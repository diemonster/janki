import json
import shutil
from pathlib import Path
from zipfile import ZipFile

import pytest

pytest.importorskip("genanki")

from japanese_anki.config import ProjectConfig
from japanese_anki.exporters.anki import AnkiBuildError, build_deck
from japanese_anki.io import DataError
from japanese_anki.models import ExampleSentence, VocabularyRecord

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_builds_importable_package(tmp_path: Path) -> None:
    config = ProjectConfig.load(PROJECT_ROOT)
    output = tmp_path / "verbs.apkg"
    result = build_deck(PROJECT_ROOT / "data/decks/verbs.yaml", config, output)

    assert result.note_count == 3
    assert result.card_types == ("recognition", "production")
    assert output.exists()
    with ZipFile(output) as archive:
        assert "collection.anki2" in archive.namelist()


def test_audio_is_resolved_against_media_dir_not_the_deck(tmp_path: Path) -> None:
    """`janki audio` stores paths relative to the project's media_dir, which is
    what the config names. Resolving them against the deck's own directory
    looked equivalent while nothing had audio and stopped being equivalent the
    moment a real clip existed — the exporter went looking in data/decks/audio/
    for a file living in data/media/audio/."""
    root = tmp_path
    (root / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "vocabulary.json"\n'
        'deck_dir = "decks"\n'
        'media_dir = "media"\n'
        'template_dir = "templates/japanese-study"\n'
        'dist_dir = "dist"\n',
        encoding="utf-8",
    )
    shutil.copytree(
        PROJECT_ROOT / "templates" / "japanese-study",
        root / "templates" / "japanese-study",
    )
    clip = root / "media" / "audio" / "janki-abc.wav"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"RIFF....WAVEfake")
    # The image branch changed base directory in the same commit and needs its
    # own coverage. Both paths are media-dir-relative on purpose: `data/decks`
    # and `data/media` are siblings, so a "../media/..." path resolves the same
    # under either base and would pin nothing.
    picture = root / "media" / "img" / "bridge.png"
    picture.parent.mkdir(parents=True)
    picture.write_bytes(b"\x89PNG fake")
    record = VocabularyRecord(
        id="word:橋:はし",
        expression="橋",
        reading="はし",
        meanings=["bridge"],
        audio="audio/janki-abc.wav",
        image="img/bridge.png",
    )
    (root / "vocabulary.json").write_text(
        json.dumps([record.to_dict()], ensure_ascii=False), encoding="utf-8"
    )
    (root / "decks").mkdir()
    (root / "decks" / "d.yaml").write_text(
        'name: D\ndeck:\n  source: "../vocabulary.json"\n', encoding="utf-8"
    )

    result = build_deck(
        root / "decks" / "d.yaml",
        ProjectConfig.load(root),
        root / "out.apkg",
    )

    assert result.media_count == 2, "the clip and the image were both found"


def _project(root: Path) -> None:
    (root / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "vocabulary.json"\n'
        'deck_dir = "decks"\n'
        'media_dir = "media"\n'
        'template_dir = "templates/japanese-study"\n'
        'dist_dir = "dist"\n',
        encoding="utf-8",
    )
    shutil.copytree(
        PROJECT_ROOT / "templates" / "japanese-study",
        root / "templates" / "japanese-study",
    )
    (root / "decks").mkdir()
    (root / "decks" / "d.yaml").write_text(
        'name: D\ndeck:\n  source: "../vocabulary.json"\n', encoding="utf-8"
    )


def _write_records(root: Path, records: list[VocabularyRecord]) -> None:
    (root / "vocabulary.json").write_text(
        json.dumps([r.to_dict() for r in records], ensure_ascii=False), encoding="utf-8"
    )


def _fields(apkg: Path) -> tuple[list[str], list[str]]:
    """The notetype's field names and the first note's values."""
    import sqlite3
    import tempfile
    from zipfile import ZipFile

    with ZipFile(apkg) as z:
        name = "collection.anki21" if "collection.anki21" in z.namelist() else "collection.anki2"
        db = Path(tempfile.mkdtemp()) / "c.db"
        db.write_bytes(z.read(name))
    con = sqlite3.connect(db)
    models = json.loads(con.execute("select models from col").fetchone()[0])
    model = next(iter(models.values()))
    values = con.execute("select flds from notes order by id limit 1").fetchone()[0]
    con.close()
    return [f["name"] for f in model["flds"]], values.split("\x1f")


def test_new_fields_are_appended_at_the_end(tmp_path: Path) -> None:
    """Appended, never inserted: a note's values are positional, so inserting
    shifts every value after it on every note that already exists — and an
    append is one-way besides, since the field is in every collection that has
    imported the deck."""
    _project(tmp_path)
    _write_records(tmp_path, [VocabularyRecord(
        id="word:橋:はし", expression="橋", reading="はし", meanings=["bridge"],
        pitch_accent=["LHL"], frequency_rank=200,
    )])

    build_deck(tmp_path / "decks" / "d.yaml", ProjectConfig.load(tmp_path), tmp_path / "o.apkg")

    names, values = _fields(tmp_path / "o.apkg")
    assert names[-8:] == [
        "PitchAccent", "FrequencyRank", "ExampleAudio", "KanjiInfo",
        "CasualJapanese", "CasualFurigana", "CasualEnglish", "CasualAudio",
    ]
    assert len(values) == len(names), "the positional list stayed parallel"
    assert values[names.index("FrequencyRank")] == "200", "and carries its value"


@pytest.mark.parametrize(
    ("rank", "expected"),
    [(200, "200"), (0, "0"), (None, "")],
    ids=["a-rank", "rank-zero", "no-rank"],
)
def test_frequency_rank_renders_zero_as_a_value_and_none_as_a_hole(
    tmp_path: Path, rank: int | None, expected: str
) -> None:
    """0 is a rank — the most frequent word there is — not a missing one, which
    is what `models.py` says about it too. Without this the whole field could
    regress to blank and every test still passed: the value list stays 22 long
    either way."""
    _project(tmp_path)
    _write_records(tmp_path, [VocabularyRecord(
        id="word:橋:はし", expression="橋", reading="はし", meanings=["bridge"],
        frequency_rank=rank,
    )])

    build_deck(tmp_path / "decks" / "d.yaml", ProjectConfig.load(tmp_path), tmp_path / "o.apkg")

    names, values = _fields(tmp_path / "o.apkg")
    assert values[names.index("FrequencyRank")] == expected


def test_the_pitch_diagram_reaches_the_card(tmp_path: Path) -> None:
    _project(tmp_path)
    _write_records(tmp_path, [VocabularyRecord(
        id="word:橋:はし", expression="橋", reading="はし", meanings=["bridge"],
        pitch_accent=["LHL"],
    )])

    build_deck(tmp_path / "decks" / "d.yaml", ProjectConfig.load(tmp_path), tmp_path / "o.apkg")

    names, values = _fields(tmp_path / "o.apkg")
    pitch = values[names.index("PitchAccent")]
    # 橋 is odaka: the fall lands on the particle, so the last mora carries the
    # drop and the particle slot is low. Pinned to the exact span rather than a
    # bare `'drop' in pitch`, which stays green if the drop moves to another
    # mora or stops being high — the two things this test is named for.
    assert '<span class="mora high drop rise">し</span>' in pitch
    assert 'class="mora particle low"' in pitch


def test_a_pattern_that_does_not_fit_is_left_out_rather_than_drawn_wrong(
    tmp_path: Path,
) -> None:
    """`render_pitch_html` refuses a mismatched pattern, and a card is the last
    place to start guessing at an alignment janki declined to guess anywhere
    else."""
    _project(tmp_path)
    _write_records(tmp_path, [VocabularyRecord(
        id="word:橋:はし", expression="橋", reading="はし", meanings=["bridge"],
        pitch_accent=["LH"],
    )])

    build_deck(tmp_path / "decks" / "d.yaml", ProjectConfig.load(tmp_path), tmp_path / "o.apkg")

    names, values = _fields(tmp_path / "o.apkg")
    assert values[names.index("PitchAccent")] == ""


def test_example_audio_is_packaged_and_tagged(tmp_path: Path) -> None:
    _project(tmp_path)
    clip = tmp_path / "media" / "audio" / "janki-ex.wav"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"RIFF....WAVEfake")
    _write_records(tmp_path, [VocabularyRecord(
        id="word:橋:はし", expression="橋", reading="はし", meanings=["bridge"],
        examples=[ExampleSentence(japanese="橋を渡ります。", audio="audio/janki-ex.wav")],
    )])

    result = build_deck(
        tmp_path / "decks" / "d.yaml", ProjectConfig.load(tmp_path), tmp_path / "o.apkg"
    )

    names, values = _fields(tmp_path / "o.apkg")
    assert values[names.index("ExampleAudio")] == "[sound:janki-ex.wav]"
    assert result.media_count == 1


def test_a_deck_relative_media_path_still_resolves(tmp_path: Path) -> None:
    """Kept as a fallback: it is what the exporter did before `media_dir`
    existed, and a hand-written note may still carry a path meant that way."""
    _project(tmp_path)
    clip = tmp_path / "decks" / "sounds" / "hand.wav"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"RIFF....WAVEfake")
    _write_records(tmp_path, [VocabularyRecord(
        id="word:橋:はし", expression="橋", reading="はし", meanings=["bridge"],
        audio="sounds/hand.wav",
    )])

    result = build_deck(
        tmp_path / "decks" / "d.yaml", ProjectConfig.load(tmp_path), tmp_path / "o.apkg"
    )

    assert result.media_count == 1


def test_a_verbatim_sound_tag_warns_rather_than_going_quietly(tmp_path: Path) -> None:
    """No file is packaged for a hand-written tag, so the card is silent unless
    that media is already in the collection — a silent-mute trap for a pipeline
    that generates its own audio."""
    _project(tmp_path)
    _write_records(tmp_path, [VocabularyRecord(
        id="word:橋:はし", expression="橋", reading="はし", meanings=["bridge"],
        audio="[sound:something-i-recorded.mp3]",
    )])

    result = build_deck(
        tmp_path / "decks" / "d.yaml", ProjectConfig.load(tmp_path), tmp_path / "o.apkg"
    )

    assert result.media_count == 0
    assert any("verbatim" in w and "word:橋:はし" in w for w in result.warnings)
    names, values = _fields(tmp_path / "o.apkg")
    assert values[names.index("Audio")] == "[sound:something-i-recorded.mp3]"


def test_missing_media_names_both_places_it_looked(tmp_path: Path) -> None:
    _project(tmp_path)
    _write_records(tmp_path, [VocabularyRecord(
        id="word:橋:はし", expression="橋", reading="はし", meanings=["bridge"],
        audio="audio/gone.wav",
    )])

    with pytest.raises(AnkiBuildError) as caught:
        build_deck(
            tmp_path / "decks" / "d.yaml", ProjectConfig.load(tmp_path), tmp_path / "o.apkg"
        )

    message = str(caught.value)
    # Both *full* paths, not the substrings "media" and "decks": pytest names
    # tmp_path after the test, so `test_missing_media_names_both_0` contains
    # "media" no matter what the exporter says, and the deck path contributes
    # "decks" on its own. Either substring passed with half the message deleted.
    assert str(tmp_path / "media" / "audio" / "gone.wav") in message
    assert str(tmp_path / "decks" / "audio" / "gone.wav") in message


# --- one bad pattern must not take the good ones with it --------------------


def _build(root: Path):
    return build_deck(root / "decks" / "d.yaml", ProjectConfig.load(root), root / "o.apkg")


def test_a_malformed_second_pattern_leaves_the_first_one_drawn(tmp_path: Path) -> None:
    """`render_pitch_html` refuses a whole *list* when any member does not fit,
    so a malformed second entry used to discard a perfectly good primary and the
    card got no diagram at all. Validation rates the mismatch a warning, not an
    error, so the build proceeds and the loss was silent."""
    _project(tmp_path)
    _write_records(tmp_path, [VocabularyRecord(
        id="word:橋:はし", expression="橋", reading="はし", meanings=["bridge"],
        pitch_accent=["LHL", "LH"],
    )])

    result = _build(tmp_path)

    names, values = _fields(tmp_path / "o.apkg")
    pitch = values[names.index("PitchAccent")]
    assert pitch.count('<span class="pitch">') == 1, "the pattern that fits is drawn"
    assert (
        '<span class="mora high drop rise">し</span>' in pitch
    ), "and drawn as odaka, which LHL is"
    assert any("'LH' does not fit" in w for w in result.warnings), "the drop is reported"


def test_the_audio_accent_leads_the_diagram(tmp_path: Path) -> None:
    """`select_pattern` prefers `audio_accent` when choosing what `janki audio`
    forces into the clip. A diagram built from `pitch_accent` alone drew heiban
    onto a card whose audio said odaka — one accent seen, another heard, on the
    card whose whole job is telling those apart."""
    _project(tmp_path)
    _write_records(tmp_path, [VocabularyRecord(
        id="word:橋:はし", expression="橋", reading="はし", meanings=["bridge"],
        pitch_accent=["LHH"], audio_accent="LHL",
    )])

    _build(tmp_path)

    names, values = _fields(tmp_path / "o.apkg")
    pitch = values[names.index("PitchAccent")]
    # One diagram, not two. `select_pattern` never falls through to
    # `pitch_accent` once `audio_accent` is set, so drawing both would put an
    # accent on the card that no clip says and that the curator overrode on
    # purpose — asserting 橋 has two accepted accents on the card whose job is
    # telling 橋 from 端.
    assert pitch.count('<span class="pitch">') == 1
    assert (
        '<span class="mora high drop rise">し</span>' in pitch
    ), "and it is odaka, which the clip says"
    assert 'class="mora particle high"' not in pitch, "the overridden heiban is gone"


def test_an_audio_accent_alone_still_draws(tmp_path: Path) -> None:
    """`audio_accent` is hand-typed curation that no importer writes, so a record
    can carry it with `pitch_accent` empty. That short-circuited to no diagram
    while the clip was voiced from the very pattern that was ignored."""
    _project(tmp_path)
    _write_records(tmp_path, [VocabularyRecord(
        id="word:橋:はし", expression="橋", reading="はし", meanings=["bridge"],
        audio_accent="LHL",
    )])

    _build(tmp_path)

    names, values = _fields(tmp_path / "o.apkg")
    assert (
        '<span class="mora high drop rise">し</span>' in values[names.index("PitchAccent")]
    )


def test_a_repeated_pattern_is_drawn_once(tmp_path: Path) -> None:
    _project(tmp_path)
    _write_records(tmp_path, [VocabularyRecord(
        id="word:橋:はし", expression="橋", reading="はし", meanings=["bridge"],
        pitch_accent=["LHL"], audio_accent="LHL",
    )])

    _build(tmp_path)

    names, values = _fields(tmp_path / "o.apkg")
    assert values[names.index("PitchAccent")].count('<span class="pitch">') == 1


# --- media: precedence, collisions, and the image field ---------------------


def test_the_media_dir_wins_when_both_roots_hold_the_same_path(tmp_path: Path) -> None:
    """The precedence only has an effect when both files exist, and then it
    decides which clip the card plays. With the file under one root only,
    reversing the order left every test green."""
    _project(tmp_path)
    for root, payload in ((tmp_path / "media", b"the media-dir clip"),
                          (tmp_path / "decks", b"the deck-dir clip")):
        (root / "audio").mkdir(parents=True, exist_ok=True)
        (root / "audio" / "same.wav").write_bytes(payload)
    _write_records(tmp_path, [VocabularyRecord(
        id="word:橋:はし", expression="橋", reading="はし", meanings=["bridge"],
        audio="audio/same.wav",
    )])

    _build(tmp_path)

    with ZipFile(tmp_path / "o.apkg") as package:
        media = json.loads(package.read("media").decode("utf-8"))
        index = next(key for key, name in media.items() if name == "same.wav")
        assert package.read(index) == b"the media-dir clip"


def test_two_files_that_would_flatten_to_one_name_are_refused(tmp_path: Path) -> None:
    """Anki stores media by basename and genanki packages by basename, so two
    different files with one name become one on import — the second overwrites
    the first, `media_count` still says two, and a card plays another word's
    audio. Content-addressed names make this unreachable for generated clips;
    a hand-written path can still do it."""
    _project(tmp_path)
    (tmp_path / "media" / "audio").mkdir(parents=True)
    (tmp_path / "decks" / "sounds").mkdir(parents=True)
    (tmp_path / "media" / "audio" / "hand.wav").write_bytes(b"one")
    (tmp_path / "decks" / "sounds" / "hand.wav").write_bytes(b"two")
    _write_records(tmp_path, [
        VocabularyRecord(id="word:橋:はし", expression="橋", reading="はし",
                         meanings=["bridge"], audio="audio/hand.wav"),
        VocabularyRecord(id="word:箸:はし", expression="箸", reading="はし",
                         meanings=["chopsticks"], audio="sounds/hand.wav"),
    ])

    with pytest.raises(AnkiBuildError) as caught:
        _build(tmp_path)

    message = str(caught.value)
    assert "hand.wav" in message
    assert "word:橋:はし" in message and "word:箸:はし" in message, "both records named"


def test_a_sound_tag_in_the_image_field_is_refused(tmp_path: Path) -> None:
    """The passthrough belongs to audio fields. In the image field the same
    value is a mis-mapped record, and passing it through both warned about
    silence for something that makes no sound and wrote the value into the card
    unescaped — the one field value here that skipped `html.escape`."""
    _project(tmp_path)
    _write_records(tmp_path, [VocabularyRecord(
        id="word:橋:はし", expression="橋", reading="はし", meanings=["bridge"],
        image="[sound:x]<b>markup</b>",
    )])

    with pytest.raises(AnkiBuildError, match="Image for word:橋:はし does not exist"):
        _build(tmp_path)


def test_two_records_may_share_one_file(tmp_path: Path) -> None:
    """The direction the collision guard must *not* fire in. Every other media
    test uses one record and one file, so refusing any second claim on a
    basename — including the same file — left the suite green while breaking
    every deck where two notes share an asset."""
    _project(tmp_path)
    (tmp_path / "media" / "audio").mkdir(parents=True)
    (tmp_path / "media" / "audio" / "shared.wav").write_bytes(b"one clip, two cards")
    _write_records(tmp_path, [
        VocabularyRecord(id="word:橋:はし", expression="橋", reading="はし",
                         meanings=["bridge"], audio="audio/shared.wav"),
        # The same file by a different spelling: `.resolve()` collapses them, so
        # this also pins that the guard compares resolved paths and not strings.
        VocabularyRecord(id="word:箸:はし", expression="箸", reading="はし",
                         meanings=["chopsticks"], audio="../media/audio/shared.wav"),
    ])

    result = _build(tmp_path)

    assert result.media_count == 1, "one file, packaged once"
    names, values = _fields(tmp_path / "o.apkg")
    assert values[names.index("Audio")] == "[sound:shared.wav]"


def test_names_that_differ_only_in_case_still_collide(tmp_path: Path) -> None:
    """Anki's media folder is what flattens these, and on macOS it is neither
    case- nor composition-sensitive. Comparing basenames byte-for-byte waved
    `Hand.wav` and `hand.wav` through to the outcome the guard exists to stop."""
    _project(tmp_path)
    (tmp_path / "media" / "audio").mkdir(parents=True)
    (tmp_path / "decks" / "sounds").mkdir(parents=True)
    (tmp_path / "media" / "audio" / "Hand.wav").write_bytes(b"one")
    (tmp_path / "decks" / "sounds" / "hand.wav").write_bytes(b"two")
    _write_records(tmp_path, [
        VocabularyRecord(id="word:橋:はし", expression="橋", reading="はし",
                         meanings=["bridge"], audio="audio/Hand.wav"),
        VocabularyRecord(id="word:箸:はし", expression="箸", reading="はし",
                         meanings=["chopsticks"], audio="sounds/hand.wav"),
    ])

    with pytest.raises(AnkiBuildError) as caught:
        _build(tmp_path)

    message = str(caught.value)
    assert "Hand.wav" in message and "hand.wav" in message, "both names as written"


def test_a_pattern_with_no_reading_to_draw_it_over_is_reported(tmp_path: Path) -> None:
    """A hand-written inline note can carry an accent and no reading.
    Validation's own length check is gated on `reading` too, so without this the
    pattern went nowhere and nothing anywhere said so."""
    _project(tmp_path)
    _write_records(tmp_path, [VocabularyRecord(
        id="word:ひらがな:", expression="ひらがな", meanings=["hiragana"],
        pitch_accent=["LHHH"],
    )])

    result = _build(tmp_path)

    names, values = _fields(tmp_path / "o.apkg")
    assert values[names.index("PitchAccent")] == ""
    assert any("no reading to draw it over" in w for w in result.warnings)


def _case_insensitive(root: Path) -> bool:
    probe = root / "CaseProbe.tmp"
    probe.write_bytes(b"")
    try:
        return (root / "caseprobe.tmp").exists()
    finally:
        probe.unlink()


def test_two_spellings_of_one_file_are_not_a_collision(tmp_path: Path) -> None:
    """`resolve()` rebuilds the path from the components as written — it fixes
    neither case nor Unicode composition — so on the very filesystem the
    collision key accounts for, two references to *one* file compared unequal
    and aborted the build, advising "rename one" when there is only one.

    Skipped where the filesystem really does distinguish them, because there the
    two spellings are two files and refusing them is correct."""
    _project(tmp_path)
    if not _case_insensitive(tmp_path):
        pytest.skip("case-sensitive filesystem: these are genuinely two files")
    (tmp_path / "media" / "audio").mkdir(parents=True)
    (tmp_path / "media" / "audio" / "Shared.wav").write_bytes(b"one file, two spellings")
    _write_records(tmp_path, [
        VocabularyRecord(id="word:橋:はし", expression="橋", reading="はし",
                         meanings=["bridge"], audio="audio/Shared.wav"),
        VocabularyRecord(id="word:箸:はし", expression="箸", reading="はし",
                         meanings=["chopsticks"], audio="audio/shared.wav"),
    ])

    result = _build(tmp_path)

    assert result.media_count == 1, "one file on disk, packaged once"
    with ZipFile(tmp_path / "o.apkg") as package:
        media = json.loads(package.read("media").decode("utf-8"))
    assert sorted(media.values()) == ["Shared.wav"], "under one name, not two"


def test_meanings_are_capped_with_the_remainder_named(tmp_path: Path) -> None:
    """jpdb hands back every sense a word has — する has 17 — and a recognition
    card is not a dictionary entry. Capped at display: the record keeps all of
    them, so `janki status`, a search, and a human choosing which sense matters
    all still see the full list."""
    _project(tmp_path)
    _write_records(tmp_path, [VocabularyRecord(
        id="word:する:する", expression="する", reading="する",
        meanings=[f"sense {n}" for n in range(1, 18)],
    )])

    build_deck(tmp_path / "decks" / "d.yaml", ProjectConfig.load(tmp_path), tmp_path / "o.apkg")

    names, values = _fields(tmp_path / "o.apkg")
    meanings = values[names.index("Meanings")]
    assert "sense 4" in meanings and "sense 5" not in meanings, "four, in jpdb's order"
    assert "+13 more senses" in meanings, "and the count is kept, not dropped"


def test_a_record_inside_the_cap_gets_no_note(tmp_path: Path) -> None:
    _project(tmp_path)
    _write_records(tmp_path, [VocabularyRecord(
        id="word:橋:はし", expression="橋", reading="はし", meanings=["bridge", "span"],
    )])

    build_deck(tmp_path / "decks" / "d.yaml", ProjectConfig.load(tmp_path), tmp_path / "o.apkg")

    names, values = _fields(tmp_path / "o.apkg")
    assert "more sense" not in values[names.index("Meanings")]


def test_one_hidden_sense_is_singular(tmp_path: Path) -> None:
    _project(tmp_path)
    _write_records(tmp_path, [VocabularyRecord(
        id="word:橋:はし", expression="橋", reading="はし",
        meanings=["a", "b", "c", "d", "e"],
    )])

    build_deck(tmp_path / "decks" / "d.yaml", ProjectConfig.load(tmp_path), tmp_path / "o.apkg")

    names, values = _fields(tmp_path / "o.apkg")
    assert "+1 more sense<" in values[names.index("Meanings")]


def test_a_deck_may_set_its_own_cap(tmp_path: Path) -> None:
    _project(tmp_path)
    (tmp_path / "decks" / "d.yaml").write_text(
        "deck:\n  name: T\n  max_meanings: 2\n  source: ../vocabulary.json\nnotes: []\n",
        encoding="utf-8",
    )
    _write_records(tmp_path, [VocabularyRecord(
        id="word:橋:はし", expression="橋", reading="はし", meanings=["a", "b", "c"],
    )])

    build_deck(tmp_path / "decks" / "d.yaml", ProjectConfig.load(tmp_path), tmp_path / "o.apkg")

    names, values = _fields(tmp_path / "o.apkg")
    assert "+1 more sense" in values[names.index("Meanings")]


def test_the_cap_can_be_turned_off(tmp_path: Path) -> None:
    _project(tmp_path)
    (tmp_path / "janki.toml").write_text(
        (tmp_path / "janki.toml").read_text(encoding="utf-8") + "\n[cards]\nmax_meanings = 0\n",
        encoding="utf-8",
    )
    _write_records(tmp_path, [VocabularyRecord(
        id="word:橋:はし", expression="橋", reading="はし",
        meanings=[f"s{n}" for n in range(10)],
    )])

    build_deck(tmp_path / "decks" / "d.yaml", ProjectConfig.load(tmp_path), tmp_path / "o.apkg")

    names, values = _fields(tmp_path / "o.apkg")
    meanings = values[names.index("Meanings")]
    assert "s9" in meanings and "more sense" not in meanings


@pytest.mark.parametrize(
    "written",
    ["yes", "4.9", "all", "[1, 2]"],
    ids=["yaml-true", "a-float", "a-word", "a-list"],
)
def test_a_deck_max_meanings_that_is_not_an_integer_is_refused(
    tmp_path: Path, written: str
) -> None:
    """Deck files are YAML 1.1, so `max_meanings: yes` arrives as `True` and a
    bare `int()` turns it into 1 — every card in the deck showing one gloss and
    "+N more senses" with nothing printed to say why. `all` was worse: `int()`
    raised a `ValueError`, which is not a `JankiError`, so `janki build` ended
    in a traceback that named no deck file."""
    _project(tmp_path)
    _write_records(tmp_path, [VocabularyRecord(
        id="word:する:する", expression="する", reading="する",
        meanings=[f"sense {n}" for n in range(1, 18)],
    )])
    deck = tmp_path / "decks" / "d.yaml"
    deck.write_text(
        f'name: D\ndeck:\n  source: "../vocabulary.json"\n  max_meanings: {written}\n',
        encoding="utf-8",
    )

    with pytest.raises(DataError, match="max_meanings must be an integer"):
        build_deck(deck, ProjectConfig.load(tmp_path), tmp_path / "o.apkg")


def test_a_deck_may_still_set_its_own_cap(tmp_path: Path) -> None:
    """The refusal above must not cost the setting itself."""
    _project(tmp_path)
    _write_records(tmp_path, [VocabularyRecord(
        id="word:する:する", expression="する", reading="する",
        meanings=[f"sense {n}" for n in range(1, 18)],
    )])
    deck = tmp_path / "decks" / "d.yaml"
    deck.write_text(
        'name: D\ndeck:\n  source: "../vocabulary.json"\n  max_meanings: 2\n',
        encoding="utf-8",
    )

    build_deck(deck, ProjectConfig.load(tmp_path), tmp_path / "o.apkg")

    names, values = _fields(tmp_path / "o.apkg")
    meanings = values[names.index("Meanings")]
    assert "sense 2" in meanings and "sense 3" not in meanings
    assert "+15 more senses" in meanings


def test_the_record_keeps_every_sense(tmp_path: Path) -> None:
    """The cap is a card decision. Nothing is discarded on the way in — this
    project's rule — so the file still holds all 17.

    Both halves are asserted together on purpose. The disk half alone restates
    what the fixture just wrote: `build_deck` has no write path to the
    normalized file, so deleting the cap, breaking `_meanings_html`, or moving
    the trim into `resolve_deck_records` all left it green. It is the pair —
    four on the card *while* 17 are on disk — that says where the cap lives.
    """
    _project(tmp_path)
    record = VocabularyRecord(
        id="word:する:する", expression="する", reading="する",
        meanings=[f"sense {n}" for n in range(1, 18)],
    )
    _write_records(tmp_path, [record])

    build_deck(tmp_path / "decks" / "d.yaml", ProjectConfig.load(tmp_path), tmp_path / "o.apkg")

    names, values = _fields(tmp_path / "o.apkg")
    meanings = values[names.index("Meanings")]
    assert "sense 5" not in meanings and "+13 more senses" in meanings, "capped on the card"

    stored = json.loads((tmp_path / "vocabulary.json").read_text(encoding="utf-8"))
    assert len(stored[0]["meanings"]) == 17, "and not on the way in"


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("使う", "%E4%BD%BF%E3%81%86"),
        ("Q&A", "Q%26A"),
        ("C#", "C%23"),
    ],
    ids=["japanese", "an-ampersand", "a-hash"],
)
def test_the_lookup_query_is_percent_encoded_not_entity_escaped(
    tmp_path: Path, expression: str, expected: str
) -> None:
    """This field's only job is to be a URL query. `html.escape` turned `Q&A`
    into `Q&amp;A`, which the webview decodes back to `Q&A` — so Shirabe
    received `w=Q` and jpdb searched for `Q`. A `#` truncated at the fragment."""
    _project(tmp_path)
    _write_records(tmp_path, [VocabularyRecord(
        id=f"word:{expression}:x", expression=expression, reading="x", meanings=["m"],
    )])

    build_deck(tmp_path / "decks" / "d.yaml", ProjectConfig.load(tmp_path), tmp_path / "o.apkg")

    names, values = _fields(tmp_path / "o.apkg")
    assert values[names.index("ShirabeQuery")] == expected


def test_every_back_template_offers_both_lookups(tmp_path: Path) -> None:
    """A card that shows a word should let you go and read about it. Shirabe is
    a deep link into the iPhone app and does nothing on a desktop; jpdb is a web
    page and works everywhere — so a deck with only the first strands anyone
    studying at a computer."""
    for name in ("recognition-back.html", "production-back.html", "reading-back.html"):
        markup = (PROJECT_ROOT / "templates" / "japanese-study" / name).read_text(
            encoding="utf-8"
        )
        assert "shirabelookup://search?w={{ShirabeQuery}}" in markup, name
        # Both links take `ShirabeQuery`, which is percent-encoded at export
        # time — entity escaping does not protect a query delimiter, and doing
        # it in template JS would not survive AnkiWeb, which strips scripts.
        assert "jpdb.io/search?q={{ShirabeQuery}}" in markup, name


def test_the_jpdb_link_reaches_the_card(tmp_path: Path) -> None:
    """Through the built package, not just the template on disk: the template
    is only real once it is inside a notetype."""
    _project(tmp_path)
    _write_records(tmp_path, [VocabularyRecord(
        id="word:使う:つかう", expression="使う", reading="つかう", meanings=["to use"],
    )])

    build_deck(tmp_path / "decks" / "d.yaml", ProjectConfig.load(tmp_path), tmp_path / "o.apkg")

    import sqlite3
    import tempfile
    with ZipFile(tmp_path / "o.apkg") as package:
        name = (
            "collection.anki21"
            if "collection.anki21" in package.namelist()
            else "collection.anki2"
        )
        db = Path(tempfile.mkdtemp()) / "c.db"
        db.write_bytes(package.read(name))
    con = sqlite3.connect(db)
    models = json.loads(con.execute("select models from col").fetchone()[0])
    templates = next(iter(models.values()))["tmpls"]
    con.close()

    # Per template, not over the concatenation of all of them: joining first
    # means one template carrying the link satisfies the assertion for every
    # card type, and dropping it from a single template goes unnoticed.
    assert templates, "the notetype has card types at all"
    for template in templates:
        back = template["afmt"]
        assert "jpdb.io/search?q={{ShirabeQuery}}" in back, template["name"]
        assert "shirabelookup://" in back, f"{template['name']}: the existing one survived"


# --- the kanji reference section --------------------------------------------


def _kanji_file(root: Path, entries: dict) -> None:
    (root / "kanji.json").write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
    config = (root / "janki.toml").read_text(encoding="utf-8")
    (root / "janki.toml").write_text(
        config.replace('dist_dir = "dist"', 'dist_dir = "dist"\nkanji_file = "kanji.json"'),
        encoding="utf-8",
    )


KANJI_使 = {
    "使": {
        "stroke_count": 8, "grade": 3, "jlpt": 4,
        "meanings": ["use", "send on a mission"],
        "readings": [
            {"kind": "on", "reading": "シ",
             "examples": [{"written": "大使", "pronounced": "たいし", "gloss": "ambassador"}]},
            {"kind": "kun", "reading": "つか.う",
             "examples": [{"written": "使う", "pronounced": "つかう", "gloss": "to use"}]},
        ],
        "strokes": ["M1,1L9,9", "M2,2L8,8", "M3,3L7,7"],
    }
}


def test_the_kanji_section_reaches_the_card(tmp_path: Path) -> None:
    """Reference data, not record content: 使 is looked up once and read by
    every record whose expression contains it."""
    _project(tmp_path)
    _kanji_file(tmp_path, KANJI_使)
    _write_records(tmp_path, [VocabularyRecord(
        id="word:使う:つかう", expression="使う", reading="つかう", meanings=["to use"],
    )])

    build_deck(tmp_path / "decks" / "d.yaml", ProjectConfig.load(tmp_path), tmp_path / "o.apkg")

    names, values = _fields(tmp_path / "o.apkg")
    section = values[names.index("KanjiInfo")]
    assert "<summary>使</summary>" in section
    assert "N4" in section and "8画" in section
    assert "大使" in section and "たいし" in section, "the on'yomi example"
    assert "使う" in section and "to use" in section, "the kun'yomi example"


def test_a_stroke_cell_per_stroke_each_adding_one(tmp_path: Path) -> None:
    """The progression is the whole point — a single finished glyph says nothing
    about the order it is written in. Cell n draws strokes 1..n, with the newest
    marked so the eye lands on what changed."""
    _project(tmp_path)
    _kanji_file(tmp_path, KANJI_使)
    _write_records(tmp_path, [VocabularyRecord(
        id="word:使う:つかう", expression="使う", reading="つかう", meanings=["to use"],
    )])

    build_deck(tmp_path / "decks" / "d.yaml", ProjectConfig.load(tmp_path), tmp_path / "o.apkg")

    names, values = _fields(tmp_path / "o.apkg")
    section = values[names.index("KanjiInfo")]
    # Counted by cell, not by `<svg`: the block may carry other SVG elements
    # (a defs block, say) that are not stroke cells.
    cells = section.split('<span class="stroke-cell">')[1:]
    assert len(cells) == 3, "three strokes, three cells"
    assert all(cell.count('class="new"') == 1 for cell in cells), "exactly one newest"

    # Each cell draws its own stroke as the new one, over the accumulated
    # picture so far. Cells reference shared `<defs>` rather than repeating path
    # data, so "one more stroke than the last" is a property of that chain: cell
    # n is stage n-1 plus stroke n, and stage n-1 was built the same way.
    for index, cell in enumerate(cells):
        assert f'class="new" href="#kanji-0-stroke-{index}"' in cell, f"cell {index}"
        assert (f'href="#kanji-0-stage-{index - 1}"' in cell) is (index > 0), f"cell {index}"

    defs = section.split('<div class="stroke-order">')[0]
    for index in range(3):
        stage = defs.split(f'<g id="kanji-0-stage-{index}">')[1].split("</g>")[0]
        expected = [f'#kanji-0-stroke-{index}']
        if index:
            expected.insert(0, f'#kanji-0-stage-{index - 1}')
        assert [ref.split('"')[0] for ref in stage.split('href="')[1:]] == expected


def test_a_word_with_no_kanji_gets_no_section(tmp_path: Path) -> None:
    _project(tmp_path)
    _kanji_file(tmp_path, KANJI_使)
    _write_records(tmp_path, [VocabularyRecord(
        id="word:する:する", expression="する", reading="する", meanings=["to do"],
    )])

    build_deck(tmp_path / "decks" / "d.yaml", ProjectConfig.load(tmp_path), tmp_path / "o.apkg")

    names, values = _fields(tmp_path / "o.apkg")
    assert values[names.index("KanjiInfo")] == ""


def test_a_kanji_not_looked_up_yet_is_simply_absent(tmp_path: Path) -> None:
    """A build must not require a network round trip, or fail because one
    character could not be fetched. The section is an extra, not a gate."""
    _project(tmp_path)
    _kanji_file(tmp_path, KANJI_使)
    _write_records(tmp_path, [VocabularyRecord(
        id="word:橋:はし", expression="橋", reading="はし", meanings=["bridge"],
    )])

    result = build_deck(
        tmp_path / "decks" / "d.yaml", ProjectConfig.load(tmp_path), tmp_path / "o.apkg"
    )

    assert result.note_count == 1
    names, values = _fields(tmp_path / "o.apkg")
    assert values[names.index("KanjiInfo")] == ""


def test_each_kanji_of_a_compound_gets_its_own_block(tmp_path: Path) -> None:
    """So opening one does not open both, and the summary can carry the
    character it is about."""
    _project(tmp_path)
    entries = dict(KANJI_使)
    entries["用"] = {
        "stroke_count": 5, "jlpt": 4, "meanings": ["utilize"],
        "readings": [{"kind": "on", "reading": "ヨウ", "examples": []}],
        "strokes": ["M1,1L9,9"],
    }
    _kanji_file(tmp_path, entries)
    _write_records(tmp_path, [VocabularyRecord(
        id="word:使用:しよう", expression="使用", reading="しよう", meanings=["use"],
    )])

    build_deck(tmp_path / "decks" / "d.yaml", ProjectConfig.load(tmp_path), tmp_path / "o.apkg")

    names, values = _fields(tmp_path / "o.apkg")
    section = values[names.index("KanjiInfo")]
    assert section.count("<details") == 2
    assert section.index("<summary>使</summary>") < section.index("<summary>用</summary>"), (
        "in the order the word is written"
    )


# --- the casual sentence ----------------------------------------------------


def _with_registers(polite: str = "使います。", casual: str = "使う？") -> VocabularyRecord:
    return VocabularyRecord(
        id="word:使う:つかう", expression="使う", reading="つかう", meanings=["to use"],
        examples=[
            ExampleSentence(japanese=polite, english="polite", register="polite"),
            ExampleSentence(japanese=casual, english="casual", register="casual"),
        ],
    )


def test_both_registers_reach_the_card(tmp_path: Path) -> None:
    """A learner meets both and they are not interchangeable — a textbook
    teaches ます first and a friend never uses it, so a card showing only one
    teaches half the word."""
    _project(tmp_path)
    _write_records(tmp_path, [_with_registers()])

    build_deck(tmp_path / "decks" / "d.yaml", ProjectConfig.load(tmp_path), tmp_path / "o.apkg")

    names, values = _fields(tmp_path / "o.apkg")
    assert values[names.index("ExampleJapanese")] == "使います。"
    assert values[names.index("CasualJapanese")] == "使う？"
    assert values[names.index("CasualEnglish")] == "casual"


def test_the_casual_example_coming_first_does_not_empty_the_main_slot(
    tmp_path: Path,
) -> None:
    """Nothing orders `examples`: `enrich --ai` appends them in whatever order
    the model answered, and a hand-written record follows the schema doc, which
    says a card has a slot for each register without saying which comes first.

    Taking `examples[0]` and blanking it when it was the casual one meant a
    record whose casual sentence happened to lead showed *no* main example — the
    polite sentence was on the record, paid for, and read by no field, with
    nothing to report it: the record has examples, so `status` sees no gap and
    `janki review` reads the record rather than the built note."""
    _project(tmp_path)
    _write_records(tmp_path, [VocabularyRecord(
        id="word:使う:つかう", expression="使う", reading="つかう", meanings=["to use"],
        examples=[
            ExampleSentence(japanese="使う？", english="casual", register="casual"),
            ExampleSentence(japanese="使います。", english="polite", register="polite"),
        ],
    )])

    build_deck(tmp_path / "decks" / "d.yaml", ProjectConfig.load(tmp_path), tmp_path / "o.apkg")

    names, values = _fields(tmp_path / "o.apkg")
    assert values[names.index("ExampleJapanese")] == "使います。"
    assert values[names.index("CasualJapanese")] == "使う？"


def test_a_record_with_only_a_casual_example_does_not_show_it_twice(tmp_path: Path) -> None:
    """Otherwise the same sentence fills both slots and the card claims a
    polite/casual contrast it does not have."""
    _project(tmp_path)
    _write_records(tmp_path, [VocabularyRecord(
        id="word:使う:つかう", expression="使う", reading="つかう", meanings=["to use"],
        examples=[ExampleSentence(japanese="使う？", register="casual")],
    )])

    build_deck(tmp_path / "decks" / "d.yaml", ProjectConfig.load(tmp_path), tmp_path / "o.apkg")

    names, values = _fields(tmp_path / "o.apkg")
    assert values[names.index("CasualJapanese")] == "使う？"
    assert values[names.index("ExampleJapanese")] == ""


def test_an_example_with_no_register_stays_the_polite_one(tmp_path: Path) -> None:
    """Every example written before the field existed carries no register, and
    is the ordinary 〜ます sentence. Putting one in the casual slot would label
    it as something it is not."""
    _project(tmp_path)
    _write_records(tmp_path, [VocabularyRecord(
        id="word:使う:つかう", expression="使う", reading="つかう", meanings=["to use"],
        examples=[ExampleSentence(japanese="使います。")],
    )])

    build_deck(tmp_path / "decks" / "d.yaml", ProjectConfig.load(tmp_path), tmp_path / "o.apkg")

    names, values = _fields(tmp_path / "o.apkg")
    assert values[names.index("ExampleJapanese")] == "使います。"
    assert values[names.index("CasualJapanese")] == ""


def test_two_casual_examples_leave_the_main_slot_empty(tmp_path: Path) -> None:
    """Excluding the one object `example_in("casual")` returned had the same
    shape of bug one step along: with two casual sentences the *second* landed
    in the main slot, which the template renders with no register heading — so
    the card presents a casual sentence as the neutral form of the word. Chosen
    by register now, not by identity."""
    _project(tmp_path)
    _write_records(tmp_path, [VocabularyRecord(
        id="word:使う:つかう", expression="使う", reading="つかう", meanings=["to use"],
        examples=[
            ExampleSentence(japanese="使う？", english="casual", register="casual"),
            ExampleSentence(japanese="使うの？", english="also casual", register="casual"),
        ],
    )])

    build_deck(tmp_path / "decks" / "d.yaml", ProjectConfig.load(tmp_path), tmp_path / "o.apkg")

    names, values = _fields(tmp_path / "o.apkg")
    assert values[names.index("ExampleJapanese")] == ""
    assert values[names.index("CasualJapanese")] == "使う？"


def test_the_preview_shows_the_sentence_the_deck_will(tmp_path: Path) -> None:
    """`janki preview` is billed as a stand-in for the card. It read
    `first_example` while the exporter had moved on, so for a record whose
    casual example comes first the two showed different sentences."""
    from japanese_anki.preview import build_preview

    _project(tmp_path)
    _write_records(tmp_path, [VocabularyRecord(
        id="word:使う:つかう", expression="使う", reading="つかう", meanings=["to use"],
        examples=[
            ExampleSentence(japanese="使う？", english="casual", register="casual"),
            ExampleSentence(japanese="使います。", english="polite", register="polite"),
        ],
    )])

    build_deck(tmp_path / "decks" / "d.yaml", ProjectConfig.load(tmp_path), tmp_path / "o.apkg")
    names, values = _fields(tmp_path / "o.apkg")
    page = build_preview(tmp_path / "decks" / "d.yaml", tmp_path / "p.html").read_text(
        encoding="utf-8"
    )

    # Both slots, in the same places the card puts them.
    assert values[names.index("ExampleJapanese")] == "使います。"
    assert values[names.index("CasualJapanese")] == "使う？"
    main, _, casual_block = page.partition('class="example-casual"')
    assert "使います。" in main, "the main slot holds the polite sentence"
    assert "使う？" in casual_block, "and the casual one is labelled as casual"


def test_an_example_that_reaches_no_slot_is_reported(tmp_path: Path) -> None:
    """A card has one example slot and one casual slot. A record carrying a
    third sentence has one that reaches no field on any card type — and
    `janki audio --examples` has already voiced it and committed the clip.

    The clip is not shipped: `_resolve_media` only ever sees the two slots, so
    it never enters the package. It is worse than that — it sits in
    `data/media/` referenced by no card, and `janki audio --prune` will not
    reclaim it either, because prune walks `record.examples` and the record
    still names it. Said out loud at build time, the way a pitch pattern with
    no reading to draw it over is."""
    _project(tmp_path)
    _write_records(tmp_path, [VocabularyRecord(
        id="word:使う:つかう", expression="使う", reading="つかう", meanings=["to use"],
        examples=[
            ExampleSentence(japanese="使う？", english="casual", register="casual"),
            ExampleSentence(japanese="使うの？", english="also casual", register="casual"),
        ],
    )])

    result = build_deck(
        tmp_path / "decks" / "d.yaml", ProjectConfig.load(tmp_path), tmp_path / "o.apkg"
    )

    assert any(
        "使うの？" in warning and "reaches no field" in warning
        for warning in result.warnings
    ), result.warnings


def test_the_preview_shows_a_casual_only_record(tmp_path: Path) -> None:
    """`enrich --ai` returns a polite and a casual sentence, and the polite one
    can fail `qc.example_contains_target` and be dropped — leaving a record
    whose only example is casual. The card shows it under "Casually"; a preview
    built on the main slot alone showed nothing at all."""
    from japanese_anki.preview import build_preview

    _project(tmp_path)
    _write_records(tmp_path, [VocabularyRecord(
        id="word:使う:つかう", expression="使う", reading="つかう", meanings=["to use"],
        examples=[ExampleSentence(japanese="使う？", english="casual", register="casual")],
    )])

    page = build_preview(tmp_path / "decks" / "d.yaml", tmp_path / "p.html").read_text(
        encoding="utf-8"
    )

    assert "使う？" in page
