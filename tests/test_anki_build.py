import json
import shutil
from pathlib import Path
from zipfile import ZipFile

import pytest

pytest.importorskip("genanki")

from japanese_anki.config import ProjectConfig
from japanese_anki.exporters.anki import AnkiBuildError, build_deck
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


def test_the_three_new_fields_are_appended_at_the_end(tmp_path: Path) -> None:
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
    assert names[-3:] == ["PitchAccent", "FrequencyRank", "ExampleAudio"]
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
    # drop and the particle slot is low.
    assert 'class="mora high drop"' in pitch
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
    assert 'class="mora high drop"' in pitch, "and drawn as odaka, which LHL is"
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
    first = values[names.index("PitchAccent")].split("</span></span>")[0]
    assert 'class="mora high drop"' in first, "odaka — what the clip says — is drawn first"


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
    assert 'class="mora high drop"' in values[names.index("PitchAccent")]


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
