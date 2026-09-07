"""``janki kanji-notes`` — the owner's direct route to character cards.

Five explicit characters, one command: prepare the notes, create or extend the
deck that ships them, and build the package. `--dry-run` stops after the
preparation, which is why the counts it prints are real rather than estimated.

The reference and provider lookups are replaced at the seams the services read
them through at call time, so nothing here reaches the network; everything
downstream of them — the store writers, the deck file, genanki — is the real
thing, and the package is opened afterwards and counted.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
import zipfile
from pathlib import Path

import pytest
import yaml

from japanese_anki import cli, jpdb_kanji, kanji
from japanese_anki.application import deck_package
from japanese_anki.config import ProjectConfig
from japanese_anki.jpdb_kanji import (
    BoundExample,
    CharacterReadings,
    ReadingGroup,
    ReadingUsage,
)

TEMPLATES = Path(__file__).resolve().parents[1] / "templates" / "japanese-study"

#: The owner's explicit targets for this milestone.
TARGETS = ("物", "特", "鳥", "料", "理")


def project(tmp_path: Path) -> Path:
    """A repository with the real templates copied in.

    Copied rather than pointed at: a package plan binds every template it
    reads and refuses one outside the repository it is building, which is the
    rule that keeps a built deck describable by paths inside its own checkout.
    """
    root = tmp_path / "repo"
    (root / "data" / "decks").mkdir(parents=True)
    shutil.copytree(TEMPLATES, root / "templates" / "japanese-study")
    (root / "janki.toml").write_text(
        "[paths]\n"
        'kanji_notes_file = "data/kanji_notes.json"\n'
        'kanji_file = "data/kanji.json"\n'
        'jpdb_readings_file = "data/jpdb_readings.json"\n'
        f'jpdb_html_cache = "{tmp_path / "cache"}"\n',
        encoding="utf-8",
    )
    return root


def example(written: str, pronounced: str) -> BoundExample:
    return BoundExample(
        written=written,
        pronounced=pronounced,
        gloss="cooking",
        furigana=f"{written}[{pronounced}]",
        source_url=f"https://jpdb.io/vocabulary/1/{written}/{pronounced}",
    )


class Lookups:
    """Both sources, answering from a script and recording every request."""

    def __init__(self) -> None:
        self.looked_up: list[str] = []
        self.fetched: list[str] = []

    def fetch_kanji(self, character: str, **_kwargs: object) -> kanji.KanjiInfo:
        self.looked_up.append(character)
        return kanji.KanjiInfo(
            character=character,
            stroke_count=11,
            meanings=("logic", "reason"),
            readings=(kanji.Reading(kind="on", reading="リ"),),
            strokes=("M10,10 L20,20",),
        )

    def fetch_character(
        self, character: str, *, html_cache: Path, refresh: bool = False, **_kw: object
    ) -> CharacterReadings:
        self.fetched.append(character)
        return CharacterReadings(
            character=character,
            source_url=f"https://jpdb.io/kanji/{character}",
            fetched_at_utc="2026-09-07T00:00:00Z",
            sha256="b" * 64,
            groups=(
                ReadingGroup(
                    source_class=jpdb_kanji.COMMON_CELL_CLASS,
                    readings=(
                        ReadingUsage(
                            label="リ",
                            href=f"/kanji-reading/{character}/リ",
                            percent_text="(84%)",
                            percent=84,
                            percent_less_than=False,
                            examples=(example("料理", "りょうり"),),
                        ),
                    ),
                ),
            ),
        )


@pytest.fixture
def lookups(monkeypatch: pytest.MonkeyPatch) -> Lookups:
    answers = Lookups()
    monkeypatch.setattr(kanji, "fetch_kanji", answers.fetch_kanji)
    monkeypatch.setattr(jpdb_kanji, "fetch_character", answers.fetch_character)
    return answers


def notes_and_cards(package: Path) -> tuple[int, int]:
    """The built package's own note and card counts, read out of its database."""
    with zipfile.ZipFile(package) as archive, tempfile.TemporaryDirectory() as into:
        name = (
            "collection.anki21"
            if "collection.anki21" in archive.namelist()
            else "collection.anki2"
        )
        archive.extract(name, into)
        connection = sqlite3.connect(Path(into) / name)
        try:
            notes = connection.execute("select count(*) from notes").fetchone()[0]
            cards = connection.execute("select count(*) from cards").fetchone()[0]
        finally:
            connection.close()
    return notes, cards


def only_deck(root: Path) -> Path:
    decks = sorted((root / "data" / "decks").glob("*.yaml"))
    assert len(decks) == 1, decks
    return decks[0]


def test_five_characters_become_five_notes_and_five_cards(
    tmp_path: Path, lookups: Lookups, capsys: pytest.CaptureFixture[str]
) -> None:
    """The milestone's own target, end to end and counted in the package."""
    root = project(tmp_path)

    assert (
        cli.main(
            ["--root", str(root), "kanji-notes", *TARGETS, "--deck-name", "Genki II Kanji"]
        )
        == 0
    )

    package = root / "dist" / f"{only_deck(root).stem}.apkg"
    assert notes_and_cards(package) == (5, 5)
    out = capsys.readouterr().out
    assert "5 character notes, 5 card(s)" in out
    assert "5 character notes" in out.split("Built")[-1]


def test_the_deck_and_the_notes_land_together(
    tmp_path: Path, lookups: Lookups
) -> None:
    root = project(tmp_path)

    cli.main(["--root", str(root), "kanji-notes", "理", "--deck-name", "Kanji"])

    section = yaml.safe_load(only_deck(root).read_text(encoding="utf-8"))["deck"]
    assert section["kind"] == "kanji"
    assert section["include_ids"] == ["kanji:理"]
    stored = json.loads((root / "data" / "kanji_notes.json").read_text("utf-8"))
    assert list(stored["notes"]) == ["理"]


def test_a_dry_run_shows_the_real_counts_and_changes_no_canonical_data(
    tmp_path: Path, lookups: Lookups, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path)

    assert (
        cli.main(
            [
                "--root",
                str(root),
                "kanji-notes",
                *TARGETS,
                "--deck-name",
                "Genki II Kanji",
                "--dry-run",
            ]
        )
        == 0
    )

    out = capsys.readouterr().out
    assert "5 character notes, 5 card(s)" in out
    assert "Dry run: no canonical data changed." in out
    assert lookups.fetched == list(TARGETS), "it really did look them up"
    for name in ("kanji_notes.json", "kanji.json", "jpdb_readings.json"):
        assert not (root / "data" / name).exists()
    assert list((root / "data" / "decks").glob("*.yaml")) == []
    assert not (root / "dist").exists()


def test_adding_to_an_existing_deck_reports_both_counts(
    tmp_path: Path, lookups: Lookups, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path)
    cli.main(["--root", str(root), "kanji-notes", "理", "料", "--deck-name", "Kanji"])
    deck = only_deck(root)
    capsys.readouterr()

    assert (
        cli.main(["--root", str(root), "kanji-notes", "鳥", "--deck", str(deck)]) == 0
    )

    out = capsys.readouterr().out
    assert "this batch: 1 character note, 1 card(s)" in out
    assert "the deck will hold: 3 note(s), 3 card(s)" in out
    assert notes_and_cards(root / "dist" / f"{deck.stem}.apkg") == (3, 3)


def test_a_rerun_asks_for_nothing_writes_nothing_and_still_builds(
    tmp_path: Path, lookups: Lookups, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path)
    cli.main(["--root", str(root), "kanji-notes", "理", "--deck-name", "Kanji"])
    deck = only_deck(root)
    lookups.looked_up.clear()
    lookups.fetched.clear()
    capsys.readouterr()

    assert cli.main(["--root", str(root), "kanji-notes", "理", "--deck", str(deck)]) == 0

    out = capsys.readouterr().out
    assert lookups.looked_up == [] and lookups.fetched == []
    assert "Already applied: every file held these exact bytes." in out


def test_an_edit_between_the_apply_and_the_build_ships_no_package(
    tmp_path: Path,
    lookups: Lookups,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The package plan is captured before the applied state is checked, so an
    edit landing in between is already inside the plan the builder would
    accept and re-check. Only the applied-state check refuses it, which is why
    the build runs after that check rather than before it."""
    root = project(tmp_path)
    plan_package = deck_package.plan_deck_package

    def edit_then_plan(
        config: ProjectConfig, deck_path: Path, **kwargs: object
    ) -> deck_package.DeckPackagePlan:
        store = root / "data" / "kanji_notes.json"
        document = json.loads(store.read_text(encoding="utf-8"))
        document["notes"]["理"]["meanings"] = ["nobody confirmed this"]
        store.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return plan_package(config, deck_path, **kwargs)

    monkeypatch.setattr(deck_package, "plan_deck_package", edit_then_plan)

    assert (
        cli.main(["--root", str(root), "kanji-notes", "理", "--deck-name", "Kanji"])
        == 1
    )

    assert "character-batch-stale" in capsys.readouterr().err
    assert list((root / "dist").glob("*.apkg")) == [], "no package was published"


def test_reading_and_production_need_what_they_need(
    tmp_path: Path, lookups: Lookups, capsys: pytest.CaptureFixture[str]
) -> None:
    """A production card asks for a character from a hint, and janki does not
    write the hint. A reading card asks one example the source bound."""
    root = project(tmp_path)

    assert (
        cli.main(
            [
                "--root",
                str(root),
                "kanji-notes",
                "理",
                "--deck-name",
                "Kanji",
                "--directions",
                "recognition,production",
            ]
        )
        == 1
    )
    assert "production card needs a disambiguating cue" in capsys.readouterr().err

    assert (
        cli.main(
            [
                "--root",
                str(root),
                "kanji-notes",
                "理",
                "--deck-name",
                "Kanji",
                "--directions",
                "recognition,reading,production",
                "--production-cue",
                "理=the 理 of 料理",
            ]
        )
        == 0
    )
    package = root / "dist" / f"{only_deck(root).stem}.apkg"
    assert notes_and_cards(package) == (1, 3)


def test_a_word_is_refused_as_a_character_target(
    tmp_path: Path, lookups: Lookups, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path)

    assert (
        cli.main(["--root", str(root), "kanji-notes", "料理", "--deck-name", "Kanji"])
        == 1
    )

    assert "one character" in capsys.readouterr().err
    assert lookups.looked_up == []


def test_naming_both_a_deck_and_a_deck_name_is_refused(
    tmp_path: Path, lookups: Lookups
) -> None:
    root = project(tmp_path)

    with pytest.raises(SystemExit):
        cli.main(
            [
                "--root",
                str(root),
                "kanji-notes",
                "理",
                "--deck",
                str(root / "data" / "decks" / "kanji.yaml"),
                "--deck-name",
                "Kanji",
            ]
        )


def test_a_production_cue_without_an_equals_sign_says_so(
    tmp_path: Path, lookups: Lookups, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path)

    assert (
        cli.main(
            [
                "--root",
                str(root),
                "kanji-notes",
                "理",
                "--deck-name",
                "Kanji",
                "--production-cue",
                "just some text",
            ]
        )
        == 1
    )

    assert "CHARACTER=TEXT" in capsys.readouterr().err


def test_an_ordinary_build_of_a_character_deck_works(
    tmp_path: Path, lookups: Lookups, capsys: pytest.CaptureFixture[str]
) -> None:
    """`janki build` is the other route to the same package, and a character
    deck must not fall through to the word-deck reader."""
    root = project(tmp_path)
    cli.main(["--root", str(root), "kanji-notes", *TARGETS, "--deck-name", "Kanji"])
    deck = only_deck(root)
    (root / "dist" / f"{deck.stem}.apkg").unlink()
    capsys.readouterr()

    assert cli.main(["--root", str(root), "build", str(deck)]) == 0

    assert "5 character notes" in capsys.readouterr().out
    assert notes_and_cards(root / "dist" / f"{deck.stem}.apkg") == (5, 5)


def test_validate_and_status_read_a_character_deck_without_complaining(
    tmp_path: Path, lookups: Lookups, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path)
    cli.main(["--root", str(root), "kanji-notes", "理", "--deck-name", "Kanji"])
    capsys.readouterr()

    assert cli.main(["--root", str(root), "validate"]) == 0
    assert cli.main(["--root", str(root), "status"]) == 0

    output = capsys.readouterr()
    assert "kanji:理" not in output.out, "a character is not an unexported word"
    assert "skipping deck" not in output.err
