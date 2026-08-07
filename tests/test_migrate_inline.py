"""`janki migrate-inline`: inline deck notes become normalized records.

The one thing this command must never do is change what a deck exports. The
tests are built around that: the two deck files this repository actually shipped
with inline notes are frozen under ``tests/fixtures/inline-deck/``, every deck is
built through the fake-genanki harness before and after the migration, and the
resulting notes — GUIDs, field values and tags — must match exactly.

The second deck in that fixture is the interesting one. It reads every record in
the normalized file with no filter, so migrating the first deck's notes into that
file would silently hand it three notes carrying the other deck's GUIDs.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from test_anki_builder_contract import FakeDeck, FakeModel, FakeNote, FakePackage

from japanese_anki import cli, ledger
from japanese_anki.config import ProjectConfig
from japanese_anki.exporters import anki

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DECKS = Path(__file__).parent / "fixtures" / "inline-deck"

# The three records that lived inline in data/decks/verbs.yaml. Written out
# rather than derived: an id is a durable address, and a test that recomputed
# them would agree with any drift instead of catching it.
VERB_IDS = ["word:行く:いく", "word:話す:はなす", "word:食べる:たべる"]

CONFIG = f"""
[paths]
normalized_file = "data/normalized/vocabulary.json"
deck_dir = "data/decks"
ledger_file = "data/ledger.json"
template_dir = "{PROJECT_ROOT / "templates/japanese-study"}"
dist_dir = "dist"

[anki]
model_id_base = 1607392310
"""


# --- harness ----------------------------------------------------------------


def _project(tmp_path: Path, decks: dict[str, dict[str, Any]] | None = None) -> Path:
    """A project laid out like the real one, with hand-written decks."""
    (tmp_path / "janki.toml").write_text(CONFIG, encoding="utf-8")
    (tmp_path / "data" / "normalized").mkdir(parents=True)
    (tmp_path / "data" / "normalized" / "vocabulary.json").write_text("[]\n", encoding="utf-8")
    deck_dir = tmp_path / "data" / "decks"
    deck_dir.mkdir()
    for stem, deck in (decks or {}).items():
        (deck_dir / f"{stem}.yaml").write_text(
            yaml.safe_dump(deck, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
    return tmp_path


def _fixture_project(tmp_path: Path) -> Path:
    """The repository's own decks as they were before this milestone."""
    root = _project(tmp_path)
    for path in sorted(FIXTURE_DECKS.glob("*.yaml")):
        shutil.copy(path, root / "data" / "decks" / path.name)
    return root


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


def _built(deck_path: Path, root: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Everything a built deck would carry, via the fake genanki harness.

    Reaching through ``build_deck`` rather than ``resolve_deck_records`` is the
    point: GUIDs, escaped field values and cleaned tags are what Anki sees, and
    they are what must be identical on both sides of the migration.
    """
    decks: list[FakeDeck] = []
    models: list[FakeModel] = []

    def make_deck(deck_id: int, name: str) -> FakeDeck:
        deck = FakeDeck(deck_id, name)
        decks.append(deck)
        return deck

    def make_model(model_id: int, name: str, **kwargs: Any) -> FakeModel:
        model = FakeModel(model_id, name, **kwargs)
        models.append(model)
        return model

    monkeypatch.setattr(
        anki,
        "genanki",
        SimpleNamespace(
            Model=make_model,
            Deck=make_deck,
            Note=FakeNote,
            Package=FakePackage,
            guid_for=lambda value: f"guid:{value}",
        ),
    )
    output = root / "dist" / f"{deck_path.stem}.apkg"
    anki.build_deck(deck_path, ProjectConfig.load(root), output)
    deck = decks[-1]
    model = models[-1]
    return {
        "deck_id": deck.deck_id,
        "deck_name": deck.name,
        "description": deck.description,
        "model_id": model.model_id,
        "model_name": model.name,
        "templates": [template["name"] for template in model.templates],
        "notes": [
            (note.guid, tuple(note.fields), tuple(note.tags)) for note in deck.notes
        ],
    }


def _every_deck(root: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, Any]]:
    return {
        path.name: _built(path, root, monkeypatch)
        for path in sorted((root / "data" / "decks").glob("*.yaml"))
    }


def _migrate(root: Path, deck: str = "verbs.yaml") -> int:
    return cli.main(
        ["--root", str(root), "migrate-inline", str(root / "data" / "decks" / deck)]
    )


def _deck_config(root: Path, name: str) -> dict[str, Any]:
    raw = yaml.safe_load((root / "data" / "decks" / name).read_text(encoding="utf-8"))
    return raw


def _records(root: Path) -> list[dict[str, Any]]:
    return json.loads(
        (root / "data" / "normalized" / "vocabulary.json").read_text(encoding="utf-8")
    )


# --- the repository's own migration -----------------------------------------


def test_every_deck_builds_exactly_the_same_notes_after_migration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _fixture_project(tmp_path)
    before = _every_deck(root, monkeypatch)
    assert [note[1][0] for note in before["verbs.yaml"]["notes"]] == VERB_IDS
    assert before["personal-vocabulary.yaml"]["notes"] == []

    assert _migrate(root) == 0

    assert _every_deck(root, monkeypatch) == before


def test_the_records_move_into_the_normalized_file_under_the_same_ids(
    tmp_path: Path,
) -> None:
    root = _fixture_project(tmp_path)

    assert _migrate(root) == 0

    stored = _records(root)
    assert [record["id"] for record in stored] == sorted(VERB_IDS)
    # Byte-identical ids, not merely equivalent ones: the GUID is derived from
    # the id, so any normalization of it orphans the card's review history.
    inline = yaml.safe_load((FIXTURE_DECKS / "verbs.yaml").read_text(encoding="utf-8"))
    assert sorted(note["id"] for note in inline["notes"]) == [
        record["id"] for record in stored
    ]
    by_id = {record["id"]: record for record in stored}
    for note in inline["notes"]:
        for key, value in note.items():
            if key == "source":
                # Stored records carry the whole SourceReference; the note only
                # wrote the keys it cared about.
                assert by_id[note["id"]][key].items() >= value.items()
            else:
                assert by_id[note["id"]][key] == value, key


def test_the_migrated_deck_keeps_its_identity_and_pins_its_membership(
    tmp_path: Path,
) -> None:
    root = _fixture_project(tmp_path)
    before = yaml.safe_load((FIXTURE_DECKS / "verbs.yaml").read_text(encoding="utf-8"))["deck"]

    assert _migrate(root) == 0

    raw = _deck_config(root, "verbs.yaml")
    assert "notes" not in raw
    deck = raw["deck"]
    for key in ("name", "deck_id", "description", "output", "cards"):
        assert deck[key] == before[key]
    assert deck["source"] == "../normalized/vocabulary.json"
    assert deck["include_ids"] == VERB_IDS


def test_a_deck_reading_the_same_file_does_not_silently_gain_those_notes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _fixture_project(tmp_path)

    assert _migrate(root) == 0

    deck = _deck_config(root, "personal-vocabulary.yaml")["deck"]
    assert deck["exclude_ids"] == VERB_IDS
    out = capsys.readouterr().out
    assert "personal-vocabulary.yaml" in out
    assert "exclude_ids" in out


def test_editing_a_deck_file_in_place_keeps_its_comments(tmp_path: Path) -> None:
    root = _fixture_project(tmp_path)
    original = (FIXTURE_DECKS / "personal-vocabulary.yaml").read_text(encoding="utf-8")

    assert _migrate(root) == 0

    text = (root / "data" / "decks" / "personal-vocabulary.yaml").read_text(encoding="utf-8")
    for line in original.splitlines():
        if line.startswith("#"):
            assert line in text
    # Only the one key was added; every original line survives untouched.
    assert set(original.splitlines()) - set(text.splitlines()) == set()


# --- idempotence ------------------------------------------------------------


def test_running_it_twice_changes_nothing_and_says_so(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _fixture_project(tmp_path)
    assert _migrate(root) == 0
    capsys.readouterr()
    touched = [
        root / "data" / "normalized" / "vocabulary.json",
        root / "data" / "decks" / "verbs.yaml",
        root / "data" / "decks" / "personal-vocabulary.yaml",
        root / "data" / "ledger.json",
    ]
    before = {path: path.read_bytes() for path in touched}

    assert _migrate(root) == 0

    assert {path: path.read_bytes() for path in touched} == before
    assert "Nothing to migrate" in capsys.readouterr().out


# --- the ledger -------------------------------------------------------------


def test_migrated_records_are_registered_in_the_ledger(tmp_path: Path) -> None:
    root = _fixture_project(tmp_path)

    assert _migrate(root) == 0

    book = ledger.load(root / "data" / "ledger.json")
    assert sorted(book.records) == sorted(VERB_IDS)
    for record_id in VERB_IDS:
        assert book.records[record_id]["sources"] == [
            {
                "type": "manual",
                "ref": "data/decks/verbs.yaml",
                "seen_at": book.records[record_id]["added_at"],
            }
        ]


# --- merge semantics --------------------------------------------------------


def _shared_source() -> dict[str, Any]:
    return {"type": "shirabe", "imported_from": "export.csv", "row": 2}


def test_the_inline_note_wins_over_a_normalized_record_it_disagrees_with(
    tmp_path: Path,
) -> None:
    root = _project(
        tmp_path,
        {
            "verbs": {
                "deck": {"name": "Verbs"},
                "notes": [
                    _raw("話す", "はなす", meanings=["to talk"], usage_notes="curated")
                ],
            }
        },
    )
    (root / "data" / "normalized" / "vocabulary.json").write_text(
        json.dumps(
            [_raw("話す", "はなす", meanings=["to speak"], romaji="hanasu")],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert _migrate(root) == 0

    stored = _records(root)[0]
    # The inline value wins where both are filled; the normalized value stays
    # where the note said nothing.
    assert stored["meanings"] == ["to talk"]
    assert stored["usage_notes"] == "curated"
    assert stored["romaji"] == "hanasu"


def test_a_disagreement_the_merge_cannot_resolve_is_refused_before_anything_is_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(
        tmp_path,
        {
            "verbs": {
                "deck": {"name": "Verbs"},
                "notes": [
                    {
                        "id": "word:話す:はなす",
                        "expression": "話す",
                        "reading": "はなす",
                        "meanings": ["to talk"],
                        "source": {"type": "manual"},
                    }
                ],
            }
        },
    )
    normalized = root / "data" / "normalized" / "vocabulary.json"
    normalized.write_text(
        json.dumps([_raw("話す", "はなす")], ensure_ascii=False), encoding="utf-8"
    )
    before = normalized.read_bytes()
    deck_before = (root / "data" / "decks" / "verbs.yaml").read_bytes()

    assert _migrate(root) == 1

    assert "source" in capsys.readouterr().err
    assert normalized.read_bytes() == before
    assert (root / "data" / "decks" / "verbs.yaml").read_bytes() == deck_before


def test_a_note_with_no_id_migrates_under_the_id_its_cards_already_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(
        tmp_path,
        {
            "verbs": {
                "deck": {"name": "Verbs"},
                "notes": [{"expression": "話す", "reading": "はなす", "meanings": ["to talk"]}],
            }
        },
    )
    before = _built(root / "data" / "decks" / "verbs.yaml", root, monkeypatch)

    assert _migrate(root) == 0

    assert [record["id"] for record in _records(root)] == ["word:話す:はなす"]
    assert _built(root / "data" / "decks" / "verbs.yaml", root, monkeypatch) == before


# --- refusals and the decks left alone --------------------------------------


def test_a_deck_reading_some_other_file_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(
        tmp_path,
        {
            "verbs": {
                "deck": {"name": "Verbs", "source": "../normalized/other.json"},
                "notes": [_raw("話す", "はなす")],
            }
        },
    )
    (root / "data" / "normalized" / "other.json").write_text("[]\n", encoding="utf-8")

    assert _migrate(root) == 1

    assert "other.json" in capsys.readouterr().err
    assert _records(root) == []


def test_an_inline_note_the_deck_filters_out_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(
        tmp_path,
        {
            "verbs": {
                "deck": {"name": "Verbs", "exclude_ids": ["word:話す:はなす"]},
                "notes": [_raw("話す", "はなす"), _raw("行く", "いく")],
            }
        },
    )

    assert _migrate(root) == 1

    assert "word:話す:はなす" in capsys.readouterr().err
    assert _records(root) == []


def test_a_deck_that_already_reads_the_normalized_file_is_not_pinned(
    tmp_path: Path,
) -> None:
    root = _project(
        tmp_path,
        {
            "verbs": {
                "deck": {"name": "Verbs", "source": "../normalized/vocabulary.json"},
                "notes": [_raw("話す", "はなす", source=_shared_source())],
            }
        },
    )

    assert _migrate(root) == 0

    deck = _deck_config(root, "verbs.yaml")["deck"]
    # Its membership rule already covered the migrated record; pinning it would
    # freeze out every future import instead of preserving anything.
    assert "include_ids" not in deck
    assert [record["id"] for record in _records(root)] == ["word:話す:はなす"]


def test_a_deck_that_asks_for_a_migrated_id_by_name_keeps_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(
        tmp_path,
        {
            "verbs": {
                "deck": {"name": "Verbs"},
                "notes": [_raw("話す", "はなす"), _raw("行く", "いく")],
            },
            "wanted": {
                "deck": {
                    "name": "Wanted",
                    "source": "../normalized/vocabulary.json",
                    "include_ids": ["word:話す:はなす"],
                },
                "notes": [],
            },
        },
    )

    assert _migrate(root) == 0

    deck = _deck_config(root, "wanted.yaml")["deck"]
    assert deck["exclude_ids"] == ["word:行く:いく"]
    assert deck["include_ids"] == ["word:話す:はなす"]
    assert "include_ids" in capsys.readouterr().err


def test_a_deck_that_does_not_read_the_normalized_file_is_left_alone(
    tmp_path: Path,
) -> None:
    root = _project(
        tmp_path,
        {
            "verbs": {"deck": {"name": "Verbs"}, "notes": [_raw("話す", "はなす")]},
            "other": {"deck": {"name": "Other"}, "notes": [_raw("行く", "いく")]},
        },
    )
    untouched = (root / "data" / "decks" / "other.yaml").read_bytes()

    assert _migrate(root) == 0

    assert (root / "data" / "decks" / "other.yaml").read_bytes() == untouched


def test_a_deck_whose_content_changes_for_another_deck_says_so(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(
        tmp_path,
        {
            "verbs": {
                "deck": {"name": "Verbs"},
                "notes": [
                    _raw("話す", "はなす", meanings=["to talk"], source=_shared_source())
                ],
            },
            "shared": {
                "deck": {"name": "Shared", "source": "../normalized/vocabulary.json"},
                "notes": [],
            },
        },
    )
    (root / "data" / "normalized" / "vocabulary.json").write_text(
        json.dumps([_raw("話す", "はなす")], ensure_ascii=False), encoding="utf-8"
    )

    assert _migrate(root) == 0

    err = capsys.readouterr().err
    assert "shared.yaml" in err
    assert "word:話す:はなす" in err
    # It already exported that record, so it is not excluded from it.
    assert "exclude_ids" not in _deck_config(root, "shared.yaml")["deck"]


def test_an_unreadable_deck_is_a_warning_not_a_silent_promise(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(
        tmp_path,
        {"verbs": {"deck": {"name": "Verbs"}, "notes": [_raw("話す", "はなす")]}},
    )
    (root / "data" / "decks" / "broken.yaml").write_text("deck: [1, 2\n", encoding="utf-8")

    assert _migrate(root) == 0

    assert "broken.yaml" in capsys.readouterr().err
