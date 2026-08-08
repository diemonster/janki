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
from test_import_ledger import make_unwritable

from japanese_anki import cli, ledger, migrate
from japanese_anki.config import ProjectConfig
from japanese_anki.exporters import anki
from japanese_anki.io import DataError

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
            elif key == "examples":
                # Same tolerance one level down: a stored example carries every
                # schema field, and the note wrote only the ones it had. M2.2's
                # `ExampleSentence.audio` is the first such field — empty until
                # `janki audio` fills it, so the migration still moved the
                # example across unchanged.
                for stored_example, written in zip(
                    by_id[note["id"]][key], value, strict=True
                ):
                    assert stored_example.items() >= written.items()
                    assert stored_example["audio"] == ""
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


def test_a_second_migration_appends_to_exclude_ids_without_eating_the_comments(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The first migration creates `exclude_ids:`; every migration after it is
    # appending an id to a list that now exists. That is an addition like any
    # other, so it must not cost a curated deck file its comment block, its
    # quote styles or its blank lines.
    root = _fixture_project(tmp_path)
    assert _migrate(root) == 0
    guarded = root / "data" / "decks" / "personal-vocabulary.yaml"
    after_first = guarded.read_text(encoding="utf-8")
    (root / "data" / "decks" / "adjectives.yaml").write_text(
        yaml.safe_dump(
            {"deck": {"name": "Adjectives"}, "notes": [_raw("高い", "たかい")]},
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    capsys.readouterr()

    assert _migrate(root, "adjectives.yaml") == 0

    text = guarded.read_text(encoding="utf-8")
    # Every line the first migration left is still there, in order, plus one.
    assert [line for line in text.splitlines() if line not in ("  - word:高い:たかい",)] == (
        after_first.splitlines()
    )
    assert _deck_config(root, "personal-vocabulary.yaml")["deck"]["exclude_ids"] == [
        *VERB_IDS,
        "word:高い:たかい",
    ]
    assert "re-serialized" not in capsys.readouterr().err


def test_a_deck_shape_the_text_edit_cannot_handle_says_the_comments_are_gone(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The parse-back-and-compare safety net still falls back to re-serializing
    # an unrecognised shape, and that deletes hand-written documentation from a
    # curated file. It may do that; it may not do it quietly.
    root = _fixture_project(tmp_path)
    guarded = root / "data" / "decks" / "personal-vocabulary.yaml"
    guarded.write_text(
        "# keep me\n"
        "deck:\n"
        '  name: "Inbox"\n'
        '  source: "../normalized/vocabulary.json"\n'
        "  exclude_ids: [word:古い:ふるい]\n",  # a flow sequence: no item lines to append after
        encoding="utf-8",
    )

    assert _migrate(root) == 0

    err = capsys.readouterr().err
    assert "re-serialized" in err
    assert "personal-vocabulary.yaml" in err
    assert "# keep me" not in guarded.read_text(encoding="utf-8")
    # The meaning is preserved even though the formatting is not.
    assert _deck_config(root, "personal-vocabulary.yaml")["deck"]["exclude_ids"] == [
        "word:古い:ふるい",
        *VERB_IDS,
    ]


def test_a_text_edit_that_would_mean_something_else_is_caught_and_discarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The parse-back-and-compare guard is the only thing standing between a
    # heuristic text edit and a corrupted curated deck file — the module
    # docstring's "the shortcut can never produce a file that means something
    # else" — and deleting it whole used to leave the suite green. Here the
    # edit is well-formed YAML that says the wrong thing, which no shape check
    # upstream can catch: only reading the result back does.
    root = _fixture_project(tmp_path)
    guarded = root / "data" / "decks" / "personal-vocabulary.yaml"
    real_append = migrate._append_to_sequence

    def wrong(lines, block, indent, key, current, value):
        edited = real_append(lines, block, indent, key, current, value)
        if edited is None:
            return None
        return [line.replace("word:", "WRONG:") for line in edited]

    assert _migrate(root) == 0  # creates exclude_ids as a block sequence
    (root / "data" / "decks" / "adjectives.yaml").write_text(
        yaml.safe_dump(
            {"deck": {"name": "Adjectives"}, "notes": [_raw("高い", "たかい")]},
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(migrate, "_append_to_sequence", wrong)
    capsys.readouterr()

    assert _migrate(root, "adjectives.yaml") == 0

    assert "re-serialized" in capsys.readouterr().err
    assert "WRONG:" not in guarded.read_text(encoding="utf-8")
    assert _deck_config(root, "personal-vocabulary.yaml")["deck"]["exclude_ids"] == [
        *VERB_IDS,
        "word:高い:たかい",
    ]


def test_a_rewrite_of_exclude_ids_that_is_not_an_append_falls_back(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # `value[:len(current)] != current` refuses anything but an append. A deck
    # whose owner reordered exclude_ids by hand is exactly that: appending the
    # tail would produce a list nobody asked for, in the wrong order.
    root = _fixture_project(tmp_path)
    guarded = root / "data" / "decks" / "personal-vocabulary.yaml"
    guarded.write_text(
        "# keep me\n"
        "deck:\n"
        '  name: "Inbox"\n'
        '  source: "../normalized/vocabulary.json"\n'
        "  exclude_ids:\n"
        "    - word:古い:ふるい\n"
        "    - word:安い:やすい\n",
        encoding="utf-8",
    )
    # The migration appends to what is there, so force a genuine rewrite by
    # having the deck already name one of the ids about to be added, out of
    # order: the merged list is not a prefix-preserving extension.
    raw = yaml.safe_load(guarded.read_text(encoding="utf-8"))
    reordered = ["word:安い:やすい", "word:古い:ふるい", "word:新しい:あたらしい"]

    assert migrate.rewrite_deck_file(guarded, raw, {"exclude_ids": reordered}) is True

    text = guarded.read_text(encoding="utf-8")
    assert "# keep me" not in text  # the fallback fired and said so via its return
    assert yaml.safe_load(text)["deck"]["exclude_ids"] == reordered


def test_a_comment_at_the_key_indent_falls_back_rather_than_guessing(
    tmp_path: Path,
) -> None:
    # `len(items) != len(current)` refuses a block whose item lines do not
    # correspond one-for-one with the values parsed from it. A comment at the
    # *mapping's* indent, between two items, closes the block for the text scan
    # while YAML reads straight past it — so the scan sees one item where the
    # parse saw two, and nothing in the text can say where the last one ends.
    root = _fixture_project(tmp_path)
    guarded = root / "data" / "decks" / "personal-vocabulary.yaml"
    guarded.write_text(
        "deck:\n"
        '  name: "Inbox"\n'
        '  source: "../normalized/vocabulary.json"\n'
        "  exclude_ids:\n"
        "    - word:古い:ふるい\n"
        "  # migrated out of verbs.yaml\n"
        "    - word:安い:やすい\n",
        encoding="utf-8",
    )
    raw = yaml.safe_load(guarded.read_text(encoding="utf-8"))
    updated = [*raw["deck"]["exclude_ids"], "word:高い:たかい"]

    assert migrate.rewrite_deck_file(guarded, raw, {"exclude_ids": updated}) is True

    text = guarded.read_text(encoding="utf-8")
    assert "# migrated out of verbs.yaml" not in text  # the fallback fired
    assert yaml.safe_load(text)["deck"]["exclude_ids"] == updated


def test_an_item_indent_comment_is_kept_by_the_text_edit(tmp_path: Path) -> None:
    # The other side of the same line: a comment indented with the items does
    # not close the block, so the append still happens and the comment survives.
    root = _fixture_project(tmp_path)
    guarded = root / "data" / "decks" / "personal-vocabulary.yaml"
    guarded.write_text(
        "deck:\n"
        '  name: "Inbox"\n'
        '  source: "../normalized/vocabulary.json"\n'
        "  exclude_ids:\n"
        "    - word:古い:ふるい\n"
        "    # migrated out of verbs.yaml\n"
        "    - word:安い:やすい\n",
        encoding="utf-8",
    )
    raw = yaml.safe_load(guarded.read_text(encoding="utf-8"))
    updated = [*raw["deck"]["exclude_ids"], "word:高い:たかい"]

    assert migrate.rewrite_deck_file(guarded, raw, {"exclude_ids": updated}) is False

    text = guarded.read_text(encoding="utf-8")
    assert "    # migrated out of verbs.yaml" in text
    assert yaml.safe_load(text)["deck"]["exclude_ids"] == updated


def test_a_failed_guard_rewrite_leaves_the_normalized_file_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Guards are written before the records for a reason: a guard rewrite that
    # fails after vocabulary.json is already written leaves every other deck
    # reading that file resolving the migrated records with no exclude_ids —
    # two .apkg files carrying one GUID, the hazard this module exists to
    # prevent.
    root = _fixture_project(tmp_path)
    normalized = root / "data" / "normalized" / "vocabulary.json"
    before = normalized.read_bytes()
    real_write = migrate.atomic_write_text

    def refuse(path: Path, text: str) -> None:
        if Path(path).name == "personal-vocabulary.yaml":
            raise DataError(f"Could not write {path}: Permission denied")
        real_write(path, text)

    monkeypatch.setattr(migrate, "atomic_write_text", refuse)

    assert _migrate(root) == 1

    assert "error:" in capsys.readouterr().err
    assert normalized.read_bytes() == before


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
        # The reference is the record's own source, exactly as `status
        # --rebuild` reconstructs it — not the deck the note used to live in.
        assert book.records[record_id]["sources"] == [
            {
                "type": "manual",
                "ref": "",
                "seen_at": book.records[record_id]["added_at"],
            }
        ]


def test_a_rebuild_after_a_migration_leaves_the_ledger_byte_identical(
    tmp_path: Path,
) -> None:
    # `status --rebuild` is the documented ledger-recovery command and the
    # ledger is a git-tracked file. If migrate wrote a reference of any other
    # shape, one rebuild would append a second, near-duplicate entry to every
    # migrated record and report it as a recovery.
    root = _fixture_project(tmp_path)
    assert _migrate(root) == 0
    before = (root / "data" / "ledger.json").read_text(encoding="utf-8")

    assert cli.main(["--root", str(root), "status", "--rebuild"]) == 0

    assert (root / "data" / "ledger.json").read_text(encoding="utf-8") == before


def test_migrating_after_a_rebuild_registers_nothing_and_says_so(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The same divergence in the other order, and the reporting that goes with
    # it: `record_added` returned False for all three, so claiming three
    # registrations would be a transcript describing work that did not happen.
    root = _fixture_project(tmp_path)
    assert cli.main(["--root", str(root), "status", "--rebuild"]) == 0
    seeded = (root / "data" / "ledger.json").read_text(encoding="utf-8")
    capsys.readouterr()

    assert _migrate(root) == 0

    assert (root / "data" / "ledger.json").read_text(encoding="utf-8") == seeded
    assert "Ledger: registered 0 new record(s) and 0 new source sighting(s)." in (
        capsys.readouterr().out
    )


def test_a_ledger_that_cannot_be_saved_still_gets_a_full_transcript(
    tmp_path: Path, request: pytest.FixtureRequest, capsys: pytest.CaptureFixture[str]
) -> None:
    # Every file but the ledger is already written by this point. Printing only
    # `error: ...` would tell the user the migration did not happen over a deck
    # it has already rewritten.
    #
    # A real unwritable directory, not a monkeypatched `Ledger.save`: a
    # monkeypatch raising LedgerError tests the one shape a real save almost
    # never has. Every single-process I/O failure goes through the atomic
    # writer and comes back a DataError, which used to escape this reporting.
    root = _fixture_project(tmp_path)
    (root / "janki.toml").write_text(
        CONFIG.replace('ledger_file = "data/ledger.json"', 'ledger_file = "ledger/ledger.json"'),
        encoding="utf-8",
    )
    (root / "ledger").mkdir()
    make_unwritable(root / "ledger", request)

    assert _migrate(root) == 1

    captured = capsys.readouterr()
    assert "Migrated 3 inline note(s)" in captured.out
    assert "Ledger: NOT written" in captured.out
    assert "registered 3" not in captured.out
    assert "janki status --rebuild" in captured.err
    assert [record["id"] for record in _records(root)] == sorted(VERB_IDS)


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


def test_a_deck_whose_membership_is_already_closed_gets_no_dead_exclude_ids(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # `resolve_deck_records` applies include_ids first, so a deck that has one
    # can never gain a record from the normalized file. Every exclude_ids entry
    # written into it would be provably dead config — and migrate pins the decks
    # it migrates, so those are exactly the decks that would accumulate it.
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
    assert "exclude_ids" not in deck
    assert deck["include_ids"] == ["word:話す:はなす"]
    # Membership is what include_ids says and nothing else, which is why an
    # exclude entry for 行く would have changed nothing.
    _, resolved = anki.resolve_deck_records(root / "data" / "decks" / "wanted.yaml")
    assert [record.id for record in resolved] == ["word:話す:はなす"]
    # The id it asked for by name is still announced: that one is a real change.
    err = capsys.readouterr().err
    assert "include_ids" in err
    assert "word:話す:はなす" in err


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


def test_migrate_inline_refuses_an_identity_clash_before_any_merge_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The reason `_print_merge_summary`'s no-flag branch is unreachable here.

    migrate-inline prefers the inline note for every non-empty mergeable field,
    so the only conflict it *could* report is an identity one — and
    `_check_merged_faithfully` refuses those outright, because two copies that
    disagree about what a record is cannot be merged by a rule. So the command
    errors instead of printing a conflict summary, which is why no test drives
    that branch through this command: it has no path to it.
    """
    root = tmp_path
    (root / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "vocabulary.json"\n'
        'ledger_file = "ledger.json"\n'
        'deck_dir = "decks"\n'
        'staging_dir = "staging"\n',
        encoding="utf-8",
    )
    existing = {
        "id": "word:ATM:エーティーエム",
        "expression": "ＡＴＭ",
        "reading": "エーティーエム",
        "meanings": ["ATM"],
        "source": {"type": "shirabe", "imported_from": "export.csv"},
    }
    (root / "vocabulary.json").write_text(
        json.dumps([existing], ensure_ascii=False), encoding="utf-8"
    )
    (root / "decks").mkdir()
    (root / "decks" / "d.yaml").write_text(
        "name: D\n"
        "notes:\n"
        "  - id: word:ATM:エーティーエム\n"
        "    expression: ATM\n"
        "    reading: エーティーエム\n"
        "    meanings: [ATM]\n"
        # Same provenance as the normalized copy on purpose: without it the
        # note defaults to a `manual` source, `source` joins the overruled
        # fields, and the refusal this test names would fire on that instead —
        # green even if the identity protection were removed.
        "    source: {type: shirabe, imported_from: export.csv}\n",
        encoding="utf-8",
    )

    code = cli.main(
        ["--root", str(root), "migrate-inline", str(root / "decks" / "d.yaml")]
    )

    captured = capsys.readouterr()
    assert code == 1
    assert "keeps its expression rather than the note's" in captured.err
    assert "Conflicts" not in captured.out
    assert "--prefer-incoming" not in captured.out
