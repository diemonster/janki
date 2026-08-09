"""The shared import pipeline: merge, records, ledger, and what it prints.

Every importer lands through `cli.run_import`, so the ledger learns about a
record from Milestone 1 rather than Milestone 2, and `import-jpdb` (M2.5)
inherits the whole sequence instead of re-deriving it. See DESIGN_V2
"The ledger" and IMPLEMENTATION_PLAN M1.8.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import yaml

from japanese_anki import cli, ledger
from japanese_anki.config import ProjectConfig
from japanese_anki.io import DataError
from japanese_anki.models import SourceReference, VocabularyRecord

CSV = "Word,Reading,Definition\n話す,はなす,to speak\n電話,でんわ,telephone\n"


def _project(tmp_path: Path, csv_text: str = CSV) -> tuple[Path, Path]:
    (tmp_path / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "vocabulary.json"\n'
        'ledger_file = "ledger.json"\n'
        'staging_dir = "staging"\n',
        encoding="utf-8",
    )
    source = tmp_path / "export.csv"
    source.write_text(csv_text, encoding="utf-8")
    return tmp_path, source


def _import(root: Path, source: Path, *extra: str) -> int:
    return cli.main(["--root", str(root), "import-shirabe", str(source), *extra])


def make_unwritable(directory: Path, request: pytest.FixtureRequest) -> None:
    """Take away write permission for the rest of the test, then give it back.

    The restore is a finalizer rather than a `finally`: a failing assertion must
    not leave a directory tmp_path cleanup cannot empty.
    """
    if os.geteuid() == 0:
        pytest.skip("root ignores directory modes")
    directory.chmod(0o555)
    request.addfinalizer(lambda: directory.chmod(0o755))


def _unwritable_ledger_project(
    tmp_path: Path, request: pytest.FixtureRequest, csv_text: str = CSV
) -> tuple[Path, Path]:
    """A project whose ledger directory refuses writes, and nothing else does.

    The ledger gets a directory of its own so the import can still write
    vocabulary.json and the staging file — the point is a *late* failure, over
    work that already landed.
    """
    (tmp_path / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "vocabulary.json"\n'
        'ledger_file = "ledger/ledger.json"\n'
        'staging_dir = "staging"\n',
        encoding="utf-8",
    )
    source = tmp_path / "export.csv"
    source.write_text(csv_text, encoding="utf-8")
    (tmp_path / "ledger").mkdir()
    make_unwritable(tmp_path / "ledger", request)
    return tmp_path, source


def _deck(root: Path, stem: str, deck: dict) -> Path:
    deck_dir = root / "data" / "decks"
    deck_dir.mkdir(parents=True, exist_ok=True)
    path = deck_dir / f"{stem}.yaml"
    path.write_text(
        yaml.safe_dump(deck, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return path


def _entries(root: Path) -> dict[str, dict]:
    return json.loads((root / "ledger.json").read_text(encoding="utf-8"))["records"]


def _sources(root: Path, record_id: str) -> list[dict]:
    return _entries(root)[record_id]["sources"]


# --- what an import writes to the ledger ------------------------------------


def test_an_import_registers_every_record_it_landed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, source = _project(tmp_path)

    assert _import(root, source) == 0

    assert sorted(_entries(root)) == ["word:話す:はなす", "word:電話:でんわ"]
    assert "Ledger: registered 2 new record(s) and 2 new source sighting(s)." in (
        capsys.readouterr().out
    )


def test_a_source_reference_carries_only_type_ref_and_date(tmp_path: Path) -> None:
    # Identity is every key but `seen_at`, so a row number or a deck name here
    # would make the same sighting look like a different one to anything that
    # reconstructs references from the record's own source.
    root, source = _project(tmp_path)

    assert _import(root, source) == 0

    references = _sources(root, "word:話す:はなす")
    assert len(references) == 1
    assert set(references[0]) == {"type", "ref", "seen_at"}
    assert references[0]["type"] == "shirabe"
    assert references[0]["ref"] == "export.csv"


def test_the_reference_matches_what_status_rebuild_would_write(tmp_path: Path) -> None:
    # `status --rebuild` derives references from `record.source`. If the import
    # wrote a different shape, the first rebuild after any import would grow a
    # second near-duplicate reference on every record, forever.
    root, source = _project(tmp_path)
    assert _import(root, source) == 0
    before = (root / "ledger.json").read_text(encoding="utf-8")

    assert cli.main(["--root", str(root), "status", "--rebuild"]) == 0

    assert (root / "ledger.json").read_text(encoding="utf-8") == before


def test_a_second_identical_import_leaves_the_ledger_byte_identical(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, source = _project(tmp_path)
    assert _import(root, source) == 0
    first = (root / "ledger.json").read_text(encoding="utf-8")
    capsys.readouterr()

    assert _import(root, source) == 0

    assert (root / "ledger.json").read_text(encoding="utf-8") == first
    assert "Ledger: registered 0 new record(s) and 0 new source sighting(s)." in (
        capsys.readouterr().out
    )


def test_only_genuinely_new_ids_are_registered_as_added(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The ledger already knows this word from an earlier import whose records
    # file is gone. The merge calls it "added" — the ledger must not, or the
    # date the record entered the collection is rewritten every time.
    root, source = _project(tmp_path)
    book = ledger.load(root / "ledger.json")
    book.record_added("word:話す:はなす", at="2020-01-01")
    book.save()

    assert _import(root, source) == 0

    assert _entries(root)["word:話す:はなす"]["added_at"] == "2020-01-01"
    assert "Ledger: registered 1 new record(s) and 2 new source sighting(s)." in (
        capsys.readouterr().out
    )


def test_a_re_import_from_a_second_file_appends_a_reference(tmp_path: Path) -> None:
    # The merge leaves the record's own `source` alone (first source sticks),
    # so this later sighting has nowhere to go but the ledger.
    root, source = _project(tmp_path)
    assert _import(root, source) == 0
    other = tmp_path / "later.csv"
    other.write_text(CSV, encoding="utf-8")

    assert _import(root, other) == 0

    assert [reference["ref"] for reference in _sources(root, "word:話す:はなす")] == [
        "export.csv",
        "later.csv",
    ]
    stored = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    assert stored[0]["source"]["imported_from"] == "export.csv"


def test_replace_still_wires_the_ledger_and_keeps_the_first_dates(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, source = _project(tmp_path)
    assert _import(root, source) == 0
    capsys.readouterr()

    assert _import(root, source, "--replace", "--yes") == 0

    # --replace makes every incoming record "added" again; the ledger knows
    # better on both counts.
    assert "Ledger: registered 0 new record(s) and 0 new source sighting(s)." in (
        capsys.readouterr().out
    )
    assert sorted(_entries(root)) == ["word:話す:はなす", "word:電話:でんわ"]


def test_replace_drops_the_ledger_entries_of_the_records_it_discarded(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Nothing else reconciles the ledger with vocabulary.json. An orphaned entry
    # permanently over-reports the collection, and once `build --only-new` reads
    # `exports` (M5.6) a dead entry silently keeps a re-imported record out of a
    # deck.
    root, source = _project(tmp_path)
    assert _import(root, source) == 0
    smaller = tmp_path / "tiny.csv"
    smaller.write_text("Word,Reading,Definition\n話す,はなす,to speak\n", encoding="utf-8")
    capsys.readouterr()

    assert _import(root, smaller, "--replace", "--yes") == 0

    assert sorted(_entries(root)) == ["word:話す:はなす"]
    assert "Dropped 1 ledger entry for records --replace discarded." in (
        capsys.readouterr().out
    )
    stored = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    assert [record["id"] for record in stored] == ["word:話す:はなす"]


def test_replace_keeps_the_entry_of_a_record_a_deck_still_carries_inline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The collection is the normalized file *plus every deck's inline notes*
    # (status.collect_records). A record --replace dropped from one file but
    # still exported by a deck has not left the collection, and its entry holds
    # `added_at` and `exports` that `status --rebuild` documents as
    # unreconstructible by anything.
    root, source = _project(tmp_path)
    assert _import(root, source) == 0
    _deck(
        root,
        "extra",
        {
            "deck": {"name": "Extra"},
            "notes": [
                {"id": "word:電話:でんわ", "expression": "電話", "reading": "でんわ"}
            ],
        },
    )
    book = ledger.load(root / "ledger.json")
    book.record_export("word:電話:でんわ", "extra", at="2026-01-02")
    book.save()
    smaller = tmp_path / "tiny.csv"
    smaller.write_text("Word,Reading,Definition\n話す,はなす,to speak\n", encoding="utf-8")
    capsys.readouterr()

    assert _import(root, smaller, "--replace", "--yes") == 0

    out = capsys.readouterr().out
    assert "Dropped" not in out
    assert "still in the collection" in out
    assert _entries(root)["word:電話:でんわ"]["exports"] == {"extra": "2026-01-02"}


def test_replace_into_another_file_prunes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # --output can point --replace at a file the collection does not contain,
    # so "discarded from it" says nothing about the records file the ledger is
    # keyed against. Exit 0 and "Dropped 1 ledger entry" was simply false.
    root, source = _project(tmp_path)
    assert _import(root, source) == 0
    other = root / "other.json"
    other.write_text((root / "vocabulary.json").read_text(encoding="utf-8"), encoding="utf-8")
    smaller = tmp_path / "tiny.csv"
    smaller.write_text("Word,Reading,Definition\n話す,はなす,to speak\n", encoding="utf-8")
    capsys.readouterr()

    assert (
        _import(root, smaller, "--output", str(other), "--replace", "--yes") == 0
    )

    out = capsys.readouterr().out
    assert "Dropped" not in out
    assert "not the records file" in out
    assert sorted(_entries(root)) == ["word:話す:はなす", "word:電話:でんわ"]
    stored = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    assert [record["id"] for record in stored] == ["word:話す:はなす", "word:電話:でんわ"]


def test_replace_prunes_nothing_while_a_deck_cannot_be_read(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A deck that will not parse may be the very thing still carrying the id.
    # Absence cannot be proved, so nothing is removed — an orphaned entry is
    # recoverable and a deleted one is not.
    root, source = _project(tmp_path)
    assert _import(root, source) == 0
    (root / "data" / "decks").mkdir(parents=True)
    (root / "data" / "decks" / "broken.yaml").write_text(
        "deck:\nnotes: not-a-list\n", encoding="utf-8"
    )
    smaller = tmp_path / "tiny.csv"
    smaller.write_text("Word,Reading,Definition\n話す,はなす,to speak\n", encoding="utf-8")
    capsys.readouterr()

    assert _import(root, smaller, "--replace", "--yes") == 0

    out = capsys.readouterr().out
    assert "Dropped" not in out
    assert "could not be read in full" in out
    assert sorted(_entries(root)) == ["word:話す:はなす", "word:電話:でんわ"]


def test_replace_over_an_unreadable_records_file_still_asks_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The count is unknown, so `bool(count)` is False — but the file exists and
    # may hold every curated record the user owns. Dropping the `unreadable or`
    # would have --replace overwrite it with no prompt at all.
    root, source = _project(tmp_path)
    (root / "vocabulary.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "n")

    assert _import(root, source, "--replace") == 1

    captured = capsys.readouterr()
    assert "could not read the existing records" in captured.err
    assert "Aborted: nothing was written." in captured.err
    assert (root / "vocabulary.json").read_text(encoding="utf-8") == "{not json"
    assert not (root / "ledger.json").exists()


def test_replace_over_an_unreadable_records_file_proceeds_on_yes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root, source = _project(tmp_path)
    (root / "vocabulary.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "y")

    assert _import(root, source, "--replace") == 0

    assert sorted(_entries(root)) == ["word:話す:はなす", "word:電話:でんわ"]
    stored = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    assert [record["id"] for record in stored] == ["word:話す:はなす", "word:電話:でんわ"]
    assert "Dropped" not in capsys.readouterr().out


def test_replace_keeps_the_entries_of_the_records_it_re_imported(tmp_path: Path) -> None:
    # A record present on both sides was not discarded: its entry keeps its
    # original added_at and its accumulated sources.
    root, source = _project(tmp_path)
    assert _import(root, source) == 0
    book = ledger.load(root / "ledger.json")
    book.record_added("word:話す:はなす", at="2020-01-01")
    assert _import(root, source, "--replace", "--yes") == 0

    assert sorted(_entries(root)) == ["word:話す:はなす", "word:電話:でんわ"]
    assert _entries(root)["word:話す:はなす"]["added_at"] != ""


def test_a_ledger_that_cannot_be_saved_still_prints_the_whole_summary(
    tmp_path: Path, request: pytest.FixtureRequest, capsys: pytest.CaptureFixture[str]
) -> None:
    # vocabulary.json and the staging file are both already written when the
    # ledger is saved. Printing only `error: ...` tells the user the import did
    # not happen over records it has already replaced.
    #
    # The failure is a real one — an unwritable directory — not a monkeypatched
    # `Ledger.save`. A monkeypatch raising LedgerError tests the one shape a
    # real save almost never has (it is reachable only from the concurrent
    # writer check); every single-process I/O failure reaches the atomic writer
    # and comes back as DataError, which used to sail past this reporting
    # entirely.
    root, source = _unwritable_ledger_project(tmp_path, request, CSV + "本,,book\n")

    assert _import(root, source) == 1

    captured = capsys.readouterr()
    assert "Imported 2 source rows into" in captured.out
    assert "Merge result:" in captured.out
    assert "Held 1 row(s)" in captured.out
    assert "Ledger: NOT written" in captured.out
    assert "registered 2" not in captured.out
    assert "janki status --rebuild" in captured.err
    stored = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    assert len(stored) == 2
    assert (root / "staging" / "shirabe-export-needs-reading.yaml").exists()


def test_the_records_are_written_before_the_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The ledger is metadata `status --rebuild` can reconstruct; the records are
    # not reconstructible from anything. Saving the ledger first would leave it
    # claiming records vocabulary.json does not contain.
    root, source = _project(tmp_path)
    book = ledger.load(root / "ledger.json")
    book.record_added("word:古い:ふるい", at="2020-01-01")
    book.save()
    before = (root / "ledger.json").read_text(encoding="utf-8")

    def refuse(path: Path, records: list) -> None:
        raise DataError(f"Could not write {path}: Permission denied")

    monkeypatch.setattr(cli, "save_records_json", refuse)

    assert _import(root, source) == 1

    assert "error:" in capsys.readouterr().err
    assert (root / "ledger.json").read_text(encoding="utf-8") == before


def test_a_record_the_merge_leaves_unchanged_is_not_registered_as_added(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # `record_added` is called only for `added` outcomes. Calling it for every
    # incoming record would report a re-import of an unchanged record as a new
    # arrival — and would rewrite the date the record entered the collection.
    root, source = _project(tmp_path, "Word,Reading,Definition\n話す,はなす,to speak\n")
    assert _import(root, source) == 0
    (root / "ledger.json").unlink()
    capsys.readouterr()

    assert _import(root, source) == 0

    out = capsys.readouterr().out
    assert "Merge result: 0 added, 0 filled, 1 unchanged, 0 conflicting" in out
    assert "Ledger: registered 0 new record(s) and 1 new source sighting(s)." in out


def test_a_held_row_is_not_registered(tmp_path: Path) -> None:
    # Rows without a usable reading go to staging, not to vocabulary.json.
    # Registering them would have the ledger claim janki holds a record that
    # exists nowhere in the collection.
    root, source = _project(tmp_path, CSV + "本,,book\n")

    assert _import(root, source) == 0

    assert sorted(_entries(root)) == ["word:話す:はなす", "word:電話:でんわ"]


def test_an_unreadable_ledger_aborts_before_anything_is_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The ledger is read before the records are replaced and before a staging
    # file is written, so a broken one costs the user nothing but a re-run.
    root, source = _project(tmp_path, CSV + "本,,book\n")
    (root / "ledger.json").write_text("{not json", encoding="utf-8")

    assert _import(root, source) == 1

    assert "error:" in capsys.readouterr().err
    assert not (root / "vocabulary.json").exists()
    assert not (root / "staging").exists()


# --- what the pipeline prints -----------------------------------------------


def test_the_summary_lines_keep_their_order(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Header, merge outcome, what was held back, what the ledger learned, then
    # the importer's warnings. The held-rows message is produced before the
    # records are written and printed here, in reading order.
    root, source = _project(tmp_path, CSV + "本,,book\n")

    assert _import(root, source) == 0

    out = capsys.readouterr().out
    positions = [
        out.index("Imported 2 source rows into"),
        out.index("Merge result:"),
        out.index("Held 1 row(s)"),
        out.index("Ledger: registered"),
        out.index("Warnings:"),
    ]
    assert positions == sorted(positions)


# --- the pipeline is not Shirabe's ------------------------------------------


def test_another_importer_names_its_own_source_unit_and_staging_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The signature M2.5's `import-jpdb` calls: nothing about a CSV file, a
    # filename stem, or the word "row" is baked in.
    root, _ = _project(tmp_path)
    config = ProjectConfig.load(root)
    record = VocabularyRecord(
        id="word:話す:はなす",
        expression="話す",
        reading="はなす",
        meanings=["to speak"],
        source=SourceReference(type="jpdb", imported_from="deck:Mining"),
    )
    held = VocabularyRecord(id="word:本:", expression="本", reading="")

    assert (
        cli.run_import(
            config,
            [record],
            source_type="jpdb",
            source_ref="deck:Mining",
            output_path=config.normalized_file,
            unit="vocabulary entries",
            staging_stem="jpdb-mining",
            needs_reading=[held],
            held_unit="vocabulary entries",
        )
        == 0
    )

    out = capsys.readouterr().out
    assert "Imported 1 vocabulary entries into" in out
    # The held line takes the caller's unit too — a deck sync has no rows.
    assert "Held 1 vocabulary entries whose reading janki cannot use" in out
    assert "row(s)" not in out
    assert (root / "staging" / "jpdb-mining-needs-reading.yaml").exists()
    seen_at = _entries(root)["word:話す:はなす"]["added_at"]
    assert _sources(root, "word:話す:はなす") == [
        {"type": "jpdb", "ref": "deck:Mining", "seen_at": seen_at}
    ]
    assert "word:本:" not in _entries(root)


# --- a misshapen ledger refuses before it costs anything ---------------------


def _misshapen(root: Path, record_id: str = "word:話す:はなす", **keys: object) -> None:
    """Write a ledger whose entry holds a structured key of the wrong type.

    `load` accepts far looser input than the mutators do — it checks only that
    each entry is an object — so a hand-edited or older ledger really can carry
    `"audio": null`, and this module's own comments cite that as a real input.
    """
    payload = {"version": 1, "records": {record_id: {"added_at": "2026-08-01", **keys}}}
    (root / "ledger.json").write_text(json.dumps(payload), encoding="utf-8")


def test_a_misshapen_ledger_stops_an_import_before_anything_is_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The ordering `_land_import` promises: everything that can refuse an import
    happens before `vocabulary.json` is replaced. Refusing from a *mutator*
    instead fires after the records are on disk and after rows have been staged,
    so the command has done nearly all of its work and reports none of it — no
    counts, no merge summary, no held-rows notice, just an error."""
    root, source = _project(tmp_path)
    _misshapen(root, audio=None)

    assert _import(root, source) == 1

    captured = capsys.readouterr()
    assert "'audio' as NoneType, not a list" in captured.err
    assert not (root / "vocabulary.json").exists(), "the records were never replaced"
    assert not (root / "staging").exists(), "and nothing was staged"


def test_the_error_names_a_repair_that_works(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--rebuild` must be able to do what the message tells the user to do.
    While the refusal came from a mutator, `status --rebuild` hit the same raise
    on the same entry, leaving "delete the ledger" as the only way out — and
    that destroys `exports` and `enriched`, which no rebuild can reconstruct."""
    root, source = _project(tmp_path)
    _misshapen(
        root,
        audio=None,
        exports={"verbs": "2026-07-01"},
        enriched=[{"at": "2026-07-01", "kind": "jpdb", "model": "jpdb", "fields": ["reading"]}],
    )
    capsys.readouterr()  # drain the config banner
    assert cli.main(["--root", str(root), "status"]) == 1
    assert "'janki status --rebuild' repairs" in capsys.readouterr().err

    assert cli.main(["--root", str(root), "status", "--rebuild"]) == 0

    out = capsys.readouterr().out
    assert "repaired: word:話す:はなす had 'audio' of the wrong type" in out
    entry = _entries(root)["word:話す:はなす"]
    assert entry["audio"] == [], "the bad key was reset"
    assert entry["exports"] == {"verbs": "2026-07-01"}, "and the history kept"
    assert entry["enriched"][0]["kind"] == "jpdb"
    assert entry["added_at"] == "2026-08-01", "including the original date"

    assert _import(root, source) == 0, "and the import the bad shape blocked now runs"


def test_a_non_dict_exports_is_refused_rather_than_crashing(tmp_path: Path) -> None:
    """`exports` was the one structured key left unchecked, and the only one
    whose bad shape still produced a raw traceback: `setdefault` returns the
    existing list and `[].get(stem)` is an AttributeError, which `main` does not
    catch, so the user gets a stack trace instead of a clean error."""
    root, _ = _project(tmp_path)
    _misshapen(root, exports=[])

    with pytest.raises(ledger.LedgerError, match="'exports' as list, not a dict"):
        ledger.load(root / "ledger.json")

    book = ledger.load(root / "ledger.json", repair=True)
    assert book.repaired == {"word:話す:はなす": ["exports"]}
    assert book.record_export("word:話す:はなす", "verbs") is True


@pytest.mark.parametrize(
    ("key", "value"),
    [("enriched", {"jpdb": {"at": "2026-07-01"}}), ("exports", ["verbs"])],
)
def test_a_repair_never_discards_what_a_rebuild_cannot_put_back(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], key: str, value: object
) -> None:
    """`sources` and `audio` are safe to empty — the rebuild in the same command
    refills them from the records and the media directory. `enriched` and
    `exports` are reconstructible by nothing at all, so emptying them destroys
    exactly the history the refusal message promises to keep: the record reads
    as un-enriched and the next `janki enrich` pays for it again."""
    root, _ = _project(tmp_path)
    _misshapen(root, **{key: value})
    capsys.readouterr()

    assert cli.main(["--root", str(root), "status", "--rebuild"]) == 0

    out = capsys.readouterr().out
    assert f"kept as '{key}_unreadable'" in out, "and says where it went"
    entry = _entries(root)["word:話す:はなす"]
    assert entry[key] == ([] if key == "enriched" else {}), "the key is usable again"
    assert entry[f"{key}_unreadable"] == value, "and the old value survived"


def test_a_repair_of_a_rebuildable_key_says_the_rebuild_will_fill_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The other half: a user must be able to tell a lossless `audio` reset from
    a lossy one, which a message saying only "a structured key" cannot do."""
    root, _ = _project(tmp_path)
    _misshapen(root, sources=None)
    capsys.readouterr()

    assert cli.main(["--root", str(root), "status", "--rebuild"]) == 0

    out = capsys.readouterr().out
    assert "'sources' of the wrong type, reset to empty; the rebuild below" in out
    assert "sources_unreadable" not in out
    assert "sources_unreadable" not in _entries(root)["word:話す:はなす"]
