"""The shared import pipeline: merge, records, ledger, and what it prints.

Every importer lands through `cli.run_import`, so the ledger learns about a
record from Milestone 1 rather than Milestone 2, and `import-jpdb` (M2.5)
inherits the whole sequence instead of re-deriving it. See DESIGN_V2
"The ledger" and IMPLEMENTATION_PLAN M1.8.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from japanese_anki import cli, ledger
from japanese_anki.config import ProjectConfig
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
        )
        == 0
    )

    out = capsys.readouterr().out
    assert "Imported 1 vocabulary entries into" in out
    assert (root / "staging" / "jpdb-mining-needs-reading.yaml").exists()
    seen_at = _entries(root)["word:話す:はなす"]["added_at"]
    assert _sources(root, "word:話す:はなす") == [
        {"type": "jpdb", "ref": "deck:Mining", "seen_at": seen_at}
    ]
    assert "word:本:" not in _entries(root)
