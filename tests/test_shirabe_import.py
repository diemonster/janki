from pathlib import Path

from japanese_anki.importers.shirabe import import_file, inspect_file


FIXTURE = Path(__file__).parent / "fixtures" / "shirabe-sample.csv"


def test_inspection_detects_expected_columns() -> None:
    result = inspect_file(FIXTURE)
    assert result.mapping["expression"] == "Word"
    assert result.mapping["reading"] == "Reading"
    assert result.mapping["meanings"] == "Definition"
    assert "Frequency" in result.unknown_headers


def test_import_preserves_unknown_fields_and_splits_meanings() -> None:
    result = import_file(FIXTURE)
    assert len(result.records) == 4
    record = result.records[0]
    assert record.id == "word:話す:はなす"
    assert record.meanings == ["to speak", "to talk"]
    assert "shirabe" in record.tags
    assert record.source.raw_fields["Frequency"] == "120"
