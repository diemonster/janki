from pathlib import Path

from japanese_anki.importers.shirabe import import_file, inspect_file
from japanese_anki.staging import annotations

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


def test_a_kana_only_row_keeps_its_reading_defaulted_id(tmp_path: Path) -> None:
    # Guard: the reading is defaulted to the expression *before* the ID is
    # minted. Moving that defaulting would re-ID every kana-only record already
    # in Anki and orphan its review history.
    source = tmp_path / "kana.csv"
    source.write_text(
        "Word,Reading,Definition\nありがとう,,thank you\nさようなら,さようなら,goodbye\n",
        encoding="utf-8",
    )

    result = import_file(source)

    assert [record.id for record in result.records] == [
        "word:ありがとう:ありがとう",
        "word:さようなら:さようなら",
    ]
    assert result.records[0].reading == "ありがとう"
    assert result.needs_reading == []
    # The fixture's kana row pins the same shape from the committed data.
    assert import_file(FIXTURE).records[3].id == "word:ありがとう:ありがとう"


def test_a_kanji_row_without_a_reading_is_held_back_for_review(tmp_path: Path) -> None:
    source = tmp_path / "export.csv"
    source.write_text(
        "Word,Reading,Definition\n話す,,to speak\n電話,でんわ,telephone\n", encoding="utf-8"
    )

    result = import_file(source)

    assert [record.id for record in result.records] == ["word:電話:でんわ"]
    assert [record.id for record in result.needs_reading] == ["word:話す:"]
    # The malformed ID is kept exactly as minted — it is the review signal
    # `janki validate` surfaces until a human confirms the reading.
    held = result.needs_reading[0]
    assert held.expression == "話す"
    assert held.reading == ""
    assert annotations(held) == {"hold_reason": "missing reading"}
    assert held.source.raw_fields["Definition"] == "to speak"
    # The row is reported, never silently dropped.
    assert any(
        warning.startswith("export.csv:2:") and "held back for reading review" in warning
        for warning in result.warnings
    )
