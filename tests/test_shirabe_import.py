from pathlib import Path

import pytest

from japanese_anki.importers import shirabe
from japanese_anki.importers.shirabe import ShirabeImportError, import_file, inspect_file
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
    # The committed fixture's last row has an empty Reading cell for the same
    # reason, so the defaulting is pinned by the checked-in data too.
    from_fixture = import_file(FIXTURE).records[3]
    assert from_fixture.source.raw_fields["Reading"] == ""
    assert from_fixture.reading == "ありがとう"
    assert from_fixture.id == "word:ありがとう:ありがとう"


def test_the_committed_fixture_mints_the_ids_it_always_has() -> None:
    # Every record id ever minted is permanent: Anki derives a note GUID from
    # it, so a change here silently orphans review history. Widening which rows
    # are held back must never change how a surviving row is identified.
    assert [record.id for record in import_file(FIXTURE).records] == [
        "word:話す:はなす",
        "word:電話:でんわ",
        "word:食べる:たべる",
        "word:ありがとう:ありがとう",
    ]


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


def test_a_supplementary_plane_kanji_row_is_held_back_like_any_other(tmp_path: Path) -> None:
    # 𠮟 (U+20B9F) and 𩸽 (U+29E3D) live above the BMP. A kanji test that stops
    # at U+9FFF calls them kana, defaults the reading to the expression, and
    # mints word:𠮟る:𠮟る — the unfixable id this whole rule exists to prevent.
    source = tmp_path / "jis2004.csv"
    source.write_text(
        "Word,Reading,Definition\n𠮟る,,to scold\n𩸽,,atka mackerel\n", encoding="utf-8"
    )

    result = import_file(source)

    assert result.records == []
    assert [record.id for record in result.needs_reading] == ["word:𠮟る:", "word:𩸽:"]
    assert [record.reading for record in result.needs_reading] == ["", ""]
    assert all(annotations(record) == {"hold_reason": "missing reading"} for record in
               result.needs_reading)


def test_a_row_with_only_a_kanji_reading_is_held_back(tmp_path: Path) -> None:
    # The empty-Word fallback copies the Reading column into the expression.
    # Run before the reading checks and left untested, it turns one malformed
    # column into a well-formed-looking id: word:話す:話す, reading 話す.
    source = tmp_path / "reading-only.csv"
    source.write_text(
        "Word,Reading,Definition\n,話す,to speak\n,ありがとう,thank you\n", encoding="utf-8"
    )

    result = import_file(source)

    # The kana row still imports: only the reading that is not a reading is held.
    assert [record.id for record in result.records] == ["word:ありがとう:ありがとう"]
    assert [record.id for record in result.needs_reading] == ["word:話す:話す"]
    assert annotations(result.needs_reading[0]) == {"hold_reason": "reading contains kanji"}
    assert any(
        "reading for 話す is written in kanji" in warning for warning in result.warnings
    )


def test_a_field_longer_than_the_default_csv_limit_still_imports(tmp_path: Path) -> None:
    # csv's default field cap is 128KB; a long pasted article in a Notes cell
    # exceeds it and used to crash the import with a raw _csv.Error.
    article = "こ" * 200_000
    source = tmp_path / "long.csv"
    source.write_text(
        f"Word,Reading,Definition,Notes\nありがとう,ありがとう,thanks,{article}\n",
        encoding="utf-8",
    )

    result = import_file(source)

    assert [record.id for record in result.records] == ["word:ありがとう:ありがとう"]
    assert result.records[0].usage_notes == article


def test_a_field_over_the_generous_cap_is_a_clean_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Shrink the cap so the failure mode is testable without a gigabyte file.
    monkeypatch.setattr(shirabe, "_CSV_FIELD_LIMIT", 100)
    source = tmp_path / "big.csv"
    source.write_text(
        f"Word,Reading,Definition\nありがとう,ありがとう,{'x' * 200}\n", encoding="utf-8"
    )

    with pytest.raises(ShirabeImportError) as excinfo:
        import_file(source)

    message = str(excinfo.value)
    assert "big.csv" in message
    assert "could not parse the CSV" in message


def _write_late_bad_byte(path: Path, rows_before_bad: int) -> None:
    """A CSV whose first 8KB+ is valid UTF-8 with an invalid byte further on."""
    filler = "".join(
        f"かな{index},かな{index},filler row number {index}\n" for index in range(rows_before_bad)
    )
    payload = ("Word,Reading,Definition\n" + filler).encode("utf-8")
    # Far past the sniffing sample *and* the text layer's read-ahead, so the
    # bad byte is first decoded inside the row loop, not the wrapped open.
    assert len(payload) > 65536
    path.write_bytes(payload + b"\xff\xff,bad,row\n")


def test_a_bad_byte_past_the_sniffing_sample_is_a_clean_import_error(tmp_path: Path) -> None:
    # Only the first 8KB used to be decoded inside the wrapped sample read; a
    # bad byte later surfaced from the row loop as a raw UnicodeDecodeError.
    source = tmp_path / "corrupt.csv"
    _write_late_bad_byte(source, rows_before_bad=2000)

    with pytest.raises(ShirabeImportError) as excinfo:
        import_file(source)

    message = str(excinfo.value)
    assert "corrupt.csv" in message


def test_a_bad_byte_in_the_inspection_sample_is_a_clean_error(tmp_path: Path) -> None:
    source = tmp_path / "corrupt.csv"
    padding = "な" * 64_000  # row 2 pushes the bad row past sample and read-ahead
    payload = f"Word,Reading,Definition\nかな,かな,{padding}\n".encode()
    source.write_bytes(payload + b"\xff\xff,bad,row\n")

    with pytest.raises(ShirabeImportError) as excinfo:
        inspect_file(source, sample_size=5)

    assert "corrupt.csv" in str(excinfo.value)
