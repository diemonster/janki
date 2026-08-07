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
