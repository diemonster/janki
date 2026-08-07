"""Curation-safe merge: existing values win, imports only fill holes.

The module this replaces asserted the opposite (incoming-wins) — that was the
bug, not the contract. See docs/DESIGN_V2.md "Curation-safe merge".
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from japanese_anki import cli
from japanese_anki.io import DataError, merge_records, save_records_json
from japanese_anki.models import ExampleSentence, SourceReference, VocabularyRecord

FIXTURE = Path(__file__).parent / "fixtures" / "shirabe-sample.csv"


def _curated() -> VocabularyRecord:
    return VocabularyRecord(
        id="word:話す:はなす",
        expression="話す",
        reading="はなす",
        furigana="話[はな]す",
        meanings=["to speak"],
        examples=[ExampleSentence(japanese="毎日話す。", english="I speak every day.")],
        tags=["curated"],
        usage_notes="Curated note",
        source=SourceReference(type="manual", imported_from="hand-written.md", row=3),
    )


def _imported(**overrides: object) -> VocabularyRecord:
    values: dict[str, object] = {
        "id": "word:話す:はなす",
        "expression": "話す",
        "reading": "はなす",
        "meanings": ["to speak", "to talk"],
        "part_of_speech": "verb",
        "tags": ["shirabe"],
        "source": SourceReference(type="shirabe", imported_from="export.csv", row=2),
    }
    values.update(overrides)
    return VocabularyRecord(**values)  # type: ignore[arg-type]


# --- merge_records ----------------------------------------------------------


def test_import_fills_empty_fields_and_never_overwrites_curation() -> None:
    merged, outcomes = merge_records([_curated()], [_imported()])

    record = merged[0]
    assert record.furigana == "話[はな]す"
    assert record.usage_notes == "Curated note"
    assert record.examples == [
        ExampleSentence(japanese="毎日話す。", english="I speak every day.")
    ]
    # meanings were already non-empty: the import does not get to touch them.
    assert record.meanings == ["to speak"]
    # part_of_speech was empty, so the import fills it.
    assert record.part_of_speech == "verb"
    assert outcomes["word:話す:はなす"].filled_fields == ["part_of_speech", "tags"]


# A non-empty, *different* value for every mergeable field, so the both-sides-
# populated branch is exercised for each one. AGENTS.md names examples, notes,
# conjugations and furigana specifically as things an import must never erase.
_CURATED_VALUES: dict[str, object] = {
    "furigana": "話[はな]す",
    "romaji": "hanasu",
    "meanings": ["to speak"],
    "part_of_speech": "verb",
    "verb_group": "godan",
    "transitivity": "intransitive",
    "examples": [ExampleSentence(japanese="毎日話す。", english="I speak every day.")],
    "conjugations": {"past": "話した", "te_form": "話して"},
    "usage_notes": "Curated note",
    "audio": "../media/audio/janki-curated.wav",
    "image": "../media/img/curated.png",
}
_IMPORTED_VALUES: dict[str, object] = {
    "furigana": "話[はなし]す",
    "romaji": "hanashisu",
    "meanings": ["to talk"],
    "part_of_speech": "noun",
    "verb_group": "ichidan",
    "transitivity": "transitive",
    "examples": [ExampleSentence(japanese="CSV の例。", english="From the CSV.")],
    "conjugations": {"past": "WRONG"},
    "usage_notes": "Imported note",
    "audio": "from-import.mp3",
    "image": "from-import.png",
}


@pytest.mark.parametrize("name", sorted(_CURATED_VALUES))
def test_every_populated_field_survives_an_import_that_disagrees(name: str) -> None:
    """One field at a time: curation wins and the disagreement is reported.

    Without this, a merge that unconditionally overwrote examples,
    conjugations, audio or image passed the whole suite.
    """
    curated = _curated()
    existing = [replace(curated, **{name: _CURATED_VALUES[name]})]
    overrides: dict[str, object] = {name: _IMPORTED_VALUES[name]}
    if name != "meanings":
        # Match the curated meanings so the field under test is the only clash.
        overrides["meanings"] = ["to speak"]
    incoming = [_imported(**overrides)]

    merged, outcomes = merge_records(existing, incoming)

    assert getattr(merged[0], name) == _CURATED_VALUES[name]
    outcome = outcomes["word:話す:はなす"]
    assert name in [field_name for field_name, _, _ in outcome.conflicts]
    assert name not in outcome.filled_fields
    assert outcome.label == "conflicting"


def test_merged_records_do_not_alias_the_import() -> None:
    """Filling a field must copy it, or later edits to the import leak in."""
    incoming_examples = [ExampleSentence(japanese="例。", english="Example.")]
    existing = [replace(_curated(), examples=[], usage_notes="")]
    merged, _ = merge_records(existing, [_imported(examples=incoming_examples)])

    incoming_examples.append(ExampleSentence(japanese="MUTATED", english="MUTATED"))

    assert [example.japanese for example in merged[0].examples] == ["例。"]


def test_a_duplicate_id_in_the_stored_file_is_refused_not_silently_dropped() -> None:
    """Two stored records sharing an id would collapse last-wins on merge."""
    first = replace(_curated(), usage_notes="CURATED NOTE A")
    second = replace(_curated(), usage_notes="CURATED NOTE B", examples=[])

    with pytest.raises(DataError) as excinfo:
        merge_records([first, second], [_imported()])

    assert "word:話す:はなす" in str(excinfo.value)


def test_tags_are_a_sorted_union() -> None:
    merged, outcomes = merge_records([_curated()], [_imported(tags=["shirabe", "n5"])])

    assert merged[0].tags == ["curated", "n5", "shirabe"]
    assert "tags" in outcomes["word:話す:はなす"].filled_fields


def test_first_source_sticks() -> None:
    merged, _ = merge_records([_curated()], [_imported()])

    assert merged[0].source.type == "manual"
    assert merged[0].source.imported_from == "hand-written.md"
    assert merged[0].source.row == 3


def test_conflicts_are_reported_and_resolved_toward_the_existing_record() -> None:
    existing = _curated()
    merged, outcomes = merge_records(
        [existing], [_imported(furigana="話[わ]す", usage_notes="Imported note")]
    )

    outcome = outcomes["word:話す:はなす"]
    assert outcome.label == "conflicting"
    assert outcome.conflicts == [
        ("furigana", "話[はな]す", "話[わ]す"),
        ("meanings", ["to speak"], ["to speak", "to talk"]),
        ("usage_notes", "Curated note", "Imported note"),
    ]
    assert merged[0].furigana == "話[はな]す"
    assert merged[0].meanings == ["to speak"]
    assert merged[0].usage_notes == "Curated note"


def test_prefer_incoming_restores_incoming_wins_for_named_fields_only() -> None:
    merged, outcomes = merge_records(
        [_curated()],
        [_imported(usage_notes="Imported note")],
        prefer_incoming=("meanings",),
    )

    outcome = outcomes["word:話す:はなす"]
    assert merged[0].meanings == ["to speak", "to talk"]
    assert "meanings" in outcome.filled_fields
    # Fields not named keep existing-wins, so usage_notes is still a conflict.
    assert outcome.label == "conflicting"
    assert outcome.conflicts == [("usage_notes", "Curated note", "Imported note")]
    assert merged[0].usage_notes == "Curated note"


def test_prefer_incoming_still_ignores_an_empty_incoming_value() -> None:
    merged, outcomes = merge_records(
        [_curated()], [_imported(meanings=[])], prefer_incoming=("meanings",)
    )

    assert merged[0].meanings == ["to speak"]
    assert "meanings" not in outcomes["word:話す:はなす"].filled_fields


@pytest.mark.parametrize("name", ["tags", "source", "id", "expression", "reading"])
def test_prefer_incoming_rejects_protected_fields(name: str) -> None:
    with pytest.raises(DataError) as excinfo:
        merge_records([_curated()], [_imported()], prefer_incoming=(name,))

    assert name in str(excinfo.value)


def test_prefer_incoming_rejects_unknown_fields() -> None:
    with pytest.raises(DataError) as excinfo:
        merge_records([_curated()], [_imported()], prefer_incoming=("meaning",))

    assert "meaning" in str(excinfo.value)
    assert "usage_notes" in str(excinfo.value)  # the message lists valid fields


def test_a_none_valued_field_is_empty_and_gets_filled() -> None:
    # Guards M2.2's `frequency_rank: int | None`: None must read as a hole an
    # import can fill, exactly like "" and [].
    existing = _curated()
    existing.romaji = None  # type: ignore[assignment]

    merged, outcomes = merge_records([existing], [_imported(romaji="hanasu")])

    assert merged[0].romaji == "hanasu"
    assert "romaji" in outcomes["word:話す:はなす"].filled_fields


def test_zero_is_a_value_not_a_hole() -> None:
    # Stand-in for M2.2's `frequency_rank: 0`: a falsy number is data, so the
    # import must report a conflict instead of quietly filling it.
    existing = _curated()
    existing.romaji = 0  # type: ignore[assignment]

    merged, outcomes = merge_records([existing], [_imported(romaji="hanasu")])

    assert merged[0].romaji == 0
    assert ("romaji", 0, "hanasu") in outcomes["word:話す:はなす"].conflicts
    assert "romaji" not in outcomes["word:話す:はなす"].filled_fields


def test_outcomes_cover_exactly_the_incoming_ids() -> None:
    untouched = VocabularyRecord(
        id="word:食べる:たべる", expression="食べる", reading="たべる"
    )
    new = VocabularyRecord(id="word:犬:いぬ", expression="犬", reading="いぬ")

    merged, outcomes = merge_records([_curated(), untouched], [_imported(), new])

    assert set(outcomes) == {"word:話す:はなす", "word:犬:いぬ"}
    assert outcomes["word:犬:いぬ"].label == "added"
    assert [record.id for record in merged] == [
        "word:犬:いぬ",
        "word:話す:はなす",
        "word:食べる:たべる",
    ]


def test_an_id_carried_twice_keeps_both_rows_reports() -> None:
    # One import can list a word twice (two jpdb decks, two CSV rows). The
    # second pass must not erase what the first one reported.
    merged, outcomes = merge_records(
        [_curated()],
        [_imported(), _imported(usage_notes="Imported note")],
    )

    outcome = outcomes["word:話す:はなす"]
    assert set(outcomes) == {"word:話す:はなす"}
    assert outcome.filled_fields == ["part_of_speech", "tags"]
    assert outcome.conflicts == [
        ("meanings", ["to speak"], ["to speak", "to talk"]),
        ("usage_notes", "Curated note", "Imported note"),
    ]
    assert merged[0].part_of_speech == "verb"


def test_an_import_that_adds_nothing_is_unchanged() -> None:
    existing = _curated()
    incoming = _imported(meanings=[], part_of_speech="", tags=["curated"])

    merged, outcomes = merge_records([existing], [incoming])

    outcome = outcomes["word:話す:はなす"]
    assert outcome.label == "unchanged"
    assert outcome.filled_fields == []
    assert merged[0].to_dict() == existing.to_dict()


# --- janki import-shirabe ---------------------------------------------------


def _project(tmp_path: Path, records: list[VocabularyRecord] | None = None) -> Path:
    (tmp_path / "janki.toml").write_text(
        '[paths]\nnormalized_file = "vocabulary.json"\n', encoding="utf-8"
    )
    if records is not None:
        save_records_json(tmp_path / "vocabulary.json", records)
    return tmp_path


def _stored(root: Path) -> dict[str, dict]:
    payload = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    return {record["id"]: record for record in payload}


def _import(root: Path, *extra: str) -> int:
    return cli.main(["--root", str(root), "import-shirabe", str(FIXTURE), *extra])


def test_import_prints_outcome_counts_and_conflicts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(tmp_path, [_curated()])

    assert _import(root) == 0

    out = capsys.readouterr().out
    assert "Merge result: 3 added, 0 filled, 0 unchanged, 1 conflicting" in out
    assert "word:話す:はなす meanings: existing to speak | incoming to speak; to talk" in out
    assert (
        "word:話す:はなす usage_notes: existing Curated note "
        "| incoming Common conversational verb" in out
    )

    record = _stored(root)["word:話す:はなす"]
    assert record["meanings"] == ["to speak"]
    assert record["usage_notes"] == "Curated note"
    assert record["tags"] == ["curated", "godan", "shirabe", "verb"]
    assert record["source"]["type"] == "manual"


def test_import_prefer_incoming_overwrites_the_named_fields(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(tmp_path, [_curated()])

    assert _import(root, "--prefer-incoming", "meanings,usage_notes") == 0

    out = capsys.readouterr().out
    assert "Merge result: 3 added, 1 filled, 0 unchanged, 0 conflicting" in out
    assert "Conflicts" not in out
    record = _stored(root)["word:話す:はなす"]
    assert record["meanings"] == ["to speak", "to talk"]
    assert record["usage_notes"] == "Common conversational verb"
    assert record["furigana"] == "話[はな]す"  # untouched fields still win


@pytest.mark.parametrize("value", ["tags", "frequency_rank"])
def test_import_rejects_bad_prefer_incoming_before_writing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], value: str
) -> None:
    root = _project(tmp_path, [_curated()])

    assert _import(root, "--prefer-incoming", value) == 1

    assert value in capsys.readouterr().err
    assert list(_stored(root)) == ["word:話す:はなす"]


def test_replace_prompts_on_a_tty_and_a_no_answer_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(tmp_path, [_curated()])
    prompts: list[str] = []
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt) or "n")

    assert _import(root, "--replace") == 1

    assert prompts == ["Replace 1 existing records? [y/N] "]
    assert capsys.readouterr().err.strip() == "Aborted: nothing was written."
    assert list(_stored(root)) == ["word:話す:はなす"]


def test_replace_proceeds_when_confirmed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(tmp_path, [_curated()])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "y")

    assert _import(root, "--replace") == 0

    assert "Merge result: 4 added, 0 filled, 0 unchanged, 0 conflicting" in (
        capsys.readouterr().out
    )
    assert _stored(root)["word:話す:はなす"]["usage_notes"] == "Common conversational verb"


def test_replace_without_a_tty_proceeds_unprompted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The prompt is fat-finger protection, not CI protection.
    root = _project(tmp_path, [_curated()])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr("builtins.input", _never_called)

    assert _import(root, "--replace") == 0

    assert _stored(root)["word:話す:はなす"]["meanings"] == ["to speak", "to talk"]


def test_replace_with_yes_skips_the_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path, [_curated()])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", _never_called)

    assert _import(root, "--replace", "--yes") == 0

    assert _stored(root)["word:話す:はなす"]["furigana"] == ""


def test_replace_into_a_fresh_output_file_never_prompts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", _never_called)

    assert _import(root, "--replace") == 0

    assert len(_stored(root)) == 4


def _never_called(prompt: str = "") -> str:
    raise AssertionError(f"input() should not have been called (prompt: {prompt!r})")
