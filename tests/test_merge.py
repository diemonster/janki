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
from japanese_anki.io import (
    MERGEABLE_FIELDS,
    DataError,
    merge_records,
    parse_prefer_incoming,
    save_records_json,
)
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


def test_filling_examples_detaches_the_inner_sentences_too() -> None:
    """Copying only the outer list leaves the ExampleSentence objects shared."""
    incoming_examples = [ExampleSentence(japanese="例。", english="Example.")]
    existing = [replace(_curated(), examples=[], usage_notes="")]
    merged, _ = merge_records(existing, [_imported(examples=incoming_examples)])

    incoming_examples[0].japanese = "MUTATED"

    assert [example.japanese for example in merged[0].examples] == ["例。"]


def test_an_added_record_does_not_alias_the_imports_containers() -> None:
    """replace() is a shallow copy: every container would still be the import's."""
    incoming = _imported(
        examples=[ExampleSentence(japanese="例。", english="Example.")],
        conjugations={"past": "話した"},
    )
    merged, outcomes = merge_records([], [incoming])
    assert outcomes["word:話す:はなす"].label == "added"

    incoming.tags.append("MUTATED")
    incoming.meanings.append("MUTATED")
    incoming.examples.append(ExampleSentence(japanese="MUTATED"))
    incoming.examples[0].japanese = "MUTATED"
    incoming.conjugations["past"] = "MUTATED"
    incoming.source.raw_fields["Word"] = "MUTATED"

    record = merged[0]
    assert record.tags == ["shirabe"]
    assert record.meanings == ["to speak", "to talk"]
    assert [example.japanese for example in record.examples] == ["例。"]
    assert record.conjugations == {"past": "話した"}
    assert record.source.raw_fields == {}


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


def test_the_schema_additions_merge_without_being_enumerated_anywhere() -> None:
    # MERGEABLE_FIELDS is derived from `dataclasses.fields(VocabularyRecord)`,
    # so M2.2's three fields merge with no edit to io.py. Verified here rather
    # than assumed: an enumeration that silently missed them would leave the
    # new fields unfillable and unreported.
    assert {"pitch_accent", "audio_accent", "frequency_rank"} <= set(MERGEABLE_FIELDS)

    merged, outcomes = merge_records(
        [_curated()],
        [_imported(pitch_accent=["LHHH"], audio_accent="LHHL", frequency_rank=0)],
    )

    record = merged[0]
    assert record.pitch_accent == ["LHHH"]
    assert record.audio_accent == "LHHL"
    # 0 is a rank, not a hole: an empty field accepted it.
    assert record.frequency_rank == 0
    filled = outcomes["word:話す:はなす"].filled_fields
    assert {"pitch_accent", "audio_accent", "frequency_rank"} <= set(filled)


def test_a_stored_frequency_rank_of_zero_conflicts_instead_of_being_refilled() -> None:
    # The other half of the same rule, on the field it was written for: once a
    # rank is stored, even the falsy one, a differing import is a conflict.
    existing = _curated()
    existing.frequency_rank = 0

    merged, outcomes = merge_records([existing], [_imported(frequency_rank=5000)])

    assert merged[0].frequency_rank == 0
    assert ("frequency_rank", 0, 5000) in outcomes["word:話す:はなす"].conflicts


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


def test_an_id_added_then_filled_within_one_import_is_still_added() -> None:
    # The store gained a record and no pre-existing curation was touched, so
    # reporting "0 added, 1 filled" would contradict the record count.
    first = VocabularyRecord(
        id="word:犬:いぬ", expression="犬", reading="いぬ", usage_notes="A"
    )
    second = VocabularyRecord(id="word:犬:いぬ", expression="犬", reading="いぬ", romaji="inu")

    merged, outcomes = merge_records([], [first, second])

    outcome = outcomes["word:犬:いぬ"]
    assert outcome.label == "added"
    assert outcome.filled_fields == ["romaji"]
    assert len(merged) == 1
    assert merged[0].usage_notes == "A"
    assert merged[0].romaji == "inu"


def test_an_id_added_then_conflicting_within_one_import_reports_the_conflict() -> None:
    # Guards the fold order: "added" must not mask a conflict the summary
    # would then never mention.
    first = VocabularyRecord(
        id="word:犬:いぬ", expression="犬", reading="いぬ", usage_notes="A"
    )
    second = VocabularyRecord(
        id="word:犬:いぬ", expression="犬", reading="いぬ", usage_notes="B"
    )

    _, outcomes = merge_records([], [first, second])

    outcome = outcomes["word:犬:いぬ"]
    assert outcome.label == "conflicting"
    assert outcome.conflicts == [("usage_notes", "A", "B")]


def test_a_merge_that_would_only_resort_tags_leaves_the_hand_edited_order() -> None:
    # "Unchanged" has to mean the file did not change: normalizing the order of
    # a hand-edited tag list is a write the summary would never admit to.
    existing = replace(_curated(), tags=["zeta", "alpha"])

    merged, outcomes = merge_records(
        [existing], [_imported(tags=["alpha"], meanings=[], part_of_speech="")]
    )

    outcome = outcomes["word:話す:はなす"]
    assert merged[0].tags == ["zeta", "alpha"]
    assert outcome.label == "unchanged"
    assert "tags" not in outcome.filled_fields


def test_prefer_incoming_of_only_separators_is_refused() -> None:
    # The flag was passed but names nothing; merging as if it were absent
    # would hide the typo.
    with pytest.raises(DataError) as excinfo:
        parse_prefer_incoming(",")

    assert "--prefer-incoming" in str(excinfo.value)


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


@pytest.mark.parametrize("value", ["tags", "not_a_field"])
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

    # The prompt names the ledger too: --replace now prunes the entries of the
    # records it discards, and that is not recoverable from the records file.
    assert prompts == ["Replace 1 existing records (their ledger entries go too)? [y/N] "]
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


def test_an_identity_conflict_is_marked_as_a_hand_fix_in_the_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The header's remedy (--prefer-incoming) refuses identity fields, so a
    # conflict line for one must say so instead of sending the user into an
    # error message.
    root = _project(
        tmp_path,
        [
            VocabularyRecord(
                id="word:ATM:エーティーエム",
                expression="ＡＴＭ",  # full width: NFKC-equal to the import, byte-different
                reading="エーティーエム",
                meanings=["cash machine"],
            )
        ],
    )
    source = tmp_path / "atm.csv"
    source.write_text(
        "Word,Reading,Definition\nATM,エーティーエム,cash machine\n", encoding="utf-8"
    )

    assert cli.main(["--root", str(root), "import-shirabe", str(source)]) == 0

    out = capsys.readouterr().out
    assert (
        "word:ATM:エーティーエム expression (identity — resolve by hand; "
        "--prefer-incoming refuses it): existing ＡＴＭ | incoming ATM" in out
    )
    # Non-identity conflicts keep the plain shape the header's remedy fits.
    assert _stored(root)["word:ATM:エーティーエム"]["expression"] == "ＡＴＭ"


def test_eof_at_the_replace_prompt_reads_as_no(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Closed stdin or Ctrl-C at the prompt must abort, not traceback.
    root = _project(tmp_path, [_curated()])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    def _closed(prompt: str) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", _closed)

    assert _import(root, "--replace") == 1

    assert "Aborted: nothing was written." in capsys.readouterr().err
    assert list(_stored(root)) == ["word:話す:はなす"]


def test_replace_recovers_a_corrupt_output_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # --replace is the one command that can recover an unreadable file; an
    # unparseable one must warn and confirm, not block.
    root = _project(tmp_path)
    (root / "vocabulary.json").write_text("{not json", encoding="utf-8")

    assert _import(root, "--replace", "--yes") == 0

    assert "warning: could not read the existing records" in capsys.readouterr().err
    assert len(_stored(root)) == 4


def test_replace_recovers_a_file_with_malformed_record_fields(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Valid JSON whose nested types are wrong (a hand-edit's string `examples`)
    # used to escape as an AttributeError traceback from models.py — from the
    # very command whose point is recovering the file.
    root = _project(tmp_path)
    (root / "vocabulary.json").write_text(
        json.dumps([{"expression": "x", "reading": "x", "examples": "broken"}]),
        encoding="utf-8",
    )

    assert _import(root, "--replace", "--yes") == 0

    err = capsys.readouterr().err
    assert "warning: could not read the existing records" in err
    assert "examples" in err
    assert len(_stored(root)) == 4


def test_a_merge_import_over_a_malformed_file_is_a_clean_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Without --replace nothing can be recovered, but the failure must still be
    # janki's one-line error naming the file and field, not a traceback.
    root = _project(tmp_path)
    (root / "vocabulary.json").write_text(
        json.dumps([{"expression": "x", "reading": "x", "conjugations": "dict form"}]),
        encoding="utf-8",
    )

    assert _import(root) == 1

    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "vocabulary.json" in err
    assert "conjugations" in err


# --- M7.6T authority keys travel with their fields ----------------------------


def test_hold_flags_travel_with_the_examples_they_describe() -> None:
    # furigana_unverified says "nobody checked these", learner_load_hold says
    # "audio must not voice these". If the examples land and either key does
    # not, the store holds sentences with nothing saying so.
    bare = replace(_curated(), examples=[])
    flagged = _imported(
        examples=[ExampleSentence(japanese="毎日日本語を話します。")],
        source=SourceReference(
            type="extract",
            imported_from="page.jpg",
            raw_fields={
                "furigana_unverified": "aaaa1111",
                "learner_load_hold": "bbbb2222",
            },
        ),
    )

    merged, _ = merge_records([bare], [flagged])

    assert merged[0].source.raw_fields["furigana_unverified"] == "aaaa1111"
    assert merged[0].source.raw_fields["learner_load_hold"] == "bbbb2222"


def test_hold_flags_do_not_travel_when_the_examples_do_not() -> None:
    # A flag describing examples that were not kept is a lie in the other
    # direction.
    unfilled = _imported(
        examples=[ExampleSentence(japanese="毎日日本語を話します。")],
        source=SourceReference(
            type="extract",
            imported_from="page.jpg",
            raw_fields={"learner_load_hold": "bbbb2222"},
        ),
    )

    merged, _ = merge_records([_curated()], [unfilled])

    assert "learner_load_hold" not in merged[0].source.raw_fields


def test_a_reviewers_acceptance_travels_with_the_sentence_it_covers() -> None:
    # The stamp is fingerprint-bound to the accepted sentence, so carrying it
    # can never bless other text — and dropping it here made the next AI pass
    # treat the reviewer's own sentence as machine text.
    from japanese_anki.identifiers import short_fingerprint
    from japanese_anki.models import example_accepted

    sentence = "毎日日本語を話します。"
    bare = replace(_curated(), examples=[])
    accepted = _imported(
        examples=[ExampleSentence(japanese=sentence)],
        source=SourceReference(
            type="extract",
            imported_from="page.jpg",
            raw_fields={"example_authority": short_fingerprint(sentence)},
        ),
    )

    merged, _ = merge_records([bare], [accepted])

    assert example_accepted(merged[0], merged[0].examples[0])


def test_a_provisional_mark_travels_with_the_field_it_binds() -> None:
    # An incoming model claim that filled a hole must stay provisional in the
    # merged record: dropping the mark here is what turned unreviewed model
    # glosses into permanently "curated" values the dictionary never revisits.
    from japanese_anki.models import mark_provisional, provisional_fields

    hole = replace(_curated(), meanings=[], part_of_speech="noun (curated)")
    claimed = mark_provisional(
        _imported(
            meanings=["a model gloss"],
            part_of_speech="noun",
            source=SourceReference(type="extract", imported_from="page.jpg"),
        )
    )

    merged, _ = merge_records([hole], [claimed])

    assert merged[0].meanings == ["a model gloss"]
    # meanings filled, so its mark travelled; part_of_speech kept the curated
    # value, so the incoming claim about it did not.
    assert provisional_fields(merged[0]) == ["meanings"]


def test_an_acceptance_of_identical_text_survives_an_unchanged_merge() -> None:
    # The documented remediation path: re-reviewing a legacy record whose
    # staged sentence matches the store fills nothing — and a carry gated on
    # "the field was written" silently un-accepted the reviewer's stamp while
    # the merge reported "unchanged".
    from japanese_anki.identifiers import short_fingerprint
    from japanese_anki.models import example_accepted

    sentence = "毎日日本語を話します。"
    legacy = replace(
        _curated(),
        examples=[ExampleSentence(japanese=sentence)],
        source=SourceReference(type="extract", imported_from="page.jpg"),
    )
    restamped = _imported(
        examples=[ExampleSentence(japanese=sentence)],
        source=SourceReference(
            type="extract",
            imported_from="page.jpg",
            raw_fields={"example_authority": short_fingerprint(sentence)},
        ),
    )

    merged, _ = merge_records([legacy], [restamped])

    assert example_accepted(merged[0], merged[0].examples[0])


def test_a_curated_fill_on_an_extract_record_mints_acceptance() -> None:
    # The merged record keeps its first-seen extract origin, which demands a
    # stamp nobody could type for a Shirabe row — but the incoming sentence is
    # the user's own data, curated by arrival. The fill event is the
    # provenance.
    from japanese_anki.models import example_accepted

    hole = replace(
        _curated(),
        examples=[],
        source=SourceReference(type="extract", imported_from="page.jpg"),
    )
    shirabe = _imported(
        examples=[ExampleSentence(japanese="毎日日本語を話します。")]
    )

    merged, _ = merge_records([hole], [shirabe])

    assert merged[0].source.type == "extract"
    assert example_accepted(merged[0], merged[0].examples[0])


def test_a_curated_fill_clears_the_mark_its_value_replaced() -> None:
    # The old mark bound the old value; surviving the overwrite would make it
    # a standing false claim about text the model never wrote — and the next
    # enrich would report a human edit that never happened.
    from japanese_anki.models import mark_provisional, provisional_entries

    marked = mark_provisional(
        replace(
            _curated(),
            meanings=["a model gloss"],
            part_of_speech="",
            source=SourceReference(type="extract", imported_from="page.jpg"),
        )
    )
    curated_fill = _imported(meanings=["a hand-written meaning"], part_of_speech="")

    merged, _ = merge_records([marked], [curated_fill], ("meanings",))

    assert merged[0].meanings == ["a hand-written meaning"]
    assert provisional_entries(merged[0]) == []
    assert "provisional_fields" not in merged[0].source.raw_fields
