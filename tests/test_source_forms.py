"""The optional canonical `source_forms` table, from the wire to the card.

It covers exactly one capability — contracts §4.5/§4.6 and DESIGN amendment
D — through the public interfaces that own it: `VocabularyRecord.from_dict` /
`to_dict`, `io.merge_records`, `io.load_records`, the real `exporters.anki`
build read back out of the packaged `collection.anki2`, and the real
`card_preview` HTML. `models.SourceFormsTable` is deliberately never imported:
every case drives the canonical wire `{columns: [{id,label}], cells: {id: str}}`
through `from_dict`, which is the shape §4.5 specifies and the shape a staging
document actually holds.

Two groups, marked by their names:

* **Baseline** tests state invariants that held before the field existed and
  must still hold now. They are here so a red run can be read: if one of these
  fails, the fixture is wrong, not the contract.
* The rest state the field's own contract. Each names, in its docstring, the
  one production mutant it was written to catch.

Nothing here reads Japanese. Every assertion is over identifiers, container
types, byte equality, HTML structure and counts; the Japanese strings are
synthetic fixture input, never judged. The records are built in a temporary
directory, no repository `data/` or `dist/` path is read or written, and no
model, provider or network call is made.

Layout transport, capture recovery, job CAS, the curation barrier and the S6
finish projection are deliberately absent: they belong to the S5 production
owner's other seams, not to this field.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import re
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pytest
import yaml

pytest.importorskip("genanki")

from japanese_anki import io as janki_io  # noqa: E402
from japanese_anki.config import ProjectConfig  # noqa: E402
from japanese_anki.exporters.anki import (  # noqa: E402
    FIELD_NAMES,
    build_deck,
)
from japanese_anki.models import ModelError, VocabularyRecord  # noqa: E402

#: The checkout under test, by this suite's own convention.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_SOURCE = PROJECT_ROOT / "templates"


#: Cited by the two helpers below, so a failure names the clause rather than
#: only the assertion that tripped.
CONTRACT = (
    "contracts §4.5 (`source_forms` on the record) / §4.6 (the Conjugations "
    "field selects it) / DESIGN amendment D"
)

_MISSING = object()


# --- fixture material ---------------------------------------------------------


def _base_wire() -> dict[str, Any]:
    """One synthetic vocabulary record as it is spelled on the wire.

    A fresh mapping per call: several tests mutate their copy, and a shared
    module-level dict would leak one test's edit into the next.
    """
    return {
        "id": "word:話す:はなす",
        "expression": "話す",
        "reading": "はなす",
        "furigana": "話[はな]す",
        "meanings": ["to speak"],
        "part_of_speech": "verb",
        "verb_group": "godan",
        "transitivity": "intransitive",
        "examples": [
            {
                "japanese": "毎日話します。",
                "furigana": "毎日[まいにち] 話[はな]します。",
                "english": "I speak every day.",
                "register": "polite",
            }
        ],
        "conjugations": {"Te-form": "話して", "Past": "話した"},
        "tags": ["lesson"],
        "source": {"type": "manual", "imported_from": "synthetic.csv", "row": 1},
    }


#: The declared columns most tests use. Ids are deliberately opaque and their
#: order is deliberately *not* their sort order, so a table that came back
#: sorted rather than in the declared ordinal order fails.
COLUMNS = [
    {"id": "c9", "label": "Plain"},
    {"id": "c3", "label": "Polite"},
    {"id": "c7", "label": "Negative"},
]


def _table(columns: list[dict[str, str]], cells: dict[str, str]) -> dict[str, Any]:
    """The planned §4.5 wire: `{columns: [{id,label}], cells: {id: str}}`."""
    return {"columns": copy.deepcopy(columns), "cells": dict(cells)}


def _record(source_forms: Any = _MISSING, **overrides: Any) -> VocabularyRecord:
    wire = _base_wire()
    wire.update(overrides)
    if source_forms is not _MISSING:
        wire["source_forms"] = source_forms
    return VocabularyRecord.from_dict(wire)


def _emitted(record: VocabularyRecord) -> dict[str, Any]:
    """The record's `source_forms` payload, or a failure naming §4.5."""
    payload = record.to_dict()
    if "source_forms" not in payload:
        pytest.fail(
            f"`to_dict()` emitted no `source_forms` key. {CONTRACT}. "
            f"Emitted keys: {sorted(payload)}"
        )
    return payload["source_forms"]


def _attached(record: VocabularyRecord) -> Any:
    """The in-memory table, or a failure naming the missing field.

    A bare `record.source_forms` would raise `AttributeError` here, which reads
    in a log like a broken fixture rather than like an unimplemented field.
    """
    value = getattr(record, "source_forms", _MISSING)
    if value is _MISSING:
        pytest.fail(
            "`VocabularyRecord` has no `source_forms` attribute. Not "
            f"implemented yet: {CONTRACT}. Declared fields: "
            f"{[item.name for item in dataclasses.fields(VocabularyRecord)]}"
        )
    return value


# --- a synthetic project the real exporters can build -------------------------


def _project(root: Path) -> ProjectConfig:
    """A scratch repository holding the real templates and nothing else.

    The templates are the shipped ones, so the Conjugations rows asserted below
    are the rows a learner would see. No `data/` or `dist/` path in the
    repository is touched: everything lives under pytest's `tmp_path`.
    """
    (root / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "vocabulary.json"\n'
        'deck_dir = "decks"\n'
        'media_dir = "media"\n'
        'template_dir = "templates/japanese-study"\n'
        'dist_dir = "dist"\n'
        'kanji_notes_file = "kanji_notes.json"\n'
        'patterns_file = "patterns.json"\n',
        encoding="utf-8",
    )
    shutil.copytree(TEMPLATE_SOURCE, root / "templates")
    for name in ("decks", "media", "dist"):
        (root / name).mkdir()
    (root / "vocabulary.json").write_text("[]", encoding="utf-8")
    return ProjectConfig.load(root)


def _deck(root: Path, wires: list[dict[str, Any]]) -> tuple[ProjectConfig, Path]:
    """Write the records as the collection file and declare one word deck."""
    config = _project(root)
    (root / "vocabulary.json").write_text(
        json.dumps(wires, ensure_ascii=False), encoding="utf-8"
    )
    deck_path = root / "decks" / "lesson.yaml"
    deck_path.write_text(
        "deck:\n"
        "  name: Lesson\n"
        '  source: "../vocabulary.json"\n'
        "  cards:\n"
        "    recognition: true\n"
        "    production: true\n"
        "    reading: true\n",
        encoding="utf-8",
    )
    return config, deck_path


def _notes(package: Path) -> tuple[list[tuple[str, str]], dict[str, dict]]:
    """`(guid, fields)` per note and the notetypes, as the package ships them."""
    with ZipFile(package) as archive, tempfile.TemporaryDirectory() as scratch:
        archive.extract("collection.anki2", scratch)
        connection = sqlite3.connect(Path(scratch) / "collection.anki2")
        try:
            rows = connection.execute("select guid, flds from notes").fetchall()
            models = json.loads(
                connection.execute("select models from col").fetchone()[0]
            )
        finally:
            connection.close()
    return rows, models


def _built_fields(root: Path, wire: dict[str, Any]) -> dict[str, str]:
    """Build one record's deck for real and return its note fields by name."""
    config, deck_path = _deck(root, [wire])
    output = root / "dist" / "lesson.apkg"
    build_deck(deck_path, config, output)
    rows, _models = _notes(output)
    assert len(rows) == 1, f"expected one note, packaged {len(rows)}"
    return dict(zip(FIELD_NAMES, rows[0][1].split("\x1f"), strict=True))


def _rows_in(field_html: str) -> list[tuple[str, str]]:
    """The `(label, value)` pairs the Conjugations field draws, in order.

    Parsed from the exporter's own markup rather than asserted as one blob, so
    a test can say "this row is present with an empty value" and "that column
    drew no row at all" as separate facts — which is the whole of §4.6.
    """
    return [
        (match.group(1), match.group(2))
        for match in re.finditer(
            r'<div class="conjugation-row">'
            r'<span class="conjugation-label">(.*?)</span>'
            r'<span class="conjugation-value">(.*?)</span>'
            r"</div>",
            field_html,
            flags=re.DOTALL,
        )
    ]


# =============================================================================
# Baseline — true today, and must stay true once the field lands
# =============================================================================


def test_baseline_an_ordinary_record_declares_no_source_forms_key() -> None:
    """§4.5: "No existing record's serialized bytes change."

    Stated as the exact key set rather than as "the key is absent", so adding
    the field as a *dense* key — `source_forms: null` on every record — fails
    here instead of silently rewriting every line of `vocabulary.json`.
    """
    record = _record()

    payload = record.to_dict()

    assert set(payload) == {
        "id",
        "expression",
        "reading",
        "furigana",
        "romaji",
        "meanings",
        "part_of_speech",
        "verb_group",
        "transitivity",
        "examples",
        "conjugations",
        "tags",
        "usage_notes",
        "audio",
        "image",
        "pitch_accent",
        "audio_accent",
        "frequency_rank",
        "source",
    }
    assert VocabularyRecord.from_dict(payload).to_dict() == payload


def test_baseline_absent_null_and_empty_table_spellings_are_the_same_bytes() -> None:
    """§4.5's one deliberately defined empty table: `None`, an absent key, and
    "no columns *and* no cells" all serialize to the same ordinary record.

    Byte equality rather than object equality: this is a statement about the
    file a merge or a repair writes back, and `json.dumps` is what writes it.
    """
    spellings = [
        _record(),
        _record(source_forms=None),
        _record(source_forms=_table([], {})),
    ]

    written = {
        json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True)
        for record in spellings
    }

    assert len(written) == 1, sorted(written)
    assert "source_forms" not in next(iter(written))


def test_baseline_merging_fills_a_hole_and_reports_a_conflict() -> None:
    """The merge harness itself, over a field that exists today.

    If this fails, the two `source_forms` merge tests below are failing for the
    fixture rather than for the contract.
    """
    existing = _record(conjugations={}, usage_notes="curated")
    incoming = _record(conjugations={"Te-form": "話して"}, usage_notes="from the model")

    merged, outcomes = janki_io.merge_records([existing], [incoming])

    outcome = outcomes[existing.id]
    assert outcome.label == "conflicting"
    assert "conjugations" in outcome.filled_fields
    assert ("usage_notes", "curated", "from the model") in outcome.conflicts
    assert merged[0].conjugations == {"Te-form": "話して"}
    assert merged[0].usage_notes == "curated"


def test_baseline_an_ordinary_card_keeps_its_fields_guid_and_directions(
    tmp_path: Path,
) -> None:
    """§4.6: "`FIELD_NAMES`, `model_id` derivation, GUIDs …, card directions …
    are unchanged."

    The whole note, not a sample of it. An unlabelled record is the shape the
    repository is full of, and the field's promise is that adding it moves
    none of these values.
    """
    import genanki

    wire = _base_wire()
    config, deck_path = _deck(tmp_path, [wire])
    output = tmp_path / "dist" / "lesson.apkg"

    result = build_deck(deck_path, config, output)
    rows, models = _notes(output)

    assert result.card_types == ("recognition", "production", "reading")
    assert [guid for guid, _ in rows] == [genanki.guid_for(wire["id"])]
    notetype = next(iter(models.values()))
    assert [field["name"] for field in notetype["flds"]] == [
        "RecordID",
        "Expression",
        "Reading",
        "Furigana",
        "Romaji",
        "Meanings",
        "PartOfSpeech",
        "VerbGroup",
        "Transitivity",
        "ExampleJapanese",
        "ExampleFurigana",
        "ExampleRomaji",
        "ExampleEnglish",
        "Conjugations",
        "UsageNotes",
        "Audio",
        "Image",
        "ShirabeQuery",
        "Source",
        "PitchAccent",
        "FrequencyRank",
        "ExampleAudio",
        "KanjiInfo",
        "CasualJapanese",
        "CasualFurigana",
        "CasualEnglish",
        "CasualAudio",
    ]
    assert [template["name"] for template in notetype["tmpls"]] == [
        "Recognition",
        "Production",
        "Reading",
    ]
    fields = dict(zip(FIELD_NAMES, rows[0][1].split("\x1f"), strict=True))
    assert fields["RecordID"] == wire["id"]
    assert fields["Expression"] == "話す"
    assert fields["Reading"] == "はなす"
    assert _rows_in(fields["Conjugations"]) == [
        ("Te-form", "話して"),
        ("Past", "話した"),
    ]


def test_baseline_the_real_preview_draws_the_computed_conjugation_rows(
    tmp_path: Path,
) -> None:
    """The owner's review surface, through the real exporters and Anki.

    The preview adds no field logic of its own (`card_preview` overlays
    serialized text into a mirrored project), so this is the fixture proof for
    the preview test below rather than a second exporter assertion.
    """
    pytest.importorskip(
        "anki.collection",
        reason="the `anki` library renders the preview; it is the preview extra",
    )
    from japanese_anki.card_preview import render_card_preview

    config, deck_path = _deck(tmp_path, [_base_wire()])

    preview = render_card_preview(config, deck_path)

    answers = "".join(card.answer_html for card in preview.cards)
    assert _rows_in(answers)[:2] == [("Te-form", "話して"), ("Past", "話した")]


# =============================================================================
# The field's own contract — contracts §4.5/§4.6, DESIGN amendment D
# =============================================================================


def test_the_record_carries_an_optional_source_forms_field() -> None:
    """The canary. §4.5: `VocabularyRecord.source_forms: SourceFormsTable | None
    = None`.

    Separate from every behaviour below so one line of the log says whether the
    field exists at all.
    """
    names = [item.name for item in dataclasses.fields(VocabularyRecord)]

    assert "source_forms" in names, CONTRACT
    assert _attached(_record()) is None


def test_declared_columns_with_no_cells_stay_present() -> None:
    """§4.5: "A table that declares columns is never empty, so provided columns
    are never silently discarded."

    The counterpart of the omission rule: only *no columns and no cells* is the
    empty table. A source that printed three headings and no forms for this
    word still recorded three headings.
    """
    record = _record(source_forms=_table(COLUMNS, {}))

    emitted = _emitted(record)

    assert emitted["columns"] == COLUMNS
    assert emitted["cells"] == {}
    assert _attached(record) is not None


def test_a_blank_cell_is_kept_and_an_omitted_one_stays_absent() -> None:
    """§4.5: "preserves every supplied cell **verbatim** — no trim, no
    blank-dropping", and DESIGN amendment D: "A blank printed cell is a declared
    row with an empty value; an absent column stays absent."

    This is the exact distinction `conjugations` cannot carry
    (`models.py:256-262` drops a blank value), and the reason the field exists
    rather than being folded into that map.
    """
    record = _record(source_forms=_table(COLUMNS, {"c9": "", "c3": "話します"}))

    emitted = _emitted(record)

    assert emitted["cells"] == {"c9": "", "c3": "話します"}
    assert "c7" not in emitted["cells"]


def test_cell_text_survives_a_round_trip_byte_for_byte() -> None:
    """The verbatim rule against the values a trim would quietly change.

    Whitespace only, an ideographic space, a leading and a trailing space and a
    combining mark: each one is a different way the existing `.strip()` and
    truthiness filters would rewrite a printed cell. Not a Japanese judgement —
    the assertion is string identity.
    """
    cells = {
        "c9": " 　",
        "c3": "　話します ",
        "c7": "é",
    }
    record = _record(source_forms=_table(COLUMNS, cells))

    emitted = _emitted(record)
    again = VocabularyRecord.from_dict(record.to_dict())

    assert emitted["cells"] == cells
    assert _emitted(again)["cells"] == cells


def test_two_columns_may_share_a_display_label_and_keep_their_order() -> None:
    """DESIGN amendment D: "two columns may share a display label because their
    identities differ", and §4.5's ordered `columns` tuple.

    The shared label is the case a label-keyed map cannot represent at all; the
    scrambled ids are the case a sorted or set-backed `columns` cannot.
    """
    columns = [
        {"id": "z1", "label": "Plain"},
        {"id": "a2", "label": "Plain"},
        {"id": "m3", "label": "Polite"},
    ]
    cells = {"z1": "話す", "a2": "話さない", "m3": "話します"}
    record = _record(source_forms=_table(columns, cells))

    emitted = _emitted(record)

    assert emitted["columns"] == columns
    assert [column["id"] for column in emitted["columns"]] == ["z1", "a2", "m3"]
    assert emitted["cells"] == cells


def test_a_duplicate_column_id_refuses() -> None:
    """§4.5: `from_dict` "refuses a duplicate column id".

    Structural, not semantic: two identities that are the same identity make
    the cell map ambiguous, which is a fact about the artifact.
    """
    columns = [{"id": "c9", "label": "Plain"}, {"id": "c9", "label": "Polite"}]

    with pytest.raises(ModelError):
        _record(source_forms=_table(columns, {"c9": "話す"}))


def test_a_cell_id_absent_from_the_columns_refuses() -> None:
    """§4.5: `from_dict` "refuses a cell id absent from `columns`".

    The record-level half of the rule; the request-level half (a model key
    outside the layout's identities) is the layout owner's, not this field's.
    """
    with pytest.raises(ModelError):
        _record(source_forms=_table(COLUMNS, {"c9": "話す", "cX": "話しました"}))


def test_a_structural_refusal_names_the_column_id_it_refused() -> None:
    """Assumption, not a quoted clause — see the handoff.

    §4.5 says both cases refuse but names neither the exception type nor the
    message. This asserts the narrowest reading of the repository's own rule
    that an input is never discarded silently: the offending identity appears
    in the text a person will read. Drop this test rather than widen the
    contract if the S5 owner settles it differently.
    """
    with pytest.raises(ModelError) as unknown:
        _record(source_forms=_table(COLUMNS, {"cX": "話しました"}))
    with pytest.raises(ModelError) as duplicate:
        _record(
            source_forms=_table(
                [{"id": "c9", "label": "Plain"}, {"id": "c9", "label": "Polite"}], {}
            )
        )

    assert "cX" in str(unknown.value)
    assert "c9" in str(duplicate.value)


def test_the_emitted_table_is_plain_json_containers() -> None:
    """§4.5: `to_dict` "emits **plain JSON containers** for the field — a list
    of column mappings, not the dataclass tuple `asdict` would hand through".

    Types, exactly, because `json.dumps` accepts a tuple and `yaml.safe_dump`
    accepts one too: both would hide a leaked `asdict` shape here and expose it
    later, in the writer that does not.
    """
    record = _record(source_forms=_table(COLUMNS, {"c9": "話す"}))

    emitted = _emitted(record)

    assert type(emitted) is dict
    assert type(emitted["columns"]) is list
    assert [type(column) for column in emitted["columns"]] == [dict, dict, dict]
    assert {type(key) for column in emitted["columns"] for key in column} == {str}
    assert type(emitted["cells"]) is dict
    assert {type(key) for key in emitted["cells"]} == {str}
    assert {type(value) for value in emitted["cells"].values()} == {str}
    assert set(emitted) == {"columns", "cells"}


def test_the_json_and_yaml_collection_writers_agree(tmp_path: Path) -> None:
    """§4.5: "Both YAML writers and the JSON collection writer then see the
    shapes they already write."

    Through `io.load_records`, which is the reader both the deck resolver and
    the staging path go through, so this covers the real file round trip rather
    than a serializer identity.
    """
    record = _record(source_forms=_table(COLUMNS, {"c9": "", "c3": "話します"}))
    payload = record.to_dict()
    _emitted(record)  # fail here, not vacuously below, while the field is absent

    as_json = tmp_path / "vocabulary.json"
    as_json.write_text(json.dumps([payload], ensure_ascii=False), encoding="utf-8")
    as_yaml = tmp_path / "vocabulary.yaml"
    yaml_text = yaml.safe_dump([payload], allow_unicode=True, sort_keys=True)
    as_yaml.write_text(yaml_text, encoding="utf-8")

    from_json = janki_io.load_records(as_json)
    from_yaml = janki_io.load_records(as_yaml)

    # A leaked `asdict` tuple still dumps as a YAML sequence, so the tag check
    # is about anything else the dataclass could hand through.
    assert "!!python" not in yaml_text
    assert [item.to_dict() for item in from_json] == [payload]
    assert [item.to_dict() for item in from_yaml] == [payload]
    assert _emitted(from_yaml[0])["cells"] == {"c9": "", "c3": "話します"}


def test_filling_the_hole_deep_copies_the_incoming_table() -> None:
    """§4.5: "`_copy_value` is `copy.deepcopy`, which detaches a frozen nested
    dataclass".

    `SourceFormsTable.cells` is a plain mutable dict inside a frozen wrapper,
    so `frozen=True` detaches nothing. `io._copy_value` is what keeps the
    merged store from aliasing an import the caller may still be writing to.
    """
    existing = _record()
    incoming = _record(source_forms=_table(COLUMNS, {"c9": "話す"}))

    merged, outcomes = janki_io.merge_records([existing], [incoming])

    outcome = outcomes[existing.id]
    assert "source_forms" in outcome.filled_fields
    landed = _attached(merged[0])
    supplied = _attached(incoming)
    assert landed is not supplied
    assert landed.cells is not supplied.cells
    supplied.cells["c9"] = "MUTATED"
    supplied.cells["c3"] = "ADDED"
    assert _emitted(merged[0])["cells"] == {"c9": "話す"}


def test_a_conflicting_table_keeps_the_existing_record_and_reports_it() -> None:
    """§4.5: "Two differing non-empty tables are reported as a conflict with the
    existing value kept, and `source` remains the first sighting's."

    DESIGN amendment D's "never silently overridden", at the merge that would
    do the overriding.
    """
    existing = _record(source_forms=_table(COLUMNS, {"c9": "話す"}))
    incoming = _record(
        source_forms=_table(COLUMNS, {"c9": "はなす"}),
        source={"type": "extract", "imported_from": "part-2.pdf", "row": 4},
    )

    merged, outcomes = janki_io.merge_records([existing], [incoming])

    outcome = outcomes[existing.id]
    assert outcome.label == "conflicting"
    assert "source_forms" not in outcome.filled_fields
    assert [name for name, _old, _new in outcome.conflicts] == ["source_forms"]
    assert _emitted(merged[0])["cells"] == {"c9": "話す"}
    assert merged[0].source.imported_from == "synthetic.csv"


def test_a_declared_but_unfilled_table_is_a_value_the_merge_will_not_fill() -> None:
    """§4.5: "`is_empty` is true only for `None` and empty containers, so a
    present table is a value rather than a hole. That is why the empty spelling
    must not exist in memory as a present object: it would look non-empty to
    the merge."

    The other side of that sentence: a table that declares columns and holds no
    cells *is* present, so a second source's forms do not quietly land on top
    of the first source's headings.
    """
    existing = _record(source_forms=_table(COLUMNS, {}))
    incoming = _record(source_forms=_table(COLUMNS, {"c9": "話す"}))

    merged, outcomes = janki_io.merge_records([existing], [incoming])

    outcome = outcomes[existing.id]
    assert outcome.label == "conflicting"
    assert _emitted(merged[0])["cells"] == {}


def test_the_exporter_prefers_a_present_table_over_the_computed_map(
    tmp_path: Path,
) -> None:
    """§4.6: "the note builder selects `record.source_forms` when present and
    `record.conjugations` otherwise", and "A record whose printed table is
    present and whose `conjugations` later fills from the dictionary engine
    keeps both; only the printed table renders."

    Through the real build and the packaged note, because the field a learner
    reads is the one the exporter wrote, not the one a helper returned.
    """
    wire = _base_wire()
    wire["source_forms"] = _table(COLUMNS, {"c9": "話す", "c3": "話します"})

    fields = _built_fields(tmp_path, wire)

    assert _rows_in(fields["Conjugations"]) == [("Plain", "話す"), ("Polite", "話します")]
    assert "Te-form" not in fields["Conjugations"]
    assert "話して" not in fields["Conjugations"]


def test_the_card_draws_a_blank_row_and_omits_an_absent_column(
    tmp_path: Path,
) -> None:
    """§4.6: "A blank cell renders its declared row with an empty value; an
    absent column renders no row."

    The printed blank is study content — the source said this form is not used
    — so it reaches the card as a row, while a column the source never filled
    for this word does not.
    """
    wire = _base_wire()
    wire["source_forms"] = _table(COLUMNS, {"c9": "話す", "c3": ""})

    fields = _built_fields(tmp_path, wire)

    assert _rows_in(fields["Conjugations"]) == [("Plain", "話す"), ("Polite", "")]


def test_an_all_absent_table_suppresses_the_computed_fallback(
    tmp_path: Path,
) -> None:
    """The selection rule at its edge: the table is present, so it selects; it
    declares no filled row, so the field is empty.

    A fallback here would print derived forms under a source that printed none
    — the silent override amendment D forbids, arriving through the exporter
    instead of through the merge.
    """
    wire = _base_wire()
    wire["source_forms"] = _table(COLUMNS, {})

    fields = _built_fields(tmp_path, wire)

    assert fields["Conjugations"] == ""


def test_the_real_preview_draws_the_selected_source_form_rows(
    tmp_path: Path,
) -> None:
    """§4.6: "`card_preview` builds through the real exporters, so the owner's
    HTML preview shows the real rendered rows rather than a summary."

    The owner's card review is the HTML, so the selection has to hold there and
    not only in the package.
    """
    pytest.importorskip(
        "anki.collection",
        reason="the `anki` library renders the preview; it is the preview extra",
    )
    from japanese_anki.card_preview import render_card_preview

    wire = _base_wire()
    wire["source_forms"] = _table(COLUMNS, {"c9": "話す", "c3": ""})
    config, deck_path = _deck(tmp_path, [wire])

    preview = render_card_preview(config, deck_path)

    answers = "".join(card.answer_html for card in preview.cards)
    assert ("Plain", "話す") in _rows_in(answers)
    assert ("Polite", "") in _rows_in(answers)
    assert "Te-form" not in answers
