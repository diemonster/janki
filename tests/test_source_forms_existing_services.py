"""A printed source table meets the two existing services that index the wire.

``source_forms`` is the first **top-level** record field ``to_dict`` may omit,
and two long-standing services read a record's wire mapping by subscripting it:
``repairs._validate_record_shape`` derives its allowed key set from an empty
record's serialization, and the staged-replacement seam in ``staging.py``
digests and compares ``to_dict()[field]``. Neither helper is a caller's
entry point, so every case here drives the caller that broke — ``janki
repair`` through ``read_safe_document``, and the card-revision
plan/stage/review/promotion path — rather than the helper underneath it.

Nothing here reads Japanese: the cells are opaque strings bound to opaque
column identities, and every assertion is over keys, digests, exit codes and
container shapes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from japanese_anki import cli, repairs, staging
from japanese_anki.application import assistant_card_revision_review, card_change_staging
from japanese_anki.application.assistant_context import AssistantContextBroker
from japanese_anki.application.promotion import plan_promotion
from japanese_anki.config import ProjectConfig
from japanese_anki.io import load_records, save_records_json
from japanese_anki.models import (
    ExampleSentence,
    SourceFormColumn,
    SourceFormsTable,
    SourceReference,
    VocabularyRecord,
)

OPERATION_ID = "c0f6a1e6-1c22-4f2b-8a54-2f4a3f8e0a71"
REQUEST_FINGERPRINT = "b" * 64

#: Opaque owner-assigned identities and the owner's own display labels. The
#: layout is not consulted here; only the canonical field shape is.
PRINTED_TABLE = SourceFormsTable(
    columns=(
        SourceFormColumn(id="col-1", label="Plain"),
        SourceFormColumn(id="col-2", label="Polite"),
    ),
    cells={"col-1": "はなす", "col-2": ""},
)
REPLACEMENT_TABLE = SourceFormsTable(
    columns=(
        SourceFormColumn(id="col-1", label="Plain"),
        SourceFormColumn(id="col-2", label="Polite"),
    ),
    cells={"col-1": "はなす", "col-2": "はなします"},
)


# --------------------------------------------------------------------------
# Defect 1: `janki repair` over a record that carries the field
# --------------------------------------------------------------------------


def _repair_record(**overrides: Any) -> VocabularyRecord:
    values: dict[str, Any] = {
        "id": "word:ねこ:ねこ",
        "expression": "ねこ",
        "reading": "ねこ",
        "meanings": ["cat"],
        "source": SourceReference(type="test", imported_from="case.yaml"),
    }
    values.update(overrides)
    return VocabularyRecord(**values)


def _repair_project(
    tmp_path: Path, records: list[VocabularyRecord]
) -> tuple[Path, Path, Path]:
    (tmp_path / "janki.toml").write_text("", encoding="utf-8")
    normalized = tmp_path / "data" / "normalized" / "vocabulary.json"
    staging_dir = tmp_path / "data" / "staging"
    staging_dir.mkdir(parents=True)
    save_records_json(normalized, records)
    return tmp_path, normalized, staging_dir


def test_repair_check_and_apply_read_a_collection_whose_record_prints_a_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """One promoted table must not make the whole collection unrepairable.

    The romaji repair has nothing to do with the printed forms, but it is read
    through the same ``read_safe_document``: if that reader refuses the
    document, every ordinary repair on every record in it refuses with it.
    """
    root, normalized, _staging_dir = _repair_project(
        tmp_path,
        [
            _repair_record(romaji="wrong", source_forms=PRINTED_TABLE),
            _repair_record(
                id="word:犬:いぬ",
                expression="犬",
                reading="いぬ",
                meanings=["dog"],
                romaji="wrong",
            ),
        ],
    )
    before = json.loads(normalized.read_text(encoding="utf-8"))

    check_arguments = [
        "--root",
        str(root),
        "repair",
        str(normalized),
        "--check",
        "record-romaji-from-reading",
        "--format",
        "json",
    ]
    assert cli.main(check_arguments) == 0, "the reader refused the document"
    checked = json.loads(capsys.readouterr().out)
    assert [
        (change["record_id"], change["field"], change["new"])
        for change in checked["changes"]
    ] == [
        ("word:ねこ:ねこ", "romaji", "neko"),
        ("word:犬:いぬ", "romaji", "inu"),
    ]

    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    applied = cli.main(
        [
            "--root",
            str(root),
            "repair",
            str(normalized),
            "--apply",
            "record-romaji-from-reading",
            "--expected-plan",
            checked["repair_plan_fingerprint"],
            "--format",
            "json",
        ]
    )

    assert applied == 0
    written = load_records(normalized)
    assert [item.romaji for item in written] == ["neko", "inu"]
    assert written[0].source_forms == PRINTED_TABLE
    after = json.loads(normalized.read_text(encoding="utf-8"))
    assert after[0]["source_forms"] == before[0]["source_forms"], (
        "the repair rewrote a field it may not target"
    )
    assert "source_forms" not in after[1], "an absent table stayed absent"


def test_repair_reads_an_active_staging_document_whose_record_prints_a_table(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every staging document a layout-bound extraction writes carries one."""
    root, _normalized, staging_dir = _repair_project(tmp_path, [_repair_record()])
    review = staging_dir / "lesson-8.yaml"
    staging.write_staging(
        review,
        [_repair_record(romaji="wrong", source_forms=PRINTED_TABLE)],
        {"source_file": "lesson-8.pdf", "review_notes": "Nothing here is approved."},
    )

    code = cli.main(
        [
            "--root",
            str(root),
            "repair",
            str(review),
            "--check",
            "record-romaji-from-reading",
            "--format",
            "json",
        ]
    )

    assert code == 0, "the reader refused an ordinary staging review"
    checked = json.loads(capsys.readouterr().out)
    assert [change["field"] for change in checked["changes"]] == ["romaji"]


def test_repair_still_refuses_a_record_field_outside_the_schema(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Admitting the schema field must not admit anything else.

    ``source_form_layout`` is deliberately the near miss: it is a real key the
    patch writes, but into ``source.raw_fields``, never onto the record.
    """
    root, normalized, staging_dir = _repair_project(tmp_path, [_repair_record()])
    raw = json.loads(normalized.read_text(encoding="utf-8"))
    raw[0]["source_form_layout"] = {"layout_id": "layout-1", "revision": 1}
    normalized.write_text(
        json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    with pytest.raises(repairs.RepairError, match="unknown field\\(s\\): source_form_layout"):
        repairs.read_safe_document(root, normalized, staging_dir, normalized)

    refused = cli.main(
        [
            "--root",
            str(root),
            "repair",
            str(normalized),
            "--check",
            "record-romaji-from-reading",
        ]
    )

    assert refused == 1
    assert "unknown field(s): source_form_layout" in capsys.readouterr().err


def test_the_printed_table_is_admitted_to_the_schema_without_becoming_repairable(
    tmp_path: Path,
) -> None:
    """Structural admission is not permission: no repair may target the field."""
    allowlist = frozenset(
        {
            "furigana",
            "romaji",
            "examples[*].furigana",
            "examples[*].romaji",
            "audio",
            "image",
            "frequency_rank",
        }
    )
    assert allowlist == repairs.AUTOMATIC_FIELDS, "the M7.1 automatic-repair allowlist"
    assert all(
        "source_forms" not in declaration.allowed_fields
        and "source_forms" not in declaration.input_fields
        for declaration in repairs.REGISTRY.select(())
    )

    seed = repairs.REGISTRY.get("record-romaji-from-reading")
    assert seed is not None
    with pytest.raises(repairs.RepairError, match="source_forms"):
        repairs.RepairRegistry(
            [replace(seed, code="test-table-target", allowed_fields=("source_forms",))]
        )

    root, normalized, staging_dir = _repair_project(
        tmp_path, [_repair_record(romaji="wrong", source_forms=PRINTED_TABLE)]
    )
    document = repairs.read_safe_document(root, normalized, staging_dir, normalized)
    plan = repairs.build_plan(
        document, repairs.REGISTRY.select(["record-romaji-from-reading"])
    )

    assert [change.field for change in plan.changes] == ["romaji"]


# --------------------------------------------------------------------------
# Defect 2: the staged-replacement seam a `revise` proposal reaches
# --------------------------------------------------------------------------


def _revision_record(**overrides: Any) -> VocabularyRecord:
    values: dict[str, Any] = {
        "id": "word:話す:はなす",
        "expression": "話す",
        "reading": "はなす",
        "romaji": "hanasu",
        "meanings": ["to speak"],
        "part_of_speech": "verb",
        "verb_group": "godan",
        "usage_notes": "Used for speaking a language.",
        "examples": [
            ExampleSentence(
                japanese="日本語を話します。",
                furigana="日本語[にほんご]を 話[はな]します。",
                english="I speak Japanese.",
                register="polite",
            )
        ],
        "source": SourceReference(type="manual", imported_from="lesson-8.csv"),
    }
    values.update(overrides)
    return VocabularyRecord(**values)


def _revision_project(tmp_path: Path, record: VocabularyRecord) -> ProjectConfig:
    (tmp_path / "janki.toml").write_text("", encoding="utf-8")
    normalized = tmp_path / "data" / "normalized"
    normalized.mkdir(parents=True)
    _write_collection(normalized / "vocabulary.json", [record])
    decks = tmp_path / "data" / "decks"
    decks.mkdir(parents=True)
    (decks / "all.yaml").write_text(
        "deck:\n"
        "  name: All words\n"
        "  deck_id: 1234\n"
        "  model_id: 5678\n"
        '  source: "../normalized/vocabulary.json"\n',
        encoding="utf-8",
    )
    return ProjectConfig.load(tmp_path)


def _write_collection(path: Path, records: list[VocabularyRecord]) -> None:
    path.write_text(
        json.dumps([item.to_dict() for item in records], ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def _provenance() -> card_change_staging.CardRevisionProvenance:
    return card_change_staging.CardRevisionProvenance(
        operation_id=OPERATION_ID,
        request_fingerprint=REQUEST_FINGERPRINT,
        provider="claude-code",
        model="claude-opus-5",
    )


def _stage_table_proposal(
    config: ProjectConfig,
    current: VocabularyRecord,
    table: SourceFormsTable | None,
) -> card_change_staging.CardChangeStagingResult:
    """The exact shape a captured `revise` answer decodes into.

    ``field_replacements`` is the wire the model returns, so this drives the
    same ``_proposals_from_field_values`` → ``_changes`` →
    ``field_replacement_block`` path ``run_card_revision`` does, without a paid
    call and without any private content.
    """
    wire = None if table is None else table.to_dict()
    plan = card_change_staging.plan_card_change_staging(
        config,
        [current],
        field_replacements={current.id: {"source_forms": wire}},
        provenance=_provenance(),
    )
    return card_change_staging.stage_card_change_staging(
        config, plan, expected_fingerprint=plan.fingerprint
    )


def _record_owner_review(path: Path, record_ids: list[str]) -> None:
    revision = hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
    staging.record_card_revision_review(path, record_ids, expected_revision=revision)


def test_a_revision_adding_a_printed_table_stages_and_promotes(
    tmp_path: Path,
) -> None:
    """The ordinary case: every pre-S5 record has no table at all."""
    current = _revision_record()
    config = _revision_project(tmp_path, current)

    result = _stage_table_proposal(config, current, PRINTED_TABLE)

    staged, meta = staging.read_staging(result.staging_path)
    assert [item.source_forms for item in staged] == [PRINTED_TABLE]
    assert meta["card_revision"]["fields"] == {current.id: ["source_forms"]}
    assert meta["field_replacements"]["records"] == {
        current.id: {"source_forms": staging.replacement_fingerprint(current, "source_forms")}
    }

    _record_owner_review(result.staging_path, [current.id])
    promotion = plan_promotion(config, result.staging_path)

    assert not promotion.is_blocked, promotion.blocked
    assert len(promotion.merging) == 1
    assert promotion.merging[0].landing.source_forms == PRINTED_TABLE


def test_a_revision_replacing_a_printed_table_stages_and_promotes(
    tmp_path: Path,
) -> None:
    """A nonempty existing value only moves under explicit replacement authority."""
    current = _revision_record(source_forms=PRINTED_TABLE)
    config = _revision_project(tmp_path, current)

    result = _stage_table_proposal(config, current, REPLACEMENT_TABLE)

    staged, meta = staging.read_staging(result.staging_path)
    assert [item.source_forms for item in staged] == [REPLACEMENT_TABLE]
    assert meta["field_replacements"]["records"][current.id]["source_forms"] == (
        staging.replacement_fingerprint(current, "source_forms")
    )
    assert staging.authorized_field_replacements(meta, [current], staged) == {
        current.id: ("source_forms",)
    }

    _record_owner_review(result.staging_path, [current.id])
    promotion = plan_promotion(config, result.staging_path)

    assert not promotion.is_blocked, promotion.blocked
    assert promotion.merging[0].landing.source_forms == REPLACEMENT_TABLE
    assert promotion.merging[0].landing.source_forms.cells["col-2"] == "はなします"


def test_a_printed_table_edited_after_staging_refuses_at_promotion(
    tmp_path: Path,
) -> None:
    """The bound old value is what makes the proposal about a known record."""
    current = _revision_record(source_forms=PRINTED_TABLE)
    config = _revision_project(tmp_path, current)
    result = _stage_table_proposal(config, current, REPLACEMENT_TABLE)
    _record_owner_review(result.staging_path, [current.id])

    edited = replace(
        current,
        source_forms=SourceFormsTable(
            columns=PRINTED_TABLE.columns,
            cells={"col-1": "はなす", "col-2": "typed by hand"},
        ),
    )
    _write_collection(config.normalized_file, [edited])

    refused = plan_promotion(config, result.staging_path)

    assert refused.is_blocked
    assert "field-replacements-stale" in refused.blocked
    assert f"{current.id}.source_forms" in refused.blocked


def test_a_printed_table_removed_after_staging_refuses_at_promotion(
    tmp_path: Path,
) -> None:
    """Removal is the direction that leaves the old wire key absent entirely."""
    current = _revision_record(source_forms=PRINTED_TABLE)
    config = _revision_project(tmp_path, current)
    result = _stage_table_proposal(config, current, REPLACEMENT_TABLE)
    _record_owner_review(result.staging_path, [current.id])

    _write_collection(config.normalized_file, [replace(current, source_forms=None)])

    refused = plan_promotion(config, result.staging_path)

    assert refused.is_blocked
    assert "field-replacements-stale" in refused.blocked
    assert f"{current.id}.source_forms" in refused.blocked


def test_the_replacement_block_refuses_an_old_table_the_record_does_not_carry(
    tmp_path: Path,
) -> None:
    """The block's own old-value guard, in both wire and dataclass spelling."""
    del tmp_path
    current = _revision_record(source_forms=PRINTED_TABLE)
    stale = SourceFormsTable(
        columns=PRINTED_TABLE.columns, cells={"col-1": "stale", "col-2": ""}
    )

    for old in (stale.to_dict(), stale):
        with pytest.raises(staging.StagingError, match="old value is different"):
            staging.field_replacement_block(
                [current],
                {current.id: {"source_forms": (old, REPLACEMENT_TABLE.to_dict())}},
            )

    absent = staging.field_replacement_block(
        [replace(current, source_forms=None)],
        {current.id: {"source_forms": (None, PRINTED_TABLE.to_dict())}},
    )

    assert absent["records"][current.id]["source_forms"] == (
        staging.replacement_fingerprint(replace(current, source_forms=None), "source_forms")
    )


def test_enrichment_dataclass_old_values_still_bind_at_the_same_seam() -> None:
    """`enrich` builds its change map from attributes, not from the wire.

    Both callers of :func:`staging.field_replacement_block` must keep working
    against one representation, so the nested dataclass spelling stays valid.
    """
    current = _revision_record()
    proposed_examples = [
        *current.examples,
        ExampleSentence(
            japanese="友達と話した。",
            furigana="友達[ともだち]と 話[はな]した。",
            english="I spoke with a friend.",
            register="casual",
        ),
    ]

    block = staging.field_replacement_block(
        [current],
        {
            current.id: {
                "examples": (current.examples, proposed_examples),
                "usage_notes": (current.usage_notes, "A better note."),
            }
        },
    )

    assert block["records"][current.id] == {
        "examples": staging.replacement_fingerprint(current, "examples"),
        "usage_notes": staging.replacement_fingerprint(current, "usage_notes"),
    }
    with pytest.raises(staging.StagingError, match="old value is different"):
        staging.field_replacement_block(
            [current],
            {current.id: {"examples": (proposed_examples, current.examples)}},
        )


def test_existing_field_replacement_digests_are_unchanged() -> None:
    """Baselines recorded against the pre-fix reader, pasted as literals.

    A digest that moved would silence every already-staged proposal's
    old-value binding, so these are written down rather than recomputed from
    the code under test.
    """
    current = _revision_record()

    assert {
        name: staging.replacement_fingerprint(current, name)
        for name in ("usage_notes", "examples", "meanings", "frequency_rank", "conjugations")
    } == {
        "usage_notes": "08323f43d5c66c4431543a54c01cfedf55347cf18e050c3139d936638281e82f",
        "examples": "b030664d5187eb2eac35d7a781af3b95539a671a9c751d4ed2858ed267e9d8f3",
        "meanings": "a514d00d662dbe33cdcaf3086771e8db4bbbd49e08e0e64e93a91c5ddb1f605f",
        "frequency_rank": "5d7023eab1c7661db07b70480994a7dc3b2d57656b01eebf62561dd58bdb695e",
        "conjugations": "11480c64cf809de19597ad801a7eb37681f0175a872fe41b1d87b152810d6b01",
    }


# --------------------------------------------------------------------------
# Defect 3: the Assistant review planner is a third reader of the same wire
# --------------------------------------------------------------------------


def _revision_resource_id(config: ProjectConfig) -> str:
    catalog = json.loads(AssistantContextBroker(config).catalog().wire)
    resource = next(
        item
        for item in catalog["data"]["resources"]
        if item.get("proposal_kind") == "card_revision"
    )
    return str(resource["resource_id"])


def test_the_assistant_review_planner_reads_a_proposal_that_adds_a_table(
    tmp_path: Path,
) -> None:
    """`plan_card_revision_review` subscripts both wires the same way.

    It is the third service to index a record's serialization by field name,
    and the only one on the Assistant's route to reviewing a staged revision.
    A record with no table omits the key, so the ordinary add-a-table proposal
    — every pre-S5 record is one — raised a bare `KeyError` that no handler
    converts into a typed refusal, with the paid answer already staged.

    Mutant: index `old_wire[name]`/`new_wire[name]` in
    `assistant_card_revision_review._prepare`.
    """
    current = _revision_record()
    config = _revision_project(tmp_path, current)
    result = _stage_table_proposal(config, current, PRINTED_TABLE)

    review = assistant_card_revision_review.plan_card_revision_review(
        config,
        resource_id=_revision_resource_id(config),
        record_ids=[current.id],
    )

    assert review.projection["selection"]["changes"] == [
        {
            "record_id": current.id,
            "expression": current.expression,
            "field": "source_forms",
            # Absent spelled `None`, the canonical spelling this whole delta
            # establishes — and the same old value the staged digest binds.
            "old_value": None,
            "proposed_value": PRINTED_TABLE.to_dict(),
        }
    ]

    # The review is durable and the promotion it authorizes still lands.
    assistant_card_revision_review.execute_card_revision_review(config, review)
    _staged, reviewed = staging.read_staging(result.staging_path)
    assert reviewed["card_revision_review"]["accepted_record_ids"] == [current.id]
    promotion = plan_promotion(config, result.staging_path)
    assert not promotion.is_blocked, promotion.blocked
    assert promotion.merging[0].landing.source_forms == PRINTED_TABLE


def test_the_review_planner_still_reads_a_proposal_that_replaces_a_table(
    tmp_path: Path,
) -> None:
    """The present-key direction is unchanged; `.get` is not a widening."""
    current = _revision_record(source_forms=PRINTED_TABLE)
    config = _revision_project(tmp_path, current)
    _stage_table_proposal(config, current, REPLACEMENT_TABLE)

    review = assistant_card_revision_review.plan_card_revision_review(
        config,
        resource_id=_revision_resource_id(config),
        record_ids=[current.id],
    )

    assert review.projection["selection"]["changes"] == [
        {
            "record_id": current.id,
            "expression": current.expression,
            "field": "source_forms",
            "old_value": PRINTED_TABLE.to_dict(),
            "proposed_value": REPLACEMENT_TABLE.to_dict(),
        }
    ]


# --------------------------------------------------------------------------
# Defect 4: removing the table is not something a revision can express
# --------------------------------------------------------------------------


def test_a_revision_removing_the_printed_table_refuses_and_scopes_its_route(
    tmp_path: Path,
) -> None:
    """The refusal is right, and what it tells the owner has to be true too.

    `SourceFormsTable.from_dict` canonicalizes both `null` and the empty table
    to absence, and `io.merge_records` reads absence as a hole to keep the
    existing value in. So a removal proposal that was admitted would stage,
    review as a deletion, and then promote to no change at all — a claimed
    success over a card that still prints its old table. Refused where the
    shape is checked instead of being accepted here and ignored three services
    later.

    The remediation half is what this case pins. Every record a revision can
    name is canonical — `_require_canonical_records` has already compared it
    to `data/normalized/vocabulary.json` byte for byte — and `--drop-table`
    writes a study job's staged copies, never that file. A message that sent
    the owner to `janki study curate` for the table they actually asked about
    buys them a second refusal, or an edit to a staged copy while the canonical
    table stays exactly where it was. So the route it names is scoped to the
    unpromoted staged proposal it really reaches, and the message says plainly
    that a canonically stored table is not a revision's to take off. It names
    no desk control, because `_STUDY_JOB_ACTIONS` is closed and has none.

    Mutant: accept `value is None` in
    `card_change_staging._validate_replacement_shape`'s `source_forms` branch,
    drop the empty-table half of the same test, or restore the unscoped
    "Delete it as an explicit local curation edit instead … or the same
    control on that job's desk" remediation sentence.
    """
    current = _revision_record(source_forms=PRINTED_TABLE)
    config = _revision_project(tmp_path, current)

    for removal in (None, {"columns": [], "cells": {}}):
        with pytest.raises(card_change_staging.CardChangeStagingError) as refusal:
            card_change_staging.plan_card_change_staging(
                config,
                [current],
                field_replacements={current.id: {"source_forms": removal}},
                provenance=_provenance(),
            )
        message = str(refusal.value)
        assert "would remove its 'source_forms' table" in message

        # What the refusal actually leaves: the table this names lives in the
        # canonical collection, and no revision takes it off there.
        assert "canonically" in message

        # The one route it names, scoped to the case where it is true.
        route = message[message.index("janki study curate") :]
        assert "--drop-table" in route
        assert "promoted" in route
        assert message.index("canonically") < message.index("janki study curate")

        # No desk control is claimed, here or anywhere.
        assert "desk" not in message

    # Nothing was staged, and the canonical record still carries its table.
    assert not config.staging_dir.exists()
    assert load_records(config.normalized_file)[0].source_forms == PRINTED_TABLE

    # The shape check still admits every real replacement, so the refusal is
    # about removal rather than about the field.
    card_change_staging._validate_replacement_shape(
        current.id, "source_forms", REPLACEMENT_TABLE.to_dict()
    )
