"""Owner-bound layouts, cross-source curation, and the promotion barrier.

Three capabilities meet in this module because they share one job document and
one lock order:

* `study_job.append_layout` — the sole creator of an immutable
  `(layout_id, revision)`, appending the revision and repointing its part
  binding in one compare-and-swap write;
* `staging.prepare_record_update` / `apply_prepared_update` and
  `application/study_curation` — one owner decision, recorded before its first
  write, applied to every occurrence of an identity across a job's parts;
* the curation barrier in `promotion.decide_promotion` **and**
  `promotion.execute_promotion`, under the shared coordination guard that every
  mutation entry takes before any other janki lock.

Every provider is faked and no model, provider or network is reached. Nothing
here reads Japanese: every assertion is over identifiers, hashes, byte
equality, key sets and counts. Each case's docstring names the one production
mutant it was written to catch.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any

import pytest
from test_extraction_batch import _stream as _settled_stream
from test_revision_provider import FakeClaudeRunner, _which
from test_workbench_fixtures import RESPONSES

from conftest import seed_prompts
from japanese_anki import claude_client, extract, staging
from japanese_anki.application import (
    assistant_context,
    extraction_batch,
    promotion,
    source_parts,
    study_curation,
    study_job,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.models import VocabularyRecord

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL = "claude-opus-5"

#: The owner's own layout, as they would write it into a file. Opaque
#: identities, printed positions, the exact printed strings they recorded and
#: the display labels they chose. Nothing derives any of it.
LAYOUT_WIRE: dict[str, Any] = {
    "layout_id": "layout-7c1f2a",
    "revision": 1,
    "columns": [
        {
            "column_id": "col-9f3a71",
            "ordinal": 1,
            "label_witnesses": ["plain form "],
            "display_label": "Plain",
        },
        {
            "column_id": "col-2b8d04",
            "ordinal": 2,
            "label_witnesses": ["polite form", "polite （ます）"],
            "display_label": "Polite",
        },
    ],
}


def _layout(**overrides: Any) -> extract.TableLayout:
    wire = json.loads(json.dumps(LAYOUT_WIRE))
    wire.update(overrides)
    return extract.TableLayout.from_wire(wire)


# --- a scratch project with a real corpus, deck and staging ---------------------


def _project(tmp_path: Path) -> ProjectConfig:
    seed_prompts(tmp_path)
    (tmp_path / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "vocabulary.json"\n'
        'deck_dir = "decks"\n'
        'media_dir = "media"\n'
        'template_dir = "templates/japanese-study"\n'
        'dist_dir = "dist"\n'
        'staging_dir = "staging"\n'
        'scan_inbox = "inbox"\n'
        'patterns_file = "patterns.json"\n'
        'operations_file = "operations.json"\n'
        "[ai]\n"
        'extract_provider = "claude-code"\n'
        f'extract_model = "{MODEL}"\n',
        encoding="utf-8",
    )
    shutil.copytree(
        PROJECT_ROOT / "templates", tmp_path / "templates", dirs_exist_ok=True
    )
    (tmp_path / "decks").mkdir(exist_ok=True)
    (tmp_path / "media").mkdir(exist_ok=True)
    (tmp_path / "vocabulary.json").write_text("[]", encoding="utf-8")
    return ProjectConfig.load(tmp_path)


def _no_api(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the Anthropic API was reached")

    monkeypatch.setattr(claude_client, "prepare_paid_client", refuse)
    monkeypatch.setattr(claude_client, "parse_call", refuse)


def _deck(config: ProjectConfig) -> Path:
    path = config.deck_dir / "lesson.yaml"
    path.write_text(
        "deck:\n"
        "  name: Lesson\n"
        "  source: ../vocabulary.json\n"
        "  cards:\n"
        "    recognition: true\n",
        encoding="utf-8",
    )
    return path


def _parent(config: ProjectConfig) -> Path:
    path = config.scan_inbox / "book.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.7 book")
    return path


RECIPE_ID = "3f1a0c1e-2b8d-4c6a-9f0e-5d4c3b2a1908"


class _PlannedPart:
    """One planned part, in the shape the publication writer reads."""

    def __init__(self, ordinal: int, name: str, data: bytes) -> None:
        self.ordinal = ordinal
        self.target_name = name
        self.sha256 = hashlib.sha256(data).hexdigest()
        self.byte_length = len(data)
        self.page_index = ordinal - 1
        self.page_rotate = 0
        self.page_size_pt = (612.0, 792.0)
        self.pixel_rect = (0, 0, 100, 100)
        self.regions: tuple[Any, ...] = ()
        self.thumbnail_png_base64 = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "target_name": self.target_name,
            "sha256": self.sha256,
            "byte_length": self.byte_length,
            "page_index": self.page_index,
            "page_rotate": self.page_rotate,
            "page_size_pt": list(self.page_size_pt),
            "pixel_rect": list(self.pixel_rect),
            "regions": [list(region) for region in self.regions],
        }


def _job(config: ProjectConfig) -> study_job.StudyJob:
    return study_job.open_study_job(
        config,
        kind="source_extraction",
        parent_source=_parent(config),
        deck_path=_deck(config),
    )


def _publish(
    config: ProjectConfig,
    job_id: str,
    payloads: dict[str, bytes],
    *,
    recipe_id: str = RECIPE_ID,
    plan_fingerprint: str = "f" * 64,
) -> None:
    """Publish these parts into the corpus under one real receipt.

    The plan is hand-built in the shape the publication writer reads, exactly
    as `test_study_job` does: what these cases are about is the job's binding
    and the staged rows, not PDFium's output, and a real render would make them
    depend on an optional dependency.

    A receipt is immutable and job-independent, so a second job publishing a
    different part set brings its own recipe id.
    """
    parent = config.scan_inbox / "book.pdf"
    parts = tuple(
        _PlannedPart(ordinal, name, data)
        for ordinal, (name, data) in enumerate(payloads.items(), start=1)
    )
    plan = source_parts.SourcePartsPlan(
        recipe_id=recipe_id,
        parent_name=parent.name,
        parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
        renderer="pdfium",
        renderer_version="1",
        encoder="pillow",
        encoder_version="1",
        render_dpi=200,
        plan_fingerprint=plan_fingerprint,
        parts=parts,
        recipe_sha256="a" * 64,
        payloads=tuple(payloads.values()),
    )
    study_job.publish_job_source_parts(
        config, job_id, plan, publish_token=plan.plan_fingerprint
    )


#: What each published part holds. Bytes, not pages: the publication writer
#: puts exactly these into the corpus, and the request identity is computed
#: over them.
PART_PAYLOADS: dict[str, bytes] = {
    "part-1.png": b"\x89PNG part one",
    "part-2.png": b"\x89PNG part two",
}


def _lone_part(config: ProjectConfig) -> Path:
    """One durable corpus part, for the cases that need no job."""
    path = config.scan_inbox / "part-1.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(PART_PAYLOADS["part-1.png"])
    return path


#: The three cell states §4.4 distinguishes, one per part, so the two parts
#: disagree about one identity and agree about nothing else.
FIRST_CELLS = {"col-9f3a71": "話す", "col-2b8d04": ""}
SECOND_CELLS = {"col-9f3a71": "はなす", "col-2b8d04": "話します"}


def _answer(cells: dict[str, str]) -> dict[str, Any]:
    answer = json.loads(
        (RESPONSES / "table_exhaustive.json").read_text(encoding="utf-8")
    )
    for entry in answer["candidates"]:
        entry["conjugations"] = dict(cells)
    return answer


def _stage(config: ProjectConfig, source: Path, cells: dict[str, str]) -> Path:
    """One real layout-bound extraction, staged through the sole writer."""
    runner = FakeClaudeRunner(reply=_settled_stream(_answer(cells)))
    plan = extraction_batch.plan_extraction_batch(
        config,
        [source],
        mode=extract.LAYOUT_MODE,
        layouts=(_layout(),),
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
    )
    outcome = extraction_batch.dispatch_extraction_batch(
        config,
        plan,
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
        provider_spawn=runner.spawn,
    )
    assert outcome.committed_count == 1
    return extract.staging_path(config.staging_dir, source.name)


def _curated_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ProjectConfig, str, Path, Path]:
    """A job whose two published parts stage one identity differently."""
    _no_api(monkeypatch)
    config = _project(tmp_path)
    job = _job(config)
    _publish(config, job.header.job_id, PART_PAYLOADS)
    first_staging = _stage(
        config, config.scan_inbox / "part-1.png", FIRST_CELLS
    )
    second_staging = _stage(
        config, config.scan_inbox / "part-2.png", SECOND_CELLS
    )
    return config, job.header.job_id, first_staging, second_staging


# =============================================================================
# append_layout — one immutable revision, one CAS write
# =============================================================================


def test_append_layout_appends_the_revision_and_binds_its_part_in_one_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§4.1: the owner's Save appends the revision and repoints the binding.

    One compare-and-swap write, because two writes could leave a binding
    pointing at a revision that does not exist, or a revision nothing points at
    with the owner believing they had bound it.

    Mutant: split `append_layout` into `record_choice` after a layout-only
    save, or drop `bind=` and repoint separately.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    job = _job(config)

    saved = study_job.append_layout(
        config,
        job.header.job_id,
        _layout(),
        bind=("part-1.png",),
        expected_revision=job.revision,
    )

    assert set(saved.layouts) == {"layout-7c1f2a@1"}
    assert saved.layouts["layout-7c1f2a@1"] == LAYOUT_WIRE
    assert saved.choices["part_layout_bindings"] == {
        "part-1.png": ["layout-7c1f2a", 1]
    }
    # One save: the document on disk already holds both halves.
    reread = study_job.load_study_job(config, job.header.job_id)
    assert reread.revision == saved.revision
    assert study_job.job_layout_bindings(reread)["part-1.png"].to_wire() == (
        LAYOUT_WIRE
    )


def test_a_layout_revision_is_immutable_and_a_stale_save_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§4.1/§6.2: `append_layout` refuses an existing key whose content differs.

    A dispatched child may already have been sent under that revision, so
    neither a choice edit nor a re-Save can rewrite it.

    Mutant: overwrite `layouts[key]` unconditionally in `append_layout`.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    job = _job(config)
    saved = study_job.append_layout(
        config, job.header.job_id, _layout(), expected_revision=job.revision
    )

    different = json.loads(json.dumps(LAYOUT_WIRE))
    different["columns"][0]["display_label"] = "Dictionary"
    with pytest.raises(study_job.StudyJobError) as refusal:
        study_job.append_layout(
            config,
            job.header.job_id,
            extract.TableLayout.from_wire(different),
            expected_revision=saved.revision,
        )
    assert "immutable" in str(refusal.value)
    assert "layout-7c1f2a revision 1" in str(refusal.value)
    assert study_job.load_study_job(config, job.header.job_id).layouts == (
        saved.layouts
    )

    # The same bytes twice is a stale editor unless the caller says otherwise.
    with pytest.raises(study_job.StudyJobError):
        study_job.append_layout(
            config, job.header.job_id, _layout(), expected_revision=saved.revision
        )
    again = study_job.append_layout(
        config,
        job.header.job_id,
        _layout(),
        bind=("part-2.png",),
        allow_identical=True,
        expected_revision=saved.revision,
    )
    assert again.choices["part_layout_bindings"] == {
        "part-2.png": ["layout-7c1f2a", 1]
    }

    # A new revision is a new key beside the old one, never a replacement.
    third = study_job.append_layout(
        config,
        job.header.job_id,
        extract.TableLayout.from_wire({**different, "revision": 2}),
        bind=("part-1.png",),
        expected_revision=again.revision,
    )
    assert set(third.layouts) == {"layout-7c1f2a@1", "layout-7c1f2a@2"}


def test_an_ordinary_choice_edit_cannot_reach_the_layout_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§6.2: `record_choice` refuses a payload naming another namespace.

    A binding may be repointed at a revision this job already recorded — that
    edits neither revision — but it cannot bring one into existence.

    Mutant: let `record_choice` write `layouts`, or drop the
    binding-points-at-a-recorded-revision check.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    job = _job(config)

    with pytest.raises(study_job.StudyJobError) as refusal:
        study_job.record_choice(
            config,
            job.header.job_id,
            {"layouts": {"layout-x@1": LAYOUT_WIRE}},
            expected_revision=job.revision,
        )
    assert "layouts" in str(refusal.value)

    with pytest.raises(study_job.StudyJobError) as unknown:
        study_job.record_choice(
            config,
            job.header.job_id,
            {"part_layout_bindings": {"part-1.png": ["layout-x", 1]}},
            expected_revision=job.revision,
        )
    assert "never recorded" in str(unknown.value)

    saved = study_job.append_layout(
        config,
        job.header.job_id,
        _layout(),
        bind=("part-1.png",),
        expected_revision=job.revision,
    )
    repointed = study_job.record_choice(
        config,
        job.header.job_id,
        {"part_layout_bindings": {"part-2.png": ["layout-7c1f2a", 1]}},
        expected_revision=saved.revision,
    )
    assert repointed.layouts == saved.layouts


def test_one_confirmed_batch_carries_one_mode_over_this_jobs_bound_parts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§4.2: the job pins the mode per part and never leaves it `None`.

    An omitted mode is pinned from the owner's own bindings; an explicitly
    named incompatible one still refuses, because a bound part sent under
    another mode would ask `extract-auto` for a table nobody described. A
    batch whose parts do not share a mode is two confirmations rather than one
    request with two shapes.

    Mutant: drop `_batch_layouts` from `plan_job_extraction_batch` and forward
    `**plan_options` alone.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    job = _job(config)
    _publish(config, job.header.job_id, PART_PAYLOADS)
    saved = study_job.append_layout(
        config,
        job.header.job_id,
        _layout(),
        bind=("part-1.png",),
        expected_revision=study_job.load_study_job(
            config, job.header.job_id
        ).revision,
    )

    runner = FakeClaudeRunner(reply=_settled_stream(_answer(FIRST_CELLS)))
    options = {
        "provider_env": {},
        "provider_runner": runner,
        "provider_which": _which,
    }

    # One part bound, one not: one mode cannot describe both.
    with pytest.raises(study_job.StudyJobError) as mixed:
        study_job.plan_job_extraction_batch(
            config, job.header.job_id, mode=extract.LAYOUT_MODE, **options
        )
    assert "part-2.png" in str(mixed.value)
    assert "separate batches" in str(mixed.value)

    study_job.append_layout(
        config,
        job.header.job_id,
        _layout(revision=2),
        bind=("part-2.png",),
        expected_revision=saved.revision,
    )

    # Both bound and no mode named: the job pins its own.
    pinned = study_job.plan_job_extraction_batch(
        config, job.header.job_id, **options
    )
    assert [child.expectation.mode for child in pinned.children] == [
        extract.LAYOUT_MODE,
        extract.LAYOUT_MODE,
    ]
    assert [
        child.expectation.table_layout.revision for child in pinned.children
    ] == [1, 2]

    # An explicitly named incompatible mode still refuses.
    with pytest.raises(study_job.StudyJobError) as wrong:
        study_job.plan_job_extraction_batch(
            config, job.header.job_id, mode="table", **options
        )
    assert "table" in str(wrong.value)
    assert "Nothing was planned" in str(wrong.value)

    plan = study_job.plan_job_extraction_batch(
        config, job.header.job_id, mode=extract.LAYOUT_MODE, **options
    )
    assert [
        child.expectation.table_layout.revision for child in plan.children
    ] == [1, 2]
    assert plan.job_id == job.header.job_id


# =============================================================================
# The narrow staging prepare/apply pair
# =============================================================================


def test_a_prepared_update_changes_one_cell_and_nothing_else(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§6.4: comments, key order, quoting, coverage and accounting untouched.

    The prepared text is re-read and must parse to exactly the records given,
    with byte-identical metadata, before it can be applied.

    Mutant: skip the re-read self-check in `staging.prepare_record_update`, or
    route the change through `write_staging` instead of the round-trip diff.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    part = _lone_part(config)
    path = _stage(config, part, FIRST_CELLS)
    original = path.read_text(encoding="utf-8")
    # A reviewer's own line, and a value the two YAML dialects disagree about.
    edited = original.replace(
        "records:", "# a reviewer's note nobody may delete\nrecords:", 1
    )
    path.write_text(edited, encoding="utf-8")
    before_records, before_meta = staging.read_staging(path)
    index = next(
        position
        for position, record in enumerate(before_records)
        if record.source_forms is not None
    )
    after = list(before_records)
    table = after[index].source_forms
    after[index] = dataclass_replace(
        after[index],
        source_forms=dataclass_replace(
            table, cells={**table.cells, "col-2b8d04": "話します"}
        ),
    )

    prepared = staging.prepare_record_update(path, after)

    assert prepared.applied is True
    assert prepared.sha256_before == hashlib.sha256(
        edited.encode("utf-8")
    ).hexdigest()
    assert "# a reviewer's note nobody may delete" in prepared.text
    reparsed, reparsed_meta = staging.read_staging_text(
        prepared.text, source=str(path)
    )
    assert reparsed == after
    assert reparsed_meta == before_meta
    # Nothing is written until the caller applies it.
    assert path.read_text(encoding="utf-8") == edited

    staging.apply_prepared_update(path, prepared)
    assert path.read_text(encoding="utf-8") == prepared.text
    landed, landed_meta = staging.read_staging(path)
    assert landed[index].source_forms.cells["col-2b8d04"] == "話します"
    assert landed_meta == before_meta


def test_only_the_explicit_operation_can_delete_a_cell_or_a_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§6.4: `_apply_changes` writes only changed keys and never removes one.

    Emptying a cell is `replace` with `""` — the printed-blank spelling — and
    removing it is `remove`. The two are different facts about the source, so
    the round-trip diff alone cannot express the second.

    Mutant: make `FieldOperation`'s `remove` fall through to `_apply_changes`,
    or accept a `remove` naming a key the row does not hold.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    part = _lone_part(config)
    path = _stage(config, part, FIRST_CELLS)
    records, _meta = staging.read_staging(path)
    index = next(
        position
        for position, record in enumerate(records)
        if record.source_forms is not None
    )
    record = records[index]

    # Without the explicit operation the stale key survives the diff, and the
    # self-check refuses rather than writing a document that disagrees.
    dropped = list(records)
    dropped[index] = dataclass_replace(record, source_forms=None)
    with pytest.raises(staging.StagingError) as refusal:
        staging.prepare_record_update(path, dropped)
    assert "did not preserve" in str(refusal.value)

    prepared = staging.prepare_record_update(
        path,
        dropped,
        operations=(
            staging.FieldOperation(
                row_index=index,
                record_id=record.id,
                field="source_forms",
                action="remove",
            ),
        ),
    )
    reparsed, _ = staging.read_staging_text(prepared.text, source=str(path))
    assert reparsed[index].source_forms is None
    assert "source_forms" not in prepared.text.split("records:")[1].split(
        "\n  - "
    )[index + 1]

    # One cell, by index *and* by record id.
    table = record.source_forms
    without_cell = list(records)
    without_cell[index] = dataclass_replace(
        record,
        source_forms=dataclass_replace(
            table, cells={"col-2b8d04": table.cells["col-2b8d04"]}
        ),
    )
    cell_removed = staging.prepare_record_update(
        path,
        without_cell,
        operations=(
            staging.FieldOperation(
                row_index=index,
                record_id=record.id,
                field="source_forms",
                action="remove",
                key="col-9f3a71",
            ),
        ),
    )
    landed, _ = staging.read_staging_text(cell_removed.text, source=str(path))
    assert landed[index].source_forms.cells == {"col-2b8d04": ""}

    with pytest.raises(staging.StagingError) as mismatch:
        staging.prepare_record_update(
            path,
            list(records),
            operations=(
                staging.FieldOperation(
                    row_index=index,
                    record_id="word:not-this-one:x",
                    field="source_forms",
                    action="remove",
                ),
            ),
        )
    assert "word:not-this-one:x" in str(mismatch.value)


def test_applying_a_prepared_update_is_bound_to_the_bytes_it_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The compare-and-swap `apply_prepared_update` performs.

    Mutant: write with `atomic_write_text` instead of
    `atomic_write_text_bound(expected_revision=prepared.sha256_before)`.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    part = _lone_part(config)
    path = _stage(config, part, FIRST_CELLS)
    records, _meta = staging.read_staging(path)
    index = next(
        position
        for position, record in enumerate(records)
        if record.source_forms is not None
    )
    after = list(records)
    table = after[index].source_forms
    after[index] = dataclass_replace(
        after[index],
        source_forms=dataclass_replace(
            table, cells={**table.cells, "col-2b8d04": "話します"}
        ),
    )
    prepared = staging.prepare_record_update(path, after)

    path.write_text(
        path.read_text(encoding="utf-8") + "\n# someone else edited it\n",
        encoding="utf-8",
    )
    moved = path.read_text(encoding="utf-8")

    with pytest.raises(JankiError):
        staging.apply_prepared_update(path, prepared)
    assert path.read_text(encoding="utf-8") == moved


# =============================================================================
# Cross-source curation
# =============================================================================


def test_curation_reads_the_current_staged_values_and_shows_disagreement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§6.1: the editable records decide, and a conflict is disclosed.

    Nothing merges first-wins, and no combined accounting record is written.

    Mutant: read the immutable `candidate_accounting` block instead of the
    editable `records` list, or resolve a conflict by taking the first part.
    """
    config, job_id, first, second = _curated_job(tmp_path, monkeypatch)
    before = {path: path.read_bytes() for path in (first, second)}

    groups = study_curation.read_curation_groups(config, job_id)

    assert groups
    conflicting = [group for group in groups if group.conflicting]
    assert conflicting, "the two parts staged different printed cells"
    group = conflicting[0]
    assert {item.part_name for item in group.occurrences} == {
        "part-1.png",
        "part-2.png",
    }
    assert {
        item.part_name: item.source_forms.cells for item in group.occurrences
    } == {"part-1.png": FIRST_CELLS, "part-2.png": SECOND_CELLS}
    # A read.
    assert {path: path.read_bytes() for path in (first, second)} == before


def test_one_decision_lands_in_every_occurrence_and_is_recorded_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§6.2/§6.3: the intent precedes the first write and names both files.

    The chosen value lands in every occurrence's own staged record for that
    identity, so whichever part promotes last writes the same bytes and cannot
    overwrite the choice.

    Mutant: append the intent *after* the writes in `apply_curation`, or write
    only the file the decision was read from.
    """
    config, job_id, first, second = _curated_job(tmp_path, monkeypatch)
    groups = study_curation.read_curation_groups(config, job_id)
    group = next(item for item in groups if item.conflicting)

    # A value neither part staged, so *both* occurrences move: a decision that
    # only changed the file it was read from would prove nothing here.
    plan = study_curation.plan_curation(
        config,
        job_id,
        (
            study_curation.CurationChoice(
                record_id=group.record_id,
                action="replace",
                key="col-2b8d04",
                value="話しました",
            ),
        ),
        decision="the polite column prints 話しました here",
    )

    assert sorted(plan.paths) == sorted(
        [
            str(first.relative_to(config.root)),
            str(second.relative_to(config.root)),
        ]
    )
    # Planning writes nothing and records no curation intent.
    assert [
        item.kind
        for item in study_job.load_study_job(config, job_id).intents
        if item.kind == "curation"
    ] == []

    outcome = study_curation.apply_curation(config, plan)

    job = study_job.load_study_job(config, job_id)
    intent = job.intent(outcome.intent_id)
    assert intent.kind == "curation"
    assert intent.bindings["decision"] == "the polite column prints 話しました here"
    # The complete prepared file text, not a digest: a digest cannot be
    # replayed after a crash.
    for entry in intent.reserves["prepared"]:
        assert entry["content_after"]
        assert (
            hashlib.sha256(entry["content_after"].encode("utf-8")).hexdigest()
            == entry["sha256_after"]
        )
    landed = {}
    for path in (first, second):
        records, _meta = staging.read_staging(path)
        landed[path.name] = {
            record.id: (
                None if record.source_forms is None else record.source_forms.cells
            )
            for record in records
        }
    assert landed[first.name][group.record_id] == {
        **FIRST_CELLS,
        "col-2b8d04": "話しました",
    }
    assert landed[second.name][group.record_id] == {
        **SECOND_CELLS,
        "col-2b8d04": "話しました",
    }
    assert set(outcome.written) == set(plan.paths)
    assert study_curation.open_curation_barriers(config) == ()


def _one_cell_plan(config: ProjectConfig, job_id: str, value: str) -> Any:
    """One decision that moves a cell in every part that staged the identity."""

    group = next(
        item
        for item in study_curation.read_curation_groups(config, job_id)
        if item.conflicting
    )
    return study_curation.plan_curation(
        config,
        job_id,
        (
            study_curation.CurationChoice(
                record_id=group.record_id,
                action="replace",
                key="col-2b8d04",
                value=value,
            ),
        ),
        decision=f"the polite column prints {value} here",
    )


def test_a_staged_file_is_prechecked_over_its_exact_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§6.3 step 4: the precheck answers the question the digests were computed for.

    Every other digest in this transaction — the prepared snapshot's, the
    barrier's and the writer's own compare-and-swap — is over the file's exact
    bytes. A file holding CRLF line endings is unchanged as far as all three
    are concerned, so the precheck must agree rather than refuse a decision it
    already recorded.

    Mutant: read the bound file through newline-translating text I/O in
    `study_curation._write_prepared`.
    """
    config, job_id, first, second = _curated_job(tmp_path, monkeypatch)
    crlf = first.read_bytes().replace(b"\n", b"\r\n")
    first.write_bytes(crlf)
    plan = _one_cell_plan(config, job_id, "話しました")
    entry = next(
        item
        for item in plan.prepared
        if (config.root / item.staging_path) == first
    )
    assert entry.sha256_before == hashlib.sha256(crlf).hexdigest()

    outcome = study_curation.apply_curation(config, plan)

    assert outcome.state == "applied"
    assert set(outcome.written) == {item.staging_path for item in plan.prepared}
    for item in plan.prepared:
        landed = (config.root / item.staging_path).read_bytes()
        assert hashlib.sha256(landed).hexdigest() == item.sha256_after
    assert study_curation.open_curation_barriers(config) == ()
    # The outcome's evidence is what those files really hold, measured with
    # the locks still held rather than restated from the intent.
    recorded = study_job.load_study_job(config, job_id).outcomes[-1]
    assert dict(recorded.observed) == {
        item.staging_path: hashlib.sha256(
            (config.root / item.staging_path).read_bytes()
        ).hexdigest()
        for item in plan.prepared
    }


def test_a_crlf_variant_of_the_recorded_result_is_not_a_finished_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§6.3 step 6: complete means the bound files *are* at `sha256_after`.

    A copy of the recorded text with translated line endings is not those
    bytes. Skipping the write for it and closing the intent would report a
    decision as landed over a file that does not hold it.

    Mutant: read the bound file through newline-translating text I/O in
    `study_curation._write_prepared`, so the variant hashes to `sha256_after`.
    """
    config, job_id, first, second = _curated_job(tmp_path, monkeypatch)
    plan = _one_cell_plan(config, job_id, "話しました")
    job = study_job.load_study_job(config, job_id)
    intent = study_curation._curation_intent(plan)
    study_job.append_intent(config, job_id, intent, expected_revision=job.revision)
    entry = next(
        item
        for item in plan.prepared
        if (config.root / item.staging_path) == first
    )
    variant = entry.content_after.replace("\n", "\r\n").encode("utf-8")
    first.write_bytes(variant)

    with pytest.raises(study_curation.StudyCurationError) as refusal:
        study_curation.resume_curation(config, job_id, intent.intent_id)

    assert entry.staging_path in str(refusal.value)
    assert first.read_bytes() == variant
    assert [
        barrier.intent_id
        for barrier in study_curation.open_curation_barriers(config)
    ] == [intent.intent_id]


def test_a_write_that_did_not_land_the_recorded_bytes_leaves_the_intent_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An outcome's `observed` is evidence about the files, not a restatement.

    `IntentOutcome.observed` is documented as pairs actually observed, and
    completion is proven from them: if the writer did not leave the recorded
    bytes behind, the decision is not complete and nothing may close it.

    Mutant: record `item.sha256_after` in `_settle` and drop the post-write
    measurement from `_write_prepared`.
    """
    config, job_id, first, second = _curated_job(tmp_path, monkeypatch)
    plan = _one_cell_plan(config, job_id, "話しました")
    real = study_curation.staging.apply_prepared_update

    def short(path: Path, prepared: Any) -> Path:
        # A writer that lands something other than the recorded text. Nothing
        # in janki does this today; the point is that the outcome is measured
        # rather than assumed, so it could not be reported as complete.
        return real(
            path,
            dataclass_replace(prepared, text=prepared.text + "\n# later\n"),
        )

    monkeypatch.setattr(
        study_curation.staging, "apply_prepared_update", short
    )

    with pytest.raises(study_curation.StudyCurationError) as refusal:
        study_curation.apply_curation(config, plan)

    assert "after this decision was written" in str(refusal.value)
    saved = study_job.load_study_job(config, job_id)
    curation = [item for item in saved.intents if item.kind == "curation"]
    assert len(curation) == 1
    assert curation[0].intent_id not in saved.closed_intent_ids
    # The barrier stays up over exactly the files the decision bound, and the
    # owner's route out of it is the explicit abandon decision below.
    assert [
        barrier.intent_id
        for barrier in study_curation.open_curation_barriers(config)
    ] == [curation[0].intent_id]


def test_an_interrupted_decision_resumes_from_its_exact_recorded_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§6.3 step 6: skip `sha256_after`, apply `sha256_before`, refuse neither.

    Nothing is recomputed from today's staged values on resume: the recorded
    text is the decision.

    Mutant: recompute the prepared text in `resume_curation` from the current
    staged values instead of replaying `content_after`.
    """
    config, job_id, first, second = _curated_job(tmp_path, monkeypatch)
    group = next(
        item
        for item in study_curation.read_curation_groups(config, job_id)
        if item.conflicting
    )
    plan = study_curation.plan_curation(
        config,
        job_id,
        (
            study_curation.CurationChoice(
                record_id=group.record_id,
                action="replace",
                key="col-2b8d04",
                value="話しました",
            ),
        ),
        decision="the polite column prints 話しました here",
    )
    job = study_job.load_study_job(config, job_id)

    # The crash: the intent is durable and no file has been written.
    intent = study_curation._curation_intent(plan)
    study_job.append_intent(config, job_id, intent, expected_revision=job.revision)
    barriers = study_curation.open_curation_barriers(config)
    assert [barrier.intent_id for barrier in barriers] == [intent.intent_id]
    assert barriers[0].remaining == barriers[0].paths

    outcome = study_curation.resume_curation(config, job_id, intent.intent_id)

    assert outcome.state == "applied"
    assert set(outcome.written) == set(barriers[0].paths)
    assert study_curation.open_curation_barriers(config) == ()
    # The recorded text, byte for byte, is what landed.
    for entry in plan.prepared:
        landed = (config.root / entry.staging_path).read_text(encoding="utf-8")
        assert landed == entry.content_after
        assert (
            hashlib.sha256(landed.encode("utf-8")).hexdigest()
            == entry.sha256_after
        )

    # Resuming a closed decision is refused; outcomes are append-only.
    with pytest.raises(study_curation.StudyCurationError):
        study_curation.resume_curation(config, job_id, intent.intent_id)


def test_one_mismatched_file_refuses_the_whole_decision_before_any_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§6.3 step 4: all locks, then a precheck of *every* bound file.

    Per-file compare-and-swap alone permits a partial application whose
    remaining files then refuse, leaving a decision recovery cannot finish.

    Mutant: precheck and write each file in one pass, so the first file lands
    before the second is compared.
    """
    config, job_id, first, second = _curated_job(tmp_path, monkeypatch)
    group = next(
        item
        for item in study_curation.read_curation_groups(config, job_id)
        if item.conflicting
    )
    plan = study_curation.plan_curation(
        config,
        job_id,
        (
            study_curation.CurationChoice(
                record_id=group.record_id,
                action="replace",
                key="col-2b8d04",
                value="話しました",
            ),
        ),
        decision="the polite column prints 話しました here",
    )
    # Two bound files, so a per-file compare-and-swap could apply one of them.
    assert len(plan.prepared) == 2
    job = study_job.load_study_job(config, job_id)
    intent = study_curation._curation_intent(plan)
    study_job.append_intent(config, job_id, intent, expected_revision=job.revision)

    # Somebody edits one of the bound files after the decision was recorded.
    moved = config.root / plan.prepared[-1].staging_path
    moved.write_text(
        moved.read_text(encoding="utf-8") + "\n# a later hand edit\n",
        encoding="utf-8",
    )
    snapshot = {
        config.root / entry.staging_path: (
            config.root / entry.staging_path
        ).read_bytes()
        for entry in plan.prepared
    }

    with pytest.raises(study_curation.StudyCurationError) as refusal:
        study_curation.resume_curation(config, job_id, intent.intent_id)

    assert plan.prepared[-1].staging_path in str(refusal.value)
    assert "neither" in str(refusal.value)
    for path, bytes_before in snapshot.items():
        assert path.read_bytes() == bytes_before
    # The intent stays open: nothing closed a decision that never happened.
    assert [
        barrier.intent_id
        for barrier in study_curation.open_curation_barriers(config)
    ] == [intent.intent_id]


def test_a_pending_intent_survives_an_ordinary_choice_edit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§6.2: an ordinary choice edit cannot clear or rewrite a pending intent.

    Mutant: let `record_choice` rebuild the document from the choices alone.
    """
    config, job_id, _first, _second = _curated_job(tmp_path, monkeypatch)
    group = next(
        item
        for item in study_curation.read_curation_groups(config, job_id)
        if item.conflicting
    )
    chosen = group.occurrences[0]
    plan = study_curation.plan_curation(
        config,
        job_id,
        (
            study_curation.CurationChoice(
                record_id=group.record_id,
                action="replace",
                key="",
                value=chosen.forms_wire,
            ),
        ),
        decision="use the first part's printed forms",
    )
    job = study_job.load_study_job(config, job_id)
    intent = study_curation._curation_intent(plan)
    study_job.append_intent(config, job_id, intent, expected_revision=job.revision)

    edited = study_job.record_choice(
        config,
        job_id,
        {"part_selections": ["part-1.png"]},
        expected_revision=study_job.load_study_job(config, job_id).revision,
    )

    assert edited.choices["part_selections"] == ["part-1.png"]
    assert intent.intent_id in [item.intent_id for item in edited.intents]
    assert intent in edited.intents
    assert [
        barrier.intent_id
        for barrier in study_curation.open_curation_barriers(config)
    ] == [intent.intent_id]


def test_curation_deletes_a_cell_without_touching_the_rows_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§6.4: an explicit delete leaves the metadata byte-identical.

    Deleting a printed cell is an ordinary local edit, not an extraction
    defect, and it never costs a paid retry.

    Mutant: route the delete through `staging.write_staging`, which re-renders
    the file from records and drops the reviewer's own bytes.
    """
    config, job_id, first, _second = _curated_job(tmp_path, monkeypatch)
    before_meta = staging.read_staging(first)[1]
    records, _ = staging.read_staging(first)
    subject = next(
        record for record in records if record.source_forms is not None
    )

    plan = study_curation.plan_curation(
        config,
        job_id,
        (
            study_curation.CurationChoice(
                record_id=subject.id, action="remove", key="col-2b8d04"
            ),
        ),
        decision="the source printed no polite column for this row",
    )
    study_curation.apply_curation(config, plan)

    after_records, after_meta = staging.read_staging(first)
    changed = next(
        record for record in after_records if record.id == subject.id
    )
    assert changed.source_forms.cells == {"col-9f3a71": "話す"}
    assert "col-2b8d04" not in changed.source_forms.cells
    # The declared column stays declared: absent is a fact about this row.
    assert [column.id for column in changed.source_forms.columns] == [
        "col-9f3a71",
        "col-2b8d04",
    ]
    assert after_meta == before_meta
    assert after_meta["candidate_accounting"] == before_meta["candidate_accounting"]
    assert after_meta["prompt_provenance"] == before_meta["prompt_provenance"]


# =============================================================================
# The promotion barrier, in both entries, under one guard
# =============================================================================


def _open_barrier(config: ProjectConfig, job_id: str, path: Path) -> str:
    """Publish one curation intent over this file without applying it."""
    records, _meta = staging.read_staging(path)
    subject = next(
        record for record in records if record.source_forms is not None
    )
    plan = study_curation.plan_curation(
        config,
        job_id,
        (
            study_curation.CurationChoice(
                record_id=subject.id,
                action="replace",
                key="col-2b8d04",
                value="話します",
            ),
        ),
        decision="the polite column prints 話します here",
    )
    job = study_job.load_study_job(config, job_id)
    intent = study_curation._curation_intent(plan)
    study_job.append_intent(config, job_id, intent, expected_revision=job.revision)
    return intent.intent_id


def test_a_pending_decision_blocks_the_planning_gate_naming_its_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§6.5: `decide_promotion` gains one gate, with the structural group.

    Because it is inside `decide_promotion`, it applies identically to `janki
    promote`, the workbench promotion, the Assistant's typed promotion action
    and this job's own finish.

    Mutant: delete the `curation-pending` gate from `decide_promotion`.
    """
    config, job_id, first, second = _curated_job(tmp_path, monkeypatch)
    intent_id = _open_barrier(config, job_id, first)

    decision = promotion.decide_promotion(config, first, skip_reading_check=True)

    assert decision.is_blocked
    assert decision.gate == "curation-pending"
    assert intent_id in str(decision.error)
    assert job_id in str(decision.error)
    # A file no open decision names is not blocked by this gate.
    other = promotion.decide_promotion(config, second, skip_reading_check=True)
    assert other.gate != "curation-pending"


def test_a_decision_taken_before_publication_still_refuses_at_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§6.5: a planning gate cannot close the race; the execution rechecks.

    An intent can be published while every bound staging file still holds its
    `sha256_before` bytes, so the wire compare-and-swap proves only that the
    *file* did not change: a promotion that decided before publication would
    still pass it and consume a file a durable decision is about to rewrite.

    Mutant: gate only `decide_promotion` and confirm a pre-barrier decision
    still executes.
    """
    config, job_id, first, _second = _curated_job(tmp_path, monkeypatch)
    # A decision taken *before* the barrier exists, and not blocked by it.
    decision = promotion.PromotionDecision(
        source=first.name,
        staging_path=first.resolve(),
        state="nothing",
        repository=promotion._repository_binding(config),
        reading_check="explicit_skip",
    )
    assert not decision.is_blocked
    assert promotion.execute_promotion(config, decision).state == "nothing"

    intent_id = _open_barrier(config, job_id, first)

    with pytest.raises(JankiError) as refusal:
        promotion.execute_promotion(config, decision)
    assert intent_id in str(refusal.value)


def test_an_unreadable_job_document_refuses_rather_than_lifting_the_barrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§6.5: a barrier that is missing or malformed is never skipped.

    A skipped barrier is a lifted one, and an unreadable document is exactly
    where an open intent would be invisible.

    Mutant: catch and skip the unreadable job in `open_curation_barriers`.
    """
    config, job_id, first, _second = _curated_job(tmp_path, monkeypatch)
    study_job.study_job_path(config, job_id).write_text(
        "not json at all", encoding="utf-8"
    )
    before = first.read_bytes()

    decision = promotion.decide_promotion(config, first, skip_reading_check=True)

    assert decision.is_blocked
    assert job_id in str(decision.error)
    assert first.read_bytes() == before


def test_the_finish_reaches_promotion_through_the_unguarded_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§6.5: the inner promotion call site, and the guard's exclusivity.

    The revision finish already holds `.janki-audio-operation` when it reaches
    promotion, and `io.exclusive_path_lock` is deliberately non-reentrant. So
    the finish takes the guard at its own outermost entry and reaches the
    **unguarded** promotion entry from inside; acquiring the guard there would
    invert the order and deadlock.

    Two separate claims, and neither is the whole ordering rule: which entry
    the inner call site names, and that the guard is exclusive between
    threads. The ordering of the four outermost `with` headers themselves is
    proven by running them against a held guard in
    `tests/test_study_curation_lock_order.py`, which is where a swapped header
    is caught; this case is blind to that and does not claim it.

    Mutant: call `assistant_promotion.execute_promotion_action` (the guarded
    entry) from `card_revision_finish._execute_record_locked`.
    """
    from japanese_anki.application import assistant_promotion, card_revision_finish

    source = Path(card_revision_finish.__file__).read_text(encoding="utf-8")
    assert "execute_promotion_action_under_guard" in source
    assert "assistant_promotion.execute_promotion_action(" not in source
    # Both entries exist and the guarded one is the wrapper.
    assert hasattr(assistant_promotion, "execute_promotion_action")
    assert hasattr(assistant_promotion, "execute_promotion_action_under_guard")

    # The guard is non-reentrant, so taking it twice in one process must not be
    # attempted: this proves it, and proves the finish's order is the safe one.
    config = _project(tmp_path)
    config.staging_dir.mkdir(parents=True, exist_ok=True)
    entered = threading.Event()
    finished = threading.Event()

    def hold() -> None:
        with study_curation.curation_guard(config):
            entered.set()
            finished.wait(timeout=5)

    holder = threading.Thread(target=hold)
    holder.start()
    assert entered.wait(timeout=5)
    waiter_done = threading.Event()

    def wait_for_guard() -> None:
        with study_curation.curation_guard(config):
            waiter_done.set()

    waiter = threading.Thread(target=wait_for_guard)
    waiter.start()
    assert not waiter_done.wait(timeout=0.5), "the guard is exclusive"
    finished.set()
    holder.join(timeout=5)
    assert waiter_done.wait(timeout=5)
    waiter.join(timeout=5)


def test_a_replacement_shape_for_the_printed_table_needs_its_whole_wire(
    tmp_path: Path,
) -> None:
    """§4.5: `_validate_replacement_shape` gains a `source_forms` case.

    Its default is `valid = True` for an unrecognized field name, so without
    the case a bare cell map would be accepted by the constructor as an empty
    table and silently discard the source's declared columns.

    Mutant: delete the `source_forms` branch from
    `card_change_staging._validate_replacement_shape`.
    """
    from japanese_anki.application import card_change_staging

    complete = {
        "columns": [{"id": "col-9f3a71", "label": "Plain"}],
        "cells": {"col-9f3a71": "話す"},
    }
    card_change_staging._validate_replacement_shape("word:x:y", "source_forms", complete)

    for bad in (
        {"cells": {"col-9f3a71": "話す"}},
        {"columns": [{"id": "col-9f3a71"}], "cells": {}},
        {"columns": [], "cells": {"col-9f3a71": 1}},
        {"columns": [{"id": "a", "label": "A"}], "cells": {}, "extra": 1},
        "source_forms",
    ):
        with pytest.raises(card_change_staging.CardChangeStagingError):
            card_change_staging._validate_replacement_shape(
                "word:x:y", "source_forms", bad
            )

    # Removal is refused here rather than admitted: both spellings of it
    # canonicalize to absence, which `io.merge_records` reads as a hole, so a
    # staged removal would review as a deletion and promote to nothing.
    # Deleting a table is curation's explicit `FieldOperation` (§6.4), proven
    # by `test_only_the_explicit_operation_can_delete_a_cell_or_a_table`.
    for removal in (None, {"columns": [], "cells": {}}):
        with pytest.raises(
            card_change_staging.CardChangeStagingError, match="--drop-table"
        ):
            card_change_staging._validate_replacement_shape(
                "word:x:y", "source_forms", removal
            )


def test_a_printed_blank_survives_the_repair_pass_round_trip() -> None:
    """`repairs._require_canonical_record` demands `to_dict → from_dict → to_dict`.

    A trimmed cell or an asymmetric empty spelling breaks every repair pass, so
    the blank a source printed has to survive it exactly.

    Mutant: trim cell values in `SourceFormsTable.from_dict` or drop blank ones.
    """
    from japanese_anki import repairs

    record = VocabularyRecord.from_dict(
        {
            "id": "word:話す:はなす",
            "expression": "話す",
            "reading": "はなす",
            "meanings": ["to speak"],
            "source_forms": {
                "columns": [
                    {"id": "col-9f3a71", "label": "Plain"},
                    {"id": "col-2b8d04", "label": "Polite"},
                ],
                "cells": {"col-9f3a71": " 　", "col-2b8d04": ""},
            },
        }
    )

    repairs._require_canonical_record(record, "record-romaji-from-reading")

    payload = record.to_dict()
    assert payload["source_forms"]["cells"] == {
        "col-9f3a71": " 　",
        "col-2b8d04": "",
    }
    assert VocabularyRecord.from_dict(payload).to_dict() == payload


def test_control_characters_are_caught_in_ids_labels_and_cells(
    tmp_path: Path,
) -> None:
    """§4.5: `validation._control_characters` walks `to_dict()` generically.

    Column ids, labels and cells are covered with no new validator, which is
    why the canonical serialization has to emit plain containers.

    Mutant: emit the dataclass tuple from `to_dict` so the walker's list branch
    stops reaching the column mappings.
    """
    from japanese_anki import validation

    record = VocabularyRecord.from_dict(
        {
            "id": "word:話す:はなす",
            "expression": "話す",
            "reading": "はなす",
            "meanings": ["to speak"],
            "source_forms": {
                "columns": [{"id": "col-9f3a71", "label": "Pla\x07in"}],
                "cells": {"col-9f3a71": "話\x00す"},
            },
        }
    )

    issues = validation.validate_records([record], tmp_path / "vocabulary.json")

    messages = " ".join(issue.message for issue in issues)
    assert "source_forms.columns[0].label" in messages
    assert "source_forms.cells.col-9f3a71" in messages
    assert "U+0007" in messages
    assert "U+0000" in messages


# =============================================================================
# The Assistant's own routes, over the real services
# =============================================================================


def test_the_desk_shows_the_bound_layout_and_settles_a_disagreement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The adapter's S5 routes, driven against the real stores.

    The two views are provenance displays and the apply is the owner's own
    bound control: it records the decision before its first write and writes
    through the same service the CLI calls.

    Mutant: have `apply_study_job_curation` read the identity and the part from
    anywhere but its bound target, or write without recording the intent.
    """
    from test_workbench_assistant_integration import _adapter

    config, job_id, first, second = _curated_job(tmp_path, monkeypatch)
    study_job.append_layout(
        config,
        job_id,
        _layout(),
        bind=("part-1.png",),
        expected_revision=study_job.load_study_job(config, job_id).revision,
    )
    adapter = _adapter(config)

    layout_text = adapter.study_job_layout_text(job_id=job_id)
    assert "part-1.png — layout layout-7c1f2a revision 1" in layout_text
    assert "col-9f3a71 → Plain" in layout_text
    assert "printed: plain form " in layout_text
    assert "No layout is bound for: part-2.png" in layout_text

    curation_text = adapter.study_job_curation_text(job_id=job_id)
    assert "the parts disagree about" in curation_text
    assert "part-1.png" in curation_text and "part-2.png" in curation_text

    group = next(
        item
        for item in study_curation.read_curation_groups(config, job_id)
        if item.conflicting
    )
    told = adapter.apply_study_job_curation(
        job_id=job_id, target=f"{group.record_id}|part-1.png"
    )

    assert "staged file(s)" in told
    landed, _meta = staging.read_staging(second)
    settled = next(record for record in landed if record.id == group.record_id)
    assert settled.source_forms.cells == FIRST_CELLS
    saved = study_job.load_study_job(config, job_id)
    curation = [item for item in saved.intents if item.kind == "curation"]
    assert len(curation) == 1
    assert curation[0].intent_id in saved.closed_intent_ids


def test_a_bare_extraction_of_a_layout_bound_part_refuses_at_the_desk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§4.2: a layout-bound part planned under `auto` refuses.

    The Assistant's single-source extraction route hard-codes `mode=None`, so
    without this check a bound part would silently be sent to `extract-auto`.

    Mutant: delete `_refuse_layout_bound_part` from
    `assistant_adapter.prepare_source_extraction`.
    """
    from test_workbench_assistant_integration import _adapter

    from japanese_anki.workbench.assistant import RevisionRefusal

    _no_api(monkeypatch)
    config = _project(tmp_path)
    job = _job(config)
    _publish(config, job.header.job_id, PART_PAYLOADS)
    study_job.append_layout(
        config,
        job.header.job_id,
        _layout(),
        bind=("part-1.png",),
        expected_revision=study_job.load_study_job(
            config, job.header.job_id
        ).revision,
    )
    adapter = _adapter(config)

    with pytest.raises(RevisionRefusal) as refusal:
        adapter.prepare_source_extraction(
            source_path=config.scan_inbox / "part-1.png"
        )
    assert "part-1.png" in str(refusal.value)
    assert job.header.job_id in str(refusal.value)
    assert extract.LAYOUT_MODE in str(refusal.value)

    # A part nothing binds is not named by the reader the check consults, so
    # an ordinary extraction of it is untouched.
    assert study_job.parts_with_bound_layouts(config) == {
        "part-1.png": (job.header.job_id,)
    }
    adapter._refuse_layout_bound_part(config, config.scan_inbox / "part-2.png")


def test_a_batch_over_layout_bound_parts_refuses_on_the_same_terms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The multi-source route enforces what the single-source route does.

    Published parts are ordinary disclosed sources, so a model may name two of
    them in an `extract_batch` plan. Which mode a bound part is sent under is a
    structural artifact contract, so the check belongs beside the plan rather
    than only in the prompt — and it runs before planning, so no provider is
    probed and nothing is dispatched.

    Mutant: delete the per-source check from
    `assistant_adapter.prepare_source_extraction_batch`.
    """
    from test_workbench_assistant_integration import _adapter

    from japanese_anki.workbench.assistant import RevisionRefusal

    _no_api(monkeypatch)
    config = _project(tmp_path)
    job = _job(config)
    _publish(config, job.header.job_id, PART_PAYLOADS)
    study_job.append_layout(
        config,
        job.header.job_id,
        _layout(),
        bind=("part-2.png",),
        expected_revision=study_job.load_study_job(
            config, job.header.job_id
        ).revision,
    )
    adapter = _adapter(config)

    def unreachable(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the batch was planned over a bound part")

    monkeypatch.setattr(
        extraction_batch, "plan_extraction_batch", unreachable
    )

    with pytest.raises(RevisionRefusal) as refusal:
        adapter.prepare_source_extraction_batch(
            source_paths=[
                config.scan_inbox / "part-1.png",
                config.scan_inbox / "part-2.png",
            ]
        )

    assert "part-2.png" in str(refusal.value)
    assert job.header.job_id in str(refusal.value)
    assert extract.LAYOUT_MODE in str(refusal.value)


def _injected_planner(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real batch planner, with only the subscription transport faked."""

    real = extraction_batch.plan_extraction_batch

    def planned(config: Any, sources: Any, **options: Any) -> Any:
        options.setdefault("provider_env", {})
        options.setdefault(
            "provider_runner",
            FakeClaudeRunner(reply=_settled_stream(_answer(FIRST_CELLS))),
        )
        options.setdefault("provider_which", _which)
        return real(config, sources, **options)

    monkeypatch.setattr(extraction_batch, "plan_extraction_batch", planned)


def test_the_assistants_own_batch_route_pins_the_mode_this_job_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§4.2 and §9.2, on the only Assistant route to a job's batch.

    No mode is model-emittable and the adapter carries none, so a job whose
    parts the owner bound has to pin `table-layout` from those bindings itself.
    The exact confirmation then names the pinned mode and each child's layout
    revision, which is what the owner is agreeing to send.

    Mutant: forward `plan_options.get("mode")` unchanged out of
    `plan_job_extraction_batch`, so the Assistant's mode-less call refuses.
    """
    from test_workbench_assistant_integration import _adapter

    _no_api(monkeypatch)
    config = _project(tmp_path)
    job = _job(config)
    job_id = job.header.job_id
    _publish(config, job_id, PART_PAYLOADS)
    saved = study_job.append_layout(
        config,
        job_id,
        _layout(),
        bind=("part-1.png",),
        expected_revision=study_job.load_study_job(config, job_id).revision,
    )
    study_job.append_layout(
        config,
        job_id,
        _layout(revision=2),
        bind=("part-2.png",),
        expected_revision=saved.revision,
    )
    _injected_planner(monkeypatch)
    adapter = _adapter(config)

    prepared = adapter.prepare_study_job_batch(job_id=job_id, deck_scope="")

    assert [child.expectation.mode for child in prepared.plan.children] == [
        extract.LAYOUT_MODE,
        extract.LAYOUT_MODE,
    ]
    wire = "\n".join((*prepared.effects, *prepared.disclosures))
    assert extract.LAYOUT_MODE in wire
    assert "layout-7c1f2a revision 1" in wire
    assert "layout-7c1f2a revision 2" in wire


# =============================================================================
# The owner's layout editor, end to end through the Assistant
# =============================================================================


def _job_resource_id(config: ProjectConfig, job_id: str) -> str:
    """The opaque identity the model sees for this job. Never its path."""

    broker = assistant_context.AssistantContextBroker(config)
    catalog = json.loads(broker.catalog().wire)["data"]
    for entry in catalog["resources"]:
        if entry["kind"] == "study_job" and broker.study_job_id(
            entry["resource_id"]
        ) == job_id:
            return str(entry["resource_id"])
    raise AssertionError("the study job was not disclosed to the model")


def _layout_sidecar(config: ProjectConfig) -> Any:
    """One real sidecar whose layout editor store is the real adapter's."""

    from test_workbench_assistant_integration import _adapter

    from japanese_anki.workbench.assistant_http import create_assistant_sidecar

    adapter = _adapter(config)
    sidecar = create_assistant_sidecar(adapter, deck_choices=())
    sidecar.server.test_adapter = adapter
    sidecar.start()
    return sidecar


def _http(
    sidecar: Any, method: str, path: str, body: bytes | None = None
) -> tuple[int, bytes]:
    import http.client

    connection = http.client.HTTPConnection(
        "127.0.0.1", sidecar.server.server_address[1], timeout=5
    )
    headers = (
        {"Content-Type": "application/json", "Origin": sidecar.origin}
        if body is not None
        else {}
    )
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    payload = response.read()
    connection.close()
    return response.status, payload


def _editor_path(text: str) -> str:
    """The local path of the editor link this reply rendered."""

    url = text.split("](")[1].split(")")[0]
    return "/" + url.split("/", 3)[3]


def _save(
    sidecar: Any, path: str, payload: dict[str, Any]
) -> tuple[int, dict[str, Any]]:
    status, body = _http(
        sidecar, "POST", path, json.dumps(payload, ensure_ascii=False).encode("utf-8")
    )
    return status, json.loads(body)


def test_the_owner_writes_a_layout_in_the_editor_and_the_batch_binds_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§4.1 and §9.1, from unbound parts to a correctly bound batch.

    The model may open the editor and nothing more; the owner types the
    printed headings, the display labels and the order inside it, and janki
    mints every machine identifier there — no `layout_id`, `revision` or
    `column_id` is invented in a text editor or carried between two commands.
    Saving appends the immutable revision and repoints the parts in one write,
    and the job then pins its own batch mode from those bindings.

    Mutant: have `open_layout_editor` answer with the provenance listing again
    instead of opening the editor, or mint no identity in
    `_save_layout_for_owner`.
    """
    from test_workbench_assistant_integration import _agent_intent, _agent_result

    _no_api(monkeypatch)
    config = _project(tmp_path)
    job = _job(config)
    job_id = job.header.job_id
    _publish(config, job_id, PART_PAYLOADS)
    sidecar = _layout_sidecar(config)
    try:
        adapter = sidecar.server.test_adapter
        result = _agent_result(
            answer="You choose the columns; I cannot see the page.",
            intents=(
                _agent_intent(
                    kind="open_layout_editor",
                    resource_ids=(_job_resource_id(config, job_id),),
                    instruction="Open the layout editor for this job.",
                ),
            ),
        )
        reply = adapter._prepare_agent_intent(config, result=result, deck_scope="")

        # The model's intent opens the owner's editor; it saves nothing.
        assert reply.action is None
        assert "/layouts/" in reply.text
        path = _editor_path(reply.text)
        status, served = _http(sidecar, "GET", path)
        assert status == 200
        assert b"part-1.png" in served and b"part-2.png" in served
        assert study_job.load_study_job(config, job_id).layouts == {}

        # The owner's own save: their witnesses, their labels, their order.
        status, saved = _save(
            sidecar,
            path,
            {
                "action": "save",
                "base_layout_id": "",
                "base_revision": None,
                "columns": [
                    {
                        "display_label": "Form",
                        "label_witnesses": ["plain form ", "辞書形"],
                    },
                    {"display_label": "Form", "label_witnesses": ["polite form"]},
                ],
                "bind": ["part-1.png", "part-2.png"],
            },
        )

        assert status == 200 and saved["ok"] is True
        first = study_job.job_layout_bindings(
            study_job.load_study_job(config, job_id)
        )
        assert set(first) == {"part-1.png", "part-2.png"}
        layout = first["part-1.png"]
        assert layout.layout_id == saved["layout_id"] and layout.revision == 1
        # Duplicate display labels are representable, because the identities
        # differ; the printed witnesses are kept exactly as they were typed.
        assert [column.display_label for column in layout.columns] == ["Form", "Form"]
        assert [column.ordinal for column in layout.columns] == [1, 2]
        assert layout.columns[0].label_witnesses == ("plain form ", "辞書形")
        assert layout.columns[1].label_witnesses == ("polite form",)
        minted = [column.column_id for column in layout.columns]
        assert len(set(minted)) == 2 and all(value.strip() for value in minted)

        # An edit: the retained column keeps its identity, the dropped one is
        # gone, and the new one gets its own. The old revision is untouched.
        status, edited = _save(
            sidecar,
            path,
            {
                "action": "save",
                "base_layout_id": saved["layout_id"],
                "base_revision": 1,
                "columns": [
                    {
                        "column_id": minted[0],
                        "display_label": "Dictionary",
                        "label_witnesses": ["plain form "],
                    },
                    {"display_label": "Past", "label_witnesses": ["past form"]},
                ],
                "bind": ["part-1.png", "part-2.png"],
            },
        )

        assert status == 200 and edited["revision"] == 2
        recorded = study_job.load_study_job(config, job_id)
        assert set(recorded.layouts) == {
            f"{saved['layout_id']}@1",
            f"{saved['layout_id']}@2",
        }
        assert recorded.layouts[f"{saved['layout_id']}@1"]["columns"][0][
            "display_label"
        ] == "Form"
        second = study_job.job_layout_bindings(recorded)["part-2.png"]
        assert second.revision == 2
        assert second.columns[0].column_id == minted[0]
        assert second.columns[0].display_label == "Dictionary"
        assert second.columns[1].column_id not in minted

        # And the batch this job would send is bound to what was saved.
        _injected_planner(monkeypatch)
        plan = study_job.plan_job_extraction_batch(config, job_id)
        assert [child.expectation.mode for child in plan.children] == [
            extract.LAYOUT_MODE,
            extract.LAYOUT_MODE,
        ]
        assert [
            child.expectation.table_layout.identity for child in plan.children
        ] == [second.identity, second.identity]
    finally:
        sidecar.close()


def test_a_stale_or_invented_layout_save_is_refused_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The editor's save binds what the owner opened, and nothing else.

    A column identity is janki's to mint, so one the posted body invented is
    refused rather than recorded; a part this job never published is refused;
    and a token this workbench is no longer holding cannot write at all.

    Mutant: accept any posted `column_id`, or read the job from the request
    body instead of the editor's own session.
    """
    from test_workbench_assistant_integration import _adapter

    _no_api(monkeypatch)
    config = _project(tmp_path)
    job = _job(config)
    job_id = job.header.job_id
    _publish(config, job_id, PART_PAYLOADS)
    sidecar = _layout_sidecar(config)
    try:
        adapter = sidecar.server.test_adapter
        offer = adapter.open_layout_editor(job_id=job_id)
        path = "/" + offer.url.split("/", 3)[3]
        column = {"display_label": "Plain", "label_witnesses": ["plain form"]}

        status, invented = _save(
            sidecar,
            path,
            {
                "action": "save",
                "columns": [{**column, "column_id": "col-9f3a71"}],
                "bind": ["part-1.png"],
            },
        )
        assert status == 409 and invented["ok"] is False
        assert "col-9f3a71" in invented["error"]

        status, unpublished = _save(
            sidecar,
            path,
            {"action": "save", "columns": [column], "bind": ["part-9.png"]},
        )
        assert status == 409 and "part-9.png" in unpublished["error"]

        status, unbound = _save(
            sidecar, path, {"action": "save", "columns": [column], "bind": []}
        )
        assert status == 409 and "at least one published part" in unbound["error"]

        # A token this workbench never minted, and one it no longer holds.
        status, unknown = _save(
            sidecar,
            "/".join(path.split("/")[:-1]) + "/" + "z" * 43,
            {"action": "save", "columns": [column], "bind": ["part-1.png"]},
        )
        assert status == 409 and unknown["ok"] is False

        assert study_job.load_study_job(config, job_id).layouts == {}
        assert study_job.job_layout_bindings(
            study_job.load_study_job(config, job_id)
        ) == {}
        # An unserved workbench says which control saves the same revision.
        with pytest.raises(Exception) as refusal:
            _adapter(config).open_layout_editor(job_id=job_id)
        assert "janki study layout" in str(refusal.value)
    finally:
        sidecar.close()


def test_the_jobs_own_resume_finishes_an_interrupted_curation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§6.5: the barrier this job's own control created has a way out of it.

    The recorded payload *is* the artifact — `reserves["prepared"]` holds the
    complete prepared text — so neither the job's status nor its resume may
    report it as an artifact janki cannot find. Resume replays the recorded
    decision through the curation service and appends its outcome.

    Mutant: drop the `curation` branch from `_resolve_intent`, so the intent
    resolves to nothing and resume reports a missing artifact.
    """
    from test_workbench_assistant_integration import _adapter

    config, job_id, first, second = _curated_job(tmp_path, monkeypatch)
    group = next(
        item
        for item in study_curation.read_curation_groups(config, job_id)
        if item.conflicting
    )
    plan = study_curation.plan_curation(
        config,
        job_id,
        (
            study_curation.CurationChoice(
                record_id=group.record_id,
                action="replace",
                key="col-2b8d04",
                value="話しました",
            ),
        ),
        decision="the polite column prints 話しました here",
    )
    # The crash: the intent is durable and no file has been written.
    intent = study_curation._curation_intent(plan)
    study_job.append_intent(
        config,
        job_id,
        intent,
        expected_revision=study_job.load_study_job(config, job_id).revision,
    )

    status = study_job.study_job_status(config, job_id)
    assert status.pending_curation_intents == (intent.intent_id,)
    assert intent.intent_id not in status.unresolved_intents

    told = _adapter(config).resume_study_job(
        job_id=job_id, progress=lambda _label: None
    )

    assert "could not resolve" not in told
    assert intent.intent_id in told
    for entry in plan.prepared:
        landed = (config.root / entry.staging_path).read_text(encoding="utf-8")
        assert landed == entry.content_after
    assert study_curation.open_curation_barriers(config) == ()
    assert intent.intent_id in study_job.load_study_job(
        config, job_id
    ).closed_intent_ids
    assert study_curation.pending_curation_refusal(config, second) == ""


def _wedged_decision(
    config: ProjectConfig, job_id: str, moved: Path
) -> Any:
    """The state no replay can leave: a recorded decision, then an edit.

    Exactly the sequence §6.3 step 6 plans for followed by an ordinary
    supported reviewer action. The intent is durable and one bound file is now
    at neither digest it recorded, so `resume_curation` refuses and will keep
    refusing however many times it is run.
    """

    plan = _one_cell_plan(config, job_id, "話しました")
    intent = study_curation._curation_intent(plan)
    study_job.append_intent(
        config,
        job_id,
        intent,
        expected_revision=study_job.load_study_job(config, job_id).revision,
    )
    moved.write_text(
        moved.read_text(encoding="utf-8") + "\n# a later hand edit\n",
        encoding="utf-8",
    )
    return plan, intent


def test_an_unsatisfiable_decision_is_closed_by_the_owners_own_abandonment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§6.2: abandonment is an explicit bound local action, not a rollback.

    S5 ships a durable barrier, so it has to ship the thing that retracts one.
    A crash between the fsynced intent and its writes, followed by any ordinary
    edit of a bound file — the staging editor, a coverage approval, hand-edited
    YAML — leaves the file at neither recorded digest. From there every resume
    refuses forever, no superseding intent may be appended while the
    predecessor is open, and the barrier blocks promotion of *every* file the
    decision named. This closes it: it records the mixed snapshot it measured,
    reverts nothing, deletes no evidence, and lets the owner decide again.

    Mutant: fill the `abandoned` outcome's `observed` from the intent's own
    recorded digests instead of measuring the files under the locks.
    """
    config, job_id, first, second = _curated_job(tmp_path, monkeypatch)
    plan, intent = _wedged_decision(config, job_id, second)
    wedged = {
        config.root / entry.staging_path: (
            config.root / entry.staging_path
        ).read_bytes()
        for entry in plan.prepared
    }

    # The state it is in: no replay can finish it, and it holds both files.
    with pytest.raises(study_curation.StudyCurationError):
        study_curation.resume_curation(config, job_id, intent.intent_id)
    barrier = study_curation.open_curation_barriers(config)[0]
    assert barrier.unsatisfiable == (plan.prepared[-1].staging_path,)
    assert study_curation.pending_curation_refusal(config, first) != ""
    assert study_curation.pending_curation_refusal(config, second) != ""
    # The refusal names the route that is actually open, and carries the digest
    # that route needs, rather than pointing at a resume that cannot work.
    refusal = study_curation.pending_curation_refusal(config, first)
    assert "--abandon" in refusal and barrier.intent_sha256 in refusal

    outcome = study_curation.abandon_intent(
        config,
        job_id,
        intent.intent_id,
        expected_intent_sha256=barrier.intent_sha256,
    )

    assert outcome.state == "abandoned"
    assert outcome.written == ()
    # Evidence, measured: each path at the digest it really holds, including
    # the edited one that matches neither recorded value.
    assert dict(outcome.observed) == {
        entry.staging_path: hashlib.sha256(
            (config.root / entry.staging_path).read_bytes()
        ).hexdigest()
        for entry in plan.prepared
    }
    recorded = study_job.load_study_job(config, job_id).outcomes[-1]
    assert recorded.state == "abandoned"
    assert recorded.observed == outcome.observed
    assert recorded.consequences["reverted"] is False
    assert recorded.consequences["at_neither"] == [plan.prepared[-1].staging_path]

    # Nothing was written and nothing was restored: abandoning is not undoing.
    for path, bytes_before in wedged.items():
        assert path.read_bytes() == bytes_before
    # The intent and its evidence stay in the log; only an outcome was added.
    saved = study_job.load_study_job(config, job_id)
    assert intent in saved.intents
    assert intent.intent_id in saved.closed_intent_ids

    # The barrier is lifted, both files can be promoted again, and the owner's
    # superseding decision — refused while the predecessor was open — applies.
    assert study_curation.open_curation_barriers(config) == ()
    assert study_curation.pending_curation_refusal(config, first) == ""
    assert study_curation.pending_curation_refusal(config, second) == ""
    fresh = _one_cell_plan(config, job_id, "話しました")
    settled = study_curation.apply_curation(config, fresh)
    assert settled.state == "applied"
    assert set(settled.written) == {item.staging_path for item in fresh.prepared}


def test_an_abandonment_binds_the_exact_decision_it_was_read_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§9.1: an owner control binds the exact resource and snapshot digest.

    Closing a durable decision without writing it is the owner's call over one
    decision they read. A control carrying any other digest is refused, so a
    stale desk row or a mistyped id cannot close a decision nobody looked at,
    and an already-closed one cannot be closed twice.

    Mutant: drop the `expected_intent_sha256` comparison from
    `study_curation.abandon_intent`.
    """
    config, job_id, _first, second = _curated_job(tmp_path, monkeypatch)
    _plan, intent = _wedged_decision(config, job_id, second)
    digest = study_curation.curation_intent_digest(intent)

    for wrong in ("", "0" * 64, digest[:-1] + ("0" if digest[-1] != "0" else "1")):
        with pytest.raises(study_curation.StudyCurationError) as refusal:
            study_curation.abandon_intent(
                config, job_id, intent.intent_id, expected_intent_sha256=wrong
            )
        assert digest in str(refusal.value)
    assert [
        barrier.intent_id
        for barrier in study_curation.open_curation_barriers(config)
    ] == [intent.intent_id]

    study_curation.abandon_intent(
        config, job_id, intent.intent_id, expected_intent_sha256=digest
    )

    # One outcome, once: outcomes are append-only and never rewritten.
    with pytest.raises(study_curation.StudyCurationError, match="already has an"):
        study_curation.abandon_intent(
            config, job_id, intent.intent_id, expected_intent_sha256=digest
        )
    assert [
        outcome.state
        for outcome in study_job.load_study_job(config, job_id).outcomes
        if outcome.intent_id == intent.intent_id
    ] == ["abandoned"]


def test_a_bound_file_that_is_gone_is_part_of_the_snapshot_not_a_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Abandoning is what an owner does once the bound set has moved.

    A path this decision named is no longer readable at all. Refusing to close
    the decision over that would be the F1 wedge again, one level up: the
    barrier would hold the *other* bound file forever because of a file that
    no longer exists. It is recorded as what it is and the intent closes.

    Mutant: hash the bound path through `_digest` in `abandon_intent`, so an
    unreadable one raises instead of being recorded.
    """
    config, job_id, first, second = _curated_job(tmp_path, monkeypatch)
    plan, intent = _wedged_decision(config, job_id, second)
    second.unlink()

    outcome = study_curation.abandon_intent(
        config,
        job_id,
        intent.intent_id,
        expected_intent_sha256=study_curation.curation_intent_digest(intent),
    )

    assert dict(outcome.observed)[plan.prepared[-1].staging_path] == (
        study_curation.MISSING_DIGEST
    )
    assert dict(outcome.observed)[plan.prepared[0].staging_path] == (
        hashlib.sha256(first.read_bytes()).hexdigest()
    )
    recorded = study_job.load_study_job(config, job_id).outcomes[-1]
    assert recorded.state == "abandoned"
    assert recorded.consequences["at_neither"] == [plan.prepared[-1].staging_path]
    assert not second.exists()
    assert study_curation.pending_curation_refusal(config, first) == ""


def test_a_plan_that_no_longer_fits_records_no_decision_at_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§6.3 step 1: re-read the whole bound set under the guard, then record.

    Both surfaces plan and then apply, and `plan_curation` reads the files
    outside the coordination guard. Without a re-read under it, a file that
    moved in between makes an intent that is unsatisfiable the moment it
    becomes durable — a barrier born already needing to be abandoned, over a
    decision the owner never got to make.

    Mutant: delete the pre-record precheck from `study_curation.apply_curation`,
    so the intent is appended before anything checks the files.
    """
    config, job_id, first, second = _curated_job(tmp_path, monkeypatch)
    plan = _one_cell_plan(config, job_id, "話しました")
    assert len(plan.prepared) == 2
    second.write_text(
        second.read_text(encoding="utf-8") + "\n# a later hand edit\n",
        encoding="utf-8",
    )
    snapshot = {
        config.root / entry.staging_path: (
            config.root / entry.staging_path
        ).read_bytes()
        for entry in plan.prepared
    }

    with pytest.raises(study_curation.StudyCurationError) as refusal:
        study_curation.apply_curation(config, plan)

    assert plan.prepared[-1].staging_path in str(refusal.value)
    assert "nothing was recorded" in str(refusal.value)
    # No intent, so no barrier, and no file moved.
    saved = study_job.load_study_job(config, job_id)
    assert [item for item in saved.intents if item.kind == "curation"] == []
    assert study_curation.open_curation_barriers(config) == ()
    assert study_curation.pending_curation_refusal(config, first) == ""
    for path, bytes_before in snapshot.items():
        assert path.read_bytes() == bytes_before

    # Planned again over the files as they now stand, it applies normally.
    study_curation.apply_curation(
        config, _one_cell_plan(config, job_id, "話しました")
    )
    assert study_curation.open_curation_barriers(config) == ()


def test_the_desk_closes_an_unsatisfiable_decision_without_writing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Assistant half of §6.2, against the real stores.

    The primary surface (§9.3) must reach the recovery its own controls can
    land the owner in. The desk renders one close control per open decision,
    carrying the intent id and the digest of the exact intent that render read.

    Mutant: have `abandon_study_job_curation` read the intent from anywhere but
    its bound target, or drop the digest half of that target.
    """
    from test_workbench_assistant_integration import _adapter

    from japanese_anki.workbench.assistant import RevisionRefusal

    config, job_id, first, second = _curated_job(tmp_path, monkeypatch)
    _plan, intent = _wedged_decision(config, job_id, second)
    adapter = _adapter(config)

    rendered = [
        action
        for choice in adapter.list_study_job_choices()
        if choice.job_id == job_id
        for action in choice.actions
        if action.action == "abandon-curation"
    ]
    assert len(rendered) == 1
    assert rendered[0].target == (
        f"{intent.intent_id}|{study_curation.curation_intent_digest(intent)}"
    )
    assert "no replay can finish it" in rendered[0].label

    # A target naming the right decision at the wrong digest is refused.
    with pytest.raises(RevisionRefusal):
        adapter.abandon_study_job_curation(
            job_id=job_id, target=f"{intent.intent_id}|{'0' * 64}"
        )
    assert study_curation.open_curation_barriers(config) != ()

    told = adapter.abandon_study_job_curation(
        job_id=job_id, target=rendered[0].target
    )

    assert intent.intent_id in told
    assert "without writing it" in told
    assert study_curation.open_curation_barriers(config) == ()
    saved = study_job.load_study_job(config, job_id)
    assert saved.outcomes[-1].state == "abandoned"
    assert study_curation.pending_curation_refusal(config, first) == ""


# =============================================================================
# §6.2 — an unresolved decision blocks its successor, and a replan says so
# =============================================================================


#: The second job's own recipe. A publication receipt is immutable and keyed by
#: its recipe id, so a different part set is a different recipe.
OTHER_RECIPE_ID = "8c2b5d41-6e7a-4f19-b0c3-2a9d8e7f6b51"


def _conflicting_groups(config: ProjectConfig, job_id: str) -> list[Any]:
    """Every identity this job's parts currently disagree about."""

    return [
        item
        for item in study_curation.read_curation_groups(config, job_id)
        if item.conflicting
    ]


def _barrier_digest(config: ProjectConfig, job_id: str, intent_id: str) -> str:
    """The digest the desk and the CLI both bind that decision by."""

    return study_curation.curation_intent_digest(
        study_job.load_study_job(config, job_id).intent(intent_id)
    )


def test_an_open_decision_blocks_the_next_one_over_the_same_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§6.2: a decision never finished or abandoned still blocks its successor.

    The crash §6.3 step 6 plans for leaves a durable intent whose bound files
    still hold their `sha256_before` bytes, so a resume can still finish it.
    Recording a *second* decision over those same files would land bytes that
    are neither digest the first recorded: the first becomes unsatisfiable the
    moment the second is written, its barrier holds every file it named, and a
    decision that succeeded has left the job needing a recovery action. The
    successor refuses instead — under the coordination guard, before anything
    is recorded — and names the two routes out of the state that is actually
    there.

    Mutant: delete the unresolved-predecessor block from
    `study_curation.apply_curation`.
    """
    config, job_id, first, second = _curated_job(tmp_path, monkeypatch)
    open_id = _open_barrier(config, job_id, first)
    snapshot = {path: path.read_bytes() for path in (first, second)}
    barrier = study_curation.open_curation_barriers(config)[0]
    # The satisfiable case: nothing has moved, so a replay can still land.
    assert barrier.intent_id == open_id
    assert barrier.unsatisfiable == ()

    plan = _one_cell_plan(config, job_id, "話しました")
    with pytest.raises(study_curation.StudyCurationError) as refusal:
        study_curation.apply_curation(config, plan)

    told = str(refusal.value)
    assert open_id in told and job_id in told
    assert f"--resume {open_id}" in told
    assert f"--abandon {open_id} --expect {barrier.intent_sha256}" in told
    assert "Nothing was recorded and nothing was written." in told
    # No second intent, so no second barrier, and no file moved.
    saved = study_job.load_study_job(config, job_id)
    assert [
        item.intent_id for item in saved.intents if item.kind == "curation"
    ] == [open_id]
    assert [
        item.intent_id for item in study_curation.open_curation_barriers(config)
    ] == [open_id]
    for path, before in snapshot.items():
        assert path.read_bytes() == before

    # The route it named finishes the predecessor, and the replan then records
    # the edge back to the decision it settles again.
    finished = study_curation.resume_curation(config, job_id, open_id)
    assert finished.state == "applied"
    replanned = study_curation.apply_curation(
        config, _one_cell_plan(config, job_id, "話しました")
    )

    assert replanned.state == "applied"
    assert replanned.supersedes == open_id
    recorded = study_job.load_study_job(config, job_id).intents[-1]
    assert recorded.kind == "curation"
    assert recorded.supersedes == open_id


def test_an_open_decision_about_this_identity_blocks_it_from_another_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§6.2's successor rule is about the decision, not only about the files.

    A recorded decision can bind a file this plan does not write — the row it
    settled is no longer staged there, or the file it named is gone. It is
    still this job's open decision about this identity, so the replan waits
    for it to reach an outcome. The refusal says which of the two routes is
    open, and the durable log enforces the same thing under its own
    compare-and-swap: a successor superseding an intent with no outcome is
    refused by `study_job.append_intent`, whatever a caller checked first.

    Mutant: consult only the paths in `unresolved_curation_predecessors`, so
    this decision is recorded ahead of the one it settles again.
    """
    config, job_id, first, second = _curated_job(tmp_path, monkeypatch)
    plan = _one_cell_plan(config, job_id, "話しました")
    snapshot = {path: path.read_bytes() for path in (first, second)}
    bound_elsewhere = study_curation.PreparedFileUpdate(
        staging_path="staging/part-9.png.yaml",
        sha256_before="1" * 64,
        sha256_after="2" * 64,
        content_after="records: []\n",
        record_ids=(plan.choices[0].record_id,),
    )
    open_elsewhere = study_job.ActionIntent(
        intent_id=study_job.new_intent_id(),
        kind="curation",
        decided_at="2026-01-01T00:00:00+00:00",
        reserves={"prepared": [bound_elsewhere.to_dict()]},
        bindings={
            "job_id": job_id,
            "decision": "the polite column printed 話します in that part",
            "choices": [
                {
                    "record_id": plan.choices[0].record_id,
                    "field": "source_forms",
                    "action": "replace",
                    "key": "col-2b8d04",
                    "value": "話します",
                }
            ],
        },
    )
    study_job.append_intent(
        config,
        job_id,
        open_elsewhere,
        expected_revision=study_job.load_study_job(config, job_id).revision,
    )
    assert bound_elsewhere.staging_path not in set(plan.paths)

    with pytest.raises(study_curation.StudyCurationError) as refusal:
        study_curation.apply_curation(config, plan)

    told = str(refusal.value)
    assert open_elsewhere.intent_id in told
    assert "settles the same identity" in told
    assert f"--abandon {open_elsewhere.intent_id} --expect" in told
    assert "Nothing was recorded and nothing was written." in told
    saved = study_job.load_study_job(config, job_id)
    assert [
        item.intent_id for item in saved.intents if item.kind == "curation"
    ] == [open_elsewhere.intent_id]
    for path, before in snapshot.items():
        assert path.read_bytes() == before

    # The same refusal, from the durable log itself: whatever a caller checked,
    # a successor cannot be appended over a predecessor with no outcome.
    with pytest.raises(JankiError, match="no outcome yet"):
        study_job.append_intent(
            config,
            job_id,
            study_curation._curation_intent(
                plan, supersedes=open_elsewhere.intent_id
            ),
            expected_revision=saved.revision,
        )


def test_a_replan_after_a_closed_decision_records_the_edge_it_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§6.2: the successor carries `supersedes: intent_id`, and nothing else.

    An owner who changes their mind gets a replan — a fresh intent computed
    against fresh snapshots — and the original intent and its evidence are
    never deleted or rewritten. There is no `superseded_by` to write, so the
    link lives on the successor, where `study_job.append_intent` is the thing
    enforcing that the predecessor already has an outcome.

    Mutant: record the successor with no `supersedes`, so the durable log
    keeps no link between a replan and the decision it replaced.
    """
    config, job_id, _first, second = _curated_job(tmp_path, monkeypatch)
    plan, intent = _wedged_decision(config, job_id, second)
    original_wire = intent.to_dict()
    study_curation.abandon_intent(
        config,
        job_id,
        intent.intent_id,
        expected_intent_sha256=study_curation.curation_intent_digest(intent),
    )
    stale = {item.staging_path: item.sha256_before for item in plan.prepared}
    moved = plan.prepared[-1].staging_path
    at_replan = hashlib.sha256(second.read_bytes()).hexdigest()

    outcome = study_curation.apply_curation(
        config, _one_cell_plan(config, job_id, "話しましょう")
    )

    assert outcome.state == "applied"
    assert outcome.supersedes == intent.intent_id
    saved = study_job.load_study_job(config, job_id)
    successor = saved.intent(outcome.intent_id)
    assert successor.supersedes == intent.intent_id
    # Fresh snapshots, not the abandoned decision's: the successor was computed
    # over the file as it now stands, which is neither digest that one recorded.
    fresh = {
        str(item["staging_path"]): str(item["sha256_before"])
        for item in successor.reserves["prepared"]
    }
    assert fresh[moved] == at_replan
    assert fresh[moved] != stale[moved]
    # The original is immutable evidence: still recorded, byte-identical wire,
    # and its own outcome is still the one it was closed with.
    assert saved.intent(intent.intent_id).to_dict() == original_wire
    curation_ids = {
        item.intent_id for item in saved.intents if item.kind == "curation"
    }
    assert [
        (item.intent_id, item.state)
        for item in saved.outcomes
        if item.intent_id in curation_ids
    ] == [(intent.intent_id, "abandoned"), (outcome.intent_id, "applied")]

    # A third decision on that identity supersedes the second, not the first.
    third = study_curation.apply_curation(
        config, _one_cell_plan(config, job_id, "話しません")
    )
    assert third.supersedes == outcome.intent_id


def test_a_later_unrelated_decision_is_neither_blocked_nor_recorded_as_a_replan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only an *unresolved* predecessor blocks, and only a replan is an edge.

    Two ordinary things stay ordinary. A decision about a different identity
    is not a replan of the settled one — it carries no `supersedes`, because
    the durable log would otherwise claim it replaced a decision it has
    nothing to do with. And another job's open decision over its own parts
    blocks nothing here: the block is bound to the files this decision would
    write and to the identities it settles again.

    Mutant: block on any open barrier in the project, or take `supersedes`
    from the latest curation intent rather than the one being settled again.
    """
    config, job_id, first, second = _curated_job(tmp_path, monkeypatch)
    settled = study_curation.apply_curation(
        config, _one_cell_plan(config, job_id, "話しました")
    )
    assert settled.state == "applied"
    assert settled.supersedes == ""

    # A second job, its own published part, its own open decision.
    other = _job(config)
    _publish(
        config,
        other.header.job_id,
        {"part-3.png": b"\x89PNG part three"},
        recipe_id=OTHER_RECIPE_ID,
        plan_fingerprint="e" * 64,
    )
    third = _stage(config, config.scan_inbox / "part-3.png", FIRST_CELLS)
    elsewhere = _open_barrier(config, other.header.job_id, third)
    assert elsewhere in {
        item.intent_id for item in study_curation.open_curation_barriers(config)
    }
    assert study_curation.pending_curation_refusal(config, first) == ""

    later = _conflicting_groups(config, job_id)[1]
    plan = study_curation.plan_curation(
        config,
        job_id,
        (
            study_curation.CurationChoice(
                record_id=later.record_id,
                action="replace",
                key="col-2b8d04",
                value="食べました",
            ),
        ),
        decision="the polite column prints 食べました here",
    )
    outcome = study_curation.apply_curation(config, plan)

    assert outcome.state == "applied"
    assert outcome.supersedes == ""
    assert set(outcome.written) == set(plan.paths)
    assert set(plan.paths) == {
        study_curation._relative(config, first),
        study_curation._relative(config, second),
    }
    # The other job's decision was neither closed nor touched by any of this.
    assert elsewhere in {
        item.intent_id for item in study_curation.open_curation_barriers(config)
    }
    assert study_curation.pending_curation_refusal(config, third) != ""


def test_a_curation_control_rendered_before_an_open_decision_refuses_at_the_desk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§9.1: a control minted over one state does not land on another.

    The desk renders its adopt controls from the identities its read saw. A
    decision recorded between that render and the click — a crash-recovered
    intent, the CLI, another window — is exactly the case that would wedge:
    the click would write over the before-state that decision is waiting for.
    Both surfaces reach the same service, so both refuse, and the desk already
    offers the close control the refusal names.

    Mutant: consult the open barriers in `cli_study._command_curate` rather
    than in the shared `study_curation.apply_curation`, so the desk's own
    control walks straight past it.
    """
    from test_workbench_assistant_integration import _adapter

    from japanese_anki.workbench.assistant import RevisionRefusal

    config, job_id, first, second = _curated_job(tmp_path, monkeypatch)
    adapter = _adapter(config)
    rendered = [
        action
        for choice in adapter.list_study_job_choices()
        if choice.job_id == job_id
        for action in choice.actions
        if action.action == "adopt-forms"
    ]
    assert rendered

    # The decision that lands between the render and the click.
    open_id = _open_barrier(config, job_id, first)
    snapshot = {path: path.read_bytes() for path in (first, second)}

    with pytest.raises(RevisionRefusal) as refusal:
        adapter.apply_study_job_curation(job_id=job_id, target=rendered[0].target)

    told = str(refusal.value)
    assert open_id in told
    assert "--resume" in told and "--abandon" in told
    for path, before in snapshot.items():
        assert path.read_bytes() == before
    saved = study_job.load_study_job(config, job_id)
    assert [
        item.intent_id for item in saved.intents if item.kind == "curation"
    ] == [open_id]
    # The owner is not stranded at the refusal: the desk renders the close
    # control for that exact decision.
    assert [
        action.target
        for choice in adapter.list_study_job_choices()
        if choice.job_id == job_id
        for action in choice.actions
        if action.action == "abandon-curation"
    ] == [f"{open_id}|{_barrier_digest(config, job_id, open_id)}"]

    # Once the predecessor has an outcome, the same control lands and the desk
    # says which closed decision this one was recorded over.
    assert study_curation.resume_curation(config, job_id, open_id).state == "applied"

    told = adapter.apply_study_job_curation(job_id=job_id, target=rendered[0].target)

    assert f"recorded over closed decision {open_id}" in told
    replan = study_job.load_study_job(config, job_id).intents[-1]
    assert replan.kind == "curation"
    assert replan.supersedes == open_id


def test_a_scoped_copy_carries_the_printed_table_with_the_rest_of_the_row(
    tmp_path: Path,
) -> None:
    """§1 step 9: a canonical `source_forms` rides onto the scoped copy.

    The scoped copy is a new record minted under the destination's scope; the
    shared original keeps its own id, GUID and history. Both carry the table
    its source printed.

    Mutant: rebuild the scoped copy field by field instead of `replace`-ing the
    current record, so a newly added field silently stops travelling.
    """
    from japanese_anki.application.assignment import (
        STANDALONE_COPY_FROM_KEY,
        AssignableWordDeck,
        _destination_record,
    )
    from japanese_anki.exporters.anki import DeckSelection

    record = VocabularyRecord.from_dict(
        {
            "id": "word:話す:はなす",
            "expression": "話す",
            "reading": "はなす",
            "meanings": ["to speak"],
            "source_forms": {
                "columns": [{"id": "col-9f3a71", "label": "Plain"}],
                "cells": {"col-9f3a71": "話す"},
            },
        }
    )
    destination = AssignableWordDeck(
        stem="standalone",
        path=tmp_path / "standalone.yaml",
        name="Standalone",
        intake_tag="verbs-intake",
        selection=DeckSelection(
            include_tags=frozenset({"verbs-intake"}),
            intake_tag="verbs-intake",
            scope_id="a1b2c3d4",
        ),
        scope_id="a1b2c3d4",
    )

    copied = _destination_record(record, destination, ())

    assert copied.id != record.id
    assert copied.source.raw_fields[STANDALONE_COPY_FROM_KEY] == record.id
    assert copied.source_forms == record.source_forms
    assert copied.to_dict()["source_forms"] == record.to_dict()["source_forms"]
