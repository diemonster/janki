"""How a bound table layout rides into the saved request, and back out of it.

The **request/provenance transport** for a layout-bound extraction, per
contracts §4.1–§4.4 and plan §10.2's "Layout and `source_forms`" row:

* the extraction wire schema stays byte-identical while the request changes;
* opaque `column_id`s, not printed labels, are what the request names, and two
  columns may share one display label;
* an ordinary extraction's provenance grows no key, and a saved ordinary
  capture normalizes to no layout;
* mode and layout must pair, in both directions;
* the whole frozen layout — not a `(layout_id, revision)` pair — is what the
  user turn and the request fingerprint are computed over;
* two inputs plan under their own aligned layouts in one `plan_extraction`
  call, and the whole-input staging-collision precheck still runs first;
* the **captured provenance**, never today's job choices, defines the layout.

Out of scope on purpose, because another candidate or another milestone owns
them: the `source_forms` serializer, merge, exporter and preview cases;
cross-part curation; the study job's `append_layout` and its CAS write; any
judgment about Japanese. Nothing here reads a printed header, matches a label,
or inspects a cell's text — every assertion is over identifiers, hashes and
key sets.

**Seam discipline.** A layout object is only ever built through the native
deserializers — `extract.TableLayout.from_wire` and
`extract.layout_from_provenance` — so no local stand-in can prove itself
instead of the implementation. The frozen `table_layout` JSON is spelled in
exactly one helper (`_frozen_layout_block`), so the wire contract is asserted
in one place and read everywhere else.

Every case's docstring names the one production mutant it was written to
catch. Nothing here reads Japanese: every assertion is over identifiers,
hashes, key sets and counts.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import uuid
from pathlib import Path
from typing import Any

import pytest
from test_extract import candidate
from test_revision_provider import FakeClaudeRunner, _which

from conftest import seed_prompts
from japanese_anki import claude_client, extract, prompts, staging
from japanese_anki.application.extraction import plan_extraction
from japanese_anki.config import ProjectConfig
from japanese_anki.inputs import PreparedInput

# --- the independent pre-S5 baseline ------------------------------------------
#
# Measured before any layout code existed and copied here verbatim. The literal
# constants and the checked-in artifact are two witnesses to the same fact, and
# each test below asserts they agree: regenerating either one from the code
# under test would turn this check into a tautology, which is precisely the
# failure mode a fingerprint baseline exists to prevent.

BASELINE_DIR = Path(__file__).resolve().parent / "fixtures"
BASELINE_RESPONSE_SCHEMA_FINGERPRINT = (
    "5447a46cd799d057b49f4f023ea1a4f881e617f0d79f0d22ef207f890fb03346"
)
BASELINE_WIRE_SCHEMA_SHA256 = (
    "cf45457974c455aa76481267ae40f1587e5d19ee4854423b1e17ef99635d2ec3"
)

#: §4.3: unchanged by this milestone, stated as a literal rather than read back
#: from the module whose constant it is checking.
EXPECTED_SCHEMA_VERSION = 5

#: The mode contracts §4.2 adds to `extract.MODES`. Also a literal: a test that
#: asked `extract.MODES` what the new mode is called would pass against a
#: module that added nothing.
LAYOUT_MODE = "table-layout"

#: The exact provenance key set an ordinary extraction writes today, on the
#: Anthropic API path. Written out rather than imported from `staging`, so a
#: change to the validator's own constant cannot make this agree with itself.
ORDINARY_PROVENANCE_KEYS = frozenset(
    {
        "source_sha256",
        "mode",
        "provider",
        "model",
        "response_schema_version",
        "response_schema_fingerprint",
        "system_prompt_fingerprint",
        "style_guide_fingerprint",
        "user_prompt_fingerprint",
        "request_fingerprint",
    }
)

#: What a shared-provider request saves beside those.
PROVIDER_PROVENANCE_KEYS = frozenset({"provider_manifest", "provider_channels"})

MODEL = "claude-opus-5"
STYLE_GUIDE = "style guide text\n"
SYSTEM = "system template text\n"
SOURCE_SHA256 = "ab" * 32


# --- the frozen layout block --------------------------------------------------


def _column(
    column_id: str,
    ordinal: int,
    witnesses: tuple[str, ...],
    display_label: str,
) -> dict[str, Any]:
    """One column of the frozen block, spelled per contracts §4.1."""
    return {
        "column_id": column_id,
        "ordinal": ordinal,
        "label_witnesses": list(witnesses),
        "display_label": display_label,
    }


def _frozen_layout_block(
    *columns: dict[str, Any],
    layout_id: str = "layout-7c1f2a",
    revision: int = 1,
) -> dict[str, Any]:
    """The exact frozen JSON §4.3 says `prompt_provenance` records.

    **The one guess in this module, and it is deliberate.** §4.1 fixes the
    field *names* — `layout_id`, `revision`, `columns`, and per column
    `column_id`, `ordinal`, `label_witnesses`, `display_label` — but no
    approved document fixes the JSON container spelling or `revision`'s type,
    because `TableLayout` does not exist in the pre-S5 tree and S4 reserves
    only the `layouts` namespace. Every case below routes through here, so
    reconciling this helper with the real serializer is one edit rather than
    eighteen.
    """
    return {
        "layout_id": layout_id,
        "revision": revision,
        "columns": [dict(column) for column in columns],
    }


def _layout(block: dict[str, Any]) -> extract.TableLayout:
    """A layout object, built only through the native deserializer.

    Never a locally defined stand-in: a fake would prove itself rather than the
    implementation, and the retry planner reads layouts back through exactly
    this deserializer.
    """
    return extract.TableLayout.from_wire(block)


PLAIN_BLOCK = _frozen_layout_block(
    _column("col-9f3a71", 1, ("plain form ",), "Plain"),
    _column("col-2b8d04", 2, ("polite form", "polite （ます）"), "Polite"),
)


# --- project and inputs -------------------------------------------------------


@pytest.fixture
def project(tmp_path: Path) -> ProjectConfig:
    """One throwaway project on the subscription transport.

    The subscription provider rather than the API path on purpose: it is the
    one that *probes* before it plans, so "refused before any provider
    request" is an observable fact about a fake runner rather than a claim.
    """
    seed_prompts(tmp_path)
    (tmp_path / "janki.toml").write_text(
        "[ai]\n"
        'extract_provider = "claude-code"\n'
        f'extract_model = "{MODEL}"\n',
        encoding="utf-8",
    )
    return ProjectConfig.load(tmp_path)


def _prepared(
    config: ProjectConfig,
    name: str,
    *,
    subdir: str = "",
    body: bytes = b"page one",
) -> PreparedInput:
    """One durable corpus input, without a copy step this module is not about."""
    parent = config.scan_inbox / subdir if subdir else config.scan_inbox
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / name
    data = b"%PDF-1.7 " + body
    path.write_bytes(data)
    return PreparedInput(
        kind="document",
        media_type="application/pdf",
        data_b64="ZmFrZQ==",
        origin_path=path,
        source_sha256=hashlib.sha256(data).hexdigest(),
    )


def _provenance(item: PreparedInput, *, mode: str | None, **extra: Any) -> dict[str, Any]:
    """`prompt_provenance` on the API path: the exact historical key set."""
    return extract.prompt_provenance(
        item,
        model=MODEL,
        style_guide=STYLE_GUIDE,
        system=SYSTEM,
        mode=mode,
        source_sha256=item.source_sha256 or SOURCE_SHA256,
        **extra,
    )


def _plan(
    config: ProjectConfig,
    prepared: list[PreparedInput],
    runner: FakeClaudeRunner,
    *,
    mode: str | None,
    **extra: Any,
) -> Any:
    """The real planner, with the suite's real fake subscription transport."""
    return plan_extraction(
        config,
        prepared,
        mode=mode,
        model=MODEL,
        style_guide=STYLE_GUIDE,
        system=SYSTEM,
        provider="claude-code",
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
        **extra,
    )


def _coverage_meta(provenance: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The smallest staging metadata `_validate_prompt_provenance` accepts.

    A v3+ provenance is validated together with the pattern answer written
    beside it, so both halves have to name the same request and the same
    review run. Nothing else in the staged document is relevant here.
    """
    run_id = str(uuid.uuid4())
    meta = {
        "prompt_provenance": provenance,
        "review_run_id": run_id,
        "pattern_set": {"prompt_provenance": provenance, "review_run_id": run_id},
    }
    return meta, {"source_fingerprint": provenance.get("source_sha256")}


def _wire_schema() -> dict[str, Any]:
    return claude_client.wire_schema(extract.candidate_schema())


def _baseline_artifact() -> dict[str, Any]:
    return json.loads((BASELINE_DIR / "extraction-wire-schema.json").read_bytes())


def _conjugations_description(schema: dict[str, Any]) -> str:
    defs = schema.get("$defs", {})
    assert "CandidateRecord" in defs, "wire schema has no CandidateRecord definition"
    properties = defs["CandidateRecord"].get("properties", {})
    assert "conjugations" in properties, "CandidateRecord has no conjugations property"
    return properties["conjugations"]["description"]


# --- the response schema does not move ----------------------------------------


def test_extraction_wire_schema_fingerprint_matches_the_independent_pre_s5_baseline() -> None:
    """§4.3's central promise: the answer contract is untouched by the layout.

    `provider_plan_from_provenance` re-checks this fingerprint for **every**
    historical capture, so a schema edit made for the layout's sake would
    strand every answer already paid for.

    Mutant: change `EXTRACTION_SCHEMA_VERSION`, or add any field to
    `extract.candidate_schema()` for the layout's benefit.
    """
    recorded = json.loads((BASELINE_DIR / "extraction-wire-schema-baseline.json").read_bytes())
    assert recorded["response_schema_fingerprint"] == BASELINE_RESPONSE_SCHEMA_FINGERPRINT

    fingerprint = prompts.schema_fingerprint(_wire_schema())
    assert fingerprint == BASELINE_RESPONSE_SCHEMA_FINGERPRINT, (
        "the extraction wire schema moved; baseline was measured at "
        f"{recorded['base_commit']} with anthropic {recorded['anthropic_version']} "
        f"and pydantic {recorded['pydantic_version']}"
    )
    assert extract.EXTRACTION_SCHEMA_VERSION == EXPECTED_SCHEMA_VERSION


def test_extraction_wire_schema_bytes_match_the_independent_pre_s5_baseline_artifact() -> None:
    """Byte-for-byte, not just hash-for-hash.

    A fingerprint says two things differ; the artifact says *what* differs,
    which is what a reviewer needs when the answer contract is supposed to be
    frozen for the whole milestone.

    Mutant: reorder, rename or retype any property in `candidate_schema()`.
    """
    raw = (BASELINE_DIR / "extraction-wire-schema.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == BASELINE_WIRE_SCHEMA_SHA256

    current = _wire_schema()
    assert json.loads(raw) == json.loads(json.dumps(current))
    # The baseline artifact's own serialization: sorted keys, two-space indent,
    # one trailing newline. Asserted so the checked-in file stays regenerable
    # without a hand-written recipe.
    assert (json.dumps(current, indent=2, sort_keys=True) + "\n").encode("utf-8") == raw


def test_conjugations_field_description_stays_byte_identical_to_the_baseline() -> None:
    """The described trap in §4.3, stated as a check rather than a comment.

    The field's description reads as printed *labels* mapped to forms, which
    is semantically wrong once the keys are opaque ids — and correcting it
    would change `response_schema_fingerprint` and refuse every historical
    capture. The id semantics belong in the labelled user turn and the new
    template, so this string is required to stay exactly as it is.

    Mutant: reword the `conjugations` description at `extract.py:253-259` to
    describe column ids.
    """
    assert _conjugations_description(_wire_schema()) == _conjugations_description(
        _baseline_artifact()
    )


# --- an ordinary extraction is untouched --------------------------------------


def test_ordinary_extraction_provenance_carries_no_layout_key(
    project: ProjectConfig,
) -> None:
    """No layout key appears where no layout was bound — on either transport.

    §4.3 records `table_layout` **only when a layout is given**, which is what
    keeps an already-written capture's provenance bytes, and the manifest
    fingerprints computed over them, unchanged.

    Mutant: emit `table_layout: None` unconditionally in `prompt_provenance`.
    """
    item = _prepared(project, "ordinary.pdf")
    for mode in (None, "table", "prose"):
        provenance = _provenance(item, mode=mode)
        assert "table_layout" not in provenance
        assert set(provenance) == set(ORDINARY_PROVENANCE_KEYS)

    runner = FakeClaudeRunner()
    planned = _plan(project, [item], runner, mode="table").targets[0].provenance
    assert "table_layout" not in planned
    assert set(planned) == set(ORDINARY_PROVENANCE_KEYS | PROVIDER_PROVENANCE_KEYS)


def test_prompt_provenance_validator_refuses_an_added_layout_key_on_an_ordinary_mode(
    project: ProjectConfig,
) -> None:
    """§4.3: ordinary modes retain their exact existing key sets.

    The validator's contract is set *equality*, so the layout work must not
    relax it into a subset test on the way to admitting one new key for one
    new mode.

    Mutant: replace the exact key-set comparison in
    `staging._validate_prompt_provenance` with `expected <= set(provenance)`.
    """
    validate = staging._validate_prompt_provenance
    item = _prepared(project, "ordinary.pdf")
    provenance = _provenance(item, mode="table")

    meta, block = _coverage_meta(provenance)
    validate(meta, block)  # the unchanged shape still passes

    smuggled = dict(provenance)
    smuggled["table_layout"] = PLAIN_BLOCK
    meta, block = _coverage_meta(smuggled)
    with pytest.raises(staging.StagingError) as refusal:
        validate(meta, block)
    assert "exact" in str(refusal.value)

    # And a key the layout deserializer has no opinion about, so only the
    # set-*equality* comparison can be what refuses it.
    extra = dict(provenance)
    extra["table_layout_notes"] = "the owner's own note"
    meta, block = _coverage_meta(extra)
    with pytest.raises(staging.StagingError) as widened:
        validate(meta, block)
    assert "exact" in str(widened.value)


# --- what the request names ---------------------------------------------------


def test_layout_user_turn_names_every_opaque_column_id_in_printed_order(
    project: ProjectConfig,
) -> None:
    """The labelled block is ids, ordinals and witnesses — not a header read.

    §4.1 mints `column_id` opaquely at review and §4.4 forbids any printed
    label from becoming a key, so the ids are what the request must carry, in
    the printed order the ordinals record. The ordinary ask survives beside
    them.

    Mutant: send `display_label` as the key of the requested map instead of
    `column_id`.
    """
    item = _prepared(project, "table.pdf")
    plain = extract.prompt_for(item.origin_path.name, ())
    bound = extract.prompt_for(
        item.origin_path.name, (), layout=_layout(PLAIN_BLOCK)
    )

    assert f"Source file: {item.origin_path.name}" in bound
    assert len(bound) > len(plain)
    for column in PLAIN_BLOCK["columns"]:
        assert column["column_id"] in bound
        for witness in column["label_witnesses"]:
            assert witness.strip() in bound
        assert column["display_label"] in bound
    first, second = (column["column_id"] for column in PLAIN_BLOCK["columns"])
    assert bound.index(first) < bound.index(second)


def test_two_columns_may_share_one_display_label_and_stay_distinct_ids(
    project: ProjectConfig,
) -> None:
    """§4.1: two columns may share a display label because the ids differ.

    The duplicate-heading collapse that made a printed label unusable as a key
    is exactly what opaque ids remove, so a repeated label must be
    representable end to end rather than refused or de-duplicated.

    Mutant: key the requested map, or the frozen block's columns, by
    `display_label`.
    """
    block = _frozen_layout_block(
        _column("col-aa01", 1, ("past",), "Past"),
        _column("col-bb02", 2, ("past, alternate",), "Past"),
        layout_id="layout-shared-label",
    )
    item = _prepared(project, "table.pdf")
    bound = extract.prompt_for(item.origin_path.name, (), layout=_layout(block))
    assert "col-aa01" in bound and "col-bb02" in bound

    provenance = _provenance(item, mode=LAYOUT_MODE, layout=_layout(block))
    frozen = provenance["table_layout"]
    labels = [column["display_label"] for column in frozen["columns"]]
    ids = [column["column_id"] for column in frozen["columns"]]
    assert labels == ["Past", "Past"]
    assert ids == ["col-aa01", "col-bb02"]


def test_layout_from_provenance_round_trips_the_frozen_block_in_printed_order(
    project: ProjectConfig,
) -> None:
    """One canonical spelling, or batch preview breaks later.

    `_require_own_staging` compares the staged `prompt_provenance` with the
    child's planned copy after canonical JSON, so a layout serialized one way
    at plan time and another at staging time is a preview failure a long way
    from its cause. Witnesses are "exact printed strings preserved verbatim" —
    including the trailing space and the non-ASCII character here, neither of
    which is read as language.

    Mutant: trim or NFC-normalize `label_witnesses` inside the serializer.
    """
    item = _prepared(project, "table.pdf")
    provenance = _provenance(item, mode=LAYOUT_MODE, layout=_layout(PLAIN_BLOCK))
    assert provenance["table_layout"] == PLAIN_BLOCK

    again = _provenance(
        item, mode=LAYOUT_MODE, layout=_layout(provenance["table_layout"])
    )
    assert again["table_layout"] == PLAIN_BLOCK
    assert again["request_fingerprint"] == provenance["request_fingerprint"]


# --- mode and layout must pair ------------------------------------------------


def test_mode_and_layout_must_pair_and_ordinary_provenance_yields_no_layout(
    project: ProjectConfig,
) -> None:
    """Both directions of §4.3's pairing rule, plus the no-backfill rule.

    A saved ordinary capture must normalize to *no layout* rather than to an
    invented empty one: the absence is the historical fact, and manufacturing
    a layout for it would let a pre-S5 answer be re-read as a labelled table.

    Mutant: return an empty layout instead of `None` for a provenance with no
    `table_layout`, and drop the "any other mode carrying one refuses" arm.
    """
    deserialize = extract.layout_from_provenance
    item = _prepared(project, "ordinary.pdf")

    for mode in (None, "table", "prose"):
        assert deserialize(_provenance(item, mode=mode)) is None

    bound = _provenance(item, mode=LAYOUT_MODE, layout=_layout(PLAIN_BLOCK))
    missing = {key: value for key, value in bound.items() if key != "table_layout"}
    # A named refusal type, never a bare `Exception`: this module's own
    # missing-seam signal is an `AssertionError`, and a wide `raises` would
    # swallow it and report a green test against absent code.
    with pytest.raises(extract.ExtractError) as refusal:
        deserialize(missing)
    assert LAYOUT_MODE in str(refusal.value)

    mispaired = dict(bound)
    mispaired["mode"] = "table"
    with pytest.raises(extract.ExtractError) as refusal:
        deserialize(mispaired)
    assert "table" in str(refusal.value)


def test_validator_requires_the_layout_for_the_new_mode_and_derives_its_modes_from_extract_modes(
    project: ProjectConfig,
) -> None:
    """§4.3: `table_layout` required **iff** the mode is `table-layout`.

    The mode allowlist has to be derived from `extract.MODES` plus `auto`
    rather than restated as a literal triple, or the validator refuses the
    artifacts the same milestone teaches the planner to write.

    Mutant: keep the literal `{"auto", "table", "prose"}` allowlist in
    `staging._validate_prompt_provenance`.
    """
    validate = staging._validate_prompt_provenance
    item = _prepared(project, "table.pdf")
    assert LAYOUT_MODE in extract.MODES, (
        f"§4.2: extract.MODES must gain {LAYOUT_MODE!r}; found {extract.MODES}"
    )

    bound = _provenance(item, mode=LAYOUT_MODE, layout=_layout(PLAIN_BLOCK))
    meta, block = _coverage_meta(bound)
    validate(meta, block)

    stripped = {key: value for key, value in bound.items() if key != "table_layout"}
    meta, block = _coverage_meta(stripped)
    with pytest.raises(staging.StagingError):
        validate(meta, block)

    for mode in extract.MODES:
        provenance = (
            bound
            if mode == LAYOUT_MODE
            else _provenance(item, mode=mode)
        )
        meta, block = _coverage_meta(provenance)
        validate(meta, block)


# --- what the layout moves, and what it must not ------------------------------


def test_a_bound_layout_moves_the_user_and_request_hashes_and_never_the_response_schema_hash(
    project: ProjectConfig,
) -> None:
    """The milestone's headline acceptance, at the seam that computes both.

    The user turn changes, so `user_prompt_fingerprint` and
    `request_fingerprint` must change — it is a different request. The answer
    contract does not, so `response_schema_fingerprint` must be the
    pre-milestone baseline byte for byte. Style guide, system template and
    source are held equal here, so the layout is the only difference left.

    Mutant: drop the labelled block from `prompt_for` while still recording
    `table_layout` in the provenance.
    """
    item = _prepared(project, "table.pdf")
    plain = _provenance(item, mode="table")
    bound = _provenance(item, mode=LAYOUT_MODE, layout=_layout(PLAIN_BLOCK))

    assert bound["user_prompt_fingerprint"] != plain["user_prompt_fingerprint"]
    assert bound["request_fingerprint"] != plain["request_fingerprint"]
    assert (
        bound["response_schema_fingerprint"]
        == plain["response_schema_fingerprint"]
        == BASELINE_RESPONSE_SCHEMA_FINGERPRINT
    )
    assert bound["response_schema_version"] == plain["response_schema_version"]
    assert set(bound) == set(ORDINARY_PROVENANCE_KEYS) | {"table_layout"}


def test_every_part_of_the_frozen_layout_moves_the_request_fingerprint(
    project: ProjectConfig,
) -> None:
    """A layout is its columns, not its name.

    The owner's display wording and the exact printed witnesses are both part
    of what the model is told, so either one differing is a different request.
    An identity that changed while the columns did not is still a different
    frozen block, which is what a retry restores and what a refusal names.

    Mutant: build the labelled block from `(layout_id, revision)` and the
    column ids alone, dropping witnesses and labels from the turn.
    """
    item = _prepared(project, "table.pdf")
    base = _provenance(item, mode=LAYOUT_MODE, layout=_layout(PLAIN_BLOCK))

    relabelled = _frozen_layout_block(
        _column("col-9f3a71", 1, ("plain form ",), "Dictionary"),
        _column("col-2b8d04", 2, ("polite form", "polite （ます）"), "Polite"),
        revision=2,
    )
    rewitnessed = _frozen_layout_block(
        _column("col-9f3a71", 1, ("plain form",), "Plain"),
        _column("col-2b8d04", 2, ("polite form", "polite （ます）"), "Polite"),
        revision=3,
    )
    for variant in (relabelled, rewitnessed):
        other = _provenance(item, mode=LAYOUT_MODE, layout=_layout(variant))
        assert other["user_prompt_fingerprint"] != base["user_prompt_fingerprint"]
        assert other["request_fingerprint"] != base["request_fingerprint"]

    # Same columns, different revision: the request need not differ, but the
    # frozen block must, or a retry cannot name the revision it restored.
    rerevised = _frozen_layout_block(*PLAIN_BLOCK["columns"], revision=9)
    assert (
        _provenance(item, mode=LAYOUT_MODE, layout=_layout(rerevised))["table_layout"]
        != base["table_layout"]
    )


# --- aligned planning ---------------------------------------------------------


def test_two_inputs_plan_under_their_own_aligned_layouts_in_one_call(
    project: ProjectConfig,
) -> None:
    """§4.2's settlement: one mode per batch, a different layout per child.

    The `layouts` sequence is aligned to the complete `prepared` sequence, so
    each planned target carries its own frozen layout and its own request
    identity, and the batch planner needs nothing more than to forward the
    sequence into this one call.

    Mutant: plan every input under `layouts[0]`.
    """
    first = _prepared(project, "one.pdf", body=b"one")
    second = _prepared(project, "two.pdf", body=b"two")
    other = _frozen_layout_block(
        _column("col-cc03", 1, ("stem",), "Stem"), layout_id="layout-other"
    )
    layout_one = _layout(PLAIN_BLOCK)
    layout_two = _layout(other)

    runner = FakeClaudeRunner()
    plan = _plan(
        project,
        [first, second],
        runner,
        mode=LAYOUT_MODE,
        layouts=(layout_one, layout_two),
    )

    assert [target.item.origin_path.name for target in plan.targets] == [
        "one.pdf",
        "two.pdf",
    ]
    assert plan.targets[0].provenance["table_layout"] == PLAIN_BLOCK
    assert plan.targets[1].provenance["table_layout"] == other
    assert (
        plan.targets[0].provenance["request_fingerprint"]
        != plan.targets[1].provenance["request_fingerprint"]
    )
    for target in plan.targets:
        assert target.provenance["mode"] == LAYOUT_MODE


def test_a_misaligned_layouts_sequence_refuses_before_any_provider_request(
    project: ProjectConfig,
) -> None:
    """Length and mode mismatches are refused before anything is planned.

    §4.3 requires exactly one non-null layout per input for `table-layout` and
    none for any other mode, refused "before planning a provider request" —
    the same shape as the existing `lineage` length check. The fake
    subscription runner records the login probe, so "before" is observable
    rather than asserted.

    Mutant: validate the `layouts` sequence inside the per-input loop, after
    the first provider plan has been built.
    """
    first = _prepared(project, "one.pdf", body=b"one")
    second = _prepared(project, "two.pdf", body=b"two")
    layout = _layout(PLAIN_BLOCK)

    for kwargs in (
        {"mode": LAYOUT_MODE, "layouts": (layout,)},
        {"mode": LAYOUT_MODE, "layouts": (layout, None)},
        {"mode": LAYOUT_MODE, "layouts": ()},
        {"mode": "table", "layouts": (layout, layout)},
    ):
        runner = FakeClaudeRunner()
        with pytest.raises(extract.ExtractError) as refusal:
            _plan(project, [first, second], runner, **kwargs)
        assert "layout" in str(refusal.value).casefold()
        assert runner.calls == []


def test_a_whole_input_staging_collision_refuses_before_any_layout_request(
    project: ProjectConfig,
) -> None:
    """The pre-payment cross-source check survives the per-input layout loop.

    `plan_extraction` makes one `staging_targets` call over the **whole**
    input set; a per-input loop carrying each input's layout would lose it,
    and the second source would silently overwrite the first's candidates
    after both had been paid for. Both layouts are correctly aligned here, so
    the collision is the only thing left to refuse.

    Mutant: move `extract.staging_targets` inside the per-input loop, one
    input at a time.
    """
    first = _prepared(project, "same.pdf", subdir="chapter-1", body=b"one")
    second = _prepared(project, "same.pdf", subdir="chapter-2", body=b"two")
    layouts = (_layout(PLAIN_BLOCK), _layout(PLAIN_BLOCK))

    runner = FakeClaudeRunner()
    with pytest.raises(extract.ExtractError) as refusal:
        _plan(project, [first, second], runner, mode=LAYOUT_MODE, layouts=layouts)
    assert refusal.value.code == "extract-staging-target-collision"
    assert runner.calls == []


# --- the captured request is the authority ------------------------------------


def test_the_captured_provenance_not_todays_choices_defines_the_layout(
    project: ProjectConfig,
) -> None:
    """§4.3: replay reads the frozen layout from the saved request manifest.

    The binding in `choices` is mutable and repointable; the capture is not.
    So the deserializer takes the provenance and nothing else — no job, no
    config, no store — and a saved capture keeps naming the layout it was sent
    with however the owner has since repointed the part.

    Mutant: give `layout_from_provenance` a `job=`/`config=` parameter and
    resolve `part_layout_bindings` when the saved block is absent.
    """
    deserialize = extract.layout_from_provenance
    parameters = list(inspect.signature(deserialize).parameters.values())
    assert len(parameters) == 1, (
        "layout_from_provenance must resolve from the saved provenance alone; "
        f"found {[parameter.name for parameter in parameters]}"
    )

    item = _prepared(project, "table.pdf")
    runner = FakeClaudeRunner()
    saved = _plan(
        project, [item], runner, mode=LAYOUT_MODE, layouts=(_layout(PLAIN_BLOCK),)
    ).targets[0].provenance
    frozen = json.dumps(saved, sort_keys=True, ensure_ascii=False)

    # Today's choices move on: the same part is re-planned under a new layout.
    repointed = _frozen_layout_block(
        _column("col-9f3a71", 1, ("plain form ",), "Dictionary"),
        _column("col-dd04", 2, ("volitional",), "Volitional"),
        revision=2,
    )
    later = _plan(
        project,
        [item],
        FakeClaudeRunner(),
        mode=LAYOUT_MODE,
        layouts=(_layout(repointed),),
        force=True,
    ).targets[0].provenance
    assert later["table_layout"] == repointed

    assert json.dumps(saved, sort_keys=True, ensure_ascii=False) == frozen
    assert saved["table_layout"] == PLAIN_BLOCK
    assert deserialize(saved) is not None
    assert (
        _provenance(item, mode=LAYOUT_MODE, layout=deserialize(saved))["table_layout"]
        == PLAIN_BLOCK
    )


def test_replanning_from_a_saved_provenance_reproduces_its_request_fingerprint(
    project: ProjectConfig,
) -> None:
    """The retry seam, at the planner the retry planner calls.

    `plan_extraction_batch_retry` restores each selected child's **complete**
    frozen layout through `layout_from_provenance(child.provenance)` and
    passes those objects as `layouts`; the identity pair alone is not a layout
    and is never sent to the planner. Re-planning from the saved provenance
    must therefore reproduce the identical request — which is exactly what
    `revalidate_extraction_request` compares before any reservation.

    Mutant: pass `(layout_id, revision)` to the planner and rebuild the
    columns from today's job document.
    """
    item = _prepared(project, "table.pdf")
    saved = _plan(
        project,
        [item],
        FakeClaudeRunner(),
        mode=LAYOUT_MODE,
        layouts=(_layout(PLAIN_BLOCK),),
    ).targets[0].provenance

    restored = extract.layout_from_provenance(saved)
    replanned = _plan(
        project,
        [item],
        FakeClaudeRunner(),
        mode=saved["mode"],
        layouts=(restored,),
        force=True,
    ).targets[0].provenance

    assert replanned["request_fingerprint"] == saved["request_fingerprint"]
    assert replanned["user_prompt_fingerprint"] == saved["user_prompt_fingerprint"]
    assert replanned["table_layout"] == saved["table_layout"] == PLAIN_BLOCK


# --- the one structural check over the answer ---------------------------------


def test_an_unknown_returned_column_id_refuses_naming_the_part_and_layout_identity(
    project: ProjectConfig,
) -> None:
    """§4.4: the parsed key set must be a subset of the request's ids.

    An unbound key would otherwise become a card row with no label binding and
    no witness. The refusal names the offending key, the part and the
    `(layout_id, revision)` because those three are what an owner needs to fix
    it — and the check is over identifiers only: nothing here reads a cell.

    Mutant: turn the subset test into a no-op, or downgrade it to dropping the
    unknown key silently.
    """
    item = _prepared(project, "table.pdf")
    layout = _layout(PLAIN_BLOCK)
    supplied = candidate(conjugations={"col-9f3a71": "x", "col-nope-42": "y"})

    with pytest.raises(extract.ExtractError) as refusal:
        extract.build_records([supplied], item, layout=layout)
    message = str(refusal.value)
    assert "col-nope-42" in message
    assert item.origin_path.name in message
    assert PLAIN_BLOCK["layout_id"] in message
    assert str(PLAIN_BLOCK["revision"]) in message


# =============================================================================
# The whole batch: manifest, dispatch, unsent resume, retry, recovery
# =============================================================================
#
# The cases above stop at `plan_extraction`. These drive the real batch
# planner, the real dispatcher over the suite's own fake subscription
# transport, the real staging writer and the real capture recovery, because
# "the saved manifest carries the layout" is only worth anything if the file
# the batch writes carries the rows.

import shutil  # noqa: E402
from dataclasses import replace as dataclass_replace  # noqa: E402

from test_capture_recovery import _stream as _capture_stream  # noqa: E402
from test_extraction_batch import _stream as _settled_stream  # noqa: E402
from test_workbench_fixtures import RESPONSES  # noqa: E402

from japanese_anki import operations  # noqa: E402
from japanese_anki.application import capture_recovery, extraction_batch  # noqa: E402
from japanese_anki.application.extraction import (  # noqa: E402
    revalidate_extraction_request,
)
from japanese_anki.errors import JankiError  # noqa: E402

OTHER_BLOCK = _frozen_layout_block(
    _column("col-cc03", 1, ("stem",), "Stem"),
    _column("col-dd04", 2, ("volitional",), "Volitional"),
    layout_id="layout-other",
    revision=4,
)


def _batch_project(tmp_path: Path) -> ProjectConfig:
    """A throwaway project the real exporters and preview can build in."""
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
    repo_root = Path(__file__).resolve().parents[1]
    shutil.copytree(
        repo_root / "templates", tmp_path / "templates", dirs_exist_ok=True
    )
    (tmp_path / "decks").mkdir(exist_ok=True)
    (tmp_path / "media").mkdir(exist_ok=True)
    (tmp_path / "vocabulary.json").write_text("[]", encoding="utf-8")
    return ProjectConfig.load(tmp_path)


def _corpus_source(config: ProjectConfig, name: str, body: bytes) -> Path:
    path = config.scan_inbox / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.7 " + body)
    return path


#: One printed blank, one filled cell, one column this row has no cell for.
#: The three states §4.4 distinguishes, in one fixture answer.
SUPPLIED_CELLS: dict[str, str] = {"col-9f3a71": "話す", "col-2b8d04": ""}


def _layout_answer(cells: dict[str, str] | None = None) -> dict[str, Any]:
    """The suite's own table fixture, with layout-keyed conjugations."""
    answer = json.loads(
        (RESPONSES / "table_exhaustive.json").read_text(encoding="utf-8")
    )
    supplied = SUPPLIED_CELLS if cells is None else cells
    for entry in answer["candidates"]:
        entry["conjugations"] = dict(supplied)
    return answer


def _batch_runner(cells: dict[str, str] | None = None) -> FakeClaudeRunner:
    return FakeClaudeRunner(reply=_settled_stream(_layout_answer(cells)))


def _no_api(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the Anthropic API was reached")

    monkeypatch.setattr(claude_client, "prepare_paid_client", refuse)
    monkeypatch.setattr(claude_client, "parse_call", refuse)


def _plan_batch(
    config: ProjectConfig,
    sources: list[Path],
    runner: FakeClaudeRunner,
    **kwargs: Any,
) -> Any:
    return extraction_batch.plan_extraction_batch(
        config,
        sources,
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
        **kwargs,
    )


def test_a_confirmed_layout_batch_freezes_a_different_layout_per_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§4.2: one mode per batch, each child bound to its own layout revision.

    The manifest is the durable record of what one confirmation bought, so the
    frozen layout has to survive its round trip and has to be *absent* from an
    ordinary child — which is what keeps a jobless manifest's bytes, its
    `manifest_sha256` and its consent fingerprint exactly what they were.

    Mutant: serialize `table_layout` unconditionally in
    `ExtractionBatchChild.to_dict`, or plan every child under `layouts[0]`.
    """
    _no_api(monkeypatch)
    config = _batch_project(tmp_path)
    sources = [
        _corpus_source(config, "one.pdf", b"one"),
        _corpus_source(config, "two.pdf", b"two"),
    ]
    runner = _batch_runner()

    plan = _plan_batch(
        config,
        sources,
        runner,
        mode=LAYOUT_MODE,
        layouts=(_layout(PLAIN_BLOCK), _layout(OTHER_BLOCK)),
    )

    wire = json.loads(plan.manifest_bytes)
    assert [child["table_layout"] for child in wire["children"]] == [
        PLAIN_BLOCK,
        OTHER_BLOCK,
    ]
    assert [
        child.expectation.table_layout.to_wire() for child in plan.children
    ] == [PLAIN_BLOCK, OTHER_BLOCK]
    assert (
        plan.children[0].provenance["table_layout"]
        == plan.children[0].expectation.table_layout.to_wire()
    )
    # Read back exactly, including the pairing check both halves enforce.
    assert extraction_batch.ExtractionBatchPlan.from_dict(wire) == plan

    ordinary = _plan_batch(config, sources, runner, force=True)
    ordinary_wire = json.loads(ordinary.manifest_bytes)
    assert all("table_layout" not in child for child in ordinary_wire["children"])
    assert all(
        child.expectation.table_layout is None for child in ordinary.children
    )


def test_a_manifest_whose_two_halves_name_different_layouts_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§4.3: the expectation and the saved provenance name the identical layout.

    Neither is read from a job's current choices, so a hand-edited manifest is
    the only way they can disagree — and janki will not rebuild a request from
    two answers about what was sent.

    Mutant: drop the expectation/provenance comparison in
    `ExtractionBatchChild.from_dict`.
    """
    _no_api(monkeypatch)
    config = _batch_project(tmp_path)
    source = _corpus_source(config, "one.pdf", b"one")
    plan = _plan_batch(
        config,
        [source],
        _batch_runner(),
        mode=LAYOUT_MODE,
        layouts=(_layout(PLAIN_BLOCK),),
    )

    tampered = json.loads(plan.manifest_bytes)
    tampered["children"][0]["table_layout"] = OTHER_BLOCK
    with pytest.raises(extract.ExtractError) as refusal:
        extraction_batch.ExtractionBatchPlan.from_dict(tampered)
    assert "layout" in str(refusal.value).casefold()

    missing = json.loads(plan.manifest_bytes)
    missing["children"][0].pop("table_layout")
    with pytest.raises(extract.ExtractError):
        extraction_batch.ExtractionBatchPlan.from_dict(missing)


def _dispatch(config: ProjectConfig, plan: Any, runner: FakeClaudeRunner) -> Any:
    return extraction_batch.dispatch_extraction_batch(
        config,
        plan,
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
        provider_spawn=runner.spawn,
    )


def test_a_dispatched_layout_batch_stages_source_forms_and_its_own_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole way through: the file the batch writes carries the rows.

    A supplied empty string is a printed blank and an omitted identity is
    absent, the untouched provider map stays beside the card as evidence, the
    exact request metadata that produced it is written beside that, and the
    computed map is left empty because its keys would be opaque identities
    rather than display labels.

    Mutant: drop the `layout=` argument from `complete_extraction`'s
    `build_records` call, so the staged rows carry no table.
    """
    _no_api(monkeypatch)
    config = _batch_project(tmp_path)
    source = _corpus_source(config, "one.pdf", b"one")
    runner = _batch_runner()
    plan = _plan_batch(
        config,
        [source],
        runner,
        mode=LAYOUT_MODE,
        layouts=(_layout(PLAIN_BLOCK),),
    )

    outcome = _dispatch(config, plan, runner)

    assert outcome.committed_count == 1
    staged = config.staging_dir / "one.pdf.yaml"
    records, meta = staging.read_staging(staged)
    table = records[0].source_forms
    assert table is not None
    assert [column.id for column in table.columns] == [
        "col-9f3a71",
        "col-2b8d04",
    ]
    assert [column.label for column in table.columns] == ["Plain", "Polite"]
    # Blank stays blank; the third declared column has no cell here at all.
    assert table.cells == {"col-9f3a71": "話す", "col-2b8d04": ""}
    assert records[0].conjugations == {}
    raw = records[0].source.raw_fields
    assert json.loads(raw["source_conjugations"]) == SUPPLIED_CELLS
    assert json.loads(raw["source_form_layout"]) == PLAIN_BLOCK
    # The staged provenance is the request that was sent, and it validates.
    assert meta["prompt_provenance"]["table_layout"] == PLAIN_BLOCK
    assert meta["prompt_provenance"]["mode"] == LAYOUT_MODE
    assert staging.validate_coverage_facts(meta) is not None
    # `table-layout` is a table mode at every branch: an answer with units is
    # unmeasured and blocking exactly as `table` is.
    assert meta["coverage"]["status"] == "unmeasured"
    assert meta["coverage"]["blocking"] is True


def test_the_batch_preview_reads_its_own_layout_bound_staging_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The staged provenance and the child's planned copy must stay equal.

    `_require_own_staging` compares them after canonical JSON, so a layout
    serialized one way at plan time and another at staging time is a preview
    failure a long way from its cause.

    Mutant: canonicalize the frozen layout differently in `prompt_provenance`
    than in the child's expectation.
    """
    _no_api(monkeypatch)
    config = _batch_project(tmp_path)
    source = _corpus_source(config, "one.pdf", b"one")
    runner = _batch_runner()
    plan = _plan_batch(
        config,
        [source],
        runner,
        mode=LAYOUT_MODE,
        layouts=(_layout(PLAIN_BLOCK),),
    )
    _dispatch(config, plan, runner)

    projection = extraction_batch.render_extraction_batch_preview(config, plan.batch_id)

    assert projection.preview.card_count > 0
    answers = "".join(card.answer_html for card in projection.preview.cards)
    assert "話す" in answers
    assert "Plain" in answers


def test_an_unsent_resume_replans_each_child_under_its_frozen_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§4.3: the expectation's frozen layout, on initial dispatch and resume.

    `revalidate_extraction_request` is the one seam a single call, a batch's
    initial dispatch and its unsent resume all pass through, and its existing
    exact request-fingerprint comparison is what refuses a layout that changed
    or went missing since the confirmation.

    Mutant: stop passing `expected.table_layout` into `plan_corpus_extraction`
    from `revalidate_extraction_request`.
    """
    _no_api(monkeypatch)
    config = _batch_project(tmp_path)
    sources = [
        _corpus_source(config, "one.pdf", b"one"),
        _corpus_source(config, "two.pdf", b"two"),
    ]
    runner = _batch_runner()
    plan = _plan_batch(
        config,
        sources,
        runner,
        mode=LAYOUT_MODE,
        layouts=(_layout(PLAIN_BLOCK), _layout(OTHER_BLOCK)),
    )

    for child in plan.children:
        fresh = revalidate_extraction_request(
            config,
            child.expectation,
            provider_env={},
            provider_runner=runner,
            provider_which=_which,
        )
        assert (
            fresh.target.provenance["request_fingerprint"]
            == child.request_fingerprint
        )
        assert (
            fresh.target.provenance["table_layout"]
            == child.expectation.table_layout.to_wire()
        )

    # A different layout under the same confirmation is a different request.
    moved = dataclass_replace(
        plan.children[0].expectation, table_layout=_layout(OTHER_BLOCK)
    )
    with pytest.raises(JankiError) as refusal:
        revalidate_extraction_request(
            config,
            moved,
            provider_env={},
            provider_runner=runner,
            provider_which=_which,
        )
    assert "changed" in str(refusal.value)


def test_a_retry_restores_the_complete_frozen_layout_from_each_saved_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§4.3: all columns, witnesses and labels come from the saved provenance.

    The `(layout_id, revision)` pair alone is not a layout and is never sent to
    the planner, and no current job lookup resolves a retry's request — this
    batch has no job at all, and it still restores.

    Mutant: pass `(layout_id, revision)` to the planner and rebuild the columns
    from today's job document, or drop `layouts=` from
    `plan_extraction_batch_retry`.
    """
    _no_api(monkeypatch)
    config = _batch_project(tmp_path)
    sources = [
        _corpus_source(config, "one.pdf", b"one"),
        _corpus_source(config, "two.pdf", b"two"),
    ]
    runner = _batch_runner()
    plan = _plan_batch(
        config,
        sources,
        runner,
        mode=LAYOUT_MODE,
        layouts=(_layout(PLAIN_BLOCK), _layout(OTHER_BLOCK)),
        concurrency_limit=1,
    )
    broken = FakeClaudeRunner(reply=b"not a stream at all\n")
    calls = {"count": 0}

    def spawn(command: list[str], **kwargs: Any) -> Any:
        calls["count"] += 1
        if calls["count"] == 1:
            return broken.spawn(command, **kwargs)
        return runner.spawn(command, **kwargs)

    extraction_batch.dispatch_extraction_batch(
        config,
        plan,
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
        provider_spawn=spawn,
    )

    retry = extraction_batch.plan_extraction_batch_retry(
        config,
        plan.batch_id,
        [1],
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
    )

    assert retry.children[0].expectation.mode == LAYOUT_MODE
    assert retry.children[0].expectation.table_layout.to_wire() == PLAIN_BLOCK
    assert retry.children[0].provenance["table_layout"] == PLAIN_BLOCK
    # The request itself is unchanged: retrying is about authority, not content.
    assert (
        retry.children[0].request_fingerprint
        == plan.children[0].request_fingerprint
    )
    assert retry.children[0].operation_id != plan.children[0].operation_id


def test_an_ordinary_jobless_retry_still_restores_no_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other direction of the same seam: absence is not backfilled.

    Mutant: have `plan_extraction_batch_retry` supply an empty layout instead
    of `None` for a child whose saved provenance carries none.
    """
    _no_api(monkeypatch)
    config = _batch_project(tmp_path)
    source = _corpus_source(config, "one.pdf", b"one")
    runner = _batch_runner()
    plan = _plan_batch(config, [source], runner, concurrency_limit=1)
    broken = FakeClaudeRunner(reply=b"not a stream at all\n")
    extraction_batch.dispatch_extraction_batch(
        config,
        plan,
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
        provider_spawn=broken.spawn,
    )

    retry = extraction_batch.plan_extraction_batch_retry(
        config,
        plan.batch_id,
        [1],
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
    )

    assert retry.children[0].expectation.table_layout is None
    assert "table_layout" not in retry.children[0].provenance
    assert (
        retry.children[0].request_fingerprint
        == plan.children[0].request_fingerprint
    )


def test_the_owner_facing_capture_render_draws_the_layouts_source_form_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The recovery render builds records directly, so it needs the layout too.

    `capture_recovery._records_for` calls `normalize_response` and
    `build_records` rather than going through `complete_extraction`, so without
    the captured layout a bound proposal would draw with no source-form rows
    while staging the identical proposal produced them.

    Mutant: drop the `layout=` argument from `capture_recovery._records_for`.
    """
    _no_api(monkeypatch)
    config = _batch_project(tmp_path)
    source = _corpus_source(config, "one.pdf", b"one")
    first = _layout_answer()
    second = _layout_answer({"col-9f3a71": "はなす"})
    runner = FakeClaudeRunner(
        reply=_capture_stream(tool_arguments=[first, second])
    )
    plan = _plan_batch(
        config,
        [source],
        runner,
        mode=LAYOUT_MODE,
        layouts=(_layout(PLAIN_BLOCK),),
    )
    _dispatch(config, plan, runner)
    operation_id = plan.children[0].operation_id
    journal = operations.OperationJournal.load(config.operations_file)
    assert journal.operations[operation_id].state == "result_captured"
    output = tmp_path / "proposals.html"

    capture_recovery.render_capture_proposals(config, operation_id, output)

    inspected = capture_recovery.inspect_capture_proposals(config, operation_id)
    pages = [
        output.parent / f"{output.stem}-{group.proposal_sha256[:12]}.html"
        for group in inspected.groups
    ]
    assert pages
    drawn = "".join(page.read_text(encoding="utf-8") for page in pages)
    # The owner's display labels, from the layout the capture itself saved.
    assert "Plain" in drawn
    assert "Polite" in drawn
    assert not (config.staging_dir / "one.pdf.yaml").exists()


# --- the mode branches inside extract ------------------------------------------


def test_both_table_modes_are_named_explicitly_at_every_mode_branch() -> None:
    """§4.3: `normalize_response` and `coverage_block` gain the new mode.

    Today's guards tested `mode == "table"`, so an unlisted mode string would
    silently skip both — and an empty `table-layout` answer would become a
    non-blocking `selection` coverage block.

    Mutant: restore the literal `mode == "table"` comparison in either
    `extract.normalize_response` or `extract.coverage_block`.
    """
    parsed = _Parsed(
        candidates=(candidate(source_kind="prose", inclusion_reason="worth it"),),
        source_units=(),
        model_reported_unit_count=0,
    )
    with pytest.raises(extract.ExtractError) as refusal:
        extract.normalize_response(parsed, LAYOUT_MODE, "table.pdf")
    assert refusal.value.code == "extract-table-prose-candidate"

    empty = extract.ExtractionResult(
        candidates=(), source_units=(), model_reported_unit_count=0
    )
    for mode in ("table", LAYOUT_MODE):
        block = extract.coverage_block(empty, source_sha256="ab" * 32, mode=mode)
        assert block["status"] == "unmeasured"
        assert block["blocking"] is True
    prose = extract.coverage_block(empty, source_sha256="ab" * 32, mode="prose")
    assert prose["status"] == "selection"
    assert prose["blocking"] is False


class _Parsed:
    """The minimum `normalize_response` reads off a decoded answer."""

    def __init__(
        self,
        *,
        candidates: Any,
        source_units: Any,
        model_reported_unit_count: int,
    ) -> None:
        self.candidates = candidates
        self.source_units = source_units
        self.model_reported_unit_count = model_reported_unit_count
        self.document_kind = "vocabulary"
        self.document_title = ""
        self.patterns: list[Any] = []


def test_the_prompt_name_resolves_the_new_complete_template(
    project: ProjectConfig,
) -> None:
    """§4.2: a complete file, not `extract-table.md` plus a rule block.

    Mutant: map `table-layout` to `extract-table` in `extract.prompt_name`.
    """
    assert extract.prompt_name(LAYOUT_MODE) == "extract-table-layout"
    text = prompts.load(project.root, extract.prompt_name(LAYOUT_MODE))
    assert text != prompts.load(project.root, "extract-table")
    # It asks for the request's own ids and forbids reading a heading.
    flat = " ".join(text.split())
    assert "Printed source-form columns for this page" in flat
    assert "Never use a printed heading, a display label" in flat
    assert "gets no key at all" in flat
    assert "is an empty string under its id" in flat


def test_the_frozen_layout_lands_beside_the_card_as_written_evidence(
    project: ProjectConfig,
) -> None:
    """§4.7: `source_conjugations` untouched, plus the exact request metadata.

    Mutant: sort or re-key `source_form_layout`, or stop writing it.
    """
    item = _prepared(project, "table.pdf")
    supplied = {"col-2b8d04": "  話します  ", "col-9f3a71": ""}
    built = extract.build_records(
        [candidate(conjugations=supplied)], item, layout=_layout(PLAIN_BLOCK)
    )
    raw = built.records[0].source.raw_fields

    assert json.loads(raw["source_conjugations"]) == supplied
    assert json.loads(raw["source_form_layout"]) == PLAIN_BLOCK
    assert built.records[0].conjugations == {}
    table = built.records[0].source_forms
    assert table is not None
    # Printed order, not supplied order, and verbatim cells.
    assert [column.id for column in table.columns] == [
        "col-9f3a71",
        "col-2b8d04",
    ]
    assert table.cells == supplied


# =============================================================================
# Every mode consumer, refusing explicitly rather than widening silently
# =============================================================================
#
# `extract.MODES` is read by the CLI's `--mode` choices, the batch command,
# the workbench's paid form, its offered-mode GET route and the Assistant's
# jobless batch route. Growing that tuple makes the new mode typeable, postable
# and reachable by URL at every one of them, and a request sent under it with
# no bound layout would ask the model to key its answer by identities nobody
# supplied. Each site is checked here, and each names its own mutant.


def test_the_bare_extract_command_refuses_the_layout_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`janki extract --mode table-layout` names files, not a job's parts.

    It refuses before the inputs are preserved and long before the login
    probe, and it names the two commands that do bind a layout.

    Mutant: delete the `refuse_unbound_layout_mode` call from
    `cli.command_extract`.
    """
    from japanese_anki import cli

    root = tmp_path / "project"
    root.mkdir()
    (root / "janki.toml").write_text("", encoding="utf-8")
    page = tmp_path / "page.pdf"
    page.write_bytes(b"%PDF-1.7 page")

    code = cli.main(
        ["--root", str(root), "extract", str(page), "--mode", LAYOUT_MODE]
    )

    assert code == 1
    told = capsys.readouterr()
    assert "janki study layout" in told.err
    assert "Nothing was sent." in told.err
    # It refused before preserving the file into the durable corpus.
    assert not list((root / "data").rglob("page.pdf"))


def test_the_batch_command_refuses_the_layout_mode_before_planning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The batch route names files too, so it has no binding either.

    Mutant: delete the `refuse_unbound_layout_mode` call from
    `cli_extract_batches.run_batch_extraction`.
    """
    from japanese_anki import cli

    root = tmp_path / "project"
    root.mkdir()
    (root / "janki.toml").write_text("", encoding="utf-8")
    pages = []
    for name in ("one.pdf", "two.pdf"):
        page = tmp_path / name
        page.write_bytes(b"%PDF-1.7 " + name.encode("utf-8"))
        pages.append(str(page))

    code = cli.main(
        ["--root", str(root), "extract", *pages, "--mode", LAYOUT_MODE]
    )

    assert code == 1
    assert "janki study layout" in capsys.readouterr().err

    # And at the batch route itself, which is a public seam of its own rather
    # than only whatever `command_extract` happens to check before it.
    import argparse

    from japanese_anki import cli_extract_batches

    with pytest.raises(extract.ExtractError) as refusal:
        cli_extract_batches.run_batch_extraction(
            ProjectConfig.load(root),
            argparse.Namespace(
                mode=LAYOUT_MODE,
                model=None,
                concurrency=2,
                force=False,
                yes=True,
            ),
            [],
        )
    assert refusal.value.code == "extract-layout-unbound"


def test_the_workbench_paid_form_refuses_a_posted_layout_mode() -> None:
    """The desk's paid form carries a source, a model and a fingerprint.

    No owner-bound layout reaches it, so a posted `table-layout` would buy a
    request whose column identities nobody supplied.

    Mutant: delete the explicit `table-layout` branch from
    `workbench.dispatch.parse_dispatch_form`, leaving only the `MODES` check.
    """
    from japanese_anki.workbench import dispatch

    def form(mode: str) -> bytes:
        return (
            "action=extract&csrf=c&dispatch=d&model=claude-opus-5"
            f"&mode={mode}&replacement=0&request_fingerprint={'a' * 64}"
        ).encode()

    ordinary = dispatch.parse_dispatch_form(form("table"))
    assert ordinary.mode == "table"

    with pytest.raises(dispatch.DispatchFormError) as refusal:
        dispatch.parse_dispatch_form(form(LAYOUT_MODE))
    assert "bound" in str(refusal.value)


def test_the_offered_mode_route_never_honours_the_layout_mode() -> None:
    """§10.2: the server's offered-mode route refuses it by name.

    The radios and the query-string reader are two halves of one decision, so
    the reader derives its allowlist from the rendered set rather than from
    `extract.MODES` — which is reachable by URL the moment that tuple grows.

    Mutant: read `_wants_mode`'s allowlist from `extract.MODES` again.
    """
    from japanese_anki.workbench import render, server

    assert LAYOUT_MODE in extract.MODES
    assert LAYOUT_MODE not in render.OFFERED_EXTRACTION_MODES
    assert frozenset({"table", "prose"}) == render.OFFERED_EXTRACTION_MODES

    class _Route:
        def __init__(self, path: str) -> None:
            self.path = path

    wants = server._WorkbenchHandler._wants_mode
    assert wants(_Route("/t/source/x?mode=table")) == "table"
    assert wants(_Route(f"/t/source/x?mode={LAYOUT_MODE}")) is None


def test_the_jobless_assistant_batch_route_refuses_the_layout_mode(
    tmp_path: Path,
) -> None:
    """That route names sources the owner disclosed, not a job's bound parts.

    Mutant: delete the `refuse_unbound_layout_mode` call from
    `workbench.assistant_batch_surface.plan_batch`.
    """
    from japanese_anki.workbench import assistant_batch_surface

    config = _batch_project(tmp_path)
    source = _corpus_source(config, "one.pdf", b"one")

    with pytest.raises(extract.ExtractError) as refusal:
        assistant_batch_surface.plan_batch(
            config, [source], mode=LAYOUT_MODE
        )
    assert refusal.value.code == "extract-layout-unbound"


def test_the_api_transport_sends_the_layout_it_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The explicit Anthropic API path builds its own turn inside the client.

    A layout-bearing provenance beside a turn that never carried the block
    would describe a request that was not sent, so the layout has to reach
    `extract_candidates` as well as `prompt_provenance`.

    Mutant: drop the `layout=` parameter from `extract.extract_candidates`, or
    stop passing `extract.layout_from_provenance(target.provenance)` into it
    from `application.extraction.run_extraction_call`.
    """
    from test_extract import FakeCall, prepared, source_unit, table_ok
    from test_extract import candidate as table_candidate

    from japanese_anki.application import extraction as extraction_module

    item = prepared(tmp_path, "table.pdf")
    layout = _layout(PLAIN_BLOCK)
    supplied = table_candidate(
        source_kind="table",
        section="vocabulary",
        ordinal=1,
        conjugations={"col-9f3a71": "話す"},
    )
    call = FakeCall(table_ok(supplied, units=[source_unit()], count=1))
    monkeypatch.setattr(extract.claude_client, "parse_call", call)

    extract.extract_candidates(
        item,
        model=MODEL,
        style_guide=STYLE_GUIDE,
        system=SYSTEM,
        mode=LAYOUT_MODE,
        client=object(),
        layout=layout,
    )

    sent = "".join(
        block["text"]
        for block in call.calls[0]["content"]
        if isinstance(block, dict) and block.get("type") == "text"
    )
    for column in PLAIN_BLOCK["columns"]:
        assert column["column_id"] in sent
        assert column["display_label"] in sent

    # And the transport hands it the layout its own target recorded, rather
    # than one rebuilt from anywhere else.
    seen: dict[str, Any] = {}

    def capture(*args: Any, **kwargs: Any) -> Any:
        seen.update(kwargs)
        raise RuntimeError("stop after the arguments are bound")

    monkeypatch.setattr(extract, "extract_candidates", capture)
    target = extraction_module.ExtractionTarget(
        item=item,
        staging_path=tmp_path / "staging" / "table.pdf.yaml",
        patterns_path=tmp_path / "patterns.json",
        source_sha256="ab" * 32,
        provenance={"mode": LAYOUT_MODE, "table_layout": PLAIN_BLOCK},
    )
    plan = extraction_module.ExtractionPlan(
        model=MODEL,
        mode=LAYOUT_MODE,
        targets=(target,),
        style_guide=STYLE_GUIDE,
        system=SYSTEM,
        skip_list=(),
        known=frozenset(),
        provider="anthropic-api",
    )
    transport = extraction_module.ExtractionTransport(
        provider="anthropic-api", client=object()
    )
    with pytest.raises(RuntimeError):
        extraction_module.run_extraction_call(
            transport, plan, target, capture=lambda _response: None
        )
    assert seen["layout"] is not None
    assert seen["layout"].to_wire() == PLAIN_BLOCK
