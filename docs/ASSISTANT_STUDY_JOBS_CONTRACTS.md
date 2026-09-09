# Assistant study jobs — detailed contracts

**Status: proposed / not implemented.** These contracts are part of the
[implementation plan](ASSISTANT_STUDY_JOBS_PLAN.md); they do not authorize
implementation or live content calls. DESIGN.md leads. Section numbers are
shared with the plan: sections 1, 10 and 11 live there. Bare source module
paths resolve under `src/japanese_anki/`.

## 2. Durable state and authority

Every path derives from an existing `ProjectConfig` field. **This plan adds no
`[paths]` config key.**

### 2.1 Study job document

`<config.operations_file.parent>/study_jobs/<job_id>.json`, derived beside the
journal exactly as `BATCH_DIR_NAME` is; `data/study_jobs/` by default. `job_id`
is a UUID validated as `_valid_batch_id` validates a batch id, so it can never
name a path outside its store. **Committed.** Writer: new
`application/study_job.py`. One file, one CAS'd revision, **five top-level
namespaces** — the same document §6.2 describes, and **no additional store**.
There is no `events` namespace anywhere in this bundle:

| namespace | contents | rule |
|---|---|---|
| header | `job_id`, `created_at`, `kind`, `parent_source_name`/`parent_sha256`, `deck_path`, `deck_sha256` | immutable at creation |
| choices | directions and audio preferences, part selections, per-part `part_layout_bindings` references, and the owner decisions of §9.1 | mutable, whitelisted keys, CAS-bound |
| layouts | `TableLayout` revisions keyed `(layout_id, revision)` (§4.1) | immutable snapshots, append-only |
| intents | `ActionIntent` and `CurationIntent`, with `supersedes` edges | append-only, immutable once written |
| outcomes | `IntentOutcome` for either intent kind | append-only, immutable once written |

Four writers, all CAS-bound (§6.2). `record_choice` writes whitelisted `choices`
keys **only** and refuses a payload naming any other top-level namespace —
`header`, `layouts`, `intents` or `outcomes` — so an ordinary choice edit can
neither rewrite a layout revision nor remove, rewrite or supersede a pending
intent. `append_layout` is the **only** way a `(layout_id, revision)` comes into
existence: it refuses a key that already exists whose content differs, and
accepts a byte-identical repeat only when the caller explicitly asked for an
idempotent re-append. An existing revision is never rewritten by any writer,
whether or not an intent or a dispatched batch child bound it. Because the
owner's layout Save both appends the revision and repoints the part's binding,
`append_layout` takes the optional binding update and performs **one** CAS write
rather than two independently failing ones; `part_layout_bindings` holds mutable
references to immutable revisions and nothing else. A job document alone grants
no spending, discard or approval authority — the sentence the batch manifest
already lives under.

**No progress fields.** Progress is derived at read time from
`OperationJournal`, `extraction_batch.extraction_batch_status`, the
`data/staging/done/` archives, `ledger.json` and the finish receipt.

```
open_study_job(config, *, kind, parent_source, deck_path) -> StudyJob
record_choice(config, job_id, choice, *, expected_revision) -> StudyJob
append_layout(config, job_id, layout, *, bind=(), allow_identical=False,
              expected_revision) -> StudyJob
append_intent(config, job_id, intent, *, expected_revision) -> StudyJob
append_outcome(config, job_id, outcome, *, expected_revision) -> StudyJob
discover_actions(config, job_id) -> tuple[ActionReference, ...]
study_job_status(config, job_id) -> StudyJobStatus
```

`ActionIntent` carries `intent_id`, `kind` (`source_parts` | `extract_batch` |
`retry` | `curation` | `finish`), the exact ids it reserves — recipe id **plus
that receipt's sha256**; batch id plus manifest sha256 plus every reserved child
operation id; finish receipt id — and the input bindings it was planned over. It
is appended through `append_intent` and fsynced **before** the owning service is
launched, and it is closed only by an appended `IntentOutcome` carrying the
discovered `ActionReference` — never by editing the intent. `append_intent` and
`append_outcome` are the one unified pair for `ActionIntent` and `CurationIntent`
alike. There is no `reserve_action`, `close_action` or `update_choices` in this
bundle, and no alias is kept for them.

### 2.2 Source-part preparation receipts

`<config.operations_file.parent>/source_parts/<recipe_id>.json`, same derivation
rule; `data/source_parts/` by default. **Committed, immutable forever**, never
edited. It is publication authority, unlike `SourcePartLineage`, which is
documentary. It is **job-independent**: a retry, a second job, another deck or a
`janki source-parts` run may reuse the same parts. Reuse is therefore bound by
**recipe id + receipt sha256 + the owner's publish binding** recorded in the
consuming job's intent — never by an embedded job id. If a receipt records the
job that first requested it, that field is documentary and no reader gates reuse
on it.

### 2.3 Study finish authority

`config.staging_dir / "done" / "study" / study-finish-<fingerprint>.json`,
following `kanji_finish._finish_directory` and
`card_revision_finish._finish_directory`. **Committed.** The immutable owner
authority is written before the first effect; per-phase intents and receipts
advance under CAS inside `exclusive_path_lock` (§7.1). Invisible to
`finish._done_archive_batches`, which scans only direct entries of `done/`
carrying staging suffixes. No hand editing, and no deletion while unfinished.

### 2.4 Backlinks that survive a crash

A child artifact must be discoverable when the job's reference write did not
land. Concrete extension: an optional `job_id: str = ""` on
`ExtractionBatchPlan`, on `_execution_receipt_bytes` and on the finish
authority, **serialized only when nonempty** — `ExtractionBatchPlan.to_dict`
already emits `destination_deck`, `retry_of` and `discards` only when set, so a
jobless CLI batch keeps byte-identical manifest wire, `manifest_sha256` and
consent `fingerprint`. That shape stays a current, fully supported input; there
is no legacy reader and nothing is backfilled. The source-part receipt carries
no binding job id (§2.2); every batch and finish artifact *created for* a job
carries its `job_id` and must match it.

Recovery is by **exact id and hash**: `discover_actions` matches a reserved id
from the intent log against the artifact at that exact path, compares the saved
manifest/receipt hash, and for batch and finish artifacts compares the embedded
`job_id`. It never picks "the newest", never orders candidates by mtime or path,
and never adopts a mismatched artifact.

### 2.5 Sole writers, unchanged

`OperationJournal` remains the money authority; batch manifests and
`<batch_id>.execution.json` the execution authority; `staging.py`, the
`data/staging/done/` archives and `promotion.PromotionBatch` receipts the
content authority; `ledger.json` and its `pending_audio` WAL the media
authority; `deck_package` the package authority. The three new stores
coordinate; they reproduce no writer and add no journal.

### 2.6 What a job does not confer

A job's local preferences mint no scope and authorize no deck: the job binds
`deck_path` and that file's sha256 as its **initial** binding and reads the
scope from the saved definition, which is the authority. A job/deck scope
disagreement refuses at `_destination_binding`. A direction preference recorded
in `choices` changes the persisted deck definition and its card set **never**:
if the owner wants different directions, that is an explicit deck-change action
on the existing deck or an explicit new-deck creation, so no card set is
silently minted or migrated. No reviewed shared identity moves into a scope and
no scoped identity becomes shared. The CLI derives a scope by the same contract;
there is no free-form `--scope`.

---

## 3. Source preparation

### 3.1 Rendering dependency and isolation

There is no PDF rasterizer in the repository (`pyproject.toml` declares `genanki`,
`PyYAML`, `ruamel.yaml`, `websockets`; extras add `anthropic`, `pydantic`, `anki`,
ChatKit/Agents, `playwright`; `inputs.py` shells to `sips` for HEIC only). A new
optional extra `sources = ["pypdfium2", "Pillow"]` is required; Pillow owns PNG
composition and encoding, because a hand-written `zlib`/`struct` encoder is code
this project would own forever. Exact constraints are resolved and locked in the
implementing milestone and recorded in every recipe. The ordinary build/import
path must not import either package, matching `card_preview.preview_unavailable()`.
`dev` must take `japanese-anki[sources]` so the rendering and publication tests
run instead of skipping (§10).

The [pypdfium2 API documentation](https://pypdfium2.readthedocs.io/en/stable/python_api.html)
states that PDFium is not thread-safe, even across different documents. Rendering
therefore runs in **one bounded worker subprocess** from an explicit `spawn`
context, one document at a time, closing every page and document handle
explicitly. It never runs inside the batch's thread pool.

### 3.2 The recipe is owner-produced, and only the owner authors geometry

```
SourcePartRecipe(recipe_id, parent_name, parent_sha256, render_dpi,
                 parts=(PartSpec(page_index, page_rotate, regions=((x0,y0,x1,y1), …)), …),
                 renderer, renderer_version, encoder, encoder_version)
```

- Coordinates are normalized fractions in `[0,1]`, origin at the top-left of the
  rendered page **after `/Rotate` is applied**, so a recipe is stable under DPI
  change and unambiguous under rotation. Each part records `page_index` (0-based),
  the page's `/Rotate` as read from the document, `render_dpi`, `page_size_pt` and
  the resulting pixel rect. A `/Rotate` the recipe did not record refuses.
- **Two part forms only:** a whole rendered page, or an explicitly reviewed region
  list composed top to bottom (header-plus-body). No third form, no detector, no
  segmentation.
- Geometry reaches `plan_source_parts` from exactly two producers: the inline
  region editor's serialized output, or `--recipe FILE` read by the CLI. Both mint
  an opaque `recipe_token` bound to the recipe bytes; the typed broker rejects a
  `plan_source_parts` argument set carrying inline geometry. The model may emit
  `open_source_part_editor` (a UI intent that writes nothing) but may not author,
  widen or alter a rectangle — it never saw the source, because intake places no
  bytes in model context. This is a **local action binding on where geometry comes
  from**, not an additional confirmation: preparation still asks the owner for
  nothing beyond the request and the choice they already made.
- **What geometry proves:** which pixels of which page were rasterized at which
  DPI and stacked in which order, and that the bytes hash to the recorded value.
  **What it does not prove:** that a rectangle is a table, a row or a column, that
  a header belongs to a body crop, or what a column means. Those are owner
  statements or model reports, never janki's conclusions.

### 3.3 Plan, receipt, publication

```
plan_source_parts(config, source_name, recipe, *, recipe_token) -> SourcePartsPlan
execute_source_parts(config, plan, *, publish_token) -> SourcePartsReceipt
```

`SourcePartsPlan` carries `recipe_id`, `parent_name`, `parent_sha256`,
`renderer`/`encoder` identity and resolved versions, `plan_fingerprint`, and per
part `ordinal`, `target_name`, `sha256`, `byte_length`, `page_index`, `regions`
and a contact-sheet thumbnail. Target names are
`<parent-stem>--p<page>-r<recipe8>-<nn>.png`, derived from the recipe fingerprint
so a resumed publication produces identical names. Neither call takes a job id —
the service is usable from the CLI and the source catalogue before study jobs
exist, and a job later references `recipe_id` (§2.1).

`execute_source_parts` writes the immutable receipt
(`<config.operations_file.parent>/source_parts/<recipe_id>.json`) **first**, then
publishes each absent part into `config.scan_inbox`. `publish_token` is the
owner's explicit publish action bound to the exact previewed `plan_fingerprint`;
a plan alone publishes nothing. Publication is ordinary local work: no paid call,
no provider disclosure, no edit of an original, and **no new approval gate**.

Publication uses a new helper `inputs.publish_derived_part(name, data, scan_inbox,
inbox_root)`. It must not go through `inputs._copy_into_inbox`, which for bytes
from outside the durable root takes the content-stamped rename branch
(`inputs.py:356-360`) and would store the part under a name the receipt never
bound. Under `exclusive_path_lock(durable_root)` the helper compares the exact
planned name and bytes through the existing `_durable_namesakes` /
`_durable_name_collision` (`inputs.py:175-221`): same name and same bytes is a
reuse; same name and different bytes refuses the whole recipe; a free name is
written atomically. Originals are never touched, nothing is renamed, and an
interrupted publication resumes from the receipt and republishes only absent
bytes. A re-render whose bytes differ from the receipt refuses and overwrites
nothing; different parts need a new recipe id and a new receipt. **Byte
provenance is a check the prompt cannot be asked for** — no template can make a
file hash to a value, and DESIGN puts file provenance in janki's own logic.

### 3.4 Then, and only then, the paid batch

`plan_extraction_batch` requires every source to be a durable corpus file and
enforces it through `inputs.prepare_corpus_input(source, config.scan_inbox)`,
whose bound-bytes check refuses any path outside that root; children bind the
prepared item's `origin_path`, and resume re-prepares `child.source` the same
way. So the fingerprints a confirmation must enumerate exist only after
publication: publish, then plan, then confirm once. Existing finite-batch
semantics are unchanged — one atomic reservation, concurrency 2 by default and 4
at most (`MAX_CONCURRENCY`), one journal claim per child, per-child capture and
staging. Each child carries the part's own path and hash plus a
`SourcePartLineage` whose `descriptor` cites `recipe_id` (documentary; the
receipt is the executable authority). PNG is a supported corpus suffix, and
identical-byte parts do not trip the duplicate-request refusal because the user
turn carries the filename. Rendering happens only when the owner asks; nothing is
prepared speculatively.

---

## 4. Table layout and card fields

### 4.1 One owner-bound layout per layout-bound part

`TableLayout` lives in the job document's append-only `layouts` namespace (§2.1),
immutable per `(layout_id, revision)` and created only by `append_layout`, which
refuses an existing key whose content differs:

```
TableLayout(layout_id, revision, columns=(TableColumn(column_id, ordinal,
            label_witnesses=(exact printed strings, …), display_label), …))
```

`column_id` is minted locally and opaquely at review; `ordinal` is printed order;
`label_witnesses` are exact strings preserved verbatim; `display_label` is the one
owner-chosen display string, and two columns may share it because the IDs are
distinct. A layout is reusable across parts and sources — that is the point of a
stable `layout_id`. Binding is per part and lives in `choices` as a **mutable
reference only**: `part_layout_bindings[(recipe_id, target_name)] = (layout_id,
revision)`, with **exactly one** binding per layout-bound part. Repointing a
binding edits neither revision, and the owner's Save appends the new revision and
repoints the binding in one CAS write (§2.1). Zero bindings makes an ordinary unlabelled part; two refuse at
plan time naming both layouts. No code reads a header, infers a column's meaning
or matches a label; the owner does that language work in the editor.

### 4.2 A complete new template, and one mode per confirmed batch

`prompts/extract-table-layout.md` is a **complete** file, not `extract-table.md`
plus a rule block, satisfying the AGENTS rule that a branching prompt is two
prompts. `extract.MODES` becomes `("table", "prose", "table-layout")`, so
`prompt_name` resolves it and the existing validators inherit it: the CLI mode
choices, `workbench/dispatch.py`, `workbench/server.py`, `promote.py`. The study
job **pins the mode per part** and never leaves it `None`; a layout-bound part
planned under `auto` refuses rather than silently sending `extract-auto`.

**A batch's mode is one scalar, and this plan does not pretend otherwise.**
`plan_extraction_batch` takes a single `mode`, loads one system template from it,
and passes one `mode` into `plan_extraction` for every prepared input; the retry
planner re-plans the whole retry under `selected[0].expectation.mode`. Mixed
per-part modes therefore **do not** pass the current API unchanged, and no
per-child mode extension is proposed here. Settlement: **one confirmed batch carries one mode.** Its aligned
`layouts` sequence may carry a different frozen layout for each child under
that mode. Different tables therefore fit one `table-layout` batch; ordinary
parts needing another extraction mode use a separate confirmed batch. A layout
difference alone does not add a confirmation.

The **shared decoder-prevention paragraph** — the tool argument is the schema
object itself, every REQUIRED field present, no unsupported extra property, no
wrapper object and no JSON-string payload — is added verbatim to all four
complete extraction templates (`extract-auto.md`, `extract-table.md`,
`extract-prose.md`, `extract-table-layout.md`). No Python instruction branch, and
no per-template variant of the wording.

### 4.3 Labelled request metadata: the exact adapter changes

The layout is a labelled data turn, which AGENTS already keeps in Python, so no
prompt prose branches. `EXTRACTION_SCHEMA_VERSION = 5`, `candidate_schema()` and
`conjugations: dict[str, str]` are **unchanged**: the model returns the supplied
IDs as keys and each row's supplied cell as the value. The
`response_schema_fingerprint` is therefore identical, so
`provider_plan_from_provenance`'s contract check
(`application/extraction.py:647-658`) still passes for captures made before this
work. The user turn does change, so `user_prompt_fingerprint` and
`request_fingerprint` differ for a layout-bound request — which is correct, since
it is a different request.

Nothing here is inherited by magic. The complete list of signature and behaviour
changes:

- `extract.prompt_for(source_name, known=(), *, layout=None)` appends one labelled
  block: per column, its `column_id`, ordinal, exact printed witnesses and display
  label.
- `extract.prompt_provenance(…, layout=None)` passes the same layout into that
  user turn and records `table_layout` — the exact frozen JSON of the block —
  **only when a layout is given**, so no existing capture's provenance bytes
  change and an old capture still normalizes to no layout.
- `extract.layout_from_provenance(provenance)`, a new pure deserializer. A
  `table-layout` provenance with no `table_layout` refuses; any other mode
  carrying one refuses.
- `extract.normalize_response(parsed, mode, source_name)` gains `table-layout` in
  its mode guards explicitly. Today those guards test `mode == "table"` and
  `mode == "prose"`, so an unlisted mode string would silently skip both.
- `staging._validate_prompt_provenance` extends its exact-key contract with
  `table_layout` **required if and only if** the mode is `table-layout`; it
  validates that value through `extract.layout_from_provenance`. Ordinary modes
  retain their exact existing key sets. Its mode allowlist derives from
  `extract.MODES` plus `auto`, replacing the literal auto/table/prose triple.
  The response-schema version and fingerprint remain unchanged; saved ordinary
  artifacts are read without backfilling a layout key.
- `extract.coverage_block` treats `table-layout` exactly as `table` at every
  mode branch, including `has_table`; its prose-only coverage path remains
  prose-only. An answer with no source units or candidates is still
  `unmeasured` and blocking for either table mode. The block's schema is
  unchanged, and the ordinary CLI promotion path enforces the same coverage
  decision as the job finish.
- `extract.build_records(candidates, prepared, known_ids=(), *, layout=None)` and
  `extract._raw_fields(candidate, prepared, *, layout=None)`.
- `application/extraction.extraction_provider_plan(item, …, layout=None)`, which
  builds the planned user turn that becomes the saved manifest and channels.
- `application/extraction.plan_extraction(config, prepared, *, mode, layouts=(),
  …)` receives layouts aligned to the complete `prepared` sequence. For
  `table-layout`, exactly one non-null layout per input is required; other modes
  accept no layout. Length and mode mismatches refuse before planning a provider
  request. Its one existing `extract.staging_targets` call still receives the
  **whole** input set, preserving the pre-payment cross-source staging-path
  collision check. The existing per-input loop passes that input's layout to
  both `extraction_provider_plan` and `extract.prompt_provenance`; the batch
  planner forwards the aligned sequence in its one call to `plan_extraction`.
  A single-source caller supplies a one-element sequence. Retry restores the
  full saved layout for each selected child through this same seam.
- `ExtractionDispatchExpectation` gains a frozen `table_layout` snapshot,
  populated from `extract.layout_from_provenance(target.provenance)` when the
  expectation is planned. Its serialized field is required only for
  `table-layout` and omitted for ordinary modes, preserving their existing
  wire and fingerprints. Deserialization enforces the same mode/layout pairing;
  a batch child's expectation and saved provenance must name the identical
  frozen layout. Neither is read from the current job choices.
- `application/extraction.plan_corpus_extraction(..., layout=None)` carries
  that single source's layout into the aligned `plan_extraction` sequence.
  `revalidate_extraction_request` passes the **expectation's frozen layout**
  when it calls that planner. Thus `_revalidate_children` reproduces the
  confirmed request on initial dispatch, unsent resume and selected retry,
  and `_run_children` sends the freshly revalidated target with that same
  layout. The existing exact request-fingerprint comparison is unchanged;
  omitting or changing a layout refuses before reservation or dispatch.
- `application/extraction.complete_extraction` passes
  `extract.layout_from_provenance(target.provenance)` into `build_records`.
  **Replay reads the frozen layout from the saved request manifest, never from the
  current mutable job document.** Both recovery paths inherit that for free
  because each builds its target from the captured provenance, exactly as they
  already read the saved `mode` back rather than today's.
- `plan_extraction_batch(config, sources, *, mode, layouts=(), …)` — `layouts` is
  a sequence aligned to `sources`, refused for a length mismatch exactly as
  `lineage` already is. Under the fixed mode every entry must be present for `table-layout`
  and may differ between children; every entry is absent for other modes.
- `plan_extraction_batch_retry` restores each selected child's **complete frozen
  layout** through `extract.layout_from_provenance(child.provenance)` — the
  identical layout its expectation froze — and passes those objects as `layouts`.
  It restores `mode` from the saved expectation as it already does. The
  `(layout_id, revision)` pair alone is not a layout and is never sent to the
  planner; all columns, witnesses and labels come from the saved provenance,
  including for a jobless batch. No current job lookup resolves a retry layout.
  The existing binding checks still refuse a changed request fingerprint.

### 4.4 Structural checks the prompt cannot be asked for

The parsed map's key set must be a **subset** of the request's `column_id` set.

- An **unknown ID** refuses that part, naming the exact offending key, the part
  name and the `(layout_id, revision)`. An unbound key would otherwise become a
  card row with no label binding and no witness.
- A **duplicate ID** cannot survive `dict[str, str]`; opaque IDs remove the
  duplicate-heading collapse that made that reachable, so no printed label is ever
  the key.
- A **schema type error** — a non-string key or value — refuses at the
  serialization boundary, where it already does.
- An **omitted known ID is absent.** A **present empty string is blank.** That
  distinction is carried, not inferred.

The template asks for a complete, faithful transcription of every printed cell;
the code validates artifact identifiers and nothing else. **No rule inspects
Japanese, a printed header, or a cell's text to decide whether a column was
absent** — the only evidence used is whether the model supplied the ID. A response
that supplies no keys at all is a valid **all-absent table**: the declared columns
are still the layout's, and the table is preserved with no cells. Nothing guesses
which column was meant, and no fuzzy or nearest-label matcher appears anywhere.

### 4.5 The optional canonical `source_forms` field

`models.py` gains:

```
@dataclass(frozen=True, slots=True)
class SourceFormColumn:  id: str; label: str
@dataclass(frozen=True, slots=True)
class SourceFormsTable:  columns: tuple[SourceFormColumn, ...]; cells: dict[str, str]
```

and `VocabularyRecord.source_forms: SourceFormsTable | None = None`.

- `from_dict` accepts `{"columns": [{"id","label"}], "cells": {id: str}}`, refuses
  a cell id absent from `columns`, refuses a duplicate column id, and preserves
  every supplied cell **verbatim** — no trim, no blank-dropping. It returns `None`
  for an absent or null key, and for the one deliberately defined **empty table**:
  no columns *and* no cells. A table that declares columns is never empty, so
  provided columns are never silently discarded.
- `to_dict` **omits** `source_forms` under exactly that same rule — `None`, or no
  columns and no cells — as it already omits an unset `spoken_japanese`. One
  canonical spelling on both sides is not a nicety: `repairs._require_canonical_record`
  (`repairs.py:477-488`) requires `to_dict → from_dict → to_dict` to be equal, and
  either a trimmed cell or an asymmetric empty spelling breaks it. No existing
  record's serialized bytes change.
- `to_dict` emits **plain JSON containers** for the field — a list of column
  mappings, not the dataclass tuple `asdict` would hand through — the same kind of
  post-processing it already does for `examples`. Both YAML writers and the JSON
  collection writer then see the shapes they already write.
- `validation._control_characters` walks `record.to_dict()` generically, including
  mapping keys and list items, so column ids, labels and cells are covered with no
  new validator.
- `io.merge_records` picks the field up because `_CONTENT_FIELDS` is derived from
  the dataclass; `_copy_value` is `copy.deepcopy`, which detaches a frozen nested
  dataclass; and `is_empty` is true only for `None` and empty containers, so a
  present table is a value rather than a hole. That is why the empty spelling must
  not exist in memory as a present object: it would look non-empty to the merge.
  Two differing non-empty tables are reported as a conflict with the existing
  value kept, and `source` remains the first sighting's — so *which* part supplied
  a table is read from that part's `data/staging/done/` archive.
- `application/card_change_staging._validate_replacement_shape` gains a
  `source_forms` case requiring the complete canonical JSON shape; the field is
  outside `enrich.ENRICHABLE_FIELDS` and `enrich.AI_FIELDS`, so no dictionary or
  AI pass writes it, and it is **not added to the automatic-repair allow-list** —
  no repair may target it.

### 4.6 What the card shows

`exporters/anki.py` renders the **same** `Conjugations` field: `_conjugation_html`
gains an ordered-rows form and the note builder selects `record.source_forms` when
present and `record.conjugations` otherwise. A blank cell renders its declared row
with an empty value; an absent column renders no row. `FIELD_NAMES`, `model_id`
derivation, GUIDs (`genanki.guid_for(record.id)`), card directions and the three
writing passes are unchanged, and no historical content is backfilled or
re-fingerprinted. `card_preview` builds through the real exporters, so the owner's
HTML preview shows the real rendered rows rather than a summary. A record whose
printed table is present and whose `conjugations` later fills from the dictionary
engine keeps both; only the printed table renders, and the finish projection must
say so plainly, since the field it shows has to be the field the exporter selects.

### 4.7 What is preserved beside the card

`extract._raw_fields` keeps writing `source_conjugations` — now the exact ID-keyed
provider map, untouched — and gains `source_form_layout`, the JSON of the exact
request metadata that produced it (`layout_id`, `revision`, and per column its
ordinal, witnesses and display label). `build_records` constructs `source_forms`
from those two facts in the layout's ordinal order and leaves `conjugations` empty
for a layout-bound candidate. `candidate_accounting` and pattern metadata keep their existing structure;
the coverage block's schema is unchanged, but its mode behavior and prompt
provenance validator are explicitly extended in §4.3. No accounting, layout or
provenance is backfilled onto a historical artifact. Ordinary unlabelled extraction is a distinct current input shape that
stays fully supported — not a compatibility shim.

---

## 5. Captured-answer recovery

### 5.1 Inspection is read-only and discloses no source bytes

`inspect_capture_proposals(config, operation_id) -> CaptureRecoveryPlan` in a new
`application/capture_recovery.py`. Pure and offline: no journal change, no
staging, no paid call, no login probe, no prompt read. It reads through
`OperationJournal.read_reply`, which already refuses an entry under `cleanup`, an
interrupted capture and a recorded-but-unreadable artifact, then unwraps with the
shared `extraction_capture_parts` so a capture with no saved manifest is refused
rather than reinterpreted.

It accepts `result_captured` only; `committed` inspects read-only for diagnostics.
`authorized`/`dispatching`/`running` refuse and name the owner's existing
`janki operations --end`. `outcome_unknown` refuses: its evidence is never resent,
and strengthening it to `result_captured` remains the existing owner service over
frames already on disk.

Per proposal it returns `frame_index`, `block_index`, `tool_use_id`, an RFC 6901
`json_pointer`, `proposal_sha256`, the enumerated envelope shape that matched and
the schema verdict; per capture it returns `request_fingerprint`,
`capture_sha256`, `response_schema_fingerprint` and the current `source_sha256` /
`staging_path` / `patterns_path` / `destination_deck_sha256` bindings. **No reply
bytes, no candidate content and no source bytes enter model context** — hashes,
pointers and verdicts only, so an inspection cannot become an unmanifested
disclosure.

### 5.2 The closed list of lossless envelope shapes

Today's decoder reads `structured_output` on the single `subtype: "success"`,
`is_error: false` result frame (`application/revision_provider.py:1046-1087`).
The salvage reader adds, and admits, only:

1. that same result-frame `structured_output`;
2. a `tool_use` block for the structured-output tool whose `input` validates
   against `extract.candidate_schema()` directly;
3. shape 2 where `input` is an object whose **sole** key is `input` and whose
   value validates — any other key set refuses;
4. shape 2 where `input` is a JSON **string** that parses totally and exactly to a
   validating object — a partial or trailing-garbage parse refuses.

Nothing else, and no parser exception is added per observed bad output. **No
unknown-field dropping, no stray-token deletion, no missing-fact filling.** An
extra property, or a missing `source_kind`, is a refusal naming the exact pointer.
The template fix in §4.2 is the first and preferred remedy and now actually ships
on every template the study job can send.

### 5.3 Who selects, and what the owner can see before selecting

- **Exactly one valid proposal:** `stage_capture_proposal(config, operation_id,
  selection)` runs under the **original** batch confirmation's staging authority —
  this is the same operation's own answer reaching its ordinary durable
  destination — and needs no new gate.
- **More than one valid proposal:** staging requires an explicit local owner
  selection, bound one-use to the exact tuple `(operation_id, capture_sha256,
  response_schema_fingerprint, frame_index, block_index, tool_use_id,
  json_pointer, proposal_sha256)`. A content hash alone is not a location: two
  structurally identical proposals in one capture hash the same, and provenance
  that cannot say which block it came from is not provenance. The model may not
  supply or narrow any part of that tuple, and the ungated single-proposal action
  refuses when the capture holds more than one. On the CLI the same decision is
  the owner's typed `--proposal SHA256 [--at POINTER]`; the two surfaces demand
  the same thing. This is a local selection capability, **not a new paid consent**
  and not a second confirmation ladder — nothing is billed and no new operation id
  is minted.
- **One content, several locations:** proposals are grouped by content hash, and
  each group lists **every** matching location in deterministic order (frame,
  block, pointer). Selecting the group records all of them; selecting one location
  requires the full tuple. A group is never collapsed to an arbitrary
  representative, and an ambiguous provenance is never written.
- A capture holding a **valid successful terminal is authoritative**: recovery
  uses the ordinary `recover_extraction_from_capture` and refuses the salvage path
  even when an earlier tool argument differs. The reader never picks the latest or
  the largest.
- **The owner does not choose between opaque hashes.** A separate local action
  `render_capture_proposals(config, operation_id, output_path)` writes an
  immutable HTML page of each proposal's cards through the **real exporters** —
  the same way the batch preview draws saved proposals — beside a raw JSON
  diagnostic of the exact parsed object. It writes no staging, makes no paid call
  and grants no approval. Its bytes are owner-facing only: like any private
  unfinished paid reply, they never enter ordinary model context, so the model
  still sees only hashes, pointers and verdicts.

### 5.4 Staging the recovered answer

Staging revalidates every binding the existing `_recover_child`
already checks — operations journal and pattern store identity, destination deck,
corpus re-preparation and `source_sha256`, and the replacement revision —
normalizes through the pure `extract.normalize_response` and completes through the
existing `application/extraction.complete_extraction`. `result_captured →
committed` is a legal advance; no new writing path is created and no paid
redispatch hides inside recovery.

Provenance rides the **existing extraction serializer seam**: `complete_extraction`
composes the staging `meta`, and a `capture_recovery` block — frame index,
pointer, `tool_use_id`, envelope shape, every location in the selected group,
`proposal_sha256`, `capture_sha256`, and who selected it — is added there, so it
is written by the sole staging writer, survives partial promotion, and reaches
`data/staging/done/` unchanged. S2 adds `capture_recovery` to `staging.META_KEYS`
as a recognized extraction metadata field. The captured artifact stays write-once: the
original terminal error and raw bytes are preserved exactly, and nothing edits
them.

### 5.5 A missing field is a diagnostic and a fresh retry

An owner may not hand-supply a missing schema field, and the reason is not the
automatic-repair allow-list — that list governs janki's own repairs of records,
not what a person may type. The reason is that the captured reply is immutable
evidence of what was paid for: a staged record built partly from a hand-typed
field would no longer describe the answer the journal says was billed, and
`complete_extraction`'s identity check exists to keep that impossible.
Settlement: refuse, emit the exact diagnostic, and offer one newly authorized
single-child retry through `plan_extraction_batch_retry` with a fresh operation
id; an unchanged request fingerprint is valid for a fresh authorization.

### 5.6 Refusals that preserve evidence

Unknown or changed `response_schema_fingerprint`, invalid UTF-8 in a frame, an
incomplete final frame, a missing capture, a capture with no manifest, or any
changed binding — each preserves the request and the reply exactly and refuses.
No provider success is ever forged. Leaving the response schema unchanged removes
exactly one refusal, the response-contract mismatch; it does **not** make every
historical capture recoverable, because the source, staging, patterns and
destination bindings must still hold and a malformed answer is still malformed.

---

## 6. Cross-source curation

### 6.1 Current staged values decide

Proposals are read from each part's staging document's **editable `records` list**
— the current edited state — joined for display to its immutable
`candidate_accounting`, which is never edited, never backfilled, and never the
source of a choice. Cross-part collision groups are computed **at read time** from
the per-part documents, as `render_extraction_batch_preview` already does for a
batch under `_require_own_staging`. No part's document is rewritten to create a
combined accounting record and no synthetic multi-source staging document is
fabricated. Where two parts propose one identity differently the review discloses
the conflict; nothing merges first-wins. A choice among competing Japanese values
is the owner's, or comes from the model's existing extraction pass — never from
new local logic.

### 6.2 One job document: mutable choices, append-only intents

The job document has one immutable header and five top-level namespaces in one
CAS-bound file — exactly the list §2.1 gives, with no `table_layouts` key inside
`choices` and no `events` namespace:

```
header     job_id, created_at, kind, parent_source_name/sha256, deck_path/deck_sha256
choices    directions, audio selections, part_layout_bindings, owner decisions (§9.1)
layouts    TableLayout revisions keyed (layout_id, revision)  append-only, immutable
intents    (ActionIntent | CurationIntent, …)                 append-only, immutable once written
outcomes   (IntentOutcome, …)                                 append-only, immutable once written
```

```
PreparedFileUpdate(staging_path, sha256_before, sha256_after, content_after,
                   record_ids, operations)
CurationIntent(intent_id, job_id, decided_at, decision, supersedes,
               prepared=(PreparedFileUpdate, … sorted by path))
IntentOutcome(intent_id, state, at, observed=((path, sha256), …), consequences)
```

`content_after` is the **complete prepared file text**, not a digest: a digest
cannot be replayed after a crash, and a resume that had to recompute the payload
from a store that has since moved is not a resume.

Four writers in `application/study_job.py`, all CAS-bound to the file's prior
revision through `io.atomic_write_text_bound`: `record_choice` accepts only a
whitelist of `choices` keys and **refuses a payload that names any other
top-level namespace** — `header`, `layouts`, `intents` or `outcomes`;
`append_layout` is the sole creator of a `(layout_id, revision)` and refuses an
existing key whose content differs, so neither a choice edit nor a re-Save can
rewrite a revision a dispatched child bound; `append_intent` and `append_outcome`
append only — one unified pair for `ActionIntent` and `CurationIntent` alike —
refusing any edit, reorder or duplicate id of an existing entry. An ordinary
job-choice edit therefore cannot overwrite or clear an intent or a layout, and
neither is a second log file — each is a namespace of the one committed job
record. Capture replay is unaffected either way: it reads the frozen layout from
the saved request manifest, never from this document (§4.3).

**No mutable field lives inside an immutable intent.** There is no
`superseded_by`: the successor carries `supersedes: intent_id`, and completion or
abandonment is an appended `IntentOutcome`. An owner who changes their mind gets a
**replan** — a fresh intent computed against fresh snapshots — and the original
intent and its evidence are never deleted or rewritten. No paid call is involved.

**Abandonment is an explicit bound local action**, not a rollback:
`abandon_intent(config, job_id, intent_id, *, expected_intent_sha256)` re-reads
every bound path under the locks of §6.3, records the current mixed snapshot —
which paths are at `sha256_after`, which at `sha256_before`, which at neither —
and its consequences, and appends the `abandoned` outcome. Nothing is reverted, no
file is restored, no evidence is deleted. The successor plans fresh `sha256_before`
values from those observed states; a partially applied intent that was never
finished or abandoned still blocks its successor.

The chosen value lands in **every occurrence's own staged record** for that
identity, taken from the current editable records — every source occurrence is
preserved, and identity dedup happens only in projection and in card, audio and
package work — so whichever part promotes last writes the same bytes and cannot
overwrite the choice.

### 6.3 All locks, then all prechecks, then any write

1. Acquire the **staging mutation coordination lock** (§6.5) first, before any
   other lock.
2. Compute every `PreparedFileUpdate` from a single read of each file, and append
   the intent durably.
3. Acquire **every** affected path's `exclusive_path_lock` in sorted path order —
   sorted so two concurrent curations cannot deadlock.
4. **While holding all of them, precheck every bound file**: each must hash to its
   `sha256_before`, or already to its `sha256_after` (a resumed entry). One
   mismatch refuses the whole decision, naming the path and both digests, before
   anything is written. Per-file CAS alone is insufficient: it permits a partial
   application whose remaining files then refuse, leaving a decision that recovery
   cannot finish.
5. Write each file under CAS on `sha256_before`, in the same sorted order.
6. **Restart proof:** re-running the recorded intent skips entries already at
   `sha256_after`, applies entries still at `sha256_before`, and refuses an entry
   at neither. The intent is complete when every entry is at `sha256_after`, and
   only then is the `applied` outcome appended.

### 6.4 The writer stays the writer, and can delete a key

The coordinator computes and supplies exact expected data; it reproduces no writer
logic. `staging.py` gains a narrow prepare/apply pair over machinery it already
has — `render_staging_update` already renders the round-trip edit without writing,
and `atomic_write_text_bound` already performs the bound replacement:

```
staging.FieldOperation(row_index, record_id, field, action, key=None, value=None)
    action ∈ {"replace", "remove"}
staging.prepare_record_update(path, records, *, operations=())
    -> PreparedStagingUpdate(text, sha256_before, sha256_after, applied)
staging.apply_prepared_update(path, prepared)   # caller's lock, CAS on sha256_before
```

Both live in `staging.py`; `application/study_curation.py` calls them and holds
the locks. This is the same seam `assistant_assignment.py` and
`workbench/server.py` already use, promoted into the sole writer module so an
application service does not depend on a workbench helper.

**Deleting a cell or a table is an ordinary local curation edit, not an extraction
defect, and it must never cost a paid retry.** `_apply_changes`
(`staging.py:1817-1842`) writes only changed keys and never removes one, which is
exactly right for its own job — annotating a reviewed row without adding twenty
empty schema fields — and it keeps that job unchanged. Structural deletion lives
only in the new explicit path: a `FieldOperation` names one row by index **and**
by `record_id`, and a mismatch refuses. `remove` on `source_forms` deletes that
key from the row's mapping in the round-trip document; `remove` with a `key`
deletes that one `cells` entry. This is the same class of targeted document edit
`record_coverage_approval` already performs, not a new framework. Every prepared
text is then re-read with `read_staging_text` and must equal the expected records
with **byte-identical** metadata before it can be applied — the self-check the
surgical example-authority writer already uses — so comments, key order, quoting,
unknown keys, coverage and `candidate_accounting` are provably untouched.

Emptying a cell is `replace` with `""` — the printed-blank spelling. Removing the
column entirely is `remove`. Row add and remove remain the workbench's dedicated
staging editor over `write_staging`; the positional annotator is not repurposed
into an implicit row edit.

### 6.5 A shared barrier, race-safe at the commit

An intent with no terminal outcome **is** the barrier — there is no separate
marker file, so the barrier cannot outlive or precede its own evidence, and it is
cleared only by a durable appended `applied` or `abandoned` outcome. Nothing
clears it automatically and no timeout exists.

`promotion.decide_promotion` gains one gate, returning `blocked(...,
gate="curation-pending")` naming the intent id and the remaining paths, placed
with the other structural gates ahead of coverage. Because that gate is inside
`decide_promotion`, it applies identically to `janki promote`, the workbench
promotion, the Assistant's typed promotion action and this job's own finish. A job
document referenced by a barrier that is missing or malformed **refuses without
touching any file**; it is never skipped, because a skipped barrier is a lifted
one.

**A planning gate cannot close the race, and `expected_wire` cannot either.** An
intent can be published while every bound staging file still holds its
`sha256_before` bytes: the wire CAS proves only that the *file* did not change, so
a promotion that decided before publication would still pass it and consume a file
a durable decision is about to rewrite. The barrier therefore needs a shared
mutation lock:

- The **staging mutation coordination lock** is `exclusive_path_lock(config.staging_dir
  / ".janki-curation-lock")` — a lock name, not a file janki writes, modelled on
  the existing `.janki-audio-operation` operation lock.
- Intent publication and curation acquire it **before** all staging path locks
  (§6.3 step 1).
- Every mutation entry that can call promotion acquires that same guard
  **before any existing janki lock**, then preserves its existing internal lock
  order. The execution rechecks the active barrier while holding the guard
  through its writes; planning alone is insufficient.
- Apply this narrow change to `cli.py`, `workbench/server.py`,
  `application/assistant_promotion.py` and its already-locked callers in
  `application/card_revision_finish.py` and `workbench/assistant_adapter.py`,
  as well as the new `study_finish`. Factor a guarded/locked entry so an inner
  promotion call never tries to acquire the guard after its caller acquired a
  staging, canonical, audio-operation or finish-record lock.
- The finish-record CAS lock is released before acquiring this outer guard.
  Checkpoints may take their short CAS lock under the guard, consistently.
  Test an existing revision finish interleaved with cross-source curation,
  as well as the new study finish. No code path may invert this ordering.

### 6.6 What curation never does

Different identities never silently merge or migrate: no reviewed shared identity
moves into a scope and no scoped identity becomes shared. Per-part review and
coverage remain separate, one decision per part. Every staged copy of one identity
must agree after an intent completes, and a completed intent that left two copies
disagreeing is a refusal naming both paths. A decision that was not durably
recorded before its first write does not exist — and cannot, because the record
precedes the writes.

## 7. Review, promotion and finish

Parts reach this section already prepared, extracted, curated, assigned to a durable
deck, and previewed. **Apply and finish** is one immutable authority over that whole
set, one phase chain, and one prepared payload per owning writer. Nothing here mints a
deck, a scope, an owner, or a coverage verdict; nothing here re-reads Japanese. Every
structural check below exists because an artifact and its provenance — which bytes
landed, under which authority, proven by which receipt — cannot be owned by a Japanese
prompt. None of it is model-quality validation.

### 7.1 Phase chain and the per-writer prepare/persist/apply/recover contract

`authorized → reviewed → promoted → enriched → audio_complete → packaged → complete`,
advanced by `_advance`-shaped CAS writes inside `exclusive_path_lock` on the finish
record, following `kanji_finish._advance` (`application/kanji_finish.py:386-418`). The
immutable authority is written before the first effect (`_write_new:374-383`).

Every phase **P** runs the same four steps. What an intent may *contain* is the
writer's fact, not a universal rule.

1. **Prepare** — side-effect-free **on canonical and on every published target**. It
   reads freely, replays the fact book, and may render into a private scratch path; it
   publishes nothing another reader treats as content. It yields `P_intent`.
2. **Persist** — one CAS write adds `P_intent` to the finish record. `_replace_record`
   refuses any write that changes an already-present `*_intent`; an ordinary
   job-choice edit cannot reach the finish record at all.
3. **Apply** — the writer that owns those files consumes the intent through the narrow
   entrypoints named in §7.5. `study_finish` coordinates and copies no writer.
4. **Receipt** — one CAS `_advance` to the next state carrying `P_receipt`.

Three intent shapes, because three kinds of write exist here:

- **Known before the write** — review, coverage, promotion, enrichment, reference
  facts. The intent carries, for **each file and each sub-write**, either the complete
  after-payload or an exact replayable delta, plus that component's own
  `expected_before` and `expected_after`. This is `character_notes`' existing shape
  (`ProposedFile:88-118`, `execute_character_notes:918-968`).
- **Unknown result, journaled** — paid audio. Clip bytes cannot exist before the call.
  Paid operation ids are reserved **one per clip at the existing `before_paid_dispatch`
  seam** (`application/audio.py:1046`), immediately before that clip is sent — not all
  at owner-plan time, where none of them is knowable. The journal and
  `ledger.pending_audio` capture every unknown outcome (§7.10).
- **Artifact first** — the package. An APKG SHA cannot be predicted, so the order is
  private build → actual SHA → intent → publish (§7.11).

**"Fixed now" covers only nondiscretionary local metadata**: a timestamp, a date, an
event id, the selected archive name, the content-addressed receipt identity. **Every date-bearing value in a prepared payload is frozen by its owning writer's
prepare**, so a resume on a later day replays those exact bytes instead of recomputing a
date or a hash. Coverage's `approved_at`, which `coverage._approval_payload:402` computes
with `date.today()`, and `ledger.record_export(at=)` (`ledger.py:1011-1017`) are not the
only seams that need a parameter. The promotion and enrichment ledger writes reach
`_iso_date(at=None)` (`ledger.py:199`) today and take the same treatment:
`ledger.record_added(at=)` (`:805`, date at `:792`), `ledger.record_source_seen(seen_at=)`
(`:817-825`, date at `:846`) and `ledger.record_enriched(at=)` (`:856-866`, date at
`:924`). Every one of those parameters already exists and is simply unthreaded; §7.7 and
§7.8 thread them from the prepared intent, so a prepared ledger component's
`expected_after` is the same digest on the day of the crash and the day after it.
Nothing a provider, a builder, or a later read decides is ever fixed in advance.

A crash between 2 and 4 recovers **from `P_intent` only**. A value that was returned but
never persisted is not evidence, and neither is a writer's state label. Recovery
re-resolves each bound component's fresh state, matches it against that component's own
before/after, and only then adopts a fresh CAS token; a component at neither digest
refuses and names both digests and the component.

### 7.2 One plan-time projection for the whole batch

The finish performs per-part staging review and owner coverage acceptance, so both must
be modelled before the owner clicks. Two facts make the current planner insufficient:
`decide_promotion` reads canonical from disk (`application/promotion.py:2424`) and its
only injection is the *staging* snapshot `_record_snapshot` (`:2190-2195`), and a table
part with `unmeasured` coverage returns `state="blocked"`, `gate="coverage"`
(`:2554-2555`) before any merge exists.

**New pure seam in `application/promotion.py`**, beside
`project_ai_enrichment_review_promotion:1881` and
`project_card_revision_review_promotion:1977`:

```
project_source_extraction_review_promotion(
    config, staging_path, *, expected_revision, review_record_ids,
    review_patterns, pattern_store, coverage_approval, collection,
    collection_revision, witness,
) -> PromotionDecision
```

- `expected_revision` refuses a staging file that moved (`:1996` shape).
- `review_record_ids` are exactly the rows `ReviewPanel.submit` would flag; the
  projection applies the writer's own `set_example_flags(record, EXAMPLE_AUTHORITY_KEY,
  …)` (`workbench/review.py:643-649`) to those rows, so `example_accepted` and
  `io._merge_one`'s drop of unaccepted examples (`io.py:4329-4348`) see post-review
  state.
- `review_patterns` contributes this part's owner-selected mark to §7.6's
  single prepared pattern-store payload. Prepare that payload from the captured
  store for **all parts before the promotion fold**: apply
  `patterns.render_reviewed_update` once per selected source entry whose mark
  is false, leave already-true marks unchanged, and never interpret a false
  choice as unreviewing an entry. `pattern_store` is the resulting mapping from
  `patterns.load_store_text(prepared_after_text)`; if no mark changes, it is the
  captured live snapshot. Every part sees this same aggregate after-store, so
  one part's choice cannot erase another's selected mark. This is a projection
  only; the store is first written by the confirmed `reviewed` phase.
- `coverage_approval` is `None` when the part's coverage already resolves; otherwise the
  exact prepared owner payload with its frozen `approved_at`, inserted at
  `meta["coverage"]["approval"]` so `promote.check_coverage` (`:2332`) passes in the
  projection exactly as after the writer runs.
- `collection` / `collection_revision` are the injected canonical snapshot.
- `witness` is the frozen keyed fact book (§7.3).

**Named narrow changes**, all in `application/promotion.py` unless stated:

1. `decide_promotion(..., _collection_snapshot: tuple[Sequence[VocabularyRecord],
   RecordsRevision] | None = None)`, replacing the `load_records_snapshot` read at
   `:2424` when supplied.
2. `PromotionDecision.projected: bool = False`, true whenever an injection was used.
   `execute_promotion` refuses a projected decision at function entry, ahead of
   the `nothing`, `pattern_only` and `archive_retry` branches as well as the
   normal landing path. The existing preview refusal at `:2659-2664` is too
   late: earlier branches can already archive and remove staging files. **A chained canonical projection can never
   be replayed as a transaction**, and this refusal is preserved verbatim below for
   enrichment.
3. The `client` annotation widens to the `DictionaryLookup` protocol (§7.3) so a replay
   book reaches `promote.check_readings` at `:2434` unchanged.
4. `decide_promotion` gains `_pattern_store_snapshot:
   Mapping[str, patterns.PatternSet] | None = None`, alongside its other
   injections. `check_pattern_review(..., pattern_store=None)` consults that
   supplied mapping instead of `patterns.load_store(config.patterns_file)`;
   ordinary callers keep the live read. The source-review projection always
   supplies its frozen `pattern_store`. Source, extraction-run and prompt
   provenance checks remain unchanged. Any pattern-store injection also sets
   `projected=True`, preserving the executor's refusal at entry. After the
   prepared review payload lands, normal execution sees the identical store
   through its ordinary read. A selected valid pattern-only review therefore
   projects `pattern_only`; an unresolved unreviewed set still projects the
   existing blocker.
5. New `io.records_json_text(records) -> str`, extracted from
   `save_records_json_locked:4125-4143` and reused by `deck_package._canonical_records_text`
   and by the fold below, so the chain's synthetic revisions are byte-identical to a save.

**The fold.** Parts run in one fixed order (published part name, ascending). Part 1
projects against `(C0, records_revision(canonical))`. Part *k* projects against
`(carry_{k-1}, RecordsRevision(canonical, text_{k-1}))`, where `carry_0 = C0` and
`text_0` is that revision's text. **The carry advances only for a part whose projected
state is `lands`** (`:2557`): then `carry_k := decision_k.merged` and
`text_k := io.records_json_text(carry_k)`. For every other state the canonical state is
carried unchanged — `carry_k := carry_{k-1}` and `text_k := text_{k-1}`.
`PromotionDecision.merged` defaults to the empty tuple (`:1738`) and is populated only at
`:2507`, after every non-landing return has already left (`nothing_lands:2476-2482`,
`archive_retry:2367`, `pattern_only:2388`, `nothing:2372`), so an empty `merged` is
**never** read as the projected collection and the empty-collection digest never enters
the chain. The authority binds, per part, its projected state, `expected_before =
sha256(text_{k-1})` and `expected_after = sha256(text_k)` — necessarily equal for a
non-landing part, because `execute_promotion` writes no canonical for one
(`:2726-2760`) — plus its landed / held / excluded id sets and its archive-retry ids.
`carry_N` is the single post-promotion canonical projection every later phase reads. A
projected `blocked` state is not a disposition: it stays a blocker under §7.7 and §9.6
unless an explicit exact owner disposition excludes or defers that whole part.

**Merge disposition is disclosed, never overridden.** `io.merge_records` fills holes; an
existing nonempty canonical value wins, and `source` stays the first sighting
(`io.py:4440-4441`). Where a part proposes a different nonempty value the review shows
the merged projection and the conflict; changing an existing value requires the separate
authorized `revise` path.

**`source_forms` keeps exactly the interface §4 defines** — optional
`columns: [{id, label}]` and `cells: {id: string}`, with empty and absent preserved as
distinct states. It is an ordinary hole-fillable `_CONTENT_FIELDS` member here
(`io._merge_one:4326`), it is not in `enrich.ENRICHABLE_FIELDS`, no phase rewrites a
nonempty one, and it survives promotion, enrichment, package projection and the `done/`
archive untouched. It affects no notetype and no GUID: `FIELD_NAMES` and
`genanki.guid_for(record.id)` are unchanged, and unaffected notetypes and GUIDs stay
byte-identical.

**Held rows are never promised.** A held row stays in its live staging file with its
reason; the authority records it as held, and `landed` never includes it.

### 7.3 The keyed dictionary fact book

Two free jpdb reads decide visible content: the promote reading witness
(`promote._dictionary_verdict:732-740` → `enrich.dictionary_readings:350`) and dictionary
enrichment. `jpdb.JpdbClient` exposes `ping:427`, `list_user_decks:431`,
`list_deck_vocabulary:439`, `lookup_vocabulary:481` and `parse:519`; **only the last two
are reachable on this path** — `enrich._parse:279` calls `parse`, `enrich._readings_for:293,302`
calls `lookup_vocabulary`, and nothing else on the promote or enrich route touches the
client. New in `enrich.py`:

- **`DictionaryFactBook`** — a **keyed request → response mapping**, not a consumptive
  sequence. The key is `(method, canonical argument wire)`; the value is the response
  wire. A bound query answers **any number of times, in any order**, which is required
  because the same expression is parsed once by the witness and again by enrichment, and
  the fold visits parts in an order the preview does not repeat. Two *conflicting*
  answers recorded for one key refuse **at prepare**, while the recording client still
  holds both. An unrecorded query refuses at replay.
- `RecordingDictionaryClient(client)` — records while answering.
- `ReplayDictionaryClient(book)` — answers only from the book; an unknown key raises
  `EnrichError`.
- `DictionaryLookup` — a `Protocol` with `parse` and `lookup_vocabulary` only, the
  annotation `decide_promotion`, `promote.check_readings` and §7.8's helper widen to.

Facts are fetched **once**, before the preview, through the recording client. The
projection, the preview, the authority and every phase apply use the replay client.
**No phase refetches after review**, so no dictionary refresh can refuse or change what
the owner approved. The book is persisted in the authority; its fingerprint is compared
at each apply.

### 7.4 The immutable authority

`StudyFinishAuthority` binds: `job_id`; the deck file's path and bytes and its saved
scope; per part `(staging_path, sha256)` **after assignment and before the finish's own
review write**, its `review_record_ids`, its prepared review and coverage payloads, and
its landed/held/excluded ids; the one aggregate prepared pattern-store
before/after payload; the ordered chained canonical digests
`sha256(text_0) … sha256(text_N)`; the approved preview `sha256` and the
approved-content fingerprint (every visible field except `audio` and
`examples[*].audio`); the dictionary fact-book fingerprint; the bound enrichment field
values with `projected_input_sha256` / `projected_output_sha256`
(`card_revision_finish.py:1016-1017` pattern); the prepared reference-fact write payloads
(§7.9); the exact prospective clip enumeration computed **after** enrichment (§7.10);
the resolved provider/model/billing profile and request template for every paid call;
the whole-deck package **projection** wire and `output_path` (§7.11); and the
configuration fingerprint. That projection is planned with
`plan_vocabulary_deck_package_revision` over the final projected **post-audio**
records and `RecordsRevision(canonical, io.records_json_text(records_after_audio))`,
plus §7.9's prepared reference after-hashes and §7.10's media enumeration. Those
records come from the audio-reference projection over the post-enrichment
collection (§7.10). `text_N` is only the post-promotion checkpoint; using it here
would bind stale canonical bytes that enrichment and audio are authorized to change. It is therefore **exact for the whole deck** — deck input, canonical,
both reference stores, templates, card set, counts, record ids and output path — and
carries `None` for exactly those enumerated audio media slots whose WAV bytes cannot
exist before the paid call. It is a projection, not the concrete build plan: the package
`preparation_id`, the staged artifact SHA, the realized media hashes and the inventory
are **not** here, because they are minted in §7.11 after the artifact exists. Reviewed visible bytes may change
afterwards only along these planned media paths; anything else refuses.

### 7.5 Writer-owned entrypoints, and lock ordering

Each phase's prepare/apply/recover lives in the module that owns the files, factored out
of the writer that already writes them. The coordinator holds no copy of any of them.

| phase | prepared intent | writer-owned prepare / apply / recover | recovery proof |
|---|---|---|---|
| `reviewed` | §7.6 per-file payloads | `workbench/review.py` `ReviewPanel.prepare` (factored from `submit:604-723`); `assistant_staging_review.{prepare,apply_prepared,recover_prepared}_staging_review` around `execute_staging_review:349`; pure `staging.render_coverage_approval` composed with review rendering into the one prepared after-payload applied by the factored review writer; standalone `approve_coverage_as_owner` is not called by this phase | §7.6 |
| `promoted` | per part in fold order: landed/held/excluded ids, `expected_before/after`, frozen archive name and receipt identity | `assistant_promotion.execute_promotion_action:510` → `promotion.execute_promotion:2574`; new `promotion.recover_promotion_intent` | §7.7 |
| `enriched` | bound field values, fact-book fingerprint, canonical before/after, prepared ledger attribution, plus §7.9's reference payloads | `enrichment.{prepare,apply_prepared,recover_prepared}_dictionary_enrichment` around `commit_dictionary_enrichment:219`; `character_notes.{prepare,apply_prepared,recover_prepared}_reference_facts` | §7.8, §7.9 |
| `audio_complete` | clip enumeration + per-clip reservations minted at dispatch | `audio.execute_targeted_audio_locked:1033` with `before_paid_dispatch=` | §7.10 |
| `packaged` | `DeckPackagePreparation` | `deck_package.{prepare,publish_prepared,recover_prepared}_deck_package` | §7.11 |
| `complete` | preview intent | `card_preview` + `assistant_packages` resolver registry | §7.11 |

Every apply prechecks its **entire** bound set under locks before **any** write, persists
its intent before the first mutation, and matches before/after **per component** —
including the intermediate review→coverage state within `reviewed`, and the
canonical→ledger→archive→prune sequence within `promoted`.

**Lock ordering.** §6's shared curation coordination lock is acquired **first** by
everything in this section that touches a staged file or canonical. Publication of a
curation intent and every curation update take it ahead of their sorted per-file staging
locks; the `promoted` phase's **execution** takes it before
`promotion.execute_promotion`'s live-staging → archive → deck-dir → canonical → ledger
sequence (`_finish_record_review:1440-1459`, `commit_canonical_state:2821-2849`,
`:2861-2875`), checks the pending-curation barrier under it, and **holds it through
every write**. One global lock always ahead of the per-path locks is what stops a
curation holding file A and waiting for B from deadlocking a promotion holding B and
waiting for A. A plan-time `blocked` / `gate` verdict alone is not sufficient: it is read
when a page renders and acted on when a button is clicked, the shape two callers can
both pass.

### 7.6 `reviewed`

`ReviewPanel.submit` writes two files through two independent bound replaces with a
deliberate crash window between them (`workbench/review.py:698-716`), and reports
`PartialReviewError` when it can. "Monotonic" is therefore a safety property, not a
recovery proof: a crash before the coordinator receives anything leaves an intent and no
receipt.

The exact after-text of both files is already computable without writing —
`staging.render_example_authority_updates:1893` and `patterns.render_reviewed_update:469`
are pure renderers over captured bytes. `ReviewPanel.prepare` returns them as
`(path, before_sha256, after_text | None)` per file, plus the expected marker and
`fingerprint(meta, reviewed)`. `submit` keeps its behaviour by calling `prepare` and then
the shared apply, so nothing about the interactive panel changes.

Review and coverage target the same staging file. Their pure preparations
are composed in memory into **one final after-payload for that path**, with the
owner's exact coverage payload and frozen approval date. The pattern store has
its own single final after-payload. The writer preparation API accepts the
coverage payload and composes two **pure** renderers over the same captured bytes:
the existing `staging.render_example_authority_updates:1893`, and a new
`staging.render_coverage_approval(captured_text, approval, *, replace_existing=False)
-> str` factored out of `staging._record_coverage_approval_unlocked:2166-2210`, which
today reads its own path through `_load_document_snapshot` and writes, and so cannot be
used as a renderer. The writing wrapper keeps its exact behaviour and refusals by
calling that renderer and then its existing bound write;
`staging.record_coverage_approval:2214` and
`record_coverage_approval_under_lock:2230` are otherwise unchanged. The preparation does
not publish an intermediate review-only staging document.

Apply and recovery precheck one before/after pair per distinct path under the
locks before writing anything. A path at its final after-state is complete; a
path at its initial before-state is pending; a third state refuses. Thus a
crash between the staging and pattern writes is recoverable without demanding
that one file simultaneously match two different intermediate revisions.
Ordinary review and coverage actions keep their existing supported entrypoints
and use the same factored writers.

### 7.7 `promoted`: per part, artifact-proven

The promotion service gains `prepare_source_promotion` and
`apply_prepared_source_promotion`, with `recover_promotion_intent`.
They factor the current executor's canonical, ledger, archive and live-staging
publication code; the coordinator does not reproduce it. Preparation freezes
the selected archive name, receipt ID, metadata and each distinct path's full
after-payload (or an exact replayable delta/deletion), its before binding and
after hash. It also freezes **every date the ledger component carries**: the factored
writers take and thread `record_added(at=)`, `record_source_seen(seen_at=)` and
`record_enriched(at=)` from the intent rather than letting `_iso_date(at=None)` read the
clock at apply time (§7.1), so the ledger component's `expected_after` is stable and the
pinned canonical→ledger split recovers on a later day instead of refusing at a third
state. Persist this per-part intent before the first mutation. Recovery
prechecks the complete component vector, then finishes only missing writes.
An archive retry is bound to that prepared archive and receipt, not a guessed
filename or a receipt reconstructed from a lost return value.


For an unstarted part, execution re-decides from disk under the curation and
promotion locks with the replay witness; its state and held/excluded/exact-retry
ID sets must equal the projected bindings. For `lands` and `nothing_lands`,
`decision.output_revision` must hash to this part's `expected_before` (the
preceding checkpoint). Only `lands` compares
`io.records_json_text(decision.merged)` to `expected_after`. The other three
states have no `output_revision`: read canonical under the held locks and
require its actual bytes to match `expected_before`; never dereference a missing
snapshot or hash the default empty `merged`. Every non-landing state keeps
`expected_after == expected_before`.

The prepared writer intent binds the actual effects of each state separately:

| state | prepared effects and recovery evidence |
|---|---|
| `nothing` | No writer effect; retain the existing live file and its metadata. Canonical remains unchanged. |
| `pattern_only` | Freeze the pattern-only archive name, metadata and complete after-payload, then archive and remove the live file through `_complete_pattern_only_review`. Preserve source and candidate accounting in that archive; do not invent a vocabulary promotion receipt. |
| `archive_retry` | Bind the existing archive, its exact retry records and available promotion receipt; `_finish_record_review` completes the archive and prunes/removes the live file. Previously archived accepted IDs are proved by their existing receipt, never reported as newly landed. |
| `nothing_lands` | Write the held reasons and retain held rows. Exact retry rows, if any, are archived/pruned by `_finish_record_review`, with their existing receipt retained; without exact retries there is no archive result or promotion receipt to require. |

All file effects above use the same prepared before/after component vector,
including archive publication and live-file deletion. A partially applied intent
recovers from that vector before any fresh decision could misclassify its own
writes as a new state. The canonical fold never advances for these states.

Completion of part *k* is proven from artifacts. **Live-file presence is irrelevant** in
both directions: a missing file is not completion, and a surviving one is not failure —
a part that legitimately keeps its live file because rows were held recovers by the same
rule. A part with new landings completes only when all of these hold: canonical hashes to
`expected_after`; `data/staging/done/<name>` contains a `promotion_batches` entry whose
`promoted_ids` equal the bound landed ids and whose receipt id is the content-addressed
`_promotion_receipt_id:306-331` value the intent froze; the source-bound archive
identity matches; and every required ledger and recovery bookkeeping write named in the
intent is present. A returned state label proves none of this.

`execute_promotion` may return `landed_ledger_incomplete` or `landed_ai_ledger_incomplete`
(`:1758-1766`), and it may die before the coordinator receives either. Both are recorded
when seen, and recovery reaches the same verdict from the intent alone. **A part in
either state cannot advance the job to `complete`**: the phase finishes only when the
owning writer's bookkeeping actually finishes, which is the same `promote` rerun the
data-lifecycle rules already prescribe. The finish reports the exact outstanding part.

A part with no new landings uses its state-specific evidence above, not the
new-landing receipt rule. Where exact earlier landings exist, keep their bound
archive/receipt association in the aggregate scope and prove those accepted IDs
from it; do not relabel them as newly landed or require another promotion.
A wholly held or excluded part with no earlier accepted landings has no
vocabulary promotion receipt. It stays outstanding until an explicit owner
exclusion or deferral covers it under §9. Such a disposition retains the source
and accounting, plus the live file or archive the writer actually leaves; it
never fabricates a receipt or requires a live file after the writer removed it.

Completed earlier parts are a proven prefix, not snapshots the current
canonical file must still equal. On resume, validate their retained receipt
and source/owner associations, then compare current canonical only with the
latest completed prefix or the active next intent's before/after state.
Later authorized enrichment/audio uses the same forward frontier rule.
External states outside that recorded evolution refuse. This avoids comparing
the final collection against every obsolete intermediate digest.

### 7.8 `enriched`

`enrich.enrich_records` is not a seam this section can borrow as-is. It reaches jpdb
through the client on every target it does not skip (`enrich.py:679, 716`); its scope
refusal is written against the canonical file — "in the normalized file. Enrichment reads
that file only … or a staged row (finish its reading review first)" (`:627-634`); and the
decision of *which* records and *which* stores it ever sees belongs to
`enrichment._plan`, which performs the canonical read at `application/enrichment.py:163`
and supplies `kanji_store` from `config.kanji_file` at `:180`. Injecting a projection
into that pair today means lying in a diagnostic or lying to a reader.

**Named refactor, in `enrich.py`.** Factor the record-and-facts decision into a helper
over explicit records and their revision:

```
enrich.decide_enrichment(client, records, revision, *, ids, force_fields, kanji_store)
    -> EnrichResult
```

`revision` is the `RecordsRevision` those records came from, and the scope refusal is
restated over *that* revision's path rather than "the normalized file". `enrich_records`
keeps its exact signature and delegates after the caller's ordinary read, so
`enrichment._plan`, `plan_dictionary_enrichment` and the CLI are behaviourally unchanged.

**Named extension, in `application/enrichment.py`:** `plan_dictionary_enrichment_revision(
config, client, records, revision, ids, *, force_fields, kanji_store)`, modelled on
`audio.plan_targeted_audio_revision:581-623`. It keeps `_plan`'s ledger preflight
(`:174`) and passes the supplied `kanji_store` straight to
`enrich.decide_enrichment` over the injected post-promotion collection. That
argument is the frozen store from §7.9's prepared after-text, never a read of
its pre-write live path. It returns a decision carrying `projected: bool = True`. `commit_dictionary_enrichment`
refuses a projected decision before any write, exactly as `execute_promotion` refuses a
projected `PromotionDecision`. There is **no monkeypatching of `load_records_snapshot`
and no temporary canonical file** written to make `_plan` read a projection.

The preview renders proposal plus those exact facts. The apply re-plans with the ordinary
`plan_dictionary_enrichment` over real canonical for exactly those ids, **applying the
saved fact book's values with no network refetch** and reading the reference
store already applied by §7.9, refuses unless every computed value
equals the bound one (naming both), then commits with `expected_fingerprint`.
`source_forms` is untouched throughout: it is not enrichable, and the apply preserves it
byte for byte. `commit_dictionary_enrichment` already prepares its ledger attribution
before the canonical write and returns `committed_ledger_incomplete` (`:264-292`); that
state blocks `complete` on the same rule as promotion's split, and recovery reaches it
from the intent when the process died first. Its `book.record_enriched(…)` attribution
call (`application/enrichment.py:266-271`) takes the prepared intent's frozen `at=`
(§7.1), so that ledger component's `expected_after` does not move when the resume
happens on a later day. The free promotion reading witness is
untouched: a contradicted reading is held, silence passes.

### 7.9 Reference facts the reviewed cards already need

A word card reads the character stores at build time and never fetches:
`exporters/anki.py:1300-1303` loads `config.kanji_file` and `config.jpdb_readings_file`,
and `deck_package` already binds both as optional `source_inputs` (`:757-758`). A newly
promoted verb bringing a character neither store covers therefore ships a card with a
hole unless the facts are prepared first. **This is reference enrichment of existing word
cards. It mints no character note, no `kanji:` identity and no character deck**, and it
changes nothing in the curated `data/kanji_notes.json`.

`application/kanji_addition.py` cannot do this as written: its scope binds canonical
bytes (`_canonical_snapshot:141-148`, `plan.canonical_fingerprint`) so it cannot describe
a projection before the preview, and it fetches and writes in one call (`:406-513`) so it
cannot hand a prepared payload to a later apply.

**Named extension, in `application/character_notes.py`**, reusing that module's existing
`_snapshot:371`, `_reference:399`, `_facts:421` and `_serialized:386`:

```
prepare_reference_facts(config, characters, *, refresh_readings=False)
    -> ReferenceFactsPreparation
apply_prepared_reference_facts(config, preparation)
recover_prepared_reference_facts(config, preparation)
```

`characters` is `kanji.kanji_in(record.expression)` (`kanji.py:143`) over the post-promotion
projection. Preparation reuses saved facts, fetches only the missing ones through
`kanji.fetch_kanji:262` and `jpdb_kanji.fetch_character:937` under the on-demand lookup
preference, and returns two `ProposedFile`s — `config.kanji_file` and
`config.jpdb_readings_file` — each with `before_sha256` and the exact after-text
`kanji.save_store` / `jpdb_kanji.save_readings` would produce. **Refresh is explicit
only.** The preview renders over those frozen projected files, not over the live ones.

The after-texts bind the finish authority, and apply is a local committed write under
before/after CAS. `kanji.save_store:577` uses `atomic_write_text` and
`jpdb_kanji.save_readings:1093` calls `atomic_write_text_bound` with no expectation, so
neither can CAS on its own: the apply writes each prepared payload with
`atomic_write_bytes_bound(expected_revision=…, expected_absent=…)`, exactly as
`execute_character_notes:958-963` does. Reference preparation runs before the word-enrichment projection, which
uses its frozen projected `kanji_store`. Apply the reference components first
within `enriched`, then the prepared vocabulary and ledger components, so an
ordinary commit re-plan observes the same reference facts used in the preview.
Each distinct path has one before/after pair; none is refetched or regenerated
after approval.

A lookup that is missing or unavailable surfaces as an **exact missing state in the
preview** for the owner to decide on. No background scraping starts, no ranking or
reading is guessed, and nothing audits the language.

### 7.10 `audio_complete`

**Sentence audio is included by default, independently of word audio.** Resolve
the effective `include_example_audio` from the job's saved audio choice,
defaulting to `true` when none exists. Only an explicit owner control or CLI
choice can save an opt-out; an omitted model option cannot become `false`.
The default is a proposal, not spending authority. The one finish confirmation
discloses the exact provider/model, API billing when applicable, and separate
word and sentence counts: expected example slots, unique clip requests, current,
recoverable and provider-required. It binds that effective choice and the exact
requests; no extra per-phase confirmation is introduced. Reuse current clips,
recover exact captured/WAL work without another charge, and generate only the
remaining authorized requests. No silent switch to a different provider or to
a word-only result is permitted.

Derive the expected sentence slots **independently of the audio plan's clip
list**, from every nonblank example's `japanese` field in the final accepted
post-enrichment records. Bind each slot to its record, example position and
exact displayed/spoken input. The existing identity-addressed audio machinery
deduplicates identical displayed sentences within one record only when their
effective spoken inputs also agree: all their slots must be linked, even when
one clip satisfies several. If the effective inputs diverge at the same target,
retain the existing structural refusal in
`src/japanese_anki/audio_cmd.py:1163-1190`
(`prepare_example_audio_profiles`), invoked by
`application/audio.py:466-473` during planning before confirmation. Its current
diagnostic names the record and duplicate displayed sentence; no new positional
diagnostic is required. Never choose one input automatically.
Slot count and unique clip count are different numbers. The plan's example
requests must cover this
expected set exactly when sentence audio is enabled; an empty request list
cannot pass merely because every selected clip is current. This counts fields
and artifact identities; it does not judge Japanese or classify conjugation
cells as sentences. A source job uses only its accepted selection; whole-deck
audio work uses the entire explicitly resolved deck.

The authority binds the prospective enumeration from `plan_targeted_audio_revision` over
the **final post-enrichment merged reviewed records**, not the post-promotion ones.
`ledger.word_audio_request:550` derives its provider input from `_selected_pitch_pattern`
and `record.reading`, and `pitch_accent` and `furigana` are exactly what dictionary
enrichment writes (`enrich.ENRICHABLE_FIELDS:128-137`), while `example_audio_request:570`
follows a reviewed `spoken_japanese`. Planning over pre-enrichment records would bind a
`request_input` and `content_fingerprint` that enrichment then changes, and
`_validate_audio_evolution:1728-1740` would refuse the finish's own work.

Per clip: `record_id`, `kind`, `target`, `request_input`, `forced_accent`,
`content_fingerprint`, `provider`, `initial_state`, `initial_media_sha256` — the exact
fields `_validate_audio_evolution:1692-1772` compares. The whole request and profile
configuration is frozen before the confirmation; a clip identity absent from the
enumeration, or a count change, refuses. Evolution is request-bound and forward-only
(`_ALLOWED_CLIP_EVOLUTION:166-170`).

**What is knowable, and what is not.** Target names are `fp(record.id)` and
`fp(record.id + example.japanese)` (`ledger.py:519-528`), which enrichment does not
touch, so once the facts are settled the post-audio canonical text **is** computable via
`_project_audio_references:617-698` and the phase's canonical `expected_after` is a real
digest. The WAV bytes are not: `_projected_package_media:701-731` binds a known SHA for a
`current` clip, the bound recovery hash for a `recoverable` one, and **`None`** for a
`provider-required` one. There is therefore **no pre-known WAV hash and no pre-computed
ledger digest for this phase**. Its proof is the writer-owned reservations plus the
`pending_audio` WAL and its `data/media/audio/.pending/*.stage` bytes, which the job
reports and never clears. Each paid clip is reserved durably in the finish record at
`before_paid_dispatch` (`application/audio.py:1046`) as `_reserve_paid_clip:1903` does. On
restart `_reconcile_reservations:1828-1900` is the model: an `authorized` reservation
becomes `failed_before_send` and is forgotten; removed evidence refuses rather than
billing again; an unknown outcome is never resent; nonempty frames are never silently
discarded. The build's ledger before-digest is taken after this phase completes, under
the `.janki-audio-operation` lock the finish holds through publication.

Before `audio_complete`, every expected enabled slot must have its canonical
`examples[*].audio` reference written, pointing to the bound target whose bytes
exist and whose ledger entry is current for the exact request and profile.
File creation alone is insufficient. Missing references, missing/stale bytes,
an incomplete writer transaction or an unaccounted paid attempt leave the job
unfinished and resumable. An explicit owner opt-out binds zero new example
requests, preserves existing references/media, and is recorded and displayed
as omitted by owner; it is never reported as successful sentence generation.

### 7.11 `packaged`, then `complete`

New in `application/deck_package.py`; the finish only coordinates.

```
prepare_deck_package(config, projection: DeckPackagePlan, *, audio_completion)
    -> DeckPackagePreparation
publish_prepared_deck_package(config, preparation) -> DeckPackageResult
recover_prepared_deck_package(config, preparation) -> DeckPackageResult | None
assert_projection_realized(projection: DeckPackagePlan, fresh: DeckPackagePlan,
                           *, audio_completion) -> None
```

Two narrow extensions make §7.4's projection plannable and checkable.
`_plan_vocabulary_revision_unlocked` and `plan_vocabulary_deck_package_revision` gain
`reference_sha256: Mapping[Path, str | None]` beside the existing `media_sha256`. It is
consulted **only** where `_input(config, …, optional=True)` reads live bytes for
`config.kanji_file` and `config.jpdb_readings_file` (`deck_package.py:757-758`), and its
values are §7.9's prepared after-hashes for exactly those two stores. It is **not an
arbitrary input override**: any other path, a repeated path, or a malformed hash refuses,
and no other `source_inputs`, `deck_input` or `template_inputs` entry can be supplied.

`assert_projection_realized` is the comparator prepare uses against the authority's
projection, and it is narrower than `_build_compatible:2037-2063`. Both plans must carry
the same field set, and every field must be **equal** — `deck_input`, `source_inputs`
(the record source at the post-audio canonical digest of §7.10, both reference stores at
§7.9's prepared after-hashes), `template_inputs`, `kind`, `deck_name`, `variant`,
`card_types`, `note_count`, `card_count`, `record_ids`, `output_path`,
`configuration_fingerprint` and `conjugation_plan` — with exactly two allowances. First,
`output_revision`/`output_identity` must still be the state the projection bound
(`_output_binding:374-385`), which is what `output_before` records. Second,
`media_inputs` must carry the identical path set, where a path the projection bound to a
hash must still equal it and a path the projection bound to `None` must now be a real
sha256 that is **this finish's own `audio_complete` result**, proven by the immutable
`audio_completion` proof in that phase's receipt. Per resolved slot it records the
bound clip request/profile, target and actual byte hash, checked against the audio
writer's current ledger currency and bytes under its operation lock. A paid clip
also binds its reservation and accounted journal operation; an unpaid VOICEVOX
clip has no paid operation to require. Active recovery uses the existing WAL, but
successful finalization removes that WAL: neither a live WAL row nor a retained
capture is a package precondition. The proof is saved after the existing writer
finishes and before package preparation, revalidated against the current exact
ledger currency and bytes; arbitrary files on disk cannot supply it. The plan `fingerprint` (`plan_fingerprint` on the wire)
differs by construction and is not compared. Any other difference — a reference store
edited outside the prepared payload, a canonical change, an added or vanished media path,
a template or deck edit — refuses, naming the input.

`DeckPackagePreparation` binds the authority `projection` and that realization proof
alongside `plan` — **the fresh concrete plan**, which every later strict `_assert_same`
compares against — plus `preparation_id`, `staged_path` with its own
`staged_sha256` (also `package_sha256`: publication copies those exact bytes), `staged_identity` and the `expected_directory_identity` of the
validated private directory (`io.prepare_bound_directory:474`), `inventory`,
`output_before`, `ledger_before_sha256`, the frozen export delta and `ledger_after_sha256`.
Both the projection proof and the concrete plan are persisted before publication.

**Prepare** takes the same lock set as `_execute_nonconjugation_locked:857-862`, re-plans
fresh under those locks, then calls `assert_projection_realized(projection, fresh, audio_completion=…)` —
**never an unconditional `_assert_same(projection, fresh)`, which the finish's own §7.9
reference writes and §7.10 clips would refuse, stranding a job that has already spent
its paid audio** — then `_assert_pending_audio_clear`, then builds into a private
`config.dist_dir/".janki-prepared"/<preparation_id>.apkg` with `output_expected_absent=True`
— **the confirmed target is not touched**. It verifies the builder result against the plan
(`:881-890`), hashes the staged bytes, re-plans and `_assert_same_after_output:815`, and
computes:

- **`inventory`** — the whole deck, read from the archive under the **current genanki
  `collection.anki2` contract**: the layout `package.write_to_file` produces
  (`exporters/anki.py:1202`) — `collection.anki2`, the `media` JSON manifest, and its
  numbered members — through stdlib `zipfile` and `sqlite3` on a private extracted copy.
  It records each note's GUID (`genanki.guid_for(id)`) **and its exact field bytes**, the
  deck id, the enabled card directions (`plan.card_types`), the template and stylesheet
  hashes (`plan.template_inputs`), the media filenames and their byte hashes
  (`plan.media_inputs`), `deck_input.sha256`, `configuration_fingerprint` and
  `output_path`. Every one of those is validated **against the bound whole-deck
  projection the build consumed, note by note** — a GUID set and a count are not the
  check. Post-audio media hashes are populated only from the allowed request-bound
  `current`/WAL evolution, never from whatever now sits on disk. An archive that is not in
  that format **refuses with a named unsupported-format diagnostic**; nothing is guessed
  at, and there is no second supported layout.
- **`output_before`** — the target's exact `(output_revision, output_identity)` or proven
  absence, from `_output_binding:374-385`.
- **frozen export delta** — one `record_export(record_id, deck_stem, gaps=…, at=<frozen>)`
  per `plan.record_ids`, using `ledger.record_export`'s existing `at` parameter so no date
  is computed after a write, plus `ledger_before_sha256` and the resulting
  `ledger_after_sha256`.

The finish persists this preparation as the build intent, then calls **publish**: re-plan
and `_assert_same` against the persisted concrete `preparation.plan` — strict from here
on, because the projection has already been realized; re-verify the staged bytes still hash to `staged_sha256` at the bound
inode inside the bound directory; publish them with `atomic_write_bytes_bound(target,
data, expected_revision=…, expected_identity=…, expected_absent=…,
expected_directory_identity=…)` bound to `output_before`; then apply the frozen delta and
save. `Ledger.save` is guarded and refuses a ledger that moved since it was read
(`ledger.py:679-684`), so **the ledger contract here is exactly before → after**: there is
no permissive merge onto an arbitrary current ledger, because that would both contradict
that refusal and let this job discard an owner change made in between.

**Recovery** consumes the intent and nothing else, and **recognizes its own output before
any strict re-plan** — after the target lands, `_output_binding` returns the new revision
and a blind `_assert_same` would refuse the finish's own work:

| observed | action |
|---|---|
| target == `package_sha256`, ledger == `ledger_after_sha256` | complete; record the receipt |
| target == `package_sha256`, ledger == `ledger_before_sha256` | apply the frozen delta, save, complete |
| target == `package_sha256`, ledger is a third state | **refuse**, naming both digests; the ledger is preserved |
| target matches `output_before` | re-plan, `_assert_same`, publish as above |
| staged artifact missing, target matches `output_before` | free rebuild from the **same bound input inventory** under a **new** `preparation_id`, recorded as an immutable new attempt that links as superseding — never a mutation of the old intent |
| target is any third revision | **refuse**, naming both digests; the external file is preserved |

`dist/` contents are disposable, which is what makes the rebuild free — it is not content
authority, and a rebuild mints an artifact receipt, never new content authority.
`_build_compatible:2037-2063` compares the *plan* against the authority only; it is never
proof that the target on disk is this job's output, and a valid zip, a matching filename
or an existing package is not adoption evidence. APKG bytes are not reproducible across
builds: the SHA proves *this exact artifact*, the inventory proves *what it contains*.

**`packaged` is the durable build proof; `complete` is minted only after the preview
receipt.** The
`packaged` receipt carries `output_path`, `package_sha256`, `inventory_fingerprint`,
whole-deck counts and `ledger_after_sha256`; a preview failure leaves the job at
`packaged`, and resume renders **only the preview** — no rebuild, no republication, no
paid call. The download offer is minted only from a verified `complete` receipt;
`assistant_packages`' closed resolver registry re-resolves it and re-verifies
`package_sha256` on every read (`workbench/assistant_packages.py:90-118`), and a restart
remints an in-memory token from that durable receipt.

Sentence-audio proof also crosses the artifact boundary. Preserve the current
word note's two export slots (`main_example()` and `example_in("casual")`);
the actual exporter determines which stored examples reach fields and which
fields the enabled card directions draw. Before confirmation, disclose any
additional stored examples using that existing export warning, with stored
and exported slot counts separate. Native sentence generation still covers all
expected stored examples and links them canonically; this adds no note fields
or new card directions. For each exported expected slot, the note's stored
sound reference and the APKG media entry must match the proven target/hash.
Check the stored card fields and packaged bytes, not only the audio plan or
aggregate media count; word clips cannot satisfy sentence coverage. The final
interactive HTML preview resolves the sentence clips drawn by its cards into playable
audio using the real card rendering. A preview before generation marks missing
planned clips as pending; after an explicit opt-out it states that sentence
audio was omitted. Never mark a generated clip playable before its media exists.

### 7.12 Aggregate scope and counts

`StudyFinishScope` in `application/study_finish.py` holds `members: tuple[FinishScope,
...]`, each from the **unchanged** `resolve_finish_scope:411`, plus a deduped
`(record_id, owner_stem, deck_path)` projection and its own fingerprint. `FinishScope`,
`resolve_finish_scope`, `execute_finish_build` and every consumer are untouched; no
replacement pathway is created. Every part/receipt association is retained; two archives
may legitimately receipt one final id. Card, audio and package work runs over the deduped
identities after proving duplicate receipts agree on the owner deck; disagreement refuses.

**Four counts stay four numbers**, never summed: source occurrences per part; receipts per
part; unique canonical identities in the job's selection; and the **whole-deck** package
totals, which are legitimately larger. Every count the finish reports says explicitly
whether it is the full deck inventory or the job subset. The package criterion is
therefore: the archive's GUID set, field bytes and media names **contain** the selection's,
and **equal** the whole-deck inventory the `packaged` receipt minted, which §7.11
validated note by note against the bound whole-deck projection the build consumed. The
authority binds that projection, not an inventory: the realized WAV and archive hashes
do not exist at authority time. The rationale is the one `kanji_finish` records at
`:224-233`.

### 7.13 Proof obligations

Crash mutants this section's contracts must be pinned at: between the two review writes,
with the staging marker landed and the pattern mark not; after the coverage payload
landed, before its receipt; after part *k* of *N* promoted, before the receipt, with the
live file both gone and surviving with held rows; after a part returns
`landed_ledger_incomplete`, and after the same split with the process dying first; after
enrichment's canonical write, before its ledger save; after one reference store landed and
before the other; during the audio WAL with a reservation whose evidence was removed;
after the package intent, before publication; after publication, before the ledger; after
the ledger, before the `packaged` receipt; after `packaged` with the preview failing; a
target replaced externally between intent and publication; a ledger moved externally
between intent and publication; a staged artifact deleted between intent and publication;
a curation update and a promotion execution contending for the shared lock; and a stale
unrelated edit to one bound staging file with all locks held.

Three further obligations are not crash mutants but the same kind of proof:

- **Projection realization (§7.11).** With §7.9's two reference stores written in
  `enriched` and new clips produced in `audio_complete`, package prepare **succeeds**:
  the fresh concrete plan differs from the authority projection only in the enumerated
  media slots. Each of a reference store edited outside the prepared payload, a media
  path appearing or vanishing, an enumerated slot filled by bytes no clip request,
  journal operation or WAL row proves, and any other bound input change **refuses**,
  naming the input. A crash after the concrete plan is persisted and before the ledger
  write retains the authority and resumes from the persisted preparation, never from a
  fresh projection.
- **A part that lands nothing (§7.2, §7.7).** A part whose every row is held by the
  reading witness sits inside the 18-part fold and the parts after it still land: the
  carry is unchanged, its `expected_after` equals its `expected_before`, no archive or
  receipt is minted, and the job stays outstanding until §9's owner disposition covers
  it.
- **A date boundary across a canonical→ledger split (§7.7, §7.8).** The promotion and
  enrichment ledger-split mutants above are re-run with the resume happening on the
  following day: the frozen `added_at` / `seen_at` / `at` values replay from the intent,
  each ledger component still hashes to its bound `expected_after`, and the resume
  completes instead of refusing at a third state.

## 8. Media capabilities

New `application/deck_capabilities.py`: one fixed table keyed off the existing
`anki.deck_kind` (`exporters/anki.py:973`, `KNOWN_DECK_KINDS` at `:970`). No
plugin framework.

```
@dataclass(frozen=True, slots=True)
class DeckMediaCapability:
    kind: str
    parses_source_as_vocabulary: bool
    owns_drill_examples: bool
    synthesizes_word_audio: bool
    synthesizes_example_audio: bool
    packages_media: bool

def capability(kind: str) -> DeckMediaCapability      # refuses an unmapped known kind
def durable_media_owners(config, deck_path, *, revision=None) -> list[VocabularyRecord]
```

Durable ownership and synthesis/packaging are separate columns. For **every**
kind, durable ownership is every durable record version the deck file declares —
`deck.source` records plus inline `notes:` overrides, exactly what
`deck_declared_record_versions` collects — plus that kind's synthetic owners.

| kind | durable media owners | word audio | example audio | packaged media |
|---|---|---|---|---|
| `""` / `vocabulary` | declared record versions | yes | yes | yes |
| `pattern` | declared record versions | no | no | no |
| `conjugation` | declared record versions **plus** `drill_examples` synthetic owners | no | yes | yes |
| `kanji` | character notes in its own curated store, not parsed as vocabulary | no | no | none |

`audio._all_durable_audio_records` keeps canonical records, every deck's
declared record versions and conjugation drill owners.
`audio._audio_deck_source_paths` keeps **every** deck's `source:` in the lock and
staleness set regardless of kind, so `_assert_audio_owners_current` still
refuses on a changed dependency and every source a census reads is
lock-validated.

**The one behavioural change:** a `kanji` deck's `source:` — the curated
character store `config.kanji_notes_file` — is no longer handed to the
vocabulary resolver in the audio census. Removing it costs no prune protection:
`kanji_cards` reports `media_count=0` and `prune_unreferenced` reads only
`record.audio` and `example.audio`.

**An unknown kind refuses before any census, publication or prune.** `deck_kind`
already refuses an unknown `kind:` string; this refusal covers a *known* kind
reaching a capability table with no row. It never implies zero media, never
yields an empty owner set, and never makes an existing clip look unreferenced.

**The repository-wide packaging block stays.**
`deck_package._assert_pending_audio_clear` refuses packaging while *any*
repository audio WAL row exists — a deliberate unchanged constraint, not an open
decision. An unrelated open `pending_audio` row blocks this job's build, and the
job says so plainly and names the exact `janki audio` rerun that finishes it.

---

## 9. Assistant, CLI and completion

### 9.1 Who may do what

A model-emitted closed intent may **read**, **plan** (rendering an owner
confirmation), **open a local owner editor or prepare a non-executing choice**,
or **dispatch and resume work the owner already authorized exactly**, after
fresh binding checks and with no discretion left in it. It may never mint or
select an owner decision, author region geometry, choose or alter a layout
binding, execute curation, supply a **job** coverage reason, review flag or
zero-landed or held-row disposition, choose among ambiguous capture proposals,
widen scope into new protected work, or write canonical content.

| action | who acts | authority |
|---|---|---|
| `study_job_status`, `inspect_source_layout`, `inspect_capture_proposals` | model or owner (read: ids, hashes, states, counts, pointers, verdicts) | none |
| `open_source_part_editor`, `open_layout_editor`, `open_curation_editor`, `open_review_editor` (review flags, coverage reason and audio preference), `open_disposition_editor` | model may open; the owner acts inside | none |
| region publish, layout save, curation apply, choice edit, `create_study_job` | **owner only**, direct bound action → local CAS write | the action itself; no second dialog |
| sentence-audio include/opt-out control within the review editor | **owner only**, bound action → `choices.include_example_audio` CAS write | the action itself, with no written reason or second dialog; missing choice defaults to include; no paid call or canonical write |
| review-flag selection, a part's coverage reason, a zero-landed part's disposition, a held row's exclusion or deferral | **owner only**, direct bound action → local CAS write into the job's `choices`, saved against the exact `job_id`/part/staging-sha256/rendering fingerprints it was taken over | the action itself; no second dialog and no paid work. It writes no staging review mark, no coverage approval and no canonical byte: §7 projects these saved decisions, and the one **Apply and finish** confirmation authorizes their exact writes |
| new destination deck | owner | existing exact deck-creation confirmation (the local job write rides inside it) |
| `extract_study_parts` | model plans, owner confirms | one-use confirmation |
| `retry_study_parts` | model plans, owner confirms | **fresh** one-use confirmation |
| `stage_capture_proposal` | model may dispatch when inspection reports exactly one valid proposal, emitting the operation's opaque resource identity and nothing else; otherwise owner-only | none in the single-proposal case — the original batch confirmation's own answer reaching staging, with the sole valid location resolved locally from that identity; otherwise a one-use selection bound to the owner-supplied location tuple, which no model-emittable field can carry |
| `resume_study_job`, reserved-unsent dispatch | model may request; runs only under recorded authority after fresh binding checks | none new |
| `finish_study_job` | model plans, owner confirms | one confirmation |
| promotion, canonical and saved-fact writes | executed under the finish authority | that authority, never a bare model action |

The audio preference is job-wide: its owner control binds `job_id` and the
job document's CAS revision only. It carries no part, staging hash or rendering
fingerprint, and staging/rendering changes do not stale it. The finish
confirmation re-discloses and binds the effective choice together with fresh
exact clip requests. Saving a preference never changes an already-recorded
finish authority.

An owner control — the region editor's publish, the layout editor's save, the
curation editor's apply, the review and disposition editors' save, the CLI's
`--recipe`/`--layout`/`--choose`/`--flag`/`--reason`/`--exclude`/`--defer` —
binds the exact resource, snapshot digest and choice it was rendered from and
calls the CAS service directly. A review, coverage or disposition decision binds
`job_id`, the part, that part's current staging sha256 and the fingerprint of the
rendering it was taken over. The rendering fingerprint binds card fields,
templates and the relevant source/deck/reference snapshots; it excludes checkbox
selections and the job file's CAS revision, so saving another owner's choice does
not invalidate an earlier one. The finish revalidates all four, and a **stale**
saved choice refuses rather than being applied, so the owner decides again over
the fresh rendering. That binding is **provenance for one reversible local record
of the owner's decision; it is not paid consent, and it writes no canonical byte,
no staging review mark and no coverage approval** — §7's one **Apply and finish**
confirmation authorizes those exact writes from the projected decisions, so the
current §1 step 10 → step 11 sequence is unchanged and nothing is written early.
It can never be spent as any other consent. A reversible owner Save gets no
redundant confirmation dialog and buys nothing.
This does not weaken the broker: a **model-proposed** staged write still goes
through the existing plan plus one-use capability, exactly as `assign_cards`
(`workbench/assistant_adapter.py:1941-1995`) and `approve_coverage` (`:2218-2279`)
do, and a coverage reason stays the owner's literal text
(`_require_owner_literal`), never model-authored. Ambiguous capture selection is
settled by the owner against a **local** rendering of the real cards (§5.3);
that rendering's Japanese never enters ordinary model context.

**Schema extensions.** `ai_schema.AssistantActionIntent.kind` is a closed
`Literal` under `extra="forbid"`, so each intent above is added there or cannot
be emitted. `AssistantActionOptions` (also `extra="forbid"`) gains only
`retry_child_indices: list[int]`; `concurrency_limit` and `deck_scope` are
reused unchanged. `include_example_audio` belongs to the job's owner-controlled
`choices`, **not** `AssistantActionOptions`: the earlier proposed model field
is removed from this plan. The model may open the review editor's audio control and plan using
the effective saved/default choice; it cannot emit an opt-out. The owner's
control saves the Boolean choice via `record_choice` without another
confirmation; spending still requires the one exact finish confirmation.
An audio preference needs no written reason. It gains **no capture-selection field**:
`capture_proposal_sha256` is not added anywhere, because anything on
`AssistantActionOptions` is model-emittable by construction — it is a field of
`AssistantActionIntent`, itself a field of `AssistantAgentAnswer.action_intents`.
The model's `stage_capture_proposal` argument set is the operation's opaque
resource identity and nothing else, and its only unambiguous route is the
single-valid-proposal case the service itself refuses when the capture holds more
than one (§5.3). **Every piece of §5.3's location tuple — `capture_sha256`,
`response_schema_fingerprint`, `frame_index`, `block_index`, `tool_use_id`,
`json_pointer`, `proposal_sha256` — travels only on the owner control or the
CLI's typed `--proposal SHA256 [--at POINTER]`, and never as a model-emittable
scalar.** No **job** coverage reason, review flag or disposition is added to
the model schema: those carry on the owner control and explicit CLI arguments
of §9.3. The existing `AssistantActionOptions.coverage_reason` remains for the
separate `approve_coverage` action, whose verbatim owner-message relay still
passes `_require_owner_literal`; this plan does not remove or widen it. `application/assistant_context.ResourceKind`
gains `"source_part"` (S3) and `"study_job"` (S4), each with a discovery entry,
`_opaque_id` minting and a `snapshot` branch. Snapshots disclose ids, hashes,
states and counts — never private unfinished paid reply bytes.

### 9.2 Assistant surface

`extract_study_parts` renders one batch naming every published part's name and
sha256, the parent and recipe, each child's source, request fingerprint,
provider/model and billing identity, the pinned mode and layout revision, the
concurrency limit, and every write. One confirmation buys precisely those
children; it never becomes one paid parent call and their answers never merge
into one synthetic provenance.

### 9.3 CLI — the same services, explicitly

Operation-scoped recovery ships first, with no job dependency, on the existing
batch command: `janki extract-batch inspect-capture --operation ID` and
`janki extract-batch recover --operation ID [--proposal SHA256] [--at POINTER]
[--yes]`. Source preparation ships next as `janki source-parts`, also
job-independent. `janki study` is added later with subparsers, registered as
`cli_extract_batches.add_batch_parser` is; each subcommand calls the same
still-supported service, and none is a shim over a retired one:

- `janki study new --source NAME --deck STEM` (or `--create-deck NAME
  [--standalone] [--directions …]`, which runs the existing deck-creation
  confirmation and then binds the created deck)
- `janki study parts JOB --recipe FILE [--publish]` · `janki study layout JOB
  --part PART --layout FILE`
- `janki study extract JOB [--mode MODE] [--concurrency N] [--yes]`
- `janki study curate JOB (--column … | --choose …)`
- `janki study recover JOB --operation ID [--proposal SHA256] [--at POINTER]`
- `janki study assign JOB --part PART --deck STEM [--records ID …]`
- `janki study review JOB --part PART --rendering FINGERPRINT
  (--flag ID … | --flag-file FILE |
  --flag-none) [--patterns | --no-patterns]`
- `janki study coverage JOB --part PART --rendering FINGERPRINT
  (--reason TEXT | --reason-file FILE)`
- `janki study disposition JOB --part PART --rendering FINGERPRINT
  (--exclude | --defer) [--records ID …]
  (--reason TEXT | --reason-file FILE)`
- `janki study status [JOB]` · `janki study preview JOB --output FILE`
- `janki study finish JOB [--example-audio | --no-example-audio] [--yes]` · `janki study resume JOB`

Assignment has no CLI today. Concrete extension: factor
`assistant_assignment.plan_assignment_for_paths(config, *, proposal_path,
destination_path, record_ids, instruction)` out of the existing `_prepare`, have
the resource-id-keyed `plan_assignment` delegate to it, and give the CLI the same
call. `execute_assignment` stays the sole writer. `--yes` is how the owner
answers a consent prompt in advance; a non-TTY without it refuses.

`janki study preview` prints the rendered-content fingerprint beside its
HTML output path; the HTML also carries it. Each owner-decision command requires
that value through `--rendering FINGERPRINT`. The service compares the supplied
fingerprint with the freshly resolved rendering before saving any choice and
refuses drift in its source/deck/reference/template inputs. It never silently
substitutes a newly computed fingerprint for the one the owner supplied. The
Assistant editor supplies its displayed rendering's fingerprint in its bound
control, so both surfaces enforce the same preview-to-decision binding.

**The owner-decision commands, settled.** `review`, `coverage` and `disposition`
are the CLI carriers for exactly the decisions §7.2/§7.4 project and §9.6
requires, and for no others. Each **saves** the owner's choice into the job
document's `choices` namespace (§2.1) through the same service the Assistant's
editor calls, bound to `job_id`, the part, that part's current staging sha256 and
the fingerprint of the rendering it was taken over. **None of them writes
anything else** — no staging review mark, no coverage approval, no canonical byte
and no paid call — so the §1 step 10 → step 11 order is preserved. `janki study
finish` projects them and its one confirmation authorizes the exact writes
through the factored `ReviewPanel.prepare`/apply with the pure
`staging.render_coverage_approval` composed into the same final staging payload.
The finish never calls the standalone `approve_coverage_as_owner` as a second
write; that route remains available separately as described below. A saved choice whose bound
job/part/staging/rendering fingerprint has moved is **stale and refuses**; the
owner decides again over the fresh rendering, and nothing is carried forward
silently.

Their options are complete and mutually exclusive: exactly one of `--flag ID …`,
`--flag-file FILE` or `--flag-none`, so "no rows flagged" is stated rather than
inferred from an omitted argument; for a part carrying a pattern set, exactly one
of `--patterns` or `--no-patterns` (§9.1's existing `review_patterns` choice,
never inferred), and neither is accepted for a part without one; exactly one of
`--exclude` or `--defer`, with `--records ID …` selecting held rows and its
absence meaning the whole zero-landed part; and exactly one of `--reason TEXT` or
`--reason-file FILE`, required, refusing a blank or whitespace-only reason as
`coverage._approval_payload` already does. The reason is the owner's literal text
typed on the command line or written in an owner-authored file — nothing derives
it from model output, and **`--yes` never manufactures a reason, a review flag or
a disposition**; it answers only a consent prompt that already exists. `janki
study finish`'s mutually exclusive `--example-audio` / `--no-example-audio`
flags set the owner's saved audio choice via the same CAS service as the
Assistant control. With neither flag, use the saved choice or default to
sentence audio included; parser omission is `None`, never an implicit `false`.
The positive flag can re-enable a previously opted-out choice. They select
§9.1's `include_example_audio`, the same sentence scope `janki audio --examples`
makes. These names avoid confusing audio selection with row-selection flags.
There is no separate old opt-in-only interpretation or fallback flag path.

The existing protected owner coverage route is untouched and stays usable: the
workbench control and the Assistant's `approve_coverage` action still record a
standalone `authority: repository-owner` approval through
`coverage.approve_coverage_as_owner` for a part outside a job. `promote
--accept-coverage` / `--reaccept-coverage` remain the separate **paid**
model-authority route (`authority: model`) and are never what a study job's
finish uses, since the finish batch enumerates no such call. These three
commands and their Assistant editors ship with the finish in **S6**, whose file
list already carries `application/coverage.py`, `workbench/review.py` and
`cli.py`; S4 may reserve the `choices` keys in the job schema without shipping a
writer or a CLI for them.

### 9.4 Busy behaviour

`assistant_agent` authorizes every ordinary model turn through the journal, so a
live extraction batch **blocks ordinary paid chat**. That is correct and
unchanged: nothing queues the owner's message and there is no hidden queued paid
turn. Local controls stay available. `MANAGE_EXTRACTION_BATCHES_MESSAGE` is
already handled ahead of the journal and busy guards
(`workbench/assistant.py:3202`); the same treatment is given to
`study_job_status`, the job preview selector, `inspect_capture_proposals` and
the `MANAGE_OPERATIONS_MESSAGE` route (`:3139`), so a blocking operation can
always be settled. Resume and retry selectors stay blocked by `_busy_threads`
while that thread's worker is alive and become available after it frees or after
a restart, driven by durable receipts. When a batch blocks a paid turn the reply
names the job, the batch, `k of n` children settled, what is in flight and the
available local controls, replacing the generic "Earlier model call needs
attention" branch (`:3276-3293`) **for this case only**.

### 9.5 The effective extraction frontier

Readiness is computed over **effective children**, derived from confirmed
execution, never from file ordering.

- An edge from an old child to a new one exists only when the retry batch has a
  **confirmed execution receipt** matching its manifest (`_confirmed_execution`),
  that batch's `discards` names the old child's exact operation id, and the new
  child's `source_sha256`, part ancestry and bound layout lineage match the old
  child's.
- A merely **prepared** retry manifest is a no-op: no edge, no supersession.
- A **successful** child is never superseded; `_RETRY_REFUSALS` already refuses
  retrying a `committed` child, and a discard edge out of one refuses.
- Forks, cycles and disjoint-purpose chains **refuse** with both ids named.
  Nothing resolves them by "latest".
- Discarded attempts stay listed as history with their disclosed cost; evidence
  accounting is never deleted to tidy the frontier.
- An unknown outcome is never redispatched, occupies a slot, blocks unrelated
  paid work including ordinary chat, and is never counted as progress. Its retry
  needs a fresh confirmation acknowledging the uncertain prior cost and
  explicitly discarding its bound evidence.

`assistant_adapter.resume_extraction_batch` reports `complete` when
`pending + unknown == 0`. That is batch-surface eligibility, not artifact
readiness, and it is never reused as job completion.

### 9.6 Job completion

A study job is complete only when **all** of these hold:

1. Every part has exactly one effective child, `committed` with
   `bookkeeping_complete` true. **Every known paid attempt is accounted for**:
   no live or unaccounted operation carries this job id, and every discarded or
   unknown attempt is listed with its disclosed cost.
2. **Per part, by disposition.** A part with accepted IDs, newly landed or
   already archived, has a validated `data/staging/done/` archive and the exact
   source-bound `promotion_batches` receipts retained in its scope. Their
   selected `promoted_ids` account for exactly the accepted IDs the authority
   bound for that part; every owning writer's bookkeeping is finished. A
   returned `landed_ledger_incomplete` or `landed_ai_ledger_incomplete` blocks
   completion until the prescribed writer recovery finishes it. A part with
   no accepted IDs has an explicit owner disposition with their reason, plus
   its source part, immutable `candidate_accounting`, and the live staging file
   where its writer retains one, otherwise its bound archive (§7.7's state
   table). Pattern-only archives are valid evidence without a vocabulary
   promotion receipt; no receipt is fabricated for a part that promoted nothing.
3. Every requested accepted card landed. A row held back — for example by the
   jpdb reading witness — keeps the job **incomplete** unless the owner
   explicitly excluded or deferred it from this job's exact scope, recorded with
   their reason and the evidence and without deleting the row.
4. The aggregate scope re-resolves to exactly the selected canonical projection,
   with agreeing owner bindings (§7.10).
5. `ledger.pending_audio` is empty and every selected clip is current by exact
   request and profile. Sentence inclusion is resolved as §7.10 specifies;
   every independently derived expected sentence slot is linked in canonical
   content and covered by current media, or the receipt records the owner's
   explicit opt-out. A word-only plan cannot vacuously satisfy this condition.
6. The package exists and its bytes hash to the receipt's `package_sha256`; its
   **whole-deck** inventory — every note GUID, the field bytes and template and
   stylesheet hashes, every media name and sha256, and the note/card/media
   counts — **equals** the inventory the `packaged` receipt recorded, which §7.11
   validated note by note against the bound whole-deck projection the build consumed,
   and the job's selection GUIDs and their media are contained in it. Selection coverage is a separate
   containment check and is never compared against whole-deck totals.
   Each exported sentence slot's sound reference also resolves inside the
   actual APKG to its proven bytes; stored examples beyond the note's export
   slots remain explicitly disclosed, with their canonical references and
   media retained (§7.11). The export-ledger entry is recorded.
7. The finish receipt state is `complete`, reached in this order: package
   receipt, then the final preview receipt, then `complete` and the download
   offer. A preview that fails after the package is proven leaves a packaged
   receipt whose resume re-attempts **only the preview**; the job never reports a
   failed preview after it is complete.

Four numbers stay four numbers and are never summed: source-row occurrences per
part, effective children per part, unique canonical identities, and whole-deck
package totals.

### 9.7 Failure semantics

**Free jpdb.** Bounded retries only, for known read-only endpoint operations,
using the existing `RETRYABLE_STATUSES` / `RETRYABLE_ERRORS` plus transport
timeouts and TLS failures. Saved facts are reused, refresh stays explicit,
builds stay offline, and this budget never applies to paid dispatch.

**Captured-first offline recovery** runs before login and prompt checks, as
`resume_extraction_batch` already does, so a logged-out session or an edited
prompt cannot strand an already-paid reply behind a sibling's preflight.
Reserved unsent siblings resume under the existing all-unsent preflight — one
changed binding refuses that whole set.

**Failures are recorded, not forgotten.** A failed combined preview render, a
refused build, or package bytes that landed while the export-ledger write failed
each record their exact evidence in the owning phase intent, and `janki study
status` keeps reporting them. Nothing treats a missing `dist/` artifact as
"never attempted", and no `complete` state or download is minted while any of
them is open.

**Download.** `assistant_packages._resolve` hard-wires
`kanji_finish.inspect_kanji_finish` (`workbench/assistant_packages.py:92`). It
becomes a closed resolver registry `{"kanji_finish": …, "study_finish": …}` with
the receipt kind recorded when the offer is minted; an unknown kind refuses.
Tokens stay in-memory convenience handles: after a restart a fresh token is
minted from the durable completed authority, resolved by exact receipt id and
re-verified `package_sha256` on every read.

---
