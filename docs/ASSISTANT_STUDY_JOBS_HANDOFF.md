# Assistant study jobs — S6 and S7 handoff

This document is the working prompt for the agent that resumes this project. It
is self-contained: everything needed to start is here or in the repository
documents it names.

**Your task.** Implement **S6**, then **S7**, of
[`ASSISTANT_STUDY_JOBS_PLAN.md`](ASSISTANT_STUDY_JOBS_PLAN.md).
[`DESIGN.md`](DESIGN.md) leads; when it and any other document disagree, DESIGN
wins and the other changes.
[`ASSISTANT_STUDY_JOBS_CONTRACTS.md`](ASSISTANT_STUDY_JOBS_CONTRACTS.md) §7–§9
gives the exact contracts (review/promotion/finish, media capabilities,
Assistant/CLI/completion); plan §10.1–§10.3 gives the file inventory, the test
rules and the parallel schedule. Read `AGENTS.md` first — its rules are not
restated here except where S6 touches them.

---

## 1. What is authorized, and what is not

The owner has authorized implementing this plan. That authorization covers
**code, tests and documents** inside the approved S6/S7 scope: ordinary
signature, wire-shape and file-ownership choices inside the approved contracts
are yours, and you should not ask the owner to approve them or reopen clean
S0–S5.

It authorizes nothing else. Every protected decision keeps its **exact existing
gate**, unchanged and unwidened by this document:

- **Paid content calls.** `janki extract`, `revise`, `promote --accept-coverage`,
  OpenAI Realtime audio and the `anthropic-api` revision provider each still
  need their own exact authorization. S6/S7 are offline work: **make no paid
  provider or content call of any kind**, and add no default that could become
  one. A default-on proposal is a proposal, never spending authority.
- **Disclosure** of private source or card bytes to a paid model.
- **Discard/forget decisions** over journalled operations, and any
  `operations --end` / `--forget` decision. Do not auto-forget an operation to
  clear a merge.
- **Identity decisions**: ambiguous new-identity resolution, existing-identity
  migration, accepted-risk approval, coverage approval where it is still the
  owner's to write.

An agent may not infer, generate, grant or widen one of those, and may not
answer an owner prompt as the owner.

---

## 2. The base: what is complete, committed and settled

Local branch `main`, repository `/Users/brandonivers/Development/ankigen`.

| what | commit |
|---|---|
| S4 — study job, job actions, CLI, busy replies, deck-creation integration | `2a3338e716dd78f4dd955d6e3f7c0e991231dfb1` |
| DESIGN amendments C and D | `f05a3e88d12351195237cddc488b6c369320a205` |
| Owner deck-parent rename (`Janki Study Decks`) | `d34b12a8cc683419b1501460b6fbc3239234eca8` |
| **S5 — layout, `source_forms`, cross-part curation** | **`0e5300788dcfbdbd43902b763e0528a709407c85`** |

**S5 is complete, reviewed, gated and committed.** That commit is the
authoritative S5 base and the authoritative statement of S5's APIs. Several
intermediate code-only snapshots and patches exist from S5's correction passes;
they were candidates that have since been integrated into `0e53007`. Where any
snapshot, patch or earlier handoff text disagrees with the committed code, **the
committed code wins** — read it.

**Integrated `make gates` at `0e53007`** (primary checkout, root-run): exit 0,
**5642 passed, 800 warnings, Ruff clean, sample deck build passed, 562.97s**.
Log: `/tmp/janki-study-s5-v3-main-gates.log` (local file, not committed).

**Independent reviews, all closed:**

| scope | verdict |
|---|---|
| Assistant surfaces and owner layout/abandon controls | CLEAN |
| D1/D2 propagation and functional follow-ups | closed; its one diagnostic-only finding closed separately |
| Diagnostic correction (refusal text) | CLEAN; both reviewed file hashes verified unchanged in the combined final tree |
| Curation F2/F3/F4 | closed |
| Final curation predecessor block and `supersedes`, including direct owner callers | CLEAN |

Integrity of that final tree: full-patch sha256
`6d4f7535984cc014c1c78a64a5dbd422273ac0c4def6b61e5b5b1c19c6a49af7`, 50 verified
source bindings, frozen review snapshot `/tmp/janki-study-s5-v3-code-review`
with hashes `/tmp/janki-study-s5-v3-review-hashes.json` (both local only).

**Not implemented, by anyone, at this commit:**

- **S6** — no `application/study_finish.py` exists; no `janki study review`,
  `coverage`, `disposition` or `finish` control exists; `_DEFERRED_CHOICE_KEYS`
  still refuses those choice keys by name and `"finish"` is still an unwritable
  intent kind.
- **S7** — no scenario module exists. The prepared fixture bank, helper probes
  and any pre-authored test candidates are **preparation, not acceptance**: a
  green prep or probe run proves nothing about the journey.
- **DESIGN amendment F** — prepared but **not applied**. `docs/DESIGN.md` is at
  sha256 `6975958572e6c0147df8037a459e11d556757aca4923dbd6e29f2518aa1186c4`,
  which is the pre-F byte state. See §6.3 for how to land it.

**Repository state check at S5 close.** Native read-only `janki operations`
reported 19 retained entries, all committed; nothing was forgotten or changed.
No `data/`, `dist/` or source-content byte changed in S5, and no paid content
call was made.

**Push.** Local `main` is ahead of `origin/main`; nothing has been pushed to
GitHub yet. The owner already asked for integration on `main` — that
authorization stands and needs no re-confirmation. Re-check remote state before
any eventual push. Do not treat pushing as a new approval gate, and do not ask
for permission already given.

---

## 3. Owner content and repository rules — do not touch

The owner's verb deck is **delivered and accepted**: 187 notes, 561 media
(374 sentence clips + 187 word clips), stable GUIDs,
`dist/brandon-japanese-201-verbs-*.apkg` and the audio preview HTML. It needs no
rebuild beyond the completed parent rename. Do not restart content delivery, do
not rebuild it, and do not reuse its artifacts as S7 fixtures.

The deck-parent rename at `d34b12a` is the owner's completed decision — parent
`Janki Study Decks` across all 17 deck YAMLs, project/default config, docs and
the config test, with 17 decks rebuilt offline. **Do not revert or redo it.**

Also preserve: the owner's untracked inbox source PDFs, the generated `dist/`
previews, and the `.claude/hooks/DISABLED` marker. Never `git add -A` — commit
explicit paths. Never rewrite the live journal or ledger. Never modify files
under `data/inbox/`. Tests write temporary repositories, never the owner's
collection, journal, media or source files.

---

## 4. The S5 APIs you build on

These are read out of the committed code at `0e53007`. Treat them as the
existing surface; §5 lists what does **not** exist yet.

### 4.1 `extract.py` — layout datatype, wire and mode

```python
LAYOUT_MODE = "table-layout"
MODES: tuple[str, ...] = ("table", "prose", LAYOUT_MODE)
TABLE_MODES: frozenset[str] = frozenset({"table", LAYOUT_MODE})
LAYOUT_MODE_REFUSAL: str

class TableColumn:  column_id; ordinal; label_witnesses; display_label; to_wire()
class TableLayout:  layout_id; revision; columns; column_ids; identity; to_wire(); from_wire(raw, *, where=...)

def layout_from_provenance(provenance: Mapping[str, Any]) -> TableLayout | None
def refuse_unbound_layout_mode(mode: str | None, *, control: str) -> None
```

The frozen wire is `{"layout_id", "revision", "columns":[{"column_id",
"ordinal", "label_witnesses", "display_label"}]}` and it is the only spelling.
Identities, witnesses and labels are preserved unnormalized; two columns may
share a `display_label`; a duplicate `column_id` refuses.
`layout_from_provenance` takes **exactly one parameter** — the saved provenance,
never a job or a store.

Changed signatures all take a keyword `layout: TableLayout | None = None`:
`prompt_for`, `prompt_provenance`, `extract_candidates`, `build_records`,
`_raw_fields`. `response_schema_fingerprint`, `EXTRACTION_SCHEMA_VERSION` and
`candidate_schema()` are byte-identical; only the request fingerprint moves.
`normalize_response` and `coverage_block` branch on `TABLE_MODES`, so an empty
`table-layout` answer is `unmeasured` and blocking exactly as `table` is.
`build_records` refuses a parsed key outside the request's `column_id` set,
naming the key, the part and `<layout_id> revision <n>`
(`code="extract-layout-unknown-column"`).

### 4.2 `models.py`, `io.py`, `exporters/anki.py`

`SourceFormColumn(id, label)` and `SourceFormsTable(columns, cells)` with
`from_dict` / `to_dict` / `is_blank()` / `rows()`;
`VocabularyRecord.source_forms: SourceFormsTable | None = None`, placed before
`source`. Absent, `null` and the one defined empty table canonicalize to a
single value, so no existing record's serialized bytes change. Cells are
preserved verbatim: a printed blank and an absent column stay different facts.
`io.merge_records` picks the field up automatically; a present table is a value,
not a hole. `exporters/anki._conjugation_field` selects `record.source_forms`
when present and `record.conjugations` otherwise — same `Conjugations` field, no
notetype, GUID or card-direction change.

### 4.3 `staging.py` — the narrow prepare/apply pair

```python
FIELD_OPERATION_ACTIONS = ("replace", "remove")
class FieldOperation:          row_index; record_id; field; action; key=None; value=None
class PreparedStagingUpdate:   text; sha256_before; sha256_after; applied

def prepare_record_update(path, records, *, operations=()) -> PreparedStagingUpdate
def apply_prepared_update(path, prepared) -> Path        # caller's lock, CAS on sha256_before
```

A `FieldOperation` names its row by index **and** `record_id`; a mismatch
refuses. Emptying a cell is `replace` with `""`; removing a cell is `remove`
with a `key`; removing the table is `remove` without one; removing something
already absent refuses. `_validate_prompt_provenance` derives its mode allowlist
from `extract.MODES` plus `auto` and requires `table_layout` if and only if the
mode is `table-layout`.

### 4.4 Extraction transport

`application/extraction.py` carries `layout=` / `layouts=` (aligned one-per-input
under `table-layout`, none otherwise; both mismatches refuse with
`code="extract-layout-misaligned"` before anything is planned).
`ExtractionDispatchExpectation` gained `table_layout` and enforces the pairing in
`__post_init__`; `revalidate_extraction_request` reproduces the expectation's
frozen layout; `complete_extraction` and `application/capture_recovery.py` read
the layout back from the **saved manifest**, never from a job document.
`application/extraction_batch.py`'s `plan_extraction_batch(..., layouts=())`
aligns to `sources` (`code="extract-batch-layouts"`); a child's `table_layout`
is emitted only when set, so jobless/layoutless manifest bytes,
`manifest_sha256` and consent fingerprints are unchanged; retries restore each
child's **complete** layout through `layout_from_provenance`, never an identity
pair and never a job lookup.

### 4.5 `application/study_job.py`

```python
def append_layout(config, job_id, layout, *, bind=(), allow_identical=False, expected_revision) -> StudyJob
def job_layout_bindings(job) -> dict[str, extract.TableLayout]
def job_part_layouts(config, job_id) -> dict[str, str]        # part -> "id@revision"
def parts_with_bound_layouts(config) -> dict[str, tuple[str, ...]]
```

`append_layout` is the sole creator of a `(layout_id, revision)`: it refuses a
differing re-save of an existing key, refuses an identical re-save unless
`allow_identical`, and appends the revision **and** repoints every part in `bind`
in one compare-and-swap write. `_WRITABLE_INTENT_KINDS` gained `"curation"`;
`"finish"` is still refused by name. `StudyJobStatus` gained
`layout_bindings` and `pending_curation_intents`, both derived at read time and
both in `to_dict()` — **any other constructor of `StudyJobStatus` must supply
them.** `plan_job_extraction_batch` refuses a mixed bound/unbound batch, a bound
batch under any mode but `table-layout`, and `table-layout` with nothing bound.

### 4.6 `application/study_curation.py` — curation, resume and abandon

```python
CURATION_LOCK_NAME = ".janki-curation-lock"
CURATION_FIELDS = ("source_forms",)
MISSING_DIGEST = "missing"
class StudyCurationError(JankiError)

def curation_guard(config) -> ContextManager[None]
def read_curation_groups(config, job_id) -> tuple[CurationGroup, ...]
def plan_curation(config, job_id, choices, *, decision: str) -> CurationPlan
def apply_curation(config, plan) -> CurationOutcome
def resume_curation(config, job_id, intent_id) -> CurationOutcome
def abandon_intent(config, job_id, intent_id, *, expected_intent_sha256: str) -> CurationOutcome
def curation_intent_digest(intent: study_job.ActionIntent) -> str
def unresolved_curation_predecessors(config, plan) -> tuple[CurationBarrier, ...]
def open_curation_barriers(config) -> tuple[CurationBarrier, ...]
def pending_curation_refusal(config, staging_path) -> str      # "" when nothing blocks
```

Dataclasses: `CurationOccurrence`, `CurationGroup`, `CurationChoice`,
`PreparedFileUpdate`, `CurationPlan`, `CurationOutcome` (with `observed` and
`supersedes`), `CurationBarrier` (with `intent_sha256` and `unsatisfiable`).

Semantics that S6 must not break:

- The curation intent is an ordinary `study_job.ActionIntent` with
  `kind="curation"`. `reserves["prepared"]` holds one `PreparedFileUpdate` per
  file including the **complete prepared file text**; `bindings` holds the
  owner's `decision` and exact choices.
- `apply_curation` takes the coordination guard and first refuses when
  `unresolved_curation_predecessors` is non-empty. It locks and prechecks the
  whole bound set **before** `append_intent` (a stale plan records nothing),
  releases those path locks, appends the fsynced intent **before any effect**,
  then takes every
  affected path's lock in sorted order, prechecks all of them, writes in the
  same order, and appends the `applied` outcome only when every entry is at
  `sha256_after`. One mismatch refuses the whole decision and leaves the intent
  open.
- A fresh intent records `supersedes` = the latest closed curation intent of
  this job whose recorded choices name a `(record_id, field)` subject the plan
  settles again. The edge is **derived structurally** from the job's own
  append-only log, never proposed by a model, and `study_job.append_intent` is
  the enforcer.
- `resume_curation` replays the recorded text and recomputes nothing.
- `abandon_intent` refuses a non-`curation` kind, an already-closed intent and
  any digest but `curation_intent_digest(intent)`; it takes every bound path's
  lock in sorted order, measures each path (an unreadable bound path is recorded
  as `MISSING_DIGEST`), and appends a `state="abandoned"` outcome. It **reverts
  nothing, restores nothing and deletes no evidence.**
- An intent with no terminal outcome **is** the barrier — there is no marker
  file. `pending_curation_refusal` is bound to the exact staging path. An
  unreadable job document **refuses** rather than being skipped (§6.5's "a
  skipped barrier is a lifted one"); this is the widest blast radius in S5 and
  is deliberate.

### 4.7 Promotion guard and the guarded/unguarded split

`promotion.decide_promotion` carries the `curation-pending` gate in the
structural group, immediately after the archive gate and **ahead of** coverage,
returning `blocked(..., gate="curation-pending")` (including for an unreadable
job document). `promotion.execute_promotion` **rechecks** the same barrier after
its blocked-decision check and before every branch.

```python
# application/assistant_promotion.py
def execute_promotion_action(config, expected, *, client_factory=None, progress=None)      # takes curation_guard, then delegates
def execute_promotion_action_under_guard(config, expected, *, client_factory=None, progress=None)
```

Entries that take `curation_guard` **before** every existing janki lock:
`cli.command_promote`, `workbench/server._promote`,
`assistant_promotion.execute_promotion_action`, and
`card_revision_finish.plan_card_revision_finish` / `plan_ai_enrichment_finish` /
`execute_card_revision_finish` / `resume_card_revision_finish` (each outside
`.janki-audio-operation`). `card_revision_finish._execute_record_locked` calls
the `_under_guard` entry because `io.exclusive_path_lock` is **not re-entrant**.

**Two rules for S6's `study_finish`:**

1. It must take `study_curation.curation_guard` at its own **outermost** entry
   and retain that one acquisition while applying its prepared writers;
   never nest a second acquisition.
2. Its per-part writer entrypoints are `promotion.py`'s **prepared writer**
   entrypoints (contracts §7.7: `prepare_source_promotion` /
   `apply_prepared_source_promotion` / `recover_promotion_intent`), **not** the
   Assistant broker's `execute_promotion_action`, which re-plans from
   `expected.resource_id` through `plan_promotion_action` and asserts an
   Assistant plan fingerprint. A study-job part is a staging path with a
   prepared intent, not a broker-resolved proposal. Earlier S5 handoff prose
   that pointed at the broker's live re-planner is superseded by the approved
   plan and contracts and by the S6 mapper's settlement.

### 4.8 Surfaces

**CLI (`cli_study.py`).** `janki study` currently offers `new`, `parts`,
`layout`, `curate`, `extract`, `recover`, `assign`, `status`, `preview`,
`resume`. S6 adds `review`, `coverage`, `disposition`, `finish`.

```
janki study layout JOB [--part PART --layout FILE] [--rebind]
janki study curate JOB [--record ID --column CELL_ID (--set TEXT | --blank | --remove)]
                       [--record ID --drop-table] [--choose RECORD_ID=PART]
                       [--note TEXT] [--resume INTENT]
                       [--abandon INTENT --expect SHA256]
```

`--expect` is required with `--abandon`; `--resume` with `--abandon` refuses.
Bare `janki study curate JOB` prints every open decision with its intent digest
and which route is open, ahead of the existing listing; bare `janki study layout
JOB` prints the provenance back. The layout file is the owner's own JSON in the
§4.1 wire — janki reads no header, matches no label and mints no identity.
`janki study status` carries a `Layout <part>: <id@revision>` line per binding
and a pending-curation line. A curation that replaced a closed decision prints
`recorded over closed decision <intent_id>, which keeps its own record and
evidence in this job's log`. Refusals reach the owner through `cli.main`'s
existing `JankiError` handler: `error: …` on stderr, exit 1.

**`ai_schema.py`.** `AssistantActionIntent.kind` gained exactly three literals:
`inspect_source_layout`, `open_layout_editor`, `open_curation_editor`. **No new
`AssistantActionOptions` field.** No intent id, digest, column identity, printed
witness, display label, layout revision, geometry, revision or curation choice
is model-emittable. Keep it that way in S6: an audio preference belongs in the
job's owner choices, never on `AssistantActionOptions`.

**`workbench/assistant.py` — `RevisionCallbacks` grew five methods.** Any other
implementer, **including test fakes**, must have all five:

```python
study_job_layout_text(*, job_id: str) -> str | Awaitable[str]
open_study_job_layout_editor(*, job_id: str) -> str | Awaitable[str]
study_job_curation_text(*, job_id: str) -> str | Awaitable[str]
apply_study_job_curation(*, job_id: str, target: str) -> str | Awaitable[str]
abandon_study_job_curation(*, job_id: str, target: str) -> str | Awaitable[str]
```

```python
_STUDY_JOB_ACTIONS  = ("status", "preview", "resume", "inspect-capture", "parts",
                       "include-part", "exclude-part", "layout", "edit-layout",
                       "curate", "adopt-forms", "abandon-curation")
_STUDY_JOB_READS    = ("status", "preview", "inspect-capture", "parts",
                       "layout", "edit-layout", "curate")
_STUDY_JOB_CHOICES  = ("include-part", "exclude-part")
_STUDY_JOB_APPLIES  = ("adopt-forms", "abandon-curation")
_STUDY_JOB_TARGETED = (*_STUDY_JOB_CHOICES, *_STUDY_JOB_APPLIES)
```

`target` is nonempty exactly for `_STUDY_JOB_TARGETED`. The `adopt-forms` target
is `"<record_id>|<part_name>"`; the `abandon-curation` target is
`"<intent_id>|<intent_sha256>"`, minted at render time, so a tampered digest is
refused by the capability check **and** by the service.

**`workbench/assistant_adapter.py`.**

```python
open_layout_editor(*, job_id: str) -> LayoutEditorOffer     # the owner's editor session
open_study_job_layout_editor(*, job_id: str) -> str         # the model-reachable read that offers it
study_job_layout_text(*, job_id: str) -> str
study_job_curation_text(*, job_id: str) -> str
apply_study_job_curation(*, job_id: str, target: str) -> str
abandon_study_job_curation(*, job_id: str, target: str) -> str
_curation_actions(...)          _MAX_CURATION_CONTROLS = 24
_curation_abandon_actions(...)  _MAX_ABANDON_CONTROLS = 8
_refuse_layout_bound_part(...)
```

`study_job_layout_text` takes **only** `job_id` in the committed code —
editing is a separate route (`open_study_job_layout_editor` →
`open_layout_editor`), not a flag on the text read. `prepare_source_extraction`
refuses a source some job binds a layout to, naming the job and the mode;
`study_job_status_text` shows layout bindings and the pending-curation count.

**`workbench/layout_editor.py`** (new in S5) owns the owner's local layout
editor: `compose_layout`, `render_layout_editor`, `LocalLayoutEditorStore`,
`LayoutEditorOffer` / `LayoutEditorSession` / `LayoutEditorDocument`,
`LayoutEditorError`. Saving appends an immutable revision and points the ticked
parts at it: a local decision, no model call, no cost.

**Other mode consumers**, each refusing explicitly through
`extract.refuse_unbound_layout_mode`: `cli.command_extract`,
`cli_extract_batches.run_batch_extraction`, `workbench/dispatch.parse_dispatch_form`,
`workbench/assistant_batch_surface.plan_batch`. `workbench/render.py` exposes
`OFFERED_EXTRACTION_MODES` derived from its own radios and
`workbench/server._wants_mode` reads that, so a URL-supplied
`?mode=table-layout` falls back to the default.

**`application/card_change_staging.py`.** `_validate_replacement_shape` refuses
`source_forms: None` and `{"columns": [], "cells": {}}`, with a message that
states the canonical fact and scopes its route:
`janki study curate JOB --record ID --drop-table` drops the table from a study
job's **staged copies**, reaching the card only while its proposal is still
unpromoted. It promises no canonical deletion path. `_changes` reads
`to_dict().get(field)` because the new field is sparse.

---

## 5. Seams that do **not** exist yet

These are contract-agreed S6 work. Do not cite them as existing symbols, and do
not let a test import one before it is written:

- `application/study_finish.py` — the whole coordinator, its authority and
  per-phase intent shapes.
- `io.records_json_text(records)`. **Not present at `0e53007`**:
  `io.save_records_json_locked` (`io.py:4125-4152`) still inlines
  `json.dumps(payload, ensure_ascii=False, indent=2)`. Extract it **with
  `allow_nan=False`** and have `save_records_json_locked` call it — a literal
  extraction would drop the package planner's non-finite refusal
  (`deck_package.py:700` already passes `allow_nan=False`, and
  `_plan_vocabulary_revision_unlocked` refuses unless
  `revision.text == _canonical_records_text(records)`, so the synthetic text
  must come from the same function). Output is byte-identical for every record
  that can round-trip. This is artifact structure — janki's own logic, no owner
  decision. `io.py` is the integration lead's file; agree the patch order.
- `coverage._approval_payload(..., approved_at: str | None = None)` — today it
  reads `date.today()` inline (`coverage.py:402`); the freeze parameter is the
  seam.
- `staging.render_coverage_approval(...)`, `ReviewPanel.prepare(...)`,
  `decide_promotion(..., _collection_snapshot=None, _pattern_store_snapshot=None)`,
  `check_pattern_review(..., pattern_store=None)`, widened `client:`
  annotations, `character_notes.prepare_reference_facts` /
  `apply_prepared_reference_facts` / recover, `enrich.decide_enrichment(...)`,
  `deck_package`'s `reference_sha256=` and `assert_projection_realized`, frozen
  `at=` threaded into `record_added` / `record_source_seen` / peers, and the
  frozen keyed `DictionaryFactBook`.
- The `janki study review` / `coverage` / `disposition` / `finish` commands and
  their Assistant controls.

---

## 6. First actions, before any S6 code

### 6.1 Refresh the interface mapping against final S5

`/tmp/janki-study-s6-existing-writers.md` maps every existing writer entrypoint,
the date-bearing bytes by owner, the smallest added seams and the settled
contract contradictions — but it was written **before** S5 landed. Its **§8
lists 12 concrete checks** to re-run against the real committed S5 symbols:

1. line drift at the pinned base; 2. `decide_promotion`'s parameter list and
gate order; 3. `execute_promotion`'s entry; 4. lock order and the concrete
staging-mutation lock holder; 5. `staging.py`'s prepare/apply surface;
6. `source_forms` transport through serializer → merge → exporter → preview;
7. `io.records_json_text` (see §5 — it does not exist); 8. job CAS signatures
and the whitelisted `choices` keys; 9. `extract.MODES` and
`staging._validate_prompt_provenance`; 10. `assistant_context.ResourceKind`'s
`"study_job"`; 11. `deck_package.py`'s reference-input sites; 12.
`plan_targeted_audio_revision` and the `before_paid_dispatch` seam.

A ready task description for that refresh is
`/tmp/janki-study-s6-interface-refresh-task.md`, whose output is
`/tmp/janki-study-s6-interface-contract.md`. Both are conveniences (§11); if
they are gone, run the same 12 checks yourself against the committed code and
the contracts.

### 6.2 Record the frozen interface handoff before branching

Write down exact argument names, result shapes, module ownership and small
synthetic input/output examples **before** branching P/E/B. Distinguish existing
symbols from new agreed contract symbols, and claim no signature that does not
yet exist. Plan §10.3 makes this ordinary engineering coordination — it does not
ask the owner to approve interfaces.

### 6.3 Verify and apply DESIGN amendment F before the behaviour it authorizes

Plan §10.3 wave 4: F precedes the affected behaviour. F is **prepared and
literally verified but not applied**. The patch is
`/tmp/janki-study-s6-design.patch`
(sha256 `d45648c96dde224b687599e8b4138dcb03122f1fdbf0114e1ce1e2cfe035dc9c`),
taking `docs/DESIGN.md` from
`6975958572e6c0147df8037a459e11d556757aca4923dbd6e29f2518aa1186c4` to
`a0ed40f4345feed68399e5b3e350ac61d1fda4990f28ef9d1c09ed6585d0ceae`, with
`/tmp/janki-study-s6-design-verification.json` and
`/tmp/janki-study-s6-design-handoff.md`. That handoff's §2 records the one
judgment call: F replaces the **W7-gap sentence** ("Extending the same
receipt-backed **Apply and finish** contract to that flow remains W7 work…"),
not the preceding paragraph's receipt/resume design sentence. Re-verify the
starting DESIGN bytes, commit F on its own, then land the behaviour. If the
patch file is unavailable, apply amendment F from plan §11 by hand to the same
sentence and verify the result the same way.

---

## 7. S6 structure and ownership

Three writer packages in isolated worktrees, plus you as integration lead
(plan §10.3). **No two workers edit one file.** Paths are under
`src/japanese_anki/`.

**S6-P — review and promotion:** `staging.py`, `workbench/review.py`,
`application/coverage.py`, `application/assistant_staging_review.py`,
`application/assistant_promotion.py`, `application/promotion.py`, plus the
`application/promotion_action.py` annotation widening.

**S6-E — reference facts and enrichment:** `enrich.py`,
`application/enrichment.py`, `application/character_notes.py`, `kanji.py`,
`jpdb_kanji.py`, and the frozen `DictionaryFactBook` in one owned module.

**S6-B — package preparation and recovery:** `application/deck_package.py`.

**Integration lead owns:** `application/study_finish.py`, `card_preview.py`,
`docs/CARD_DESIGN.md`, `io.py`, `ai_schema.py`,
`application/assistant_context.py`, `application/assistant_staging_actions.py`,
`workbench/assistant.py`, `workbench/assistant_adapter.py`,
`workbench/assistant_packages.py`, `cli.py` / `cli_study.py`, and the finish's
audio coordination. `application/assistant_staging_actions.py` sits in S6's file
list but in no package row: S6-P hands off the `coverage.py` patch and the lead
applies any matching call-site edit (expected: zero, since that seam is purely
additive).

**Integrate all three real writer packages before connecting the `study_finish`
coordinator and the Assistant/CLI controls.** No production stub, placeholder
writer, alternate path or compatibility shim exists to make an early merge
possible; fakes live only in tests. No new finish action is exposed until every
phase has its real writer and recovery proof. If a shared signature or wire
shape moves, stop dependent callers, update the handoff, rebase.

**Settled conclusions to preserve** (from the mapper and the contracts): the
finish uses `promotion.py`'s prepared writer entrypoints, never the Assistant
broker's live re-planner (§4.7); review then coverage compose **one** final
payload per staging path; references prepare from post-promotion records into
`kanji_file` and `jpdb_readings_file` only — **no `kanji_notes` entry, no
character note, no kanji card**; every frozen ledger date replays exactly across
a next-day resume; strict package comparisons survive; `source_forms` is
preserved untouched through projection, promotion, enrichment and the archive.

---

## 8. S6 obligations that are easy to fake — pin each to a named mutant

Every new test is named for the behaviour, names one production mutant, is
proven to fail against it, and the mutation is restored.

- **Sentence audio defaults on**, independently of word audio. Only an explicit
  owner control or CLI choice saves an opt-out; an omitted model option can
  never become `false`. The default is a proposal, not spending authority, and
  the finish confirmation still enumerates exact requests with provider/model,
  billing and **separate** word/sentence counts.
- **Derive the expected slot census independently of the audio plan's clip
  list**, from every nonblank example `japanese` in the final post-enrichment
  records. Slot count and unique-clip count are different numbers; an empty
  request list must not pass because every selected clip is "current".
- **Every expected enabled slot is linked canonically** — including several
  slots served by one clip — before `audio_complete`.
- **Each exported sound reference resolves inside the actual APKG** to its
  proven bytes: check stored field bytes and packaged media, not aggregate
  counts or GUID sets. Word clips can never satisfy sentence coverage.
- **The final HTML preview actually plays** the sentence clips its cards draw,
  using the real card rendering; pending and owner opt-out states stay explicit
  and distinct. Preview and published APKG are two artifacts and two separate
  proofs.
- **Recovery is finite and request-bound**: resume from the durable per-phase
  intent's before/after component vector, never a persisted phase string; an
  unknown paid outcome is never resent; removed evidence refuses rather than
  re-billing.
- **An unpaid VOICEVOX clip has no paid journal operation.** Its completion
  proof is bound request/profile + target + actual byte hash + ledger currency;
  the paid reservation and accounted operation are an additional field, not a
  requirement.
- **Source forms create no audio requests** — printed cells never reach the
  sentence transport.
- Four counts stay four numbers and are never summed (contracts §7.12, §9.6).

---

## 9. S7

Offline synthetic acceptance: the whole plan §1 Assistant journey plus the CLI
proven equivalent over the same services, with fake providers and temporary
repositories, then `make gates` on the integrated checkout. **No live private
PDF and no paid run is required**, and none is authorized here.

Inputs are prepared under `/tmp/janki-study-s7-prep/` (fixture bank, helpers,
`tools/check_bank.py`), with `/tmp/janki-study-s7-prep-handoff.md` (pinned
counts, `INTEGRATION_POINTS.md` §B–§E, open questions R1 and C1–C5),
`/tmp/janki-study-s7-prep-validation.md`,
`/tmp/janki-study-s7-helper-probe-report.md` (41/41 probes, two helper defects
found and fixed) and `/tmp/janki-study-s7-lead-notes.md`. The bank self-check
and all 41 helper probes have run; the latter exercised the existing S0–S3
interfaces with fake transports. The complete S7 journey and `s7_assertions`
acceptance helpers remain unexecuted. Those preparation results do not
establish S7 acceptance.

Run the bank self-check **first**, then resolve the integration points against
the real S5/S6 handoffs, then write the scenario modules:

```
/Users/brandonivers/Development/ankigen/.venv/bin/python \
  /tmp/janki-study-s7-prep/tools/check_bank.py \
  --repo /Users/brandonivers/Development/ankigen
```

`--repo` takes the janki checkout whose `src/` and live
`extract.candidate_schema()` the responses are validated against; without it the
tool runs the structural level only. It refuses with exit 2 if that path has no
`src/` directory or if pydantic (the `ai` extra, installed by `[dev]`) is
missing — run it with a checkout's own `.venv/bin/python`, and substitute your
worktree's path for both arguments when you work outside the primary checkout.
It writes nothing and imports no provider.

Playback proof: the local environment has Playwright in the dev extra and
Chromium installed under `~/Library/Caches/ms-playwright`;
`tests/test_card_preview.py::_launch_installed_chrome` and
`tests/test_workbench_browser.py` show the repository's real-browser launch
pattern. **An installed browser failing to launch is a failed check, not a
skip.** Verify the final HTML's audio element starts, `currentTime` advances, it
reaches `ended` with a finite duration and no media error — and prove each
embedded/exported slot's bytes independently, because one representative
playback check does not prove slot coverage. Do not reuse the 201-verbs
artifacts as fixtures or count their previous playback as this proof.

---

## 10. Operating rules

- **Every development, planning or review model launch goes through that
  checkout's `scripts/claude-subscription.py`**, Claude Opus 5 pinned, **xhigh**
  for implementation and **max** for planning/review. A subscription refusal
  stops the launch: there is no API fallback and no other provider.
- At most **3 external model workers** alongside root. Each lane gets its own
  branch, worktree and `.venv`. Root owns the single final integrated
  `make gates` per milestone; workers run focused tests plus named mutation
  proofs and stop. Do not re-run a green set to reformat a count. Use
  checkout-root commands: a shared editable venv otherwise tests the primary
  worktree and reports green for code your branch never changed.
- Code reviews run in frozen code-only snapshots that exclude `data/` and
  `dist/` entirely, with no Bash; re-review is narrowed to the findings. Never
  mutate a checkout while its tests or review run.
- Before activating prompt or Assistant-schema changes in the owner's checkout,
  check operation status read-only. Do not auto-forget operations. Complete or
  recover affected work under its original revision and existing authority
  first; if an end, retry or discard needs a new owner decision, leave that
  activation pending and preserve the evidence.
- Failing test first, then the fix. A wrong or thin **model answer** is a
  template problem, not a janki defect — expand the prompt, never add code that
  reads Japanese.
- Pre-release: no legacy paths. When an approach is superseded, delete it in the
  same change.
- Keep owner-facing updates short.
- Prefer durable repository documents and final artifact paths over `/tmp`.

---

## 11. Reference artifacts outside the repository

The `/tmp/janki-study-*` files named in this document — the S6 writer mapping
and interface-refresh task, the amendment F patch and its verification, the S7
prep bank, helpers, probe report and lead notes, the S5 evidence, review reports
and gates log — are **local convenience references**. They were backed up under
`.git/janki-study-checkpoints/2026-09-09-s5` and
`.../2026-09-10-deck-parent-rename`, which are **local recovery aids only, not
committed study stores**.

None of them is authoritative code, and none of them is acceptance. The
authoritative sources are, in order: `docs/DESIGN.md`, this plan and its
contracts, and the committed code at `0e53007`. **If one of these helpers is
unavailable, do not recreate S0–S5 work to regenerate it** — use the committed
plan and contracts and inspect the final code directly, then proceed with S6.

---

## 12. Limits of the S5 evidence — do not overclaim

These were reviewed and accepted as nonfindings. They are **not** S6
prerequisites; record them accurately if you cite S5's coverage.

- Path-only overlap blocking (a different identity sharing a staging file, or
  another job sharing it) is statically reviewed correct but **not** isolated by
  a regression mutant. Do not claim exhaustive mutation coverage.
- A resumed replan retains `supersedes` durably, but current resume messages do
  not echo that link; ordinary apply messages do.
- The contract has one `supersedes` edge. The public owner controls settle one
  choice; direct multi-choice service use records only the latest matching
  closed predecessor.
- The two path-lock acquisitions in the curation writer remain, so a
  non-curation writer can change a file between them. The exact refusal and the
  owner-bound `abandon_intent` provide recovery. This was explicitly a
  nonfinding, not an S6 prerequisite.
