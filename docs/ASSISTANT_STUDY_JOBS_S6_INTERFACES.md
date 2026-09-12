# S6 interface record

Plan §10.3 wave 4 asks the integration lead to pin the base commit, the exact
helper signatures and the prepared-payload shapes **before** the S6 writer
packages start. This document is that record. It is ordinary engineering
coordination, not an owner approval gate.

**Base.** `main` at `0e53007` (S5 behaviour), plus `f05a3e8` (amendments C/D),
`d34b12a` (owner deck-parent rename), `728f895` (S5 documentation) and the
model-configuration and DESIGN-amendment-F commits that open S6.

**Execution shape.** The owner chose **sequential implementation in the primary
checkout** on 2026-09-10: S6-P, then S6-E, then S6-B, then the `study_finish`
coordinator and the Assistant/CLI surfaces, then S7. Plan §10.3 permits this
explicitly — "When fewer workers are available, run the same packages
sequentially without changing their contracts." Package **contracts, ownership
and integration order are unchanged**; only the concurrency is. The
no-two-workers-edit-one-file rule is trivially satisfied, and the shared-file
sequencing the lead owned across lanes becomes ordinary commit ordering.

---

## 1. Handoff §6.1 — the twelve checks, re-run against committed S5

Re-run against the committed code, not against the pre-S5 mapping. The
headline: **the contracts' line citations have drifted and the symbol names
have not.** Cite symbols, never line numbers.

| # | check | result |
|---|---|---|
| 1 | line drift at the pinned base | Drifted. `application/promotion.py` is uniformly **+30** from the contract's citations; `staging.py` drifts **+75** at `render_example_authority_updates` and **+293** at the coverage writers; `application/audio.py` is **+8** at `execute_targeted_audio_locked`; `io.py` and `workbench/review.py` have **no** drift. Treat every `path:line` in the contracts as documentary. |
| 2 | `decide_promotion` parameter list and gate order | `decide_promotion(config, staging_path, *, source="", client=None, skip_reading_check=None, _record_snapshot=None)`. Gate order confirmed: archive → **`curation-pending`** → structure → coverage → patterns → rewritable → collection → readings → accounting → ledger → merge → deck. The `curation-pending` gate sits in the structural group ahead of coverage, as handoff §4.7 states. |
| 3 | `execute_promotion` entry | `execute_promotion(config, decision)`. Its first three acts are the repository-binding compare, the `is_blocked` re-raise, and the `_pending_curation` **recheck**. §7.2's `projected` refusal must go **at function entry**, ahead of all three, so it precedes the `nothing` / `pattern_only` / `archive_retry` branches. |
| 4 | lock order and the concrete staging-mutation lock holder | `study_curation.curation_guard(config)` takes `exclusive_path_lock(config.staging_dir / ".janki-curation-lock")` and is the outermost lock. `io.exclusive_path_lock` is **not re-entrant** — it opens a fresh handle per call — which is why `card_revision_finish._execute_record_locked` calls the `_under_guard` promotion entry. `study_finish` takes the guard once at its outermost entry and never nests a second acquisition. |
| 5 | `staging.py` prepare/apply surface | Present: `FieldOperation`, `PreparedStagingUpdate`, `prepare_record_update`, `apply_prepared_update`, `render_example_authority_updates`, `render_staging_update`, `render_staging_prune`. The coverage writers are `_record_coverage_approval_unlocked`, `record_coverage_approval`, `record_coverage_approval_under_lock`. **`render_coverage_approval` does not exist** — it is S6-P's to factor out. |
| 6 | `source_forms` transport | Confirmed end to end. `models.py` defines `SourceFormColumn` / `SourceFormsTable` and the sparse `VocabularyRecord.source_forms`; `io._CONTENT_FIELDS` is *derived* (`_RECORD_FIELDS` minus `id`/`tags`/`source`) so `merge_records` picks the field up with **no `io.py` edit**; `exporters/anki._conjugation_field` selects it; it is **absent from `enrich.ENRICHABLE_FIELDS`**, so no enrichment pass can write it. §7.2's "ordinary hole-fillable `_CONTENT_FIELDS` member, not enrichable" holds as written. |
| 7 | `io.records_json_text` | **Absent, as the handoff states.** `save_records_json_locked` inlines `json.dumps(payload, ensure_ascii=False, indent=2)` with **no `allow_nan=False`**, while `deck_package._canonical_records_text` passes `allow_nan=False` and wraps failures as `DeckPackageError`. §2 below settles the extraction. |
| 8 | job CAS signatures and whitelisted `choices` keys | `_WRITABLE_INTENT_KINDS = ("source_parts", "extract_batch", "retry", "curation")` — `"finish"` is in `INTENT_KINDS` but refused by name. `_WRITABLE_CHOICE_KEYS = ("directions", "part_selections", "part_layout_bindings")`; all five of `include_example_audio`, `review_flags`, `review_patterns`, `coverage_reasons`, `dispositions` sit in `_DEFERRED_CHOICE_KEYS`, each refusing with the name of the editor that does not exist yet. S6 moves all five across and adds `"finish"`. |
| 9 | `extract.MODES` and `staging._validate_prompt_provenance` | `LAYOUT_MODE = "table-layout"`, `MODES = ("table", "prose", LAYOUT_MODE)`, `TABLE_MODES = frozenset({"table", LAYOUT_MODE})`. The provenance validator derives its allowlist from `extract.MODES` plus `auto`. S6 changes neither. |
| 10 | `assistant_context.ResourceKind` | Carries `"source_part"` and `"study_job"` with discovery, `_opaque_id` minting and `snapshot` branches already registered. S6 extends the job snapshot; it adds no kind. |
| 11 | `deck_package.py` reference-input sites | `_plan_vocabulary_revision_unlocked` and `plan_vocabulary_deck_package_revision` both take `media_sha256`; the two optional reference reads are the `_input(config, "kanji reference store", config.kanji_file, optional=True)` and `_input(config, "jpdb reading facts", config.jpdb_readings_file, optional=True)` calls inside `_plan_vocabulary_revision_unlocked`. Those two sites — and no others — are where §7.11's `reference_sha256` is consulted. |
| 12 | `plan_targeted_audio_revision` and `before_paid_dispatch` | Both present in `application/audio.py`; `execute_targeted_audio_locked(..., before_paid_dispatch=...)` threads the callback down to the dispatch site. **The clip-evolution machinery the contracts cite by bare line number lives in `application/card_revision_finish.py`**, not in `audio.py`: `_ALLOWED_CLIP_EVOLUTION`, `_project_audio_references`, `_projected_package_media`, `_validate_audio_evolution`, `_reconcile_reservations`, `_reserve_paid_clip`. That module — with `kanji_finish` for `_advance` / `_write_new` — is the model `study_finish` follows. |

---

## 2. The shared seams, and who lands them

These are the only seams more than one package touches. The lead lands them
**before** the packages that consume them, so no package invents a signature.

### 2.1 `io.records_json_text` — lead, boundary 2

```python
def records_json_text(records: Sequence[VocabularyRecord]) -> str
```

Sorts by `record.id`, serializes with `ensure_ascii=False, indent=2,
allow_nan=False`, appends the trailing newline. `save_records_json_locked`
calls it. Output is byte-identical for every record that can round-trip.

**Exception contract.** It is a pure serializer and raises no `DataError`. What
comes out is what the encoder raises, unwrapped: `ValueError` for a non-finite
float — `NaN`, `Infinity` and `-Infinity` alike — and `TypeError` for a value
`json` cannot encode. A `record.to_dict()` that raises propagates as itself.
That preserves the existing saver contract: `test_io_atomic.py` pins an
unwrapped `ValueError` from `to_dict` and a `TypeError` from the encoder, with
the original file untouched in both cases. A wrapper here would break that
contract. A caller
wanting its own vocabulary catches and re-raises — which is exactly what
`deck_package._canonical_records_text`'s
`except (AttributeError, TypeError, ValueError)` does.

`allow_nan=False` is **not** a literal extraction, and that is deliberate: a
literal one would drop the package planner's non-finite refusal, and
`_plan_vocabulary_revision_unlocked` requires `revision.text ==
_canonical_records_text(records)`, so the fold's synthetic text and the
planner's text must come from the same function. A non-finite float that
`save_records_json_locked` previously wrote as invalid JSON (`NaN`) now
refuses. Pre-release rules make that breaking change free, and it is a
strengthening.

`deck_package._canonical_records_text` delegates to it and keeps its own
`DeckPackageError` wrapping, so the planner's refusal text is unchanged.

This is artifact structure — janki's own logic — and no owner decision.

### 2.2 `coverage._approval_payload(..., approved_at=None)` — S6-P

The date freeze §7.1 requires. Today the function reads `date.today()` inline.
The parameter defaults to `None` and preserves today's behaviour for every
existing caller; the finish supplies the frozen date from its prepared intent.

### 2.3 Ledger date parameters — S6-P and S6-E, already present

`record_added(at=)`, `record_source_seen(seen_at=)`, `record_enriched(at=)` and
`record_export(at=)` **all already exist** and are simply unthreaded — each
reaches `_iso_date(at=None)` and reads the clock. S6 threads them from the
prepared intent. No signature changes; the work is call-site threading, and its
proof is the next-day resume mutants of §7.13.

### 2.4 Study-job schema — lead, with the surfaces

`_WRITABLE_INTENT_KINDS` gains `"finish"`; all five `_DEFERRED_CHOICE_KEYS`
move into `_WRITABLE_CHOICE_KEYS` **together with the editors that write
them**, never ahead of them. `CHOICE_KEYS` is derived and needs no edit.

---

## 3. Package boundaries, unchanged from plan §10.3

| package | owns | must not touch |
|---|---|---|
| **S6-P** | `staging.py`, `workbench/review.py`, `application/coverage.py`, `application/assistant_staging_review.py`, `application/assistant_promotion.py`, `application/promotion.py`, the `application/promotion_action.py` annotation widening | `io.py`, `deck_package.py`, `enrich.py` |
| **S6-E** | `enrich.py`, `application/enrichment.py`, `application/character_notes.py`, `kanji.py`, `jpdb_kanji.py`, and the frozen `DictionaryFactBook` in one owned module | `promotion.py`, `deck_package.py` |
| **S6-B** | `application/deck_package.py` | everything else |
| **lead** | `application/study_finish.py`, `card_preview.py`, `docs/CARD_DESIGN.md`, `io.py`, `ai_schema.py`, `application/assistant_context.py`, `application/assistant_staging_actions.py`, `workbench/assistant.py`, `workbench/assistant_adapter.py`, `workbench/assistant_packages.py`, `cli.py` / `cli_study.py`, `application/study_job.py`, the finish's audio coordination | the three packages' writer modules |

`application/assistant_staging_actions.py` sits in S6's file list but in no
package row: S6-P hands off the `coverage.py` patch and the lead applies any
matching call-site edit. Expected: zero, since that seam is purely additive.

**No production stub, placeholder writer, alternate path or compatibility shim
is introduced to make an early merge possible.** Fakes live only in tests, and
no finish action is exposed until every phase has its real writer and recovery
proof.

---

## 4. Commit boundaries

The owner asked for a code review, a manual verification of its findings, the
fixes and a re-review at each boundary, up to four rounds, on 2026-09-10.

| # | boundary | state |
|---|---|---|
| 0 | Review and planning models moved to Claude Fable 5.1 | committed `390ee81` |
| 1 | DESIGN amendment F and corrected handoff | committed `d512904` |
| 2 | Shared serializer seam and this interface record | implemented in this revision; `make gates`: 5651 passed, 800 warnings, Ruff clean, sample build; nine production mutations caught |
| 3 | S6-P — review and promotion | pending |
| 4 | S6-E — reference facts and enrichment | pending |
| 5 | S6-B — package preparation and recovery | pending |
| 6 | `study_finish` coordinator, Assistant and CLI surfaces, `card_preview` | pending |
| 7 | S7 — the complete offline journey | pending |
