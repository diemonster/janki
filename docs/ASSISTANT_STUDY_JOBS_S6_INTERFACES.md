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

### 2.5 `enrich.DictionaryLookup` — lead ownership, landed inside S6-P

```python
class DictionaryLookup(Protocol):
    def parse(self, text, *, token_fields=…, vocabulary_fields=…,
              forced_furigana=None, encoding=…) -> jpdb.ParseResult: ...
    def lookup_vocabulary(self, pairs, fields=…, *, batch_size=…) -> list[dict[str, Any]]: ...
```

Copied verbatim from `jpdb.JpdbClient`, so `JpdbClient` satisfies it
structurally with no change. The defaults are the real module constants rather
than `...` on purpose: S6-E's fact-book key is `(method, canonical argument
wire)`, and a replay client can only canonicalize *effective* arguments if the
defaults are written down. `fields` stays positional-or-keyword because
`enrich._readings_for` passes it positionally. Not `@runtime_checkable` —
nothing isinstance-checks it.

The same seam widens the three reachable `enrich.py` annotations — `_parse`,
`_readings_for`, `dictionary_readings` — and adds `DictionaryLookup` to
`enrich.__all__`. Zero behaviour change.

**Ownership exception.** `enrich.py` is S6-E's module and §3 forbids S6-P from
touching it, but S6-P cannot annotate `decide_promotion`,
`promote.check_readings` or `promotion_action` without the type, and the replay
witness must reach `promote._dictionary_verdict → enrich.dictionary_readings`
at apply time. The Protocol and those three annotations are therefore
**lead-owned and landed inside the S6-P revision**. S6-E retains
`DictionaryFactBook`, `RecordingDictionaryClient`, `ReplayDictionaryClient` and
every behavioural change in `enrich.py`. It is an ordinary signature under
existing plan authority and needs no owner approval.

### 2.6 `ledger.Ledger.serialized_text()` — lead ownership, landed inside S6-P

```python
def serialized_text(self) -> str
```

§7.7 and §7.8 both need a frozen ledger `expected_after`, and the only
serializer was the private `Ledger._serialized`. It is **renamed**, not
aliased: `_save_locked` is the one call site and no test referenced the private
name, so there is no second entry to keep in step. Pure — reads no file, reads
no clock. `ledger.load_snapshot(path, wire)` already exists and is what
`prepare_source_promotion` uses to rebuild the exact object from
`decision.ledger_revision`.

```python
def save_under_lock(self) -> None
```

The second half of the same seam, landed in the round-1 fixes. §7.5 requires
one promotion transaction to hold canonical **and** the ledger from before its
vector is measured until after its last write, and `io.exclusive_path_lock` is
not re-entrant, so the writer inside that span cannot call `save()`. Both
public entries funnel through the unchanged `_save_locked`, and every failure
is still a `LedgerError`.

### 2.6b `status.surviving_ids_from(config, record_ids)` — S6-P

```python
def surviving_ids_from(config, record_ids: Iterable[str]) -> tuple[set[str], list[str]]
```

The owning definition of the surviving-id pair; `surviving_ids(config, records)`
is now exactly this over `record.id`, because the records contribute nothing
else. `promotion._require_deck_inputs` takes the ids rather than the records so
a prepared intent can re-ask the question with the collection ids its decision
was taken against — at recovery the canonical file may already hold the very
landing the check is guarding. No behaviour change for any existing caller.

### 2.7 `promote.py` annotations — S6-P

`promote.check_readings(..., client: enrich.DictionaryLookup | None = None)`
and `promote._dictionary_verdict(client: enrich.DictionaryLookup, …)`. §7.3
requires the widening and `promote.py` appears in no package row; it is S6-P's.
No behaviour change, and no import cycle: `promote` already imports `enrich`,
and `enrich` imports neither `promote` nor `promotion`.

The same widening now covers the rest of §7.3's reachable seam, landed in the
round-1 fixes and **type-only**:

```python
promotion_action.resolve_promotion_for_execution(..., client_factory: Callable[[], DictionaryLookup])
assistant_promotion.execute_promotion_action(..., client_factory: Callable[[], enrich.DictionaryLookup] | None = None)
assistant_promotion.execute_promotion_action_under_guard(..., client_factory=…)
promotion.plan_promotion(..., client: enrich.DictionaryLookup | None = None)
```

Every existing default still constructs `jpdb.JpdbClient(jpdb.api_key_from_env())`
— `assistant_promotion`'s fallback factory is unchanged — and no new network
path exists. `promotion.py` no longer imports `jpdb` at all.

### 2.8 `study_curation.staging_curation_guard(staging_dir)` — S6-P

```python
@contextlib.contextmanager
def staging_curation_guard(staging_dir: Path) -> Iterator[None]
```

The same lock file, taken by a caller that holds paths rather than a
`ProjectConfig`. `curation_guard(config)` is now exactly this with the staging
directory read off the configuration, so there is **one** lock and one
convention. It exists because `workbench.review.ReviewPanel` is deliberately
opened from three explicit paths — an approval binds the exact files it read —
and is nonetheless an entry that writes staged bytes (§4 below).

---

## 3. Package boundaries, unchanged from plan §10.3

| package | owns | must not touch |
|---|---|---|
| **S6-P** | `staging.py`, `workbench/review.py`, `application/coverage.py`, `application/assistant_staging_review.py`, `application/assistant_promotion.py`, `application/promotion.py`, the `application/promotion_action.py` annotation widening, plus `promote.py`'s two annotations (§2.7) and `study_curation.staging_curation_guard` (§2.8) | `io.py`, `deck_package.py`; `enrich.py` and `ledger.py` except the lead seams of §2.5–§2.6 |
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

## 4. S6-P as implemented

Signatures copied from the source, not from the plan. Where the brief and the
implementation differ, the implementation is what is written here and the
difference is named.

### 4.1 Pure renderers

```python
# staging.py
def render_coverage_approval(captured_text: str, approval: Mapping[str, Any], *,
                             replace_existing: bool = False,
                             source: str = "<captured staging file>") -> str
def render_staging_finish(captured_text: str, keep: Sequence[bool],
                          held: Sequence[VocabularyRecord], *,
                          source: str) -> tuple[str | None, int]
def render_staging_document(records: Iterable[VocabularyRecord],
                            meta: Mapping[str, Any] | None = None, *,
                            source: str) -> str
```

`_record_coverage_approval_unlocked`, `finish_staging_under_lock` and
`_write_staging_unlocked` are each now that renderer plus their existing bound
write, so every refusal and every byte is unchanged.

`render_staging_finish` returns `(None, removed)` when the prepared document is
byte-identical to the captured one, which distinguishes "no row departs" from
"write these same bytes again". It and `render_staging_document` are additions
beyond the contract's named list: §7.7 requires the live-staging and archive
components' complete after-payloads frozen before the first mutation, and both
writers were read-modify-write.

### 4.2 The frozen coverage date

```python
# application/coverage.py
def _approval_payload(decision, *, authority, reason, approved_at: str | None = None)

@dataclass(frozen=True, slots=True)
class OwnerCoverageApproval:
    staging_path: Path
    staging_revision: str
    payload: Mapping[str, Any]          # carries the frozen approved_at

def prepare_owner_coverage_approval(config, decision, *, reason,
                                    approved_at: str | None = None) -> OwnerCoverageApproval
```

`approved_at=None` keeps `date.today()` for every existing owner and model
caller. A supplied value must be exactly `YYYY-MM-DD`.
`approve_coverage_as_owner` is now `prepare_owner_coverage_approval` followed by
`staging.record_coverage_approval`, so there stays one payload builder and one
writer.

### 4.3 The prepared review

```python
# staging.py — shared by both phases
PREPARED_COMPONENT_ROLES = ("staging", "patterns", "canonical", "ledger",
                            "archive", "live_staging")

@dataclass(frozen=True, slots=True)
class PreparedComponent:
    role: str
    path: str
    expected_before: str | None     # None means absent
    expected_after: str | None      # None means absent
    after_text: str | None = None   # the complete payload, never a delta
    @property
    def writes(self) -> bool        # expected_after != expected_before
    @property
    def removes(self) -> bool
```

Four states stay distinct: *write*, *creation* (`expected_before is None`),
*deletion* (`expected_after is None`), and *no change* (the two digests equal —
**including both `None`**, a bound file that is absent and stays absent, such as
the pattern store of a project that never extracted anything). A present empty
`{}` document is **not** absence. This corrects the brief, which proposed
making both-absent unconstructible.

```python
# workbench/review.py
def review_target_paths(staging_path, *, staging_dir, patterns_path) -> tuple[Path, Path]

@dataclass(frozen=True, slots=True)
class PreparedReview:
    components: tuple[staging.PreparedComponent, ...]   # exactly ["staging", "patterns"]
    record_ids: tuple[str, ...] = ()
    review_patterns: bool = False
    expected_authority: Mapping[str, str] = {}          # record id -> exact wire value
    coverage_approved_at: str | None = None

class ReviewPanel:
    def prepare(self, *, record_ids, review_patterns,
                coverage_approval: Mapping[str, Any] | None = None) -> PreparedReview
    def submit(self, *, record_ids, review_patterns) -> ReviewOutcome          # guard outermost
    def submit_under_guard(self, *, record_ids, review_patterns) -> ReviewOutcome

def apply_prepared_review(prepared) -> ReviewOutcome
def apply_prepared_review_under_locks(prepared) -> ReviewOutcome
def recover_prepared_review(prepared) -> ReviewOutcome
def recover_prepared_review_under_locks(prepared) -> ReviewOutcome
def review_path_locks(prepared)
```

**The aggregate, which §7.2 and §7.6 require and which round 1 added.** A study
job reviews several parts and every part that marks a pattern set writes the
*same* file. One `PreparedReview` per part binds that one path at a different
after-digest, so whichever landed second would find the store at neither of its
own — and because the whole vector is prechecked, that part's *staging* half
could never land either, by apply or by recovery. The fold already consumed an
aggregate store it had no writer for.

```python
# workbench/review.py
@dataclass(frozen=True, slots=True)
class ReviewBatchRequest:
    part_name: str
    staging_path: Path
    record_ids: tuple[str, ...] = ()
    review_patterns: bool = False
    coverage_approval: Mapping[str, Any] | None = None
    expected_revision: str | None = None

@dataclass(frozen=True, slots=True)
class PreparedReviewPart:
    part_name: str; source: str
    staging: staging.PreparedComponent          # this part's whole staging payload
    record_ids: tuple[str, ...] = ()
    review_patterns: bool = False
    expected_authority: Mapping[str, str] = {}
    coverage_approved_at: str | None = None

@dataclass(frozen=True, slots=True)
class PreparedReviewBatch:
    parts: tuple[PreparedReviewPart, ...]       # ascending published part name
    patterns: staging.PreparedComponent         # ONE final payload for the store

@dataclass(frozen=True, slots=True)
class ReviewBatchOutcome:
    parts: Mapping[str, ReviewOutcome]

class PartialReviewBatchError(ReviewPanelError):     # .outcome: ReviewBatchOutcome

class ReviewPanel:
    def prepare_part(self, *, part_name="", record_ids, review_patterns,
                     coverage_approval=None,
                     retain_marked_patterns: bool = False) -> PreparedReviewPart
    def validate_actions(self, record_ids, review_patterns, *,
                         retain_marked_patterns: bool = False) -> tuple[str, ...]
    def render_reviewed_pattern_store(self, captured_text: str) -> str
    @property
    def pattern_selectable(self) -> bool   # lineage only: warning-free, entry present
    @property
    def pattern_marked(self) -> bool       # the captured entry is already reviewed
    @property
    def pattern_reviewable(self) -> bool   # selectable and not yet marked (unchanged)

def prepare_review_batch(requests, *, staging_dir, patterns_path,
                         collection_name="") -> PreparedReviewBatch
def apply_prepared_review_batch(batch) -> ReviewBatchOutcome
def apply_prepared_review_batch_under_locks(batch) -> ReviewBatchOutcome
def recover_prepared_review_batch(batch) -> ReviewBatchOutcome
def recover_prepared_review_batch_under_locks(batch) -> ReviewBatchOutcome
def review_batch_path_locks(batch)

# application/assistant_staging_review.py
@dataclass(frozen=True, slots=True)
class PreparedStagingReviewBatch:
    repository_root: Path
    batch: PreparedReviewBatch
    @property
    def fingerprint(self) -> str

def prepare_staging_review_batch(config, requests) -> PreparedStagingReviewBatch
def apply_prepared_staging_review_batch(config, prepared) -> ReviewBatchOutcome       # takes the guard
def apply_prepared_staging_review_batch_under_guard(config, prepared) -> ReviewBatchOutcome
def recover_prepared_staging_review_batch(config, prepared) -> ReviewBatchOutcome     # takes the guard
def recover_prepared_staging_review_batch_under_guard(config, prepared) -> ReviewBatchOutcome
```

`prepare_review_batch` takes every path's lock at once, opens each panel under
them — so every part sees the **same** captured store bytes — and composes the
selected marks one after another over that one capture:
`patterns.render_reviewed_update` once per selected entry **whose mark is
false**, over the accumulating text. A part whose choice is false contributes
nothing, which is not an unreview. If no mark changes, the component is bound
and unwritten, and an absent store stays absence (`expected_before is None`)
rather than becoming a present `{}`.

**A selected mark that is already `true` is retained, not refused** — §7.2's
own words, "apply it once per selected source entry whose mark is false, leave
already-true marks unchanged". An ordinary batch mixes a part whose grammar an
earlier pass reviewed with one still waiting; refusing the whole preparation
for the settled part loses the part that still needs its mark. The owner's
choice stays exactly as the owner made it in the durable intent
(`PreparedReviewPart.review_patterns` is not normalized to false), it
contributes no write, and the composition simply skips it — so the raw renderer
keeps its own refusal of an already-true update and is called only where the
mark changes. `prepare_review_batch` passes `retain_marked_patterns=True`;
nothing else does, so **the one-panel page keeps its display-only refusal**
("The pattern set is already reviewed and display-only") from `submit`,
`prepare` and `validate_actions` alike. The lineage half of the check is
unchanged for both: an entry whose store lineage does not match the staging run
is not selectable at all.

Apply and recovery measure **every** distinct path before any write, then write
the parts in order and the one store payload last. A part that landed before a
later failure is reported in `PartialReviewBatchError.outcome`. A completed
apply returns exactly `PreparedReviewBatch.outcome`: each part reports its own
owner-selected mark, including one whose entry was already `true` and therefore
needed no write.

**There is one low-level writer.** `PreparedReview.as_batch()` turns a single
panel into a batch of one and `_apply_prepared_review_under_locks` runs it
through the aggregate writer, mapping `PartialReviewBatchError` back to
`PartialReviewError`. The ordinary `submit` keeps its exact outcomes, refusal
messages and no-op return; a batch of one says "staging file" in a refusal and
a batch of several names the part.

Composition inside one staging path is **authority, then coverage**:
`render_coverage_approval(render_example_authority_updates(captured, …), …)`.
The coverage renderer performs a whole-document ruamel dump, so the two do not
commute byte-wise. One final after-payload per path; no intermediate
review-only document is ever published.

`prepare` accepts a selection that decides nothing — empty ids, false pattern
choice, no coverage — and returns both components bound and unwritten. A study
job folds every part it owns, including one an earlier pass already resolved,
and the alternative was fabricating an owner decision nobody made. `submit`
keeps its existing no-op return for the same input.

**Guard.** `submit` takes `study_curation.staging_curation_guard(self.staging_dir)`
outermost, ahead of the two sorted path locks, so the ordinary workbench review
and the Assistant's confirmed review hold §6.5's coordination lock like every
other staged-effect entry. A caller already inside the guard uses
`submit_under_guard`. The import is deferred to call time, the way
`promotion._pending_curation` is, because the curation service reaches the
application aggregate and `workbench.review` sits below it in the import graph.

```python
# application/assistant_staging_review.py
@dataclass(frozen=True, slots=True)
class PreparedStagingReview:
    repository_root: Path
    staging_path: Path
    part_name: str
    source_name: str
    review: PreparedReview

def prepare_staging_review(config, staging_path, *, part_name="", record_ids,
                           review_patterns, coverage_approval=None,
                           expected_revision: str | None = None) -> PreparedStagingReview
def apply_prepared_staging_review(config, prepared) -> ReviewOutcome            # takes the guard
def apply_prepared_staging_review_under_guard(config, prepared) -> ReviewOutcome
def recover_prepared_staging_review(config, prepared) -> ReviewOutcome          # takes the guard
def recover_prepared_staging_review_under_guard(config, prepared) -> ReviewOutcome
```

Every entry that writes calls `_bound_to`, which re-resolves **both** component
paths through the *live* configuration via `review_target_paths` and refuses
anything else, before effects. A repository root is not the binding: two
configurations in one root can name different pattern stores and different
active staging directories, and a substituted target holding byte-identical
content passes every digest check. The comparison is lexical so an alias cannot
stand in for the reviewed file.

### 4.4 The projection and the fold

```python
# application/promotion.py
def project_source_extraction_review_promotion(
    config, staging_path, *, expected_revision, review_record_ids, review_patterns,
    pattern_store, coverage_approval, collection, collection_revision, witness,
) -> PromotionDecision

@dataclass(frozen=True, slots=True)
class SourcePromotionPart:
    part_name: str
    staging_path: Path
    expected_revision: str
    review_record_ids: tuple[str, ...] = ()
    review_patterns: bool = False
    coverage_approval: Mapping[str, Any] | None = None

@dataclass(frozen=True, slots=True)
class SourcePromotionProjection:
    part_name: str; staging_path: Path; staging_sha256: str
    state: str; gate: str
    expected_before: str | None; expected_after: str | None
    landed_ids / held_ids / excluded_ids / archive_retry_ids: tuple[str, ...]
    decision: PromotionDecision | None      # in memory only, never serialized

@dataclass(frozen=True, slots=True)
class SourcePromotionFold:
    parts: tuple[SourcePromotionProjection, ...]
    canonical_digests: tuple[str | None, ...]      # one more entry than there are parts
    records_after: tuple[VocabularyRecord, ...]
    canonical_text_after: str | None

def fold_source_extraction_promotions(config, parts, *, collection,
                                      collection_revision, pattern_store,
                                      witness) -> SourcePromotionFold
```

Order is ascending `part_name` under plain `str` comparison. The carry advances
**only** for `state == "lands"`; every other state carries the previous records
and text and reports `expected_after == expected_before`, so the empty `merged`
of a non-landing return never enters the chain. An absent collection is `None`
at every checkpoint until the first landing, which is its own state and not an
empty collection. A `blocked` projection stays a blocker and the canonical state
is carried past it.

The projection applies `ReviewPanel.submit`'s own example flags — every nonblank
Japanese example on a selected row, with **no** `source.type == "extract"`
filter, unlike the two neighbouring `project_*` helpers — because `submit` is
the writer the finish actually runs.

### 4.5 Injections and the projected refusal

```python
def decide_promotion(config, staging_path, *, source="",
                     client: enrich.DictionaryLookup | None = None,
                     skip_reading_check: bool | None = None,
                     _record_snapshot=None, _collection_snapshot=None,
                     _pattern_store_snapshot=None) -> PromotionDecision
def check_pattern_review(config, meta, run_id, *, pattern_store=None) -> None
def _complete_pattern_only_review(..., pattern_store=None, precheck=None)
```

`PromotionDecision.projected` is `True` for **any** of the three injections, one
flag for all of them so a caller cannot supply the quiet one and reach the
writer. The refusal — `[promotion-projection-not-executable]` — is the first
statement of `prepare_source_promotion`, and `execute_promotion` is exactly
`prepare` then `apply`, so it precedes the repository compare, the `is_blocked`
re-raise, the curation recheck and therefore the `nothing`, `pattern_only`,
`archive_retry` and `nothing_lands` branches, each of which can archive or
delete a live file.

### 4.6 Prepare / apply / recover

```python
@dataclass(frozen=True, slots=True)
class PreparedSourcePromotion:
    part_name: str
    source: str
    staging_path: str
    projected_state: str                 # a DECISION_STATES member, never "blocked"
    reading_check: str                   # a READING_CHECKS member
    landed_ids / held_ids / excluded_ids / archive_retry_ids: tuple[str, ...]
    archive_name: str | None
    receipt_id: str | None
    archive_meta: Mapping[str, Any] | None
    ledger_dates: Mapping[str, str]      # added_at / seen_at / enriched_at, one ISO day
    retry_rows: tuple[Mapping[str, Any], ...]   # the exact rows this part prunes
    existing_ids / stored_ids / unreadable_decks: tuple[str, ...]
    deck_revision: str                   # "" only where the state never read the decks
    components: tuple[staging.PreparedComponent, ...]   # canonical, ledger, archive, live_staging
    @property
    def fingerprint(self) -> str
    def to_dict(self) / from_dict(cls, raw)

@dataclass(frozen=True, slots=True)
class PromotionRecovery:
    part_name: str
    state: PromotionExecutionState
    already_complete: tuple[str, ...]
    finished: tuple[str, ...]
    receipt_id: str | None
    archive_path: Path | None
    landed_ids: tuple[str, ...]

def prepare_source_promotion(config, decision, *, part_name="",
                             pattern_store=None, now: date | None = None) -> PreparedSourcePromotion
def apply_prepared_source_promotion(config, prepared, *, decision=None,
                                    witness: enrich.DictionaryLookup | None = None,
                                    ) -> PromotionExecutionResult
def recover_promotion_intent(config, prepared, *,
                             witness: enrich.DictionaryLookup | None = None,
                             ) -> PromotionRecovery                            # takes the guard
def recover_promotion_intent_under_guard(config, prepared, *,
                                         witness: enrich.DictionaryLookup | None = None,
                                         ) -> PromotionRecovery
def execute_promotion(config, decision) -> PromotionExecutionResult
```

`reading_check` is an addition beyond the brief's shape and it is required: a
resume re-decides the part, and replaying an `explicit_skip` decision as a
consulted one — or the reverse — asks a different question from the one whose
answer the owner approved. It grants nothing; the decision it reproduces was
already authorized and the fresh outcome sets are compared regardless. A
`consulted` intent resumed without a witness refuses with
`[promotion-intent-witness-required]`.

`execute_promotion(config, decision)` keeps its exact signature and return type
and is `apply_prepared_source_promotion(config, prepare_source_promotion(config,
decision, part_name=decision.source), decision=decision)`. `cli.command_promote`,
`workbench/server.py` and `assistant_promotion` are unchanged and exercise the
prepared path on every run. There is one writer, not two.

**Apply's order, which is the whole of §7.7's recovery contract:**

1. the pending-curation barrier, rechecked under the caller's guard — an
   unreadable job document refuses with promotion's own tag rather than being
   skipped, because a skipped barrier is a lifted one;
2. `_bound_intent`: the live review is a direct member of *this* configuration's
   active staging directory with a staging suffix, the part and source are
   named, the frozen archive name is a plain staging filename, and a landing
   carries the deck-input binding every later proof and comparison reads — an
   intent missing it is `[promotion-intent-invalid]` here, before the
   repository is consulted, on both entries;
3. `_precheck_components` over the **entire** vector while every one of its
   paths is locked, each role's path re-resolved from the live configuration —
   one component at neither digest refuses the whole part and names the
   component and both digests;
4. a part whose own writes have begun is **finished from its intent** —
   `_finish_intent_under_locks`, the same function `recover_promotion_intent`
   calls — never re-planned, because a fresh decision reads those writes as
   somebody else's and either refuses or describes a different transaction;
5. only a genuinely unstarted part is decided again, with the recorded
   dictionary disposition, the replay witness and the **frozen day**, and
   `_assert_intent_matches` compares state, all four disclosed id sets, the
   archive selection, the receipt, the role set and *both* sides of every
   component binding;
6. the writer's own locks are then taken by `_finish_record_review` /
   `_complete_pattern_only_review`, whose `precheck` callback re-measures the
   whole vector at the one moment nothing else can move it, and additionally
   requires an unstarted vector on a resumed pass.

Step 3's locks are released before step 6 takes its own: `exclusive_path_lock`
is not re-entrant, so holding them across the writer would deadlock. Nothing is
written between the two measurements, and step 6 refuses anything that moved.

**Lock order, and how long each lock is held.** §7.5's order is live staging →
archive → deck directory → canonical → ledger, under the shared curation guard.
`_intent_lock_paths` returns exactly that sequence — sorting the bound paths by
name instead produced its *reverse* on an ordinary layout — and a resume holds
all of them through its proof and its writes. On the writer's path
`_finish_record_review` takes the first two itself and `precheck` joins the rest
before it measures anything, keeping them until the apply returns; that is why
`commit_canonical_state` no longer takes the deck lock and calls
`io.save_records_json_locked` and `Ledger.save_under_lock`. A deck proof
released before the canonical write it authorizes describes a repository a
cooperating deck writer may already have changed.

**Recovery makes the same unstarted/started distinction, because §7.7 does.**
A part whose own writes have begun is finished from its frozen payloads and is
never re-decided — steps 1–4, exactly as before. A part at which **nothing** has
been written is not an interrupted transaction at all: there is no classification
a fresh decision could corrupt, and it goes through step 5's re-decision, the
same `_redecide_for_intent` a resumed apply calls, with the recorded dictionary
disposition, the frozen day and the replay `witness`. Recovering such a part
straight from its payloads was a way to apply a promotion whose owner
judgements had since moved by resuming it instead of applying it: a pattern-only
intent whose store entry stopped being reviewed archived the set and deleted the
live review, where the ordinary apply refused with `[patterns-unreviewed]` and
touched nothing. A `consulted` part resumed with no witness refuses with
`[promotion-intent-witness-required]` rather than silently asking the offline
question or fetching a fresh answer. The re-decision runs with the component
locks released — `decide_promotion` reads those same paths — and one fresh
`_intent_locks` acquisition then re-measures the whole vector and writes; a part
that became partly applied in that window is finished from its intent instead.

Both shapes end at the same proof. When the canonical component is still
*pending*, `_reprove_landing_authority_under_locks` re-proves exactly one
configured study-deck owner for every landed row, re-asks `_require_deck_inputs`
with the intent's own `existing_ids` / `stored_ids` / `unreadable_decks` /
`deck_revision`, and re-derives the receipt through the writer's own
`_new_promotion_batch`, comparing it to the frozen one — all under the deck lock
the caller still holds. An intent is a plan, not authority to publish a card,
and recovery is not a way around a check the ordinary promote applies. A deck
file edited after the decision refuses from either entry: the ordinary writer
at that canonical seam with `[promotion-input-stale]`, an unstarted resume one
step earlier with `[promotion-intent-stale] … deck configuration`, because
`_assert_intent_matches` compares those same bindings and no component digest
could — a deck file is not one of this intent's paths. The seam is still the
authority: a deck moved *after* the fresh decision agrees with everything the
comparison saw, and only `_require_deck_inputs` under the held lock refuses it.
`existing_ids` are the ids the collection contributed **before** this landing,
because by recovery time the canonical file may already hold it.

Both entries are idempotent. Completion is proven by the bound digests, never by
a live file's presence: a part that legitimately keeps its live file because
rows were held recovers by the same rule as one whose file the writer removed.

**A finished intent reports what the writer would have.** `_intent_execution_result`
derives the resumed `PromotionExecutionResult` from the intent's own payloads:
the pruned exact-retry rows from `retry_rows`, the held remainder by reading the
frozen live-staging payload, `removed` as the writer's own count of rows leaving
the live file, and `empty_live_retry` from those retry rows rather than from the
already-archived ids — which are every row in the archive in exactly the
empty-live case. Merge outcomes and the ledger counters are deliberately absent:
they describe what the writing pass did, and inventing them would report
bookkeeping nobody performed.

**One disposition definition.** `_disclosed_ids` is what both the fold's
projection and the prepared intent use, so §7.7's equality between them is an
identity. The intent no longer derives its own sets: that second derivation
reported every already-archived row of an `archive_retry` part as `excluded` as
well, which names a row that already landed as one still needing an owner's
exclusion or deferral.

**Dates.** `prepare_source_promotion(now=…)` defaults to `date.today()` and
freezes `ledger_dates`; apply threads them as `record_added(at=)`,
`record_source_seen(seen_at=)` and `record_enriched(at=)`, and both the resumed
re-decide and recovery replay the same day. That is what makes a next-day resume
hash to the bound `expected_after` instead of refusing at a third state.

### 4.7 What S6-P does **not** expose

No finish control, CLI flag or Assistant action. Per §3, nothing surfaces until
every phase has its real writer and recovery proof. The finish coordinator will
call `prepare_source_promotion` / `apply_prepared_source_promotion` directly —
**not** `assistant_promotion.execute_promotion_action`, which re-plans a live
Assistant resource and would discard the bound intent.

---

## 5. S6-E as implemented

Signatures copied from the source, not from the plan. Where the brief and the
implementation differ, the implementation is what is written here and the
difference is named.

### 5.1 The keyed fact book, in `enrich.py`

Contract §7.3 says "New in `enrich.py`", and that is where these live — beside
the `DictionaryLookup` Protocol §2.5 landed, so a replay client can canonicalize
*effective* arguments against the real module defaults, and so nothing has to
import a new module to raise `EnrichError`.

```python
FACT_BOOK_METHODS: tuple[str, ...] = ("parse", "lookup_vocabulary")

@dataclass(frozen=True, slots=True)
class DictionaryFactConflict:
    method: str; arguments: str; responses: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class DictionaryFactBook:
    facts: tuple[tuple[str, str, str], ...] = ()   # (method, arguments wire, response wire), sorted
    def __len__(self) / __contains__(key) / keys() / items()
    def answer(self, method: str, arguments: str) -> str
    @property
    def fingerprint(self) -> str
    def to_dict(self) / from_dict(cls, raw)

class RecordingDictionaryClient:
    def __init__(self, client: DictionaryLookup)
    calls: list[tuple[str, str]]                   # every key asked for, repeats included
    @property
    def conflicts(self) -> tuple[DictionaryFactConflict, ...]
    def freeze(self) -> DictionaryFactBook

class ReplayDictionaryClient:
    def __init__(self, book: DictionaryFactBook)
    book: DictionaryFactBook
    calls: list[tuple[str, str]]
```

Both clients implement `DictionaryLookup` with the Protocol's exact signatures
and defaults, so either satisfies `decide_promotion`, `promote.check_readings`
and every §7.8 helper with no change.

**The key** is `(method, canonical argument wire)`. The wire is
`json.dumps(effective, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
allow_nan=False)` over the *effective* arguments:

| method | keyed arguments |
|---|---|
| `parse` | `text`, `token_fields`, `vocabulary_fields`, `forced_furigana`, `encoding` |
| `lookup_vocabulary` | `pairs`, `fields`, `batch_size` |

Defaults are filled in from the Protocol's own values, so `parse("話す")` and
`parse("話す", token_fields=jpdb.DEFAULT_TOKEN_FIELDS, …, forced_furigana=None,
encoding=jpdb.DEFAULT_ENCODING)` are **one** key, and `fields` passed positionally
— which is how `enrich._readings_for` passes it — is the same key as the keyword
form. Tuples canonicalize to lists, so `[[vid, sid]]` and `[(vid, sid)]` are one
request, matching `jpdb.as_pair`. Every other explicit argument value is a
different key.

**Iterable arguments are snapshotted before the transport sees them**, and the
snapshot is what is forwarded. A generator of `pairs` or of `forced_furigana`
spans therefore reaches the live client whole; keying after the transport had
consumed it would record an empty list beside a request that carried spans.

**The response wire** is the decoded method result the client already has — not
HTTP bytes it never returns. `parse` stores
`{"tokens": result.tokens, "vocabulary": result.vocabulary}` and
`lookup_vocabulary` stores the list of row dicts, both with `sort_keys=False` so
key order is the dictionary's own. Replay rebuilds a real `jpdb.ParseResult` and
a real `list[dict]` from that wire on **every** answer, so each caller gets its
own clone and a caller's edits reach nobody. The recording client returns the
same clone, so the preview pass and the applying pass cannot diverge.

**Conflicts.** `RecordingDictionaryClient` answers live on every call — it is not
a cache. A repeated key whose answer is byte-identical is one fact; a key
answered two different ways retains **both** wires in `conflicts` and refuses at
`freeze()` with `[dictionary-fact-conflict]`, leaving the evidence intact and
re-raising on every later `freeze()`. `DictionaryFactBook.__post_init__` refuses
the same shape, and refuses a method outside `FACT_BOOK_METHODS`.

**Replay** raises `EnrichError("[dictionary-fact-missing] …")` for an unknown key
and has no transport at all, so "no phase refetches after review" is a property
of the type rather than a rule somebody remembers. `fingerprint` binds every
response as well as every key; `to_dict`/`from_dict` round-trip and `from_dict`
refuses a recorded fingerprint that does not match with
`[dictionary-fact-stale]`.

### 5.2 `enrich.decide_enrichment`, and the honest scope diagnostic

```python
def decide_enrichment(client, records, revision, *, ids=None, force_fields=(),
                      kanji_store=None) -> EnrichResult
```

`enrich_records` keeps its **exact** signature and behaviour and is now that same
decision over the same private `_decide`. The one difference is the
out-of-scope refusal, which used to claim "in the normalized file. Enrichment
reads that file only" from an entry that never opened a file:

* `enrich_records` — "*in the records this pass was given. Enrichment reads only
  those records, so an id that 'janki status --format ids' lists …*"
* `decide_enrichment` — "*in `<revision.path>`. Enrichment reads that file only,
  so …*"

The remaining advice (inline deck note, staged row) is unchanged in both. No
canonical read is fabricated and no temporary collection is written; `revision`
is the exact text and path its caller already bound, which for a study finish is
a projected post-promotion chain that is deliberately not on disk.

`source_forms` is absent from `ENRICHABLE_FIELDS` and from `AI_FIELDS`, so
neither entry can fill or overwrite one.

### 5.3 `application/enrichment.py`

```python
@dataclass(frozen=True, slots=True)
class DictionaryEnrichmentDecision:
    …                                   # unchanged fields
    projected: bool = False             # new; part of the decision fingerprint (version 2)

def plan_dictionary_enrichment_revision(
    config, client, records, revision, ids, *, force_fields=(), kanji_store,
) -> DictionaryEnrichmentDecision
```

`kanji_store` is a **required** keyword with no default: it is §7.9's frozen
projected store, and "nobody has run `janki kanji`" has to be said out loud as
`kanji_store=None` rather than defaulted into. The planner reads neither live
canonical nor the live kanji path; it keeps `_plan`'s ledger preflight, refuses a
revision whose path is not this configuration's collection with
`[dictionary-projection-unbound]`, and returns `projected=True`.

`plan_dictionary_enrichment` and `plan_all_dictionary_enrichment` are unchanged
in behaviour and return `projected=False`. Both now annotate `client` as
`enrich.DictionaryLookup`.

```python
def commit_dictionary_enrichment(config, decision, *, expected_fingerprint)          # takes the guard
def commit_dictionary_enrichment_under_guard(config, decision, *, expected_fingerprint)
```

`_require_executable` — `[dictionary-projection-not-executable]` — is the **first
statement of both**, ahead of the guard acquisition, the repository compare, the
fingerprint check and the early `nothing` return. `commit_dictionary_enrichment`
then takes `study_curation.curation_guard(config)` outermost (§7.5), and the
write holds `exclusive_path_lock` on canonical and then the ledger. Because
`exclusive_path_lock` is not re-entrant, the one writer calls
`io.save_records_json_locked` and `Ledger.save_under_lock`. `cli.command_enrich`
and `workbench/server.py` are unchanged and neither holds the guard, so both keep
using the guard-taking entry.

```python
@dataclass(frozen=True, slots=True)
class PreparedDictionaryEnrichment:
    repository_root: str; output_path: str; ledger_path: str; kanji_path: str
    record_ids: tuple[str, ...]
    force_fields: tuple[str, ...]
    enriched_at: str                                   # the frozen ISO day
    bound_values: Mapping[str, Mapping[str, Any]]      # record id -> field -> exact new value
    cleared_fields: Mapping[str, tuple[str, ...]]      # record id -> provisional marks cleared
    book_fingerprint: str = ""
    components: tuple[staging.PreparedComponent, ...]  # exactly ("canonical", "ledger")
    @property
    def changed_record_ids / cleared_record_ids / writes
    @property
    def projected_input_sha256 / projected_output_sha256    # §7.4's pair, off the canonical component
    @property
    def fingerprint(self) -> str                       # derived, so `replace()` cannot keep a stale seal
    def to_dict(self) / from_dict(cls, raw)

@dataclass(frozen=True, slots=True)
class DictionaryEnrichmentRecovery:
    state: DictionaryEnrichmentCommitState
    output_path: Path
    already_complete: tuple[str, ...]
    finished: tuple[str, ...]
    changed_record_ids / cleared_record_ids: tuple[str, ...]
    ledger_error: ledger.LedgerError | None = None

def prepare_dictionary_enrichment(config, decision, *, now=None, book=None)
def apply_prepared_dictionary_enrichment(config, prepared, *, client=None, decision=None)        # takes the guard
def apply_prepared_dictionary_enrichment_under_guard(config, prepared, *, client=None, decision=None)
def recover_prepared_dictionary_enrichment(config, prepared, *, client=None)                     # takes the guard
def recover_prepared_dictionary_enrichment_under_guard(config, prepared, *, client=None)
```

`prepare` publishes nothing, refuses a projected decision, and freezes:

* **canonical** — `expected_before = sha256(decision.output_revision.text)`,
  `after_text = io.records_json_text(result.records)`. A pass with **no** change
  and **no** cleared mark binds canonical *unwritten* rather than proposing the
  saver's normalization of bytes nobody asked it to touch.
* **ledger** — `expected_before` from the current bytes, `after_text` from
  `ledger.load_snapshot(path, wire)` plus one
  `record_enriched(kind="jpdb", model="jpdb", fields=…, at=<frozen day>)` per
  changed record, serialized through `Ledger.serialized_text()`. A **cleared-only**
  pass writes canonical and no attribution row at all, so its ledger component is
  bound and unwritten — including the absent-and-stays-absent case, which is not
  a present `{}`.

`apply` under the guard: `_bound_intent` (configuration, role order, nonempty
scope, ISO day, each role's path re-resolved from the live configuration) →
canonical and ledger locks → `_precheck_components` over the **entire** vector,
including the bound-unwritten components → a **started** intent is finished from
its frozen payloads and **never** re-planned → only a genuinely unstarted one is
re-planned. Unlike promotion's resume the re-plan runs *inside* the held locks:
`plan_dictionary_enrichment` reads canonical, the ledger and the reference store
through the bound readers and takes neither of these two path locks, so there is
no window between the classification, the decision and the write.

The re-plan is `plan_dictionary_enrichment(config, <replay client>,
prepared.record_ids, force_fields=prepared.force_fields)` — the ordinary planner,
over real canonical, reading the reference store §7.9 has already written, with
no network anywhere. A missing `client` refuses with
`[enrichment-intent-client-required]`; a client whose book fingerprint is not the
bound one refuses with `[enrichment-intent-book-mismatch]`.
`_assert_intent_matches` then compares record ids, force fields, the cleared
marks, **every** authorized field value naming both the bound and the recomputed
one, and **both** sides of each component binding. The commit is the one
`_commit_under_locks` the ordinary path uses, with `at=prepared.enriched_at`.
There is no second payload check: `_assert_intent_matches` derives the fresh
components from the same decision through the same `_prepared_from`, so the bytes
compared and the bytes written are one derivation under one lock.

`recover` makes the same started/unstarted distinction. A started pass replays
only the writes it still owes, canonical then ledger, from the frozen payloads
and the frozen day, and reports `already_complete` / `finished`. A ledger replay
that still fails is **not** an exception: it returns
`state="committed_ledger_incomplete"` with the error, which is the writer's own
split reached from the intent alone when the process died before returning it,
and it blocks a job reaching `complete` on the same rule as promotion's.

**Refusal tags.** `[dictionary-projection-not-executable]`,
`[dictionary-projection-unbound]`, `[dictionary-config-mismatch]`,
`[dictionary-plan-stale]`, `[enrichment-intent-invalid]`,
`[enrichment-intent-stale]`, `[enrichment-intent-client-required]`,
`[enrichment-intent-book-mismatch]`, `[enrichment-recovery-incomplete]`.

### 5.4 `application/character_notes.py` — reference facts

```python
REFERENCE_FILE_LABELS: tuple[str, str] = ("kanji reference store", "jpdb reading facts")

@dataclass(frozen=True, slots=True)
class MissingReferenceFact:
    character: str; store: str; detail: str
    def to_dict(self) / from_dict(cls, raw)

@dataclass(frozen=True, slots=True)
class ReferenceFactsPreparation:
    project_root: Path
    characters: tuple[str, ...]
    refresh_readings: bool
    looked_up: tuple[str, ...]
    fetched_readings: tuple[str, ...]
    missing: tuple[MissingReferenceFact, ...]
    files: tuple[ProposedFile, ...]      # exactly REFERENCE_FILE_LABELS, in that order
    fingerprint: str                     # stored, like CharacterNotesPlan's
    @property
    def changed_files(self) -> tuple[ProposedFile, ...]
    def file(self, label) -> ProposedFile | None
    def to_dict(self) / from_dict(cls, raw)

@dataclass(frozen=True, slots=True)
class ReferenceFactsResult:
    already_complete: tuple[str, ...]
    changed: tuple[str, ...]
    missing: tuple[MissingReferenceFact, ...]

def prepare_reference_facts(config, characters, *, refresh_readings=False)
def apply_prepared_reference_facts(config, preparation) -> ReferenceFactsResult
def recover_prepared_reference_facts(config, preparation) -> ReferenceFactsResult
```

Each store's own module now exposes the decoder its loader uses, so a caller
holding bytes it has already read need not read them again:

```python
# kanji.py
def parse_store(text: str | None, *, source: Any) -> KanjiStore
def load_store(path: Any) -> KanjiStore            # read_text_bound, then parse_store

# jpdb_kanji.py
def parse_readings(text: str | None, *, source: Any) -> dict[str, CharacterReadings]
def load_readings(path: Any) -> dict[str, CharacterReadings]
                                                   # read_text_bound, then parse_readings
```

`None` is the missing file both loaders still report as an empty store, and a
present but empty document is still the parse failure it was. `source` names
the path in every message, so the refusal texts — `Could not read {path}: …`,
`{path} must hold a JSON object keyed by character`, the wire-field and
per-entry messages — are unchanged, as is the malformed-saved-store refusal a
preparation raises as `CharacterNotesError`. There is one parser per format:
the loaders delegate rather than carrying a second copy.

Both preparations take that route. `prepare_reference_facts` and
`prepare_character_notes` snapshot each store once, and the text `_snapshot`
returns is what `parse_store` / `parse_readings` decode — the digest apply
compares and the store the proposal is built from come from a single read.
Reading the file a second time was a real defect rather than a tidiness point:
a store replaced and restored between the two reads left the after-payload
built on content nobody bound, and the apply's compare-and-swap accepted it
because the file was byte-identical to the snapshot, so saved entries were
lost. `prepare_character_notes` still reads `config.kanji_notes_file` through
`kanji_notes.load_notes`; that third store keeps the split read.

`characters` are single kanji, deduplicated in order — unlike a character-note
batch, which refuses a repeat, because the caller applies `kanji.kanji_in` per
record over a whole collection and a character two words share arrives twice.
Nothing here mints an identity: the check is only that each target is one kanji,
which is what both providers require of a request.

It reuses `_snapshot` (whose text it parses), `_reference`, `_facts` and
`_serialized`, and touches
exactly two paths — `config.kanji_file` and `config.jpdb_readings_file`, refused
if a configuration points both at one file. It never reads or writes
`config.kanji_notes_file`, mints no `kanji:` identity, no note and no deck.
Saved facts are reused; only missing characters are fetched;
`refresh_readings=True` re-requests exactly the named characters' JPDB pages and
**not** the KANJIDIC inventory. Raw HTML goes to `config.jpdb_html_cache` only.
The payloads come from `kanji.save_store` / `jpdb_kanji.save_readings` rendered
into a private `tempfile` scratch path, so preparation publishes nothing.

An unavailable lookup is an exact disclosed `MissingReferenceFact` carrying the
provider's own message, not a refusal and not a guess: the word cards already
exist and the hole is the owner's to decide about. The existing character-note
path is unchanged and still **refuses** a failed lookup, because a note cannot be
written without facts.

`detail` is exactly the provider's own message, with one qualifier appended and
only when both halves hold: `refresh_readings=True` *and* the facts store
already has an entry for that character. Then it reads `<provider message> The
refresh failed; the saved reading facts for this character were kept.` — a
statement about the file, which `_facts` never touched, so nothing is proposed
for it and a card still shows what was saved earlier. A character with no saved
entry, and any failure outside a refresh, keeps the bare provider message and
its existing missing-data meaning. No field, label, type or flow changes: the
store label is still `REFERENCE_FILE_LABELS[1]`.

A saved store that cannot be read is a **refusal**, not a hole and not an empty
store: preparation raises `CharacterNotesError` carrying the owning module's
message and the file's path, before any provider request, and proposes nothing.

`apply` and `recover` are the same idempotent replay, deliberately: this write is
a pure compare-and-swap over frozen bytes, so there is no classification an
interrupted pair could corrupt — only writes it still owes. Both re-prove the
binding (root, the closed label list, each path re-resolved from the live
configuration and compared lexically, the stored fingerprint, and a full
`from_dict(to_dict())` round trip), take both paths' locks in sorted-path order,
measure **both** before any write, and write only the pending one. A store
already at its after-state is safe; any third state refuses with
`[reference-facts-stale]` before anything further is written.

The low-level writer `_apply_proposed_files` is now shared with
`execute_character_notes`, whose refusal text, tolerated states and returned
`changed` labels are byte-for-byte what they were.

### 5.5 Ordering inside `enriched`

1. **Reference preparation** runs over the post-promotion collection (§7.2's
   `carry_N`), producing both stores' frozen after-texts and any missing states.
2. The **vocabulary enrichment projection** is planned next, through
   `plan_dictionary_enrichment_revision` with that frozen store as
   `kanji_store` — so the preview renders over the reference facts the owner is
   about to approve, not over the live pre-write store.
3. At apply time the **reference components are written first**, then the
   prepared vocabulary and ledger components. The enrichment apply's real
   re-plan reads the live reference store, so it must observe what step 3's
   first half wrote; running it the other way round makes the re-plan compute
   values the preview never showed, and it refuses naming both.
4. Within the enrichment component vector the order is canonical, then ledger.

### 5.6 What S6-E does **not** expose

No finish control, CLI flag or Assistant action, and no production stub,
placeholder or shim. `RecordingDictionaryClient` and `ReplayDictionaryClient` are
ordinary production types; the only fakes are in tests. The reference
preparation deliberately stops at the two reference stores — kanji notes, decks
and `kanji:` identities remain `character_notes`' existing batch and
`kanji_finish`'s.

---

## 6. S6-B as implemented

Signatures copied from the source. Two shared-file exceptions were verified and
approved against the actual code before this package started, and both are
recorded here: `exporters/anki.py` (the missing rendering seam) and
`application/audio.py` (the audio-completion proof). The verified paid-attribution
correction also changes the native `audio_cmd.py`/`tts` callbacks and the shared
ledger entry query, as described in §6.2. These are structural persistence seams;
existing finish coordinators and content are unchanged.

### 6.1 `exporters/anki.py` — one rendering, three consumers

```python
@dataclass(frozen=True, slots=True)
class RenderedNote:
    record_id: str
    fields: tuple[str, ...]          # positional, parallel to FIELD_NAMES
    tags: tuple[str, ...]            # cleaned, in note order
    media_files: tuple[Path, ...]    # absolute, this note's claims in claim order

@dataclass(frozen=True, slots=True)
class RenderedDeck:
    deck_path: Path; deck_id: int; deck_name: str; deck_description: str
    model_id: int; model_name: str
    field_names: tuple[str, ...]; card_types: tuple[str, ...]
    templates: tuple[tuple[str, str, str], ...]      # (display name, qfmt, afmt)
    notes: tuple[RenderedNote, ...]
    media_files: tuple[Path, ...]                    # deduped, sorted by name
    warnings: tuple[str, ...]
    @property
    def record_ids(self) -> tuple[str, ...]

def render_deck(deck_path, project_config, deck_config, records, *,
                allowed_missing_media: frozenset[Path] = frozenset()) -> RenderedDeck

@dataclass(frozen=True, slots=True)
class ExpandedNote:
    record_id: str
    guid: str                       # genanki.guid_for(record_id)
    card_ordinals: tuple[int, ...]  # ascending, repeats kept: genanki's own
                                    # required-field expansion of these fields
    note: Any                       # the genanki.Note the builder packages

def expand_deck(rendering: RenderedDeck, *, css: str) -> tuple[ExpandedNote, ...]
```

`_media_paths_for_records` is **deleted**. `resolve_deck_media_paths` and
`project_deck_media_paths` keep their signatures and return
`render_deck(...).media_files`; `build_deck` adds the notes `expand_deck` returns to
its deck and passes `css=_read_text(template_dir / "style.css")` into it, so the one
place a `genanki.Model` and a `genanki.Note` are built is `expand_deck` — and planning,
which calls neither, still requires neither genanki nor the stylesheet. Field rendering,
validation-before-narrowing, the public direction/output resolvers, the warnings,
`BuildResult` and every public signature are unchanged.

**Two accepted timing deltas.** `deck_config["model_id"]` and
`deck_config["deck_id"]` are now converted inside the renderer, so a deck file
carrying a number Anki cannot use is refused when its media is planned rather than
minutes later at the exporter. Valid configurations are unaffected, and an **absent**
key still means the project's default. A key that is *present* and unusable is refused
by `_deck_number` as an `AnkiBuildError` naming the deck file and the key, rather than
as whatever `int()` raises: `deck_id:` with no value is YAML for `None` and
`deck_id: [1]` is a list, both of which give a bare `TypeError` that the planning
callers — the Assistant build action, `plan_deck_package`, `card_revision_finish`, each
catching `JankiError`/`OSError`/`ValueError` — do not catch.
`_resolve_deck_records_document` stays the owner of the deck document's shape, including
its stricter non-integer `model_id` rule; it never read `deck_id` and deliberately lets
an explicit null through. Pinned by
`test_anki_build.py::test_media_planning_refuses_a_deck_number_the_build_cannot_use`,
`…::test_media_planning_names_the_deck_and_key_for_an_unusable_number` and
`test_application_deck_package.py::test_package_planning_refuses_a_deck_number_it_cannot_use`.

**The expansion is genanki's, not the enabled list.** `ExpandedNote.card_ordinals` is
`Note.cards` over the model `expand_deck` builds, which is `Model._req` over each
enabled template's `qfmt`: recognition keeps its card when `Expression` **or** `Image`
is non-empty, production when `Meanings` **or** `PartOfSpeech` is, reading when
`Expression` is. So the cards an archive carries are not one per enabled direction, and
`card_types` is not a substitute for asking. With the three shipped front templates the
two coincide for every record this project loads — `validate_record` makes a missing
meaning an error and `_string_list` drops empty ones — but an ordinary card-design edit
separates them: a production front prompting on `{{Furigana}}` legitimately expands a
record with no furigana into one card, which
`test_application_deck_package.py::test_the_receipt_counts_the_cards_the_archive_was_proven_to_carry`
builds end to end.
`test_anki_build.py::test_the_shared_expansion_follows_genanki_required_fields` pins the
rule itself against the installed genanki. `css` rides on the notetype genanki
serializes and takes no part in the expansion, so a caller that wants only the ordinals
passes `""` and compares the archived stylesheet by its own bound hash. The genanki
doubles in `test_anki_builder_contract.py` answer `cards` with one card per enabled
template, deliberately: they record what the builder wired and do not reimplement
`_req`.

**Limitation.** Expected values drawn from the owning renderer prove that the
archive faithfully carries what the renderer produced from the reviewed inputs.
They cannot catch a defect *inside* the renderer, which moves both sides
together; that stays with `test_anki_build.py` and `test_card_templates.py`. A
second independent renderer would catch it and would drift, which is why the
seam forbids one. `render_deck` also runs twice per prepare (once for the plan's
media, once for the expected values): local CPU over one deck.
The same applies to the expansion: comparing the archive against `expand_deck` proves
the archive carries the cards the builder's own notes produced, not that genanki's
required-field rule is the rule Anki would apply.

### 6.2 `application/audio.py` — the durable audio-completion proof

```python
AUDIO_COMPLETION_SCHEMA = "janki-audio-completion-proof-v1"
CLIP_EVOLUTION: Mapping[str, frozenset[str]]
ProvenSlotOrigin = Literal["reused", "recovered", "synthesized"]

class AudioProofError(JankiError): ...

def provider_plan_wire(provider: AudioProviderPlan | None) -> dict | None
def media_target_path(config: ProjectConfig, target: str) -> Path

@dataclass(frozen=True, slots=True)
class ExpectedAudioSlot:      record_id, kind, position: int | None
def expected_audio_slots(records, record_ids, *, words, examples) -> tuple[ExpectedAudioSlot, ...]

@dataclass(frozen=True, slots=True)
class ConfirmedAudioClip:     record_id, kind, target, request_input, forced_accent,
                              content_fingerprint, provider, initial_state,
                              initial_media_sha256, initial_recovery_key,
                              initial_recovery_sha256, initial_recovery_source
@dataclass(frozen=True, slots=True)
class ConfirmedAudioPlan:     repository_root, canonical_path, ledger_path, media_dir,
                              record_ids, targeted, words, examples, force,
                              word_provider, example_provider, clips
                              # clip_for(target), to_wire(), from_wire(config, raw)
def confirm_audio_plan(config, plan: AudioPlan) -> ConfirmedAudioPlan

@dataclass(frozen=True, slots=True)
class ReservedPaidAttempt:    target, operation_id, expected_request_fp
@dataclass(frozen=True, slots=True)
class ProvenPaidOperation:    operation_id, kind, model, source_file, source_sha256,
                              expected_request_fp, state: Literal["committed", "accounted"]
@dataclass(frozen=True, slots=True)
class ProvenAudioSlot:        record_id, kind, position, target, reference, media_sha256,
                              request_input, forced_accent, content_fingerprint,
                              provider, origin, paid_operation
@dataclass(frozen=True, slots=True)
class AudioCompletionProof:   schema, repository_root, canonical_path, canonical_sha256,
                              ledger_path, media_dir, record_ids, words, examples,
                              slots, fingerprint
                              # clip_count, stored_slot_count, realized_media_sha256,
                              # origin_by_media_path, slots_for(), to_wire(), from_wire()

def prove_audio_completion(config, plan: AudioPlan, *, authority: ConfirmedAudioPlan,
                           expected_slots: Sequence[ExpectedAudioSlot],
                           reservations: Sequence[ReservedPaidAttempt] = ()) -> AudioCompletionProof
def revalidate_audio_completion(config, proof: AudioCompletionProof) -> None
```

The proof binds the **whole** confirmed enumeration, not a state map:
`confirm_audio_plan` freezes the scope, the inclusion flags, both provider
profiles and every clip's request identity, provider and initial disposition
with the fixed hash that disposition implies, and proving compares the fresh
plan against all of it plus `CLIP_EVOLUTION`. A matching `current` clip in a
different voice is refused as a changed provider.

The census is independent: `expected_audio_slots` counts stored fields — a word
slot where `words` and the record has a reading, an example slot for every
nonblank `example.japanese` — and never `AudioPlan.clips`. The caller's
disclosed census is cross-checked against one re-derived from the post-audio
canonical, so under-declaring buys nothing, and revalidation re-derives it again
so an omitted slot refuses even when the payload's self-fingerprint is
recomputed. Several slots legitimately share one clip; the existing
divergent-spoken-input refusal stays in `audio_cmd.prepare_example_audio_profiles`.

Per slot the proof reads only durable state: the canonical reference
(file existence alone is never accepted), the media bytes, and
`Ledger.audio_entry_for(..., provider=, voice=, speed=, settings=)` — which is
where the voice is proven, since `content_fp` deliberately excludes it.
`audio_entry_for` returns an isolated copy of the very entry `audio_file_for`
answers with, and `audio_file_for` now delegates to it: one currency predicate,
because a paid clip's attribution has to be read from the same entry that
answered "current", not from a second, similar query.

**Money.** Only a clip the confirmation enumerated as `provider-required` **and**
`paid-network` may carry a paid attempt, and one is **required**: a coordinator
that never wires `before_paid_dispatch` gets no proof. Reuse and recovery invent
no reservation, and supplying one for them refuses.

The binding is two independently written values meeting. The dispatcher retains
`expected_request_fp` — derived from the approved request and profile before the
call left. The **paid writer** records `paid_attempt` into the clip's own WAL row
and, through commit, into its canonical audio entry: `operation_id`,
`request_fp`, `model` and `audio_sha256`, minted from the reply it is about to
persist, inside the operation's commit and therefore before the forget. Proving
requires all four to agree with the reservation, the model and the bytes read
from disk, for **every newly dispatched paid clip** — whether or not the journal
entry survives. `ProvenPaidOperation.state` classifies (`"committed"` while the
entry is present, `"accounted"` once it is not, every other state refused by
name), but neither value is ever accepted on its own.

**Two corrections to earlier descriptions of this seam.** First, the planning
report required `state == "committed"` at proving *and at every revalidation*,
which is unsatisfiable on the success path: `synthesize_journaled` ends with
`commit_result(...)` followed by `forget([operation_id])`, so a successful paid
clip's journal entry is **gone**. Second — and this is what the `paid_attempt`
record exists to fix — an earlier version of this section treated that absence as
itself the accounting, constructing the operation from the caller's reservation
whenever the id was missing. It is not: absence is equally what a refusal proven
before send, a reconciler's forget of a `failed_before_send` row, an owner's
`--forget --force` discard and a never-existent id leave behind, and meanwhile
any other authorized run may have voiced the same identity-addressed target.
A proof built on absence could therefore label another run's bytes as a
discarded attempt's accounted result. The forget decision still says the money
was dealt with; it never said *which clip* the money bought, and now the writer
does.

The writer-side seam this rests on, owned by `tts/`, `audio_cmd.py` and
`ledger.py` rather than by this package:

```python
# japanese_anki.tts
@dataclass(frozen=True, slots=True)
class PaidAttempt:            operation_id, request_fp, model, audio_sha256

class JournaledSpeechProvider(SpeechProvider, Protocol):
    def synthesize_journaled(..., persist: Callable[[bytes, PaidAttempt], _Persisted],
                             before_dispatch: Callable[[str], None] | None = None) -> _Persisted
    def reconcile_journaled(..., audio_sha256: str,
                            attach: Callable[[PaidAttempt], None] | None = None) -> None

# japanese_anki.audio_cmd
PAID_ATTEMPT = "paid_attempt"   # the ledger detail key
PersistPendingAudio = Callable[
    [ledger_mod.Ledger, str, Mapping[str, object] | None], None
]

# japanese_anki.ledger.Ledger
def audio_entry_for(record_id, *, of, content_fp, provider=None, voice=None,
                    speed=None, settings=None) -> dict | None
def merge_pending_audio(keys, *, replace=False,
                        expected: Mapping[str, Mapping[str, Any]] | None = None) -> None

# japanese_anki.application.audio — the native callback factory
def _audio_wal_persister(force: bool) -> audio_cmd.PersistPendingAudio
def _persist_audio_wal(book, keys, *, replace=False,
                       expected: Mapping[str, Mapping[str, Any]] | None = None) -> ledger.Ledger
```

The WAL callback's third argument is the exact row read before an attribution
update. Staging and adoption pass `None`; attaching a still-live captured
reply's attribution passes that predecessor. The application callback forwards
it as `{key: expected}` to `merge_pending_audio`. On an unforced run the locked
merge accepts the exact predecessor or the already-written successor, refuses
any third state (including a vanished row), and preserves unrelated durable
rows. This lets a stage interrupted before its WAL row resume through the real
application writer without another provider call. The existing explicit
`replace=True`/`--force` behavior remains separate and bypasses that comparison;
B's completion proof accepts only unforced scope.

There is one callback shape, not two: the free local branch passes `None`
beside its bytes rather than keeping a second signature alive. No ledger schema
version and no migration — `details` is free-form JSON that existing readers
already preserve, and `audio_file_for` ignores unknown fields.

**Recovery and reuse.** `reconcile_journaled` takes an `attach` callback and
offers the attempt only once the adopted bytes are proven to be that entry's own
reply, always before settling or forgetting it; a failing attach keeps both the
entry and the stage. A `--force` replacement that adopts a different valid
orphan stage at the same request key does **not** inherit the replaced row's
attempt — the same request may legitimately be voiced twice, and the second
reply's bytes are not the first call's result — so it earns a witness from the
matching live reply or has none. A clip confirmed `current` or `recoverable`
carries `paid_operation=None` and requires nothing new, and a ledger entry
written before this field existed still proves reuse and recovery unchanged.

`revalidate_audio_completion` resolves **no** provider and sends no call.
Absence of a `pending_audio` row and of a retained capture after success is the
finished state and is never read as missing proof.

**Locks.** Both entries are caller-holds: the caller must already own
`exclusive_path_lock(config.root / ".janki-audio-operation")`. Neither takes
it, neither takes the canonical or deck locks, and `io.exclusive_path_lock` is
not re-entrant.

**Scope.** The proof is evidence about artifacts. It grants no job authority and
no spending authority: a shared `repository_root`, `canonical_sha256` and
`media_dir` prove artifact scope, not that two study jobs are one, and a
coordinator must still bind the proof to its own exact phase receipt and owner
authority.

**Not centralized.** `card_revision_finish._ALLOWED_CLIP_EVOLUTION` and
`revision_finish._ALLOWED_CLIP_EVOLUTION` are **left in place**: those modules
are out of this package's scope, and changing two shipped finish modules merely
to share a constant was explicitly not authorized. `CLIP_EVOLUTION` is
therefore a third copy of the same table until the coordinator wave folds them
together. Inside `audio.py` the duplicate provider-wire builder *was* removed:
`_fingerprint` now calls `provider_plan_wire`.

### 6.3 `application/deck_package.py` — realization, preparation, publication

```python
PREPARED_PACKAGE_DIR_NAME = ".janki-prepared"
PREPARED_PACKAGE_SCHEMA = "janki-prepared-deck-package-v1"

def plan_vocabulary_deck_package_revision(config, deck_path, records, revision, *,
        media_sha256: Mapping[Path, str | None],
        reference_sha256: Mapping[Path, str | None] | None = None) -> DeckPackagePlan

def assert_projection_realized(projection: DeckPackagePlan, fresh: DeckPackagePlan, *,
                               audio_completion: AudioCompletionProof) -> None

@dataclass(frozen=True, slots=True)
class DeckPackageNote:        record_id, guid, fields, tags, card_ordinals
@dataclass(frozen=True, slots=True)
class DeckPackageInventory:   deck_id, deck_name, deck_description, model_id, model_name,
                              field_names, card_types, templates, stylesheet_sha256,
                              notes, media, deck_sha256, configuration_fingerprint,
                              output_path, stored_sentence_slots,
                              exported_sentence_slots, sentence_clips, fingerprint
@dataclass(frozen=True, slots=True)
class DeckPackageExport:      record_id, deck_stem, gaps, at
@dataclass(frozen=True, slots=True)
class DeckPackagePreparation: schema, preparation_id, projection, plan, audio_completion,
                              staged_path, staged_sha256, staged_identity,
                              expected_directory_identity,       # private stage
                              output_directory_identity,         # dist/
                              inventory,
                              output_before_revision, output_before_identity,
                              ledger_before_sha256, export_delta, ledger_after_sha256,
                              warnings, supersedes, fingerprint
                              # package_sha256, packaged_receipt(config),
                              # to_wire(config), from_wire(config, raw)

def prepare_deck_package(config, projection, *, audio_completion,
                         supersedes: DeckPackagePreparation | None = None) -> DeckPackagePreparation
def publish_prepared_deck_package(config, preparation) -> DeckPackageResult
def recover_prepared_deck_package(config, preparation) -> DeckPackageResult | None
```

**`reference_sha256`** is consulted at exactly the two
`_input(config, …, optional=True)` reads for `config.kanji_file` and
`config.jpdb_readings_file`. Any other path, a repeated canonicalized path, a
malformed hash, or a configuration pointing both stores at one file refuses. A
`None` value projects the store's **absence**, which the fresh plan has to
reproduce; it is never a wildcard. The only wildcard in the system is an
enumerated `media_inputs` path. Those bytes reach the card's character block and
not its media, so the projected records, media set and counts are unchanged by
the override — what changes is the input the realized plan must equal.

**`assert_projection_realized`** is pure: two plans and one already-revalidated
proof, no disk read. Every field is compared for equality except the constructed
`fingerprint`; `output_revision`/`output_identity` must still be the state the
projection bound, and that pair is what `output_before` records. A media path
the projection bound to a hash stays fixed; a path it bound to `None` must now
carry a real SHA-256 equal to `audio_completion.realized_media_sha256[path]`,
whose slot origin may not be `reused`. A path appearing or vanishing refuses,
naming it. The proof's `repository_root` and `canonical_sha256` must match the
plan's — an artifact-scope check, explicitly not a different-job proof. The
permissive `card_revision_finish._build_compatible` is never used, and the whole
fingerprint is never compared. `_assert_pending_audio_clear` stays a separate
gate after it, and strict `_assert_same` remains for the concrete plan from
publication onwards.

**`prepare_deck_package`** requires the caller to hold
`.janki-audio-operation`, revalidates the proof under it, then takes
`_execute_nonconjugation_locked`'s own lock set — deck dir, deck path, then the
sorted remaining `_locked_paths` — re-plans inside it, realizes the projection,
clears pending audio, and builds with the owning exporter into
`dist/.janki-prepared/<preparation_id>.apkg` with `output_expected_absent=True`.
The confirmed target and the canonical export ledger are untouched. It then
verifies the builder result against the plan, snapshots the staged bytes and
inode, binds **both** directories it will use — the private staging directory it reads
the artifact back from and, separately, the directory the confirmed target lives in —
re-plans and `_assert_same_after_output`, reads the inventory, and
freezes the export delta with one `date.today()` call whose value every later
replay uses. The two identities are distinct bindings on purpose: replacing `dist/` and
replacing `dist/.janki-prepared/` are different events, and `expected_directory_identity`
is the private one.

`preparation_id` is a fresh `secrets.token_hex(16)`, so a superseding attempt is
necessarily a new immutable record. `supersedes=` mints §7.11's free-rebuild
row: the same projection and the same proof, and the rebuilt inventory
fingerprint must equal the superseded one, which is what "from the same bound
input inventory" is checked to mean. The old intent is never edited.

**The inventory** is read from the real artifact with stdlib `zipfile` and
`sqlite3` over a private extracted copy: `collection.anki2`, the `media` JSON
manifest and its numbered members, with any other layout refused by a named
unsupported-layout diagnostic. It validates, against the owning renderer's
output over the same bytes the build consumed: the archived model and deck
identities and names, the deck description, the field names, the card templates
(`name`/`qfmt`/`afmt`), the **archived stylesheet** against the bound
`card stylesheet` template input, each note's GUID **and its exact `\x1f`-split
field bytes and tags**, each note's **card multiset** — the exact ascending template
ordinals `expand_deck` reports for that note's rendered fields, so a dropped, repeated
or added card is refused by name — each card's deck, and every packaged media name
against its bound byte hash. A GUID set, a note count, an archive self-hash or an
ordinal merely being in range is never the check: an ordinal no enabled template has is
a difference from that multiset like any other, so there is no separate bounds check
kept in step with it. A note genanki legitimately expands into fewer cards than the
deck enables is **accepted**, because the expected multiset is read from the same
expansion the builder wrote. Before those comparisons it runs the
proof-to-artifact check: for each **exported** sentence slot the note's stored
sound field must name the proven target and the packaged member must hold the
proven bytes, checked independently of word audio. Stored sentence slots,
exported sentence slots and unique clips are recorded as three numbers and never
summed.

Content only. genanki mints note and card ids, `mod`, `usn` and `col.crt` from
the build clock and the zip carries per-member timestamps, so APKG bytes are not
reproducible across builds: the artifact SHA proves *this file*, the inventory
proves *what it contains*, and the free rebuild in the recovery table has to
reproduce the second, not the first.

**Publication** revalidates the proof, re-plans and compares strictly against
the persisted concrete plan, re-verifies the staged bytes at their bound inode
inside their bound directory, **classifies the ledger before writing anything**,
publishes those exact bytes with `atomic_write_bytes_bound` bound to `output_before`
and to the prepared `output_directory_identity`, and applies the frozen
delta before→after. The ledger contract is exactly before to after: `Ledger.save`
refuses a ledger that moved, so a third state refuses and names both digests
rather than merging. The precheck is §7.5's rule that an apply verifies its **entire**
bound set before any effect: a ledger already at a third state makes the frozen delta
unappliable, and publishing first would spend the artifact write to reach a state
`publish` can no longer retry — its `_assert_same` refuses once the target has moved —
leaving only recovery able to report it. One owning classifier
(`_ledger_at_a_bound_state`) answers before/after/third for both the precheck and the
apply, which re-reads after the write because `Ledger.save` guards the baseline the
replay was computed from. Pinned by
`test_application_deck_package.py::test_publication_refuses_a_third_ledger_state_before_writing_the_target`
(target untouched, owner ledger byte-preserved) and
`…::test_publication_binds_the_output_directory_the_preparation_captured`.

**Recovery** consumes the intent and nothing else and recognizes its own output
**before** any strict re-plan. Target at `package_sha256` with the ledger at the
after-state completes; at the before-state applies the delta; at a third state
refuses, naming both. Target still at `output_before` re-plans, compares
strictly and publishes. A vanished stage with an unchanged target returns
`None`, which is the caller's cue to mint a superseding preparation — and the **whole
private directory being absent is the same vanished stage as the file being absent**,
because this scratch content is disposable. This is the same-checkout recovery case;
these checks do not establish support for rebasing a receipt into another checkout. A stage
replaced at another inode, or a private directory that **is there** under a different
identity, refuses by name
rather than publishing a file this intent never bound. `_directory_identity` therefore
lets `FileNotFoundError` through and wraps every other `OSError`, so the two outcomes
stay distinguishable by type rather than by message text. Pinned by
`test_application_deck_package.py::test_a_removed_private_staging_directory_is_a_vanished_stage`
beside the existing unlinked-file and replaced-directory nodes. Any third target revision
refuses and the external file is preserved.

**`packaged` is not `complete`.** `packaged_receipt(config)` returns the
evidence a later preview receipt is minted from — output path, package SHA,
inventory fingerprint, the whole-deck `note_count`/`card_count`/`card_types`/
`media_count`, an `audio_selection` block, the audio
proof's fingerprint, the ledger after-digest, the build warnings and any
`supersedes` link. It mints no download token and exposes no finish action.

**Two scopes, both named.** `card_count` here is `inventory.card_count` — the sum of
the per-note expansions the archive was *validated* against, note by note — and not
`plan.card_count`, which is `len(records) × len(card_types)` computed before the build.
The plan's number is the **capacity the deck's configuration allows** and stays exactly
that: the projection/fresh comparison remains strict and exact over it, and a deck is
never refused for expanding into fewer cards than its directions offer. A completed
build proof reports what the package was proven to contain, so a coordinator or preview
must display `card_count` as the archive's total and must not re-derive it from notes ×
directions. The three sentence numbers are the **job's selection**, not the deck: they
are counted from `audio_completion.slots`, so a deck record the audio phase never
covered contributes nothing to them. They therefore live in `audio_selection` together
with that proof's exact `record_ids`, which is §7.12's "every count says explicitly
whether it is the full deck inventory or the job subset" made structural rather than
conventional. `DeckPackageInventory` keeps the same three field names and the same
values; its docstrings now say which scope they are. `DeckPackageResult.card_count` from
a prepared publication is likewise the validated archive total; the ordinary
`execute_deck_package` path, which reads no archive back, still reports the plan's
number.

**Ledger digests** are taken over `Ledger.serialized_text()` rather than the raw
file, so a repository with no ledger yet has a defined before-state and every
comparison is over the same normalization the saver writes.

### 6.4 What S6-B does **not** expose

No coordinator, no finish action, no CLI flag, no Assistant surface, no preview
and no download offer. No generic receipt framework and no compatibility layer:
`DeckPackagePreparation` and `AudioCompletionProof` are concrete wires for these
two seams. No Japanese validation is added and `source_forms` is untouched. The
only fakes are in tests, which use temporary repositories throughout.

---

## 7. Commit boundaries

The owner asked for a code review, a manual verification of its findings, the
fixes and a re-review at each boundary, up to four rounds, on 2026-09-10.

| # | boundary | state |
|---|---|---|
| 0 | Review and planning models moved to Claude Fable 5.1 | committed `390ee81` |
| 1 | DESIGN amendment F and corrected handoff | committed `d512904` |
| 2 | Shared serializer seam and this interface record | committed `fdcd96a`; `make gates`: 5651 passed, 800 warnings, Ruff clean, sample build; nine production mutations caught |
| 3 | S6-P — review and promotion | **complete at `972ef32`.** §4 records its APIs. Five initial defects and two follow-up gaps are fixed with failing-first tests and 16/16 plus 9/9 mutation sweeps. Two obsolete test seams found by full gates are corrected and 2/2 further mutants caught and restored. Independent reviews 2 and 3 are clean and root-verified; final `make gates`: **5743 passed**, 800 warnings, Ruff clean, sample build (523.58s). |
| 4 | S6-E — reference facts and enrichment | **complete at `e36f638`, independently reviewed and gated.** §5 records the actual APIs. Two review rounds and root verification closed the bound-store split-read defect and corrected saved-fact disclosure. Four new correction mutants were caught; prior M22 was rerun after its anchor moved, while the other 29 prior traces remain applicable. Mutation limits remain explicit: M16 protects refusal ordering before a missing-key replay error; M26 protects the configuration/path diagnostic before fingerprint refusal. Reconstructed pre-fix runs prove old behavior, not chronological TDD. Final `make gates`: **5830 passed**, 800 warnings, Ruff clean, sample build, 516.70s. The source and test assertions stayed unchanged through the gate. No finish action is exposed. |
| 5 | S6-B — package preparation and recovery | **complete in this revision, independently reviewed and gated.** §6 records the actual APIs. Package reviews 2/3 and native paid-attribution reviews 1/2, with root verification, closed archive/count/publication defects and the production WAL recovery collision. One combined `make gates`: **5942 passed**, 1008 warnings, Ruff clean, sample build, 543.35s; all reviewed source/test bytes stayed unchanged. Focused regressions and mutation traces are retained, including initial survivors and later tests that catch them. Some mutations pin diagnostics or callback arguments rather than independent artifact rejection (including WAL M08/M10); no stronger claim is made. No finish action or final preview is exposed. |
| 6 | `study_finish` coordinator, Assistant and CLI surfaces, `card_preview` | pending |
| 7 | S7 — the complete offline journey | pending |
