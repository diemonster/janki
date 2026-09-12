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

## 5. Commit boundaries

The owner asked for a code review, a manual verification of its findings, the
fixes and a re-review at each boundary, up to four rounds, on 2026-09-10.

| # | boundary | state |
|---|---|---|
| 0 | Review and planning models moved to Claude Fable 5.1 | committed `390ee81` |
| 1 | DESIGN amendment F and corrected handoff | committed `d512904` |
| 2 | Shared serializer seam and this interface record | committed `fdcd96a`; `make gates`: 5651 passed, 800 warnings, Ruff clean, sample build; nine production mutations caught |
| 3 | S6-P — review and promotion | **complete in this revision.** §4 records its APIs. Five initial defects and two follow-up gaps are fixed with failing-first tests and 16/16 plus 9/9 mutation sweeps. Two obsolete test seams found by full gates are corrected and 2/2 further mutants caught and restored. Independent reviews 2 and 3 are clean and root-verified; final `make gates`: **5743 passed**, 800 warnings, Ruff clean, sample build (523.58s). |
| 4 | S6-E — reference facts and enrichment | pending |
| 5 | S6-B — package preparation and recovery | pending |
| 6 | `study_finish` coordinator, Assistant and CLI surfaces, `card_preview` | pending |
| 7 | S7 — the complete offline journey | pending |
