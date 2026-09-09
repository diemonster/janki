# Assistant study jobs — implementation plan

**Status: proposed / not implemented.** Written 2026-09-09 against `main` at
`55682fb`. No production code, prompt, test or data file changes with this
document; the owner authorized this planning and review loop only. Approving
the plan is not approval to implement it, and it is not consent to disclose
private source or card bytes to a paid model. Paid work is bought exactly as
DESIGN already buys it: one confirmation may enumerate one exact finite batch
and buys precisely the calls it enumerates — never one confirmation per child,
and never a re-ask on resume. A new or changed unit of paid work needs a fresh
decision; authority already recorded in the repository persists across a
restart.

`docs/DESIGN.md` leads. Where this plan and DESIGN disagree, DESIGN wins and
this plan changes. §11 holds the only amendments this plan asks DESIGN for, and
each lands before the milestone whose behaviour it authorizes.

**Path shorthand.** Bare `application/…`, `workbench/…`, `exporters/…` and bare
modules (`extract.py`, `models.py`, `io.py`, `staging.py`, `inputs.py`,
`cli.py`, `promote.py`, `enrich.py`, `ai_schema.py`, `kanji.py`,
`jpdb_kanji.py`, `card_preview.py`) resolve under `src/japanese_anki/`; test
paths under `tests/`, prompts under `prompts/`, documents under `docs/`.

**Baseline, recorded once.** `make gates` on `55682fb`: Ruff passed; pytest
reported 5 failed / 5243 passed / 762 warnings in 409.28s; the sample deck
build was not reached, because pytest stops the target first. Those five
failures are pre-existing content-dependent fixture assumptions, corrected as
S0 (§10). No production code changed to produce that baseline, and no gate
result below is claimed as already green.

**Worked example.** The delivered 201-verbs job — 18 source parts, 192
source-row occurrences, 187 unique identities — only keeps the counting rules
honest. It is never a constant, threshold or precondition; every rule is stated
over "each part" and "each identity". `data/**` and `dist/**` are outside the
scope of any review of this plan.

**Non-goals.** Automatic table, row or column detection, or any code that
infers Japanese, table meaning or what a printed label denotes; a fourth
card-writing path; a new notetype, GUID scheme or card direction; historical
content mutation, backfill or fingerprint rewrite; automatic redispatch of an
`outcome_unknown` call; a second database of what happened; widening the
automatic-repair allow-list; per-sibling isolation of stale unsent batch
children (`_revalidate_children` deliberately refuses the whole remaining
unsent set); hosted edition, AnkiConnect, mobile access.

**Review record.** [ASSISTANT_STUDY_JOBS_REVIEW.md](ASSISTANT_STUDY_JOBS_REVIEW.md)
records this plan's review history and each round's verdict.

---

This plan covers the Assistant journey, implementation milestones and proposed
DESIGN changes. The [detailed contracts](ASSISTANT_STUDY_JOBS_CONTRACTS.md)
contain sections 2–9: durable state, source preparation, card fields, captured
replies, curation, finish, media ownership and action authority. They are part
of the same proposal and must be reviewed together.

## 1. Assistant journey

One PDF of a different class from the 201-verbs source becomes a standalone
verb deck inside the Janki Assistant. The CLI (§9.3) is a fully supported
equivalent over the same application services and the same authority.

1. **Intake.** The owner attaches or selects a preserved source: ordinary
   immutable intake under `data/inbox/`, no model context, no paid call.
2. **A durable destination deck.** The batch binds one destination and
   promotion refuses a record no word deck selects
   (`require_exact_deck_ownership`, `application/assignment.py:541`, called at
   `application/promotion.py:2521`), so the deck exists first. Reusing a deck is
   a selection. A new deck is the **existing exact deck-creation confirmation**
   (`assistant_deck_creation.plan_deck_creation` → `execute_deck_creation` →
   `deck_creation.create_study_deck`), taken here rather than promised later;
   shared versus standalone is settled in that one confirmation and the scope is
   derived once and saved in the deck definition (`deck_creation._scope_id`).
   This plan promises **no fixed number of approvals**: how many the job needs
   depends on what already exists.
3. **Open the job.** A direct owner action — the Assistant's Create/Open, or
   `janki study new` — binds the parent source, the deck path and that deck
   file's sha256 and writes the local job document. That owner action *is* the
   authority for a local write, so there is **no second dialog** and no new
   confirmation ladder. When the destination is new, the job write happens
   inside the existing deck-creation action after it succeeds; the protected
   deck creation still asks exactly once, for the deck. The model may open the
   job editor and prepare a non-executing choice; it may not perform
   Create/Open/Save (§9.1).
4. **Part preparation (§3).** The owner chooses pages or regions in the inline
   visual editor — janki runs no detector and infers no Japanese — reviews a
   contact sheet of exact bytes and hashes, and publishes the parts as new
   immutable derivatives into `config.scan_inbox` under an immutable
   preparation receipt. Ordinary local work: no paid call, no original edited,
   no new approval gate. The receipt is **job-independent** (§2.2).
5. **Layout binding (§4).** The owner assigns one immutable layout revision to
   each layout-bound part, minting opaque column ids with exact printed
   witnesses and ordered display labels. A direct owner data action against the
   job document.
6. **Batch plan over durable bytes, then one paid confirmation.** Only after
   publication does `plan_extraction_batch` run, so every child's
   `source_sha256` and `request_fingerprint` describe bytes that already exist
   and can be shown. A confirmed batch carries **one scalar mode**: layout-bound
   parts go as one `table-layout` batch with the per-child layout revisions
   aligned to its sources, and ordinary parts are a separate confirmed batch
   (§4.2). One confirmation enumerates every child's source, request
   fingerprint, provider/model, billing identity, the pinned mode and layout
   revision, and the concurrency limit; then `authorize_batch` and
   bounded-parallel dispatch.
7. **Per-child recovery (§5)** where a reply arrived captured but malformed. No
   redispatch hides inside recovery.
8. **Cross-part curation (§6)** over the current edited staged values.
9. **Per-part deck assignment.** The existing `assign_cards` path
   (`assistant_assignment.plan_assignment` → `execute_assignment`) writes the
   ownership tags and, for a standalone destination, mints the scoped copy
   (`assignment._destination_record`). Extraction mints **unscoped** identities,
   so the scoped identity exists only because the owner chose this destination;
   the model never mints an identity decision. A canonical `source_forms` value
   rides onto the scoped copy with the rest of the row. Assignment happens
   **before** the final preview and before the finish authority is written, so
   that authority binds post-assignment staging bytes.
10. **Combined review of the real cards (§7).** The exact proposed normalized
    records are overlaid into one temporary collection and rendered with the
    real templates and CSS. The free reference facts the word card actually
    renders — jpdb dictionary facts, and the kanji/stroke and JPDB-reading
    references read from `data/kanji.json` and `data/jpdb_readings.json` — are
    fetched once before this review and are visible in it; the source-form
    witness table sits beside the cards as a supplementary aid. Against that
    HTML the owner saves review flags, each part's literal coverage reason, and
    any required exclusion or deferral in the job; the staging and canonical
    effects wait for step 11's one confirmation. Unresolved identity, ownership or coverage blockers
    are listed **before** the confirmation, naming every part that is not ready.
    Opening or flipping the preview approves nothing.
11. **One Apply-and-finish confirmation (§7).** The bound review flags, coverage
    decisions, promotions, reference-fact writes, selected audio, build and
    package execute under one durable finish authority. Those owner decisions
    were made in step 10; their exact writes execute under the receipt. There is
    no phase-by-phase approval ladder.
12. **Download in-thread** from a verified `complete` receipt (§9.6).

No temporary scripts, no YAML surgery, no code edits during the run, no new
notetype, no new paid writing path. Verb study is an existing-or-new
*vocabulary word profile*.

---

## 10. Milestones and verification

Each milestone lands its deletions **in the same change** — pre-release rule: no
shims, no deprecation — and any DESIGN amendment it needs lands first.

| # | milestone | depends | files | acceptance and deletions |
|---|---|---|---|---|
| S0 | Baseline fixture fixes | — | `tests/test_anki_builder_contract.py`, `tests/test_conjugation.py`, `tests/test_operations.py`, `tests/fixtures/conjugation_golden.json` | Replace the literal deck-name set with dynamic discovery through `status.deck_files` filtered by `anki.deck_kind`, keeping nonempty pairwise-disjoint ids and full canonical coverage; replace the scope-blind probe `word:assignment:assignment` with one scope-aware probe per deck matching `deck_creation._cross_selection_probe`; **delete both** live conjugation engine-shape assertions in favour of the module's synthetic golden tables plus a checked-in answer key, keeping the one live contract (every canonical `conjugations` map survives `VocabularyRecord.from_dict` unchanged); replace `operations == {}` with a load/validate/round-trip assertion accepting any valid persisted state. Acceptance: `make gates` reaches and passes the sample build, and each replacement test is proven against its named mutant **at implementation time** — no mutant is claimed proven by this document. No semantic Japanese check is added. |
| S1 | Media capabilities (§8) | S0 | new `application/deck_capabilities.py`; `application/audio.py`; `application/deck_package.py`; tests | A targeted vocabulary audio run no longer parses the character store; declared versions, inline overrides and drill owners are all protected; unknown kind refuses before any census, packaging or prune. Deletes the unconditional vocabulary parse of a `kanji` deck's `source:`. |
| S2 | Capture recovery, operation-scoped (§5) | S0 | new `application/capture_recovery.py`; `application/extraction.py`, `staging.py`; `prompts/extract-table.md`, `-auto.md`, `-prose.md`; `prompts/README.md`; `cli_extract_batches.py`, `cli.py`; tests | Keyed by `operation_id` only: no job store and no `janki study` parser exists yet, and S4's `janki study recover` later calls this same still-supported service rather than replacing it. Fixture captures of every enumerated envelope shape stage offline; non-lossless shapes refuse with exact pointers; a valid successful terminal always wins; `capture_recovery` is a recognized staging metadata key and survives the archive. |
| S3 | Source preparation (§3) + amendments A, B | S0 | new `application/source_parts.py`; `inputs.py`; `pyproject.toml`; workbench region-editor widget; `ai_schema.py`, `application/assistant_context.py`, `workbench/assistant_adapter.py`; `cli.py` (`janki source-parts`); `tests/fixtures/synthetic_table.pdf`; tests | The typed region editor needs its broker seam **now**: the closed source-editor intents (`open_source_part_editor`, recipe-token-bound `plan_source_parts`/publish) and the `source_part` `ResourceKind`, discovery and snapshot land here; S4 adds the job kind and job actions. `pyproject.toml` adds the optional `sources` extra **and `japanese-anki[sources]` to `[dev]`**, because `scripts/bootstrap.sh` installs `.[dev]` and a skip would report green for code nobody ran — the render and publication tests run for real, **no skips**. A synthetic PDF yields reviewed parts under a bound receipt published into `config.scan_inbox`; an interrupted publication resumes; rendering is confined to the worker process. Deletes `prompts/assistant-agent.md:150,153-154` and the `docs/DESIGN.md:337-338` parenthetical. |
| S4 | Study job, job actions, CLI, busy replies, deck-creation integration (§1, §2, §9) + amendment E | S1–S3 | new `application/study_job.py`; `ai_schema.py`; `application/assistant_context.py`; `application/assistant_assignment.py`; `application/extraction_batch.py`; `workbench/assistant.py`, `assistant_adapter.py`, `assistant_batch_surface.py`; `cli.py`; tests | Adds the `study_job` kind, the job intents/actions of §9.1 and the optional `job_id` of §2.4 — a jobless manifest's bytes and fingerprints are unchanged. The job uses the consistent choice/intent/outcome writers and reserves the layout and owner-decision namespaces; S5 adds `append_layout` with its datatype, and S6 adds the decision controls. A job resumes after process death with no new approval; a new destination deck is created through the existing exact confirmation before batch planning, and the local job write rides inside it; local status, preview and capture inspection answer while a batch runs; a lost backlink is rediscovered by reserved id and hash. Deletes the generic "Earlier model call needs attention" branch for the batch-blocked case only. |
| S5 | Layout, `source_forms`, cross-part curation (§4, §6) + amendments C, D | S4 | `extract.py`, `models.py`, `io.py`, `staging.py`, `exporters/anki.py`, `card_preview.py`, `application/extraction.py`, `application/extraction_batch.py`, `application/card_change_staging.py`, `application/promotion.py`, `application/study_job.py`, new `application/study_curation.py`; `application/assistant_promotion.py`, `application/card_revision_finish.py`, `workbench/assistant_adapter.py`; `cli.py`, `cli_extract_batches.py`, `workbench/dispatch.py`, `workbench/render.py`, `workbench/server.py`; `ai_schema.py`, `application/assistant_context.py`; new `prompts/extract-table-layout.md`, `prompts/README.md`, `docs/CARD_DESIGN.md`; tests | Owns the whole layout/capture-propagation/curation capability set and its broker additions, and its amendments land before any of it is enabled. `application/extraction.py` carries the layout into the saved request manifest and reads it **back from that manifest** for recovery; `cli_extract_batches.py`, `plan_extraction_batch`, retries and the one whole-input `plan_extraction` call transport aligned per-child `layouts`, retaining the cross-input staging-collision check; dispatch expectations freeze the layout, and `plan_corpus_extraction`/`revalidate_extraction_request` reproduce it on initial dispatch and resume; the exact prompt-provenance validator requires `table_layout` only for the new mode, and `coverage_block` treats both table modes identically; every `extract.MODES` consumer is updated explicitly rather than silently widened — a `table-layout` mode with no bound layout refuses at the CLI, at `workbench/dispatch.py` and on the server's offered-mode route. `append_layout` atomically appends an immutable layout revision and updates its part binding; `staging.py` gains the prepare/apply pair with explicit field/cell **delete**; the pending-curation barrier lands in **both** `promotion.decide_promotion` (plan) and `promotion.execute_promotion` (execution) under the shared coordination lock taken **before** the sorted file locks. Acceptance: the response-schema fingerprint is byte-for-byte unchanged while the request fingerprint changes; `source_forms` round-trips through serializer, merge, exporter and preview; blank versus absent holds; ordinary unlabelled extraction stays fully supported; no historical record is mutated by loading it. Deletions: none. |
| S6 | Projected review/promotion, prepared-writer finish, publication (§7) + amendment F | S5 | new `application/study_finish.py`; `application/promotion.py`, `application/coverage.py`, `staging.py`, `workbench/review.py`, `application/assistant_staging_review.py`, `application/assistant_staging_actions.py`, `application/assistant_promotion.py`, `application/enrichment.py`, `enrich.py`, `io.py`, `application/deck_package.py`, `application/character_notes.py`, `kanji.py`, `jpdb_kanji.py`; `workbench/assistant_packages.py`, `workbench/assistant_adapter.py`, `workbench/assistant.py`; `ai_schema.py`, `application/assistant_context.py`, `cli.py`; tests | Appends the finish actions, their broker schema and CLI, plus owner-only review/coverage/disposition editors and `janki study review`, `coverage`, `disposition` here; they save local choices and apply them only under the finish authority. S4 does not claim to implement these controls. Adds the source-review projection with the injected canonical chain and aggregate prepared pattern-store snapshot through `decide_promotion`/`check_pattern_review`, writer-prepared review/coverage/promotion/enrichment payloads (`coverage._approval_payload(approved_at=…)` frozen at prepare time), the `enrich.py` fact/record-snapshot decision helper, an explicit prepared `kanji_store` argument through the application projection, and frozen keyed `DictionaryFactBook`, and `deck_package` prepare/publish/recover. The package planner gains the two-store `reference_sha256` projection and `assert_projection_realized`; strict `_assert_same` remains for the resulting concrete plan. Prepared promotion/enrichment thread every frozen ledger date, and `staging.render_coverage_approval` factors the existing coverage writer into the coalesced prepare path. Free word-card reference preparation uses the prepared-facts seam in `application/character_notes.py` specified in §7.9 and binds the canonical `kanji.save_store` / `jpdb_kanji.save_readings` payloads in the authority — **no `kanji_notes` entry, no character note and no kanji card is created**. `source_forms` is preserved untouched through projection, promotion, enrichment and the archive. Acceptance: a crash at each phase boundary resumes from the durable intent rather than a lost return value; §9.6 completion holds; the package downloads in-thread. Deletes the hard-wired `kanji_finish.inspect_kanji_finish` call in `assistant_packages._resolve`. |
| S7 | Vertical-slice acceptance | S6 | tests and docs only | The whole §1 journey runs **offline end to end** over the synthetic fixture with fake providers, and the CLI is proven equivalent over the same services, including review → literal owner coverage → disposition → finish for an unmeasured table, with no paid coverage call. A live run over a real private PDF with real paid calls is **optional**, needs the owner's exact source and call consent at the time it is made, and is not required to finish this work. |

### 10.1 File inventory

**Modified:** `application/card_revision_finish.py`, `workbench/review.py`, `application/character_notes.py`, `models.py`, `extract.py`, `io.py`, `inputs.py`, `staging.py`,
`enrich.py`, `card_preview.py`, `ai_schema.py`, `exporters/anki.py`,
`application/extraction.py`, `application/extraction_batch.py`,
`application/card_change_staging.py`, `application/audio.py`,
`application/promotion.py`, `application/coverage.py`,
`application/assistant_staging_review.py`,
`application/assistant_staging_actions.py`, `application/assistant_promotion.py`,
`application/enrichment.py`, `kanji.py`,
`jpdb_kanji.py`, `application/deck_package.py`,
`application/assistant_context.py`, `application/assistant_assignment.py`,
`workbench/assistant.py`, `workbench/assistant_adapter.py`,
`workbench/assistant_batch_surface.py`, `workbench/assistant_packages.py`,
`workbench/dispatch.py`, `workbench/render.py`, `workbench/server.py`,
`cli.py`, `cli_extract_batches.py`, `pyproject.toml`,
`prompts/extract-table.md`, `prompts/extract-auto.md`,
`prompts/extract-prose.md`, `prompts/assistant-agent.md`, `prompts/README.md`,
`docs/DESIGN.md`, `docs/CARD_DESIGN.md`, `docs/WORKBENCH_PLAN.md` (one pointer
line in §W7), `AGENTS.md`, `README.md`, `docs/IMPORTING.md`.

**New:** `application/deck_capabilities.py`, `application/source_parts.py`,
`application/capture_recovery.py`, `application/study_job.py`,
`application/study_curation.py`, `application/study_finish.py`,
`prompts/extract-table-layout.md`, a workbench region/layout/curation editor
widget, `tests/fixtures/synthetic_table.pdf`,
`tests/fixtures/conjugation_golden.json`, one named test module per new service.

**Not modified:** `config.py`, `application/finish.py`, `application/build.py`,
`application/assignment.py`, `workbench/finish.py`.
`promote.py` needs no edit for the new mode — its
`mode not in {None, *extract.MODES}` check inherits it correctly for a staging
document that already carries a layout — and this plan makes no
unmodified promise for any file whose accepted-mode or payload contract grows.

### 10.2 Test rules and matrix

Failing test first, then the fix. Every new test is named for the behaviour,
names one production mutant, is proven to fail against it, and the mutation is
restored. `make gates` uses offline fakes: no live model call, no money, no
private bytes, no approved content edited to make a test pass, and no new check
that reads Japanese. Optional render dependencies are installed by `[dev]`, so
no source-render test skips.

One named module per area: **capabilities** (mixed-kind repository; unknown kind
refuses before census, packaging and prune; kanji store never parsed as
vocabulary; declared, inline and drill owners protected). **Rendering and
publication** (deterministic part names under the pinned dependency;
rotated-page rect mapping; worker-process confinement; namesake refusal and
identical-byte reuse; interrupted publication resumes; re-render mismatch
refuses). **Planning order** (unpublished paths refuse; published paths produce
children whose fingerprints match the confirmation; a jobless manifest's bytes,
`manifest_sha256` and consent fingerprint are unchanged by the optional
`job_id`). **Layout and `source_forms`** (request metadata changes the request
hash and not the response-schema hash; a supplied blank cell stays blank and an
omitted known id stays absent; an **unknown column id** refuses naming the part
and `(layout_id, revision)`; duplicate display labels are representable;
no code path matches a printed label anywhere; two different layouts retain their per-child bindings in one batch and retry, while a cross-input staging collision refuses before any call; initial confirmed dispatch and unsent resume preserve each bound layout fingerprint, while changed or omitted layouts refuse; the exact provenance validator accepts the new mode only with its layout and leaves ordinary provenance shapes unchanged; an empty `table-layout` answer is `unmeasured` and blocking exactly as in `table` mode; a `table-layout` request with no
bound layout refuses at every mode consumer; old captures recover from their
saved manifest layout, never from the current job document). **Curation**
(coordination lock before sorted file locks; all-file precheck before any write;
explicit cell/table delete leaves metadata byte-identical; resume from exact
before/after states; the barrier blocks both `decide_promotion` and
`execute_promotion`; a pending intent survives an ordinary choice edit; rewriting an existing layout
revision is refused, and append-layout plus part-binding update is one CAS save).
**Recovery** (each envelope shape; multiple valid proposals need the
owner-bound location tuple; a valid terminal wins; every refusal preserves
evidence). **Authority** (an unknown dispatch blocks unrelated paid work and is
never resent; recorded authority resumes with no new approval; a changed plan
demands a fresh one; a local owner Save is never accepted as paid or canonical
consent; Assistant and CLI carry the same literal review, coverage and disposition
choices without prematurely applying them, and stale preview bindings refuse; the CLI requires the fingerprint printed by its HTML preview and refuses template/reference/deck drift between preview and save instead of rebinding it;
model output cannot carry any capture-selection tuple component, while the existing owner-literal `approve_coverage` relay remains supported; `--yes` cannot supply missing owner decisions and blank reasons refuse; saving a second choice preserves an unchanged first choice's rendering binding). **Frontier** (a prepared-only retry manifest changes nothing; a
confirmed retry with a discard edge does; forks and cycles refuse; a committed
child is never superseded). **Crash mutants** at each boundary — after review,
mid-promotion, after enrichment before its receipt (including next-day resume for both promotion and enrichment ledger splits with their original frozen dates), during the audio WAL, after
package bytes before the ledger write, after the ledger before the download
offer, and after the package before the preview. Package preparation succeeds after the finish's own bound reference writes and resolved audio slots, including unpaid VOICEVOX and paid clips after WAL cleanup, but refuses unproved bytes, added or missing media paths, and any unbound reference or input change. Prepared coverage rendering must retain the existing writer's bytes and refusals and apply as one composed write, never through a second standalone coverage action. Projected promotion decisions refuse at function entry, including the early pattern/archive branches. Enrichment projection receives the prepared kanji store; its apply observes the same already-published store. A selected pattern-only review projects completion before any write; an unselected unreviewed set stays blocked. All parts see the same prepared pattern-store snapshot, true marks are not rendered twice, and false choices never erase existing or other selected marks. Jobless retries restore the whole saved layout, not an identity pair or a current job lookup. **Readiness** (held rows keep a
job incomplete until an explicit owner exclusion; each of `nothing`, `pattern_only`, `archive_retry` and `nothing_lands` uses its actual file effects and optional snapshot/receipt fields; a zero-landed or all-held middle part carries the prior canonical collection unchanged, needs a disposition and keeps its evidence, with no fabricated receipt; the four counts
stay separate; bound whole-deck inventory equality versus selection
containment). **Busy** (local status, preview and capture inspection answer
while a batch runs; ordinary prose gets the deterministic busy reply and
dispatches nothing).

`make gates` is expected to reach the sample build only after S0. Until then the
five baseline failures above are the only known ones, and no other failure is
treated as known.

---

## 11. DESIGN and lifecycle amendments

Each amendment lands **before** the milestone whose behaviour it authorizes.

**A — S3, replacing the `docs/DESIGN.md:337-338` parenthetical:**

> The owner may confirm one exact finite batch of explicit immutable source
> parts. A part is a preserved source file: a whole document, or a derivative
> janki rendered from one page or from explicitly reviewed regions of one page
> under a recorded recipe binding the parent, the geometry, the renderer and
> encoder versions, and each part's exact bytes. Janki renders pixels; it never
> decides that a rectangle is a table, a row, or a word. The original is never
> edited, and a derivative is a new deliberate intake that never overwrites a
> namesake.

**B — S3, Surfaces:**

> Preparing source parts is ordinary local work: an explicit preparation request
> and the owner's own page or region choice authorize it. It adds no approval
> gate, discloses nothing to a provider, and spends nothing. Region geometry
> comes only from the owner's editor or an owner-supplied recipe file; a model
> may ask for the editor and may not author or alter a coordinate. A
> preparation receipt belongs to the parts, not to one job, and any later job or
> CLI run may reuse it by its exact recipe id and receipt hash. Parts are
> published as immutable derivatives before any batch is planned, so the one
> exact paid batch confirmation names children whose bytes already exist.

**C — S5, the AI passes:**

> A source whose printed forms the owner has bound carries that binding as
> labelled request metadata: one explicit owner-assigned layout revision per
> child, with opaque stable column identities, the exact printed witnesses the
> owner recorded for each, and the owner's ordered display labels. The model
> returns those identities as keys of the existing conjugation map; the response
> contract and its fingerprint are unchanged, and only the request fingerprint
> changes. This is a complete additional extraction template, not a branch in an
> existing one and not a fourth card-writing path: `extract`, `enrich --ai` and
> `revise` remain the three. One confirmed batch carries one mode, with each child bound to its own
> explicit layout revision. Janki matches no printed label at runtime and owns no header
> detector: the returned key set must be a subset of the identities the request
> supplied, an unknown identity refuses for a person to settle, an omitted
> identity is absent and a supplied empty value is a printed blank. Ordinary
> unlabelled extraction remains a fully supported input shape.

**D — S5, Compile:**

> A record may carry an optional canonical `source_forms` table: the exact
> ordered columns its source printed, each with a stable identity and the
> owner's display label, and the cells keyed by those identities. A blank
> printed cell is a declared row with an empty value; an absent column stays
> absent; two columns may share a display label because their identities differ.
> Where it is present it selects the rows of the same Anki `Conjugations` field
> the deck already ships; where it is absent the existing map still does. No
> notetype, GUID or card direction changes, and no existing record is rewritten
> by adding this field. Per-archive provenance and the source's own witnesses
> are preserved beside it across hole-filling merges; a conflict with existing
> nonempty canonical content is disclosed under ordinary promotion, never
> silently overridden.

**E — S4, Surfaces:**

> A model may emit only closed typed intents. They may read, prepare a plan,
> open a local owner editor or prepare a choice that executes nothing, or
> dispatch and resume work the owner has already authorized exactly — a resume
> after fresh binding checks, a reserved unsent child under its original
> authority, or one unambiguously valid captured answer reaching the staging
> destination its own paid call was confirmed for. A model may not mint or
> select an owner decision, author source geometry, choose a column layout,
> execute curation, choose among competing captured proposals, widen scope into
> new protected work, or grant authority to write canonical content or saved
> facts. The new study-finish actions execute those writes only under their
> exact finish authority; existing deck-creation and other protected application
> actions retain their existing exact authorities. An owner control binds the exact
> resource, snapshot and choice it was rendered from and invokes the application
> service directly; a reversible local save asks for no second confirmation and
> is never consent for a paid or canonical effect. A model-proposed staged write
> still consumes the same one-use capability the browser and the CLI demand.

**F — S6, Surfaces, replacing the last sentence of the "Apply and finish"
paragraph:**

> That receipt-backed **Apply and finish** contract also covers a source-study
> job: one authority binds a set of reviewed source proposals, the owner's
> review and coverage decisions taken over the rendered cards, their promotions,
> the reference facts already visible in that review, the selected audio, the
> build and the package. Each phase's prepared payload is durable before its
> first effect, so an interruption resumes from that intent rather than a lost
> return value. It aggregates existing single-receipt scopes without replacing
> them and without discarding any source occurrence, and its completion is
> proven from artifacts — every accounted paid attempt, each part's exact
> disposition, and the bound whole-deck package inventory — rather than from a
> batch surface's completeness label.

### 11.1 `AGENTS.md` data-lifecycle entries

Three committed stores need the hand-edit prohibitions their peers carry at 1c
and 1d. Proposed entries, placed beside 1d:

- **`data/study_jobs/`** (derived beside the configured operations file):
  committed owner choices for one study job, its immutable layout revisions, and
  an append-only log of curation and action intents, outcomes and supersedes
  edges, plus references to receipts other stores own. Choice edits are
  compare-and-swap; an appended intent and a bound layout revision are immutable,
  and an ordinary choice edit may not replace either. A job record grants no
  spending, discard or approval authority and holds no independent progress
  state. Never hand-edit it, and never delete one while any action it reserved is
  unfinished.
- **`data/source_parts/`** (same derivation): committed, immutable publication
  receipts binding the parent source, the complete render recipe including
  renderer and encoder versions, and every planned part's exact name and expected
  hash. They are job-independent and may be reused by any later job or CLI run.
  An interrupted publication resumes against this expectation. Never edit one,
  and never delete one — publishing every part it names does not retire it — while
  any unfinished batch, finish authority or current job still references it.
- **`data/staging/done/study/`**: committed study-finish authorities. The
  immutable owner authority is written before the first effect; its phase intents
  and receipts advance under compare-and-swap. Never hand-edit or delete one
  while its finish is unfinished; it is the only authority an interruption may
  resume.

Actual new data files are created by future implementation work, not by this
planning. Every path derives from an existing configured path; no `[paths]` key
is added.

### 11.2 Other documentation

`docs/WORKBENCH_PLAN.md` §W7 gains one pointer line to this plan; historical
W5/W7 notes are not rewritten. `docs/CARD_DESIGN.md` gains the source-form
rules: what the card renders from `source_forms`, and that the witness table
beside the preview is a supplementary review aid rather than the card.
`README.md` and `docs/IMPORTING.md` gain the Assistant PDF-to-deck journey.
`prompts/README.md` records the new `extract-table-layout.md` template and the
shared tool-argument wording added to the three existing extraction files.
