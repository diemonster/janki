# Assistant study jobs — plan review record

**Status: plan, scheduling and sentence-audio review complete — no material findings remain.** This records an adversarial review of the
[implementation plan](ASSISTANT_STUDY_JOBS_PLAN.md) and its
[detailed contracts](ASSISTANT_STUDY_JOBS_CONTRACTS.md), not approval of
implemented behavior or Japanese study content. Implementation has not started.

## Scope and method

- Baseline: `55682fb61e40037409c162c42ca1b57b565ad83a` on `main`.
- Authority: `docs/DESIGN.md` and the current `AGENTS.md`.
- The planner and independent reviewers run through
  `scripts/claude-subscription.py`, using `claude-opus-5[1m]` at max effort and
  the verified subscription login. There is no API fallback.
- Reviews inspect the plan and relevant production contracts and tests. They
  exclude `data/**`, `dist/**`, and Japanese-content judgments.
- Each reviewed version is hashed. Findings are corrected before a fresh
  independent review checks the revisions and their affected contracts.

## Review rounds

### Round 1 — nine findings

Reviewed SHA-256:
`8279c10f4fa6c2fed36638fb9116884cbf610c6844e67efb1ae48072c0e5f363`.
Verdict: **FINDINGS 9**. All nine were accepted for correction; none had
passed re-review at this round.

| Finding | Defect | Required correction |
|---|---|---|
| F1 | Batch planning requires durable parts before it can name exact requests. | Publish requested derivatives under the configured scan inbox, then plan the paid batch. Local publication adds no consent gate. |
| F2 | Normalization drops blank source-form cells before rendering. | Specify their complete storage and rendering path. |
| F3 | Existing dictionary enrichment cannot act on staged proposals before review. | Prepare and bind exact fact values and the final visible projection before confirmation. |
| F4 | Unknown promotion, audio and package outputs cannot have precomputed after-digests. | Define writer-specific preparation and recovery proofs, including publication before bookkeeping. |
| F5 | The single-receipt finish type cannot represent several source associations. | Add a distinct aggregate scope retaining member receipts and a deduplicated work projection. |
| F6 | The capability table could omit durable media owners during pruning. | Preserve source, inline/override and drill owners, independently from synthesis capabilities. |
| F7 | Ordinary intake may rename a conflicting generated part. | Explicitly bind and check exact derivative names before publication. |
| F8 | A proposed live conjugation invariant rejects legitimate merged records. | Test engine results with fixtures; retain only supported structural live invariants. |
| F9 | Audio choice/provider alone does not bind the clips a finish may generate. | Confirm the exact prospective clip list and permit only its request-preserving progress. |

Additional local checks found missing treatment of duplicate table headings,
valid live journal states, confirmed retry lineage in completion, immutable
curation intents, and PDF renderer versions/thread safety. These are included
in the revision request. This round did not produce a clean verdict.

### Round 2 — three independent scopes

Reviewed SHA-256:
`2031d2c3cae613714916111037d0c31401e53a763f9c97086591a192dd40cc7b`.

The source review returned **FINDINGS 4**:

- The preventive prompt change missed the default extraction template.
- Model-proposed crop geometry lacked the required owner-origin binding.
- An ungated action let the model select among ambiguous captured answers.
- The new PDF dependencies were absent from the development test extra.

The finish review returned **FINDINGS 5**:

- Promotion planning must project the selected review and coverage decisions.
- Successive promotions need a chained canonical snapshot projection.
- The journey omitted deck creation and staged assignment.
- Recovery must handle partial promotion and unfinished ledger writes even
  while the live staging file survives.
- Package readiness compared the full deck with the smaller job selection.

The flow review returned **FINDINGS 7**:

- Assignment and coverage decisions were missing from the executable journey.
- The action table failed to distinguish model plans from owner actions.
- Closed action schemas and resource catalogues were missing from the file list.
- The recovery command depended on a job interface scheduled later.
- A DESIGN amendment was scheduled after its behavior.
- PDF dependencies were absent from the development test extra (also found by
  the source reviewer).
- New stores lacked lifecycle instructions in `AGENTS.md`.

These findings overlap; their counts are per reviewer, not a count of unique
defects. Local re-review additionally requires the actual card layout to use
the selected columns, durable package evidence before publication, full
curation payloads, and confirmation-backed retry edges. Source, finish, and
flow sections were rewritten against shared contracts. A newer draft or
manifest alone does not resolve any finding.

### Round 3 — assembled plan and contracts

The revised proposal is split into an implementation sequence and its detailed
contracts. Both documents form one reviewed bundle:

| Document | SHA-256 |
|---|---|
| `ASSISTANT_STUDY_JOBS_PLAN.md` | `e804afa3d0b885168b18fbfd6cf9cb01baf2041cc83bdad99a4ded4a6f6542cc` |
| `ASSISTANT_STUDY_JOBS_CONTRACTS.md` | `ac89caaebf072400bd353dd4221a7420f66e4320519468bed3b8ca5a19c32697` |

The source review returned **FINDINGS 3**, finish **FINDINGS 3**, and flow
**FINDINGS 5**. The per-child layout finding overlaps between source and flow.
All findings are accepted, including the two lower-severity consistency issues.

| Area | Finding | Correction required |
|---|---|---|
| Source | Exact provenance keys and a literal mode allowlist reject the new table mode. | Extend the validator explicitly; require the saved layout only for that mode. |
| Source / flow | The batch carries per-child layouts but its underlying planner accepts one layout. | Pass aligned layouts through the whole-input planner, keeping its cross-input staging collision check. |
| Source | Empty table-layout answers bypass table coverage behavior. | Apply table-mode coverage semantics at every mode branch. |
| Finish | Package validation rejects reference and audio writes made by its own finish. | Bind projected reference after-values and permit only the confirmed audio slots to resolve to proved bytes. |
| Finish | A non-landing part has an empty default merge result. | Carry the prior canonical state through non-landing decisions; never treat that default as an empty collection. |
| Finish | Promotion and enrichment replay can change ledger dates after midnight. | Freeze every date-bearing ledger argument in its prepared writer intent. |
| Flow | CLI lacks review, coverage and disposition decision carriers. | Add owner-only local controls and equivalent CLI arguments before the single finish confirmation. |
| Flow | Model output schema exposes an owner-only capture selection hash. | Remove it; the complete selection travels only through the owner control or CLI. |
| Flow | Job layout storage and writer names disagree between sections. | Use one append-only layout namespace and one consistent CAS writer contract. |
| Flow | Modified and untouched file lists contradict each other. | Reconcile them with the actual milestone changes. |

Local checks also found a stale one-layout sentence in the proposed DESIGN
amendment and overbroad wording that could forbid existing canonical actions.
Those are corrected without changing the existing action authorities. The next
round reviews the corrected bundle; round 3 is not a clean verdict.

### Round 4 — corrected contracts and integration

| Document | SHA-256 |
|---|---|
| `ASSISTANT_STUDY_JOBS_PLAN.md` | `0f72af5b75587232183eb59bc605dfeb94c13e44232353a59ede40d3cc7ef930` |
| `ASSISTANT_STUDY_JOBS_CONTRACTS.md` | `9df2a120cfe6aded2ea9a166d9c9351b16e524fc7a20e186fe1c5fd35e2cae63` |

All round-3 corrections are incorporated. Integration checks also corrected
the package projection to use the post-audio canonical revision, and the
audio-completion proof to work for unpaid VOICEVOX and paid clips whose WAL
and capture have already been cleaned up. Owner choice bindings exclude the
mutable job-file revision and checkbox state, so a later local choice does not
invalidate an earlier unchanged one.

The source/flow and finish reviewers each returned **FINDINGS 4**. All eight
are accepted and corrected:

| Area | Finding | Correction |
|---|---|---|
| Source | Dispatch-time revalidation loses the saved layout. | Freeze it in the dispatch expectation and pass it through the single-source replan on dispatch, resume and retry. |
| CLI | Owner-decision commands cannot name the preview they refer to. | Preview prints its fingerprint; decisions require `--rendering` and reject changed inputs. |
| Authority | Job-only coverage restrictions could remove the existing literal relay. | Scope the restriction to job decisions and retain `approve_coverage`'s owner-literal field. |
| Milestones | S6 omits the adapter that must handle its new controls. | Add the Assistant adapter and routed surface to S6. |
| Enrichment | The application projection cannot receive the prepared kanji store. | Add the explicit argument and pass it to the fact/record helper. |
| Promotion | Non-landing states have different snapshot, archive and deletion behavior. | Define each state's actual effects and evidence, including exact earlier landings. |
| Promotion | The proposed guard comes after early branches that can write. | Refuse projected decisions at executor entry. |
| Coverage | The standalone coverage writer conflicts with a single prepared staging write. | Compose pure coverage rendering with review into one payload; retain the standalone route separately. |

### Round 5 — corrected dispatch, decisions and writer states

| Document | SHA-256 |
|---|---|
| `ASSISTANT_STUDY_JOBS_PLAN.md` | `b5af9b633c2ca6067e92ca61b8c4784165c7e7b379458646992fb3df83bd7407` |
| `ASSISTANT_STUDY_JOBS_CONTRACTS.md` | `2c030fc8dea48c32f3bae58d798121cc685b188f8460d6f329cd9451809fd91a` |

Each reviewer returned **FINDINGS 1**. All round-4 findings closed; the
remaining issues were a retry bullet that named a layout identity pair instead
of the complete saved layout, and the missing pattern-store input to the
promotion projection. Both are corrected: retries deserialize the full frozen
layout, and every part projects against the same prepared pattern-store
snapshot, retaining the existing source/run/provenance checks.

### Round 6 — final combined review

| Document | SHA-256 |
|---|---|
| `ASSISTANT_STUDY_JOBS_PLAN.md` | `df874010aa717d8c57b09f40bc523a9611845ed9370e8d5b9b638d6abde4a818` |
| `ASSISTANT_STUDY_JOBS_CONTRACTS.md` | `cdf9713540a441a964d6042b19138457d332861ba289767f7d773eada07d1b21` |

The combined independent reviewer returned **VERDICT: CLEAN** on the exact
round-6 bundle above. It verified the last two corrections and their affected
interfaces against the relevant code, and found no remaining material plan
findings. The earlier independent source/flow and finish reviews had closed the
preceding findings; local integration checks also found no remaining material
contradiction. All accepted findings are resolved; none was waived.

This is a clean implementation-plan review, not proof that the future
implementation is correct. The milestones still require failing tests, mutation
proof and `make gates`; production implementation has not started.

## Parallel-scheduling addendum — 2026-09-09

The owner's follow-up asks which implementation work can run in parallel.
§10.3 adds a development schedule, file ownership, interface handoffs and
integration checks. The milestone scopes and detailed contracts are unchanged.
The original six-round verdict above remains tied to its recorded hashes.

A separate subscription-backed planning pass inspected shared files and
dependencies while the schedule was drafted. The schedule uses the strongest
independent wave, S1/S2/S3 after S0, keeps each of S4 and S5 in one
implementation lane, and splits S6's writer preparation into three development
packages with one integrated finish.
Additional checks against `Makefile`, `assistant_agent.recover_agent` and
extraction revalidation established the need for per-lane environments and an
activation window that preserves affected unfinished operations under their
original revision. This adds no runtime authority or automatic cleanup.

### Scheduling review 1 — one finding

| Document | SHA-256 |
|---|---|
| `ASSISTANT_STUDY_JOBS_PLAN.md` | `1111a734adefb52b31d54862db5630954d2705f423f21ade6c4efcaa965d71c3` |
| `ASSISTANT_STUDY_JOBS_CONTRACTS.md` | `cdf9713540a441a964d6042b19138457d332861ba289767f7d773eada07d1b21` |

The independent reviewer returned **FINDINGS 1**: the S6-E handoff said facts
were prepared before the review projection, reversing §7.9's dependency on the
post-promotion projection. The correction names that projection as the input
and places fact preparation before word enrichment. The review passed the
remaining scheduling checks: milestone joins, shared-file ownership, complete
lock rollout, test-only fakes, runtime ordering, authority and both surfaces.

### Scheduling review 2 — clean

| Document | SHA-256 |
|---|---|
| `ASSISTANT_STUDY_JOBS_PLAN.md` | `d717294399c26af20bc76bb6283c09b1dd164063449bc12a4e9e89a483664d1d` |
| `ASSISTANT_STUDY_JOBS_CONTRACTS.md` | `cdf9713540a441a964d6042b19138457d332861ba289767f7d773eada07d1b21` |

The fresh independent reviewer returned **VERDICT: CLEAN**. It confirmed the
corrected reference ordering, the fixture-to-real integration obligation, the
Makefile's preference for a checkout-local environment, and the prompt/schema
activation hazards and recovery boundaries. The review used an immutable
packet containing the scheduling text, the exact delta, the prior finding,
DESIGN and relevant contract/code excerpts. The integration lead computed and
rechecked the packet's hashes locally; the reviewer did not recompute them.
Both scheduling reviews used the same guarded subscription launcher and pinned
review model/effort as the original rounds. No finding was waived.

Local documentation links and `git diff --check` pass. The detailed-contracts
file is byte-identical to the round-6 version. No implementation, test,
dependency or data changes were made, so the baseline gate result below is
unchanged; the scheduling addendum did not rerun that suite.

## Sentence-audio addendum — 2026-09-09

The owner asks that study jobs generate sentence audio and link it through the
delivered deck. The proposal now includes sentence audio by default, keeps
opt-out in owner-controlled job choices, and verifies an independent census of
expected example slots through canonical references, current media, exported
sound fields, APKG bytes and the final interactive preview. Stored examples
beyond the existing two export slots remain explicitly disclosed and voiced;
the proposal adds no note fields. The same finish confirmation binds exact
provider/model, billing and requests; default inclusion authorizes no paid call.

A bounded planning pass identified the previous opt-in CLI, the proposed
model-controlled audio field and the vacuous selected-clip completeness check.
The update fixes all three. A suggested written reason for audio opt-out was
not adopted: the direct owner control records this ordinary preference without
another reason or approval gate.

### Sentence review 1 — three findings

| Document | SHA-256 |
|---|---|
| `ASSISTANT_STUDY_JOBS_PLAN.md` | `24d1ff8c7fbefd481fda4d87928e60967429bc9b4339ca3d1ac3b95939ad1227` |
| `ASSISTANT_STUDY_JOBS_CONTRACTS.md` | `d83799626b28296323095b5b78cb374e3f1c4aebaa592e177c699c2ff034698c` |

The independent review returned **FINDINGS 3**: specify the refusal for
identical displayed sentences with divergent spoken input; name the S6
preview files, owner and edit order; and define the job-wide audio choice's
CAS binding and persistence through staging/rendering changes. All were
accepted and corrected, along with an unrelated indentation nit.

### Sentence review 2 — one remaining correction

| Document | SHA-256 |
|---|---|
| `ASSISTANT_STUDY_JOBS_PLAN.md` | `7554e35a2bd78e438e8b15cd9496ee7a94c0c159ab9abcc8cbe3455693a3d44a` |
| `ASSISTANT_STUDY_JOBS_CONTRACTS.md` | `5737293460e7de4f954684b46574b353df0e4487c9032a249d20033529b87a38` |

The fresh review closed the file-ownership and CAS findings and requested an
exact citation for the existing collision refusal. Local inspection confirmed
`audio_cmd.prepare_example_audio_profiles` performs it during planning; its
diagnostic names the record and duplicate sentence, not example positions.
The contract now cites that service and its planning caller and describes the
existing diagnostic accurately. No new diagnostic or production edit is needed.

### Sentence review 3 — clean

| Document | SHA-256 |
|---|---|
| `ASSISTANT_STUDY_JOBS_PLAN.md` | `7554e35a2bd78e438e8b15cd9496ee7a94c0c159ab9abcc8cbe3455693a3d44a` |
| `ASSISTANT_STUDY_JOBS_CONTRACTS.md` | `df05d54e3dd8973ac33431a848d937cd4d566a853ad72ce59209009f5e807ce5` |

The final independent review returned **VERDICT: CLEAN**, confirming the
per-record refusal, its actual diagnostic and the invocation before dispatch.
All three reviews used immutable packets with the affected text, exact hashes,
DESIGN and relevant code/contract excerpts, through the same guarded
subscription launcher and pinned review model/effort. The integration lead
computed and rechecked the hashes locally. No material finding was waived.
The earlier clean verdicts remain bound to their own versions above; this
addendum is not a claim that any proposed feature is implemented.

Local documentation links and `git diff --check` pass. This addendum changes
only the plan, contracts and review record; it makes no production-code,
test, dependency or canonical-data change. The existing baseline gate result
below is unchanged and was not rerun for these documentation edits.

## Repository validation

`make gates` ran on 2026-09-09. Ruff passed; pytest reported **5 failed,
5,243 passed**. These are the same five baseline content-dependent test
assumptions named in the plan:

- `test_word_decks_are_nonempty_and_do_not_share_stable_ids`
- `test_each_real_intake_tag_selects_only_its_owner`
- `test_the_key_set_and_its_order_are_what_the_stored_records_already_carry`
- `test_the_curated_records_are_reproduced_form_for_form`
- `test_committed_repository_operation_journal_loads`

The sample build was not reached because the test step failed. Fixing these
tests is planned implementation work; this documentation change does not
modify production code, tests, prompts, or study data.
