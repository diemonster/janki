# Hardening from real deck work

This document defines what to do when a real deck exposes a defect. The goal is
to make one lesson protect later decks without giving an agent broad permission
to change curated Japanese.

## What self-healing means

The repository is the memory. A durable lesson is a case, fixture, rule, repair,
prompt version, decision, or test that is stored in git.

After one systemic defect is diagnosed, a later run must do one of these things:

- avoid the defect;
- apply a named deterministic repair whose evidence and conditions pass; or
- stop with a finding that states what a person must decide.

Self-healing does not change model weights. It does not use hidden memory. It
does not let a model rewrite curated records without review.

## Required workflow

Use this workflow when deck creation, review, import, enrichment, audio, render,
or build work exposes a defect.

### 1. Reproduce the defect

Preserve enough evidence to run the same boundary again:

- the SHA-256 fingerprint of the source or record input;
- a locator such as a page, region, row, record ID, or field;
- the pipeline stage;
- the exact command and relevant configuration;
- the observed result and the required result; and
- the smallest safe evidence excerpt that explains the failure.

Use repository-relative paths. Do not record secrets or absolute paths. Do not
modify `data/inbox/` to make a reproduction easier.

### 2. Classify the correction

Classify the correction before you generalize it.

A `content-specific` correction is true for this item because of its meaning or
context. Examples include a wrong gloss for one word and an ambiguous kana mark
in one photograph. Keep this correction as a curated edit.

A `systemic` correction states a repeatable pipeline rule. Examples include a
table extractor that always drops the second column and a parser that always
chooses a lower-priority part of speech.

When the class is uncertain, use `content-specific`. Keep the item in staging.
Do not create a general validator from one ambiguous example.

### 3. Preserve a systemic case

A systemic defect needs the smallest case that still fails. Use one of these
source forms:

- a synthetic fixture that contains no source-derived content and records that
  basis;
- a minimized owner-provided or source-derived fixture that the user has
  approved for redistribution; or
- an exact fingerprint and locator for an immutable private source under
  `data/inbox/`.

Do not copy a complete owner-provided work when a small crop or excerpt proves
the defect. Do not commit source-derived pixels without exact redistribution
approval. Private-source consent for a model evaluation does not grant
redistribution permission.

If the case and findings tools are not available yet, keep the implementation
task open. Record the evidence and the intended `open` or `deferred` state in
the task note. Do not call the systemic defect fixed.

### 4. Fix the earliest reliable boundary

Fix the earliest production boundary that can state the rule. Examples include
parsing, normalization, validation, prompt construction, response handling, and
rendering.

A prompt change is a valid fix for model behavior. It still needs an offline
case for downstream response handling and a live evaluation case for the prompt
behavior. A test-only copy of production logic is not a regression case.

### 5. Choose validation, repair, proposal, or hold

Add a deterministic validator only for artifact structure — identifiers, counts, file shape — never for a judgment about the Japanese; that belongs in the prompt template (AGENTS.md: the prompt does the work). Add an
automatic repair only when it meets the repair boundary below. Use a staged
proposal when a protected non-identity content field needs a human decision.
Hold the record when evidence is uncertain.

### 6. Verify and close

Run the new case, the complete accumulated case corpus, and `make gates`.

A systemic finding is fixed only when all of these statements are true:

- the finding links to a passing case;
- the case calls the production boundary;
- the production fix is present;
- the complete fixed corpus passes; and
- `make gates` passes.

If a safe fix is not ready, keep the finding `open` or `deferred`. Only the user
can approve an `accepted-risk` state.

## Automatic repair boundary

Automatic repair is default-deny. A repair can target only these fields:

- `furigana`;
- `romaji`;
- `examples[*].furigana`;
- `examples[*].romaji`;
- `audio`;
- `image`; and
- `frequency_rank`.

The repair must have evidence that proves the result. It must preserve meaning
and identity. It must have a stable code, a narrow precondition, a stated
invariant, a postcondition, an idempotence test, a positive fixture, and a
non-triggering near-miss fixture.

The canonical `source.raw_fields["janki_repairs"]` provenance entry is a
required side effect. It is not a repair target.

All other record and source fields are default-deny. This set includes:

- `id`, `expression`, and `reading`;
- meanings;
- part of speech, verb group, and transitivity;
- pitch accent and audio accent;
- Japanese and English example text;
- conjugations, tags, and usage notes; and
- unknown source fields.

A new repair does not update a stored curated record merely because the repair
exists. A protected non-identity content change must use a fingerprinted,
field-scoped proposal and explicit user acceptance. Ordinary promotion must not
consume that proposal.

Existing identity is outside both repair routes. A repair must not change or
propose a change to `id`, `expression`, or `reading` for an existing record.
Such a conflict needs a manual migration plan that states what happens to Anki
review history. A new candidate still needs identity review before its first
promotion.

The allowlist can grow only through a reviewed contract change with a concrete
invariant and adversarial fixtures.

## Study-content trust boundaries

The camera pilot (M7.6T) forced three layers apart. Every learner-facing field
now belongs to exactly one of them:

- **Source evidence** records what the source shows: locator, verbatim context,
  the excerpt, extraction confidence. It lives in `source.raw_fields`
  (`context`, `example`, `page`, `confidence`, `inclusion_reason`) and can
  prove an identity or give review context. It is never study content by
  itself: extraction keeps the excerpt under `raw_fields["example"]` and
  leaves canonical `examples` empty.
- **Provisional facts** are model claims awaiting a named authority.
  Extraction marks the semantic fields it filled in
  `raw_fields["provisional_fields"]` as `name:fingerprint` pairs, value-bound
  so a later human edit breaks the binding and reads as curated. Dictionary
  enrichment replaces a provisional `meanings` or `part_of_speech` on an exact
  identity match, shows the change in its diff, and clears the resolved mark;
  a near match, absent entry, different reading, or missing reading holds the
  claim with a named warning. A stale mark is cleared, never obeyed.
- **Study content** passed its field-specific authority. On an
  extract-sourced record, example acceptance is explicit and per sentence:
  the reviewer types `example_authority: staging-review` into the staging row
  — presence of an example proves nothing, since the large AI-enrichment
  route stages model sentences on extract rows too — and promotion replaces
  the sentinel with the accepted sentences' content fingerprints, so the
  durable stamp covers exactly the Japanese the reviewer read. Non-extract
  sources (imports, hand-written records) are the user's own data, curated by
  arrival. Enrichment pins an example's exact Japanese only when acceptance
  covers it; an unaccepted stored sentence is never described to the model,
  never silently replaced, and changes only on an explicit `--force-fields
  examples` request.

Promotion, oracle acceptance, and a structurally valid record approve nothing
semantic. The local gates enforce teaching suitability before anything paid or
audible: `validate` errors on certain fragments and false register labels
(`qc.example_content_holds` — one judgment, certainties only, no grammar
model), the audio command refuses to voice any held or failing example, every
one of these keys and the provisional
marker travels through the merge with the field it describes, and a shipping
build — word or drill deck — reports local validation failures on their own
before the review store is consulted.

The final AI review stays read-only and residual. It runs only after every
local gate passes for a complete release candidate, sees the canonical record
and a compact authority summary rather than the original PDF or image, cannot
establish authority or mutate a record, and a second broad pass needs explicit
repository-owner approval.

Records promoted before these boundaries carry no authority keys, and their
examples take the same posture as any unaccepted sentence: preserved, never
pinned, never rewritten without an explicit `--force-fields` request —
provenance existing now is not a reason to touch them. A staging file written
before the boundary may still hold machine-copied examples; promoting it
cannot mint acceptance (no reviewer typed the sentinel), so its examples land
preserved-but-unaccepted — re-extracting is still the cleaner path.

## Extraction coverage gate

Table extraction writes a `coverage` block to its staging file. The block
stores every source unit and the exact comparison with an exhaustive human
oracle. A model-reported count is diagnostic only. It cannot make coverage
complete.

Use this form to bind an approved oracle:

```console
janki extract page.pdf --mode table --coverage-oracle quality/oracles/page.yaml
```

The coverage status has these meanings:

- `matched`: The exact unit keys, context fingerprints, and dispositions match
  the approved exhaustive oracle.
- `mismatch`: At least one exact fact does not match.
- `unmeasured`: A table has no exhaustive oracle.
- `selection`: The result contains prose selection only. It does not claim
  exhaustive coverage.

`mismatch` and `unmeasured` block promotion. The repository owner can accept
the exact state after a source review. Add `coverage.approval` with these
fields: `authority`, `source_fingerprint`, `coverage_block_fingerprint`,
`accepted_dispositions`, `accepted_mismatches`, `unmeasured`, `reason`, and
`approved_at`. The first field must be `repository-owner`. The two accepted
mappings and the unmeasured value must repeat the current coverage block
exactly. A source-unit or oracle change makes the approval stale.

Promotion keeps the approval in `data/staging/done/`. For an oracle result,
promotion also reloads the current approved oracle and recomputes the coverage
facts before it writes records or the ledger. A staging file from before M7.4
has no coverage block. It stays valid as a legacy unmeasured file.

## Decisions that belong to the user

An agent can prepare evidence and a draft. It must not infer, generate, or grant
any decision in this table. It must not answer an approval prompt as the user.
A broad request such as “finish the deck” or “fix all findings” is not one of
these approvals.

| Decision | The exact approval must name |
| --- | --- |
| Live model evaluation of a private source | Case ID, source fingerprint, provider, resolved model IDs or model scope, purpose, and stored reason |
| Redistribution of owner-provided or source-derived material | Artifact fingerprint and the license or other redistribution basis |
| Human unit oracle | Source fingerprint, oracle ID, normalized oracle-content fingerprint, oracle type, and selection rubric when present |
| Manual coverage acceptance | Source and coverage-block fingerprints, each accepted mismatch or unmeasured disposition, and the reason |
| Repair proposal | Proposal-entry fingerprint and the `yes`, `no`, or `quit` answer |
| Ambiguous new identity | Source fingerprint and locator, expression, chosen reading, and reviewed evidence |
| Existing identity migration | Exact old and new identity and the review-history plan |
| Accepted risk | Finding ID, normalized risk-content fingerprint, risk, and reason |
| Live baseline | Report fingerprint, acceptance-input-manifest fingerprint, and displayed diff |

An agent may record an approval only after the user gives that exact approval.
It must not widen its scope. If approval is absent, keep the value false or
missing. Stop that route or select an input that does not need the approval.

## Agent completion rule

A content-specific edit can finish as a content-specific edit. It does not need
a manufactured global rule.

A systemic defect is not complete when only the current row is corrected. The
agent must complete the workflow in this document or preserve the defect as an
open or deferred finding. More cards and fewer visible errors in one deck are
not proof that the system learned the lesson.
