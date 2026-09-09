# Japanese Anki Project Instructions

## Purpose

This repository converts Japanese vocabulary and study material into custom
Anki decks. Shirabe Jisho CSV exports are a primary input source. The repository
is the durable source of truth; generated `.apkg` files are build artifacts.
The one-page `docs/DESIGN.md` leads: when code, plan, or any other document
disagrees with it, DESIGN.md wins and the other changes.

## The prompt does the work

**This is the first rule, and it overrides convenience.** janki asks a model to
read and parse Japanese. When the model's output is wrong or incomplete, the
first move is to *expand the prompt* — not to add code that inspects, corrects
or second-guesses what came back.

- **Do not write logic that reads Japanese.** No rule that decides a reading, a
  word boundary, a register, or whether notation describes a sentence. Japanese
  is contextual; a rule derived from one sentence is wrong for the next, and a
  growing list of exceptions is how M7.6V's retired jpdb sentence oracle got
  built. It flagged 38 of 155 examples with no true positive among them.
- **The logic this project owns is enrichment and filtering**: fetching facts
  from dictionaries, and shaping reviewed content into an Anki deck. Structural
  contracts — identifiers, fingerprints, field counts, file provenance — are
  the project's, because they are about the artifact rather than the language.
- **Prompts are templates and belong in files.** Every *system* prompt on the
  Anthropic path is Markdown under `prompts/`, sent byte for byte and re-read
  on every run. Three pieces still live in Python: the labelled data turn built
  from the record or source, the terse `Field(description=…)` labels shipped in
  the JSON schema, and the Codex provider's JSON-only/no-tools preamble. Change
  what a pass asks for by editing the file, never by adding code that fixes up
  the answer. A prompt needing a branch in its instruction prose is two prompts
  — that is why extraction's three modes are three complete files rather than
  one file plus three rule blocks.
- **There are three card-writing model calls.** `extract` reads a source once and
  returns complete proposed cards plus the patterns the source teaches;
  `enrich --ai` receives a bare vocabulary record and returns the same meanings,
  examples, and usage-note shape; `revise` receives an explicitly selected
  existing card or deck plus the owner's requested change and writes a staging
  proposal, never canonical content. ChatKit may dispatch `revise` after a
  plan-bound owner confirmation, but it is a surface over that pass rather than
  another writing path. Pattern discovery and meaning improvement are parts of
  the first two rich answers, never automatic follow-up audits. `janki patterns`
  only lists and records human review of pattern sets already written by
  `extract`.
- **janki's logic enriches the card; it never audits the model.** Review
  gates and hardening rules aimed at proving the model wrong are the same
  anti-pattern as reading Japanese in code: writing a rules engine for
  Japanese from scratch. If the output is thin, the template asks for more.
- Before adding any check: could the prompt have asked for this? If yes, change
  the prompt. Add a check only for something the model cannot be asked — and
  say in the commit why not.

## Development rules

### Content delivery is not platform work

- A request to create, revise, voice, or build Japanese study content starts in
  the **content-delivery lane**. Use the existing pipeline, validate its exact
  artifacts, and deliver the requested deck before considering platform
  improvements. Do not turn a useful enhancement, refactor, roadmap item, or
  code audit into an unstated prerequisite for that artifact.
- If existing machinery truly cannot produce the requested artifact safely,
  state the missing capability and get the owner's explicit approval before
  entering the **platform lane**. Keep that work separately scoped. A content
  request alone does not authorize a codebase change merely because one would
  make future content easier.
- Code-review findings and Japanese-content findings never share a review. A
  code review cannot block content-only delivery, and a content review cannot
  expand into a code audit. Generated-media and ledger consistency are checked
  mechanically.
- A surfaced workbench content job keeps its own progress, recovery state and
  deliverable. Assistant/UI work may improve that lane later; it cannot replace
  the current job or leave the owner waiting in a coding interface.

- **Pre-release: there are no legacy paths.** Nothing is released and nothing
  outside this repository depends on this code. When an approach is
  superseded, delete it in the same change — old code paths, compatibility
  shims, deprecated flags, config keys, tests for removed behaviour. Do not
  preserve, deprecate, or warn; remove. Breaking changes are expected and
  free, and a migration is a plain data edit, not a supported pathway.
- Use Python 3.11 or newer.
- Keep parsing, normalization, validation, and Anki generation separate.
- Prefer the standard library unless an external package materially simplifies the task.
- Add or update tests for every supported input format and behavioral change.
- Never modify files under `data/inbox/`.
- Treat `dist/` as generated output.
- Use deterministic **GUIDs** (`genanki.guid_for(record.id)`) so rebuilt decks
  update notes instead of duplicating them. Note *ids* are timestamps genanki
  mints per build and are not stable — the GUID is what Anki matches on.
- Surface source filename and row number in import errors.
- Never silently discard an input row or unknown source column.
- **Models for working on janki**: Claude Opus 5 across the board —
  implementation at extra-high effort (`.claude/settings.json`), review and
  planning at max (`.claude/agents/code-reviewer.md`, `.claude/agents/planner.md`).
  No model aliases in settings: the id is written out, so a harness alias
  change cannot silently swap the model. janki's own runtime calls follow the
  same rule (`config.py` defaults, `claude_client.DEFAULT_EFFORT`; only the
  ordinary Assistant turn takes a configured depth, `[assistant] effort`).
- **Every development, planning, or review model launch goes through
  `scripts/claude-subscription.py`.** That is the repository's only CLI entry
  point to a model. It builds the child's environment from an allowlist,
  resolves an absolute `claude`, and runs `auth status --json` under exactly
  the environment, working directory and `--safe-mode --setting-sources ''`
  the launch itself uses — so what it verified is the process that runs. It
  refuses anything short of a claude.ai first-party Pro or Max login.
  - `scripts/claude-subscription.py --check` runs that probe alone, free, and
    prints a summary carrying no account details.
  - **Not allowed substitutes:** bare `claude -p`, a hand-written `env -u
    ANTHROPIC_API_KEY claude …`, an Agent SDK or API call standing in for the
    CLI, or a settings file that re-supplies a provider. `--settings`,
    `--setting-sources`, `--bare` and any cwd-changing launch form are refused
    by the launcher rather than overridden.
  - **A refusal stops the work.** There is no API-billed fallback, and
    switching to one is not a workaround for a failed login — it is the
    mistake this exists to prevent. A valid Max login and an inherited
    `ANTHROPIC_API_KEY` are indistinguishable from the outside: the CLI runs,
    answers, exits 0, and the only evidence is a Console bill.
  - The wrapper protects this repository's entry points — the review hooks and
    anything an agent starts here. It cannot police a `claude` typed in an
    unrelated terminal; that is what `--check` and this rule are for.
  - It changes nothing about **paid content calls**. `janki extract`,
    `revise`, `promote --accept-coverage`, Realtime audio, and the explicit
    `anthropic-api` revision provider each still need their own exact
    authorization, and the Max account's extra-usage setting is the owner's.
- Run `make gates` before considering work complete: it runs ruff, pytest,
  and a sample deck build. Run it rather than its parts. Bare `pytest` and
  bare `janki` resolve through the venv's editable install to the *primary*
  worktree, so inside a linked worktree they exercise code your branch never
  changed and report green. `make gates` derives every path from the
  checkout it lives in; `conftest.py` does the same for pytest and aborts if
  the package still resolves elsewhere.

## Japanese-content rules

- Store Japanese, kana reading, furigana source, romaji, and English separately.
- Use Anki furigana notation such as `日本語[にほんご]`.
- Do not algorithmically guess furigana segmentation for mixed kanji/kana words.
- Romaji is a temporary learning aid. Vocabulary cards do not render it; any
  other learner-facing surface that does must keep it behind a disclosure
  control.
- Keep default examples appropriate for a beginner using Genki-style grammar.
- Prefer natural English; include literal English only when it teaches a useful structure.
- Identify verb group, transitivity, and common conjugations when known.
- Flag uncertain readings, pitch accent, meanings, or usage rather than guessing.

## When janki gets something wrong

*Rewritten 2026-08-15 (M8.4). This was a set of rules for operating a
findings-and-cases corpus: `quality/findings.yaml`, replay cases with pinned
fixtures, pilot reports, `janki harden status` and `replay`. Most of that value
was ordinary regression testing wearing ceremony, and it is deleted. The loop
below is what is left, and it is the loop every other project already uses.*

- A defect in janki's own machinery — an importer, packaging, provenance, a
  repair, a CLI contract — gets a **failing test first, then the fix**. The
  test goes beside its subject in `tests/`, named for the behaviour rather than
  the bug.
- **A wrong or thin model answer is not a defect in janki.** It is a template
  problem, and the fix is a clearer prompt (`prompts/`). Do not add a
  check that reads Japanese to catch it.
- Prove a new test actually catches its subject: break the production code in
  the single way the test names, confirm that test fails, and restore. A test
  that passes against its own mutation is documentation, not a guard — this
  repository has produced several, and mutation is the only thing that found
  them.
- Do not generalize one specific correction into a validator. Fix the record,
  or ask the template for something better.

- Automatic repairs are default-deny. Their only allowed target fields are
  `furigana`, `romaji`, `examples[*].furigana`, `examples[*].romaji`, `audio`,
  `image`, and `frequency_rank`. The canonical
  `source.raw_fields["janki_repairs"]` entry is required provenance, not a
  repair target. Every other record or source field is denied unless a reviewed
  contract change adds it with an invariant and adversarial fixtures.
- Never repair or propose a change to an existing record's `id`, `expression`,
  or `reading`. These fields determine Anki identity and review history.
- The user owns consent to send their own material to a paid model
  (`janki extract` asks, and `--yes` is how they answer in advance),
  redistribution approval for owner-provided material, ambiguous new-identity
  resolution, existing-identity migration, and accepted-risk approval. An agent
  must not infer, generate, grant, or widen one of these decisions. **It must
  not answer an approval prompt as the user** — which is why the extract gate
  refuses on a non-TTY rather than proceeding.
- A direct owner confirmation inside ChatKit is still the owner's decision, not
  the assistant answering for them. It is valid only when it consumes a one-use
  capability bound to the exact plan the conversation rendered. The assistant
  may never infer that confirmation from an earlier broad request, confirm its
  own proposal, or reuse or widen the capability.
- **One exact confirmed extraction batch is the exception to one paid call at a
  time.** It is a consent and scheduling shape, not a new pass: a single
  confirmation that enumerates the batch exactly authorizes its child calls,
  and each child keeps its own source, request fingerprint, provider/model and
  billing identity. It never becomes one paid parent call, and the children's
  answers never merge into one synthetic provenance. Unrelated authorization
  still refuses while any operation is live or unaccounted for.
- Retrying part of a batch is a **fresh** owner decision: one new confirmation
  binds the new exact requests together with the forget/discard decision for
  each failed operation it clears, and fresh operation ids are mandatory. A
  successful child is never retried, and an in-flight child needs the owner's
  `operations --end` decision first. An `outcome_unknown` child is already
  ended; its new retry confirmation must acknowledge the uncertain prior cost
  and explicitly discard its exact retained evidence. No unknown call is ever
  sent again automatically. An agent may not make one of these decisions
  or read it out of the owner's original batch confirmation.
- **Staging coverage acceptance is the exception, decided 2026-08-16.** It was
  on the list above until the owner measured what it cost: since M8.4 deleted
  oracles every extraction *carrying a table* lands `unmeasured` — a prose-only
  source lands `selection`, which needs no approval at all — and the approval
  block repeats
  every source unit's facts — 168 lines of YAML per page, hand-transcribed,
  with the actual judgment buried in six of them. The owner's decision moved up
  a level: not "is this page accounted for" but "may a model answer that".
  `janki promote --accept-coverage` runs `prompts/approve-coverage.md`, shows
  the model the page itself and janki's account of it, and records the verdict
  with `authority: model`, the model id and the prompt's fingerprint — so a
  card promoted on a model's word says so permanently and a reader can tell
  which asking produced it. A refusal stops the promote. The flag is opt-in;
  without it the approval is still the owner's to write.
- That exception is narrow on purpose, and it is **not** the review subsystem
  returning. Coverage is a count — is everything on the page in the record —
  which DESIGN.md puts in janki's business. The moment a prompt asks a model
  whether the *Japanese* is any good, it has become the thing M8.2 deleted.
- A broad request to finish work is not an approval for any of the above. If
  exact approval is absent, stop that route or use material that does not need
  it.

## Card-design rules

- **Card review defaults to an interactive HTML preview of the actual cards.**
  The owner's reference is `dist/201-week-2-kanji-preview.html`: navigate the
  cards, see their real layout, flip with Show Answer, and use the answer's
  disclosures independently. Apply this preference to all card types and to
  creation and revision reviews. Use the proposed fields and real templates/CSS;
  text summaries, tables and raw fields are supplementary review aids. Keep
  Janki Assistant as the entry point, linking or embedding the preview. See
  `docs/CARD_DESIGN.md` for the durable specification; the reference HTML is a
  generated artifact. This presentation preference does not add approval gates
  or make viewing a preview count as approval.
- Recognition cards are enabled by default.
- Production and reading cards are independently configurable per deck.
- Avoid producing several nearly identical cards from one note.
- Keep the note type and enabled card set stable after real review history exists.
- Make cards legible on AnkiMobile and in dark mode.
- Include an `Open in Shirabe` link, but treat the deep-link URL scheme as provisional until tested on the installed app version.

## Data lifecycle

1. `data/inbox/`: untouched source exports.
1b. `data/operations.json` and `data/.pending/`: the paid-provider-call journal
   and the exact answers it captures, written **before** each call is dispatched
   and **before** its reply is parsed or decoded. The journal is committed — it
   is the only durable record that a call was billed, and an entry in
   `outcome_unknown` or `result_captured` is money someone still has to make a
   decision about. A captured artifact is a *recovery buffer*, not an archive:
   once its operation reaches `committed` the answer has reached that operation
   kind's durable destination. Extraction becomes staging; coverage becomes its
   recorded verdict; Realtime audio becomes the exact `pending_audio` stage and
   WAL row. A successful Realtime operation then forgets its captured envelope;
   `data/staging/done/` holds extraction's eventual durable copy. Read an
   unfinished reply only through `janki operations --show-reply ID`: a complete
   captured artifact passes through byte-for-byte; an incomplete streaming
   response becomes a deterministic JSON view that preserves every exact
   committed UTF-8 frame payload and boundary. The reader does not publish a
   private write-ahead reply, adopt a replacement or unjournalled crash
   extension, settle the call, or change the operation. Ending the call adopts
   the sole valid response frame that may have been fsynced just past its
   journal head. An exact rerun may seal already-durable terminal frames and
   strengthen `outcome_unknown` to `result_captured`, but never redispatches.
   Nonempty frames from a call that may have been sent and never became
   committed output require an explicit `operations --forget --force`
   decision. An entry carrying a
   `cleanup` intent is a durable, retryable forget decision: rerun its ordinary
   `janki operations --forget`
   command until the entry disappears; it no longer blocks a new paid call but
   remains listed until cleanup succeeds. Cleanup retires only exact names in
   the still-bound pending-directory inode. A checkout that makes that namespace
   missing or replaces it is preserved: janki touches no replacement names and
   never searches for the moved directory before closing the unreachable
   binding. Never delete its evidence or edit the intent by hand. Never delete
   either by hand while an operation is unfinished, and never re-dispatch an
   `outcome_unknown` one — a fresh charge needs fresh authority.
1c. `data/assistant/` (the `[paths].assistant_dir` default): the durable
   ordinary-Assistant turn store — **committed**. Before an Assistant call is
   dispatched, its exact user message, disclosed bounded context,
   provider/model identity and request fingerprint are written here as a
   request manifest. After the journal has captured the paid reply, that same
   manifest records the exact decoded answer and provenance before the
   operation becomes `committed`. An unfinished request remains governed by
   its operation-journal entry; never delete or edit it by hand. ChatKit
   threads are convenience views of these repository records, not their
   replacement, and this store never grants revision or content authority.
1d. `data/extraction_batches/` (derived beside the configured operations file):
   durable manifests and confirmed-execution receipts for extraction batches — **committed**
   repository state, not a `data/.pending` recovery buffer. A manifest binds
   each child's expectations and reserved operation id, the source ancestry and
   labels its parts came from, the batch's concurrency limit, and the
   provenance of every retry. It never flattens the children: each page or
   slice keeps its own staging document with its own request, coverage,
   candidate-accounting and pattern metadata, and a reference to the full
   parent source is documentary — never a substitute for the sliced input whose
   bytes were actually sent. A reserved child that was never sent may resume
   under its original authority after fresh binding checks. Recovering an
   already-captured reply makes no new call and keeps its existing operation;
   sending a replacement request needs fresh confirmation and a fresh operation
   id. An exact confirmed-execution receipt may finish interrupted cleanup and
   reservation after fresh binding checks; a request manifest alone grants no
   authority. Never hand-edit either file, and never delete one while its
   execution or any child operation is unfinished.
1e. `data/source_parts/` (derived beside the configured operations file the
   same way): committed, immutable publication receipts binding the parent
   source, the complete render recipe including renderer and encoder versions,
   and every planned part's exact name and expected hash. They are
   job-independent and may be reused by any later job or CLI run. An
   interrupted publication resumes against this expectation. Never edit one,
   and never delete one — publishing every part it names does not retire it —
   while any unfinished batch, finish authority or current job still
   references it. The published parts are ordinary immutable intake under
   `data/inbox/`: a derivative never overwrites a namesake, and the canonical
   parent source is never edited.
2. `data/normalized/`: mechanical conversion into the canonical schema.
3. `data/decks/`: curated deck definitions and human edits.
4. `data/staging/`: rows an import held back for a human — **committed**, so a
   reading typed in by hand is recoverable. `janki promote` normally archives
   the file to `data/staging/done/` and deletes the original itself once every
   row has landed; do not delete one by hand, which skips the archive. Normally,
   a file that survives a promote has rows still held back, and it says so. The
   exception is a staged `enrich --ai` review whose record changes landed but
   whose ledger save failed: the live staging file is kept because it is the
   only recoverable model attribution. Once the ledger is writable, rerun the
   same promote command so it records that attribution, archives, and deletes
   the file; never delete this recovery artifact by hand. Source
   extraction metadata fingerprints the source, style guide, task template,
   labelled data turn, response schema, and complete request; the pattern copy
   carries the same provenance. `janki workbench` is a localhost-only human
   authority surface: a checked card immediately binds the exact Japanese
   examples displayed to their content fingerprints, while the manual
   `staging-review` sentinel remains available for a reviewer editing YAML by
   hand. Its separate pattern-set checkbox writes only the matching store
   entry's existing `reviewed` mark. A human can edit or remove proposed cards,
   assign them to an explicitly chosen study deck, record a scoped
   repository-owner coverage decision with their reason, and invoke the same
   promotion transaction as the CLI. Exact one-use consent can dispatch
   journaled extraction or a separately named paid coverage check over the
   named source; each paid operation is re-planned and bound to the exact
   source, request and staging state described before the click. A model
   coverage verdict records its model and prompt provenance. The workbench
   never manufactures a CLI approval flag or makes an automatic identity,
   deck, coverage, review or promotion decision. Extraction and enrichment
   require a nonblank meaning list, complete fields on every returned example,
   and an explicit usage note that may be empty. Schema-v5 extraction
   additionally requires exactly one polite and one casual example plus
   source-kind evidence before the answer can be staged; enrich keeps
   example-list cardinality flexible so it can complete only the unoccupied
   card slots. These are structural card contracts, not Japanese audits.
   Schema-v4 and newer extraction also writes
   machine-owned
   `candidate_accounting`: coverage v2 binds its parsed/canonical/unusable/
   duplicate counts and fingerprint, and every stable-ID collision group keeps
   all parsed schema proposals in original order. Reviewers may still edit,
   delete, or re-identify rows under `records`; never edit that accounting or
   backfill it onto a schema-v3/coverage-v1 paid artifact. It survives partial
   promotion and the done archive unchanged. Large `enrich --ai` staging
   records complete request and input fingerprints per record. Its
   `field_replacements` hashes
   bind each authorized field to the record id, field name, and exact old wire
   value — not the proposed value, which a reviewer may improve. If any bound
   old value changes before promote, the whole staged replacement is stale and
   nothing lands.
   A `revise` answer is a separate JSON proposal in this directory. It binds
   the exact selected deck bytes, owner instruction, model request and old
   editable values. Source promotion ignores it: the workbench shows its
   old/new deck diff, and a separate owner action revalidates that binding
   before writing the deck. Audio and build remain later, separately planned
   actions.
5. `data/staging/done/`: canonical promoted rows as they actually landed, with
   the staging metadata preserved and archival count/review text added —
   **committed**, the record of what each source yielded and what a reviewer
   accepted. An exact retry after archive success is idempotent; divergent
   same-run live/archive content refuses rather than overwriting either copy.
6. `data/kanji.json`: machine-written kanji reference data — **committed** and
   replaced by `janki kanji --refresh`; do not add fields by hand because the
   source schema does not preserve unknown keys.
6b. `data/jpdb_readings.json`: machine-written JPDB reading facts — **committed**.
    Preserve source labels, contextual groups, displayed percentages and bounds,
    missing values, reading-bound examples, URLs, retrieval times and hashes.
    Reuse saved facts by default; fetch missing requested characters on demand
    and refresh only explicitly. Full HTML is a private local cache outside the
    repository, not committed study content. Builds never fetch it.
6c. `data/kanji_notes.json`: curated character notes — **committed**, separate
    from vocabulary and refreshable reference data. Identity is
    `kanji:<character>` and does not include readings. A refresh must not replace
    curated notes implicitly. Explicit kanji preparation can populate the local
    raw cache, but canonical facts, notes and deck creation stay in one exact
    confirmed apply batch; its durable receipt owns interrupted-work recovery.
7. `data/patterns.json`: machine-written from `janki extract`'s rich source
   answer — **committed**. `janki patterns` only lists entries and marks them
   reviewed. A reviewed mark is a human judgement: extracting that source again
   preserves its reviewed store entry unless `extract --force` explicitly
   replaces it with the new unreviewed answer. The staging file still keeps the
   new answer, so neither half of the already-paid source response is lost.
8. `data/ledger.json`: machine-written operational state — **committed**, never
   hand-edited. `janki status --rebuild` reconstructs what records and media
   still prove, and is a no-op on a healthy repository: a source reference's
   `ref` is the record's own `source.imported_from`, so any divergence has
   rebuild append a duplicate. Its sparse top-level `pending_audio` block is a
   write-ahead record for paid clips whose guarded record/canonical-ledger
   transaction has not finished. Rebuild cannot recreate that exact request,
   profile, and byte hash, so never edit or remove it by hand.
9. `data/media/`: generated audio, identity-addressed with content/profile
    currency in the ledger — **committed**, so a rebuild is free
    (`docs/PROJECT_PLAN.md` design principle 6). OpenAI Realtime sentence calls
    first fsync every exact WebSocket text frame to an operation-bound spool
    before JSON parsing, then capture the terminal envelope through the
    operation journal; only its decoded finite WAV may enter the audio WAL
    below. While `pending_audio` exists,
    its paired paid bytes live under `data/media/audio/.pending/*.stage`; they
    are committed recovery data, protected from prune, and finalized by
    rerunning the exact matching `janki audio` command. Do not delete them.
    Successful audio runs automatically remove a stage only after proving that
    neither a WAL row nor any current exact request can still claim it.
10. `data/review.json`: the retired review subsystem's store — **committed**
    as history and read by nothing since M8.2. Do not extend it or wire a
    reader to it; it is a record of what a model once said, not state.
11. `dist/`: generated `.apkg` and preview files — **not** committed.

Only `dist/`'s *contents* are disposable — the directory itself is held open
by a tracked `.gitkeep` like the others. Everything under `data/` is tracked:
the media, the staging files, and four `.gitkeep`s (`data/inbox/shirabe/`,
`data/media/`, `data/staging/`, `data/assistant/`) so those directories exist
in a fresh clone whether or not they have contents yet — because the
repository, not Anki's database and not an uncommitted working tree, is the
source of truth.

A new import must not erase manually curated examples, notes, conjugations, or furigana.

## Repository review hooks

- `scripts/janki-review.sh` is the tracked implementation used by the
  post-commit advisory review and the pre-push review gate.
- **Code review and Japanese-content review are separate.** The code reviewer
  excludes `data/**` and `dist/**`; a content-only commit or push must not start
  Claude at all, and a mixed range exposes only its non-content paths to the
  code reviewer. It may inspect code that preserves or renders Japanese fields,
  but it never judges whether repository Japanese, readings, furigana,
  translations, examples, or usage notes are correct or natural. Content is
  reviewed through the workbench and owner acceptance. This owner decision
  supersedes older handoff instructions that said every commit, including a
  content-only submission, needed the code-reviewer agent.
- `scripts/bootstrap.sh` installs thin shims into Git's hooks directory. Run
  `scripts/install-review-hooks.sh` directly to refresh only those shims.
- Never overwrite an unrelated local hook. The installer refuses a conflict so
  the existing hook and the tracked shim can be combined deliberately.
- To disable both reviews in one clone, create `.claude/hooks/DISABLED` with a
  short reason. The hooks announce that state on every commit and push. Delete
  the marker to re-enable them.
