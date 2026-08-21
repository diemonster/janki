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
- **There are two card-writing model calls.** `extract` reads a source once and
  returns complete proposed cards plus the patterns the source teaches;
  `enrich --ai` receives a bare vocabulary record and returns the same meanings,
  examples, and usage-note shape. Pattern discovery and meaning improvement are
  parts of those rich answers, never follow-up paid passes. `janki patterns`
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
  same rule (`config.py` defaults, `claude_client.DEFAULT_EFFORT`).
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
- Romaji is a temporary learning aid and must remain hidden behind a disclosure control.
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

- Recognition cards are enabled by default.
- Production and reading cards are independently configurable per deck.
- Avoid producing several nearly identical cards from one note.
- Keep the note type and enabled card set stable after real review history exists.
- Make cards legible on AnkiMobile and in dark mode.
- Include an `Open in Shirabe` link, but treat the deep-link URL scheme as provisional until tested on the installed app version.

## Data lifecycle

1. `data/inbox/`: untouched source exports.
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
   entry's existing `reviewed` mark. It never edits content, promotes rows,
   accepts coverage, or calls a model. Both
   card-writing paths require a nonblank
   meaning list, complete fields on every returned example, and an explicit
   usage note that may be empty. Schema-v5 extraction additionally requires
   exactly one polite and one casual example plus source-kind evidence before
   the answer can be staged; enrich keeps example-list cardinality flexible so
   it can complete only the unoccupied card slots. These are structural card
   contracts, not Japanese audits. Schema-v4 and newer extraction also writes
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
5. `data/staging/done/`: canonical promoted rows as they actually landed, with
   the staging metadata preserved and archival count/review text added —
   **committed**, the record of what each source yielded and what a reviewer
   accepted. An exact retry after archive success is idempotent; divergent
   same-run live/archive content refuses rather than overwriting either copy.
6. `data/kanji.json`: machine-written kanji reference data — **committed** and
   replaced by `janki kanji --refresh`; do not add fields by hand because the
   source schema does not preserve unknown keys.
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
    (`docs/PROJECT_PLAN.md` design principle 6). While `pending_audio` exists,
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
the media, the staging files, and three `.gitkeep`s (`data/inbox/shirabe/`,
`data/media/`, `data/staging/`) so those directories exist in a fresh clone
whether or not they have contents yet — because the repository, not Anki's
database and not an uncommitted working tree, is the source of truth.

A new import must not erase manually curated examples, notes, conjugations, or furigana.

## Repository review hooks

- `scripts/janki-review.sh` is the tracked implementation used by the
  post-commit advisory review and the pre-push review gate.
- `scripts/bootstrap.sh` installs thin shims into Git's hooks directory. Run
  `scripts/install-review-hooks.sh` directly to refresh only those shims.
- Never overwrite an unrelated local hook. The installer refuses a conflict so
  the existing hook and the tracked shim can be combined deliberately.
- To disable both reviews in one clone, create `.claude/hooks/DISABLED` with a
  short reason. The hooks announce that state on every commit and push. Delete
  the marker to re-enable them.
