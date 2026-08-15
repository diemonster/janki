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
- **Prompts are templates and belong in files**, stated plainly enough that
  someone can read what the model was asked for without reading Python.
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
- Use deterministic note IDs and GUIDs so rebuilt decks update notes instead of duplicating them.
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

## Hardening rules

*Scope (2026-08-15): a finding records a defect in janki's own machinery —
importers, packaging, provenance, repairs. A wrong or thin model answer is a
template problem, never a finding. M8.3 slims this corpus to plain tests.*

- Follow `docs/HARDENING.md` when real deck work exposes a defect.
- Classify a correction as `content-specific` or `systemic` before you
  generalize it. When uncertain, keep it content-specific and staged.
- A systemic defect is not complete when only the current row is corrected.
  Preserve the reproduction and complete the finding, case, production fix,
  full replay, and `make gates`, or record the finding as open or deferred.
- Do not manufacture a general validator from one content-specific correction.
- Record each systemic defect in `quality/findings.yaml`. Record a
  content-specific correction only in its pilot report. Do not create a finding
  for it.
- Run `janki harden status` after you edit a finding, pilot, oracle, or case.
  Run `janki harden replay` after you change production behavior or a case.
  Status is read-only. Edit reviewed YAML by hand so comments and decisions
  remain visible.
- Do not mark a finding as `fixed` until a passing gating case and a production
  fix reference exist. Case and finding links must point both ways.
- Automatic repairs are default-deny. Their only allowed target fields are
  `furigana`, `romaji`, `examples[*].furigana`, `examples[*].romaji`, `audio`,
  `image`, and `frequency_rank`. The canonical
  `source.raw_fields["janki_repairs"]` entry is required provenance, not a
  repair target. Every other record or source field is denied unless a reviewed
  contract change adds it with an invariant and adversarial fixtures.
- Never repair or propose a change to an existing record's `id`, `expression`,
  or `reading`. These fields determine Anki identity and review history.
- The user owns live-eval consent, redistribution approval for owner-provided
  material, human unit-oracle acceptance, manual coverage acceptance,
  repair-proposal acceptance, ambiguous new-identity resolution,
  existing-identity migration, accepted-risk approval, and baseline acceptance.
  An agent must not infer, generate, grant, or widen one of these decisions. It
  must not answer an approval prompt as the user.
- An agent may record a user decision only when the approval names every exact
  fingerprint, scope, decision, and reason required by `docs/HARDENING.md`. A
  broad request to finish work is not an approval. If exact approval is absent,
  keep it false or missing and stop that route or use material that does not
  need the approval.

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
   reading typed in by hand is recoverable. Delete a staging file once its rows
   have moved into `data/normalized/` and `janki status --rebuild` has run.
5. `data/media/`: generated audio, content-addressed — **committed**, so a
   rebuild is free (`docs/PROJECT_PLAN.md` design principle 6).
6. `data/kanji.json`: machine-written kanji reference data — **committed** and
   replaced by `janki kanji --refresh`; do not add fields by hand because the
   source schema does not preserve unknown keys.
7. `data/ledger.json`: machine-written operational state — **committed**, never
   hand-edited. `janki status --rebuild` reconstructs what records and media
   still prove.
8. `dist/`: generated `.apkg` and preview files — **not** committed.

Only `dist/` is disposable. Everything under `data/` is tracked, including the
two directories that start empty (`.gitkeep`), because the repository — not
Anki's database and not an uncommitted working tree — is the source of truth.

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
