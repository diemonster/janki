# Japanese Anki Project Instructions

## Purpose

This repository converts Japanese vocabulary and study material into custom
Anki decks. Shirabe Jisho CSV exports are a primary input source. The repository
is the durable source of truth; generated `.apkg` files are build artifacts.

## Development rules

- Use Python 3.11 or newer.
- Keep parsing, normalization, validation, and Anki generation separate.
- Prefer the standard library unless an external package materially simplifies the task.
- Add or update tests for every supported input format and behavioral change.
- Never modify files under `data/inbox/`.
- Treat `dist/` as generated output.
- Use deterministic note IDs and GUIDs so rebuilt decks update notes instead of duplicating them.
- Surface source filename and row number in import errors.
- Never silently discard an input row or unknown source column.
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

- Follow `docs/HARDENING.md` when real deck work exposes a defect.
- Classify a correction as `content-specific` or `systemic` before you
  generalize it. When uncertain, keep it content-specific and staged.
- A systemic defect is not complete when only the current row is corrected.
  Preserve the reproduction and complete the finding, case, production fix,
  full replay, and `make gates`, or record the finding as open or deferred.
- Do not manufacture a general validator from one content-specific correction.
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
