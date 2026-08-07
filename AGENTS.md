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
6. `data/ledger.json`: machine-written operational state — **committed**, never
   hand-edited. `janki status --rebuild` reconstructs what records and media
   still prove.
7. `dist/`: generated `.apkg` and preview files — **not** committed.

Only `dist/` is disposable. Everything under `data/` is tracked, including the
two directories that start empty (`.gitkeep`), because the repository — not
Anki's database and not an uncommitted working tree — is the source of truth.

A new import must not erase manually curated examples, notes, conjugations, or furigana.
