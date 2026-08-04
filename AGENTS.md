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
- Run `ruff check .`, `pytest`, and at least one sample deck build before considering work complete.

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
4. `dist/`: generated `.apkg` and preview files.

A new import must not erase manually curated examples, notes, conjugations, or furigana.
