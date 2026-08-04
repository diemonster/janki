# Japanese Anki

A local, version-controlled pipeline for turning Japanese vocabulary—especially
Shirabe Jisho bookmark exports—into maintainable Anki decks.

The repository is the source of truth. Anki packages are reproducible outputs.

## What is included

- A `janki` command-line application.
- Flexible CSV inspection and Shirabe import.
- A canonical JSON/YAML vocabulary schema.
- Validation with row- and record-level diagnostics.
- Deterministic Anki note GUIDs.
- Configurable recognition, production, and reading cards.
- Mobile-friendly card templates with furigana, hidden romaji, and a Shirabe link.
- A static HTML preview command.
- Tests, fixtures, project instructions, and workflow documentation.

## Bootstrap on macOS or Linux

```bash
unzip japanese-anki.zip
cd japanese-anki
chmod +x scripts/bootstrap.sh
./scripts/bootstrap.sh
source .venv/bin/activate

# Optional: establish your own local history
git init
git add .
git commit -m "Bootstrap Japanese Anki pipeline"
```

The bootstrap script installs the project in an isolated virtual environment,
runs tests, validates the sample deck, and builds `dist/sample-verbs.apkg`.

Manual equivalent:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
pytest
janki validate data/decks/verbs.yaml
janki build data/decks/verbs.yaml
```

## First Shirabe import

Export a small Shirabe bookmark folder as CSV and copy it into the raw inbox:

```bash
cp ~/Downloads/shirabe-export.csv data/inbox/shirabe/
```

Inspect the file before importing:

```bash
janki inspect data/inbox/shirabe/shirabe-export.csv
```

Import it into the canonical normalized database:

```bash
janki import-shirabe data/inbox/shirabe/shirabe-export.csv
```

Validate the result:

```bash
janki validate data/normalized/vocabulary.json
```

Build the personal vocabulary deck, which reads all normalized records:

```bash
janki build data/decks/personal-vocabulary.yaml
```

The package will appear under `dist/` and can be imported into Anki Desktop.
Sync Anki normally to make it available in AnkiMobile.

## Common commands

```bash
# Show detected CSV columns and a few source rows
janki inspect FILE.csv

# Import or merge a Shirabe CSV
janki import-shirabe FILE.csv

# Replace normalized output rather than merging it
janki import-shirabe FILE.csv --replace

# Validate one data or deck file
janki validate PATH

# Validate every configured deck
janki validate

# Build one deck
janki build data/decks/verbs.yaml

# Build every YAML deck in data/decks
janki build --all

# Generate a browser preview
janki preview data/decks/verbs.yaml
```

## Data model

Canonical records look like this:

```yaml
id: "word:話す:はなす"
expression: "話す"
reading: "はなす"
furigana: "話[はな]す"
romaji: "hanasu"
meanings:
  - "to speak"
  - "to talk"
part_of_speech: "verb"
verb_group: "godan"
transitivity: "intransitive"
examples:
  - japanese: "毎日、妻と日本語で話します。"
    furigana: "毎日[まいにち]、妻[つま]と日本語[にほんご]で話[はな]します。"
    romaji: "Mainichi, tsuma to Nihongo de hanashimasu."
    english: "I speak Japanese with my wife every day."
conjugations:
  plain: "話す"
  negative: "話さない"
  past: "話した"
  past_negative: "話さなかった"
  te_form: "話して"
  potential: "話せる"
  passive: "話される"
tags:
  - "verb"
  - "godan"
  - "shirabe"
usage_notes: "Used for speaking or talking with someone."
source:
  type: "shirabe"
  imported_from: "shirabe-export.csv"
  row: 2
```

## Editing and review-history safety

Each record has a deterministic ID based on the expression and reading. The Anki
builder derives a stable note GUID from that ID. Rebuilding and importing the
same deck should therefore update matching notes rather than create duplicates.

Do not casually change IDs, the configured deck ID, or the enabled card set after
you have accumulated real review history. See `docs/CARD_DESIGN.md`.

## VS Code / Codex usage

Open the repository root in VS Code. `AGENTS.md` contains the standing project
instructions, and the documents under `docs/` contain design context. Useful
requests include:

> Import the newest CSV in `data/inbox/shirabe`, preserve all source columns,
> report ambiguous rows, and run the importer tests.

> Add natural Genki-level examples to the new records without exposing romaji on
> the front of any card.

> Review this deck for duplicate concepts and cards that are too similar.

## Important limitations

- Shirabe's exported header names may differ between app versions. The importer
  recognizes many common names and preserves unknown columns, but your first real
  export may require a small alias addition.
- The Shirabe deep link currently uses `shirabelookup://search?w=...`. Test it on
  your installed iPhone version before relying on it.
- The project does not generate trustworthy pitch accent or synthesized audio by
  itself. Those should come from a reliable source or an explicit later tool.
