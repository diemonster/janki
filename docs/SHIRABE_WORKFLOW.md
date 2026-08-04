# Shirabe Jisho Workflow

## Intended use

Shirabe acts as a convenient vocabulary inbox on iPhone. Anki remains the spaced
repetition system, and this repository holds the editable master data.

## Capture

1. Create a Shirabe bookmark folder named `Anki Inbox`.
2. Save words worth studying to that folder.
3. Export the folder as CSV periodically.
4. Keep exports unchanged under `data/inbox/shirabe/`.

Use dated names:

```text
2026-08-04-anki-inbox.csv
```

## Inspect before importing

```bash
janki inspect data/inbox/shirabe/2026-08-04-anki-inbox.csv
```

The command shows:

- encoding and delimiter handling;
- detected field mapping;
- unknown columns that will be preserved;
- a few source rows.

The exact Shirabe schema may vary by app version. If the mapping is wrong, add a
header alias in `src/japanese_anki/importers/shirabe.py` and add the real export
as an anonymized test fixture.

## Import

```bash
janki import-shirabe data/inbox/shirabe/2026-08-04-anki-inbox.csv
```

By default, the importer merges into `data/normalized/vocabulary.json`. Existing
human enrichment is retained when the new export does not provide a replacement.

Use `--replace` only when you intentionally want to recreate normalized output
from one source file.

## Curate

After import, add or correct:

- furigana segmentation;
- romaji;
- concise meanings;
- examples;
- part of speech and verb group;
- conjugations;
- useful tags.

A future command can automate moving selected records into dedicated curated
deck files. The starter personal deck simply reads every normalized record.

## Build

```bash
janki build data/decks/personal-vocabulary.yaml
```

## Open from AnkiMobile

The card template includes a provisional deep link:

```text
shirabelookup://search?w={{ShirabeQuery}}
```

Test this on the installed version of Shirabe. If it fails, update the link in
`templates/japanese-study/*.html`; no vocabulary data changes are required.

## No bidirectional synchronization

The starter does not synchronize Shirabe bookmarks, Anki review state, or edits
back to Shirabe. The deliberate one-way flow is:

```text
Shirabe capture -> repository curation -> Anki review
```
