# Project Plan

## Goal

Create a reproducible local pipeline that converts Japanese study material into
Anki decks without making Anki's database the only copy of the learning content.

## Architecture

```text
Shirabe CSV / manual YAML / class material
                  |
                  v
        importers and normalization
                  |
                  v
      canonical vocabulary records
                  |
                  v
        validation and curation
                  |
                  v
     templates + deterministic builder
                  |
                  v
             .apkg output
                  |
                  v
        Anki Desktop -> AnkiMobile
```

## Design principles

1. Raw exports are immutable.
2. Normalization is mechanical and reproducible.
3. Human curation lives in readable JSON or YAML.
4. Validation fails loudly rather than dropping data.
5. Note IDs and Anki GUIDs are deterministic.
6. Generated packages can always be deleted and rebuilt.
7. Japanese-learning conventions are explicit in `AGENTS.md` and the style guide.

## Milestones

### Phase 1: usable starter

- CSV inspection.
- Alias-based Shirabe import.
- Canonical schema.
- Validation.
- Recognition, production, and reading templates.
- `.apkg` generation.
- Static preview.

### Phase 2: real Shirabe fixture

- Add an export from the installed Shirabe version as an anonymized test fixture.
- Adjust aliases and parsing to match its exact columns.
- Verify the `shirabelookup://` deep link on iOS.

### Phase 3: curation helpers

- Commands for selecting records into decks.
- Interactive handling of ambiguous readings or definitions.
- Better merge reports showing added, updated, unchanged, and conflicting rows.

### Phase 4: richer cards

- Optional embedded audio.
- Images and kanji-writing cards.
- Grammar and sentence note types.
- Transitivity-pair and conjugation practice cards.

### Phase 5: direct local sync

- Optional AnkiConnect integration.
- Dry-run diff before modifying a live Anki collection.
- Explicit backup and rollback instructions.

## Non-goals for the starter

- Automatic pitch-accent generation.
- Automatic furigana segmentation for arbitrary mixed-kanji words.
- Bidirectional synchronization with Shirabe.
- Direct modification of Anki's SQLite collection.
