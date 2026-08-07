# Japanese Anki

A local, version-controlled pipeline for turning Japanese vocabulary—especially
Shirabe Jisho bookmark exports—into maintainable Anki decks.

The repository is the source of truth. Anki packages are reproducible outputs.

## What is included

- A `janki` command-line application.
- Flexible CSV inspection and Shirabe import.
- A canonical JSON/YAML vocabulary schema.
- Validation with row- and record-level diagnostics.
- Curation-safe merging: an import fills empty fields and reports conflicts
  instead of overwriting your edits.
- A word ledger (`data/ledger.json`) and a `janki status` report over it.
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

The import merges: it adds records janki has never seen, fills fields that are
empty on the ones it has, and prints what it did without overwriting anything
you have edited (see "How an import merges"). Rows whose reading it cannot use
are held back rather than imported (see "Rows held back for reading review").

Validate the result:

```bash
janki validate data/normalized/vocabulary.json
```

Build the personal vocabulary deck, which reads the normalized records (all of
them except the few its `exclude_ids` leaves to the starter deck):

```bash
janki build data/decks/personal-vocabulary.yaml
```

The package will appear under `dist/` and can be imported into Anki Desktop.
Sync Anki normally to make it available in AnkiMobile.

## Common commands

```bash
# Show detected CSV columns and a few source rows
janki inspect FILE.csv

# Import a Shirabe CSV: add new records, fill empty fields, keep existing values
janki import-shirabe FILE.csv

# Let this import overwrite the named fields instead of only filling empty ones
janki import-shirabe FILE.csv --prefer-incoming meanings,part_of_speech

# Discard the normalized records and write only this import
# (asks for confirmation on a terminal; --yes answers it)
janki import-shirabe FILE.csv --replace
janki import-shirabe FILE.csv --replace --yes

# Validate one data, deck, or staging file
janki validate PATH

# Validate every configured deck
janki validate

# Build one deck
janki build data/decks/verbs.yaml

# Build every YAML deck in data/decks
janki build --all

# Generate a browser preview
janki preview data/decks/verbs.yaml

# Report on the records and the ledger
janki status
janki status --unexported --missing-audio --duplicates

# Move a deck's inline notes into the normalized records file
janki migrate-inline data/decks/verbs.yaml
```

## How an import merges

An import never overwrites what is already in
`data/normalized/vocabulary.json`. Field by field:

- **Existing wins.** An import only fills fields that are currently empty.
  Empty means `""`, `[]`, `{}`, or absent — `0` and `false` are values, not
  holes.
- **`tags` is a union**, and the record's `source` block belongs to whichever
  import saw the record first. Later sightings go to the ledger instead, so
  re-importing never erases where a word originally came from.
- **Conflicts are reported, not resolved.** Where both sides hold a different
  non-empty value, the existing value stays and the import prints the pair:

```text
Merge result: 0 added, 0 filled, 1 unchanged, 1 conflicting
Conflicts (existing values kept; --prefer-incoming FIELD takes the import's):
  word:話す:はなす meanings: existing to speak | incoming to talk
```

`--prefer-incoming FIELD[,FIELD]` opts individual fields back into
overwriting, which is what you want for a deliberate refresh from a corrected
export. It accepts any content field of a record — everything except the five it
refuses: `id`, `expression`, `reading`, `tags`, and `source`. (The accepted
set is derived from the record schema rather than hard-coded, so it grows
with it; `janki import-shirabe --prefer-incoming nope` lists the current
names in its error.) Those five are not preferences: the first three are the
record's identity (the ID is derived from expression and reading, and the Anki
GUID from the ID), tags are always unioned, and source sticks with the first
import. A conflict on one of those — an `expression` or `reading` edited
without re-minting the ID, say — is a hand fix, and the conflict line says so.

`--replace` still means "discard the existing records and write only this
import". It now asks for confirmation when it is run on a terminal and the
output file holds records (or exists but cannot be read); `--yes` answers that
prompt. A non-interactive run proceeds without asking — the prompt is
fat-finger protection, not a lock, and the records are in git either way.

The ledger follows the records: every discarded record's ledger entry is
dropped too, and the summary says how many. That is the point of the prompt's
wording — a record that comes back in a later import comes back as new, with a
new `added_at` and no memory of the decks it used to be exported to. Leaving the
entries behind would be worse: `janki status` would over-report the collection
forever, and once `build --only-new` reads export state a dead entry would keep
a re-imported record out of its deck.

## Rows held back for reading review

A record's ID is minted from its expression *and* its reading
(`word:話す:はなす`), and the Anki GUID is derived from that ID. A row with a
missing reading would mint `word:食べ物:`, and a row whose reading column holds
kanji rather than kana would mint something that looks well formed and is not.
Neither can be corrected later without changing the ID, and changing an ID
orphans the Anki review history behind it.

So the importer holds those rows back instead of importing them. They are
written to `data/staging/shirabe-<csv-stem>-needs-reading.yaml` — with the
reason recorded per row and the remedy in a `review_notes` block at the top —
and the import prints where they went and why. The rest of the import lands
normally.

To resolve them, edit that staging file:

1. Fill in `reading` with kana for each row worth keeping.
2. Delete that row's `id:` line. The ID recorded there is the malformed one; an
   empty ID is re-minted from expression + reading when the file is read.
3. Delete the rows not worth keeping.
4. Run `janki validate data/staging/<file>.yaml`. It lists every row still
   malformed and exits non-zero until the file is clean.
5. Move the surviving records into `data/normalized/vocabulary.json`, run
   `janki status --rebuild` so the ledger learns about them, and delete the
   staging file. Without the rebuild the records exist but the ledger has never
   heard of them, so `janki status` under-reports the collection and a later
   import of the same word registers nothing.

`janki promote` will do step 5 for you; it ships in Milestone 3. There is no
reason to wait for it — a resolved staging file is finished work, and leaving it
in place only means the next import of the same export keeps mentioning it.

Re-running the same import never overwrites a staging file that already exists.
While the file still has errors it reports the path and the count and tells you
to resolve it; once `janki validate` is happy with it, the message changes to
say the review is finished and to move the records across, because re-running an
import can never consume that file itself.

`data/staging/` is committed, like everything else under `data/`. Commit a
review in progress and a hand-typed reading is recoverable; leave it
uncommitted and it exists only in your working tree.

## The ledger and `janki status`

`data/ledger.json` is machine-written, git-committed, and deliberately
minimal. It tracks *operational* state per record: when the word arrived,
every source it has been seen in, whether it has been enriched, whether audio
exists for it, and which decks it has been exported to. Card content stays in
`vocabulary.json` and never appears here; timestamps and provenance history
stay here and never appear in the records. Imports and `migrate-inline` write
it — you do not edit it by hand.

`janki status` reads both and summarizes them:

```text
Records: 3 (3 in data/normalized/vocabulary.json, 0 inline in deck files)
By source type: manual 3
Ledger: data/ledger.json — 3 record entries
Never exported: personal-vocabulary 0 of 0, verbs 3 of 3
Missing word audio: 3 of 3
Stale audio: 0
Missing enrichment: 2 (no example sentence, or no usage notes)
Missing pitch accent: 3
Staged for review: none
```

The detail flags:

```bash
janki status --unexported     # per deck, the record ids it was never built with
janki status --missing-audio  # record ids with no word audio
janki status --duplicates     # records that look like the same word twice
janki status --staged         # ids waiting in data/staging, and why
janki status --format ids     # bare ids on stdout, one per line, for piping
janki status --rebuild        # recover the ledger from records and media on disk
```

`--duplicates` looks for one expression under two IDs, and for one reading
under both a kanji and a kana spelling. Resolution is manual: pick the
surviving ID and accept that the loser's review history goes with it, because
re-IDing a record would orphan that history anyway.

`--format ids` prints nothing but record IDs on stdout — every warning and
every human-readable line moves to stderr — so it can be piped. Combined with
a detail flag it emits that flag's IDs; on its own it emits every record.

`--staged` lists the rows an import held back for reading review, with the
reason each was held. The summary always carries a `Staged for review:` line so
a review queue cannot sit unnoticed.

`--rebuild` reconstructs what is still provable after a lost or corrupted
ledger: source references from each record's own `source` block, and audio
entries from the files under `data/media`. Export state is not reconstructible
— the ledger was the only place that held it. Nothing is lost when it goes,
because GUIDs are deterministic and re-exporting a note updates it rather than
duplicating it. It is also the command that teaches the ledger about records you
moved into `vocabulary.json` by hand, and running it when nothing is missing is
a no-op: every writer records a source reference in the same shape `--rebuild`
reconstructs, so it never grows the file.

Parts of the ledger are still unwritten while the rest of the pipeline is
built. Nothing writes `enriched` at all, and the only writer of `audio` is
`--rebuild` over files you placed under `data/media` yourself (`janki enrich`
and `janki audio` arrive in Milestones 2, 4 and 5). `janki build` does not yet
mark records exported either, so `--unexported` currently lists everything.

## Inline deck notes

A deck YAML may carry `notes:` of its own. That was how the starter deck began,
and it is a dead end: enrichment and audio only ever write to
`data/normalized/vocabulary.json`, so a record living inside a deck file can
never gain examples, pitch accent or audio.

`janki migrate-inline DECK.yaml` moves those notes into the normalized file
under the same IDs — GUIDs, and the review history behind them, are preserved —
and leaves the deck as a filter over shared records (`source:` plus an
`include_ids:` list pinning exactly what it exported before). A deck that
*already* read the normalized file keeps its own membership rule instead and
gets no `include_ids:` — pinning it would freeze out the imports it exists to
receive — and the command says so.

Any other deck reading the same normalized file would suddenly resolve the moved
records and emit notes with the migrated deck's GUIDs, so those decks get an
`exclude_ids:` entry and the command says which. A deck whose own `include_ids:`
already closes its membership gets none: an exclusion there could never change
what it exports. Running it again on a deck with no inline notes does nothing.

Inline notes that *override* a normalized record (a matching `id` plus the few
fields you want to change) remain supported and are not affected.

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
- The Shirabe deep link currently uses `shirabelookup://search?w=...`. That URL
  scheme is unverified against a real installed app — test it on your iPhone
  before relying on it.
- Records now carry pitch accent and a frequency rank, but nothing fills them
  in yet — `janki enrich --jpdb` does that later in Milestone 2, from jpdb's
  dictionary data. Generated audio arrives in Milestone 5. Neither is ever
  guessed: an empty field means the dictionary did not say, and `janki status`
  counts it as missing rather than inventing a value. See `docs/DESIGN_V2.md`.
