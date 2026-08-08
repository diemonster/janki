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

# jpdb (see "Working with jpdb" below)
janki jpdb ping
janki import-jpdb --deck "Textbook Vol. 1: Lesson 1"
janki enrich --jpdb
janki import-jpdb-reviews ~/Downloads/reviews.json
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

## Working with jpdb

[jpdb.io](https://jpdb.io) is used two ways, and they are independent. It is a
**source** of records (`import-jpdb`), and it is a **dictionary** for records
that came from somewhere else (`enrich --jpdb`). A third command,
`import-jpdb-reviews`, neither imports nor enriches — it marks what you already
study there, and is documented under
[Choosing what a deck contains](#choosing-what-a-deck-contains).

### Setup

Copy your API key from the jpdb.io settings page and put it in the environment.
It is read from `JPDB_API_KEY` and nothing else — janki does not read `.env`
files, and no key is ever written to the repository:

```bash
export JPDB_API_KEY='...'
janki jpdb ping
```

`jpdb ping` needs no project, so it works from anywhere. That makes it the thing
to run when a later command reports a rejected key.

### Importing decks

```bash
# One deck, by name (repeat --deck for several)
janki import-jpdb --deck "Textbook Vol. 1: Lesson 1"

# Every deck on the account
janki import-jpdb --all-decks

# The JPDB-Export userscript's CSV instead of the API
janki import-jpdb export.csv
```

Deck names are matched ignoring case and surrounding space, and nothing further:
a name that matches nothing lists the decks you do have, and one that matches two
decks is an error rather than a guess.

Each deck is imported as its own pass, so the summary, the merge result and the
ledger entries are per deck. A word in two decks is one record with a ledger
sighting for each, which is how you can still tell where it came from.

A deck import tags each record `jpdb` and `jpdb:<deck-name>`, the name flattened
for Anki — spaces and punctuation become hyphens, so `Textbook Vol. 1: Lesson 1`
tags `jpdb:textbook-vol-1-lesson-1` and 語彙・基礎 tags `jpdb:語彙-基礎`. A deck
YAML can then select one jpdb deck with `include_tags`. A CSV import gets the
plain `jpdb` tag only, since the file does not say which deck the words are
from.

An import fills empty fields and never overwrites your edits, exactly as
[described above](#how-an-import-merges) — `--prefer-incoming`, `--replace` and
`--yes` all work the same way here. With several decks, `--replace` applies to
the first pass only; applied to every pass, the second deck would discard the
first.

jpdb states a spelling, reading, meanings, pitch accent, frequency rank and part
of speech. Romaji and the conjugation table are computed from that by the same
rules everything else uses. Anything jpdb has no answer for is left empty for a
human rather than guessed — `janki status` counts those.

Entries whose reading janki cannot use are held back for review just like CSV
rows, into `data/staging/jpdb-<deck>-<fingerprint>-needs-reading.yaml`. See
[Rows held back for reading review](#rows-held-back-for-reading-review); the
fingerprint is there so two decks whose names differ only in punctuation cannot
land in one file.

### Enriching records from the dictionary

For records that came from Shirabe, a textbook or a photograph, jpdb can fill
what they are missing:

```bash
# Every record with an empty field
janki enrich --jpdb

# Just these records
janki enrich --jpdb word:話す:はなす word:食べる:たべる

# Let it overwrite named fields instead of only filling empty ones
janki enrich --jpdb --force-fields pitch_accent,frequency_rank
```

It fills `furigana`, `romaji`, `part_of_speech`, `verb_group`, `conjugations`,
`pitch_accent` and `frequency_rank`. It shows you every proposed change as a
diff and asks before writing; `--yes` skips the question.

A record whose enrichable fields are all filled is skipped without an API call.
A record with a field jpdb had no answer for is *not*, and is looked up again on
every run — a noun has no verb group and no conjugation table, so those stay
empty however many times you ask. Re-running is cheap on a collection of verbs
and roughly one call per word on a collection of nouns, which is worth knowing
before pointing it at a few thousand records.

Two things it will not do. It **never writes `reading`** — that is half of the
record ID, so a dictionary changing it would orphan the Anki review history
behind the record. And where your reading and jpdb's disagree, it warns and
writes nothing rather than picking: both may be right, since 一日 is both
いちにち and ついたち. When your reading *is* one jpdb lists, it asks jpdb again
with that reading pinned, so a homograph is filled from the right entry rather
than whichever one jpdb reached for first.

`meanings` is deliberately left alone. A record that came from a textbook
carries the meaning that textbook taught, and a dictionary is not better placed
to decide that.

### Getting a reading suggestion for held rows

A staging file full of rows with no usable reading is the slowest part of a
review, so jpdb can propose one for each:

```bash
janki enrich --jpdb --staging data/staging/shirabe-export-needs-reading.yaml
```

Each held row gains `suggested_reading` in its `raw_fields`. It is a proposal
and nothing more: the row stays held until you type the reading into `reading`
yourself and delete the row's `id:` line, because the ID it would mint is not
correctable afterwards. The suggestion is the reading of the form as written, so
an inflected entry gets its own reading rather than its dictionary form's.

Your file is edited, not rewritten — comments you left in it, and any key janki
does not know about, are still there afterwards.

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

## Choosing what a deck contains

A deck that reads `source:` gets every record in that file. Four optional keys
narrow it, all lists, all matched exactly:

```yaml
deck:
  name: Genki 1 — verbs
  source: ../normalized/vocabulary.json
  include_ids: [word:話す:はなす]   # only these records
  include_tags: [genki-1]          # only records carrying at least one
  exclude_ids: [word:食べる:たべる] # never these records
  exclude_tags: [jpdb-known]       # never records carrying any
```

They apply in that order, so an `include_` key chooses the pool and an
`exclude_` key removes from it: a record that is both included by tag and
excluded by id is excluded. Omitting all four means "every record in the
source". Tag matching is exact — `genki-1` does not match `genki-10`.

`janki migrate-inline` writes `include_ids:` and `exclude_ids:` for you; the
tag keys are yours to maintain.

### Skipping words you already study in jpdb

The reason `exclude_tags` exists. Export your reviews from jpdb
(Settings → "Export vocabulary reviews"), then:

```bash
janki import-jpdb-reviews ~/Downloads/reviews.json
```

This creates no records. It finds the records janki already holds that appear
in the export — by jpdb `vid` where a record has one, otherwise by expression
plus reading — tags them `jpdb-known`, and stores the review count in
`source.raw_fields.jpdb_reviews`. Entries matching no record are listed rather
than dropped: those are words jpdb knows and janki does not, and
`janki import-jpdb` is what adds them.

Then any deck with `exclude_tags: [jpdb-known]` stops emitting them, so your
Anki decks cover what jpdb is not already drilling. Re-run it whenever you like
— words that are still known and still at the same count are left untouched,
and the ledger records the export once rather than once per run.

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
