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

The package will appear under `dist/`.

### Importing into Anki Desktop

**File → Import, and tick "Merge Notetypes".** It is off by default, and with it
off an import that adds a field does something else entirely and says nothing:
every existing note stays on the old notetype, and the new one is filed beside
it under the same name with a `+` appended and zero notes in it. No error, no
duplicates, review history intact — the new fields simply never reach a card.
With the box ticked the notetype upgrades in place, notes are matched by GUID,
and scheduling survives. Verified in Anki Desktop 25.09; see
[docs/NOTETYPE_UPGRADE.md](docs/NOTETYPE_UPGRADE.md).

You need this whenever a janki release adds a field — the pitch diagram,
frequency rank, and example audio arrived that way. It is harmless otherwise.

**A field append forces one full AnkiWeb sync.** Adding a field is a schema
change (verified against the `anki` library, 2026-08-08: it bumps the
collection's `scm` mark), so the next sync asks you to choose a direction
rather than merging. Sync *before* importing, so the choice is trivial: upload
your own collection and nothing is at stake. Then sync normally to make the
deck available in AnkiMobile.

### Building only what is new

A deck that has shipped 400 records and gained 3 does not need a 400-note
package:

```bash
janki build verbs --only-new
```

That includes only the records this deck has never been built with, per the
ledger's export history, and records the rest as exported so the next run knows.
A plain `janki build` records them too — the history is what makes `--only-new`
correct, and a full build that stayed quiet would make every record it shipped
look new forever. When there is nothing new it writes no package at all, rather
than replacing your last good one with an empty deck.

Before building it says what the new records are missing:

```
warning: verbs: 2 of 3 new records have no word audio
warning: verbs: 3 of 3 new records have no example sentence
Build them anyway? [y/N]
```

`--yes` skips the question; a non-interactive run proceeds and prints the
counts. A deck name works as well as a path — `verbs` is looked up under
`deck_dir`, and a real file at that path always wins.

### The whole pipeline

This is the command to run after adding words. It is the weekly workflow:

```bash
janki refresh                  # build every deck
janki refresh --deck verbs     # build only this deck
```

Runs `enrich --jpdb` → `enrich --ai` → `audio --words --examples` →
`build --only-new`, in that order, because each stage needs what the one before
it produces: jpdb fills the readings and accents `audio` needs to force a pitch,
`--ai` writes the examples `audio` then voices, and the build ships what they
finished. Skip any stage with `--no-jpdb`, `--no-ai`, `--no-audio`, `--no-build`.
A stage that fails stops the run rather than building a package from
half-enriched records.

`--deck` scopes the **build** stage only: enrichment and audio still run over
every record in the normalized file, because a word's reading and its clip
belong to the word rather than to whichever deck happens to carry it.

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

# Read vocabulary off a PDF or photo, then promote what survives review
# (see "From a photo or a PDF to cards" below)
janki extract ~/Downloads/lesson-3.pdf
janki promote data/staging/lesson-3.pdf.yaml

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
5. Run `janki promote data/staging/<file>.yaml`. It merges the surviving
   records into `data/normalized/vocabulary.json`, registers them in the ledger,
   archives them under `data/staging/done/`, and deletes the staging file once
   nothing is left held back.

Step 2 is worth doing even though `promote` re-mints a malformed ID anyway: it
is what makes `janki validate` usable as the green light before you promote,
since validate refuses an ID minted without a reading no matter what the reading
now says. The re-mint is the safety net for when you forget, not a reason to
skip it.

A resolved staging file is finished work — leaving it in place only means the
next import of the same export keeps mentioning it.

Re-running the same import never overwrites a staging file that already exists.
While the file still has errors it reports the path and the count and tells you
to resolve it; once `janki validate` is happy with it, the message changes to
say the review is finished and to run `janki promote` on it, because re-running
an import can never consume that file itself.

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

## From a photo or a PDF to cards

A handout, a textbook page, a photo of a whiteboard: `janki extract` reads the
vocabulary off it and `janki promote` decides what becomes real. Nothing in
between touches your records.

### Install the AI extra

Extraction needs the Anthropic SDK, which is not installed by default so that
every other command works without it:

```bash
python -m pip install -e '.[ai]'
export ANTHROPIC_API_KEY='...'
```

### 1. Extract

```bash
janki extract ~/Downloads/lesson-3.pdf ~/Desktop/IMG_0421.HEIC
```

Each file is copied into `data/inbox/scans/` first, and that copy — never the
path you typed — is what every extracted record cites. A desktop path will not
exist in six months; the evidence behind a card has to.

PDFs, JPEG, PNG and HEIC are accepted (HEIC converts through macOS's `sips`).
Each input produces one staging file named for it, `data/staging/lesson-3.pdf.yaml`,
holding one candidate per word with the page it was read from, the line it was
read from, and the model's own confidence.

`--mode table` transcribes a vocabulary list row by row; `--mode prose` mines
running text for words worth a card and says why. Omit it and the model judges
each page, which is right when one document holds both. `--model ID` overrides
the configured model for one run, and `--force` overwrites a staging file you
have already started reviewing.

Two things it will not do. It never writes to `vocabulary.json` — extraction
proposes, you accept. And it never accepts a truncated answer: if the model runs
out of room part-way through a page, the run fails rather than writing a file
that looks complete and quietly lost half a table.

### 2. Review

Open the staging file. It is ordinary YAML, and it is yours to edit:

```yaml
source_file: lesson-3.pdf
model: claude-opus-5
records:
  - id: 'word:話す:はなす'
    expression: 話す
    reading: はなす
    meanings: [to speak]
    source:
      type: extract
      imported_from: lesson-3.pdf
      raw_fields:
        page: '12'
        confidence: high
        context: 話す　はなす　to speak
```

Fix what is wrong, delete what is not worth a card, and fill in any reading the
model left empty — it is told to leave one blank rather than guess, because a
reading it invents becomes a permanent record ID. Words janki already has are
kept, marked `already_known`, and sorted to the bottom so your attention goes to
the new ones. Candidates it could not turn into records at all are listed in
`review_notes` with everything it did read about them.

`janki validate data/staging/lesson-3.pdf.yaml` lists every row still missing
something.

### 3. Promote

```bash
janki promote data/staging/lesson-3.pdf.yaml
```

This is the only command that writes extracted words into your collection, and
it checks the reading of every one against jpdb before it does. Three outcomes:
the reading is what jpdb reaches for (promoted), it is one jpdb lists for another
sense (promoted, with a warning — a homograph is real and you chose it), or no
entry lists it at all (held back, because a reading nobody recognises is more
likely a slip than a discovery). A word jpdb cannot resolve at all is promoted
unchecked — silence is not disagreement. Use `--skip-reading-check` to promote
offline; readings still have to be kana, because that rule is about whether an
ID can exist at all.

Rows that pass merge into `vocabulary.json` with the usual
[merge semantics](#how-an-import-merges), get their ledger entries, and are
archived to `data/staging/done/`. Rows that do not stay in the staging file with
the reason written into them, so the file always shows exactly what still needs
you. When nothing is left held back, the file is deleted.

Promote is also where the one sanctioned ID change happens: a row held back for
a missing reading carries a malformed ID, and once you supply the reading that
ID no longer describes the record. It is re-minted from expression + reading at
promote time. These records have never been in Anki, so there is no review
history to orphan — which is exactly why this is the only place an ID may
change.

Then build as usual:

```bash
janki build data/decks/verbs.yaml
```

## Writing what a dictionary cannot

`janki enrich --jpdb` fills what jpdb knows. Two more passes write what it does
not — an example sentence a beginner can read, a note on how the word is
actually used, better English glosses. Both need the AI extra and an Anthropic
key, the same ones `janki extract` uses:

```bash
python -m pip install -e '.[ai]'
export ANTHROPIC_API_KEY='...'
```

`--ai` needs `JPDB_API_KEY` as well, because it checks every sentence it writes
against jpdb's parse of it — so the live pass and `--batch-fetch` will not start
without one. `--batch-submit` does not need it: no sentence exists to check yet.
Neither does `--polish-meanings`, which writes English and asks jpdb nothing.

One pass per run. `--jpdb`, `--ai` and `--polish-meanings` each show you their
own diff, and merging two unrelated sets of proposals into one y/n is not
review.

### Examples and usage notes

```bash
# Every record with no example or no usage note
janki enrich --ai

# Just these records
janki enrich --ai word:話す:はなす

# Rewrite the examples on records you name
janki enrich --ai --force-fields examples word:話す:はなす
```

For this pass `--force-fields` widens what may be *written*, not which records
are visited: without ids it still only looks at records missing an example or a
usage note, so `--force-fields examples` on its own finds nothing to do on a
collection where every record has both. Name the records you want rewritten.

`--jpdb` is the other way round, which is worth knowing before running it over
a whole collection: there a forced field makes every record a candidate again,
including ones that were being skipped for having nothing left to fill. So
`janki enrich --jpdb --force-fields pitch_accent` with no ids is one dictionary
call per record and overwrites every curated value it names. Name ids there too.

Everything it writes is checked before you see it. A sentence that does not
contain the word is **rejected** — it may be a perfectly good sentence, but it
is not an example of this word, and the check knows the word's conjugations and
its 〜ます forms, so 話しました counts as 話す. A sentence whose furigana jpdb
does not confirm is **kept and flagged** instead of dropped: the Japanese may be
right where the segmentation is wrong, and that is a judgment for you rather
than for janki. The flag is a fingerprint of the sentence in the record's
`raw_fields`, under `furigana_unverified`, and audio generation will read it before it speaks a
sentence (Milestone 5).
Romaji is always regenerated from the furigana, whatever the model sent.

At fifty records or more the proposals go to `data/staging/ai-enrichment.yaml`
and through `janki promote` instead of a terminal diff, because nobody reads
five hundred proposed sentences in a terminal and means it. Those records
already exist, so promoting fills their empty fields.

One thing to know before combining that with `--force-fields`: promote is
existing-wins and has no flag to change it, so a forced replacement of a field
that is already full is reported as a conflict and *not* written. Below fifty
records the same command writes it directly. If you are replacing content
rather than filling holes, do it in batches small enough to take the diff
route.

### Better English glosses

```bash
janki enrich --polish-meanings              # every record
janki enrich --polish-meanings word:聞く:きく # just this one
```

This is the one pass that rewrites a field that is already full, so it is a
separate flag, it never runs as a side effect of anything else, and it asks you
about **one record at a time** — `y`, `n`, or `q` to stop. "These thirty are
fine except the fourth" is not an answer a single y/n can take. What you accept
before quitting is still written.

It calls the model as the loop runs, so `q` stops the spending the moment you
answer it: you pay for the records it *reached*, not for every record in your
collection. Two things make "reached" more than "shown to you", though.
Declining with `n` moves on to the next record, which is another call. And a
record whose glosses are already right produces no proposal at all, so it is
paid for and passed over without a prompt — nine of those before the first
question means ten calls, not one. An answer
that comes back empty means the glosses on file are already right, which is a
common and legitimate result. An answer that reduces to nothing is refused
rather than written: a card with a Japanese side and no English one is worse
than a clumsy gloss.

The prompt carries the record's own examples, because 先生に聞く and 音楽を聞く
are the same verb with two meanings a learner needs kept apart, and the
sentences the record was collected with are the only evidence of which one it
means.

### Large runs: batch mode

The Message Batches API is the same request at half price, answered within a day
rather than within seconds. Worth it for backfilling a thousand-word mining
deck; pointless for the weekly ten, which run in seconds for cents.

```bash
janki enrich --ai --batch-submit    # send, then walk away
janki enrich --ai --batch-fetch     # collect it, or hear how far along it is
janki enrich --ai --batch-forget    # give up on one that can no longer land
```

One batch at a time: two in flight would leave two answers for the same word and
no way to say which is current. The batch id and the records it covers live in
the ledger, because a submitted batch nobody kept the id of is work that was paid
for and cannot be collected. `--batch-fetch` polls once and exits — the point of
batching is that nobody is sitting there — so a batch still running just reports
its status.

Answers go through exactly the same checks as the live pass. A row the batch
reports as errored, expired or canceled is reported by name and leaves its record
alone. A row whose answer janki could not parse **keeps the batch pending**: that
answer is complete and paid for and sits on Anthropic's side for weeks, and
janki's own schema is the only thing rejecting it, so a later fetch retries
exactly those rows. `--batch-forget` is there for when they are not worth
chasing.

### What it costs

The rates are per million tokens, and the batch API halves both:

| Model            | Input  | Output |
| ---------------- | ------ | ------ |
| `claude-opus-5`  | $5.00  | $25.00 |
| `claude-sonnet-5`| $3.00  | $15.00 |
| `claude-haiku-4-5`| $1.00 | $5.00  |

`claude-sonnet-5` has introductory pricing of $2.00 / $10.00 through
2026-08-31. Check the Anthropic pricing page before planning a large run —
this table is a snapshot, not a source of truth.

What janki actually sends is small and fixed: `docs/JAPANESE_STYLE_GUIDE.md`
plus a paragraph of instructions as the system prompt — together under a
thousand tokens — and then one line per record naming the word, its reading and
what janki already knows. One call per record, every time.

That makes the output the variable, and the part worth measuring rather than
predicting: current models think before they answer, and thinking is billed as
output. **Run one record first and look at the usage in the Anthropic console**
before pointing a pass at a few thousand. A rough floor for planning is a cent or
two per record on `claude-opus-5`, half that batched — but treat a number you
measured on your own collection as the real one.

Prompt caching is asked for on the system prefix, and with the style guide as
shipped that prefix is probably below the per-model minimum for caching to
happen at all. The API does not say when it misses the minimum, so budget as if
every call re-sends it.

`janki enrich --jpdb` costs nothing in tokens — jpdb is a dictionary API, not a
model.

### Choosing the model

```toml
[ai]
extract_model = "claude-opus-5"
enrich_model = "claude-opus-5"
```

`enrich_model` covers `--ai` and `--polish-meanings`; `--model` overrides it for
one run. A batch is fetched with the model it was **submitted** under, whatever
the config says by the time you collect it, because that is what answered it.

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

`audio` is written by `janki audio`, one entry per clip, recording the engine,
the voice, the rate and any style settings that decided how it sounds — so
changing any of them makes exactly those clips stale. `exports` is written by
every build, so `--unexported` answers what a deck has never shipped, which is
what makes `build --only-new` correct. An export entry also records what the
record was *missing* when it shipped, so a word that went out silent and has a
clip now can be reported rather than silently left behind.

`enriched` *is* written, by the passes that write records directly:
`janki enrich --jpdb`, `--ai` and `--polish-meanings` each leave their own
entry, and they accumulate rather than replace, because a dictionary pass and a
writing pass describe different work.

With one gap worth knowing: a large `--ai` run — fifty records or more, or a
`--batch-fetch` of that size — writes proposals to `data/staging/` instead, and
`janki promote` then usually adds nothing to the ledger at all: every row merges
into a record that already exists, so there is no addition to record, and the
sighting it would write is the one that record's import already wrote, so the
ledger drops it as a duplicate — `registered 0 new record(s) and 0 new source
sighting(s)`. The exception is a record the ledger has never heard of, typed
into `vocabulary.json` by hand without a `status --rebuild`; that one does gain
an entry and a sighting here. So the model that wrote them survives in the staging file's
`model:` metadata and not in the ledger. If you care which model wrote a batch
of examples, keep the promoted staging file (`janki promote` archives it under
`data/staging/done/`).

## Inline deck notes

A deck YAML may carry `notes:` of its own. That was how the starter deck began,
and it is a dead end: enrichment reads records from `data/normalized/`, and
writes them back there or to `data/staging/` for review — never to a deck file.
So a record living inside a deck YAML can never gain examples, usage notes,
pitch accent or audio.

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
    furigana: "毎日[まいにち]、 妻[つま]と 日本語[にほんご]で 話[はな]します。"
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
- Pitch accent and frequency rank are filled by `janki enrich --jpdb` from
  jpdb's dictionary data. Neither is ever guessed: an empty field means the
  dictionary did not say, and `janki status` counts it as missing rather than
  inventing a value. See `docs/DESIGN_V2.md`.
- Word audio needs a pitch accent, so a record without one is skipped and
  reported rather than voiced with the engine's guess — the guess is wrong on
  exactly the homographs a pitch card exists for. `--allow-default-accent` opts
  into it deliberately and marks those clips in the ledger.
- janki does not read your Anki collection, so it cannot tell you that an import
  silently failed to upgrade the notetype. Tick **Merge Notetypes** — see
  [Importing into Anki Desktop](#importing-into-anki-desktop).

## Audio

Two engines, because the two recordings do different jobs.

**Words are spoken by VOICEVOX with their pitch accent forced.** That is the
whole reason it is here: 橋 and 箸 are the pair a card exists to tell apart, and
an engine left to guess renders them identically — measured, not assumed.
Nothing else in this project can force an accent, so nothing else voices a word.

**Sentences are read naturally**, by VOICEVOX or by OpenAI. Nothing is forced
there — janki has no accent data for a whole sentence and does not pretend to —
so the choice is about which reads Japanese better, and you should listen rather
than take a recommendation.

### Getting VOICEVOX running

It is a local engine: no account, no key, no per-character cost, and it works
offline. Either install the [VOICEVOX app](https://voicevox.hiroshiba.jp/) and
leave it open, or run the engine on its own:

```bash
docker run --rm -p 50021:50021 --name janki-voicevox \
  voicevox/voicevox_engine:cpu-latest
```

janki talks to `http://localhost:50021` by default; set `tts.voicevox_url` if
yours listens elsewhere. `janki audio` checks the engine is answering *before*
it synthesizes anything, because a run that voices forty clips and then fails on
the forty-first has written forty files and half a ledger.

If you run it in a container, note that it loads a model per speaker on demand
and keeps them: a 2 GiB colima VM gets through about nine speakers before the
container is OOM-killed. That only bites when auditioning many voices at once —
`scripts/voice-samples.py` cycles the container itself to work around it.

### Generating the audio

```bash
janki audio --words --examples
```

Both kinds are off unless asked for. It skips — and reports — anything it is not
sure of: a record with no accent pattern is not voiced with a guess, and an
example whose furigana nobody confirmed is not spoken at all. `--prune` removes
clips no record references any more, taking their ledger entries with them.

### Choosing a voice

```toml
[tts]
voicevox_speaker = 13               # speaks the words, accent forced
voicevox_speed = 0.7                # below 1 slows delivery; the engine's own
                                    # time-stretch, so the pitch does not drop
```

VOICEVOX ships 40-odd speakers, most with several styles. To hear them rather
than read a list:

```bash
python3 scripts/voice-samples.py            # every speaker, plus a page to compare them
python3 scripts/voice-samples.py --male     # just the male voices
```

That writes clips and an `index.html` to `~/Desktop/janki-voice-samples`
(`--out` to put them elsewhere). Each one runs through janki's own forced-accent
path, so the sample word carries its real accent rather than the engine's guess
— what you hear is what a card will sound like. Use `--word/--reading/--pattern`
to audition with a word you care about.

Sentences can take a different voice, or a different engine:

```toml
[tts]
voicevox_sentence_speaker = 52      # another VOICEVOX voice for sentences
```

```toml
[tts]
sentence_provider = "openai"        # OpenAI reads the sentences instead
openai_voice = "onyx"               # alloy, ash, ballad, cedar, coral, echo,
                                    # fable, marin, nova, onyx, sage, shimmer,
                                    # verse. (Cove and the other ChatGPT app
                                    # voices are a different set — not this API's.)
openai_model = "gpt-4o-mini-tts"    # or a pinned snapshot like
                                    # gpt-4o-mini-tts-2025-12-15
```

OpenAI needs `OPENAI_API_KEY` in the environment — never in `janki.toml` — and
bills per character. That model has no rate parameter, so pace is asked for in
prose through `openai_instructions`; the shipped default asks for a noticeably
slower delivery. Leave `sentence_provider` unset and one voice does everything.

Only the four models on `/v1/audio/speech` work: `tts-1`, `tts-1-hd`,
`gpt-4o-mini-tts` and its dated snapshots. The conversational models
(`gpt-audio`, `gpt-realtime`) generate speech natively, which sounds like it
should be better — but they *respond* to text rather than reading it. Asked to
read 「日本語を話しますか。」 they answer it. See M5.7 in
`docs/IMPLEMENTATION_PLAN.md` for the measurements.

### Changing a voice re-voices only what that voice said

The ledger records which engine, which voice, which rate and which style
settings made every clip, so changing any of them makes exactly those clips
stale and leaves the rest alone:

```bash
janki audio --examples       # after changing the sentence voice; words untouched
```

No `--force` needed. That flag remains for rewriting audio the settings did not
change. Filenames are content-addressed and unchanged by a re-voice, so clips
are rewritten in place and Anki's media sync picks up the new audio behind the
same `[sound:]` references.

This also means a re-voice interrupted part way — an OOM, a dropped connection —
is finished simply by running the command again.

### If you share a deck

VOICEVOX voices are free to use, **but each character carries its own terms**,
and most ask to be credited. That is a question for a deck you publish, not for
one you study alone. Check the terms for the speaker you chose at
[voicevox.hiroshiba.jp](https://voicevox.hiroshiba.jp/) and credit it in the
deck description. OpenAI audio has no such attribution requirement.
