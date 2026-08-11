# janki

Turn the Japanese you are actually studying — a textbook page, a photo of a
whiteboard, a jpdb deck, a Shirabe Jisho export — into Anki decks that keep
their review history.

The repository is the source of truth. Anki packages are reproducible outputs:
records live in git, and rebuilding a deck updates the notes in Anki rather than
duplicating them.

```bash
janki extract ~/Downloads/lesson-3.pdf   # read a page
janki promote data/staging/lesson-3.pdf.yaml
janki refresh                            # enrich, voice, build what is new
```

## Setup

```bash
./scripts/bootstrap.sh
source .venv/bin/activate
```

That installs janki and its `[dev]` extra into `.venv` — which includes the AI
support, so the whole command set works — runs the tests, and builds a sample
deck into `dist/`. Installing without bootstrap, add the extra you need:
`pip install -e '.[ai]'` for `extract`, `patterns`, `review`, and every `enrich`
pass except `--jpdb`.

What is left is two keys:

```bash
export ANTHROPIC_API_KEY='...'   # extract, patterns, review, every enrich but --jpdb
export JPDB_API_KEY='...'        # dictionary lookups; from the jpdb.io settings page
```

They are read from the environment only — never from `janki.toml`, never from a
`.env` file, and neither is ever written into the repository.

## Getting material in

Three routes, and they mix freely. All of them land in
`data/normalized/vocabulary.json`, and none of them overwrites an edit you made:
an import fills empty fields and reports conflicts instead of resolving them.

### A PDF or a photo

```bash
janki extract ~/Downloads/lesson-3.pdf ~/Desktop/IMG_0421.HEIC
```

PDF, JPEG, PNG and HEIC. Each file is copied into `data/inbox/scans/` — that
copy, not the path you typed, is what every record cites — and read into one
staging file per input, holding one candidate per word with the page and line it
came from.

Nothing reaches your collection yet. Open the staging file, fix what is wrong,
delete what is not worth a card, then:

```bash
janki validate data/staging/lesson-3.pdf.yaml   # what is still incomplete
janki promote data/staging/lesson-3.pdf.yaml    # merge the survivors in
```

Promote checks every reading against jpdb before writing, and holds back the
ones no dictionary entry recognises.

### A grammar handout

A te-form chart holds almost no vocabulary and is entirely about a *form*. That
is a different question, and a different command:

```bash
janki patterns data/inbox/scans/*.pdf
janki patterns --review 'teform_song.pdf'
```

A reviewed **lesson** document steers the sentences `enrich --ai` writes, so this
week's examples use this week's grammar. A **chart** can become its own deck —
rule cards, plus a drill deck that asks you to produce the form — and janki
checks the chart's worked examples against its own conjugation rules rather than
trusting them. See [docs/PATTERNS.md](docs/PATTERNS.md).

### jpdb and Shirabe Jisho

```bash
janki import-jpdb --deck "Textbook Vol. 1: Lesson 1"   # or --all-decks
janki import-shirabe data/inbox/shirabe/export.csv
janki import-jpdb-reviews ~/Downloads/reviews.json     # tag what jpdb already drills
```

jpdb is also the dictionary behind `enrich --jpdb`, which fills furigana, pitch
accent, part of speech, verb group and frequency for records that came from
anywhere. Rows whose reading janki cannot use are held back into
`data/staging/` for review rather than imported with an ID that cannot be
corrected later. See [docs/IMPORTING.md](docs/IMPORTING.md).

## The weekly loop

One command after adding words:

```bash
janki refresh                # every deck
janki refresh --deck verbs   # build only this one
```

`--deck` scopes the **build** stage only. Enrichment and audio still run over
every record in the collection, because a word's reading and its clip belong to
the word rather than to whichever deck carries it — worth knowing before running
it against a few thousand records.

It runs `enrich --jpdb` → `enrich --ai` → `enrich --recheck-furigana` →
`audio --words --examples` → `review` → `build --only-new`, in that order,
because each stage needs what the one before it produced: jpdb fills the accents
audio needs, `--ai` writes the sentences audio speaks, and the re-check settles
any furigana dispute before it can leave a good sentence unvoiced. Skip any
stage with `--no-jpdb`, `--no-ai`, `--no-recheck`, `--no-audio`, `--no-review`,
`--no-build`. A stage that fails stops the run rather than shipping a package
built from half-enriched records.

`build --only-new` ships only the records a deck has never been built with, so a
deck that has shipped 400 records and gained 3 produces a 3-note package. When
there is nothing new it writes no package at all, rather than replacing your last
good one with an empty deck. Before building it reports what the new records are
still missing and asks; `--yes` answers that prompt, and a non-interactive run
proceeds and prints the counts.

## Loading a deck into Anki

File → Import, **and tick "Merge Notetypes."** It is off by default, and with it
off an import that adds a field does something else entirely and says nothing:
your existing notes stay on the old notetype, and
a new one is filed beside it with a `+` appended and zero notes in it. No error,
no duplicates, review history intact — the new fields simply never reach a card.
You need it whenever a janki release adds a field. `janki status` reports that
failure after the fact; nothing can stop it while it is happening.

**Sync before you import.** Adding a field is a schema change, so the next
AnkiWeb sync asks you to choose a direction rather than merging — and if you
import first and sync second, "download" loses the import while "upload" loses
whatever you reviewed on the phone. Sync first and the choice is trivial: your
own collection is the newer side. Details and the measurements behind them:
[docs/NOTETYPE_UPGRADE.md](docs/NOTETYPE_UPGRADE.md).

## Commands

| Command | What it does |
| --- | --- |
| `janki extract FILE...` | Read vocabulary off PDFs and photos into staging |
| `janki patterns FILE...` | Read a handout for the grammar it teaches |
| `janki import-shirabe FILE.csv` | Import a Shirabe Jisho export |
| `janki import-jpdb --deck NAME` | Import a jpdb deck (`--all-decks` for every one) |
| `janki import-jpdb-reviews FILE` | Tag records jpdb already drills |
| `janki promote FILE.yaml` | Move a reviewed staging file into the collection |
| `janki enrich --jpdb` | Fill fields from the dictionary |
| `janki enrich --ai` | Write example sentences and usage notes |
| `janki kanji` | Look up stroke order and on/kun readings |
| `janki audio --words --examples` | Voice the words and the sentences |
| `janki review` | Read the finished cards for correctness |
| `janki validate [PATH]` | Check records, decks, or a staging file |
| `janki build [DECK]` | Build one deck, or `--all` |
| `janki preview DECK` | A browser preview, no Anki needed |
| `janki status` | Records, ledger, what is missing |
| `janki refresh` | enrich → recheck → audio → review → build, in order |

Every command takes `--help`. `janki build` (and so `refresh --deck`) accepts a
bare deck name as well as a path — `janki build verbs` finds
`data/decks/verbs.yaml`. `validate`, `preview` and `migrate-inline` want the path.

`refresh` runs five of these, plus `enrich --recheck-furigana`. Everything else
— the importers, `extract`, `patterns`, `promote`, `kanji`,
`enrich --polish-meanings`, `validate`, `preview` and `status` — is yours to run
when it applies. Run `janki kanji` after words with new characters arrive: a
character nobody looked up simply has no stroke-order block on the card.

## Configuration

`janki.toml` at the repository root. The keys worth knowing:

```toml
[cards]
max_meanings = 4              # senses per card; 0 shows them all, a deck may
                              # set its own. The record keeps every sense

[ai]
enrich_model = "claude-opus-5"

[tts]
voicevox_speaker = 13         # words, with the pitch accent forced
sentence_provider = "openai"  # or leave unset for VOICEVOX throughout

[review]
require = true                # a build refuses a card the gate has not passed

[anki]
profile = "User 1"            # only needed with several Anki profiles
```

## Where things live

| Path | What is in it |
| --- | --- |
| `data/normalized/vocabulary.json` | The records. The source of truth — [schema](docs/DATA_MODEL.md) |
| `data/decks/*.yaml` | One file per deck: what it contains, its Anki ids |
| `data/staging/` | Waiting for you to review — extractions, held-back rows |
| `data/inbox/` | The originals every record cites |
| `data/ledger.json` | Machine-written: what arrived when, what shipped where |
| `dist/` | Built `.apkg` files |
| `templates/japanese-study/` | The card HTML and CSS |

Everything under `data/` is committed. That is what makes a review in progress
recoverable and a build reproducible.

## Documentation

| Doc | Covers |
| --- | --- |
| [docs/IMPORTING.md](docs/IMPORTING.md) | Merge rules, held-back rows, jpdb decks, PDFs and photos, deck membership |
| [docs/ENRICHMENT.md](docs/ENRICHMENT.md) | jpdb lookups, AI sentences and glosses, batch mode, what it costs |
| [docs/PATTERNS.md](docs/PATTERNS.md) | Grammar handouts, rule decks, drill decks, how a chart is checked |
| [docs/AUDIO.md](docs/AUDIO.md) | VOICEVOX and OpenAI, choosing a voice, re-voicing |
| [docs/QUALITY.md](docs/QUALITY.md) | The review gate, the ledger and `janki status`, known limits |
| [docs/DATA_MODEL.md](docs/DATA_MODEL.md) | Every field a record can carry |
| [docs/CARD_DESIGN.md](docs/CARD_DESIGN.md) | What a card shows and why |
| [docs/NOTETYPE_UPGRADE.md](docs/NOTETYPE_UPGRADE.md) | Adding a field to a notetype already in Anki |
| [docs/SHIRABE_WORKFLOW.md](docs/SHIRABE_WORKFLOW.md) | Capturing words on the phone, end to end |
| [docs/JAPANESE_STYLE_GUIDE.md](docs/JAPANESE_STYLE_GUIDE.md) | The Japanese every generated sentence is held to |
| [AGENTS.md](AGENTS.md) | Standing instructions for working on janki itself |

Design and history, for anyone changing janki rather than using it:
[docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md),
[docs/DESIGN_V2.md](docs/DESIGN_V2.md),
[docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md).

## Known limits

- The Shirabe deep link (`shirabelookup://search?w=...`) and the jpdb search URL
  on each card back are unverified against the live app and site. Test them
  before relying on them.
- Pitch accent and frequency rank are never guessed. An empty field means the
  dictionary did not say, and word audio is skipped rather than voiced with an
  engine's guess — the guess is wrong on exactly the homographs a pitch card
  exists for.
- Rendering is verified against Anki Desktop only. AnkiMobile and AnkiDroid
  layout, and whether the Shirabe app answers its URL scheme, are still manual
  checks.
- janki never writes to your Anki collection. It reads it to report problems,
  through a temporary copy, and everything else goes through an `.apkg`.
