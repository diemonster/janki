# janki

Turn the Japanese you are actually studying — a textbook page, a photo of a
whiteboard, a jpdb deck, a Shirabe Jisho export — into Anki decks that keep
their review history.

The repository is the source of truth. Anki packages are reproducible outputs:
records live in git, and rebuilding a deck updates the notes in Anki rather than
duplicating them.

```bash
janki extract ~/Downloads/lesson-3.pdf   # names the file and model, asks first
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
deck into `dist/`. It also installs the repository's advisory post-commit review
and pre-push review gate. Existing unrelated Git hooks are never overwritten;
resolve the reported conflict by combining the hooks manually. Refresh only the
review hooks with `./scripts/install-review-hooks.sh`. To disable them in this
clone, create `.claude/hooks/DISABLED` with a short reason; delete it to
re-enable them.

Installing without bootstrap, add the AI extra for the two card-writing paths
(`extract` and `enrich --ai`) and the opt-in model-backed coverage approval
(`promote --accept-coverage`): `pip install -e '.[ai]'`. The `patterns` command
only lists and reviews results already emitted by `extract`.

Anthropic-backed model calls use your Anthropic key. Codex remains a supported
provider for immediate `enrich --ai`, and needs `codex login` once — only if
you set
`enrich_provider = "codex"`.

Provider credentials are environment variables:

```bash
export ANTHROPIC_API_KEY='...'   # extract, Anthropic enrich --ai,
                                 # AI batches, and optional coverage approval
export JPDB_API_KEY='...'        # every jpdb lookup — import-jpdb, promote's
                                 # reading check, jpdb ping,
                                 # and enrich --jpdb/--staging
export OPENAI_API_KEY='...'      # billed example-sentence audio when
                                 # sentence_provider = "openai"
```

They are read from the environment only — never from `janki.toml`, never from a
`.env` file, and none is ever written into the repository.

## Getting material in

Three routes, and they mix freely. All of them land in
`data/normalized/vocabulary.json`, and none of them overwrites an edit you made:
an import fills empty fields and reports conflicts instead of resolving them.

### A PDF or a photo

```bash
janki extract ~/Downloads/lesson-3.pdf ~/Desktop/IMG_0421.HEIC
```

`extract` is the intake command that sends **your own documents** to a paid
model, so it names the files and the model and asks before the first call. Add
`--yes` to consent in advance; without it an unattended run refuses rather
than sending, because there is nobody there to ask. Later,
`promote --accept-coverage` sends the preserved source again only when you
explicitly choose that model-backed coverage gate.

PDF, JPEG, PNG and HEIC. A file outside `data/inbox/` is copied into
`data/inbox/scans/`; a file already in the durable inbox stays where it is. That
durable file, not the path you typed, is what every record cites. Each input
gets one staging file. One source call returns the complete proposed cards —
English glosses, two annotated examples, a usage note, part of speech, source
page and line — and the grammar or usage patterns the document teaches. An
everyday polite or casual source sentence can be preserved and annotated;
formal or literary source text stays verbatim in `context`, with its register
explained in the usage note, while the two card examples keep their actual
polite/casual slots. A table row with no sentence gets beginner-friendly
examples from the same answer. The patterns also go to `data/patterns.json` as
unreviewed proposals, with a recoverable copy beside the staged cards.

Current schema-v5 runs require every selected candidate to arrive as a complete
card: at least one nonblank meaning, exactly one complete polite and one
complete casual example, plus reviewable source-kind evidence. A structurally
incomplete response is refused before either staging or patterns are written;
`enrich --ai` shares the complete value fields but keeps its returned example
list cardinality flexible because it may preserve one reviewed example and fill
only the other slot.

Schema-v4 and newer runs also write fingerprinted `candidate_accounting` beside
the editable rows. It binds the parsed, canonical, unusable, and duplicate
candidate counts into coverage v2. If several parsed proposals mint one stable
ID, janki keeps one canonical row and preserves every proposal in that
collision group, with its original response index, for comparison during
review. Do not edit or backfill this machine-owned block; deleting or correcting
the ordinary `records` rows is still the human review workflow.

Two different durable files cannot use one basename because staging and pattern
review use that basename as their key. Letter case does not make the name
unique. Janki refuses the collision before it calls a model; give each source a
unique name before you put it in the inbox.

Nothing reaches your collection yet. Open the staging file, fix what is wrong,
and delete what is not worth a card. For the final human approvals, open the
workbench:

```bash
janki workbench
```

It lists every source and what each is waiting for, and opening one shows its
proposed cards beside their source evidence. The address it prints carries
that session's key: the page is served only to this computer, and only to a
request that presents the key. It makes no model call and loads no remote
asset. Checking a card immediately
binds approval to the exact Japanese examples displayed; checking the pattern
set marks that complete current set reviewed. Unchecked items are unchanged,
and the panel never edits content or promotes records. Meanings and other card
fields are source-scoped display context, not part of the example approval. If
either underlying file changes while the page is open, submission refuses so
an older page cannot approve newer text.

The manual equivalent for a card is to type
`example_authority: staging-review` in that extract row's
`source.raw_fields`. Promotion replaces that manual sentinel with the same
sentence fingerprints the panel writes directly. Merely leaving a model
proposal in the file is not acceptance. Then:

```bash
janki validate data/staging/lesson-3.pdf.yaml   # what is still incomplete
janki promote data/staging/lesson-3.pdf.yaml    # merge the survivors in
```

Promote checks every reading against jpdb before writing, and holds back the
ones no dictionary entry recognises.

Paid answers remain attributable. Extraction records the source hash, mode,
provider, model, schema version, separate hashes of the style/task/data/wire-
schema channels, and one fingerprint of the complete provider-normalized
request; its pattern proposal carries the same provenance. Bare-word AI writes
the provider and complete-request fingerprint to the ledger or staging
metadata. When a large staged AI run may
replace an existing field, the staging file also binds that permission to the
record id, field name, and exact old value. A concurrent edit makes the whole
staged replacement stale instead of letting an older proposal overwrite it.
Promotion carries extraction accounting into the done archive unchanged. If a
process wrote the archive but did not prune the live review, the exact retry is
idempotent; a concurrent replacement or divergent same-run row is kept and
refused instead of being deleted or appended twice.

### A grammar handout

A te-form chart holds almost no vocabulary and is entirely about a *form*.
`extract` reads it once for both cards and what it teaches:

```bash
janki extract data/inbox/scans/teform_song.pdf
janki patterns
janki patterns --review 'teform_song.pdf'
```

`janki patterns` makes no model call: it lists the unreviewed pattern sets that
`extract` already wrote and marks the ones you have checked. A reviewed
**lesson** document steers the sentences `enrich --ai` writes, so this week's
examples use this week's grammar. A **chart** can become its own deck — rule
cards, plus a drill deck that asks you to produce the form. The gate is your
review of the chart; a reviewed chart's rules and worked examples ship as it
states them. See [docs/PATTERNS.md](docs/PATTERNS.md).

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

It runs `enrich --jpdb` → `enrich --ai` → `audio --words --examples` →
`build --only-new`, in that order, because each stage needs what the one before
it produced: jpdb fills the accents audio needs, and `--ai` writes the sentences
audio speaks. Skip any stage with `--no-jpdb`, `--no-ai`, `--no-audio`,
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
| `janki extract FILE...` | Read each PDF or photo once into rich staged cards and unreviewed patterns |
| `janki workbench` | Open a local page showing every source's state, and approve staged examples and grammar |
| `janki patterns [--review DOCUMENT]` | List or review patterns already emitted by `extract` |
| `janki import-shirabe FILE.csv` | Import a Shirabe Jisho export |
| `janki import-jpdb --deck NAME` | Import a jpdb deck (`--all-decks` for every one) |
| `janki import-jpdb-reviews FILE` | Tag records jpdb already drills |
| `janki repair PATH` | Show exact changes from registered safe repairs |
| `janki promote FILE.yaml` | Move a reviewed staging file into the collection |
| `janki enrich --jpdb` | Fill fields from the dictionary |
| `janki enrich --ai` | Propose English glosses, complete the polite/casual example slots, and add a usage note in one bare-word call |
| `janki kanji` | Look up stroke order and on/kun readings |
| `janki audio --words --examples` | Voice the words and the sentences |
| `janki validate [PATH]` | Check records, decks, or a staging file |
| `janki build [DECK]` | Build one deck, or `--all` |
| `janki preview DECK` | A browser preview, no Anki needed |
| `janki status` | Records, ledger, what is missing |
| `janki operations` | Paid model calls janki is still tracking. `--end ID` retires one that will never finish; `--forget ID` drops a finished one so the next call can start |
| `janki refresh` | enrich → audio → build, in order. The jpdb-backed
stage needs `JPDB_API_KEY`; without it that stage is skipped and the run exits
non-zero rather than reporting a refresh that enriched nothing |

Every command takes `--help`. `janki build` (and so `refresh --deck`) accepts a
bare deck name as well as a path — `janki build verbs` finds
`data/decks/verbs.yaml`. `validate`, `preview` and `migrate-inline` want the path.

`refresh` runs four of these. Everything else — the importers, `extract`,
`patterns`, `promote`, `kanji`, `validate`, `preview` and `status` — is yours to
run when it applies. `janki operations` is the one you should not need: it
exists for the day a paid call is interrupted, because janki refuses to start
a second one until somebody says what happened to the first. Run `janki kanji` after words with new characters arrive:
a character nobody looked up simply has no stroke-order block on the card.

## Configuration

`janki.toml` at the repository root. The keys worth knowing:

```toml
[cards]
max_meanings = 4              # senses per card; 0 shows them all, a deck may
                              # set its own. The record keeps every sense

[ai]
enrich_provider = "anthropic"   # or "codex"
enrich_model = "claude-opus-5"
enrich_reasoning_effort = "ultra"   # codex only; Anthropic depth follows
                                    # the model (claude_client.effort_for)

[tts]
voicevox_speaker = 53         # words, with the pitch accent forced
sentence_provider = "openai"  # or leave unset for VOICEVOX throughout

[anki]
profile = "User 1"            # only needed with several Anki profiles
```

One sentence that needs pronunciation help can carry an optional
`examples[].instructions` string in `vocabulary.json`. It supplements the
configured OpenAI sentence prompt and re-voices only that clip; VOICEVOX
refuses a clip instruction rather than ignoring it.

## Where things live

| Path | What is in it |
| --- | --- |
| `data/normalized/vocabulary.json` | The records. The source of truth — [schema](docs/DATA_MODEL.md) |
| `data/decks/*.yaml` | One file per deck: what it contains, its Anki ids |
| `data/staging/` | Waiting for you to review — extractions, held-back rows |
| `data/inbox/` | The originals every record cites |
| `data/patterns.json` | Extracted lesson/chart patterns and their human-reviewed marks |
| `data/ledger.json` | Machine-written: what arrived, shipped, was voiced, or is awaiting exact audio recovery |
| `data/media/` | Generated media; paid audio may briefly live under `audio/.pending/` until its guarded write finalizes |
| `data/review.json` | Frozen history: what a model said about these cards in August 2026. Nothing reads it |
| `dist/` | Built `.apkg` files |
| `templates/japanese-study/` | The card HTML and CSS |

Everything under `data/` is committed. That is what makes a review in progress
recoverable and a build reproducible. Do not remove a `.pending` audio stage by
hand; [rerun the matching audio command](docs/AUDIO.md#interrupted-audio-and-recovery)
and let janki verify and finalize it.

## Documentation

| Doc | Covers |
| --- | --- |
| [docs/DESIGN.md](docs/DESIGN.md) | The one-page design that leads — read this first |
| [docs/IMPORTING.md](docs/IMPORTING.md) | Merge rules, held-back rows, jpdb decks, PDFs and photos, deck membership |
| [docs/ENRICHMENT.md](docs/ENRICHMENT.md) | jpdb lookups, complete bare-word AI enrichment, batch mode, provenance |
| [docs/PATTERNS.md](docs/PATTERNS.md) | Grammar handouts, rule decks, drill decks, the review gate for charts |
| [docs/AUDIO.md](docs/AUDIO.md) | VOICEVOX and OpenAI, choosing a voice, re-voicing |
| [docs/QUALITY.md](docs/QUALITY.md) | What checks what, the ledger and `janki status`, known limits |
| [docs/DATA_MODEL.md](docs/DATA_MODEL.md) | Every field a record can carry |
| [docs/CARD_DESIGN.md](docs/CARD_DESIGN.md) | What a card shows and why |
| [docs/NOTETYPE_UPGRADE.md](docs/NOTETYPE_UPGRADE.md) | Adding a field to a notetype already in Anki |
| [docs/SHIRABE_WORKFLOW.md](docs/SHIRABE_WORKFLOW.md) | Capturing words on the phone, end to end |
| [prompts/](prompts/) | Every editable system prompt template. Markdown, sent byte for byte, yours to edit |
| [AGENTS.md](AGENTS.md) | Standing instructions for working on janki itself |

Design and history, for anyone changing janki rather than using it:
[docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md),
[docs/DESIGN_V2.md](docs/DESIGN_V2.md),
[docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md).

## Known limits

- The Shirabe deep link (`shirabelookup://search?w=...`) and the jpdb search URL
  on each card back are unverified against the live app and site. Test them
  before relying on them.
- Pitch accent and frequency rank are never guessed *in the record*: an empty
  field means the dictionary did not say. Word audio is still generated — a
  silent card teaches nothing — but with the engine's own accent rather than a
  forced one, tagged `accent_unverified` in the ledger and named by every
  `janki audio` run until `enrich --jpdb` fills the pattern, because the guess
  is wrong on exactly the homographs a pitch card exists for.
- Rendering is verified against Anki Desktop only. AnkiMobile and AnkiDroid
  layout, and whether the Shirabe app answers its URL scheme, are still manual
  checks.
- janki never writes to your Anki collection. It reads it to report problems,
  through a temporary copy, and everything else goes through an `.apkg`.
