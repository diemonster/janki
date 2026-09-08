# janki

Turn the Japanese you are actually studying — a textbook page, a photo of a
whiteboard, a jpdb deck, a Shirabe Jisho export — into Anki decks that keep
their review history.

The repository is the source of truth. Anki packages are reproducible outputs:
records live in git, and rebuilding a deck updates the notes in Anki rather than
duplicating them.

## Start with the workbench

After setup, open the local workbench:

```bash
janki workbench
```

The address it prints is for this computer and this workbench session. Quick
start and provider setup opens on the first dashboard view of each workbench
session; collapse it when you are done and reopen its summary whenever you need
it. The ordinary path stays in the browser:

```text
 PDF or photo
      │
      ▼
 Save on this computer ──► Read with the named model ──► Check the cards
      no upload              direct paid action             your decisions
                                                               │
                                                               ▼
 Anki ◄── Build the decks ◄── Add facts and audio ◄── Choose a study deck
```

1. Add a PDF or photo. Saving it puts a durable copy in this project; it does
   not send the source anywhere.
2. Open the source and choose the separate reading action. Before anything is
   sent, the page names the file, provider, model, purpose, and paid API call.
3. Check each proposed card against the page. Edit or remove it, approve the
   Japanese examples you actually read, and review any grammar the lesson
   teaches.
4. Choose the one study deck that should teach each new word, creating it from
   the dashboard first if needed. If janki held a card for a reading decision,
   settle that before adding it.
5. For a vocabulary table, confirm that the promised rows were all accounted
   for. You can compare the source yourself or explicitly buy a separate model
   completeness check; neither choice judges whether the Japanese is good.
6. Add the cards, then follow the finish page through dictionary facts, kanji,
   word and example audio, preview, and the final `.apkg` files. Sync Anki
   first; import with **Merge Notetypes** ticked.

A few labels matter:

- **Meaning in this lesson** is the sense the source taught. A dictionary may
  add facts about the word, but it does not replace that lesson meaning.
- **Polite** and **casual** are two complete example slots, not translations of
  one another. Source wording with a different register stays visible as source
  context instead of being squeezed into either slot.
- A **reading hold** protects the card's lasting identity. Janki leaves the
  proposal waiting when a reading is blank or contradicted; it does not guess.
- **Grammar review** means you read the lesson's proposed pattern set. It is a
  separate decision from accepting example sentences and makes no new model
  call.
- **Coverage** asks whether every promised source unit was represented, not
  whether its language is correct.
- **Study-deck ownership** gives each word one durable learning destination.
  Seeing the same word in another source adds history rather than duplicating
  the card or silently moving it.

For automation, batch work, and interrupted-call recovery, use the
[CLI reference](#commands) and [full importing guide](docs/IMPORTING.md).

## Setup

Install Git LFS once per machine — the repository's audio is stored in it:

```bash
brew install git-lfs          # macOS; apt-get install git-lfs on Debian/Ubuntu
```

Then, once per clone:

```bash
./scripts/bootstrap.sh
source .venv/bin/activate
```

`bootstrap.sh` is the supported fresh-clone setup. It configures this clone's
LFS filters, installs the hooks, pulls the media so no audio is left as a
pointer file, installs janki and its `[dev]` extra into `.venv` — which includes
the AI support, so the whole command set works — runs the tests, and builds a
sample deck into `dist/`. The hooks it installs are composites: they run
`git lfs` and the repository's advisory post-commit review and pre-push review
gate. Existing unrelated Git hooks are never overwritten; resolve the reported
conflict by combining the hooks manually. Refresh the hooks with
`./scripts/install-review-hooks.sh` — do not run `git lfs install --force`,
which would replace the composites with LFS-only hooks and silently drop the
review step. To disable the review in this clone, create
`.claude/hooks/DISABLED` with a short reason; delete it to re-enable it. That
marker stops the review only: LFS keeps running either way.

### Launching Claude for work on janki

Development, planning and review model launches go through one tracked
launcher, and the review hooks already use it:

```bash
scripts/claude-subscription.py --check  # free: verify the login, then stop
scripts/claude-subscription.py --model 'claude-opus-5[1m]' --effort xhigh -p "..."
scripts/claude-subscription.py --model 'claude-opus-5[1m]' --effort xhigh  # interactive
```

Settings files are disabled for these launches, so name the model and effort
explicitly: `xhigh` for implementation, `max` for planning and review.

It hands the CLI an allowlisted environment, resolves an absolute `claude`, and
checks `auth status --json` under exactly the environment, working directory
and `--safe-mode --setting-sources ''` the launch itself uses. Anything short of
a claude.ai first-party Pro or Max login is refused, with no fallback — an
`ANTHROPIC_API_KEY` sitting in your shell is otherwise invisible until the
Console bill arrives. Your environment is left alone; the launcher simply does
not pass it on. This covers the repository's entry points, not a `claude` you
type in some other terminal. Paid content calls — `janki extract`, `revise`,
`promote --accept-coverage`, sentence audio — are unaffected and still ask for
their own consent.

### Audio and Git LFS

Once bootstrap has run, nothing about day to day work changes: ordinary
`git add`, `git commit`, `git checkout` and `git push` handle the matching WAV
files under `data/media/audio/` and their `.pending/*.stage` recovery files
automatically. The audio stays tracked and durable — the same commits, the same
recoverable review state — LFS only changes how those bytes are stored and
transferred. JSON and Markdown remain ordinary Git objects.

Two things worth knowing. The conversion tracks audio from here forward; it does
not rewrite the binary history already in the repository, so old revisions keep
their inline blobs. And every LFS version of a file consumes storage on whatever
remote hosts it, so re-voicing the collection wholesale costs space there.

Installing without bootstrap, add the AI extra for the two card-writing paths
(`extract` and `enrich --ai`) and the opt-in model-backed coverage approval
(`promote --accept-coverage`): `pip install -e '.[ai]'`. The `patterns` command
only lists and reviews results already emitted by `extract`.

The first-run guide in the workbench keeps this distinction visible:

| Service | What it does | Connection and cost |
| --- | --- | --- |
| Anthropic Claude | Reads PDFs and photos, checks a table's completeness when asked, and is the default `enrich --ai` provider for bare-word cards | Networked, paid API. `ANTHROPIC_API_KEY` is required. Claude Pro/Max is a separate subscription. |
| jpdb | Supplies dictionary facts and witnesses readings | Networked account API, not a model call. `JPDB_API_KEY` is required. |
| JPDB kanji pages | Supplies published reading percentages and reading-bound word examples | Personal, on-demand page lookups; no account API key. Saved evidence is reused and builds stay offline. |
| KANJIDIC / KanjiVG | Supplies kanji facts and stroke diagrams | Networked reference sources; no account or API key. |
| VOICEVOX | Speaks words and, by default, examples | Local service on your computer; no API key or per-call bill. Voice terms still apply. |
| OpenAI Realtime | Optionally speaks example sentences with a stable per-record voice | Networked, paid API. `OPENAI_API_KEY` is required; ChatGPT billing is separate. |
| Codex | Alternative `enrich --ai` provider for bare-word cards | Networked; run `codex login` and select `enrich_provider = "codex"`. It is not used to read a PDF or photo. |

The **Janki** surface is a real model-backed conversation rendered in
OpenAI's hosted ChatKit UI and served by janki's custom backend. ChatKit is the
interface here, not the inference provider: by default each sent message is one
journaled Claude turn through the logged-in Claude Code Pro/Max subscription.
It can inspect bounded repository projections and turn an explicit owner
instruction into one closed, exact application plan. The model cannot execute
or confirm that plan, write Japanese directly, or use raw shell, filesystem,
Git, or network tools. An active deck is optional conversational focus; it does
not limit which supported library action Janki can plan. The uncluttered routine
page does not repeat provider, model and context-provenance details; the exact
request manifest and operation journal retain them.

For character study, tell Janki the targets and destination, for example:
“Make a kanji deck called 201 Week 2 Kanji for 物、特、鳥、料、理.” Explicit
kanji intent goes directly to character-note planning. Janki prepares five
notes and five recognition cards by default, shows the actual card content and
count, then uses one confirmation to save the notes, create a compatible deck
when needed, and build the download. Progress and interruption recovery stay in
the thread. Dictionary preparation makes no paid card-writing call; ordinary
Assistant messages still use the configured conversational provider.

The wizard's **Preview these cards** link opens an interactive HTML snapshot
of the proposed cards before you confirm. Select a card, use **Show Answer**,
and open its disclosures just as you would while studying. You can also ask
Janki to “Preview this deck” or preview selected cards in an existing deck.
Vocabulary, kanji, grammar-pattern and conjugation decks use their actual Anki
templates and available media. Generating and opening a preview runs locally;
it makes no additional model or audio call and does not accept the proposal.

The same preview accompanies staged-card and revision reviews once a study
deck is assigned and the proposed content is available. Links belong to the
running Assistant session; older previews may expire. For a standalone file, run
`janki preview data/decks/201-week-2.yaml`; it writes into `dist/` by default.
The renderer uses Anki's core library without opening the Anki app. Setup and
the `[assistant]` extra include it; a minimal install can add `[preview]`.

Recognition asks for the character's core meaning. **Show Answer** reveals
meanings, strokes, common reading examples and source-labelled JPDB
percentages. Additional readings have their own caret on the answer. Character
notes live separately from vocabulary; compounds in their examples do not
become word notes. Reading practice needs a fixed source-bound example, and
production needs an explicit disambiguating cue. These directions are optional.

Attaching one supported source of up to 128 MiB in Janki only saves an immutable
local copy under `data/inbox/`. Uploading does not add its bytes to the
conversation or send them to a model, and project-wide intake remains available
even when no single deck can be selected for conversation or revision.
Immediately afterward, the same Janki conversation shows the exact extraction
plan. One plan-bound owner-confirmation click sends the named source; there is no
second consent page or confirmation chain. “Separate” means separate authority
from the local upload, not a separate interface.

Conversation and card writing have independent transport settings:

```toml
[paths]
assistant_dir = "data/assistant"  # durable conversational turns

[assistant]
enabled = true
provider = "claude-code"          # ordinary Janki turns: local Pro/Max login
model = "claude-opus-5"           # pinned; other model ids are refused
effort = "medium"                 # this turn only; low/medium/high/xhigh/max

[ai]
revise_provider = "claude-code"   # card-writing revision pass
revise_model = "claude-opus-5"
```

Either provider can be changed independently to `"anthropic-api"` to use
`ANTHROPIC_API_KEY` and Anthropic platform billing instead. Changing the
Assistant provider does not silently reroute revision, and changing revision
does not reroute conversation. A revision confirmation shows and binds its
billing path, authentication class or subscription tier, model, and exact
request fingerprints. It stages the model's proposal and stops for owner
review. After that exact review, one **Apply and finish** action re-plans the
same proposal, applies it, voices only the selected cards whose examples were
shown, and builds the whole deck package. A durable receipt resumes an
interrupted finish without widening it or repeating completed paid audio. This
is explicit content authority over the visible proposal, not an automatic
unseen apply or a sequence of separate apply/audio/build confirmations. OpenAI
Realtime audio remains API-backed regardless of either Claude transport
setting.

The Assistant's selected-card **enrich missing fields** action is narrower: it
currently uses the Anthropic API because that transport can durably capture the
exact raw reply before parsing. It refuses the Codex enrichment transport until
that path offers the same recovery seam. The confirmation says this explicitly,
and every answer stops in staging for owner review. Once its exact old and new
fields and examples are visible, one **Apply and finish** confirmation records
that review, promotes the selected cards, generates their selected audio, and
builds the persisted focused vocabulary deck. Ordinary Janki conversation and
`revise` can still use the Claude Pro/Max subscription as configured above.

Provider credentials are environment variables:

```bash
export ANTHROPIC_API_KEY='...'   # extract, Anthropic enrich --ai, Assistant or
                                 # revise when configured for anthropic-api,
                                 # AI batches, and optional coverage approval
export JPDB_API_KEY='...'        # every jpdb lookup — import-jpdb, promote's
                                 # reading check, jpdb ping,
                                 # and enrich --jpdb/--staging
export OPENAI_API_KEY='...'      # billed example-sentence audio when
                                 # sentence_provider = "openai-realtime"
```

They are read from the server process's environment only — never from
`janki.toml`, a `.env` file, a browser form or browser storage. They are never
put in repository files, URLs, logs, or error pages. A missing key stops the
action before the private source or paid request is sent; enter the key in your
shell environment and try the named action again. An OS credential-store setup
and a double-clickable launcher are not part of v1; the supported path is the
shell environment plus `janki workbench`.

## Getting material in

Three routes, and they mix freely. All of them land in
`data/normalized/vocabulary.json`, and none of them overwrites an edit you made:
an import fills empty fields and reports conflicts instead of resolving them.

### A PDF or a photo

Use the [workbench path above](#start-with-the-workbench) for PDF, JPEG, PNG and
HEIC sources. One model pass proposes complete cards and any grammar or usage
patterns the source teaches. Nothing reaches the collection until you review
the result, choose its study decks, and use the separate add action. The finish
page then keeps its dictionary, kanji, audio and preview work limited to the
cards just added, while each final package contains its complete current deck.

The exact source handling, review contract, and scripted flags are in
[From a photo or a PDF to cards](docs/IMPORTING.md#from-a-photo-or-a-pdf-to-cards).

For CLI automation, the corresponding route begins:

```bash
janki extract ~/Downloads/lesson-3.pdf   # names the source, model and charge
janki validate data/staging/lesson-3.pdf.yaml
janki promote data/staging/lesson-3.pdf.yaml  # receipt prints; ledger failure still needs recovery
janki enrich --jpdb 'word:話す:はなす'
janki kanji 'word:話す:はなす'
janki audio --words --examples 'word:話す:はなす'
janki preview data/decks/verbs.yaml
janki build --receipt RECEIPT
```

`extract --yes` records advance consent for an unattended call; without it a
non-interactive run sends nothing. `promote --accept-coverage` is a second,
separately chosen paid source read. If either journaled source call is
interrupted, do not send it again: run `janki operations`, follow the exact
`--show-reply`, `--end` or `--forget` action it prints, and preserve
`data/operations.json` and `data/.pending/`. Interrupted paid audio is resumed
by rerunning the exact matching `janki audio` selection. See
[docs/QUALITY.md](docs/QUALITY.md) for the recovery guarantees and the current
`enrich --ai` journaling gap.

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
| `janki workbench` | Take a source through review, deck choice, adding, scoped facts/audio/preview, and a complete-deck build in the local browser |
| `janki patterns [--review DOCUMENT]` | List or review patterns already emitted by `extract` |
| `janki import-shirabe FILE.csv` | Import a Shirabe Jisho export |
| `janki import-jpdb --deck NAME` | Import a jpdb deck (`--all-decks` for every one) |
| `janki import-jpdb-reviews FILE` | Tag records jpdb already drills |
| `janki repair PATH` | Show exact changes from registered safe repairs |
| `janki promote FILE.yaml` | Move a reviewed staging file into the collection |
| `janki enrich --jpdb [ID...]` | Fill dictionary fields for selected records, or every record when IDs are omitted |
| `janki enrich --ai` | Propose English glosses, complete the polite/casual example slots, and add a usage note in one bare-word call |
| `janki kanji [ID...]` | Add missing character references and JPDB reading evidence for selected vocabulary records, or the collection when IDs are omitted |
| `janki kanji-notes CHAR... --deck-name NAME` | Prepare explicit character notes, create a character deck and build it; `--deck PATH` adds to an existing character deck, and `--dry-run` previews the batch |
| `janki audio --words --examples [ID...]` | Voice selected records, or the collection when IDs are omitted |
| `janki validate [PATH]` | Check records, decks, or a staging file |
| `janki build [DECK]` | Build one deck, or `--all` |
| `janki build --receipt ID` | Resume the workbench's exact finish batch and build every complete owner deck it touches |
| `janki preview DECK` | Interactive HTML of the actual cards; uses the `[preview]` extra, no Anki app needed |
| `janki status` | Records, ledger, what is missing |
| `janki operations` | Paid calls that block spending or need cleanup. `--show-reply ID` writes a complete reply byte-for-byte or a frame-preserving JSON view of an incomplete stream; `--end ID` settles one that will never finish and adopts its sole valid crash-extension frame; an exact provider rerun may still seal already-durable terminal frames without redispatch. `--forget ID` records the final decision, unblocks spending, and retires still-bound exact recovery names without adopting a replaced pending namespace. Uncommitted replies or frames from a possibly sent call require `--force`. |
| `janki refresh` | enrich → audio → build, in order. The jpdb-backed
stage needs `JPDB_API_KEY`; without it that stage is skipped and the run exits
non-zero rather than reporting a refresh that enriched nothing |

Every command takes `--help`. `janki build` (and so `refresh --deck`) accepts a
bare deck name as well as a path — `janki build verbs` finds
`data/decks/verbs.yaml`. A CLI promotion that lands records writes and prints
its opaque finish receipt for `janki build --receipt ID`. The receipt can be
durable even when the command then exits non-zero because the ledger save did
not land; in that case, follow the printed `janki status --rebuild` recovery
before treating the promotion as ready to finish. `validate`, `preview` and
`migrate-inline` want the path.

`refresh` runs four of these. Everything else — the importers, `extract`,
`patterns`, `promote`, `kanji`, `validate`, `preview` and `status` — is yours to
run when it applies. `janki operations` is the one you should not need: it
exists for the day a paid call or its exact cleanup is interrupted, because
janki refuses to start a second call until somebody accounts for the first and
keeps incomplete cleanup visible until it finishes. A complete reply shown by
`--show-reply` passes through exactly; an incomplete streaming response is a
JSON view preserving its exact committed frame payloads and boundaries. It
remains tracked either way, so redirect stdout to export it before making a
forget decision. Run `janki kanji` after words with
new characters arrive:
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
                                    # the model (claude_client.effort_for).
                                    # Only [assistant] effort is configurable,
                                    # and only for conversation

[tts]
voicevox_speaker = 53         # words, with the pitch accent forced
sentence_provider = "openai-realtime"  # deterministic cedar/ash/verse/marin
                                        # pool; unset keeps VOICEVOX throughout

[anki]
profile = "User 1"            # only needed with several Anki profiles
```

One sentence that needs an explicit reading can carry an optional,
human-written `examples[].spoken_japanese` string in `vocabulary.json`. It is
the exact TTS input and re-voices only that clip; the displayed `japanese` and
stable media filename stay unchanged. janki never derives the override.

## Where things live

| Path | What is in it |
| --- | --- |
| `data/normalized/vocabulary.json` | The records. The source of truth — [schema](docs/DATA_MODEL.md) |
| `data/decks/*.yaml` | One file per deck: what it contains, its Anki ids |
| `data/staging/` | Waiting for you to review — extractions, held-back rows |
| `data/inbox/` | The originals every record cites |
| `data/patterns.json` | Extracted lesson/chart patterns and their human-reviewed marks |
| `data/ledger.json` | Machine-written: what arrived, shipped, was voiced, or is awaiting exact audio recovery |
| `data/media/` | Generated media; audio is [stored in Git LFS](#audio-and-git-lfs), and paid audio may briefly live under `audio/.pending/` until its guarded write finalizes |
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
