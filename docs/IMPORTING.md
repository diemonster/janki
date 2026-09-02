# Getting words in

Every route from source material to a record in
`data/normalized/vocabulary.json`, and what happens to the rows that do not make
it. The README covers the common path; this is the whole of it.

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

## Getting a reading suggestion for held rows

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

A handout, textbook page, or photo of a whiteboard can go from source to an
Anki package in the local browser. The command line remains available for
scripts and exact recovery work, but it is not required for reviewing cards,
choosing their decks, or finishing a source.

### Browser path

Start the workbench after setup:

```bash
janki workbench
```

The address printed in the terminal is restricted to this computer and carries
a new key for that workbench session. The authority-bearing workbench page
loads no remote assets. When `[assistant] enabled = true`, its **Janki**
link opens a separately keyed loopback origin whose ChatKit UI is hosted by
OpenAI; that origin never receives the workbench URL key or CSRF token. Its
quick-start guide opens on the first dashboard view of that server session,
collapses without changing any work, and remains available to reopen. Follow
the source from top to bottom:

Janki's project-wide source intake and extraction stay available even when no
single revision deck can be selected. Its existing-card revision action does
not use a separate chat model.
It renders one deterministic application plan, then dispatches the confirmed
revision through `[ai].revise_provider`: `claude-code` uses the locally logged-in
Claude Pro/Max subscription, while `anthropic-api` uses
`ANTHROPIC_API_KEY`. Both transports use the same journal, recovery, staging,
review and apply path. The confirmation names which billing path is active;
changing the setting makes an open confirmation stale. This selector does not
change OpenAI Realtime audio.

For an existing-card change, the confirmed revision call first saves a staging
proposal; it does not alter the deck. Janki then shows the exact current and
proposed form note and examples, the example-audio provider and clip counts,
and the final package path and card count. One **Apply and finish** confirmation
applies that reviewed proposal, voices only its selected cards, and builds the
whole deck. If the browser disconnects or a later phase stops, resume the named
finish receipt rather than preparing another paid revision or audio plan.

1. **Add source material.** Choose a PDF, JPEG, PNG or HEIC file. This saves a
   durable copy on this computer and sends nothing to a provider.
2. **Read the source.** Open the saved source and use its separate reading
   action. The confirmation page names the exact file, Anthropic provider,
   model, purpose and paid API call. A missing `ANTHROPIC_API_KEY` refuses
   before the source is sent. In Janki, attaching and sending a source performs
   the same local intake and then renders that exact extraction plan in the
   conversation; one confirmation click dispatches it without another consent
   page.
3. **Check the proposals.** Compare the cards with their source evidence. Edit
   or remove a proposal explicitly; checking examples approves only the exact
   Japanese sentences shown. Review the lesson's grammar separately.
4. **Settle what is waiting.** A reading hold protects a card from receiving a
   lasting identity based on a blank or contradicted reading. A vocabulary
   table also needs a completeness decision: compare its promised source rows
   yourself, or explicitly choose the separate paid model check. Completeness
   does not judge the Japanese.
5. **Choose a study deck and add.** Create a destination from the dashboard if
   needed. Each word has one learning destination. Seeing the same word in
   another lesson records that source without silently duplicating the card or
   moving it to another deck. Review the exact add preview, then use its
   separate add button.
6. **Finish the source.** The finish page offers dictionary facts, kanji
   reference, word audio, example audio, a card preview and the deck build, in
   that order. Each provider is labelled local, networked or paid before its
   action; a paid audio button names the source, provider, model and paid
   network call. The page ends with the `.apkg` paths and the sync-first,
   **Merge Notetypes** import reminder.

Checking a card records approval for the content fingerprints of the exact
Japanese examples displayed; it does not approve the meaning, source evidence,
or any sentence added later. Grammar review is a separate mark bound to the
matching current pattern set. If the staging rows or that pattern set changes
while the review page is open, the submission refuses without a partial write;
reload and review the current text. Editing or removing a proposal remains a
separate, explicitly named action.

“Meaning in this lesson” is deliberately source-specific: it is the sense this
page taught, so dictionary enrichment does not replace it. Polite and casual
examples are two complete learning contexts, not two translations of the same
sentence. Formal or literary wording from the source remains visible as source
context with its register explained.

Closing the browser or restarting `janki workbench` does not lose a saved
review or durably completed action; unsaved form text and an unconsumed preview
are still ephemeral. The dashboard is rebuilt from the files in the
repository, not browser history or browser storage. An interrupted paid
operation remains visible and names its CLI recovery route; do not click a
fresh paid action to replace one whose outcome is unknown.

### CLI setup

Extraction needs the Anthropic SDK, which is not installed by default so that
every other command works without it:

```bash
python -m pip install -e '.[ai]'
export ANTHROPIC_API_KEY='...'
```

The key is read only from the process environment. It is never accepted by a
browser form or written to the repository, browser storage, a URL, a log, or an
error page. A missing key stops before any private source is sent.

### CLI 1: extract

```bash
janki extract ~/Downloads/lesson-3.pdf ~/Desktop/IMG_0421.HEIC
```

It asks first. Extraction is the one command that sends **your own
documents** to a paid model, so it lists the files and names the model and
waits for a `y`. A bare Enter means no. `--yes` consents in advance for a
scripted run; without it an unattended run refuses and sends nothing, because
there is nobody there to answer.

A file outside `data/inbox/` is copied into `data/inbox/scans/` first — and
that copy happens whether or not you then consent, so a refused run still
leaves the file in the inbox, which is tracked. A file already anywhere in the
durable inbox is used where it lies. This durable path
— never a desktop path — is what every extracted record cites. A desktop path
will not exist in six months; the evidence behind a card has to.

Staging and pattern review use the source basename as their key. If two
different files under the durable inbox have one basename, janki refuses them
before it calls a model or writes staging. Letter case does not make a basename
unique. Give each source a unique name before you put it in the inbox.

PDFs, JPEG, PNG and HEIC are accepted (HEIC converts through macOS's `sips`).
Each input produces one staging file named for it, `data/staging/lesson-3.pdf.yaml`,
holding one candidate per word with the page it was read from, the line it was
read from, and the model's own confidence.

Current schema-v5 extraction accepts a selected candidate only when it has at
least one nonblank meaning, exactly one complete polite and one complete casual
example, and the page/context evidence its source kind requires. A response
that misses this structural contract writes neither staging nor patterns.
`enrich --ai` shares the nonblank meaning and complete-example value contract,
but its returned example count remains flexible because an existing reviewed
example may already occupy one slot.

For schema-v4 and newer runs, the file also carries machine-owned
`candidate_accounting`. Coverage v2 binds its parsed/canonical/unusable/
duplicate counts and fingerprint. When two parsed proposals mint the same
stable ID, the editable `records` list keeps one canonical row while the
accounting block preserves the entire collision group, including original
candidate indices. Do not edit or backfill this block onto an older paid
schema-v3 file.

`--mode table` transcribes a vocabulary list row by row; `--mode prose` uses
non-exhaustive candidate selection for running text, dialogues, sentence grids,
and structured exercises that are not vocabulary lists or tables. Prose mode
also receives the collection's known expressions, so it can keep a forced
selection compact. Omit `--mode` and the model judges each page, which is right
when one document genuinely holds both table and prose shapes. `--model ID` overrides
the configured model for one run. `--force` overwrites a staging file you have
already started reviewing and replaces a reviewed stored pattern set for that
source with the new answer as unreviewed, so both need review again. `--yes`
skips the consent prompt described above.

Two things it will not do. It never writes to `vocabulary.json` — extraction
proposes, you accept. And it never accepts a truncated answer: if the model runs
out of room part-way through a page, the run fails rather than writing a file
that looks complete and quietly lost half a table.

### CLI 2: review

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

Human review still owns the ordinary rows: candidate accounting records what
the model proposed but does not require you to keep it. You may delete a row or
remove its stale `id` and re-identify it deliberately; leave the accounting
metadata unchanged.

For a CLI-only example review, read the final Japanese sentences and then set
`source.raw_fields.example_authority: staging-review` on that row. This manual
sentinel is a blanket mark, not a binding to the sentence text, so any later
edit means you must read the examples again. The browser's exact binding and
stale-page behavior are described in the Browser path above. Review a matching
grammar set with `janki patterns --review 'lesson-3.pdf'`; that command makes no
model call.

`janki validate data/staging/lesson-3.pdf.yaml` lists every row still missing
something.

### CLI 3: promote

```bash
janki promote data/staging/lesson-3.pdf.yaml
```

If a table's coverage is still unresolved, record the repository owner's
comparison under the staging file's nested `coverage.approval` key, or use the
workbench owner-confirmation form that writes that exact bound decision. The
top-level `review_notes` may explain what needs attention, but it is not
coverage authority. The alternative is the separate paid `--accept-coverage`
source read; a broad promote command is neither approval.

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

### CLI 4: finish and recover

The workbench finish page scopes facts, kanji, audio and preview to the exact
cards just added. The individual CLI commands remain useful for scripts and
collection-wide maintenance:

```bash
janki enrich --jpdb 'word:話す:はなす'          # exact promoted IDs
janki kanji 'word:話す:はなす'
janki audio --words --examples 'word:話す:はなす'
janki preview data/decks/verbs.yaml
janki build --receipt RECEIPT
```

`janki refresh` automates enrichment, audio, and build, but its enrichment and
audio steps cover the collection; `refresh --deck verbs` narrows only the build.
Record promotion writes and prints `Finish receipt: RECEIPT`. Pass that opaque
value to `janki build --receipt RECEIPT` to run the same fresh receipt-scoped
build transaction as the workbench. A receipt also exists when records and the
archive landed but the command exits non-zero because its ledger save failed;
first make the ledger writable and follow the printed `janki status --rebuild`
recovery, then continue the finish route. The `enrich`, `kanji`, and `audio`
positional IDs limit their work to the promoted records; omit them only for
intentional collection-wide maintenance.

If a journaled extraction or paid completeness-check process disappears,
inspect its durable record before doing anything else:

```bash
janki operations
janki operations --show-reply ID   # exact still-bound reply, if one exists
janki operations --end ID          # only after deciding the process is gone
janki operations --forget ID       # after accounting for the outcome and cost
```

The listed operation tells you which action is valid. `outcome_unknown` is not
permission to retry: a new request could be a second charge. Do not delete or
edit `data/operations.json` or `data/.pending/` by hand. If exact paid audio
bytes were saved before their final record write, `janki status` names the
matching `janki audio` command; rerun that same selection so it can reuse the
bytes without another provider call. See [QUALITY.md](QUALITY.md#paid-call-and-browser-recovery)
for the state and failure-reporting contract.

### Registered repairs

Use `janki repair PATH` to check a registered repair. The command shows the
input revision, each old and new value, and the repair-plan fingerprint. It does
not change the file.

Use `--apply CODE --expected-plan SHA256` for a non-interactive, derived-field
repair. The fingerprint must match the new plan. Direct apply cannot change an
ID or a protected content field.

There is no repair for a protected content field. A staged proposal flow
existed for that — `--propose CODE`, then `promote --accept-proposals` field by
field — with exactly one producer, which M8.3 deleted as model-audit logic; the
rest went with it in M8.4. Content a person wrote is changed by that person, in
the staging file or the record, or by a template clause that asks for something
better next time.

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
  intake_tag: genki-1                 # workbench assignment adds this tag
  include_ids: [word:話す:はなす]   # only these records
  include_tags: [genki-1]          # only records carrying at least one
  exclude_ids: [word:食べる:たべる] # never these records
  exclude_tags: [jpdb-known]       # never records carrying any
```

They apply in that order, so an `include_` key chooses the pool and an
`exclude_` key removes from it: a record that is both included by tag and
excluded by id is excluded. Omitting all four means "every record in the
source". Tag matching is exact — `genki-1` does not match `genki-10`.

`intake_tag` marks a word deck as a workbench assignment destination. It must
be one nonblank tag already named by `include_tags` and not by `exclude_tags`;
assigning a staged card writes that exact tag. Across the real deck corpus,
these tags are unique and a card carrying one may select only its owning word
deck. Decks without `intake_tag` still build, but the workbench does not offer
them as assignment destinations.

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
