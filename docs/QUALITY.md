# What janki checks before a card ships

The gates between a record and your collection, the ledger they read and write,
and the things janki still cannot check for you.

## What checks what

Every check janki runs is a rule about *artifacts*, not a judgement about
Japanese. `validate` knows the shape a record must have — identifiers,
balanced brackets, control characters, the shape of a pitch pattern;
`conjugation` knows which forms exist; the importers know what a source file
may claim. What none of them do is read the card as language, and that
includes the notation: brackets are counted, not parsed, so a ruby group whose
separator Anki cannot read passes `validate` and is caught by the person
reading the diff — the y/n an enrichment pass prints, or the staging file it
writes instead once a run is large enough. The enrichment template states what
a good sentence is, and stating it is the whole of the protection.

That is a deliberate reversal. A paid pass that re-read finished cards and
blocked builds on its findings ran until M8.2, with `data/review.json` as its
store; it was the audit instinct at its most expensive, and the templates now
carry the duty. `data/review.json` stays committed as history — 183 entries of
what a model said about these cards in August 2026 — and nothing reads it.

**What that pass found while it ran.** Six of twenty cards carried a part of
speech that contradicted their own verb group — jpdb returns
`["aux-v", "vt", "v1"]` for 見る and `["int", "vi", "v5", "v5r"]` for 分かる, and
janki was taking the first recognized code, so 分かる shipped as an
"interjection". That is a deterministic rule in `pos_to_part_of_speech` now,
which is where a fact about a dictionary's codes belongs.

## What Anki actually draws

`tests/test_rendered_cards.py` builds a package, imports it into a scratch
collection, and asks **Anki** to render the cards — the same path the desktop
reviewer uses. Every other template test reads the HTML off disk, and that is
blind to everything which only exists after Anki has processed it:

```bash
# delete `{{furigana:...}}` from every card back:
#   52 source-level template and build tests → still green
#   3 rendered-card tests                    → fail
```

So it pins the things a source test cannot see: that `{{furigana:話[はな]す}}`
puts はな over 話 and not over 話す, that a `{{#CasualJapanese}}` section stays
shut when there is no casual sentence, that no `{{Field}}` survives unresolved,
that `[sound:...]` is consumed rather than shown, and that both lookup links
carry a percent-encoded query.

It also demonstrates, against Anki rather than in a comment, why the prompt
template states the separator rule: with the separator missing, Anki really
does draw つま across `、妻`.

**It is not a device test.** AnkiMobile and AnkiDroid rendering, CSS and layout,
and whether the Shirabe app answers its URL scheme are all still manual.

## The workbench browser gate

HTTP handler and HTML-parser tests are necessary, but they do not prove the
browser boundary. They cannot establish what Chrome sends for a real form and
Origin header, how it treats the isolated PDF display, or its focus behavior.

`tests/test_workbench_browser.py` is the automated real-browser journey. It
drives the workbench with Playwright against an installed Chrome or Chromium
and local fake providers. A checkout without the optional Playwright test driver
skips this module at collection; with the driver present, it skips only when no
supported browser executable is installed and fails when an installed browser
cannot launch. An HTTP client is not a substitute. Keep this one journey end to
end instead of duplicating every handler test in a browser.

The automated network claim has exact edges. The main workbench test removes provider keys,
replaces every application provider, rejects non-loopback socket connections
and name resolution attempted by the Python server process, and fails if a
request issued by the tested workbench page leaves loopback or its local
`blob:` preview. It therefore cannot contact or charge a real provider through
janki. It does not instrument Chrome's own background processes and makes no
broader Chrome-egress claim; Playwright is configured to use an already
installed executable rather than downloading one during gates.

The optional assistant is tested as a different origin. Its protocol and
authority tests replace the provider and ChatKit client, while its shell test
intercepts the one documented OpenAI-hosted UI asset. This preserves the
stronger no-remote-code claim for the workbench origin instead of weakening it
to make room for ChatKit.

That one table journey checks one keyboard-reachable action and its visible
outline, a computed dark palette whose background is darker than its text, the
500px responsive breakpoint with a deliberately unbroken English value, one
exact `ruby`/`rt` pair, and a duplicate browser submission while the same
one-use paid action is already in flight. Those are regression checks for the
mechanics, not evidence that every workflow control, font fallback, source
shape, or assistive technology has passed.

Automation does not replace the owner acceptance gate. Before W6 is marked
complete, the progress note in `docs/WORKBENCH_PLAN.md` must name the browser
and version, the two real sources, the resulting `.apkg` paths, and the outcome
of both of these runs:

- one vocabulary-table source through save, paid reading, card review,
  completeness, deck choice, add, dictionary facts, kanji, word and example
  audio, preview, and build;
- one lesson or dialogue source through the same path, including its grammar
  review, without editing a repository file or running a workflow command.

That note must also identify the fixture used for CLI/workbench parity. With
the same selections, decisions, configuration and fake provider answers, both
surfaces must leave identical records, review authority, source history, merge
result, ledger, media, and deck membership. Package comparison is semantic
because genanki timestamps prevent useful byte equality.

The complete mechanical accessibility record belongs beside that evidence:
keyboard path and visible focus, an accessible name for every control, a 500px
viewport, dark mode, a Japanese font, attached furigana, and wrapping for long
English. A green unit suite alone does not establish any of these manual gates.

## Failure messages tell the whole result

Every workbench failure answers four questions in learner language:

1. What happened?
2. Which durable changes completed, including a saved source, record, archive,
   media file, ledger entry, package, or recovery artifact?
3. Could money have been spent: no, yes, or unknown?
4. What exact action should the learner take next?

“Nothing changed” is truthful only when no relevant write completed and no
request could have left the machine. A partial record/media/ledger transaction
must name each part that landed. A lost response after dispatch must say that
money may have been spent; it must not collapse into an ordinary retry prompt.
Conversely, a missing API key must say that the provider was not contacted,
while still acknowledging a source that was already saved during intake.

Tests should assert the four facts, not a loose substring such as `none`: that
word may appear in an unrelated summary and let the actual failure message
regress unnoticed.

## The ledger and `janki status`

`data/ledger.json` is machine-written, git-committed, and deliberately
minimal. Most of it tracks *operational* state per record: when the word
arrived, every source it has been seen in, whether it has been enriched,
whether audio exists for it, and which decks it has been exported to. Its
sparse top-level `pending_audio` block is the exception: a per-clip write-ahead
record for paid bytes whose guarded record/media/ledger transaction has not
finished. Card content stays in `vocabulary.json` and never appears here;
timestamps and provenance history stay here and never appear in the records.
Imports and `migrate-inline` write it — you do not edit it by hand.

`data/operations.json` is the same idea for journaled paid *model* calls, and
`janki operations` is how you read it. Extraction and the separate paid
coverage check are journaled before they are sent, and their exact replies are
written to disk before anything tries to parse them, so a crash cannot lose an
answer you already paid for. `enrich --ai` does not yet use this journal; that
is a known gap, not a recovery guarantee. janki will not start a second
journaled call while one is unaccounted for — a call still in flight, a reply
nobody turned into staging, or an outcome nothing could determine. That block
never lifts on its own, because only you can say a vanished process is gone:
`janki operations --end ID` records that, and `--forget ID` journals the final
decision once you have dealt with what it cost. That durable decision
immediately unblocks a new paid call; forget then retires exact recovery names
only while their pending-directory inode remains bound. A missing or replaced
namespace is preserved and never searched for elsewhere. If cleanup is
interrupted, the entry stays listed and the same ordinary `--forget ID` resumes
it without another `--force` decision. If a run is interrupted and the next
one refuses, that is this, and `janki operations` names the call and its
executable recovery action.

### Paid-call and browser recovery

The workbench has no second database. On restart it issues a new session URL
and reconstructs the dashboard from the durable inbox, live reviews, completed
review archives, pattern store, operation journal, ledger, and deck files. A
durably committed or saved action survives; a preview capability and text left
only in an unsubmitted browser form do not. No workflow state or credential is
recovered from cookies, local storage, or session storage.

The same repository state drives recovery in both surfaces:

- An extraction or paid completeness call is journaled before dispatch and
  its exact answer is captured before parsing. `janki operations` names
  anything still live, captured, uncertain, or awaiting cleanup.
- `janki operations --show-reply ID` streams a still-bound captured answer
  without adopting a replacement file or changing the operation.
- If a process is known to be gone, `janki operations --end ID` records whether
  it ended before sending or with an unknown outcome. An unknown outcome may
  have cost money and is never retried automatically.
- After the learner has accounted for the answer and charge,
  `janki operations --forget ID` records that decision before cleaning only
  the exact bound recovery names. The same command resumes interrupted cleanup.
- A pending paid-audio row is recovered by rerunning the exact matching audio
  selection. Saved matching bytes are finalized without another provider call;
  a different text, voice, model, instruction, or target cannot claim them.

Never repair recovery by editing or deleting `data/operations.json`,
`data/.pending/`, `data/ledger.json`, or pending audio stages. Preserve the
evidence and run the action that `janki operations` or `janki status` prints.
If a provider key is missing after restart, set it in the shell environment
that launches the workbench; the key never belongs in the recovery data or the
browser.

`janki status` reads both and summarizes them:

```text
Records: 3 (3 in data/normalized/vocabulary.json, 0 inline in deck files)
By source type: manual 3
Ledger: data/ledger.json — 3 record entries
Never exported: personal-vocabulary 0 of 0, verbs 3 of 3
Missing word audio: 3 of 3
Missing example audio: 4 of 4 sentence(s)
Stale audio: 0
Pending audio recovery: 0
Missing enrichment: 2 (no meanings or example sentence)
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

`--staged` lists every live staging row, including importer reading holdbacks
and source-extraction proposals, and shows a hold reason when one is present.
The summary always carries a `Staged for review:` line so a review queue cannot
sit unnoticed. A rich extraction may instead have no card
rows while still carrying its attributed pattern answer. Status identifies that
as pattern-only review work: review the matching `data/patterns.json` entry,
then run `janki promote` on the staging file so the zero-record run is archived.
Status checks that the store entry names the same review run and prompt; a
missing, stale, unreadable, or incomplete half gets preservation/recovery
guidance instead of a command that cannot work. Suggested commands carry the
project root explicitly, so they remain valid when status was called with
`--root` from another directory. The pattern-store key is the exact nonblank
top-level `source_file`, matching `promote`; status neither strips it nor falls
back to a nested value. Reserved review metadata such as `review_run_id`,
coverage, prompt provenance, or candidate accounting keeps a damaged zero-row
artifact in the preservation path even when another half is missing.

There is no automatic pattern-store recovery command: when the matching run is
missing or stale, preserve the staging file and restore the exact store entry
from history if available, or reconstruct it from the staged `pattern_set`
copy. Unresolved coverage remains the repository owner's decision; status never
supplies that approval. An ordinary empty staging file has no rows waiting and
no completion route status can name, but it remains tracked repository data;
status never recommends deleting it by hand.

`Pending audio recovery` is neither missing nor stale audio. It means exact
paid bytes are staged while their record reference or canonical audio-ledger
commit is unfinished. `janki build` refuses in that state; rerun the same
`janki audio` selection so janki can verify and finalize the bytes without
another provider call. A different sentence, reading, instruction, voice, or
model is not the same request and cannot adopt them. If the owner record was
deleted, status instead names the matching `janki audio … --prune` command that
retires the unreachable recovery row without requiring the speech provider.

`--rebuild` reconstructs what is still provable after a lost or corrupted
ledger: source references from each record's own `source` block, and audio
entries from the files under `data/media`. Export state is not reconstructible
— the ledger was the only place that held it. Exact pending-audio recovery is
not reconstructible either: staged bytes alone do not prove the provider
request, render profile, target, and hash that authorized adoption. Canonical
record/media state remains recoverable enough to rebuild a deck, and GUIDs are
deterministic so re-exporting a note updates it rather than duplicating it. It
is also the command that teaches the ledger about records you moved into
`vocabulary.json` by hand, and running it when nothing is missing is a no-op:
every writer records a source reference in the same shape `--rebuild`
reconstructs, so it never grows the file.

`audio` is written by `janki audio`, one entry per clip, recording the engine,
the voice, the rate and any style settings that decided how it sounds — so
changing any of them makes exactly those clips stale. `exports` is written by
every build to a deck's *own* package — a `--output` build is a throwaway and
records nothing — so `--unexported` answers what a deck has never shipped, which
is what makes `build --only-new` correct. An export entry also records what the
record was *missing* when it shipped, so a word that went out silent and has a
clip now can be reported rather than silently left behind.

`enriched` *is* written by the passes that write records directly:
`janki enrich --jpdb` and `--ai` each leave their own entry, and they
accumulate rather than replace because a dictionary pass and a complete
bare-word writing pass describe different work.

Large `--ai` runs — fifty records or more, or a `--batch-fetch` of that size —
write proposals to `data/staging/`. Promotion records an AI enrichment entry
for exactly the fields the reviewed merge actually wrote, including provider,
model, and the answer's complete request fingerprint. The archived staging
file retains the per-record input fingerprints and old-value replacement
bindings as the fuller review artifact.

## What `--only-new` promises

Three things worth knowing before scripting a build:

- When there is nothing new it writes **no package at all**, rather than
  replacing your last good one with an empty deck.
- Before building it reports what the new records are still missing —
  `warning: verbs: 2 of 3 new records have no word audio` — and asks
  `Build them anyway? [y/N]`.
- `--yes` answers that prompt. A non-interactive run proceeds and prints the
  counts rather than blocking on a question nobody is there to answer.

## Checking that an import actually landed

`janki status` reads your Anki collection and says when a deck's notetype is not
what a build would write — the **Merge Notetypes** failure described in
[NOTETYPE_UPGRADE.md](NOTETYPE_UPGRADE.md), after the fact:

```
warning: Anki: verbs: 'Japanese Study (recognition+production)+' sits beside
'Japanese Study (recognition+production)' with 0 note(s) — an import that left
'Merge Notetypes' unticked. The notes on it are on the wrong notetype.
```

It is **read-only**: the collection is copied to a temporary directory and the
copy is opened, so it works whether or not Anki is running and cannot touch what
you are studying. janki never writes to your collection.

Only your janki decks are inspected, by each deck's own model id. A collection
full of downloaded shared decks with their own `+` notetypes is neither
inspected nor mentioned — janki did not create those and cannot fix them.

With one Anki profile it finds the collection itself. With several, name one:

```toml
[anki]
profile = "User 1"
# or point straight at it:
# collection = "~/Library/Application Support/Anki2/User 1/collection.anki2"
```

Nothing here is ever an error. No Anki, no import yet, or a collection janki
cannot read are all ordinary states, and `janki status` keeps working in each.

## Editing and review-history safety

Each record has a deterministic ID based on the expression and reading. The Anki
builder derives a stable note GUID from that ID. Rebuilding and importing the
same deck should therefore update matching notes rather than create duplicates.

Do not casually change IDs, the configured deck ID, or the enabled card set after
you have accumulated real review history. See `CARD_DESIGN.md`.

## Important limitations

- Shirabe's exported header names may differ between app versions. The importer
  recognizes many common names and preserves unknown columns, but your first real
  export may require a small alias addition.
- The Shirabe deep link currently uses `shirabelookup://search?w=...`. That URL
  scheme is unverified against a real installed app — test it on your iPhone
  before relying on it.
- The desktop fallback currently searches
  `https://jpdb.io/search?q=...&lang=english`. That URL is likewise unverified
  against the live site and may change independently of janki.
- Pitch accent and frequency rank are filled by `janki enrich --jpdb` from
  jpdb's dictionary data. Neither is ever guessed: an empty field means the
  dictionary did not say, and `janki status` counts it as missing rather than
  inventing a value. See [DESIGN_V2.md](DESIGN_V2.md).
- Word audio forces the pitch accent when janki has the pattern, because the
  engine's own guess is wrong on exactly the homographs a pitch card exists
  for. A record without a pattern is still voiced — a silent card teaches
  nothing — but the run reports it and the clip is tagged `accent_unverified`
  in the ledger. Fill the pattern with `janki enrich --jpdb` and the next
  `janki audio` replaces the clip, because the bare-reading utterance and the
  usable forced-AquesTalk utterance have different content fingerprints.
- `janki status` reads your Anki collection to report an import that silently
  failed to upgrade the notetype, but only after the fact — nothing can stop the
  bad import while it is happening. Tick **Merge Notetypes** — see
  [NOTETYPE_UPGRADE.md](NOTETYPE_UPGRADE.md).
