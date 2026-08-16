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
every build to a deck's *own* package — a `--output` build is a throwaway and
records nothing — so `--unexported` answers what a deck has never shipped, which
is what makes `build --only-new` correct. An export entry also records what the
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
- Word audio needs a pitch accent, so a record without one is skipped and
  reported rather than voiced with the engine's guess — the guess is wrong on
  exactly the homographs a pitch card exists for. `--allow-default-accent` opts
  into it deliberately and marks those clips in the ledger.
- `janki status` reads your Anki collection to report an import that silently
  failed to upgrade the notetype, but only after the fact — nothing can stop the
  bad import while it is happening. Tick **Merge Notetypes** — see
  [NOTETYPE_UPGRADE.md](NOTETYPE_UPGRADE.md).
