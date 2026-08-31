# janki design

One page. This document leads: when code, plan, or another document disagrees
with it, this one wins and the other changes.

## What janki is

janki turns Japanese study material — Shirabe Jisho CSV exports, photos,
PDFs — into Anki decks. The repository is the durable source of truth;
generated `.apkg` files are build artifacts.

## The pipeline

Where the code has not yet caught up with a sentence here, an M8 milestone in
the plan names the gap. The sentence states the design, not the present.

**1. Intake.** Sources arrive and are preserved immutably under `data/inbox/`,
with provenance. Nothing ever edits a source.

**2. The AI passes (Claude Opus 5 by default; Codex is optional for immediate
bare-record enrichment).** One rich template per input shape asks
for the complete card: Japanese, natural English glosses, examples, furigana
giving each kanji's *contextual* reading in that sentence, romaji, usage, and
register. There are two paid card-writing paths. `extract` reads a preserved source
once and returns candidates, coverage facts, complete card proposals, and the
grammar patterns that source teaches in the same answer; its auto, table, and
prose modes remain three complete source templates. `enrich --ai` reads a bare
record once and returns glosses, examples, and usage together. Pattern listing
and human review remain local operations over extraction output. Prompts are
template files, readable without opening Python, and they are the quality
mechanism: **when the output is wrong or thin, expand the template.** A bare
word list has no source sentences, so its template writes them; everything
else about the card contract is shared.

Extraction's structural accounting is separate from judging the Japanese. One
deterministic staging row represents each stable word ID; if the parsed answer
proposes that ID more than once, the fingerprinted staging metadata preserves
every member of the collision group in response order. The editable `records`
list remains the human keep/correct/re-identify list.

The opt-in `promote --accept-coverage` gate is another paid model operation. It
re-reads a preserved source only to check janki's coverage bookkeeping; it does
not write or judge card content.

janki never writes code that reads Japanese — no rule that decides a reading,
a word boundary, a register, or whether notation matches a sentence. Japanese
is deeply contextual; a rule derived from one sentence is wrong for the next.
**Writing a rules engine for Japanese from scratch is this project's defining
anti-pattern.** The retired jpdb sentence oracle is the cautionary measurement:
38 of 155 examples flagged, zero true positives.

**3. Enrichment.** Facts from outside, and deterministic derivation janki
owns:

- **jpdb** — what a dictionary knows about a word: furigana, romaji, part of
  speech, verb group, transitivity, conjugations, pitch accent, frequency rank. It
  is also the reading witness at the promote gate, where a reading it
  contradicts is held back rather than minted into a permanent record ID.
  **Meanings are not on that list, and a dictionary may not write them even to
  replace a model's guess** — a card's meaning is the sense its source taught,
  and a gloss list keyed on spelling cannot tell おたふく "mumps" from the
  identically-spelled お多福 "homely woman". Only a person or `enrich --ai`,
  both of which read the card, may settle a provisional meaning, and
  `status --unsettled [field]` is how you find the ones still waiting, and
  what settles each — a mark nobody
  can see cannot do the job it exists for.
  The line is drawn at *sense*: everything still granted above is a property of
  the word rather than of the meaning taught, so an entry resolved on matching
  spelling and reading states it correctly for either sense. That leaves a
  narrower hole, and it is deliberate rather than overlooked: for an all-kana
  record spelling and reading are the same string, so a true homograph pair
  passes the guard and can differ in accent or word class. Refusing the fields
  outright would cost every legitimate kana word, and telling which pair is
  which is reading Japanese. So janki keeps writing them and a person decides.
- **KANJIDIC** — kanji meanings, readings and stroke *count*. Stroke **order**
  comes from KanjiVG, a separate source under CC BY-SA 3.0 that a shared deck
  must credit.
- **VOICEVOX** — audio for words, with the pitch accent **forced**. It is the
  only engine here that can force one, which is why it voices the words: left
  to guess, an engine renders 橋 and 箸 identically, and telling those apart is
  what a pitch card is for. A word with no usable pattern is still voiced, with
  the engine's own accent and a ledger mark saying so.
- **OpenAI Realtime** — audio for example *sentences*, read naturally by
  `gpt-realtime-1.5`. Cedar, ash, verse and marin form an equal-weight pool;
  a versioned framed SHA-256 of the stable record id picks one, so every
  example on a note keeps the same voice across rebuilds. The reviewed prompt
  asks for a clear learner pace around 75% of normal conversation while
  preserving connected Japanese phrasing. A sentence carries context that
  disambiguates, and no accent data janki has covers a whole sentence. If the
  model misreads one reviewed sentence, a human may add sparse
  `examples[].spoken_japanese`; janki sends that exact nonblank text instead
  of the displayed `japanese`, and never derives it automatically.
  The displayed sentence remains the card text and stable filename identity;
  the effective spoken input participates in the content/request fingerprint
  and audio write-ahead record, so changing it stales only that clip without
  renaming it.
- **Derivation** — mechanics janki owns: romaji transliterated from the
  model's furigana (for a bare word, from its reading), and the conjugation
  tables that fill the Conjugations field and build the drill decks.
  Deterministic grammar *generation* is derivation; judging the model's
  grammar is not.

Enrichment adds facts about *words* and *files*. It never judges sentences.

**4. Compile.** genanki builds the decks. Deterministic **GUIDs** — note ids
are timestamps genanki mints per build, and the GUID is what Anki matches on
make rebuilds update cards instead of duplicating them; the ledger records
what shipped. Word decks are durable thematic study destinations, not
per-source artifacts. Every assignable word deck declares one nonblank, unique
`intake_tag`; assigning a staged record to that deck writes that tag. Its
`intake_tag` appears in its `include_tags`, not its `exclude_tags`, and cannot
select another word deck. No record belongs to more than one word deck — a
convention the deck files keep, checked by a test over the real decks rather
than enforced by the builder, which filters each deck without looking at the
others.

## Mechanisms the pipeline rests on

Not stages, but load-bearing: the word database
(`data/normalized/vocabulary.json`) as the single durable store; stable
identities (`word:<expression>:<reading>`) so cards dedup and review history
survives rebuilds; a human approving what enters the store (staging review);
a reviewed reading checked against jpdb before it becomes an identity, where
jpdb can resolve it — silence passes; a dictionary is a witness, not a gate;
a billed call over a private source confirmed by the owner, never assumed;
provenance from every record back to its source. A paid Realtime sentence call
is journaled before dispatch; an operation-bound spool fsyncs every exact
WebSocket text frame before JSON inspection, then a terminal stream becomes
the captured envelope before PCM decoding. The operation becomes committed
only while its decoded WAV is staged and ledgered by exact request. After that,
the guarded record write must win before janki publishes canonical media and
finalizes its ordinary audio ledger entry. Either recovery layer can therefore
resume the exact interrupted request without another bill.
Staging completion similarly holds the live review and selected archive locks
through one byte-checked archive/prune transaction, so an exact interrupted
retry cannot duplicate rows or delete a concurrently replaced review.

A paid model call follows the same write-ahead shape as that paid audio
staging: journaled durably before dispatch, its exact response persisted as a
pending artifact before parsing, so neither a crash nor a parse failure can
lose an answer already paid for. `extract`, the separate
`promote --accept-coverage` completeness check, and OpenAI Realtime sentence
audio do this today. `enrich --ai` spends money and does **not** yet journal,
which is a gap to close rather than a design choice — until it does, nothing
below applies to it. The journal moves
`authorized → dispatching → running → result_captured → committed`, with
`outcome_unknown`, `failed_before_send`, `canceled_before_send`, and `expired`
as terminal or holding states. A dispatched call whose outcome is unknown is
never retried automatically — a fresh charge requires fresh authority.
`janki operations --show-reply ID` is the only advertised recovery reader. A
complete captured artifact passes through byte-for-byte. For an incomplete
streaming response, it emits a deterministic JSON view preserving every exact
committed UTF-8 frame payload and boundary. It revalidates the binding without
publishing a private reply, adopting a same-name replacement or unjournalled
crash extension, settling the call, or changing the journal. A recorded
artifact name is historical metadata, not proof that reply bytes remain
accessible. Before `operations --end` settles a nonterminal stream, it adopts
the sole structurally valid frame that may have been fsynced just before its
journal-head write was interrupted. An exact provider-specific rerun may then
strengthen `outcome_unknown` to `result_captured` only by sealing terminal
frames already on disk; it never redispatches. Nonempty response frames from a
call that may have been sent require `operations --forget --force` when they
did not become committed output; an ordinary forget cannot silently erase
partial paid output.
Normal capture keeps its terminal operation-bound marker until the
`result_captured` journal write durably records the relative name, pending
directory identity, five-field file snapshot (including ctime), and response
digest, together with the exact terminal marker's direct name, five-field
snapshot, and digest. The private hard link is retired before that snapshot,
so its removal cannot immediately stale the receipt; only after the receipt is
durable may that exact marker be finalized. A missing, replaced, or unreadable
marker falls back to the receipt-bound public answer and is never freshly
adopted or retired. Thus a crash before the journal write leaves WAL
proof, and one after it leaves receipt proof. With neither proof, a lexical
operation-id filename is never freshly adopted, even when its bytes match.

**One journaled call at a time.** The journal refuses to authorize a new one
while any entry is live or holds money nobody has accounted for — every state
except the terminal ones, plus `outcome_unknown`. `authorized` counts, because
writing the authority and marking it dispatched are two writes and excluding
it leaves a window where two runs both pass. This is enforced under the
journal's own lock at authorization, never merely displayed by a surface: a
check read when a page renders and acted on when a button is clicked is one
two callers can both pass. Nothing lifts the block automatically, because only
a person can say a vanished process is gone. `janki operations --end` records
that — as `canceled_before_send` when nothing was sent, as `outcome_unknown`
when a request left and no answer came back — and `--forget` records that they
have dealt with what it cost. Forgetting first records the exact
directory, inode and content bindings it is authorized to retire in that same
journal entry. A crash retries only those bindings without asking for a new
force decision or adopting a same-name replacement. The lexical pending
directory is part of that authority: if it is missing, symlinked/non-directory,
or stably opens as a different directory inode after a checkout, cleanup
touches no names there, never searches for the moved inode, preserves the
replacement namespace, and closes the now-unreachable binding. Other probe or
I/O failures keep the entry. The entry disappears only after its still-bound
names retire or their old namespace is proven no longer bound. That durable
forget decision means the call's money is accounted for and no longer blocks a
new authorization, while a cleanup-only entry remains listed with its exact
ordinary `--forget` retry.

## Surfaces

A local browser workbench and the CLI are the same tool twice: views and
controllers over the same repository files and the same operations. The
workbench is not a second database of what happened — the repository already
is — and it may not weaken an authority gate the CLI enforces: a consent,
review, or approval demanded at the prompt is demanded identically in the tab.

## What janki's own logic is for

Enrichment, derivation, and artifact structure: identifiers, fingerprints,
field order, packaging, dedup, provenance. janki's logic enriches the card;
it does not audit the model. If a need looks like "check whether the model's
Japanese is right," it is a template problem — ask the template for more.
