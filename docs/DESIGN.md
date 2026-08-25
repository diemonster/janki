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
- **OpenAI TTS** — audio for example *sentences*, read naturally. A sentence
  carries context that disambiguates, and no accent data janki has covers a
  whole sentence. Steering is prose instructions in config; if one clip needs
  help, the reading rides along in that clip's instructions.
- **Derivation** — mechanics janki owns: romaji transliterated from the
  model's furigana (for a bare word, from its reading), and the conjugation
  tables that fill the Conjugations field and build the drill decks.
  Deterministic grammar *generation* is derivation; judging the model's
  grammar is not.

Enrichment adds facts about *words* and *files*. It never judges sentences.

**4. Compile.** genanki builds the decks. Deterministic **GUIDs** — note ids
are timestamps genanki mints per build, and the GUID is what Anki matches on
make rebuilds update cards instead of duplicating them; the ledger records
what shipped; no record belongs to more than one word deck — a convention the deck files keep, checked by a test over the real decks rather than enforced by the builder, which filters each deck without looking at the others.

## Mechanisms the pipeline rests on

Not stages, but load-bearing: the word database
(`data/normalized/vocabulary.json`) as the single durable store; stable
identities (`word:<expression>:<reading>`) so cards dedup and review history
survives rebuilds; a human approving what enters the store (staging review);
a reviewed reading checked against jpdb before it becomes an identity, where
jpdb can resolve it — silence passes; a dictionary is a witness, not a gate;
a billed call over a private source confirmed by the owner, never assumed;
provenance from every record back to its source. Paid TTS output is staged and
ledgered by exact request before a guarded record write; only after that write
wins does janki publish canonical media and finalize its ordinary audio ledger
entry, so the exact same interrupted request can resume without another bill.
Staging completion similarly holds the live review and selected archive locks
through one byte-checked archive/prune transaction, so an exact interrupted
retry cannot duplicate rows or delete a concurrently replaced review.

A paid model call follows the same write-ahead shape as that paid audio
staging: journaled durably before dispatch, its exact response persisted as a
pending artifact before parsing, so neither a crash nor a parse failure can
lose an answer already paid for. `extract` does this today. `enrich --ai` and
`promote --accept-coverage` spend money and do **not** yet journal, which is a
gap to close rather than a design choice — until they do, nothing below
applies to them. The journal moves
`authorized → dispatching → running → result_captured → committed`, with
`outcome_unknown`, `failed_before_send`, `canceled_before_send`, and `expired`
as terminal or holding states. A dispatched call whose outcome is unknown is
never retried automatically — a fresh charge requires fresh authority.

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
when a request left and no answer came back — and `--forget` drops an entry
once they have dealt with what it cost.

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
