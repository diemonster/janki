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

**2. The AI pass (Claude Opus 5).** One template prompt per input shape asks
for everything the cards need in one answer: the Japanese, the English
translation, furigana giving each kanji's *contextual* reading as used in that
sentence, the kanji themselves, usage patterns worth teaching, register. The
answer fills the data model. Prompts are template files, readable without
opening Python, and they are the quality mechanism: **when the output is wrong
or thin, expand the template.** A bare word list (the CSV path) has no source
sentences, so it gets its own template that writes them; everything else about
the contract is the same.

janki never writes code that reads Japanese — no rule that decides a reading,
a word boundary, a register, or whether notation matches a sentence. Japanese
is deeply contextual; a rule derived from one sentence is wrong for the next.
**Writing a rules engine for Japanese from scratch is this project's defining
anti-pattern.** The retired jpdb sentence oracle is the cautionary measurement:
38 of 155 examples flagged, zero true positives.

**3. Enrichment.** Facts from outside, and deterministic derivation janki
owns:

- **jpdb** — word details: frequency, pitch accent for card display.
- **KANJIDIC** — kanji details: meanings, readings, stroke order.
- **OpenAI TTS** — audio for words and example sentences. It renders natural
  Japanese, pitch accent included. The only steering is prose instructions in
  config; if a single clip needs help, the reading rides along in that clip's
  instructions.
- **Derivation** — mechanics janki owns: romaji transliterated from the
  model's furigana (for a bare word, from its reading), and the conjugation
  tables that fill the Conjugations field and build the drill decks.
  Deterministic grammar *generation* is derivation; judging the model's
  grammar is not.

Enrichment adds facts about *words* and *files*. It never judges sentences.

**4. Compile.** genanki builds the decks. Deterministic note IDs and GUIDs
make rebuilds update cards instead of duplicating them; the ledger records
what shipped; no record belongs to more than one word deck.

## Mechanisms the pipeline rests on

Not stages, but load-bearing: the word database
(`data/normalized/vocabulary.json`) as the single durable store; stable
identities (`word:<expression>:<reading>`) so cards dedup and review history
survives rebuilds; a human approving what enters the store (staging review);
a reviewed reading checked against jpdb before it becomes an identity, where
jpdb can resolve it — silence passes; a dictionary is a witness, not a gate;
a billed call over a private source confirmed by the owner, never assumed;
provenance from every record back to its source.

## What janki's own logic is for

Enrichment, derivation, and artifact structure: identifiers, fingerprints,
field order, packaging, dedup, provenance. janki's logic enriches the card;
it does not audit the model. If a need looks like "check whether the model's
Japanese is right," it is a template problem — ask the template for more.
