# Filling in what a record is missing

jpdb is the dictionary; a model writes what a dictionary cannot. Both are
fill-empty by default, show what they propose, and preserve curation unless you
explicitly authorize named replacements.

## Enriching records from the dictionary

For records that came from Shirabe, a textbook or a photograph, jpdb can fill
what they are missing:

```bash
# Every record with an empty dictionary field
janki enrich --jpdb

# Just these records
janki enrich --jpdb word:話す:はなす word:食べる:たべる

# Let it overwrite named fields instead of only filling empty ones
janki enrich --jpdb --force-fields pitch_accent,frequency_rank
```

It fills `furigana`, `romaji`, `part_of_speech`, `verb_group`, `conjugations`,
`pitch_accent` and `frequency_rank`. It shows every proposed change as a diff
and asks before writing; `--yes` skips the question.

A record whose enrichable fields are all filled is skipped without an API call.
A record with a field jpdb had no answer for is looked up again on every run —
a noun has no verb group and no conjugation table, so those stay empty however
many times you ask. Re-running is cheap on a collection of verbs and roughly
one call per word on a collection of nouns, which is worth knowing before
pointing it at a few thousand records.

Two things it will not do. It **never writes `reading`** — that is half of the
record ID, so a dictionary changing it would orphan the Anki review history.
Where your reading and jpdb's disagree, it warns and writes nothing rather than
picking: both may be right, since 一日 is both いちにち and ついたち. When your
reading *is* one jpdb lists, janki asks again with that reading pinned so a
homograph is filled from the right entry.

`meanings` is deliberately left alone. A record that came from a textbook
carries the meaning that textbook taught, and a dictionary is not better placed
to decide that.

## Stroke order and kanji readings

```bash
janki kanji            # look up every character the collection uses
janki kanji --refresh  # re-fetch, rather than only what is new
```

Each card back gains a collapsed block per kanji in the word: stroke order
drawn one stroke at a time on graph paper, the 音/訓 readings, and a common word
for each — 音 ゼン → 前線 ぜんせん “front line”, 訓 まえ → 名前 なまえ “name”.
Collapsed because it is a reminder, not the thing being tested.

The data is looked up per *character* and shared: 前 is the same 前 in 名前 and
前線, so it is fetched once into `data/kanji.json` and read by every record that
contains it. Re-running costs one request per new character. A build never
needs the network — a character not looked up simply has no block.

The card's exact spelling-and-reading pair gets the first example row when it
is present in the reference data. The remaining rows preserve the source order
derived from JMdict's word-priority tags, with one example per reading before a
reading gets a second. Untagged entries are dropped rather than ranked last, so
a rare character shows its readings with no example rather than an obscure one
that looks endorsed. The four-row display is therefore useful to this card
first and biased toward common vocabulary after that.

JMdict's tags are a word-priority signal, not a percentage distribution over a
kanji's readings. janki does not import jpdb's displayed reading percentages;
doing that would require a supported data and redistribution contract rather
than depending on public-page HTML.

Sources are **KANJIDIC2** and **JMdict** (CC BY-SA 4.0, EDRDG) via
kanjiapi.dev, and **KanjiVG** (CC BY-SA 3.0, Ulrich Apel). A personal deck is
fine; a deck you share must credit all three — the same footing as the VOICEVOX
voice terms.

## Nothing checks the sentences

The model writes Japanese and nothing in janki audits it. The prompt states the
whole contract — notation, register, exact headword spelling, and each kanji's
contextual reading — and the model's answer is the answer. janki's logic
enriches the card; it never judges the Japanese.

That is a decision with history, not an oversight. The retired jpdb sentence
oracle flagged 38 of 155 examples with zero true positives, and local rules
that replaced it held natural casual ellipsis while missing the faults they
were written for. A wrong or thin sentence is fixed by editing the record or
strengthening its prompt template, never by another Japanese-reading rule.

The dictionaries still enrich **words**: `enrich --jpdb` fills dictionary
facts, and KANJIDIC supplies the kanji blocks. That is the division: the model
parses and writes; dictionaries report facts about words.

## The complete bare-word AI call

There are three card-writing model shapes in janki. `extract` reads a source once
and returns complete proposed cards plus what that source teaches. A CSV or
dictionary import supplies only a bare vocabulary record, so `enrich --ai`
uses `prompts/enrich-bare-word.md` and returns the same card content in one
answer. `revise` reads only explicitly selected existing card or deck content
plus the owner's exact requested change and returns a separately reviewable,
CAS-bound staging proposal; it never edits canonical content directly.

The bare-word answer contains:

- a compact, natural English gloss list;
- complete values for the unoccupied polite/casual example slots, with
  Japanese, natural English, `speech_level`, contextual Anki furigana, and
  spaced Hepburn romaji;
- a concise usage note when there is something useful and certain to say.

Reviewed lesson patterns from `data/patterns.json` ride in the labelled data
turn. The task template asks the model to use one naturally when it fits and to
write ordinary natural Japanese when none does. Recent examples from the same
run ride along as variety data.

Install the AI extra first. Anthropic is the default provider; Codex is
available for immediate `--ai` runs:

```bash
python -m pip install -e '.[ai]'
export ANTHROPIC_API_KEY='...'  # Anthropic calls and batches
# codex login                  # only for enrich_provider = "codex"
```

The AI pass needs no `JPDB_API_KEY`.

```bash
# Records missing meanings or an example
janki enrich --ai

# Just this record
janki enrich --ai word:話す:はなす

# Explicitly authorize replacements on a named record
janki enrich --ai --force-fields meanings,examples,usage_notes \
  word:話す:はなす
```

Without ids, `--force-fields` widens what may be written but does not make
otherwise-complete records targets again. Name records whose existing content
you intend to replace. Existing examples are preserved unless `examples` is
forced; preserving them still allows one generated sentence to fill an
unoccupied polite or casual slot when every stored example is accepted. An
empty usage note is a complete answer when there is no useful, certain nuance
to teach; use an explicit id when you want the model to revisit an
otherwise-complete record. An accepted extract-source example that only lacks
annotations is sent back with its exact Japanese pinned so the model can fill
its English, furigana, romaji, and speech level without replacing the sentence.

The remaining code owns fill discipline and deterministic derivation, not
Japanese quality. Existing annotations win over incoming ones. Supplied romaji
is compared with the reading expressed by the furigana; an incompatible value
is replaced by the mechanical transliteration with a warning. This comparison
is over readings, not a judgment about word boundaries or sentence quality.

For fewer than fifty targets, janki prints one field diff and asks once before
writing; `--yes` accepts it. At fifty or more, proposals go to
`data/staging/ai-enrichment.yaml` for an actual review:

```bash
janki validate data/staging/ai-enrichment.yaml
janki promote data/staging/ai-enrichment.yaml
```

The staged route supports real replacements without granting a floating
permission to overwrite whatever happens to be current. Its
`field_replacements` block hashes each target's record id, field name, and
exact **old wire value**. The proposed value is deliberately outside that
binding, so you may improve it during review. Before merging anything,
`promote` recomputes every old-value hash against the current collection. One
changed or deleted target makes the whole staged replacement stale and nothing
lands; regenerate the proposal against the current records.

## Prompt and answer provenance

The live task is readable in `prompts/enrich-bare-word.md`; the shared style
guide is `prompts/style-guide.md`. They are sent byte for byte and re-read on
every run. The Python user turn contains labelled record data, reviewed lesson
patterns, and recent examples — no hidden task instructions. The structured
response schema is the other input to the model.

Every answer gets one canonical request fingerprint over named logical and
transport channels:

1. the exact style-guide text;
2. the exact task-template text;
3. the labelled user/data turn;
4. the provider;
5. the provider-normalized prompt actually sent (including Codex's preamble);
6. the provider's actual wire response schema.

Named channels avoid concatenation ambiguity. Including provider transport and
wire schema means a Codex wrapper or SDK schema-transform change cannot
masquerade as the same asking. Immediate accepted changes record provider and
fingerprint in `data/ledger.json`. Large staged runs carry provider, request
fingerprint, data-turn fingerprint, and fields proposed for every record.
Batch submissions store provider and both fingerprints too: fetch compares the
data-turn fingerprint with the record as it exists then, and ignores an answer
whose input changed while the batch was out. The complete request fingerprint
remains attribution for the answer that actually ran.

Source extraction records more because the source itself is an input: source
SHA-256, mode, provider, model, schema version, separate fingerprints for the
style guide, task template, data turn, and wire response schema, plus the full
provider-normalized request fingerprint. Its unreviewed `PatternSet` carries
the same block.

## Large runs: Anthropic batch mode

The Message Batches API sends the same complete bare-word request at a lower
price and answers later. It is useful for a large backfill and unnecessary for
the weekly few words.

```bash
janki enrich --ai --batch-submit    # send, then walk away
janki enrich --ai --batch-fetch     # collect it, or report its status
janki enrich --ai --batch-forget    # abandon one that can no longer land
```

One batch may be pending at a time. Its id, model, selected fields, target ids,
and per-record fingerprints live in the committed ledger, so a submitted job
cannot become paid work nobody can collect. `--batch-fetch` polls once and
exits; a running batch simply reports its status. Invalid or failed rows stay
pending or are reported by id rather than disappearing. `--batch-forget`
removes only janki's pending entry; the provider's result remains reachable by
its batch id.

Batch mode is an Anthropic feature. Set `enrich_provider = "anthropic"` before
submitting. A later config change does not change the provider or model that
already answered.

## Cost and model choice

Model pricing changes. Check Anthropic's current pricing before a large source
or batch run, and measure one record from your own collection before estimating
thousands: reasoning tokens make output the variable. Batches trade latency for
lower cost. `janki enrich --jpdb` uses a dictionary API and costs no model
tokens.

```toml
[ai]
extract_model = "claude-opus-5"
enrich_provider = "anthropic"      # or "codex" for immediate --ai
enrich_model = "claude-opus-5"
enrich_reasoning_effort = "ultra" # Codex only
```

`--model` overrides `enrich_model` for one run. Anthropic reasoning depth comes
from the model-specific runtime mapping rather than a config key. Codex runs in
an empty temporary workspace with user configuration ignored, read-only sandbox
and execution-policy rules enabled, and all tools disabled, so source text in a
record cannot turn into local file access.

Prompt caching is requested for Anthropic's system prefix, but whether the
prefix is long enough to qualify depends on the provider's current rules.
Budget as if each request sends it in full.
