# Filling in what a record is missing

jpdb is the dictionary; a model writes what a dictionary cannot. Both show you
a diff before anything is written.

## Enriching records from the dictionary

For records that came from Shirabe, a textbook or a photograph, jpdb can fill
what they are missing:

```bash
# Every record with an empty field
janki enrich --jpdb

# Just these records
janki enrich --jpdb word:話す:はなす word:食べる:たべる

# Let it overwrite named fields instead of only filling empty ones
janki enrich --jpdb --force-fields pitch_accent,frequency_rank
```

It fills `furigana`, `romaji`, `part_of_speech`, `verb_group`, `conjugations`,
`pitch_accent` and `frequency_rank`. It shows you every proposed change as a
diff and asks before writing; `--yes` skips the question.

A record whose enrichable fields are all filled is skipped without an API call.
A record with a field jpdb had no answer for is *not*, and is looked up again on
every run — a noun has no verb group and no conjugation table, so those stay
empty however many times you ask. Re-running is cheap on a collection of verbs
and roughly one call per word on a collection of nouns, which is worth knowing
before pointing it at a few thousand records.

Two things it will not do. It **never writes `reading`** — that is half of the
record ID, so a dictionary changing it would orphan the Anki review history
behind the record. And where your reading and jpdb's disagree, it warns and
writes nothing rather than picking: both may be right, since 一日 is both
いちにち and ついたち. When your reading *is* one jpdb lists, it asks jpdb again
with that reading pinned, so a homograph is filled from the right entry rather
than whichever one jpdb reached for first.

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
for each — 音 ゼン → 前線 ぜんせん "front line", 訓 まえ → 名前 なまえ "name".
Collapsed because it is a reminder, not the thing being tested.

The data is looked up per *character* and shared: 前 is the same 前 in 名前 and
前線, so it is fetched once into `data/kanji.json` and read by every record that
contains it. Re-running costs one request per new character. A build never
needs the network — a character not looked up simply has no block.

Example words are ranked by JMdict's frequency tags, which matters more than it
sounds: the raw list for 前 is 740 entries opening on 前官礼遇 and 前駆体.
Untagged entries are dropped rather than ranked last, so a rare character shows
its readings with no example rather than an obscure one that looks endorsed.

Sources are **KANJIDIC2** (CC BY-SA 4.0, EDRDG) via kanjiapi.dev and
**KanjiVG** (CC BY-SA 3.0, Ulrich Apel). A personal deck is fine; a deck you
share must credit both — the same footing as the VOICEVOX voice terms.

## When jpdb and the sentence disagree

`enrich --ai` writes a sentence and jpdb checks its furigana — two independent
sources, which is the point. But jpdb is not always right: its parse reads
日本語 as **にっぽんご**, and the language is にほんご. A disagreement it wins
leaves a correct sentence unvoiced forever, since `janki audio` will not speak
a flagged example.

So a disagreement is *adjudicated*. A cheap model is shown both readings and
asked which one a native speaker uses for that sentence — a much narrower
question than "what is the reading", and one it never answers by proposing a
third. `unsure` is an answer it is told to give, and it leaves the flag alone.

```bash
janki enrich --recheck-furigana                  # re-ask, adjudicating disputes
janki enrich --recheck-furigana --no-adjudicate  # leave every dispute flagged
janki enrich --recheck-furigana --accept IDS     # your call, not jpdb's
```

`janki refresh` runs this between writing and voicing, so a dispute is settled
before it can silence a card. The ledger records **who vouched** — `jpdb`, `ai`
with the adjudicating model, or `human` — because a reading a model judged is
not the same evidence as one a dictionary confirmed, and months later that is
the only thing explaining why a sentence was trusted.

Set `[ai] adjudicate_model = ""` to turn it off entirely.

## Writing what a dictionary cannot

`janki enrich --jpdb` fills what jpdb knows. Two more passes write what it does
not — an example sentence a beginner can read, a note on how the word is
actually used, better English glosses. Both need the AI extra. Immediate
example writing uses the authenticated Codex CLI by default; meaning polish,
extraction, and Anthropic batch submission use an Anthropic key:

```bash
python -m pip install -e '.[ai]'
codex login
export ANTHROPIC_API_KEY='...'
```

`--ai` needs `JPDB_API_KEY` as well, because it checks every sentence it writes
against jpdb's parse of it — so the live pass and an AI `--batch-fetch` will not
start without one. `--batch-submit` does not need it: no sentence exists to
check yet. Neither does `--polish-meanings`, including its batch actions, which
writes English and asks jpdb nothing.

One pass per run. `--jpdb`, `--ai` and `--polish-meanings` each show you their
own diff, and merging two unrelated sets of proposals into one y/n is not
review.

### Examples and usage notes

```bash
# Every record with no example or no usage note
janki enrich --ai

# Just these records
janki enrich --ai word:話す:はなす

# Rewrite the examples on records you name
janki enrich --ai --force-fields examples word:話す:はなす
```

For this pass `--force-fields` widens what may be *written*, not which records
are visited: without ids it still only looks at records missing an example or a
usage note, so `--force-fields examples` on its own finds nothing to do on a
collection where every record has both. Name the records you want rewritten.

`--jpdb` is the other way round, which is worth knowing before running it over
a whole collection: there a forced field makes every record a candidate again,
including ones that were being skipped for having nothing left to fill. So
`janki enrich --jpdb --force-fields pitch_accent` with no ids is one dictionary
call per record and overwrites every curated value it names. Name ids there too.

Everything it writes is checked before you see it. A sentence that does not
contain the word is **rejected** — it may be a perfectly good sentence, but it
is not an example of this word, and the check knows the word's conjugations and
its 〜ます forms, so 話しました counts as 話す. A sentence whose furigana jpdb
does not confirm is **kept and flagged** instead of dropped: the Japanese may be
right where the segmentation is wrong, and that is a judgment for you rather
than for janki. The flag is a fingerprint of the sentence in the record's
`raw_fields`, under `furigana_unverified`, and audio generation will read it before it speaks a
sentence (Milestone 5).
Romaji is always regenerated from the furigana, whatever the model sent.

At fifty records or more the proposals go to `data/staging/ai-enrichment.yaml`
and through `janki promote` instead of a terminal diff, because nobody reads
five hundred proposed sentences in a terminal and means it. Those records
already exist, so promoting fills their empty fields.

One thing to know before combining that with `--force-fields`: promote is
existing-wins and has no flag to change it, so a forced replacement of a field
that is already full is reported as a conflict and *not* written. Below fifty
records the same command writes it directly. If you are replacing content
rather than filling holes, do it in batches small enough to take the diff
route.

### Better English glosses

```bash
janki enrich --polish-meanings              # every record
janki enrich --polish-meanings word:聞く:きく # just this one
janki enrich --polish-meanings --batch-submit # price a large pass as one batch
janki enrich --polish-meanings --batch-fetch  # collect and review its proposals
```

This is the one pass that rewrites a field that is already full, so it is a
separate flag, it never runs as a side effect of anything else, and it asks you
about **one record at a time** — `y`, `n`, or `q` to stop. "These thirty are
fine except the fourth" is not an answer a single y/n can take. What you accept
before quitting is still written.

It calls the model as the loop runs, so `q` stops the spending the moment you
answer it: you pay for the records it *reached*, not for every record in your
collection. Two things make "reached" more than "shown to you", though.
Declining with `n` moves on to the next record, which is another call. And a
record whose glosses are already right produces no proposal at all, so it is
paid for and passed over without a prompt — nine of those before the first
question means ten calls, not one. An answer
that comes back empty means the glosses on file are already right, which is a
common and legitimate result. An answer that reduces to nothing is refused
rather than written: a card with a Japanese side and no English one is worse
than a clumsy gloss.

The prompt carries the record's own examples, because 先生に聞く and 音楽を聞く
are the same verb with two meanings a learner needs kept apart, and the
sentences the record was collected with are the only evidence of which one it
means.

For a large pass, `--batch-submit` sends the same per-record requests through
Anthropic's Message Batches API. `--batch-fetch` waits until the answers exist,
then presents their meaning diffs locally. Answering `q` records exactly the
unreviewed proposal IDs; the next fetch resumes there without another model
call or another charge. Accepted proposals are written as they are reviewed,
and declined or unchanged rows are settled rather than shown again.

Submission also records a fingerprint of every record-specific prompt. If a
record's meanings, examples, or other prompt inputs change while the batch is
running, fetch names that record and ignores its now-stale answer instead of
overwriting the newer curation. A polish batch submitted by an older janki that
did not record those fingerprints must be explicitly forgotten and resubmitted;
janki cannot safely infer whether its answers are still current.

### Large runs: batch mode

The Message Batches API is the same request at half price, answered within a day
rather than within seconds. Worth it for backfilling a thousand-word mining
deck; pointless for the weekly ten, which run in seconds for cents.

```bash
janki enrich --ai --batch-submit    # send, then walk away
janki enrich --ai --batch-fetch     # collect it, or hear how far along it is
janki enrich --ai --batch-forget    # give up on one that can no longer land
```

Meaning polish uses the same submit/fetch/forget flags with
`--polish-meanings`; unlike AI enrichment, its fetch remains a per-record
review because it replaces curated content rather than filling empty fields.

One batch at a time: two in flight would leave two answers for the same word and
no way to say which is current. The batch id and the records it covers live in
the ledger, because a submitted batch nobody kept the id of is work that was paid
for and cannot be collected. `--batch-fetch` polls once and exits — the point of
batching is that nobody is sitting there — so a batch still running just reports
its status.

Meaning-polish also protects the two moments when the external batch, records,
and ledger cannot be written as one transaction. Janki atomically writes a
recovery journal beside the configured ledger
(`data/ledger.polish-batch-recovery.json` by default) before registering a
newly submitted batch and before landing accepted meanings. If a concurrent or
failed ledger write interrupts either handoff, the next
`janki enrich --polish-meanings --batch-fetch` restores the full descriptor or
records the accepted proposals' provenance before it checks prompt fingerprints.
The file is machine-owned operational state: do not edit or delete it. Janki
validates its ownership marker and batch id before removing it after recovery.

Answers go through exactly the same checks as the live pass. A row the batch
reports as errored, expired or canceled is reported by name and leaves its record
alone. A row whose answer janki could not parse **keeps the batch pending**: that
answer is complete and paid for and sits on Anthropic's side for weeks, and
janki's own schema is the only thing rejecting it, so a later fetch retries
exactly those rows. `--batch-forget` is there for when they are not worth
chasing.

### What Anthropic batch enrichment costs

The rates are per million tokens, and the batch API halves both:

| Model            | Input  | Output |
| ---------------- | ------ | ------ |
| `claude-opus-5`  | $5.00  | $25.00 |
| `claude-sonnet-5`| $3.00  | $15.00 |
| `claude-haiku-4-5`| $1.00 | $5.00  |

`claude-sonnet-5` has introductory pricing of $2.00 / $10.00 through
2026-08-31. Check the Anthropic pricing page before planning a large run —
this table is a snapshot, not a source of truth.

What janki actually sends is small and fixed: `JAPANESE_STYLE_GUIDE.md`
plus a paragraph of instructions as the system prompt — together under a
thousand tokens — and then one line per record naming the word, its reading and
what janki already knows. One call per record, every time.

That makes the output the variable, and the part worth measuring rather than
predicting: current models think before they answer, and thinking is billed as
output. **Run one Anthropic-backed record first and look at the usage in the
Anthropic console**
before pointing a pass at a few thousand. A rough floor for planning is a cent or
two per record on `claude-opus-5`, half that batched — but treat a number you
measured on your own collection as the real one.

Prompt caching is asked for on the system prefix, and with the style guide as
shipped that prefix is probably below the per-model minimum for caching to
happen at all. The API does not say when it misses the minimum, so budget as if
every call re-sends it.

`janki enrich --jpdb` costs nothing in tokens — jpdb is a dictionary API, not a
model.

### Choosing the model

```toml
[ai]
extract_model = "claude-opus-5"
enrich_provider = "codex"
enrich_model = "gpt-5.6-sol"
enrich_reasoning_effort = "ultra"
polish_model = "claude-opus-5"
review_model = "claude-opus-5"
```

The immediate `--ai` pass uses `enrich_provider`, `enrich_model`, and (for
Codex) `enrich_reasoning_effort`. `--model` overrides the model for one run.
Meaning polish and the card review gate remain Anthropic-backed and have their
own model settings so changing the enrichment provider cannot change them by
accident.

Codex receives imported deck text as untrusted prompt content. Janki runs the
CLI from an empty temporary workspace with user configuration ignored, keeps
the read-only sandbox and execution-policy rules enabled, and disables shell,
unified exec, multi-agent, app/plugin, browser/computer, local-image, and web
tools for that call. The model therefore has no local file-reading tool through
which a prompt embedded in a source gloss could retrieve host data.

For upgrade compatibility, an older `[ai]` table that has `enrich_model` but no
`enrich_provider` keeps its original meaning: Anthropic is selected and that
model remains the default for enrichment, meaning polish, and review. Add
`enrich_provider = "codex"` explicitly when migrating that configuration to
Codex-backed enrichment.

Message Batches are an Anthropic API feature. For an `--ai` batch, set
`enrich_provider = "anthropic"` and choose a Claude `enrich_model`; polish
batches use `polish_model`. An already pending batch remains fetchable if the
config later changes. A batch is fetched with the model it was **submitted**
under because that is what answered it.
