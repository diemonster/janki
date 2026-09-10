# Grammar from a source: `extract` then `patterns`

A conjugation chart is not vocabulary. A week's slides may contain sixty
unglossed words while really teaching 〜んだ and つもり. janki reads both facts
in the same source call: complete proposed cards for the words and an
unreviewed account of the patterns the document teaches.

```bash
janki extract data/inbox/scans/teform_song.pdf
janki patterns                            # list extracted pattern sets
janki patterns --review 'teform_song.pdf'
```

`janki patterns` makes no model call and accepts no source files. It only lists
the entries already in `data/patterns.json` or marks named entries reviewed.
Edit the JSON first when the model's account needs correction, then mark the
document reviewed.

## What `extract` writes

The source response classifies the document as one of four kinds:

- `lesson` — a handout or slide deck teaching sentence-level grammar;
- `pattern` — a chart teaching a form or transformation;
- `vocabulary` — primarily a word list, often with no patterns;
- `unknown` — no confident classification and no invented patterns.

Each pattern has the template a learner recognizes, a short English gloss,
examples found in the source, and the page or slide where it appears. Ordinary
text examples remain verbatim. When a visual chart itself encodes a complete
worked transformation through layout, its example is faithfully linearized as
self-contained plain text; an ambiguous or incomplete graphic contributes no
guessed example. The resulting `PatternSet` is written under the source basename
in `data/patterns.json` with `reviewed: false`. A complete copy also rides in the
source's staging file, so a pattern-store write problem cannot lose half of a
paid answer.

Each extraction mints one `review_run_id`. The staging metadata, its nested
proposed `pattern_set`, and the corresponding pattern-store entry all carry
that same UUID. Prompt fingerprints can repeat when the same source is asked
the same question twice; the run ID is what prevents a review of the older
answer from authorizing the newer one.

Pattern provenance is the source call's provenance: source SHA-256, mode,
provider, model, response-schema version, fingerprints for the style guide,
source task, labelled data turn and schema, and a fingerprint of the complete
request. The cards and patterns therefore name the same asking and source
bytes.

Extracting a source again does not silently revoke a prior human judgment. If
its stored pattern set is reviewed, `extract` preserves that entry while still
writing the newly paid pattern answer into the staging file. `extract --force`
explicitly replaces both the existing staging output and the stored pattern
set; the replacement is unreviewed and must pass through review again.

## The review boundary

Nothing consumes an unreviewed pattern set. That is a model's reading of a
source nobody has checked, and letting it steer every card would spread one bad
inference across the collection. Review the templates, glosses, source examples
and locations directly in `data/patterns.json`, while leaving its
`prompt_provenance` and `review_run_id` unchanged, then:

```bash
janki patterns --review '104 Week 11 Slide.pdf'
```

A reviewed **lesson** supplies labelled `Reviewed lesson patterns` data to the
complete bare-word `enrich --ai` call. Its prompt may use one naturally when it
fits the word; a pattern is never forced into a sentence merely because it is
listed. A reviewed **pattern** document steers no vocabulary examples. It is
reviewed for its own rule cards.

If a rich extraction proposes patterns but no vocabulary records, `promote`
still has finished evidence to preserve. Once the current run's pattern-store
entry is reviewed, it archives the untouched proposed `pattern_set` together
with a `reviewed_pattern_set` snapshot containing the human's corrected text;
the archive says explicitly that no records were promoted. A different run ID
or prompt provenance refuses the transition. Archive creation and live-staging
deletion are one locked compare-and-swap operation, so a concurrent forced
extraction or failed archive write leaves the live review recoverable.

## A chart becomes its own deck

A deck file with `kind: pattern` turns a reviewed chart into cards, one per
rule:

```yaml
deck:
  kind: pattern
  name: "Janki Study Decks::Te-form Rules"
  deck_id: 2059400113
  model_id: 1607392351
  document: "teform_song.pdf"
```

The human review mark is the gate. janki does not ask a dictionary or a local
Japanese rules engine to overrule what the reviewed chart states. A reviewed
chart's templates and worked examples ship as stored; an unreviewed document
builds no rule cards. Fix a bad transcription in `data/patterns.json`, not by
adding another model call or a Japanese validator.

Pattern cards have their own notetype, so adding them does not append a field
to the word notetype or force its one-directional AnkiWeb schema sync. See
[NOTETYPE_UPGRADE.md](NOTETYPE_UPGRADE.md) for the distinction.

## A deck that practises the form

The rules are the mnemonic; the drill is what you need in conversation. A deck
file with `kind: conjugation` runs every suitable verb in the collection
through `conjugation.conjugate` and asks for one form:

```yaml
deck:
  kind: conjugation
  form: te_form
  name: "Janki Study Decks::Te-form Practice"
  deck_id: 2059400114
  model_id: 1607392351
```

```text
FRONT: te form  買う（かう） → ?
BACK :          買って      to use   godan
```

These drill answers are computed from the word record's verb group rather than
transcribed from the chart. A verb outside the supported deterministic groups
produces no drill card rather than a guessed form. The reading remains on the
front where hiding it would turn a conjugation drill into a reading test. GUIDs
key on the record and form, so correcting ordinary content rebuilds the same
card rather than minting a duplicate.
