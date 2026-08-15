# Grammar from a handout: `janki patterns`

A conjugation chart is not vocabulary. This is the path from a chart or a
slide deck to rule cards, drill cards, and sentences that use the grammar you
are being taught.

`janki extract` asks a page "which vocabulary is here". That is the wrong
question for half of what a class hands you: a te-form chart contains almost no
vocabulary and is entirely about a *form*, and a week's slides contain sixty
unglossed words while really being about 〜んだ and つもり.

```bash
janki patterns data/inbox/scans/*.pdf     # read them
janki patterns                            # list what has been read
janki patterns --review '104 Week 11 Slide.pdf'
```

Nothing is used until you mark it reviewed — it is a model's reading of a slide
deck, and letting that steer every card would spread one bad inference across
the collection. A reviewed **lesson** document then steers `enrich --ai`, so the
examples it writes use the grammar you are being taught this week. A **pattern**
document (a conjugation chart) steers nothing; it is reviewed for its own sake.

## A chart becomes its own deck

A conjugation chart is a deck in itself. A deck file with `kind: pattern` turns
the rules it teaches into cards — one per rule, and a row stating two
(`くる → きて / する → して`) into one card each, which is how they are drilled:

```yaml
deck:
  kind: pattern
  name: "Brandon Japanese::Te-form Rules"
  deck_id: 2059400113
  model_id: 1607392351
  document: "teform_song.pdf"
```

**It ships only what the check below passed.** Everything on these cards is
either transcribed from the page by a model or computed by janki, so a worked
example janki disagrees with is one it has reason to think was mis-transcribed —
drilling it would teach the error. A rule with nothing checkable (`く → いて`
names an ending, not a verb) still ships: there is nothing to disagree with, and
the rule is the thing being taught. An unreviewed document builds nothing.

Pattern cards get **their own notetype**, which costs no forced sync. Measured
against `anki` 26.8.1: adding a notetype leaves the collection's `scm` mark
alone, while appending a field to an existing one bumps it — so unlike a new
field on the word notetype, this does not force the one-directional AnkiWeb sync
described in [NOTETYPE_UPGRADE.md](NOTETYPE_UPGRADE.md).

## And a deck that practises it

The rules are the mnemonic; the drill is what you need in conversation. A deck
file with `kind: conjugation` runs every verb in the collection through
`conjugation.conjugate` and asks for one form:

```yaml
deck:
  kind: conjugation
  form: te_form          # any of janki's computed forms
  name: "Brandon Japanese::Te-form Practice"
  deck_id: 2059400114
  model_id: 1607392351
```

```text
FRONT: te form  買う（かう） → ?
BACK :          買って      to use   godan
```

If you imported `teform-rules.apkg` before the drill deck existed, re-import it
with **Merge Notetypes** ticked: the shared notetype gained a `Kind` field, and
appending one is the schema change that forces a one-directional AnkiWeb sync
(see [NOTETYPE_UPGRADE.md](NOTETYPE_UPGRADE.md), and sync before you import). Without the re-import the existing cards
still read correctly — the template falls back to "Rule" for an empty field —
but they will not pick up the label.

**Every answer is computed, never transcribed** — the same rules that build the
conjugation table on the word card, so a drill card and its word card cannot
disagree, and 行く → 行って comes out right because `conjugate` knows the
exception. A verb janki declines (ゆく, whose て-form is genuinely contested, or
a record whose `verb_group` is a class name janki does not know) produces **no
card** rather than a guess.

The reading rides along on the front where it adds something: a kanji verb
cannot be conjugated without it, and hiding it would test the reading instead of
the form. GUIDs key on the record and the form, so correcting a reading rewrites
the card rather than orphaning its history.

## A chart is reviewed, then believed

A chart's rules and worked examples reach cards through one gate: a human
marks the document reviewed (`janki patterns --review`). M8.3 deleted the
checker that used to adjudicate each worked example against janki's own
conjugation tables before letting it ship — the dictionary-checks-writer shape
this project retired everywhere else. A reviewed chart's examples ship exactly
as it states them; a garbled row is fixed by editing the document's entry in
`data/patterns.json`, the same way any other reviewed content is fixed.
