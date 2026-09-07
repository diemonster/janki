# JPDB reading-percentage proof of concept

The owner selected personal, on-demand acquisition from JPDB. This experiment
captures the reading groups and percentages published on the five requested
kanji pages. It does not alter vocabulary, kanji reference data, decks, card
templates, or the Assistant workflow.

## Saved evidence

[report.json](report.json) contains the extracted facts, preserving each reading's source
label, link, group, order and displayed percentage text. Every character has
its source URL, retrieval timestamp and raw-response SHA-256. The metric is
**JPDB reported usage**. The inspected pages do not supply a corpus denominator
or a reproducible calculation method.

Full response HTML and matching manifests stay in the owner's local cache,
outside this public repository. The repository stores the extracted facts;
it does not republish the pages' unrelated mnemonics or example sentences.

## Observed values

| Kanji | Published percentages, in source order | Additional readings without percentages |
| --- | --- | --- |
| [物](https://jpdb.io/kanji/物) | もの 51%, もん 34%, ぶつ 9%, もつ 2%, ぶっ 1% | もち、も、もっ、のもの |
| [特](https://jpdb.io/kanji/特) | とく 90%, とっ 9% | どく、ことい、こって |
| [鳥](https://jpdb.io/kanji/鳥) | とり 70%, ちょう 21%, どり 6% | とっ、と、ちょ、ちょっ、か、ら、みとり、め |
| [料](https://jpdb.io/kanji/料) | りょう 100% | はやし、りょ |
| [理](https://jpdb.io/kanji/理) | り 84%, わ 14% | ことわり、め、ことわ |

These are provider labels, not a reconstructed on/kun inventory. In particular,
the source's contextual `わ` group is retained as supplied. The general
vocabulary and sentence examples elsewhere on a kanji page are not assigned to
individual reading groups. The script preserves the reading-detail links but
does not follow them.

Missing percentages remain null. Rounded displayed percentages are not
renormalized to 100%, and 料's displayed 100% does not make its other readings
zero. The parser can preserve an explicitly supplied `less than 1%` bound;
it never infers that bound from a missing percentage.

## Run on demand

From the repository root, fetch missing pages into the local cache:

```sh
.venv/bin/python scripts/jpdb_readings_poc.py \
  --cache-dir "$HOME/Library/Caches/janki/jpdb-readings-poc" \
  --fetch 物 特 鳥 料 理
```

Repeat without `--fetch` to reproduce the report offline. Even with `--fetch`,
valid cached pages are reused. Add `--refresh` alongside `--fetch` only when a
new download is wanted. There is no background refresh, crawl, or retry loop.
The command emits JSON on stdout; fetching it does not rewrite this research
report automatically.

## Boundary of the result

This establishes acquisition and faithful local reproduction of JPDB's
published values. Integration into the stroke panel and dedicated kanji cards
remains a separate implementation step under the agreed Assistant-first card
design.

## Verification

The live run captured five characters and 33 reading entries: 13 have
published numeric percentages and 20 have unknown percentages. Replaying the
real cache both offline and with `--fetch` reproduced `report.json` byte for
byte with network access disabled and left every cache file unchanged.

The 51 focused tests cover source structure, group and reading order, missing
and bounded values, cache provenance, explicit fetching, and failure behavior.
Nine deliberate implementation mutations were caught and restored, covering
ordering, bounds, accidental network access, cache reuse, hash verification,
the report's metric label, incomplete cache entries, and truncated reading
cells.

`make gates` passed: ruff clean, all 4,585 tests passed, and the sample deck
built successfully (536 existing dependency warnings).
