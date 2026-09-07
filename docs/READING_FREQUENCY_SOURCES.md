# Reading frequency sources for kanji cards

Research snapshot: 2026-09-07. This is a source comparison, not an implemented
provider choice. The card-type proposal remains a separate design discussion.

## What we want to measure

For a given character, how often does each reading occur in a stated body of
Japanese? A repeatable lookup alone does not establish that measurement:

- **Occurrence counts (tokens):** a word appearing 10,000 times contributes
  10,000 observations. This most closely answers the study question.
- **Distinct-word counts (types):** that word contributes one observation.
  This describes how many words use a reading, not how often readers meet it.
- **Example-word priority:** the current Janki stroke panel orders readings
  using a reading's best JMdict example priority. That does not measure either
  distribution; it must not be presented as a usage percentage.

Even actual occurrence counts describe a corpus, not all Japanese. Newspaper,
fiction, conversation, names, and textbook vocabulary can rank differently.

## Candidates

| Source | What is actually available | Acquisition | Fit for Janki |
| --- | --- | --- | --- |
| JPDB | Published reading percentages and example words | Public HTML; no kanji-reading percentage field found in the documented public API | Closest existing learner-facing presentation; preserve JPDB's reported groups and percentages |
| Tamaoka Kanji Database | Individual reading occurrence counts in the left/right positions of two-kanji compounds; also aggregate on/kun statistics | Website queries with CSV/XML export | Promising methodology, but the exported individual-reading frequencies do not reconcile with the same rows' compound totals; not adoptable as exported |
| CJK Dictionary Institute, `RD_OCCUR` | Corpus occurrence percentages for character readings and written forms | Commercial dataset enquiry | Promising occurrence data; coverage and denominator need clarification before selection |
| Jiten.moe | Count of dictionary word/form rows associated with each reading and having a positive frequency rank | JSON endpoint in its open-source API | Convenient integration, but its displayed percentages measure word/form counts |
| ichi.moe / Ichiran | Counts of common dictionary words using each reading | Website and downloadable database; open-source calculation | Transparent, reproducible dictionary prevalence, rather than occurrence frequency |

### JPDB

[The page for 理](https://jpdb.io/kanji/%E7%90%86) reports り at 84% and わ at
14%. The latter is illustrated through 理由 read わけ; this demonstrates why a
provider's contextual grouping cannot simply be substituted for KANJIDIC's
on/kun inventory. The public
[API documentation](https://jpdb.stoplight.io/docs/jpdb/mgsimhgxpjpqe-jpdb-io-public-api)
and its VocabularyField schema do not expose these character-reading values.
Absence from that documentation does not prove no private/export facility exists.

The calculation behind the published percentages was not independently
verified. Label them **JPDB reported usage**, retaining the source's rounding
and bounds such as `<1%`. A personal HTML adapter is technically an option;
the site's [terms prohibit automated scraping](https://jpdb.io/terms-of-use),
and HTML can change. A chosen
adapter should fetch only requested characters, cache results, refresh
explicitly, and leave existing data intact on failure. Builds should use the
saved snapshot, without fetching pages again.

### Tamaoka Kanji Database

This database covers 2,136 jōyō kanji using Mainichi newspaper material from
2000–2010. Its homepage explicitly licenses the data under CC BY-NC-SA 4.0 and
supports CSV/XML exports. [Database and license](https://www.kanjidatabase.com/)

The individual `Left1sound` / `Left1freq` and corresponding right-side fields
are useful here. Their scope is readings within **two-kanji compounds**. They
do not give a complete distribution across standalone words, inflected words,
and every longer compound. Separate aggregate on/kun ratios should not be
mistaken for rankings of individual readings.
[Available fields](https://www.kanjidatabase.com/kanji_search.php),
[methodology, especially pp. 704–705](https://tamaoka.org/scholarly/sadokuari/2017/154.pdf).

A completed seven-row proof of concept exported these fields and checked each
row's individual-reading frequencies against its own compound occurrence
totals. In all seven exported rows the individual-reading frequency values
fail to reconcile with the corresponding compound occurrence totals, so the
export cannot support trustworthy individual-reading percentages. 理 has
`Left1freq` 163 and `Right1freq` 205, against `AccLeft` 205212 and `AccRight`
163378. The exported numeric values also contradict the documented frequency
ordering: 鳥 has `Left1freq` 1 and `Left2freq` 655. The intended slot order may
itself be correct while the counts attached to it are wrong; the export alone
cannot distinguish the two cases. The proof of concept preserves the exported
values and their slot order exactly as issued, making no repairs.
[Proof-of-concept report](../data/research/tamaoka/2026-09-07/README.md).

Older Excel versions 2, 3 and 4 were inspected rather than assumed about:
none contains individual-reading occurrence counts. They carry aggregate
on/kun figures only, so they are not a substitute table.

The Tamaoka methodology remains promising, and a display would still say
**Mainichi compound occurrences** with the position and denominator retained.
This export is not ready for adoption until corrected original data or a
publisher clarification resolves the mismatch.

### CJK Dictionary Institute

`RD_OCCUR` explicitly counts occurrences of readings and expresses them as
percentages of its corpus. The examples include complete written forms with
okurigana; these are not already percentages normalized within one kanji.
Ask for corpus description, coverage inside compounds, available counts,
grouping, delivery format, and personal-use price before choosing it.

Its separate `FR_KANYOMI` ranking always places the on-reading block before
the kun-reading block, even when an on reading is rare, so that product would
not meet the requested overall ordering.
[Definitions and sample tables](https://cjki.org/reference/japfreq.htm).

### Jiten.moe

The source defines `GET /api/kanji/{character}` and groups reading-word rows
whose word/form frequency rank is positive. `TotalWords = g.Count()` counts
rows; the page divides these counts by their sum to display percentages.
It does **not** sum corpus occurrence counts for the reading. The controller's
comment about frequency weighting should not be taken as the implemented
metric. [API implementation](https://github.com/Sirush/Jiten/blob/master/Jiten.Api/Controllers/KanjiController.cs),
[percentage calculation](https://github.com/Sirush/Jiten/blob/master/Jiten.Web/app/pages/kanji/%5Bcharacter%5D.vue).

This is an attractive API alternative if the desired label is **share of
ranked dictionary forms**. It would not satisfy an unqualified promise of
reading usage frequency. Jiten's downloadable word and whole-kanji frequency
dictionaries are different products from a reading-occurrence table.
[Downloads](https://jiten.moe/frequency-dictionaries).

### ichi.moe / Ichiran

Ichiran's `kanji-word-stats` increments a reading counter for each dictionary
word returned, and `reading-info-json` divides by the common-word total.
That is dictionary coverage, with separately accounted irregular readings.
It is not weighted by how often each word occurs in text.
[Calculation](https://github.com/tshatrov/ichiran/blob/master/kanji.lisp),
[common-word selection](https://github.com/tshatrov/ichiran/blob/master/dict.lisp).

The downloadable database makes offline use possible without scraping the
website or copying its Japanese-reading logic into Janki.
[Database release](https://github.com/tshatrov/ichiran/releases/tag/ichiran-260118).

## Other sources checked

NINJAL/BCCWJ publishes word, character, and spelling frequency tables. These
are authoritative corpus resources, but the listed tables do not directly
provide the per-character contextual reading alignment required here.
Turning whole-word readings into character readings would require additional
linguistic analysis; Janki should not invent that analysis in its adapter.
[Official tables](https://clrd.ninjal.ac.jp/bccwj/bcc-chu.html).

KANJIDIC/JMdict remain appropriate reference dictionaries. Their reading
inventories and word-priority marks do not themselves establish a character's
reading-usage distribution.
[JMdict word priority documentation](https://www.edrdg.org/wiki/JMdict-EDICT_Dictionary_Project.html#Word_Priority_Marking).

## Recommendation for the next implementation discussion

**JPDB** offers the desired published presentation and is the remaining
shortlist candidate. **Tamaoka**'s methodology is still the strongest free
corpus-based approach on paper, but its export is held back rather than
shortlisted: the seven-row proof of concept above found every row's
individual-reading frequencies irreconcilable with that row's compound
occurrence totals. Jiten is the strongest convenient API alternative if
dictionary prevalence is acceptable. CJKI is a further option if broader
documented occurrence data justifies a commercial enquiry.

Only Tamaoka's own export was checked this way; no JPDB-versus-Tamaoka or
other cross-source character comparison has been carried out. That check's
finding is the mismatch recorded above, so the next step for Tamaoka is not
another pass over the same export but obtaining corrected original data or a
publisher clarification of the mismatch. Until one of those arrives, Tamaoka
stays promising but not ready for adoption. Any further source compared this
way should have its gaps and source groupings recorded alongside its numbers,
as a comparison of supplied facts rather than an LLM rating or a Japanese
audit.

Whichever source is selected, save the source URL/version, retrieval date,
raw reading label, metric, corpus scope, numeric value or bound, denominator
when supplied, and snapshot fingerprint. Sort comparable numeric values
within that source. Missing data stays unknown. Do not blend sources into one
percentage, infer frequencies from an LLM, or silently map unmatched readings
to KANJIDIC entries. The same saved evidence should serve both dedicated
kanji cards and vocabulary cards' stroke panels.
