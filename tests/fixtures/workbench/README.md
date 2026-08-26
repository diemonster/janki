# Workbench fixture corpus (W0)

Offline stand-ins for `janki extract`'s paid response, used to drive the real
pipeline (`extract.build_records`, `staging.write_staging`,
`patterns.save_store`, `promote.check_readings`, `exporters.anki.resolve_deck_records`,
`review_panel.ReviewPanel`) without a live model call. See
`docs/WORKBENCH_PLAN.md` task W0.

Every file under `responses/` is a raw dict matching the Pydantic schema
`extract.candidate_schema()` returns (the `Extraction` model: `candidates`,
`source_units`, `model_reported_unit_count`, `document_kind`,
`document_title`, `patterns`) — the same shape a real Claude structured-output
answer would take. `tests/test_workbench_fixtures.py` loads each one,
validates it against that live schema (so a schema change here breaks loudly
rather than silently drifting), and replicates `cli.command_extract`'s write
sequence to produce the staging file, pattern-store entry, and record set the
real command would write.

| File | Models W0's... |
|---|---|
| `table_exhaustive.json` | a small exhaustive vocabulary table (走る/食べる/飲む), three table candidates linked to matching source units |
| `lesson_with_grammar.json` | a lesson teaching both vocabulary (あげる/もらう) and grammar (〜てあげる/〜てもらう) in one answer |
| `pattern_only_chart.json` | a pattern-only source — zero candidates, patterns only, the `Grammar saved — no word cards to build` state |
| `same_spelling_different_reading.json` | 一日 read いちにち ("a whole day") and ついたち ("the first of the month") — same expression, distinct stable IDs |
| `shared_word_source_a.json` / `shared_word_source_b.json` | あげる proposed by two different sources; same stable ID from two extraction runs |
| `reading_holds.json` | 泊まる with a blank reading, and 走る with a reading (わしる) no dictionary lists — two different `promote.check_readings` hold reasons |

`decks/week-a.yaml` and `decks/week-b.yaml` are two assignable thematic decks.
Each declares its ordinary selector tag as `intake_tag`, and both currently
claim `word:あげる:あげる` via `include_tags`. This is the ambiguity the deck
picker in W4.1 has to make visible and resolve — the same shape
`m7-mixed-tsumori.yaml`'s `exclude_ids` already resolves for a real overlap in
the corpus.

Nothing here is a live provider response and no test in this suite makes a
network call.

## One deliberate trap: `shared_word_source_b`

Its source evidence is `「これ、あげる。」とよつばが言った。` — a manga line that
*quotes* the card's own example sentence, `これ、あげる。`. That shape is real
(dialogue sources quote themselves), and it makes a loose assertion lie:
`assert "これ、あげる。" in html` passes off the rendered **Source evidence**
block even when the Example field holds something else entirely. A panel test
written that way was green against a fixture whose example had been replaced
with `ぜんぜん違う文。`.

Assert the exact rendered field instead —
`<dt>Japanese</dt><dd lang="ja">これ、あげる。</dd>` — which only the example
can produce. Every other fixture's `context` is deliberately a *different*
sentence from its examples, so this file is the single regression case that
keeps the precise-assertion habit honest. Do not "fix" it.
