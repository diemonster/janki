# The record schema

One record per word in `data/normalized/vocabulary.json`. Everything else in
janki — a deck, a card, a staging file you are hand-editing, the fields
`enrich` fills, the five `--prefer-incoming` refuses — is described in terms of
these keys.

Canonical records look like this:

```yaml
id: "word:話す:はなす"
expression: "話す"
reading: "はなす"
furigana: "話[はな]す"
romaji: "hanasu"
meanings:
  - "to speak"
  - "to talk"
part_of_speech: "verb"
verb_group: "godan"
transitivity: "intransitive"
examples:
  - japanese: "毎日、妻と日本語で話します。"
    furigana: "毎日[まいにち]、 妻[つま]と 日本語[にほんご]で 話[はな]します。"
    romaji: "Mainichi, tsuma to Nihongo de hanashimasu."
    english: "I speak Japanese with my wife every day."
conjugations:
  plain: "話す"
  negative: "話さない"
  past: "話した"
  past_negative: "話さなかった"
  te_form: "話して"
  potential: "話せる"
  passive: "話される"
tags:
  - "verb"
  - "godan"
  - "shirabe"
usage_notes: "Used for speaking or talking with someone."
source:
  type: "shirabe"
  imported_from: "shirabe-export.csv"
  row: 2
```

Only `id`, `expression` and `reading` are required; everything else is filled
by an importer, by `janki enrich`, or by you. An empty field means nobody
knew — janki never guesses one, and `janki status` counts what is missing.

`id` is minted from the expression and the reading, and the Anki note GUID is
derived from the id, so changing either orphans the review history behind the
card. See [CARD_DESIGN.md](CARD_DESIGN.md) and
[IMPORTING.md](IMPORTING.md#rows-held-back-for-reading-review).
