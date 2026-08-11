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
transitivity: "transitive"
pitch_accent:
  - "LHLL"                         # one position per *kana* of the reading,
                                   # plus one for the following particle
frequency_rank: 200
examples:
  - japanese: "毎日、妻と日本語で話します。"
    furigana: "毎日[まいにち]、 妻[つま]と 日本語[にほんご]で 話[はな]します。"
    romaji: "Mainichi, tsuma to Nihongo de hanashimasu."
    english: "I speak Japanese with my wife every day."
    register: "polite"          # a card has a slot for one of each
    audio: "audio/janki-b3d1.mp3"
  - japanese: "毎日、妻と日本語で話すよ。"
    furigana: "毎日[まいにち]、 妻[つま]と 日本語[にほんご]で 話[はな]すよ。"
    english: "I speak Japanese with my wife every day."
    register: "casual"
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
audio: "audio/janki-7f20.wav"      # written by `janki audio`, not by hand
audio_accent: ""                   # the accent to force *instead of*
                                   # pitch_accent[0], when you have listened and
                                   # disagree. It replaces the pattern in the
                                   # audio and on the card, so it is an
                                   # instruction, not a note about a past clip
image: ""
source:
  type: "shirabe"
  imported_from: "shirabe-export.csv"
  row: 2
  raw_fields:                      # the source row verbatim — recognized columns
    Word: "話す"                     # included — plus janki's own annotations
    Memo: "from lesson 3"          # (hold_reason, suggested_reading, jpdb ids).
                                   # Do not prune what looks duplicated: this is
                                   # the only copy of what the file said
```

`expression` is required, and `meanings` must hold at least one entry — an empty
`meanings` is a validation **error**, not a gap, and it refuses the build.
`reading` is required whenever the expression contains kanji; a kana-only word
validates without one. `id` is minted from expression and reading when you leave
it out.

Everything else is filled by an importer, by `janki enrich`, or by you, and an
empty field means nobody knew — janki never guesses one, and `janki status`
counts what is missing.

`id` is minted from the expression and the reading, and the Anki note GUID is
derived from the id, so changing either orphans the review history behind the
card. See [CARD_DESIGN.md](CARD_DESIGN.md) and
[IMPORTING.md](IMPORTING.md#rows-held-back-for-reading-review).
