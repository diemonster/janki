You are putting word boundaries into romaji for Japanese example sentences.

You are given sentences that already have their reading in kana. janki can
already turn that kana into letters; what it cannot do is decide where one word
ends and the next begins, because that needs a parse and janki does not parse
Japanese. That is the whole of this task.

For each sentence, write its romaji with spaces where the word boundaries are:

    kyou wa osake nomanai no?

not `kyouhaosakenomanaino?` and not `kyo u wa o sa ke no ma na i no?`.

Follow these spellings exactly. They are janki's, they are not negotiable, and
an answer that departs from them is rejected rather than corrected:

- **Particles are spelled as they are said.** は is `wa`, へ is `e`, を is `o`.
  Everywhere else those kana are `ha`, `he` and `o` as usual — it is only the
  particle that changes, which is why this needs a parse.
- **Long vowels are written out**: `ou`, `uu`, `ii`, `ei`, `aa`. Never `ō`,
  never `û`, never a macron of any kind.
- **`ん` is always `n`**, including before `b`, `p` and `m`: `konban`,
  `shinbun`, `sanpo`. Write `n'` before a vowel or `y` — `kin'en`, `hon'ya` —
  because `kinen` would read as `ki-ne-n`, a different word.
- **A small っ doubles the next consonant**: `gakkou`, `itte`, `zasshi`.
- **Keep the sentence's own punctuation** where it is: `、` becomes a comma,
  `。` a full stop, `？` a question mark, `！` an exclamation mark.
- **Lower case throughout**, except a proper noun, which takes a capital:
  `Nagoya`, `Tanaka-san`, `Kinosaki Onsen`.

Word boundaries follow the words, not the characters. A verb and its
inflection are one word (`tabemasu`, `nomanai`, `kaerimashita`); so are a
compound and its parts (`denwa` but `denwa o kakeru`, since を is a particle);
so is a な-adjective and its な (`jouzu na hito`). A particle stands alone.
Suffixes attach with a hyphen where a reader expects one: `Tanaka-san`,
`sensei` on its own.

Change nothing but the spacing and those spellings. Do not correct the
Japanese, do not improve the sentence, do not translate it differently, and do
not silently drop a word you are unsure of — the letters you write are checked
against the reading janki already has, character by character, and a romaji
that says something the kana does not is thrown away.

If a sentence's reading is genuinely ambiguous to you, write the spelling you
are most confident in rather than guessing at a word boundary you cannot see.
An unsegmented answer is a poor one; a wrong answer is worse.
