Write **two** example sentences for the word, and a usage note if there is
something worth saying.

If the record prompt lists existing curated examples that need annotations,
return those exact Japanese strings instead of writing new examples. Fill their
empty English, furigana, and speech_level values. Do not rewrite, replace, or
add a Japanese sentence in that case.

The first sentence is polite (〜ます / 〜です); set its speech_level to
"polite". The second is the same kind of everyday sentence in **casual** plain
form, as a friend would say it; set its speech_level to "casual". Write a different sentence
rather than the same one with the ending swapped — casual speech drops
particles, uses different sentence-final forms (〜の, 〜んだ, 〜よ, 〜ね), and a
mechanical de-politening teaches none of that.

Both must contain the word itself, conjugated if that reads more naturally, and
both must be simple enough for a beginner working through Genki-style grammar.
Use the exact spelling shown in Expression. You can conjugate it, but do not
replace a kana-only expression with kanji or replace its kanji with kana.
Give each sentence's furigana in Anki notation. The whole field is read back
as the sentence, so it has to be the *same sentence* — every character of the
Japanese, in order, with readings attached — and not a list of the words that
carry kanji.

A group is `base[reading]`. The base is exactly the characters that reading
spells and nothing more, and an ASCII space separates each group from what
comes before it unless the group opens the field. Anki reads that space and
only that space: a full-width space is not a separator, and without one the
reading is drawn back over whatever precedes it. So 毎日話します。 is
`毎日[まいにち] 話[はな]します。` — not `毎日　話[はな]します。`, and not
`毎日話[はな]します。`, both of which say はな is the reading of 毎日話 and lose
毎日 from what the card speaks.

Readings are contextual and that is your judgement to make: 行 is い in 行く and
ぎょう in 銀行, 話 is はな in 話す and わ in 会話. Give the reading this sentence
uses. Kana already in the base stay as they are — they are their own reading.

Do not fill in romaji — janki generates that from the furigana and discards
whatever you send.

Say nothing you are not sure of. An empty usage note is a fine answer; an
invented nuance is not.