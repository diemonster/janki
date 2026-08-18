You are completing a Japanese vocabulary card from the labelled record data.
Return its English glosses, complete its polite and casual example slots, and
give a usage note as one rich answer.

The English gloss list is the few senses this word actually carries, ordered
with the most common relevant sense first, in the plainest accurate English.
Verbs read as “to …” — “to speak”, not “speaking” or “speech”. Current meanings
and curated examples identify the sense for this record. Keep that sense first.
Remove restatements, part-of-speech labels, and dictionary hedges; include no
sense belonging only to a different register or identity.

Existing curated examples are reviewed sentences to preserve. When Existing
curated examples requiring annotations contains sentences, return every pinned
Japanese string exactly and fill its empty English, furigana, romaji, and
speech_level values. Do not replace or omit one. A polite or casual speech_level
already present on a curated example, or assigned to a pinned one, occupies that
card slot.

Write one new natural sentence only for a polite or casual slot that is not
already occupied. If both slots are occupied, add no new sentence. A new polite
example is everyday Japanese using 〜ます or 〜です and has speech_level “polite”.
A new casual example is everyday casual plain-form Japanese, as a friend would
say it, and has speech_level “casual”. Casual speech uses its own natural
particles and sentence-final forms rather than mechanically swapping an ending.
Every new sentence contains the expression itself, conjugated where that reads
naturally, and is simple enough for a beginner using Genki-style grammar.
Use the exact
spelling supplied in Expression: a kana expression stays kana and a
kanji expression keeps its kanji. Give each sentence a natural English
translation.

Reviewed lesson patterns in the record data are grammar the learner already
knows. Use one naturally when it suits this word; an ordinary natural sentence
is better when none fits. Recent examples from this run show structures and
settings already used, so choose genuinely different situations and language.

Give each sentence furigana in Anki notation. The whole furigana field
reproduces the same Japanese sentence in order, with each kanji's contextual
reading attached. A group is base[reading]. The base contains exactly the
characters that reading spells, and an ASCII space separates each ruby group
from what precedes it unless the group opens the field. Kana already in the base
remain as written. Thus 毎日話します。 is
`毎日[まいにち] 話[はな]します。`, not `毎日話[はな]します。`. Put an ASCII space
after punctuation when a ruby group follows it: `先週[せんしゅう]、 家族[かぞく]`.

Give each sentence Hepburn romaji with spaces between words. Spell particles
as spoken: は is wa, へ is e, and を is o. Write long vowels out as ou, uu, and
ii, never with macrons. Write ん as n, with an apostrophe before a vowel or y,
so `kin'en` cannot be read as `ki-ne-n`.

The usage note is concise and teaches a useful nuance, construction,
collocation, or register distinction that the glosses do not show. An empty
usage note is appropriate when there is nothing useful and certain to add.
