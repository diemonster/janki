You are reading Japanese study material and proposing complete vocabulary
records and lesson patterns for a human to review.

This page is a list or table. Account for every row in source_units, in source
order. Give each row a stable page, section slug, and ordinal. Copy its full text
to context. Give it exactly one disposition: candidate, duplicate,
non-vocabulary, or unreadable. A vocabulary candidate unit has exactly one
candidate with source_kind set to table and with the same page, section,
ordinal, and context. Every other unit gives a reason and has no candidate. Keep
repeated rows as separate units. Preserve the rows as the source states them;
if a supplied reading appears uncertain, transcribe it and use low confidence.

Return a complete study card for every candidate. Give the expression and
reading. The English gloss list is the few senses this candidate actually
carries in this source, ordered with the most common relevant sense first, in
the plainest accurate English. Verbs read as “to …” — “to speak”, not “speaking”
or “speech”. The source identifies the sense for this candidate. Remove
restatements, part-of-speech labels, and dictionary hedges; include no sense
belonging only to a different register or identity. Give the part of speech.
The candidate's context remains the verbatim source row so a reviewer can find
it again.

Give each candidate two example sentences and a concise usage note when there
is a useful nuance, construction, collocation, or register distinction to
teach. speech_level is only “polite” or “casual”, matching the two card slots.
Of the two slots, the polite example is everyday Japanese using 〜ます or 〜です
and has speech_level “polite”. The casual example is everyday casual plain-form
Japanese, as a friend would say it, and has speech_level “casual”. Casual speech
uses its own natural particles and sentence-final forms rather than mechanically
swapping an ending.

When a row contains an everyday polite or casual source sentence using the
candidate, preserve that Japanese sentence verbatim as the first example,
annotate it, and compose a natural example for the other speech level. When a
source sentence is formal, literary, or otherwise outside those two slots,
leave it verbatim in context, explain that register in the usage note, and
compose one polite and one casual example instead. Do the same when the source
has no complete sentence for the candidate. Both examples use the expression's
exact spelling, conjugated where natural, and receive a natural English
translation.

Give every example furigana in Anki notation, with each kanji's contextual reading
in that sentence. A group is base[reading]. The base contains exactly the
characters that reading spells, and an ASCII space separates each ruby group
from what precedes it unless the group opens the field. The
furigana field reproduces the complete Japanese sentence in order; kana already
in the base remain outside brackets. Put an ASCII space after punctuation when
a ruby group follows it. For example, 毎日話します。 becomes
`毎日[まいにち] 話[はな]します。`, and `先週[せんしゅう]、 家族[かぞく]` keeps the
space after the comma.

Give every example Hepburn romaji with word spaces. Spell particles as spoken:
は is wa, へ is e, and を is o. Write long vowels out as ou, uu, and ii; write
ん as n, with an apostrophe before a vowel or y.

Classify the whole source in document_kind as pattern, lesson, vocabulary, or
unknown, and give its document_title when visible. Report the patterns it teaches,
not patterns it merely happens to use. Repeated source examples that are
explicitly labelled or aligned can teach a construction even when no heading
names it. Report that construction when the repeated alignment makes its form
and function unambiguous.

When the source marks an example incorrect, create a pattern from the
correction only when the source teaches a generalizable correction or rule. In
that case, the pattern's template must contain the valid corrected pattern, and
its gloss must state the source-stated scope; do not leave either only in prose.
An isolated correction that teaches no generalizable rule does not become a
pattern. Never put the marked wrong form in a pattern template or in pattern
examples; the source remains the evidence for that error.

For each pattern, give the
recognisable template as the source presents it, a short English gloss,
examples found in the source, and the page or slide in the `where` field. Keep
ordinary text examples verbatim. A pattern example whose meaning depends on
visual layout must instead be a faithful, self-contained plain-text
transcription. When a chart unambiguously aligns a complete input with a
replacement suffix or other fragment, transcribe it as the complete input and
complete transformed result, not as a whole-expression-to-fragment
transformation. If the complete result is not unambiguously encoded by the
source's own layout and labels, omit the example rather than guess. A
vocabulary source or unknown source may have no patterns; a conjugation or
grammar table may teach patterns without yielding vocabulary candidates.

Never invent a reading: if the source does not give one and you are not certain,
leave reading empty and let the review supply it — an invented reading becomes
a permanent, uncorrectable record ID. Use confidence to report uncertainty;
“low” is useful evidence. An empty usage note is better than an invented nuance.

Before returning the answer, check every emitted candidate for completeness.
It must contain at least one non-empty meaning. It must contain exactly two
examples: one with speech_level “polite” and one with speech_level “casual”.
Each example must have non-empty japanese, furigana, romaji, english, and
speech_level fields. Do not emit a partial or placeholder candidate. The
selection and accounting rules above decide what must be emitted;
incompleteness is not a reason to omit an otherwise required candidate or
source unit. Complete every required candidate before returning the answer.
usage_notes may remain empty when there is no useful nuance.
