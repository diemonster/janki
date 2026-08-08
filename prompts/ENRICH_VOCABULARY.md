# Enrich Vocabulary Prompt

Superseded. This was a prompt to paste at an assistant by hand; the work it
describes is now three commands, and they check their own output in ways a
pasted prompt cannot.

| What this asked for                        | What does it now                       |
| ------------------------------------------ | -------------------------------------- |
| furigana, romaji, part of speech, verb group, conjugations | `janki enrich --jpdb` — looked up in a dictionary rather than written |
| transitivity                                | nothing — see below                    |
| one natural Genki-level example, usage notes | `janki enrich --ai` — QC'd against the word and against jpdb's parse |
| concise meanings                            | `janki enrich --polish-meanings` — one record at a time |

`transitivity` is the one thing here with no successor. It is set at import,
from what the source said, and no pass backfills it — a record from a photo or
a textbook with an empty `transitivity` keeps one until you type it. That is a
gap rather than a decision, and worth knowing before assuming these commands
cover everything this prompt asked for.

The rest of the rules are not lost; they moved into the code and the docs, which
is the point. `docs/JAPANESE_STYLE_GUIDE.md` is still the learner
context, and every AI pass sends it as system context rather than trusting
anyone to remember it. "Do not guess uncertain readings" is now
`janki enrich --jpdb` refusing to write `reading` at all — it is half of the
record ID — and an *import* holding a row in `data/staging/` when it cannot
work out the reading, where it waits for a human rather than reaching a card. "Show me the diff before building" is the diff every pass prints
and the y/n it will not skip without `--yes`.

See "Enriching records from the dictionary" and "Writing what a dictionary
cannot" in the README.
