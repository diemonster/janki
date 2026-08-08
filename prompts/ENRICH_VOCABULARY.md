# Enrich Vocabulary Prompt

Superseded. This was a prompt to paste at an assistant by hand; the work it
describes is now three commands, and they check their own output in ways a
pasted prompt cannot.

| What this asked for                        | What does it now                       |
| ------------------------------------------ | -------------------------------------- |
| furigana, romaji, part of speech, verb group, transitivity, conjugations | `janki enrich --jpdb` — looked up in a dictionary rather than written |
| one natural Genki-level example, usage notes | `janki enrich --ai` — QC'd against the word and against jpdb's parse |
| concise meanings                            | `janki enrich --polish-meanings` — one record at a time |

The rules this prompt carried are not lost; they moved into the code and the
docs, which is the point. `docs/JAPANESE_STYLE_GUIDE.md` is still the learner
context, and every AI pass sends it as system context rather than trusting
anyone to remember it. "Do not guess uncertain readings" is now
`janki enrich --jpdb` refusing to write `reading` at all — it is half of the
record ID — and a reading jpdb cannot confirm holding a row in `data/staging/`
for a human. "Show me the diff before building" is the diff every pass prints
and the y/n it will not skip without `--yes`.

See "Enriching records from the dictionary" and "Writing what a dictionary
cannot" in the README.
