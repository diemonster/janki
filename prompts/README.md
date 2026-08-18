# The prompt templates janki sends

Every Anthropic system prompt is a file in this directory. It is sent **byte
for byte** — no placeholders, no template syntax, no conditional sections.
Record and source data are composed into labelled user turns in Python, and the
terse Pydantic field descriptions travel in the structured-output schema.

Edit one and the next run uses it. Nothing is cached and nothing is compiled,
so an edit can never be silently stale, and `git log prompts/` is the history
of why the asking changed.

## Which pass sends which file

| File | Sent by | As |
| --- | --- | --- |
| `style-guide.md` | every pass below except `approve-coverage` | system context, first block |
| `extract-auto.md` | `janki extract` with no `--mode` | system |
| `extract-table.md` | `janki extract --mode table` | system |
| `extract-prose.md` | `janki extract --mode prose` | system |
| `enrich-bare-word.md` | `janki enrich --ai` | system |
| `approve-coverage.md` | `janki promote --accept-coverage` | system |

The coverage check is the one pass that does not lead with the style guide.
It is not judging Japanese — it counts whether the page is accounted for —
and handing it a guide to writing good glosses is an invitation to volunteer
opinions about them, which is exactly what `approve-coverage.md` tells it not
to do.

The configured Codex enrichment provider receives the same style and task
template, but its transport adapter rejoins those blocks and prepends a small
JSON-only/no-tools preamble. That wrapper contains no Japanese-content policy;
the Markdown file remains the complete task instruction.

The **user turn** is not a file. It is the record's own data — the expression,
the reading, what janki already knows about the word — composed by Python. For
the passes that read a source, it also carries the page itself: the same
base64 image or PDF, so the model is looking at what you are looking at.
That is data, not instruction, and it is the only non-file content. The terse
schema labels and Codex transport preamble described above are the other
Python-owned pieces of request structure.

## The rules that keep this a directory of files

**One file per pass and input shape.** A prompt that would need an `if` in its
instruction prose is two prompts. That is why extraction has three files rather
than one file and three rule blocks: the modes ask for genuinely different
work, and reading `extract-table.md` should not require mentally deleting the
prose paragraphs. The cost is that shared closing paragraph, duplicated three
times. That is the intended trade — a reader of one file needs no other file.

**No placeholders.** If a prompt needs a record's data, Python puts it in the
user turn. A `{{expression}}` here would make the file something you have to
run to understand.

**Say what you want, not what to avoid doing wrong.** These files ask for
Japanese; they do not describe how janki will check the answer, because janki
does not check the answer. There is no rule engine behind this directory
grading the model's Japanese — that approach was built, measured, and deleted
(38 of 155 sentences flagged, not one true positive). If a card comes back thin
or wrong, the fix is here, in the asking.

**The paid paths record the complete asking.** Extraction stores prompt
provenance with both its candidates and its proposed source patterns. AI
enrichment fingerprints the provider, exact style guide, rich task template,
labelled record turn, provider-normalized transport prompt, and actual wire
schema. The coverage approval records its model and template fingerprint too,
so every accepted model proposal can be traced to the asking that produced it
via `git log prompts/`.

## Changing one

Edit the file, run the pass, look at the output. If a clause matters enough
that removing it should break something, `tests/test_prompts.py` is where that
assertion goes — it reads these files, so a retired clause is a failing test
rather than a silent weakening.

Today that holds for `enrich-bare-word.md`, the three `extract-*.md` files,
and `approve-coverage.md`. Assertions living in tests read the files too: they
once read Python constants holding byte-identical copies, which meant deleting
a clause from the live template changed nothing anyone would notice.
