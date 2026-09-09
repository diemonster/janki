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
| `style-guide.md` | every card-writing pass below | system context, first block |
| `extract-auto.md` | `janki extract` with no `--mode` | system |
| `extract-table.md` | `janki extract --mode table` | system |
| `extract-prose.md` | `janki extract --mode prose` | system |
| `enrich-bare-word.md` | `janki enrich --ai` | system |
| `revise-conjugation-deck.md` | confirmed conjugation-deck `revise` | system |
| `revise-cards.md` | confirmed canonical-card `revise` | system |
| `assistant-agent.md` | repository-wide ordinary Janki conversation and closed intent planning | complete system prompt |
| `approve-coverage.md` | `janki promote --accept-coverage` | system |

The coverage check and conversational Assistant do not lead with the style
guide. Coverage counts whether the page is accounted for rather than judging
Japanese. `assistant-agent.md` answers from bounded repository projections and
thread history and may return one closed intent, but it does not write cards;
the separately confirmed `revise-cards.md` pass does that work.

The configured Codex enrichment provider receives the same style and task
template, but its transport adapter rejoins those blocks and prepends a small
JSON-only/no-tools preamble. That wrapper contains no Japanese-content policy;
the Markdown file remains the complete task instruction.

The **user turn** is not a file. For card-writing passes it is the record's own
data — the expression, the reading, what janki already knows about the word —
composed by Python. For the passes that read a source, it also carries the page
itself: the same base64 image or PDF, so the model is looking at what you are
looking at. For the repository Assistant it carries bounded, fingerprinted
repository projections, optional deck focus, bounded visible thread history,
and the current message. That is data, not instruction.
The terse schema labels and Codex transport preamble described above are the
other Python-owned pieces of request structure.

## The rules that keep this a directory of files

**One file per pass and input shape.** A prompt that would need an `if` in its
instruction prose is two prompts. That is why extraction has three files rather
than one file and three rule blocks: the modes ask for genuinely different
work, and reading `extract-table.md` should not require mentally deleting the
prose paragraphs. The cost is that shared closing paragraph, duplicated three
times. That is the intended trade — a reader of one file needs no other file.

The last of those duplicated closing paragraphs asks for the answer as the bare
schema object through the structured-output tool: every required field present,
no property the schema does not define, no wrapper object and no JSON string
standing in for the object. It is byte-identical in all three files and has no
Python counterpart — no branch selects a variant of it. It is there because the
alternative is a decoder that repairs answers: a reply whose tool argument
arrived wrapped or stringified is still a reply somebody paid for, and
`janki extract-batch recover` reads exactly those shapes out of a capture and
refuses everything else. Expanding this paragraph is the first remedy; widening
that reader is not one.

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
schema. Revision fingerprints the exact selected deck bytes, owner instruction,
style/task prompts, labelled turn, model, provider, and response schema before
it can dispatch. The coverage approval records its model and template fingerprint too,
so every accepted model proposal can be traced to the asking that produced it
via `git log prompts/`.

## Changing one

Edit the file, run the pass, look at the output. If a clause matters enough
that removing it should break something, `tests/test_prompts.py` is where that
assertion goes — it reads these files, so a retired clause is a failing test
rather than a silent weakening.

Today that holds for `enrich-bare-word.md`, both `revise-*.md` files,
`assistant-agent.md`, the three `extract-*.md` files, and
`approve-coverage.md`. Assertions living in tests read the files too: they once
read Python constants holding byte-identical copies, which meant deleting a
clause from the live template changed nothing anyone would notice.
