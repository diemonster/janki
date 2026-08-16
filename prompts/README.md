# The prompts janki sends

Every instruction a model receives is a file in this directory. They are sent
**byte for byte** — no placeholders, no template syntax, no conditional
sections. What you read here is what the model reads.

Edit one and the next run uses it. Nothing is cached and nothing is compiled,
so an edit can never be silently stale, and `git log prompts/` is the history
of why the asking changed.

## Which pass sends which file

| File | Sent by | As |
| --- | --- | --- |
| `style-guide.md` | every pass | system context, first block |
| `extract-auto.md` | `janki extract` with no `--mode` | system |
| `extract-table.md` | `janki extract --mode table` | system |
| `extract-prose.md` | `janki extract --mode prose` | system |
| `enrich-examples.md` | `janki enrich --ai` | system |
| `polish-meanings.md` | `janki enrich --polish-meanings` | system |
| `patterns.md` | `janki patterns` | system |

The **user turn** is not a file. It is the record's own data — the expression,
the reading, what janki already knows about the word — composed by Python.
That is data, not instruction, and it is the only thing the model sees that is
not written here.

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

**Everything sent is fingerprinted.** The sha-256 of each file goes into
staging archives and batch records, so a card can be traced to the exact text
that produced it even after the file has moved on.

## Changing one

Edit the file, run the pass, look at the output. If a clause matters enough
that removing it should break something, `tests/test_enrich_ai.py`'s prompt
section is where that assertion goes — a retired clause should be a failing
test, not a silent weakening.
