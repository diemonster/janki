# Human unit oracles

A unit oracle pins one source SHA-256. Use `exhaustive` for a table or list and
`selection` for targets that a person selects from prose. An exhaustive oracle
uses ordered page, section, and ordinal keys. A selection oracle states its
selection rubric and does not claim that it accounts for all source content.

An oracle without `approval` is a draft. It cannot bind a pilot, case, replay
gate, promotion, or live baseline. Only the repository owner can supply the
approval. The approval must repeat the source fingerprint, oracle ID, oracle
type, selection rubric when present, and the normalized oracle-content
fingerprint that `japanese_anki.hardening.oracle_content_fingerprint` returns.

The initial downstream cases and the extraction-accounting regression case are
synthetic. They do not reproduce a model reading private source pixels, so they
do not need a human unit oracle.

## Exhaustive oracle

Use these fields:

- `id`, `source_fingerprint`, and `type: exhaustive`
- `case_ids`
- `units`, in page, section, and ordinal order

Each unit has `page`, `section`, `ordinal`, `context_fingerprint`, and one
disposition: `candidate`, `duplicate`, `non-vocabulary`, or `unreadable`.
Unit keys must be unique.

Create `context_fingerprint` with
`japanese_anki.extract.context_fingerprint`. This production function applies
Unicode NFC, changes each whitespace run to one space, and removes whitespace
at both ends. Extraction uses the same function before it compares an oracle.

## Selection oracle

Use these fields:

- `id`, `source_fingerprint`, and `type: selection`
- `case_ids`
- `targets`, where each item has `identity` and `locator`
- `selection_rubric`

Do not add `units` to a selection oracle. It does not claim that all source
content is accounted for.

## Approval

An owner approval has `authority: repository-owner`, `oracle_id`,
`source_fingerprint`, `oracle_type`, `oracle_content_fingerprint`, and
`approved_at`. It also repeats `selection_rubric` for a selection oracle.

Run `janki harden status`. Its draft-oracle output gives the content
fingerprint that the owner must review. A content change makes an existing
approval stale.

Pass approved oracles to extraction with `--coverage-oracle FILE`. Repeat the
option for a multi-input run. Each oracle must match exactly one prepared
source SHA-256. Janki checks all bindings before the first model call.
