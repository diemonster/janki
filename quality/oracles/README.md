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

The four initial cases are downstream synthetic cases. They do not reproduce a
model reading source pixels, so they do not need a human unit oracle.
