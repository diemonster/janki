# Implementation Plan (v2)

Execution plan for `docs/DESIGN_V2.md`. Written for agents: each task is
self-contained, sized for one focused session, and carries its contract,
files, tests, and known traps. Read `DESIGN_V2.md` for the *why*; this
document is the *what and in which order*. Where this plan and DESIGN_V2
disagree on a mechanism, **this plan wins** (each deliberate deviation is
marked "supersedes design").

## How to work this plan

1. **Claim before you start.** Commit a one-line edit to this file on
   `main` changing the task's `[ ]` to `[~] claimed <agent-or-branch>
   <date>`. If that commit conflicts, the task is taken — pick another.
   Flip to `[x]` in the same change that completes the task (note any
   scope shift beside it).
2. Pick any unclaimed task whose **Depends on** entries are all `[x]`.
   The lane maps per milestone are derived from the dependency lists —
   the dependency lists are authoritative. Tasks touching `cli.py` are
   the highest-contention; land them smallest-first and rebase.
3. Before starting, read: `DESIGN_V2.md` (the section named in the task),
   `AGENTS.md` (project rules), and the files listed under **Files**.
4. Definition of done, every task, no exceptions:
   - **`make gates` passes** — ruff clean, pytest green (including the
     new tests the task specifies), sample deck builds. Run `make gates`,
     not its three parts: it anchors itself to the checkout it lives in,
     while a bare `pytest`/`janki` inside a linked worktree runs the
     *primary* worktree's code and reports green for a branch it never
     executed. You do not need to set `PYTHONPATH` or name a venv.
   - README/docs updated when user-visible behavior changed
   - the checkbox here flipped to `[x]`
5. Never modify `data/inbox/`. Never change `stable_record_id`, GUID
   derivation, or an existing record's `id` (single exception: M3.4's
   malformed-ID re-mint at promote time, specified there). Never insert
   into `FIELD_NAMES` (append-only, and only in task M5.4). Schema
   fields (`pitch_accent`, `audio_accent`, `frequency_rank`,
   `ExampleSentence.audio`) are added **only** in M2.2, while the sparse
   human-owned `ExampleSentence.instructions` escape hatch is M8.1 — if an
   earlier task needs one, use defensive access, don't add fields early.
6. **No live network calls in tests.** Every client (jpdb, Claude,
   VOICEVOX, OpenAI speech) takes an injectable transport; tests use fakes with
   canned responses.
7. Dependencies policy (`AGENTS.md`: prefer stdlib): jpdb/VOICEVOX/OpenAI speech
   clients use `urllib.request` (JSON-over-POST is trivial); the
   `anthropic` SDK is justified for AI tasks and lives in the optional
   extras group `ai` created by M3.1. HEIC conversion uses `sips` via
   `subprocess` on macOS; other platforms get a clear error naming
   `pillow-heif` as the workaround (accepted scope: single-user macOS
   tool). **`ruamel.yaml` (added in M2.6)** is the one exception to
   "PyYAML is the YAML library": it exists solely for
   `staging.rewrite_staging`, which annotates a file a human is part-way
   through reviewing. PyYAML cannot round-trip comments or keys outside
   the record schema, so re-rendering that file deletes the reviewer's
   own notes — the one kind of work in this repository that exists
   nowhere else. Everything janki writes *from scratch* still goes
   through PyYAML; do not widen this.
8. **Paid model calls are milestone operations, not routine verification.** Run
   local validation and `make gates` first — they are free and they catch most
   of it. *(Through 2026-08-15 this rule also described a "final semantic
   review" over a release candidate; `janki review` was deleted in M8.2 and
   there is no such pass. The paid calls that remain are `extract`, `enrich
   --ai`, and `promote --accept-coverage`.)*
   Do not run a paid pass after each small commit or on a media-only diff. Any
   additional broad live review needs explicit repository-owner approval.
   Tests and `make gates` never make a live request.

## Conventions (introduced in M1/M3.1, used by everything after)

- All janki exceptions subclass `JankiError` (M1.2); `cli.main()`
  catches `JankiError` once. Error homes: `JpdbError` →
  `jpdb.py` (M2.1), `ExtractError` → `extract.py` (M3.3), `EnrichError`
  → `enrich.py` (M2.6), `PitchError` → `pitch.py` (M5.1), `AudioError`
  → `audio_cmd.py` (M5.3), `LedgerError` → `ledger.py` (M1.3).
- All JSON writes (records, ledger) go through one atomic
  write-temp-then-rename helper (M1.2): sorted records with stable
  (declaration-order) field keys, trailing newline, LF-only,
  `ensure_ascii=False`. The ledger may pass `sort_keys=True` at its own
  call site if it wants alphabetical keys.
- **Fingerprints** (formulas live in `ledger.py` (M1.3) — never restate them
  elsewhere). Two families, deliberately different:
  - *Filename* fp (stable addresses): word audio `fp(record.id)`;
    example audio `fp(record.id + example.japanese)`, both through the frozen
    48-bit `identifiers.short_fingerprint` formula.
  - *Content* fp (staleness detection): a raw, framed SHA-256 over the exact
    provider request. Word audio binds its forced/natural mode and the one
    string actually sent (forced AquesTalk or bare reading); example audio
    binds the exact Japanese sentence. The formulas live only in `ledger.py`.
- **Staging file shape** (M1.5): a YAML mapping with `records:` (list of
  record dicts — the shape `load_records` already accepts) plus metadata
  keys the loader ignores (`source_file`, `extracted_at`, `model`,
  `review_notes`). Per-record annotations (`hold_reason`,
  `already_known`, `suggested_reading`) are stringified into that
  record's `source.raw_fields` so records round-trip through
  `VocabularyRecord.from_dict` unchanged.
- ~~**Unverified-furigana convention**~~ *(deleted 2026-08-15 by M8.3 with the
  furigana flag subsystem it served. `furigana_unverified`,
  `io.CONTENT_ANNOTATIONS` and the flag's merge carry are all gone; nothing
  writes or reads that key. Struck rather than removed because this section is
  headed "used by everything after", and a reader who remembers the convention
  should find out here that it went.)*
- **Field-diff output** (shared helper, first built in M2.6, reused by
  M4.2/M4.3): per record, per field, one indented line of the form
  `<field>: <old> -> <new>` under a `<record id>` header.
- **Mutation testing** is safe by default and must stay that way.
  CPython validates a `.pyc` against its source's *mtime in whole seconds
  and byte size*, so a sweep's back-to-back edits — revert mutant A,
  apply mutant B — land in the same second at the same size (`==` → `!=`,
  `<` → `<=`, swapped arguments) and Python silently reuses A's bytecode.
  Mutant B never runs, its result is really A's, and a test gap is
  reported as covered. It fails *toward* false confidence, so it never
  announces itself. `conftest.py` purges `src/**/__pycache__` and sets
  `sys.dont_write_bytecode` before the first import, and the `Makefile`
  exports `PYTHONDONTWRITEBYTECODE=1` for the CLI paths pytest cannot
  reach — so run mutants through `make test` / `make gates` and do not
  bypass either. Still prove the harness before trusting a clean sweep:
  apply one mutant you are certain is unguarded and confirm it survives.

---

## Milestone 1 — Foundations

Lane map (derived from deps): {M1.2} → {M1.1, M1.6} → {M1.3, M1.5} →
{M1.4, M1.7, M1.8}. The last wave's three tasks all touch `cli.py` —
land smallest-first.

### [x] M1.2 Error base + atomic writes

*Done 2026-08-06, revised same day after adversarial review. Scope notes:
(1) the repo had 8 pre-existing `ruff` violations (unrelated files) that
made the "ruff clean" gate unsatisfiable for every task; cleared with
`ruff check --fix` in the same change. (2) Review hardening: the temp file
is uniquely named via `tempfile.mkstemp` (a fixed `path.tmp` name lets
concurrent writers corrupt each other — the contract below was amended),
content is fsynced before the rename (+ best-effort directory fsync),
symlinked targets are resolved so writes go through to the real file,
permission bits of an existing target are preserved, cleanup can never
mask the original error, output is always LF, and filesystem `OSError`s
surface as `DataError` so the CLI prints `error: ...` instead of a
traceback naming a temp file. (3) Small pre-existing clean-error gaps
fixed in passing, per the behavior-vs-HEAD review: `load_structured` and
the CSV reader wrap all `OSError`s (not just missing-file), and
`load_records` rejects non-mapping items with a `DataError` instead of an
`AttributeError` traceback.*

Depends on: — (first task; several others build on it)
Files: new `src/japanese_anki/errors.py`, `src/japanese_anki/io.py`,
`src/japanese_anki/cli.py`, all modules defining error classes,
new `tests/test_io_atomic.py`.

- Add `JankiError(Exception)` in `errors.py`. Re-parent `AnkiBuildError`,
  `ConfigError`, `DataError`, `ShirabeImportError` onto it (keep their
  names and defining modules — imports elsewhere must not break).
- `cli.main()`: replace the 4-tuple `except` (cli.py ~line 204) with
  `except JankiError`.
- Add `atomic_write_text(path, text)` to `io.py`: a **uniquely named**
  temp file in the target's directory (`tempfile.mkstemp` — never a
  fixed `path.tmp` name, which lets concurrent writers corrupt each
  other), fsync, then `os.replace`; resolve symlinks first; preserve an
  existing target's permission bits; wrap `OSError` in `DataError`.
  Route `save_records_json` through it.
- Tests: atomic write leaves no tmp file on success; original file
  intact when the serializer raises mid-write; a brand-new
  `JankiError` subclass is caught by `main()` without registration.

### [x] M1.1 Curation-safe merge + import CLI update

Depends on: M1.2
Files: `src/japanese_anki/io.py`, `src/japanese_anki/cli.py`,
`tests/test_merge.py`.
Design: DESIGN_V2 "Curation-safe merge".

Contract for `merge_records(existing, incoming, prefer_incoming=())`:

- **Empty means `""`, `[]`, `{}`, or `None`. Zero is not empty.**
  Existing wins: an incoming value lands only where the existing field
  is empty. `tags` = sorted union. `source` = existing record's source,
  unconditionally (first source sticks).
- Fields named in `prefer_incoming` use old incoming-wins behavior.
  It applies to content fields only — naming `tags`, `source`, `id`,
  `expression`, `reading`, or an unknown field is a CLI error.
- Returns `(records, outcomes)` where `outcomes` maps record id →
  `MergeOutcome` for **exactly the incoming record IDs** (untouched
  existing records do not appear). Outcome label per record in
  `{added, filled, unchanged, conflicting}` (precedence:
  conflicting > filled > unchanged) plus `filled_fields` and
  `conflicts: [(field, existing, incoming)]`.
- Tests must include: fill-only semantics, tags union, source
  preservation, conflict reporting, `prefer_incoming` override, and a
  `None`-valued field being fillable (guards M2.2's `frequency_rank`).
- **Trap:** `tests/test_merge.py::test_merge_preserves_existing_enrichment`
  asserts incoming-wins on `meanings` — the test encodes the bug the
  design fixes. Rewrite the whole module to the new contract.

CLI (`command_import_shirabe`):

- The unconditional hand-built `counts` dict at the top of the function
  and the summary print both move to the outcome-map shape (they
  KeyError otherwise). Summary: counts per outcome + one line per
  conflict (id, field, both values).
- Add `--prefer-incoming FIELD[,FIELD]` and `--yes`. `--replace`
  prompts "Replace N existing records? [y/N]" where N is the record
  count loaded from the existing output file (the `--replace` path must
  now load it just to count); non-TTY without `--yes` **proceeds** (the
  guard is fat-finger protection, not CI protection). Test via
  monkeypatched `sys.stdin.isatty` / `builtins.input`.

### [x] M1.6 Config v2

*Done 2026-08-06. Two small additions beyond the contract, both inside the
unknown-key warning: a secret-looking key in TOML (`*key*`, `*token*`,
`*secret*`, `*password*`) is warned about by name pointing at the env vars
instead of being reported as a typo, and a known section whose value is not
a table warns and falls back to defaults rather than crashing `_get` with an
`AttributeError`. TOML key for the provider is `[tts] provider` (per
DESIGN_V2); the dataclass field is `tts_provider` (per this plan).*

Depends on: M1.2 (both edit `config.py` — M1.2 re-parents `ConfigError`)
Files: `src/japanese_anki/config.py`, `janki.toml`,
`tests/test_config.py` (new).
Design: DESIGN_V2 "Configuration".

- New `ProjectConfig` fields (frozen dataclass, resolved against root):
  `ledger_file` (default `data/ledger.json`), `staging_dir`
  (`data/staging`), `media_dir` (`data/media`), `scan_inbox`
  (`data/inbox/scans`), `extract_model` / `enrich_model` (both
  `claude-opus-5`), `tts_provider` (`voicevox`), `voicevox_url`
  (`http://localhost:50021`), `voicevox_speaker` (int, 46),
  ~~`azure_voice`, `azure_region`~~ *(deleted 2026-08-16 with the rest of the
  Azure path — nothing read them; M5.7 dropped the provider)*.
  TOML sections: `[paths]`, `[ai]`, `[tts]`.
- Unknown TOML keys/sections emit a warning to stderr naming the key
  and the nearest valid one (today typos are silently ignored).
- Secrets are never read from TOML. Env vars only: `ANTHROPIC_API_KEY`,
  `JPDB_API_KEY`, `OPENAI_API_KEY` *(was `AZURE_SPEECH_KEY`; corrected
  2026-08-16)*.
- Tests: defaults when sections absent; overrides; unknown-key warning.

### [x] M1.3 Ledger module

*Done 2026-08-06. Contract as written; four call-site details downstream
tasks need: mutators change memory and return whether they changed
anything — call `ledger.save()` once per command (a save per record would
fsync the whole file per record). A source reference's identity is every
key but `seen_at`, so a re-import of the same file appends nothing.
Audio entries are keyed by `file`, so regenerating one (`--force`, new
voice) replaces its entry instead of adding a second. `missing_audio` is
word audio only (the design's "words missing audio"); example audio is
optional and per-example, and `stale_audio` covers both kinds.*

Depends on: M1.2, M1.6
Files: new `src/japanese_anki/ledger.py`, new `tests/test_ledger.py`.
Design: DESIGN_V2 "The ledger" (JSON example = schema, with one
refinement below).

- Per-record entry: `added_at`, `sources` (append-only list; appending
  an identical source ref is a no-op), `enriched` — **a list** of
  `{at, kind: "jpdb"|"ai", model, fields}` entries (kind `jpdb` writes
  `model: "jpdb"`; supersedes the design example's single object, so
  jpdb and AI passes never overwrite each other), `audio` (list of
  `{file, of: "word"|"example", provider, voice: int, content_fp, at}`),
  `exports` (`dict[deck_file_stem, iso_date | {at, missing}]`), and a top-level
  `pending_batches` section (consumed by M4.4).
- API (idempotent, persisted via the atomic writer, file sorted by
  record id): `load(path)`, `record_added`, `record_source_seen`,
  `record_enriched`, `record_audio`, `record_export`, `remove`.
- Query helpers for `status` / `build --only-new`:
  `unexported(deck_stem, ids)`, `missing_audio(records)`,
  `stale_audio(records, word_provider=…, example_provider=…)` (compare stored
  content and render profile — engine, voice, rate and provider settings —
  against the current record and configuration), `missing_enrichment(records)` — **defined
  purely from record content** (empty meanings or examples; M7.6P made
  `usage_notes` optional, so an empty note is complete), with ledger entries as
  metadata only, so a jpdb-only pass never hides a record from `enrich --ai`.
- Both fingerprint families (see Conventions) are implemented here as
  helpers; `stale_audio` uses defensive access for `pitch_accent` /
  `audio_accent` (`getattr(record, "pitch_accent", [])`) — the schema
  fields arrive in M2.2; do not add them here.
- Dates are ISO `YYYY-MM-DD` strings. `LedgerError` defined here.
- A missing ledger file is an empty ledger, never an error.

### [x] M1.5 Reading rules + staging module

*Done 2026-08-06. Both "current-code facts" below re-verified against the tree
before any edit; both held. Three small decisions inside the contract:
(1) `StagingError(JankiError)` is defined in `staging.py` (the Conventions list
of error homes predates this module); (2) the malformed-ID error is reported
when the missing-reading error is not — while the reading is still empty the
existing message states the same fault, and a staging file under review would
otherwise carry two errors per row for one fix, so the new check fires exactly
where it adds information (a reading filled in without a re-mint); (3) the
README section the validation messages point at belongs to M1.W, which still
owes it. Post-a91af32, diversion also covers rows whose reading slot is itself
written in kanji — including the empty-Word fallback where a kanji Reading
cell becomes the expression — staged with `hold_reason: "reading contains
kanji"` beside the original `"missing reading"` class; M2.5/M3.4 must handle
both `hold_reason` values.*

Depends on: M1.1 (rewrites the same CLI function), M1.2, M1.6
Files: new `src/japanese_anki/staging.py`,
`src/japanese_anki/importers/shirabe.py`,
`src/japanese_anki/validation.py`, `src/japanese_anki/cli.py`,
`tests/test_staging.py` (new), `tests/test_validation.py`,
`tests/test_shirabe_import.py`.
Design: DESIGN_V2 "Readings are ID-constitutive" + "Staging file shape".

- `staging.py`: `write_staging(path, records, meta, force=False)` /
  `read_staging(path) -> (records, meta)` implementing the pinned shape
  (see Conventions). `write_staging` refuses to overwrite unless
  `force=True`.
- Shirabe importer: rows whose expression contains kanji and whose
  reading is empty are excluded from the import result and returned on
  a new `ImportResult.needs_reading` list; the CLI writes them to
  `<staging_dir>/shirabe-<csv-stem>-needs-reading.yaml` (with
  `hold_reason: "missing reading"` in each record's `raw_fields`) and
  prints where they went and why. **Collision rule:** if that staging
  file already exists, do not overwrite — print its path and the count
  of rows that would have been written, and complete the rest of the
  import normally (the rows are still excluded).
- Staged rows keep their as-minted malformed ID (`word:<expr>:`) — that
  *is* the review signal `validate` surfaces; M3.4's promote re-mints
  it once a human has confirmed the reading.
- **Current-code facts (verified — don't "fix" the wrong thing):**
  `validation.py` *already errors* on kanji-with-empty-reading
  (`validation.py:40`); what warns-and-imports-anyway today is the
  importer (`shirabe.py:227-230`) — that is the behavior this task
  replaces with diversion. And the importer *already* defaults
  `reading = expression` for kana-only rows **before** the ID is minted
  (`shirabe.py:184-190`) — **keep that**; changing it would re-ID every
  kana-only record. Add an ID assertion for a kana-only fixture row to
  `test_shirabe_import.py` as a guard.
- `validation.py`: add the genuinely new check — **error** on malformed
  IDs of the form `word:<expr>:` where `<expr>` contains kanji (catches
  records whose reading was later fixed without re-IDing). Messages
  point at staging review generically ("route through data/staging
  review; see README") — no config plumbing into validation.

### [x] M1.4 `janki status`

*Done 2026-08-06. Contract as written; five decisions taken inside it.
(1) Where an id exists both in the normalized file and as an inline deck note,
the deck-resolved record wins — it is what that deck exports today and what
M1.7 will make authoritative. (2) A deck file that cannot be read is a warning
on stderr and a skipped deck, not a dead report: status is the command you run
to find out what is wrong. (3) Under `--format ids` every human-readable line
(warnings, the `--rebuild` report) goes to stderr so stdout is nothing but ids;
with no detail flag the id list is every record. (4) `--rebuild` writes audio
entries as `provider: "unknown"`, `voice: -1`, `rebuilt: true` — a file on disk
does not say which engine spoke it, and the configured provider would be a
plausible lie — and never replaces an existing entry that does know. (5) A
rebuilt entry gets a content fingerprint only where the filename proves the
content (example audio always; word audio only while the record carries no
accent data, since the filename covers the id and so the reading, but never the
pattern). Unprovable word audio is recorded with an empty fingerprint and
therefore reported stale, which is the safe direction. The README `status`
section is still M1.W's.*

*Amended 2026-08-07 by the end-of-milestone review; the task itself is unchanged.
`status` gained a `Staged for review:` line and a `--staged` detail flag (ids +
`hold_reason`, pipeable like the others). Held rows are the one category of
record that is not in the collection and needs a human, and nothing but the
one-off line in the import output mentioned them.*

Depends on: M1.3
Files: `src/japanese_anki/cli.py`, new `src/japanese_anki/status.py`,
`tests/test_status.py` (new).
Design: DESIGN_V2 "The ledger" (status + duplicate detection).

- `janki status`: summary table — record totals per source type, counts
  never-exported (per deck stem), missing audio / enrichment, stale
  audio. Pitch-accent-derived counts use defensive access and report
  "n/a until schema lands" before M2.2 (do **not** add the fields here).
- Flags: `--unexported`, `--missing-audio`, `--duplicates`, `--rebuild`,
  `--format ids` (bare IDs, one per line).
- `--duplicates`, three passes: (a) same expression, different ID;
  (b) same reading, different expression where one expression equals
  the other's reading (kana form) or both share a jpdb `vid` in
  `source.raw_fields`; (c) the same non-empty jpdb `vid` under more
  than one ID regardless of reading — a shared vid is the same
  dictionary word even when a reading was hand-corrected or is empty,
  which pass (b) cannot see. Vids compare numerically where possible
  (`1577980` == `"1577980.0"`) and a value that is not a positive
  integer is not a vid at all (`"None"` from a stringified null, `-`,
  `n/a`): a placeholder is shared by every record that has no vid, so
  grouping on one would report the whole file as a single duplicate
  under a heading whose remedy is deletion. A vid group whose IDs an
  earlier group already covers is not reported twice. Output: grouped
  pairs + a reminder that resolution is manual.
- `--rebuild`: reconstruct `sources` from each record's `source`, and
  `audio` entries from files in `media_dir` matching the filename
  fingerprint formulas; print that export state is not reconstructible.
- Record universe = normalized file **plus inline deck notes** (via
  `resolve_deck_records` per deck YAML) until M1.7 migrates them.

### [x] M1.7 `janki migrate-inline`

*Done 2026-08-07, and run on this repository: `data/decks/verbs.yaml`'s three
notes now live in `data/normalized/vocabulary.json`. Scope notes. (1) Migrating
into the shared normalized file would have handed
`data/decks/personal-vocabulary.yaml` — which reads every record in that file
with no filter — three notes carrying the starter deck's GUIDs. Preserving that
deck's contents is part of an honest migration, so the command gives every other
deck reading the same file an `exclude_ids:` entry for the records it did not
have before (unless its own `include_ids:` names one) and reports it. (2)
`include_ids:` is written only when the deck had no `source:` yet; a deck that
already read the normalized file keeps its own membership rule, since pinning it
would freeze out the imports it exists to receive. (3) Notes with no `id:` are
migrated too, under the id their cards already use — the plan said "with an
`id`", but leaving them behind would leave the `notes:` section this command
promises to remove. (4) `io._is_empty` became public `io.is_empty`: migrate
decides which inline fields to prefer with the merge's own emptiness rule rather
than a copy of it.*

*Amended 2026-08-07 by the end-of-milestone review; the task itself is unchanged.
Scope note (1) now reads: a deck with a **non-empty `include_ids:`** gets no
`exclude_ids` at all — `resolve_deck_records` applies `include_ids` first, so its
membership is already closed and any exclusion written into it is provably dead
config, accumulating one line per future migration in a file a human reads. Such
a deck is still reported when its own `include_ids` names a migrated id. Two more
corrections: the ledger reference migrate writes is the stored record's own
`source.type` / `source.imported_from` — the shape `status --rebuild`
reconstructs, not the deck path, which had one rebuild double every migrated
record's sources in the git-tracked ledger; and appending an id to an existing
`exclude_ids:` is now a text edit like any other addition, with the
re-serialization fallback reported as a warning rather than silently deleting a
curated file's comments.*

Depends on: M1.1, M1.3
Files: `src/japanese_anki/cli.py`, new `src/japanese_anki/migrate.py`
(new module — do not add this to `io.py`),
`tests/test_migrate_inline.py` (new).
Design: DESIGN_V2 "Inline notes migration".

- `janki migrate-inline data/decks/verbs.yaml`: move each inline note
  with an `id` into `vocabulary.json` (merge via M1.1 semantics with
  the inline record's **non-empty `MERGEABLE_FIELDS`** as
  `prefer_incoming` — inline is authoritative for content. Identity
  fields, `tags`, and `source` are excluded by M1.1's
  `validate_prefer_incoming` and need no override: the id is equal by
  construction, and the tags-union / first-source-sticks rules are the
  desired behavior), register in the ledger, and rewrite the deck
  YAML with the notes removed and an `include_ids:` list preserving
  exactly the previous deck membership.
- IDs byte-identical before/after (assert in test). Building the deck
  before and after migration produces the same note set (same GUIDs,
  same field values) — assert via the fake-genanki harness in
  `tests/test_anki_builder_contract.py`.

### [x] M1.8 Import ↔ ledger wiring + shared import summary

*Done 2026-08-07. The shared helper is `cli.run_import(config, records, *,
source_type, source_ref, output_path, unit, staging_stem, needs_reading=(),
warnings=(), prefer_incoming=(), replace=False, assume_yes=False) -> int`; it
owns the whole sequence (`--replace` confirmation, staging, merge, records
write, ledger, printing), not just the last three steps. Scope note: M1.5's
`_stage_needs_reading` took the source `Path` and hardcoded
`shirabe-<stem>-needs-reading.yaml`; it now takes `staging_stem` and
`source_ref`, since a jpdb deck sync has no file to take a stem from.*

*Amended 2026-08-07 by the end-of-milestone review; the task itself is unchanged.
The signature gained `held_unit="row(s)"` (the held-rows line was the one place a
caller's `unit` did not reach). `run_import` is now also the **named owner of
`Ledger.remove`**, which M1.3 built and nothing called: the `--replace` path drops
the ledger entry of every record it discards and reports the count, so the one
command that shrinks the collection no longer leaves entries behind that
over-report it — and, once M5.6 makes `build --only-new` read `exports`, would
silently keep a re-imported record out of a deck. And a `book.save()` that fails
is now a warning printed *after* the full summary rather than an `error:` instead
of one: vocabulary.json and the staging file are already written by then, so the
bare error told the user the import had not happened.*

*Amended 2026-08-07 by the review of that amendment. "Discarded from this file" is
not "gone from the collection": the ledger is collection-wide, the collection is
the normalized file **plus every deck's inline notes** (`status.collect_records`),
and `--output` can point `--replace` at a file the collection does not contain at
all. An id is therefore pruned only when `status.surviving_ids` can be built and
does not hold it; when it cannot be — an `--output` elsewhere, a deck that will
not parse — every entry stays and the summary says why. Removing an entry
destroys `added_at` and `exports`, which `--rebuild` documents as not
reconstructible; leaving one only over-reports, which `status` shows. Relatedly,
`Ledger.save` now re-raises the atomic writer's `DataError` as `LedgerError`, so
the warning-over-a-full-summary path above actually fires: `DataError` is a
sibling under `JankiError`, and every real filesystem failure has that shape.*

Depends on: M1.1, M1.3
Files: `src/japanese_anki/cli.py`, `tests/test_import_ledger.py` (new).

- Wire `command_import_shirabe` to the ledger: `record_added` for
  `added` outcomes, `record_source_seen` for every incoming record —
  this is what makes "later sightings go to the ledger" true from
  Milestone 1, not Milestone 2.
- Extract the import-summary printing + merge + ledger sequence into a
  shared helper in `cli.py` that `import-jpdb` (M2.5) will call — the
  design's "identical pipeline" requirement, made concrete here.

### [x] M1.W Milestone 1 wrap

*Done 2026-08-07. README only, as scoped. It settles the two sections M1.4 and
M1.5 deferred here; M1.5's validation messages turned out to be self-contained,
naming no README section, so the new staging text stands on its own rather than
being pointed at. `migrate-inline` was already documented by M1.7 and was
verified, not rewritten. Every documented command was run before committing.*

Depends on: all M1 tasks
Files: `README.md`.

- README: `status`/ledger section; note the merge-semantics change and
  `--prefer-incoming`; document the needs-reading staging flow.

---

## Milestone 2 — jpdb

Lane map: {M2.1, M2.3, M2.4 in parallel; M2.2 after M1.5} →
{M2.5, M2.6, M2.7} (M2.6 after M2.5 — shared POS/table imports and
`enrich.py` is new in M2.6; M2.5 and M2.7 both small on `cli.py`).
M2.1F is **owner-only** (it needs a live `JPDB_API_KEY`) and blocks nothing:
M2.1 ships against a labelled community-shape fixture and every task after it
proceeds on that. Nothing else in this milestone needs a human in the loop.

### [x] M2.1 jpdb API client (+ `jpdb ping`, POS table, /parse contract)

*Done 2026-08-07. Contract as written; five call-site details later tasks need.
(1) `lookup_vocabulary` attaches the requested `vid`/`sid` to every returned dict
even when they were not in `fields`, so no caller re-zips results against its own
input; `DEFAULT_LOOKUP_FIELDS` is DESIGN_V2 step 3's list unchanged. (2) `as_pair`
accepts a pair, a `{"vid","sid"}` dict (what `list_deck_vocabulary` returns) or an
object with those attributes — pairs are the wire shape, not the call shape.
(3) `parse()` takes one text and sends it as a list of one, and reads the response
tokens under either nesting; `position_length_encoding` is sent only when forced
furigana or `position`/`length` token fields are in play. (4) `pos_to_transitivity`
joins the two named POS tables: `vt`/`vi` are JMDict POS codes, and the whole point
of a single owner is that M2.5 does not build a third table for them. (5) The
fixture carries five golden furigana cases, not four — `お茶` (a bracketed group
after kana) and `日本語` (two adjacent groups) are different halves of the spacing
rule. README is untouched: `janki jpdb ping` is documented by M2.W with the rest of
the jpdb setup.*

Depends on: M1.2
Files: new `src/japanese_anki/jpdb.py`, `src/japanese_anki/cli.py`,
new `tests/test_jpdb_client.py`, new `tests/fixtures/jpdb-parse-sample.json`.
Design: DESIGN_V2 "jpdb.io > API client".

- `JpdbClient(api_key, transport=None)` — transport is a callable
  `(url, json_body, headers) -> (status, json)` defaulting to
  `urllib.request`; tests inject fakes.
- Endpoints wrapped: `ping`, `list_user_decks(fields)`,
  `list_deck_vocabulary(deck_id, fetch_occurences=False)`,
  `lookup_vocabulary(pairs, fields)` (batched, ~200 pairs per call),
  `parse(...)`. Column-oriented responses are zipped into dicts
  **inside the client**; callers never see positional rows.
- Retry with exponential backoff + jitter on 429 and `api_unavailable`
  (max ~5 tries); accept any 2xx (the spec is inconsistent: 201 on some
  endpoints). `JpdbError(JankiError)` carries the API's stable `error`
  id. Use the misspelled keys (`fetch_occurences`, `occurences`) — add
  a comment so nobody "fixes" them.
- **The `/parse` contract is pinned here** (M2.6 codes against it):
  - Request: `token_fields=["vocabulary_index", "furigana"]`,
    `vocabulary_fields=["vid", "sid", "spelling", "reading",
    "pitch_accent", "frequency_rank", "part_of_speech"]`, plus
    `position_length_encoding="utf16"` whenever positional data or
    forced furigana is involved.
  - Forced-furigana wire shape: `[[position, length, reading]]` with
    position/length in the request's encoding. Provide
    `forced_furigana_span(expression, reading, encoding)` computing the
    whole-expression span; test with an ASCII-safe and a
    UTF-16-surrogate string.
  - Token furigana shape (community-verified): a list of segments, each
    either a plain kana string or a `[text, reading]` pair. Parse it
    defensively. **Fixture:** commit
    `tests/fixtures/jpdb-parse-sample.json` hand-written to the
    community shape, wrapped so the marker cannot be mistaken for part
    of the response: the file is `{"_source": "community-documented
    shape, not a captured response", "response": {...}}`, tests read
    only the `response` half, and dropping `_source` in M2.1F changes
    nothing the client sees. **An agent may proceed on the
    community shape** — capturing a live response needs the owner's
    `JPDB_API_KEY`, which no agent has, and blocking the milestone on
    it is worse than a labelled fixture. Reconciliation is M2.1F.
  - `furigana_to_anki(segments) -> str`: `[text, reading]` →
    `text[reading]`; plain segments verbatim; a space before each
    bracketed group except at string start (Anki's furigana rule).
    話す → `話[はな]す`. Golden tests: leading-kanji, mid-kanji
    (okurigana), all-kana, multi-kanji compound. **The fixture must
    carry all four cases and the golden tests must be parametrized
    over tokens loaded from it** — otherwise M2.1F's acceptance gate
    ("the golden tests stay green against the captured response")
    tests nothing.
- **JMDict POS mapping lives here** (single owner; M2.5 and M2.6
  import it): `pos_to_verb_group` / `pos_to_part_of_speech` tables —
  `v5*`→godan, `v1`→ichidan, `vs*`→suru, `vk`→kuru, `adj-i`/`adj-na`
  adjectives, nouns et al.
- CLI: register `janki jpdb ping` (prints ok/failure from the client).

### [x] M2.1F Reconcile the /parse fixture with a live capture

Depends on: M2.1
Files: `tests/fixtures/jpdb-parse-sample.json`,
`tests/test_jpdb_client.py`, `src/japanese_anki/jpdb.py` (docstring only —
the parser needed no change), `docs/IMPLEMENTATION_PLAN.md`.

Done 2026-08-07 with a live capture. The key lives in `~/.zshrc`, which is
sourced only for *interactive* shells, so an agent's non-interactive shell
does not see it; capture with `zsh -ic` or move the export to `~/.zshenv`.

To re-capture (the request must stay this exact sentence — the five golden
furigana cases live in it, and the field lists are `DEFAULT_TOKEN_FIELDS` /
`DEFAULT_VOCABULARY_FIELDS`):

```sh
curl -sS https://jpdb.io/api/v1/parse \
  -H "Authorization: Bearer $JPDB_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"text":["日本語を話す。お茶と食べ物をたべる。"],
       "token_fields":["vocabulary_index","furigana"],
       "vocabulary_fields":["vid","sid","spelling","reading",
                            "pitch_accent","frequency_rank","part_of_speech"]}'
```

Store it as the fixture's `response` value and run `make gates`. The key is
read from the environment and must never be pasted into the repo.

**What the capture changed.** `jpdb.py`'s parser was correct — it read the
live response without modification, and `furigana_to_anki` emitted correct
Anki notation for every token. Only hand-written *expectations* were wrong:

- `たべる` (all kana) comes back as `null` furigana, not `["たべる"]`, so it
  renders `""`. That is the right field value: the note templates fall back
  to `{{Reading}}` when `{{Furigana}}` is empty.
- `日本語` is segmented **per kanji** and read `にっぽんご`
  (`日[にっ] 本[ぽん] 語[ご]`), not as the `日本` + `語` compound with
  `にほんご`. A dictionary whose primary reading disagrees with the curator's
  is precisely what M2.6's reading check exists to warn about — so the
  fixture keeps jpdb's answer verbatim.

**Two live shapes the community fixture never showed**, both already handled:

- `pitch_accent` can carry more than one pattern (`食べ物` →
  `["LHLLL","LHHLL"]`). `models.py` already types it `list[str]`, first
  entry primary.
- `/parse` returns vocabulary entries for **particles** too (`を`, `と`), and
  `part_of_speech` arrives unordered with generic and specific tags together
  (`話す` → `["vt","v5","v5s"]`). The POS tables map these correctly; M2.5
  and M2.6 must not assume `part_of_speech[0]` is meaningful.

### [x] M2.2 Schema additions

Depends on: M1.1, M1.5 (validation.py contention — land M1.5 first)
Files: `src/japanese_anki/models.py`, `src/japanese_anki/validation.py`,
`tests/test_ledger.py` (replace the `SimpleNamespace` post-M2.2 stand-ins
with real records now the fields exist — a cleanup, not a break),
`tests/test_models.py` (new or extend),
`tests/test_status.py`, `tests/test_merge.py`,
`tests/test_migrate_inline.py`, `README.md`.
Design: DESIGN_V2 "Schema changes".

- `VocabularyRecord`: `pitch_accent: list[str]` (default `[]`),
  `audio_accent: str` (`""`), `frequency_rank: int | None` (`None`).
  `ExampleSentence`: `audio: str` (`""`). Extend `from_dict` coercion.
- If M1.1's merge enumerates fields, add the new ones there.
- Validation: `pitch_accent` entries match `^[HL]+$`, **case-insensitively**
  (amended in M5.1: `pitch._LEVELS` reads `h`/`l` deliberately, and this
  check is an *error*, so a case-sensitive one made `janki build` refuse a
  whole deck over a pattern the converter speaks correctly); warn (not
  error) when `len(pattern) != len(reading) + 1` — counting **NFC** kana,
  since a decomposed `が` is two codepoints and one kana (also M5.1) —
  because the particle-slot invariant is community-verified only.
- **Do not touch `FIELD_NAMES` or the exporter** — Anki-visible fields
  ship together in M5.4.
- **Three existing tests encode the pre-M2.2 world and must be rewritten
  in the same change** (verified by applying the schema change to a
  clean checkout: 3 failures, in files the old Files list did not name):
  - `tests/test_status.py::test_pitch_accent_counts_wait_for_the_schema`
    asserts `status.pitch_accent_supported() is False` and the literal
    line `Missing pitch accent: n/a until the pitch-accent schema lands
    (M2.2)`. It becomes a real count.
  - `tests/test_merge.py::test_import_rejects_bad_prefer_incoming_before_writing`
    parametrizes on `frequency_rank` as its canonical *unknown* field.
    `io.MERGEABLE_FIELDS` is derived from `dataclasses.fields(
    VocabularyRecord)`, so M2.2 makes that name valid and the rejection
    returns 0. Pick a different sentinel (`not_a_field`).
  - `tests/test_migrate_inline.py::test_the_records_move_into_the_normalized_file_under_the_same_ids`
    compares stored example dicts key-for-key against the inline note;
    `ExampleSentence.audio` adds an `audio: ""` key. Tolerate it.
- README: update the `janki status` sample block and the pitch-accent
  sentence in "Important limitations" — both state the pre-M2.2 answer.
- PR note: first `save_records_json` after this rewrites every stored
  record with the new keys (expected one-time diff).
- **A third README passage goes stale** beyond the two already named:
  the `--prefer-incoming` accepted-field list in the merge section
  enumerates `MERGEABLE_FIELDS`, which this change grows by three.
  It is derived from `dataclasses.fields`, so describe it as derived
  rather than re-enumerating it and going stale again.

### [x] M2.3 Romaji converter

Depends on: —
Files: new `src/japanese_anki/romaji.py`, new `tests/test_romaji.py`.
Design: DESIGN_V2 "Division of labor" (romaji row).

- `kana_to_romaji(kana) -> str`, Hepburn, macron-free (`ou`/`uu` long
  vowels — matches Shirabe conventions; document the choice): digraphs
  (きょ→kyo), っ gemination (がっこう→gakkou), ~~ん before b/p/m → m
  (しんぶん→shimbun)~~ *(2026-08-17: modern Hepburn — ん is always n:
  しんぶん→shinbun, こんばん→konban)*, ん before vowels/y → n' (きんえん→kin'en),
  katakana accepted (normalized to hiragana first), ー long-vowel marks.
- Pure, table-driven, no deps. Golden tests per rule + a mixed sentence.

### [x] M2.4 Conjugation tables

*Done 2026-08-07. Contract as written, plus four refusals the contract implies
but no bullet named. (1) `verb_group` values come from `jpdb.GODAN/ICHIDAN/
SURU/KURU` by import, not by restated literal, so the writer and the reader of
the field cannot drift; a small alias table also accepts `五段`/`u-verb`/
`group1` and friends for hand-typed decks. (2) `i-adjective` is accepted as a
`verb_group` and forwarded to `conjugate_i_adjective`, since a caller holding
one "how does this inflect" field will put it there. (3) Three classes return
`{}` beyond "unknown group": an expression that cannot end the way its group
must (`高い` as godan), an expression and reading that inflect differently
(`話す`/`はなした`), and the hand-written unsafe list (`ゆく`, `有る`, `在る`,
`である`, `得る` as godan — jpdb's `v5uru` routes exactly that word here and
its negative is `得ない`, never `得らない`). (4) A compound ending in a verb
with a hand-written table is refused rather than mechanically built:
`置いてある` is `置いてない`, not `置いてあらない`. Every entry in every
exception list has a test, and a coverage test fails if a list grows an entry
no case pins.*

Depends on: —
Files: new `src/japanese_anki/conjugation.py`,
new `tests/test_conjugation.py`.
Design: DESIGN_V2 "Division of labor". Key set matches the existing
`conjugations` dict (plain, negative, past, past_negative, te_form,
potential, passive).

- `conjugate(expression, reading, verb_group) -> dict[str, str]` for
  godan (all endings), ichidan, する/くる compounds, 行く's te-form
  irregularity. い-adjective forms as a second function if cheap.
- Unknown/irregular → `{}` (empty and flagged beats guessed).
- Golden tests: one verb per godan ending, ichidan, する, くる, 行く.

### [x] M2.5 `janki import-jpdb`

*Done 2026-08-07. Contract as written; six decisions inside it, several of which
M2.6 inherits. (1) A deck import runs `run_import` **once per deck** rather than
pooling every deck's words. That is the only shape that satisfies the
`source_ref` contract — a record's `source.imported_from` is its deck name — and
it is why a word in two decks earns a ledger reference for each. `--replace` is
therefore honoured on the first deck only; otherwise deck two would discard deck
one. (2) The three sources (FILE, `--deck`, `--all-decks`) are mutually
exclusive and exactly one is required: "import everything plus this file" has no
single meaning for `--replace`. (3) `conjugate` is fed `verb_group or
part_of_speech`, because jpdb has no verb class for an い-adjective (`adj-i` is
not a verb code) and the part of speech is what carries the inflection there.
(4) `meanings_chunks` keeps one line per *sense*, glosses joined — flattening
would spill eleven fragments onto a card that should read as three meanings.
(5) `deck_tag` slugs on unicode category, not an ASCII range, so a Japanese deck
name keeps its characters and only separators collapse; `:` collapses too, since
`::` is Anki's tag-hierarchy separator. The same slug names the deck's staging
file. (6) `frequency_rank` parses to `None`, never 0, on anything unparseable —
0 is a real rank and "never looked up" must stay distinguishable. As planned,
`occurences` counts are not stored. README is untouched: jpdb setup is M2.W's.
**One contract item was not met:** "check the userscript's actual output header
before hardcoding" did not happen — the JPDB-Export userscript was unreachable
from the environment this ran in. Both `reading` and `furigana reading` map, the
guess is flagged as such in `csv_base.py` and in the test that pins it, and a
header that turns out to be neither fails loudly (the column goes unmapped and
the row is held for reading review rather than imported wrong). Verifying it
against a real export belongs to M2.W or a follow-up, and is owner-only for the
same reason M2.1F is.*

Depends on: M2.1, M2.2, M2.3, M2.4, M1.5, M1.8
Files: new `src/japanese_anki/importers/jpdb_import.py`,
`src/japanese_anki/importers/shirabe.py` (refactor),
new `src/japanese_anki/importers/csv_base.py`,
`src/japanese_anki/cli.py`, `tests/test_jpdb_import.py` (new),
`tests/test_shirabe_import.py` (must stay green).
Design: DESIGN_V2 "jpdb.io > Deck sync" + "Manual export files".

- Factor generic CSV machinery (delimiter sniffing, alias mapping,
  row→record) into `csv_base.py`; `shirabe.py` keeps its aliases and
  error type as a thin specialization; **existing shirabe tests pass
  unchanged**. While in there: `csv_base` fills `romaji` via M2.3
  whenever reading is present and romaji empty — so Shirabe imports
  get romaji too, per the design's "on import" commitment.
- `FIELD_ALIASES` gains `spelling` and the JPDB-Export userscript's
  reading header (check the userscript's actual output header before
  hardcoding; add both `reading` and `furigana reading` normalizations).
- API path: decks → pairs → batched lookup; mapping per design using
  M2.1's POS tables, M2.3 romaji, M2.4 conjugations,
  `pitch_accent`/`frequency_rank` from lookup. `source.type="jpdb"`,
  raw_fields keep vid/sid/deck name/card_state (stringified;
  `occurences` counts deliberately deferred — not stored). Auto-tags:
  `jpdb`, `jpdb:<deck-name-slug>`.
- Merge + ledger + outcome printing via M1.8's shared helper
  (`cli.run_import`). **Its `source_ref` contract binds here:** pass the
  same value the importer stored as each record's own
  `source.imported_from`, and pass no extra detail — a source
  reference's identity is every key but `seen_at`, so any other shape
  has `janki status --rebuild` append a near-duplicate reference to
  every record this importer touched. Use `held_unit=` so the held-rows
  line does not say "row(s)" about API results.
- **Both** M1.5 hold classes route through the staging rule, not just
  the obvious one: entries whose reading is empty (`hold_reason:
  "missing reading"`) *and* entries whose reading itself contains kanji
  (`contains_kanji(reading)`, `hold_reason: "reading contains kanji"`).
  The second mints `word:<kanji>:<kanji>` — well-formed-looking and
  permanently invalid — so it cannot be left to the empty-reading check.

### [x] M2.6 `janki enrich --jpdb`

*Done 2026-08-07. Contract as written; one deviation and six decisions.
**Deviation:** the two remaining jpdb wire-value normalizers (`pitch_accent`,
`frequency_rank`) moved out of M2.5's importer into `jpdb.py` beside the POS
tables, so this task touched `jpdb.py` and `importers/jpdb_import.py` beyond its
Files list. Enrichment reads the same fields off the same endpoints as the
import does, and a second definition of what `frequency_rank: 0` means is the
third POS table this plan already refused once. (1) `--staging FILE` is a
distinct *target*, not an add-on to a normal pass: it takes neither
`--force-fields` nor record ids, and writes no records and no ledger entries —
a held row is not a record yet, and the ledger describes records. (2) A parse
that resolves to more than one dictionary entry is a **warning and no write**,
not a first-token guess: an entry whose spelling *is* the expression wins, and
failing that a single resolving token, but 食べ物屋 splitting into 食べ物 + 屋
has no entry whose pitch accent describes the record. (3) `meanings` is
deliberately not enrichable. jpdb's glosses are a dictionary's; a record that
reached janki from a textbook carries what that textbook taught, and filling
that hole is M4.2's call with a pass that can read the record's examples.
(4) An empty proposal never blanks a field — including under `--force-fields`,
where the field being non-empty is exactly the case, so "jpdb had nothing" must
not read as "blank it". (5) The conjugation table is written even when the
record's reading is empty: it inflects the *expression*, and the reading only
ever guards against the two disagreeing. Romaji is not, having nothing to
transliterate. (6) The reading set for the mismatch check is gathered across
the entry's `alt_sids`, because a homograph's other reading lives on its other
sense, not on the one `/parse` happened to pick. (7) A suggestion, and any
dictionary fact, comes from the token's **furigana** rather than the entry's
`reading` wherever the two can differ: jpdb resolves an inflected surface form
to its lemma, so 行った answers with 行く's entry, and `suggested_reading: いく`
is a reading a reviewer could type into a permanent `word:行った:いく`. A record
with no reading at all cannot prove the entry is even the same word, so it is
warned and skipped rather than filled from the lemma. README is untouched: the
enrich walkthrough is M2.W's, with the rest of the jpdb setup.

**Two later additions, both from review** (see the commits after this task):
`--staging` writes through a new `staging.rewrite_staging`, which edits the
document instead of re-rendering it from records — a staging file under review
is the one place in this repository holding work that exists nowhere else, and
`write_staging`'s load-then-dump round trip silently deleted a reviewer's YAML
comments and any key outside the record schema. That is what `ruamel.yaml` was
added for, scoped to that one function; everything janki writes from scratch
still goes through PyYAML. And `cli._slug_for_file` now appends a fingerprint of
the deck name, because `Lesson 1`, `lesson-1` and `Lesson: 1` all flatten to one
slug and so shared one needs-reading file — the second deck's rows were told to
resolve a file the first deck reclaims on every re-run, advice that never
converges. The tag stays collision-prone on purpose: it is what a human types
into a deck filter.*

Depends on: M2.1, M2.2, M2.3, M2.4, M1.3, M2.5 (shared import/POS
plumbing settled first)
Files: new `src/japanese_anki/enrich.py`, `src/japanese_anki/cli.py`,
`tests/test_enrich_jpdb.py` (new).
Design: DESIGN_V2 "jpdb.io > Dictionary enrichment" + enrich rules.

- CLI shape decided now: bare `janki enrich` errors with usage; `--ai`
  errors "not implemented until M4.2"; `--force-fields FIELD[,FIELD]`
  is implemented **here** (shared flag logic reused by M4.2): named
  fields are overwritten, all others keep fill-empty. `EnrichError`
  defined here.
- Per target record (all, or `IDS...`; records with nothing fillable
  are skipped before any API call):
  1. `/parse` the expression **without** forced furigana → token
     reading + vid.
  2. Stored reading empty (post-M1.5 **no imported record has an empty
     reading** — importers default a kana-only row's reading to its
     expression before the ID is minted, and divert kanji-without-
     reading rows to staging — so this branch is reached only by a
     hand-written record or a hand-written inline deck note) or equal
     to token reading → use this response.
  3. Otherwise fetch the reading set for the vid via
     `lookup_vocabulary` (+ `alt_sids`/readings). Stored reading not in
     the set → **warning listing both, no write**. In the set → re-run
     `/parse` **with** forced furigana (M2.1 span helper) and use that
     response. (This is the mechanism behind "mismatches are warned,
     never auto-fixed" — without the unforced first pass there is
     nothing to compare.)
- Fill **empty** fields only: `furigana` (via M2.1's
  `furigana_to_anki`), `pitch_accent`, `frequency_rank`,
  `part_of_speech`/`verb_group` (M2.1 tables), `romaji` (M2.3),
  `conjugations` (M2.4). **Never write `reading`** — kana-only reading
  defaults happen at record creation (importers), not here.
- Staging assist: with `--staging FILE`, annotate **every held row** —
  both M1.5 hold classes, `hold_reason` `"missing reading"` *and*
  `"reading contains kanji"` (select on the presence of `hold_reason`,
  or on `not reading or contains_kanji(reading)`; a needs-reading file
  is not all reading-less post-M1.5) — in
  a needs-reading staging file with `suggested_reading` (from the
  unforced parse) for the human to confirm — delivers the design's
  "/parse proposes a reading" step. Write it back with
  `write_staging(..., force=True)`: the no-overwrite guard exists for
  the *import* path, and the file being annotated always exists (it is
  the one the import wrote), so without `force` this raises
  `StagingError: Staging file already exists` every time. Do the
  annotate-and-write only after the whole API pass has succeeded, so a
  mid-pass failure leaves the reviewer's file untouched.
- Show the shared field-diff before save; `--yes` skips. Ledger:
  `record_enriched(kind="jpdb", model="jpdb", fields=...)`.
- Tests: fill-empty, `--force-fields` override, reading-mismatch
  warning path (fake transport scripted for steps 1–3), homograph
  disambiguation (forced-furigana request asserted), no-write on
  populated fields, staging `suggested_reading` annotation.

### [x] M2.7 `janki import-jpdb-reviews` + deck filter docs

*Done 2026-08-07. Contract as written; one deviation and five decisions.
**Deviation:** the parsing and matching live in a new
`importers/jpdb_reviews.py`, not in `cli.py` as the Files list says. That entry's
stated reason — "do not fold into `jpdb_import.py` — M2.5 may be in flight" — was
about merge contention, and M2.5 has landed; meanwhile AGENTS.md requires parsing
to be separate from the CLI. A third module satisfies both, and `cli.py` keeps
only the command. DESIGN_V2 needed no amendment: its jpdb section already says
raw_fields rather than the ledger. (1) Every `cards_vocabulary_*` list is read,
not just `jp_en`, with counts **summed per word** — `jp_en` and `en_jp` are the
same word drilled in two directions, so reporting them separately would double
every count. A prefix match takes any vocabulary list jpdb adds later without a
code change; a non-vocabulary section (`cards_kanji_*`) is reported as skipped
rather than silently ignored. (2) Presence in the export earns the tag, whatever
the count — the filter means "words I already study in jpdb", and jpdb having
made a card is what answers that. The count is stored beside it so a finer rule
is possible later without re-importing. (3) Matching is vid first, then
expression + reading; **first match wins** where two records share either, because
two records with one vid is a duplicate `janki status --duplicates` exists to
report and tagging both would spread it. (4) A re-run whose counts have not moved
rewrites nothing — `vocabulary.json` is left byte-identical and the ledger
sighting is already idempotent, so the weekly loop is cheap and produces no git
noise. (5) Only the review count is stored, not a last-reviewed date: the plan
asked for counts, and a date is a second thing to keep in step for a filter
nothing yet reads. README documents all four deck filters and the
`exclude_tags: [jpdb-known]` recipe; both were run end to end before committing,
including building a deck that excluded a tagged word.*

Depends on: M2.1, M1.3
Files: `src/japanese_anki/cli.py` (do **not** fold into
`jpdb_import.py` — M2.5 may be in flight), `tests/test_jpdb_reviews.py`
(new), `README.md`.
Design: DESIGN_V2 "Manual export files" (reviews.json).

- Parse the reviews export (top-level `cards_vocabulary_jp_en` etc.;
  entries `{vid, spelling, reading, reviews:[...]}`). Tag matching
  records `jpdb-known`: by vid in `raw_fields` first, else
  expression+reading. Print unmatched entries (count + first few).
- **Review counts go into the record's `source.raw_fields`, not the
  ledger** (supersedes design — DESIGN_V2's jpdb section is amended to
  match in the same change).** `ledger.record_source_seen` identifies a reference by every
  key but `seen_at`, so a changing count is a *new* reference every
  time: a weekly run over 2,000 words would add 2,000 near-duplicate
  lines a week to a git-tracked file and make `status`'s provenance
  view unreadable. The ledger gets exactly one detail-free
  `record_source_seen(id, "jpdb-reviews", <export filename>)` per
  matched record, which is idempotent across every future run. If a
  count ever must live in the ledger, changing `record_source_seen`'s
  identity rule is a prerequisite task, not a call-site decision.
- README: document deck YAML filtering (`include_ids`, `include_tags`,
  `exclude_ids`, `exclude_tags` — already implemented in
  `resolve_deck_records`, currently undocumented) with the
  `exclude_tags: [jpdb-known]` recipe.

### [x] M2.W Milestone 2 wrap

*Done 2026-08-07. README only, as scoped. A "Working with jpdb" section covering
setup (`JPDB_API_KEY` and `jpdb ping`), deck and CSV imports, and the
`enrich --jpdb` walkthrough including the `--staging` reading assist. The
deck-filter and `import-jpdb-reviews` documentation stayed M2.7's and is
cross-linked rather than restated, per this task's note. Two things the writing
turned up and fixed rather than documenting as-is: a CSV import gets the plain
`jpdb` tag only (no deck to name), which the first draft blurred, and the deck
tag is the *flattened* name, which is worth an example since a reader will type
it into `include_tags`. **Command verification is partial and deliberately so:**
every command that does not need the network was run — `jpdb ping` from `/` with
no project and no key, `import-jpdb export.csv` end to end, and every documented
error path — but the API-backed flows need a live `JPDB_API_KEY`, which is
owner-only for the same reason M2.1F is. Those are covered by tests against fake
transports, not by a live run, and this note is where that gap is recorded.

**Gap partly closed 2026-08-08.** `jpdb ping`, `/parse`, `enrich --jpdb`
and `list_user_decks` were run against the live API on the real
collection. Still fake-transport only, and worth naming so this note
keeps doing its job: **`import-jpdb --deck` end to end**
(`deck/list-vocabulary` → `lookup-vocabulary`) and **`enrich --staging`**.
The first is the shape janki guesses at hardest: it fetches a
`[vid, sid]` row list and then batches `lookup-vocabulary` over it,
matching answers to requests **positionally** — so a batch that comes
back short raises and aborts the whole deck import rather than warning
(`jpdb.py`'s "the results are positional"). All against a hand-written
fake, which makes it the one to run before any deck is built from it.
(Not the `occurences` reconciliation: `import_deck` leaves
`fetch_occurences` at its `False` default and no caller in `src/` sets
it, so that branch is unreachable from this flow.) `enrich --jpdb`
filled `pitch_accent` and `frequency_rank` for all three records
(行く `LHH` rank 100, 話す `LHLL` rank 200, 食べる `LHLL` rank 200) and
recorded the pass in the ledger. The owner's decks were listed live: 584
words in Genki Vol 1, 641 across all decks — the number that decides
whether batch mode earns its keep.*

Depends on: all M2 tasks except M2.1F (owner-only; see the lane map)
Files: `README.md`.

- README: jpdb setup (API key location, `JPDB_API_KEY`), sync + enrich
  walkthrough. (Deck-filter docs are M2.7's; don't duplicate.)

---

## Milestone 3 — PDFs and photos

Lane map: {M3.1, M3.2 in parallel} → {M3.3} → {M3.4}.

### [x] M3.1 AI plumbing (single owner of the Anthropic client)

*Done 2026-08-07. Contract as written; six decisions. (1) `parse_call` returns
the 2-tuple the contract specifies and **not** `stop_details`, so M3.3's refusal
error cannot name the refusal category without widening the signature — worth
knowing before M3.3 writes that message, and a deliberate choice to keep the
contract as pinned rather than guess at what M4.2 will want too. (2) No new
error class: the Conventions list every error home and `claude_client.py` is not
one, so it raises `JankiError` directly, exactly as this task's text says. (3)
`build_client` does **not** pre-check the API key. The SDK resolves credentials
from more sources than `ANTHROPIC_API_KEY` alone, and a friendlier "key not set"
error here would be a second copy of that resolution order, wrong the first time
it gains a source. (4) A missing style guide is an error, not an empty block:
every AI pass is meant to write to this project's conventions, and dropping them
silently produces plausible output that ignores the rules the repository exists
to enforce. (5) `system_blocks` puts the cache breakpoint on the **last** block,
because caching is a prefix match — the style guide leads so the prefix is long
enough to clear the API's per-model minimum, which is silent when missed. (6)
`DEFAULT_MAX_TOKENS = 16000` is the non-streaming ceiling, and on current models
it budgets **thinking plus response** — a limit sized snugly around the expected
answer truncates mid-way, which is precisely the `max_tokens` stop reason this
module makes callers look at. **The module builds the request itself rather than
calling the SDK's `messages.parse()` helper** — a correction made after review,
and the reason the contract is satisfiable at all: `parse()` validates every
text block against the schema with no stop-reason check anywhere in its path,
so a truncated answer raises a pydantic `ValidationError` from inside the SDK.
That fails twice — it is not a `JankiError`, so the CLI cannot format it, and
the stop reason is unrecoverable from the exception, so a caller cannot tell
"declined" from "ran out of room". `parse_call` therefore sends
`output_config.format` (schema transformed by the SDK's own
`transform_schema`, so the wire shape stays the SDK's business), reads
`stop_reason` first, and validates only on `end_turn`. **The extras floor is
`anthropic>=0.121`, not a bare `anthropic`:** the bindings it actually needs —
the top-level `transform_schema` export and
`messages.create(output_config={"format": ...})` — were verified against that
installed version rather than taken from documentation, and an unverified floor
would fail at the first AI call instead of at install time. **`dev` pulls in
`ai`,** also from review: the runtime promise is that non-AI commands work
without the extra, and the lazy imports keep that true and are tested by forcing
the failure; the *suite* is a different thing and has to exercise the AI
plumbing rather than skip it. Without that, `scripts/bootstrap.sh` — which
installs `.[dev]` and then runs pytest under `set -e` — dies collecting
`tests/test_claude_client.py` on every fresh clone, and skipping instead would
report green for code nobody ran.*

Depends on: M1.2, M1.6
Files: new `src/japanese_anki/claude_client.py`, `pyproject.toml`,
`tests/test_claude_client.py` (new).

- `pyproject.toml`: optional extras group `ai = ["anthropic"]`.
- `claude_client.py`: one factory used by every AI feature (M3.3, M4.2,
  M4.3, M4.4) — lazy `anthropic` import with a friendly `JankiError`
  naming `pip install -e '.[ai]'` when missing; builds the client;
  exposes `parse_call(model, system_blocks, user_content, schema,
  client=None)` returning `(parsed, stop_reason)` so callers share the
  stop-reason discipline; system blocks carry the style guide with a
  `cache_control: {"type": "ephemeral"}` breakpoint (callers may pass
  `ttl: "1h"` for batch use).
- Tests: missing-dependency error path; fake client wiring.

### [x] M3.2 Input plumbing (formats, HEIC, inbox copy)

*Done 2026-08-07. Contract as written; seven decisions. (1) For a HEIC,
`origin_path` is the **original**, not the JPEG that was actually sent: the
camera's file is the evidence and the JPEG is a rendering of it. The conversion
goes to a temporary directory and is thrown away, so `data/inbox/` keeps what
the camera produced rather than a derived file sitting beside it looking like a
second source. (2) "Already under `data/inbox/`" uses the explicit durable
inbox root that the CLI supplies. A file anywhere in that root stays where it
is. A later M7.6 camera pilot exposed that limiting the test to `scan_inbox`
copied a root-level inbox file again and changed its durable locator. (3) A name
already taken by *different* content earns a fingerprint suffix rather than an
overwrite: every phone writes `IMG_0001`, and overwriting one with the other
destroys the evidence behind every record extracted from it. Identical bytes
reuse the existing copy. Two different files already inside the durable inbox
cannot receive a suffix without renaming an immutable source. They are refused
when their basenames collide because staging and pattern review key on that
basename. The comparison ignores letter case so it is safe on the supported
macOS filesystems. (4) Duplicates in one call are kept. Passing the same
photo twice costs tokens, but dropping the second is the silent discard this
project refuses everywhere else. (5) An unreadable path or unsupported suffix
stops the whole batch — extracting a subset would leave the user to notice it
came up short. (6) `content_block()` lives on `PreparedInput`, slightly beyond
the tuple the contract names: `claude_client` is the single owner of *the
client*, and giving it a media vocabulary would make two places to change when
a format is added. (7) `InputError` exists (the contract says "raise
`JankiError`", which it is) because M3.3 has to tell an unreadable input from a
refused extraction, and because every other module with its own failure domain
— staging, csv_base, the jpdb importers — carries one. Note this differs from
M3.1, whose contract named `JankiError` directly and which has one failure
domain; the divergence is deliberate rather than drift.*

Depends on: M1.6
Files: new `src/japanese_anki/inputs.py`, `tests/test_inputs.py` (new).

- `prepare_inputs(paths, scan_inbox) -> list[PreparedInput]` where each
  is `(content_block_kind: "document"|"image", media_type, data_b64,
  origin_path)`: `.pdf` → document; `.jpg`/`.png` → image; `.heic` →
  convert via `sips -s format jpeg` (darwin) else raise `JankiError`
  naming `pillow-heif` (accepted macOS-only scope). Files not already
  under `data/inbox/` are copied into `scan_inbox` first; the copy is
  the recorded provenance path.
- Pure besides `sips`; tests fake the subprocess.

### [x] M3.3 `janki extract`

*Done 2026-08-07. Contract as written except one item, below; six decisions.
**Deviation — `max_tokens` does not retry per-page.** The contract offers
"retry per-page (PDFs) or fail with guidance"; per-page retry means splitting a
PDF, which needs a PDF library this project does not depend on and this task was
not authorised to add. So a truncated answer fails with guidance naming the
concrete remedy (extract fewer pages, split the document, raise the budget), and
never writes a partial file. Adding `pypdf` and doing the split is a follow-up
worth its own decision, not a call to make inside this task. **Also required
widening M3.1's `parse_call`,** which M3.1's own note flagged for exactly this
moment: the contract says "fail with the category in the message", and a
2-tuple could not carry it. It now returns a `CallResult` NamedTuple —
`(parsed, stop_reason, refusal)` — with the SDK's refusal object translated into
janki's own `Refusal`, so `claude_client` stays the only module that knows what
an Anthropic response looks like. (1) The known-word list rides in the **user
turn**, not the system blocks: it changes every time the collection grows, and
anything above the cache breakpoint that changes invalidates the cached style
guide for every run. (2) Only prose mode is told what janki already has — a
table is transcribed row by row, and telling the model to skip rows would put
holes in a faithful transcription. (3) Already-known candidates are kept, marked
and sorted last rather than dropped: a silent discard is a silent discard even
for a duplicate, and the reviewer may still want this page's example sentence.
(4) `known_ids` matches on the stored id **and** the id expression+reading would
mint today, since a hand-written record may carry one that has drifted. (5)
Files are processed one at a time and written as they succeed, so a later
failure keeps the earlier files — the work already paid for is kept and the
error names what is left. (6) `candidate_schema()` is cached: a fresh class per
call would make an instance built by one call fail validation in another, and
would present an identical schema to the API as new on every request. (7) The
staging file is named for the whole source **name**, suffix included
(`worksheet.pdf.yaml`) — which is what DESIGN_V2 said, and what stops a scan and
a photo of one page from colliding. Every target is resolved before the first
API call, so a batch that would write two inputs to one file is refused rather
than paid for and then half-discarded. Both from review.*

Depends on: M3.1, M3.2, M1.5
Files: new `src/japanese_anki/extract.py`, `src/japanese_anki/cli.py`,
`tests/test_extract.py` (new).
Design: DESIGN_V2 "PDFs and photos > Step 1".

- `janki extract FILE... [--mode table|prose] [--model ID] [--force]` —
  `--model` overrides `config.extract_model` per invocation. One
  staging file per input, via M1.5's `write_staging` (which enforces
  the no-overwrite rule; `--force` passes through). `ExtractError`
  defined here.
- Claude call via M3.1 with a Pydantic `CandidateRecord` schema:
  expression, reading, meanings, part_of_speech, example (optional),
  page (int), context (str), confidence (`high|medium|low`),
  inclusion_reason (prose mode).
- **`stop_reason` discipline** (via M3.1's return): `refusal` → fail
  with the category in the message; `max_tokens` → retry per-page
  (PDFs) or fail with guidance; never accept truncated output.
- Prose mode includes the known-expression list in the prompt so the
  model skips them. Candidates matching an existing record (ID, or
  expression+reading) are annotated `already_known` and sorted last.
- Tests: fake parse_call; staging shape; already-known annotation;
  refusal and max_tokens paths; --force behavior.

### [x] M3.4 `janki promote`

*Amended 2026-08-08, from an M4.W review. `remint` documented a
precondition it never checked — "these records have never been in
`vocabulary.json` or Anki" — and M4.2's staging route made breaking it
routine. A record whose id was minted from a wrong reading keeps that id
when the reading is corrected, because the id is uncorrectable by design;
`enrich --ai` then stages it, and promote re-minted it into an id the
merge had never seen, **added a second record**, and left the curated
original with its Anki history and without the change. The message even
said "these records were never in Anki" about a record that had been
exported. `remint` and `check_readings` now take the ids the collection
already holds and leave those alone, and `command_promote` reads the
collection before any id is decided rather than after. The one sanctioned
ID change — a held row the collection has never seen — is unaffected, and
both sides are pinned. "The collection" is `status.surviving_ids` — the
normalized file **plus every deck's inline notes** — because a record
living only in a deck YAML has the same stale id and the same exported
GUID; and it asks each deck what it
**declares**, not what it builds: include/exclude filters answer "does
this deck build it", which is a different question from "does this id
exist", and a note a filter drops still holds hand-written content and a
GUID Anki may already have. (`exporters.anki.deck_declared_ids` is new
for that, and `--replace`'s ledger prune gets the same correction for
free.) A deck that will not resolve means no id can be proved absent, so
rows whose id **would change** are held back rather than promoted —
writing the id they arrived with would put it in the store permanently,
since a stored id is exempt from the re-mint that repairs it, and a held
row waits in `data/staging/`, which is committed.*

*Done 2026-08-07. Contract as written; two deviations from the Files list and
seven decisions. **Deviation 1:** the reading-set lookup lives in `enrich.py` as
a new public `dictionary_readings`, not reimplemented here — walking `alt_sids`
to find a homograph's other reading is exactly what goes subtly wrong in a
second copy, and M2.6 and this task ask the identical question. **Deviation 2:**
`staging.prune_staging` is new, so removing promoted rows keeps the surviving
rows' own keys and inline notes plus the file's header comments, instead of
re-rendering the review from records. One loss is documented rather than fought:
a comment on its own line *between* rows is attached by YAML to the row above,
so it goes when that row is promoted. Reaching into the parser's internals for a
case it does not model is the worse trade, and the residual loss is strictly
smaller than a full re-render. (1) A spelling jpdb cannot resolve to one entry
is promoted **unchecked with a warning**, not held: silence is not disagreement,
and holding those back would punish exactly the uncommon words a textbook is
most worth extracting from. (2) `--skip-reading-check` drops only the dictionary
check. The kana rule is never skippable — it is about whether an ID can exist at
all, not about whether a dictionary agrees. (3) The structural holds are
re-tested from the reading itself rather than trusted from `hold_reason`: a
staging file is hand-edited, and the question at this gate is what the reading
*is now*. (4) The ledger's source type and ref are read off each record instead
of fixed to `"pdf"`, which is what makes an enrichment-shaped file work and what
keeps `status --rebuild` from appending a near-duplicate reference to everything
promoted. (5) The `done/` archive is appended to, not replaced, so promoting a
file in two passes does not lose the first pass's rows. (6) The staging file is
pruned only of rows that actually landed, so a promote that fails part-way
leaves the review intact and re-running is safe. (7) An emptied review is
deleted: leaving it would have the next import report a file that can never be
resolved.*

Depends on: M3.3, M2.1, M1.1, M1.3
Files: new `src/japanese_anki/promote.py`, `src/japanese_anki/cli.py`,
`tests/test_promote.py` (new).
Design: DESIGN_V2 "PDFs and photos > Step 2".

- Reading cross-check per candidate against the **set** of dictionary
  readings for its spelling (unforced `/parse` → vid →
  `lookup_vocabulary` readings incl. alt_sids), three outcomes: pass /
  warn-but-pass (valid non-primary reading) / hold back (in no entry).
  `--skip-reading-check` bypasses (offline).
- Candidates whose reading is empty **or itself contains kanji** are
  always held back — unless a human filled `reading` with kana (or
  confirmed `suggested_reading` by copying it into `reading`). Both are
  M1.5 hold classes (`hold_reason` `"missing reading"` and `"reading
  contains kanji"`) and both reach here. **Malformed-ID re-mint (the
  one sanctioned ID change):** a promoted record gets its ID re-minted
  from expression+reading at promote time whenever its stored id is
  malformed — `word:<expr>:` *and* `word:<kanji>:<kanji>`. Key the test
  off the **id itself**: re-mint whenever
  `record.id != stable_record_id(record.expression, record.reading)`.
  That subsumes both shapes with no per-shape enumeration, and it is the
  only key that works — by promote time a human has replaced the kanji
  reading with kana, so `contains_kanji(reading)` is `False` in exactly
  the case the re-mint exists for. Promoting such a record unchanged
  would write into `vocabulary.json` precisely the id M1.5 exists to
  prevent, which `validation.py` then errors on forever. These records
  never entered `vocabulary.json` or Anki, so no history exists to
  orphan.
- Promotion: survivors merge into `vocabulary.json` (M1.1 semantics),
  ledger `record_added`/`record_source_seen` (`source.type="pdf"` — or
  the staging file's recorded source type: promote must also accept
  **enrichment-shaped staging files** from M4.2, whose records already
  exist and merge as updates, not adds). Promoted rows append to
  `staging/done/<file>`; the staging file is rewritten in place with
  only held-back rows (`hold_reason` set); deleted when empty.
- Tests: three-outcome routing (fake jpdb), held-back rewrite, ID
  re-mint, enrichment-update staging file, empty-file cleanup,
  idempotent re-promote of a half-done file.

### [x] M3.W Milestone 3 wrap

*Done 2026-08-07. README as scoped, plus two stale forward-references the
writing exposed and this task had to fix rather than document around: the
staging-file `review_notes` written by every import still told the reviewer that
`janki promote` "ships in Milestone 3" and gave them the three manual steps to
run instead — instructions now competing with a command that exists — and the
README said the same. Both now name `promote`, and the test that pinned the old
wording pins the new rule instead: every instruction in those notes has to be one
the reviewer can run today. One thing the walkthrough documents that is easy to
get wrong: deleting a held row's `id:` line is still worth doing even though
`promote` re-mints a malformed ID anyway, because `validate` refuses that ID
whatever the reading now says — so the deletion is what keeps `validate` usable
as the green light before promoting, and the re-mint is the safety net for
forgetting. **Command verification is partial, as in M2.W:** the review→promote
half was run end to end following the notes' own instructions (import, fill the
reading, drop the id, validate, promote), but `extract` needs a live
`ANTHROPIC_API_KEY` and is covered by tests against fakes rather than a live
run.*

Depends on: all M3 tasks
Files: `README.md`.

- README: photo-of-handout → deck walkthrough (extract → review →
  promote), `.[ai]` extra install.

---

## Milestone 4 — AI enrichment

Lane map: {M4.1 (early — only needs M2)} → {M4.2} → {M4.3} → {M4.4}.
M4.2/M4.3/M4.4 all touch `enrich.py` + `cli.py`: strictly serial.

### [x] M4.1 Example QC functions

*Done 2026-08-07. Contract as written; four decisions. (1) The furigana verdict
is on the **(kanji run, reading) pairs in order**, not on the rendered string:
jpdb does not tokenize punctuation, so comparing renderings would fail every
sentence that ends in a full stop. (2) A **missing space still fails**, and
should — I first wrote the opposite and the test caught it. `お茶[ちゃ]` puts
ちゃ over both characters instead of over 茶, so Anki renders the wrong ruby
*and* `furigana_reading` yields `ちゃ` with the お gone, which is the reading
sentence audio would speak. The space is notation, but it is notation that
decides which characters a reading belongs to. (3) `example_contains_target`
searches the plain sentence, never the furigana, which carries bracketed
readings that would match text no reader sees. A word janki has no verb class
for contributes only its dictionary form — the right answer rather than a guess.
(4) `settle_example_romaji` returns a new example rather than mutating, and
inherits `kana_to_romaji`'s refusal: kanji with no furigana yields `""`, never a
part-transliterated string. One test runs against the committed live `/parse`
capture rather than a hand-written fixture, because jpdb segments 日本語 per
kanji and reads it にっぽんご — an expectation written by hand would have quietly
"corrected" both.*

Depends on: M2.1, M2.3, M2.4
Files: new `src/japanese_anki/qc.py`, `tests/test_qc.py` (new).
Design: DESIGN_V2 "AI integration > Mechanical QC".

- Pure functions, no CLI:
  `example_contains_target(example, expression, verb_group)` (uses
  M2.4's conjugated forms), `verify_example_furigana(example,
  parse_response)` (compare against jpdb `/parse` of the sentence via
  M2.1's contract; returns verified | mismatch), and
  `settle_example_romaji(example)` (M2.3 from verified furigana; *checked*
  rather than rebuilt since 2026-08-17 — a supplied romaji that transliterates
  the reading is kept for its word spacing, which janki cannot derive).
- Unit tests with canned parse fixtures; no network.

### [x] M4.2 `janki enrich --ai`

*Done 2026-08-07. Contract as written; one deviation and six decisions.
**Deviation — `conjugation.polite_stem` is new,** outside this task's file list,
and it had to be: the QC target-word check compares against
`CONJUGATION_FORMS`, which has no polite form, so every 〜ます sentence was
rejected — and the style guide asks for beginner examples, which a beginner
textbook teaches in polite form first. The feature would have discarded almost
every good sentence. The stem lives in `conjugation.py` because "which kana does
this ending become" is that module's question wherever it is asked, and it is
deliberately **not** added to `CONJUGATION_FORMS`: being able to *match* a
sentence is a different decision from changing every stored record's
conjugation table. `qc.target_forms` spells out the polite forms rather than
matching the bare stem, which would let 食べ物 count as an example of 食べる.
(1) `--jpdb` and `--ai` are mutually exclusive per run — each shows its own
diff, and merging two unrelated sets of proposals into one y/n is not review.
(2) `--force-fields` is now pass-aware: naming a `--jpdb` field while running
`--ai` is a typo worth catching, and the error says which pass owns it.
(3) An example whose furigana jpdb did not confirm is **kept and flagged**, not
dropped — the sentence may be right where the segmentation is wrong, and a human
deciding that beats janki discarding good Japanese. (4) A missing parse flags the
example the same way a mismatch does, because "nobody checked" is exactly what
unverified means. (5) The flag appends to any fingerprints already there rather
than replacing them, since a record can accumulate flagged examples across runs.
(6) One call per record, so a refusal or a truncation costs that record and not
the run; a truncated answer is never accepted, on the extractor's reasoning — a
half-written example is a sentence that stops mid-word, and accepting one puts
it on a card.*

Depends on: M3.1, M4.1, M2.6, M3.4 (the ≥50-record path needs promote)
Files: `src/japanese_anki/enrich.py`, `src/japanese_anki/cli.py`,
`tests/test_enrich_ai.py` (new).
Design: DESIGN_V2 "AI integration".

- Targets when M4.2 landed: records with empty examples or `usage_notes`
  (content-defined, per M1.3; explicit `IDS...` overrides). M7.6P supersedes
  that target rule: current default targeting is empty meanings or examples,
  while an empty optional usage note is complete. Per record, one structured
  call via M3.1 (`enrich_model`; stop_reason discipline): system =
  style guide (cached), user = record + jpdb facts + up to 3 recently
  generated examples from other records (variety pressure). Schema:
  examples (japanese, furigana, romaji, english), usage_notes.
- Apply through M4.1's QC: examples failing the target-word check are
  rejected; furigana mismatches keep the example but write the
  unverified-furigana convention key (see Conventions); romaji always
  regenerated. Fill-empty (+ `--force-fields` from M2.6's shared
  logic); validation; shared field-diff before save (`--yes`).
- ≥ 50 target records → write results to a staging file and route
  through `janki promote` instead of a monolithic diff.
- Ledger `record_enriched(kind="ai", model=<model>, fields=...)`.

### [x] M4.3 `--polish-meanings`

*Done 2026-08-08. Contract as written; six decisions and one file added
to the list. **Added file:** `tests/test_enrich_polish.py` rather than
extending `tests/test_enrich_ai.py` — the two passes share a module and
almost nothing else, and this one's subject is what it refuses to do to
a field that is already full. (1) It is a third **pass**, exclusive with
`--jpdb` and `--ai`, not a modifier of `--ai`: it rewrites curated
English, which is a different promise from filling a hole, and one y/n
covering both would hide it. (2) `enrich.polish_meanings` is a
**generator**. The confirmation is per record, and the confirmation is
what decides whether the next call is worth making — declining the first
proposal and walking away costs one call, not one per record in the
collection. (3) `q` quits the loop and still writes what was already
accepted: it was accepted. (4) An answer that reduces to nothing is
refused rather than written — a card with a Japanese side and no English
one is worse than a clumsy gloss — and an empty list is the documented
way to say "already right", not an error. (5) The prompt carries the
record's **examples**, because 「先生に聞く」 and 「音楽を聞く」 are the
same verb with two glosses, and the sentences it was collected with are
the only evidence janki has for which sense it means. (6) The ledger kind
is `polish`, separate from `ai` although the same model does it: "this
record's glosses were replaced by a model" is a different fact about a
record than "its examples were written by one". The pass needs no jpdb
key, so `command_enrich` no longer builds a client before dispatching to
it. No gloss cap: the per-record confirm is the cap, and that is what it
is for.*

Depends on: M4.2
Files: `src/japanese_anki/enrich.py`, `src/japanese_anki/cli.py`,
`tests/test_enrich_ai.py`.

- Separate explicit operation (meanings are never empty, so fill-empty
  can't reach them): propose improved gloss lists; always print
  old → new per record; per-record confirm (or `--yes`); write + ledger.

### [x] M4.4 Batch submit/fetch

*Done 2026-08-08. Contract as written, plus one file the list omitted and
seven decisions. **Added file:** `src/japanese_anki/claude_client.py` —
the batch endpoints had to go there, because that module's whole stated
contract is being the only one that knows what an Anthropic request looks
like, and it names M4.4 in its own docstring. `parse_call` and
`batch_request` now build their request from one `_request_body`, so the
promise "the same request at half price" is structural rather than
remembered. (1) **`custom_id` is a fingerprint of the record id**, not
the id: the API wants a short ASCII identifier and `word:話す:はなす` is
the wrong alphabet and an unbounded length. The fetch side recomputes the
map from the ledger's pending list rather than storing a second copy, and
a fingerprint collision is **refused before submitting** — it takes a
birthday collision across 48 bits, but "unlikely" is not the standard for
silently writing one word's example onto another. (2) `absorb_ai_call` is
extracted and public so the batch path runs M4.2's QC *code*, not a
second copy of it. (3) **No variety pressure in a batch** — every request
is built before any answer exists. A real difference in output, and why
the live path stays the default. (4) A one-hour cache TTL: five minutes
does not survive the span a batch's requests are read over. (5) A large
fetch lands in `staging/ai-enrichment.yaml` exactly as a large live run
does, via a shared `_write_ai_result` — batch is what janki reaches for at
a thousand records, which is not a number of sentences anyone reviews in a
terminal. (6) **Anything short of a written record keeps the batch
pending**: a declined diff, a failed ledger write, an existing staging
file. Results live on Anthropic's side for weeks and fetching again is
free, so forgetting the id is the only irreversible thing in the command.
(7) **Four** ways a record comes back with nothing — errored/expired/canceled,
never mentioned, no longer in the collection, or an answer that did not
validate — each reported by name, because a batch runs for up to a day and
the collection does not hold still. The fourth is different in kind and is
the only one that holds the batch id: the answer is complete and paid for
and lives on Anthropic's side for weeks, with janki's schema the only thing
rejecting it, so clearing over it would be the one irreversible act in the
command for the one failure that was not the API's. The rule the whole
fetch is arranged around: **clear the id once every record still in the
collection has been accounted for and nothing that remains is
recoverable.** A record deleted while the batch was out is deliberate,
so its answer is moot and does not hold the id; an unreadable answer for
a record that is still here does. The all-missing guard is a separate
judgment on top of that — *every* record gone reads as an accident
rather than curation, so it refuses instead of clearing. The fetch uses the **batch's** model, not the config's: a run
submitted under one model was answered by that one.

Amended after a second review found the all-missing guard could deadlock:
a one-record batch whose single word was deleted while it was out has no
way to tell "curation" from "the collection moved", and every later fetch
raised while every later submit was refused. The fix was **`--batch-forget`** (outside the task's
surface, deliberately) rather than a weaker guard — the alternative
remedy was hand-editing `data/ledger.json`, which AGENTS.md forbids for a
file janki writes. Both guards stand: an **empty collection** is a
`--root` pointed elsewhere, and a **non-empty one holding none of the
batch's ids** is a `--replace` import or a promote that re-minted them.
Either refuses rather than clearing the id of a batch whose answers are
alive on Anthropic's side for weeks. (I removed the second guard first,
which was backwards: the escape hatch is what makes keeping it safe.)
`--force-fields` at fetch **overrides** the stored list rather than being
refused with `--ids`/`--model`, and says so — the model and the ids
describe what was asked and are settled, while this decides how an answer
already in hand is applied, and a field may have filled in the meantime.
`--batch-submit` and `--batch-forget` no longer build a jpdb client:
neither asks jpdb anything, so neither may demand a key.*

Depends on: M4.3 (same files), M1.3
Files: `src/japanese_anki/enrich.py`, `src/japanese_anki/cli.py`,
`src/japanese_anki/ledger.py` (pending_batches accessors),
`tests/test_enrich_batch.py` (new).
Design: DESIGN_V2 "Batch mode".

- `enrich --ai --batch-submit`: same requests via
  `client.messages.batches.create` (custom_id = record ID); persist
  `{batch_id, submitted_at, pending_ids}` in the ledger's
  `pending_batches`; refuse to submit while one is pending.
- `--batch-fetch`: poll once; `ended` → stream results keyed by
  custom_id, run the exact M4.2 QC/apply path, clear the entry; still
  processing → print status, exit 0. Errored/expired per-record results
  are reported and leave those records untouched.

### [x] M4.W Milestone 4 wrap

*Done 2026-08-08. README gained "Writing what a dictionary cannot" — the
two passes that write rather than look up, batch mode, a rates table and
the model config. The cost advice is deliberately not a per-record
number: janki's request is small and fixed, so the output dominates,
models think before answering and thinking bills as output, so the honest
instruction is to measure one record rather than trust a table (this one
included). Said plainly too: prompt caching is asked for on the system
prefix and the style guide as shipped is probably under the per-model
minimum, and the API is silent about missing it.
`prompts/ENRICH_VOCABULARY.md` became a migration table plus the one
honest gap — **`transitivity` has no successor pass.** It is set at
import and nothing backfills it, so claiming the prompt's rules all
survived would have been false. *(That file was deleted 2026-08-15 with
the rest of `prompts/`; the gap it recorded is carried into M8.5, which
closes it.)*

Three things the reviews turned up that were code, not prose. (1)
`_print_merge_summary` advertised `--prefer-incoming` on `promote` and
`migrate-inline`, which do not take it — following the program's own
advice gave "unrecognized arguments". (2) The identity marker on a
conflict line had to survive that fix: `expression`/`reading` clashes are
not one more field to settle by hand. (3) A test asserting `"identity" in
capsys...out` passed with the feature deleted, because pytest names
`tmp_path` after the test and promote echoes the path — a whole class of
false green, now recorded in memory.*

Depends on: all M4 tasks
Files: `README.md`, `prompts/ENRICH_VOCABULARY.md`.

- README: enrichment costs table, model configuration. Replace
  `prompts/ENRICH_VOCABULARY.md` body with a pointer at `janki enrich`.

---

## Milestone 5 — Audio

Lane map: {M5.1, M5.2, M5.5 in parallel} → {M5.3} → {M5.4} → {M5.6};
M5.7 anytime after M5.3.

### [x] M5.1 Pitch conversion + HTML renderer (merge gate: golden tests)

*Done 2026-08-08. DESIGN_V2's conversion implemented exactly, golden set
in place. **Verified against live jpdb the same day**, which is what the
goldens were previously asserting from memory: 橋 `LHL`, 箸 `HLL`, 端
`LHH`, 病院 `LLHHHH`, 授業 `HHLLLL` — every hand-written golden matches
the API exactly, and the `len(pattern) == len(reading) + 1` invariant
holds on all five. 病院 returning **six** characters for a **four**-mora
word is the part that matters: it confirms the one-character-per-*kana*
width, which is what forces the regrouping. It does **not** confirm the
"read the level off the mora's first kana" choice, as an earlier version
of this note claimed — びょ's two kana both carry `L` there, so first-
and last-kana readings agree, and 授業 is non-discriminating the same
way. No live response can confirm that rule: it only differs on a source
that writes a small kana with the *following* mora's level, which is
malformed input jpdb does not produce. It stays a defensive choice, held
in place by `test_the_moras_level_is_read_off_its_first_kana` alone. Run end to end on the real collection too:
行く → `イク'`, 話す → `ハナ'ス`, 食べる → `タベ'ル`. Five decisions worth recording. (1) **Heiban and odaka produce
the same AquesTalk string, and that is right, not a bug to fix later.**
The notation carries one mark per phrase and the engine writes heiban on
the final mora — where odaka's goes. The two differ only in the pitch of
a particle, and janki speaks a word alone, so there is nothing for the
distinction to land on. `render_pitch_html` *does* keep them apart, since
a card shows the pattern rather than speaking it, and a test asserts both
halves. (2) A mora's level is read off its **first** kana — the one that
cannot be the small one — so a source that ever writes a 拗音's small kana
with the following mora's level still converts correctly. (3) The reading
is NFC-normalized before the length check: が typed as か+U+3099 is two
codepoints and one kana, and counting codepoints would reject a pattern
that fits. (4) The diagram draws the **particle slot** as an empty mora,
because odaka's fall happens after the word and a diagram stopping at the
last kana has nowhere to show it. (5) `render_pitch_html` refuses a bare
`str` for `patterns`: a str is a `Sequence[str]` that iterates as
characters, so passing one would render a diagram per character or fail
far from the mistake. Refusals throughout rather than best efforts —
a wrong accent spoken onto a card built to teach that accent is the one
outcome worse than no audio.

Amended same day, from review. **The merge gate did not gate.** Both my
拗音 goldens were 病院, which is heiban — and under heiban the mark lands
on the last unit however the kana are grouped, so the naive per-kana
mapping passed every assertion in the file. A 拗音 gate has to have its
drop *inside* the word: 授業 `HHLLLL` → `ジュ'ギョウ`. The same flaw ran
through the っ/ん cases (both hold with them wrongly attaching) and the
first-kana case (holds if the *last* kana is read).

A second review found I had reported that verification more confidently
than I ran it — I mutated the module once and read an aggregate count,
and one of the four replacements (発表) still discriminated nothing,
while the gate caught a *different* mutation than its docstring named.
Each case is now mutated **individually** and the failing test named:
per-kana-vs-per-mora pattern consumption and 拗音 grouping (the gate,
via string and span count), っ mora-hood (`いっき` `HLLL` → `イ'ッキ`,
which is stated as a pattern rather than a claim about any word's
dictionary accent — asserting an accent class from memory in a golden
file is how a wrong one gets copied forward), the NFC kana count, the
lowercase pattern, and the ledger's delegation. Also unified with two
neighbours it had quietly forked from: `ledger._selected_pitch_pattern`
now delegates to `pitch.select_pattern` — the word-audio content
fingerprint is computed *over* that choice, so two definitions would mean
audio generated under one is never stale under the other — and
`validation` counts kana the same NFC way, having warned about patterns
the converter accepts.*

Depends on: M2.2
Files: new `src/japanese_anki/pitch.py`, new `tests/test_pitch.py`.
Design: DESIGN_V2 "Pitch-pattern conversion" — implement exactly; the
naive rule is wrong for heiban.

- `to_aquestalk(reading, pattern) -> str` (katakana + `'`):
  1. Assert `len(pattern) == len(reading) + 1` (kana positions +
     particle slot); mismatch → `PitchError` (callers flag, never
     guess). `PitchError` defined here.
  2. Regroup reading kana into morae (small ゃゅょ attach; っ and ん
     are morae); map the kana-positional pattern onto morae.
  3. Find the H→L drop. Within the word → accent mark after that mora.
     Only at the particle boundary → odaka: mark after the final mora.
     No drop anywhere → heiban: engine convention, mark on the final
     mora (VOICEVOX requires exactly one mark per phrase).
  4. Output katakana.
- `select_pattern(record) -> str | None`: `audio_accent` override,
  else `pitch_accent[0]`, else `None`.
- `render_pitch_html(reading, patterns) -> str`: span-per-mora inline
  HTML (border-top/right styling classes; all patterns, primary
  first) — pre-landed here so M5.4 stays an atomic exporter change.
- Golden tests (the merge gate): heiban (端 LHH), atamadaka (箸 HLL),
  nakadaka (卵 LHLL), odaka (橋 LHL), 拗音 (病院 びょういん), っ and
  ん words, length-mismatch error. Expected AquesTalk strings cited in
  test comments.

### [x] M5.2 VOICEVOX provider

*Done 2026-08-08. Flow exactly as specified. One deviation and three
decisions. **Deviation:** the protocol has a fourth member, `launch_hint`.
The task list gives `name`/`voice` and the reason — M5.3 writes them
without an `isinstance` ladder — and "an engine that is not running" is
the most common failure of a locally-hosted one, so the CLI needs to say
how to start it under exactly the same reasoning. (1) `available()`
never raises: a local engine being down is this system's ordinary state,
so refused connection, timeout and error status are all the same answer,
and the caller prints the hint. (2) The accent mark is stripped only on
the forced path — in ordinary text an apostrophe is punctuation, and
removing it would change what is spoken. (3) `/audio_query` answering
with anything but an object is refused rather than replaced with a
hand-built AudioQuery, which is the same quiet-failure trap as
constructing defaults.

Amended same day, from review. The substitution was gated on the
*payload* (`forced is not None`), which cannot tell "the engine answered
`null`" from "we never asked" — so a 200 with `null` fell through to the
accent `/audio_query` guessed, and an empty list synthesized to silence;
both would be ledgered as forced-accent audio. It branches on the flag
now and refuses an unusable answer. `urllib_transport` also converted
too little: `urlopen` wraps only the *request* in `URLError`, so a peer
that accepts a connection and closes it arrives as a bare
`RemoteDisconnected`, and a schemeless URL (an empty `voicevox_url`) as a
`ValueError` from `Request()` — which sat outside the `try`. Every
transport failure is a `TtsError` now, which is what lets `available()`
promise it never raises.

The ways this breaks are each pinned by the test that names them,
verified by mutation: `is_kana` reaching `/audio_query` (the silent one —
FastAPI drops it, the request succeeds, the audio is guessed), the forced
phrases never substituted, the mark left in for `/audio_query`, and a
hand-constructed AudioQuery.*

Depends on: M1.6 (not M5.1 — tests use a hand-written AquesTalk string)
Files: new `src/japanese_anki/tts/__init__.py` (provider protocol),
new `src/japanese_anki/tts/voicevox.py`, new `tests/test_voicevox.py`.
Design: DESIGN_V2 "Audio > VOICEVOX" — the endpoint choice is the
whole ballgame.

- Provider protocol: `synthesize(text_or_kana, *, forced_accent: bool)
  -> bytes` (WAV), `available() -> bool`, plus `name: str` and
  `voice: int` properties (M5.3 writes these to the ledger without
  isinstance checks).
- Transport (VOICEVOX-shaped, not the jpdb one):
  `(method, url_with_query, json_body | None) -> (status, bytes)`;
  JSON decoding inside the provider. `VoicevoxProvider(base_url,
  speaker, transport=None)`; `available()` = `GET /version`.
- Flow, exactly: `POST /accent_phrases?text=<kana>&is_kana=true
  &speaker=N` — **the only endpoint that accepts `is_kana`; on
  `/audio_query` it is silently ignored (FastAPI drops unknown query
  params) and you get normally-parsed, possibly-wrong audio** →
  `POST /audio_query` with the accent-mark-**stripped** kana (never
  "construct defaults" — the AudioQuery schema is engine-versioned) →
  replace its `accent_phrases` with the forced ones →
  `POST /synthesis?speaker=N`.
- Unit test with a fake transport records the request sequence and
  asserts (a) `is_kana=true` went to `/accent_phrases` and only there,
  (b) the accent phrases submitted to `/synthesis` are the forced ones
  (marker-value round-trip).
- `available()` failure message includes a launch hint.

### [x] M5.5 Notetype-upgrade verification (spike; blocks M5.4)

*Done 2026-08-08, in the desktop app as well as the library — and the two
disagreed, which is the finding. Full write-up in
`docs/NOTETYPE_UPGRADE.md`.

**Append + reimport upgrades the notetype in place, matches notes by GUID
and keeps review history — but only with "Merge Notetypes" enabled, and
Anki's default is off.** With it off the import keeps every existing note
on the 19-field notetype and files the incoming 22-field one as a
separate, empty notetype with `+` appended to its name. Nothing errors,
nothing is lost, scheduling survives either way — the new fields simply
never reach a note. Established three ways: library with the flag on
(19→22 in place, ivl 21 / reps 4 preserved), the owner's real GUI import
at default settings (3 notes on the old type, an empty `…+` beside it,
both studied cards still `Due 2026-08-08`), and the library with the flag
off, which reproduced the GUI result exactly and so identifies the option
as the whole difference.

**M5.4 must therefore**: document the checkbox wherever the README says
to import a rebuilt deck; detect the failure (a notetype named `…+`, or
an existing one with fewer fields than `FIELD_NAMES`) and say so in
`janki status`; and **never** renumber `model_id` to force a fresh
notetype, which would orphan every card's scheduling.

*Revised 2026-08-08.* The first and third landed in M5.4. The detector
did not, and the reason is that it was specified against the wrong
subject: `janki status` reports on **`vocabulary.json`** — "the
collection" in janki's vocabulary is the repo, not Anki's. Nothing in
janki opens `collection.anki2` or knows where it lives, so "detect the
`…+` notetype" is a new capability rather than a status tweak. It moves
to **M5.8**, which is where that capability gets decided.

One thing this spike did not record and should have: **appending a field
is a schema change**, so it forces a one-directional full AnkiWeb sync.
Verified 2026-08-08 against `anki` 26.08.1 — `models.add_field` +
`update_dict` bumps the collection's `scm` mark. The README now says to
sync before importing, so the direction you are asked to choose is the
trivial one.

The spike was worth running twice: the library alone would have shipped
"append is safe" as unconditional, and the condition is the part that
bites.*

### [x] M5.3 `janki audio`

*Done 2026-08-08, **including a live run against VOICEVOX 0.25.2** (the
arm64 engine image under colima, native rather than emulated). Contract
as written.

The live run settled the question M5.2 was built for, and the answer is
that the feature is load-bearing rather than theoretical. Asked for the
accent phrases of はし three ways:

| word | pattern | forced (`is_kana=true`) | engine's guess |
| --- | --- | --- | --- |
| 橋 odaka | `LHL` → `ハシ'` | accent **2** | accent 1 |
| 箸 atamadaka | `HLL` → `ハ'シ` | accent **1** | accent 1 |
| 端 heiban | `LHH` → `ハシ'` | accent **2** | accent 1 |

Left to guess the engine gives all three **accent 1** — so 橋 and 端 are
spoken as 箸, which is exactly the failure DESIGN_V2 predicted. Forcing
separates 箸 from the other two, and 橋 vs 箸 synthesize to different
bytes through janki's own provider (same length, different md5). It does
**not** make all three distinct, and must not be read that way: 橋 and 端
share `ハシ'` and accent 2 by design, because AquesTalk carries one mark
per phrase and the odaka/heiban difference lands on a particle janki does
not speak (`pitch.py`'s module docstring). What forcing buys is that 橋
and 端 stop being pronounced *wrongly*, not that they become
distinguishable from each other in isolation. The three real records voiced correctly, a re-run reported "3
already current" and wrote nothing, and `--examples` voiced the
sentences. Decisions: (1) at
least one of `--words`/`--examples` is required rather than defaulting —
they are different recordings made different ways, and neither is the
obvious default. (2) `--provider azure` is refused **by name** until
M5.7 rather than falling back to VOICEVOX, which would record Azure in
the ledger against VOICEVOX's audio. (3) The command checks
`available()` before spending, because "the engine is not running" is the
ordinary failure and is knowable without doing work. *(Superseded 2026-08-18
under M8.1: later failures no longer leave direct canonical writes and a
half-ledger. Every completed clip is staged and write-ahead ledgered before the
record CAS, then published/finalized only if that CAS wins.)* (4) "Is there already a clip for
this?" needs **three** facts and the ledger holds one: an entry saying
this was recorded, the record still naming that file, and the file
existing. Asking the ledger alone (which is what shipped first, as
`has_audio`) calls a record current whose reference was dropped, and then
`--prune` deletes the clip nothing appears to want. `ledger` exposes
`audio_file_for`, returning the *name* rather than a yes/no, and
`audio_cmd._is_current` checks all three. (5) `--prune` deletes only
`janki-*`: a clip somebody dropped into `data/media` by hand is theirs.
A pattern that does not fit its reading is reported and skipped, because
`to_aquestalk` refuses rather than guesses and this command must not
convert that refusal into a guess of its own.* *(Reversed 2026-08-15 under
M8.1: it takes the no-pattern fallback and is voiced with the engine's
accent, warning on every run rather than being silently absent.)*

Depends on: M5.1, M5.2, M2.2, M1.3
Files: new `src/japanese_anki/audio_cmd.py`, `src/japanese_anki/cli.py`,
`tests/test_audio_cmd.py` (new).
Design: DESIGN_V2 "Audio > Mechanics".

- Flags: `--words`, `--examples`, `--provider {voicevox,azure}`
  (overrides `[tts] provider`; azure errors "not implemented" until
  M5.7), `--force`, `--prune`, `--allow-default-accent`, `IDS...`.
  *(`--allow-default-accent` deleted 2026-08-15 under M8.1 — the fallback
  it opted into is now unconditional.)*
  `AudioError` defined here.
- `--words`: records with reading + `select_pattern` result and no
  current word audio (ledger content_fp-aware): M5.1 convert → M5.2
  synthesize → `media_dir/audio/janki-<filename fp>.wav` →
  `record.audio` (media-dir-relative) → ledger `record_audio` with the
  word content fp. Empty pattern → **skip + flag** in the summary;
  `--allow-default-accent` synthesizes without forcing (plain
  audio_query path) and ledger-tags `accent_unverified`. *(2026-08-15:
  the unforced path is what every pattern-less record takes now; the
  flag is gone and the summary reports the guess on every run.)*
- `--examples`: per example with furigana **not** listed in the
  record's unverified-furigana key (see Conventions) and no current
  audio: synthesize sentence (no accent forcing), filename/content fps
  per Conventions, set `example.audio`, ledger entry.
- Staleness combines identity-addressed filenames with content fingerprints:
  edited examples no longer match their recorded clip; `--prune` deletes
  unreferenced `janki-*` files.
  `--force` regenerates in place (same filenames; Anki media sync
  picks up content changes).

### [x] M5.4 Exporter + templates (ships all new note fields at once)

*Done 2026-08-08. `FIELD_NAMES` 19 → 22, appended in one release, with the
rule written at the append itself and in CARD_DESIGN. Verified in a real
build: notetype `1607392313` carries 22 fields, notes carry 22 values,
and the three real records package 6 media files.

Decisions. (1) One `_resolve_media` for all three media fields rather
than three copies of the same branch — that duplication is what let the
image path keep the old base directory after audio moved. (2) A verbatim
`[sound:]` tag warns and is passed through: no file is packaged, so the
card is silent unless that media is already in the collection, which for
a pipeline that generates its own audio is a trap rather than a feature.
Warnings ride out on `BuildResult` rather than printing from the
exporter, so `build` decides how loudly to say it. (3) A pattern that
does not fit its reading leaves `PitchAccent` **empty** — a card is the
last place to start guessing at an alignment refused everywhere else.
(4) The diagram draws the particle slot, which is what makes odaka
visible: the fall lands after the word, so a diagram stopping at the last
kana would make 橋 look identical to 端. Their *audio* does collapse
(M5.1); the diagram does not. (5) Production-back gains word audio and
the diagram, since a production card asks you to say the word and the
answer side should say it back.

Each behaviour pinned by mutation: swapping the appended field order,
raising instead of skipping an unfittable pattern, dropping the
`media_dir` fallback, and removing the passthrough warning all go red.*

Depends on: M5.3, M5.5, M5.1
Files: `src/japanese_anki/exporters/anki.py`,
`templates/japanese-study/*.html`, `templates/japanese-study/style.css`,
`docs/CARD_DESIGN.md`, `tests/test_anki_build.py`,
`tests/test_anki_builder_contract.py`.
Design: DESIGN_V2 "Schema changes" + "Audio > Mechanics".

- `FIELD_NAMES` += `PitchAccent`, `FrequencyRank`, `ExampleAudio` —
  appended at the end, one release, never again inserted. Parallel
  positional list in `_field_values` extended to match.
- `PitchAccent` content = M5.1's `render_pitch_html`. No card JS.
- Media resolution: `media_dir`-relative first, deck-relative
  fallback. The `[sound:...]` passthrough branch prints a warning
  naming the record.
- Templates: audio on production-back; example audio + pitch display
  on recognition-back; romaji stays behind its disclosure.
- `CARD_DESIGN.md`: append-only field rule, inline-notes `source`
  shallow-merge nuance, link `NOTETYPE_UPGRADE.md`; follow whatever
  import procedure M5.5 concluded.

### [x] M5.6 `build --only-new` + `janki refresh`

Depends on: M5.4 (exporter settled), M2.6, M4.2 (refresh calls them),
M1.3
Files: `src/japanese_anki/cli.py`,
`src/japanese_anki/exporters/anki.py` (only-new filter hook),
`tests/test_build_only_new.py` (new).
Design: DESIGN_V2 "CLI surface".

- `build DECK --only-new`: DECK = path or bare name resolved against
  `deck_dir`; filter resolved records to those without an export entry
  for the deck stem; **warn with counts** when included records lack
  audio / examples / pitch accent ("12 of 15 new records have no
  audio — continue? [y/N]"; `--yes` skips; non-TTY proceeds). On
  success, `record_export(record_id, stem, at=today)` per included record; plain
  `build` records exports too (idempotent).
- `janki refresh [--deck DECK]`: enrich --jpdb → enrich --ai →
  audio --words --examples → build --only-new, in order, per-stage
  skip flags, one-line summary per stage. Calls the command functions
  directly (no subprocesses).
- While in the exporter: route the `.apkg` write through temp-then-rename
  (genanki needs a real path — `write_to_file(tmp)` then `os.replace`) so
  an interrupted build can't leave a truncated package that a later
  `--only-new`-era run mistakes for a good one.

*Landed 2026-08-08.* Two things the plan did not name and the code needed:

- **`--only-new` with nothing new writes no package.** genanki will happily
  produce a zero-note deck, which would replace the last good `.apkg` with an
  empty one — worse than an out-of-date deck, because the file on disk is what
  the user imports.
- **Exports are recorded from `BuildResult.record_ids`, not from the deck
  file.** Recording every deck record would mark records exported that the
  package does not contain, and those are invisible to every later
  `--only-new` — they would never reach a card and nothing would report it.
  Pinning this needed a ledger seeded with an *older* export date; with both
  builds dated today the right and wrong behaviours are byte-identical.

Verified live against the real project: all four stages ran in order and each
correctly reported a no-op.

### [x] M5.7 Sentence-audio provider — **OpenAI, not Azure**

Both premises were tested and neither held.

**"A natural adult voice matters more for sentences"** was written when
VOICEVOX meant speaker 46, the default character voice. The owner chose
青山龍星 (13) at 0.7×. The premise was about a voice nobody was using.

**Reading control** was the stronger argument, and the only one that
was about correctness rather than taste: word audio forces the accent,
sentence audio forces nothing, so VOICEVOX guesses every reading in a
sentence and a wrong guess teaches a wrong reading. Azure's
`<sub alias>` fed from verified furigana would fix that. Checked
against a live engine, speaker 13: eight classic two-reading traps
(今日, 上手, 大人, 今朝, 市場, 何か, 人気, 一日中) all correct, and all
three of the project's own example sentences correct — including
城崎温泉 → キノサキオンセン, a non-obvious place-name reading. OpenJTalk,
which VOICEVOX uses underneath, is better at this than the plan assumed.

Against that: an Azure account, `AZURE_SPEECH_KEY`, per-character
billing, a network round-trip per sentence, and a second provider to
maintain — replacing something local, free, and working.

**What shipped instead** is `voicevox_sentence_speaker`: a second local
voice for example sentences, which is the part of the idea that was
actually wanted (a sentence should not sound like a longer word). It
went in as a **second provider** rather than a second voice on one
provider, because everything downstream — the voice the ledger records,
the voice `_is_current` compares — already asks *a provider*. That seam
is exactly where an Azure provider would attach, so this stays cheap to
revisit if the reading accuracy above ever stops holding, or if the deck
is shared publicly and VOICEVOX's per-character terms of use become the
deciding factor.

**Superseded 2026-08-09 by an OpenAI provider.** The owner listened to
both engines on their own sentences and preferred OpenAI's. Since the
key was already set for the AI enrichment passes, the account objection
that sank Azure did not apply, and the sentence seam landed the day
before was where it attached — `src/japanese_anki/tts/openai_tts.py`,
`tests/test_openai_tts.py`, ~30 lines of CLI and config.

What the build had to add beyond the seam:

- **The protocol's `voice` widened to `int | str`.** VOICEVOX numbers
  its speakers, OpenAI names them. Mapping `onyx` onto an index would
  put a number in a committed ledger that means nothing outside janki.
- **A `suffix` on the protocol.** The API's WAV is a *streaming* WAV
  with `0xFFFFFFFF` placeholder chunk sizes — its header claims 89,478
  seconds for a 4.5-second clip. mp3 avoids it, and a hard-coded `.wav`
  downstream would have named an mp3 file `.wav`.
- **A refusal, not a fallback.** `synthesize(forced_accent=True)`
  raises. A provider that accepted the flag and ignored it would return
  a clip for 橋 that says 箸 — the failure this milestone exists to
  prevent, arriving as working audio.

Azure remains unbuilt and the reasoning above still holds against it.

**Conversational audio models were tested too, and cannot do this job.**
`gpt-audio` and `gpt-audio-1.5` on `/v1/chat/completions` generate speech
natively rather than rendering finished text, which is the architecture
ChatGPT's voice mode uses and a fair thing to want. But janki needs the
clip to say *exactly* the sentence on the card, and they are chat models:
asked to read 「日本語を話しますか。」 they **answer** it (0/3 verbatim,
"はい、話せますよ…"), and asked to read 「すみません、駅はどこですか。」 they
give directions to an invented station (0/3, a different route each time).
`gpt-audio-mini` replied to a statement with a paragraph about Korean
food. `gpt-audio-1.5` also swapped 。for an ASCII period once on a
sentence it otherwise read correctly, changing the closing intonation.

The failure is silent and total: the card plays fluent, confident
Japanese that is not the sentence. Questions are ordinary in a beginner
deck, so this is not an edge case. **TTS renders what it is given;
conversational models respond to it** — janki needs the first, which is
the same distinction that keeps words on VOICEVOX one level down.

GPT-Live is not on the API as of 2026-08-09, and when it arrives it will
be a conversational model, so it would fail this same test. The realtime
family (`gpt-realtime-2.1`, WebRTC) is the right tool for live
conversation practice, which is a different product from a deck builder:
janki writes files to disk, with no browser, microphone, or session.

~~Depends on: M5.3
Files: new `src/japanese_anki/tts/azure.py`, `tests/test_azure_tts.py`.~~

~~- REST via the M5.2 transport shape (no SDK dep): key header, SSML
  body, voice/region from config, `AZURE_SPEECH_KEY` env.
  `<sub alias="...">` substitution fed from **verified** furigana for
  reading-ambiguous tokens. Fake-transport tests assert SSML shape and
  sub/alias injection. Wire into `audio --provider azure`.~~
**Not built.** The instructions above are the cancelled Azure plan, kept
for the record. What shipped is `tts/openai_tts.py`; see below.

### [x] M5.8 Tell the user when an import silently did not upgrade

Depends on: M5.4
Files: TBD — the shape of this task is the decision it has to make first.

Carried out of M5.5's spike (see the revision note there). With "Merge
Notetypes" left off, an import leaves the old notetype in place and files
a `…+` clone with zero notes beside it. Nothing is lost, nothing errors,
and none of the new fields reach a card — the user finds out when they
notice a diagram that never appears.

The obstacle is that janki cannot see any of this. It has no path to
`collection.anki2`, no config naming one, and reading it directly needs
Anki closed. So this task's first job is choosing how janki learns what
is in the collection:

- **Read `collection.anki2` directly** — no dependency, but only while
  Anki is shut, and a wrong guess at the profile path is worse than
  silence.
- **AnkiConnect** — an HTTP API most collections already have installed,
  works with Anki open, and would let `janki status` ask directly. Also
  the route by which a later janki could write into a live collection
  rather than shipping a package — which would make this whole failure
  mode structurally impossible, since the checkbox belongs to the import
  dialog and there would be no import dialog. (Prior art:
  `ankimcp/anki-mcp-server-addon` runs an MCP server *inside* Anki on the
  same bridge-to-the-Qt-thread pattern AnkiConnect uses.)
- **Say nothing and document harder** — the README now covers the
  checkbox and the full-sync consequence, which may be enough.

Whatever it picks, the detector must read `deck.model_id` rather than the
derived value, or it will misreport any deck that pins one.

*Done 2026-08-09. It picked **read the file**, and the choice was easy
once measured.* `src/japanese_anki/collection.py` is standard library
only: three SQL queries against a **copy** of `collection.anki2`. The
`anki` package would cost a large version-coupled dependency and
AnkiConnect an add-on install plus a running Anki, and neither buys
anything for a question with a yes-or-no answer. If janki ever needs to
*write*, that trade changes — `notetypes.config` is protobuf and writes
need USN/`mod`/`scm` bookkeeping, so writing by hand is not on the table.

The plan said direct reading "needs Anki closed". That was half right and
the fix is trivial: Anki holds an exclusive lock, so an ordinary
read-only open answers `database is locked` — but copying the file and
reading the copy works with Anki open, which is when someone actually
runs `janki status`. Three wrinkles, all found by trying it against a
real collection rather than reasoning about it:

- Anki registers a custom `unicase` collation; any query ordering by a
  collated column fails until one is registered.
- The collection is WAL-mode, so copying the main file alone can read a
  stale snapshot. The `-wal`/`-shm` sidecars are copied with it.
- `notetypes.config` is protobuf, not JSON. Nothing needed from it —
  names and field counts are plain relational tables.

`deck_notetype` is exported from the exporter so the check asks the same
question a build answers, honouring a pinned `model_id`/`model_name`
rather than recomputing them.

**Scope, deliberately narrow.** Only the notetypes this project's decks
build are inspected. The collection this was written against holds 1,815
notes on `+` clones — Yotsubato, Tofugu, a Quizlet course, all from
re-importing updated shared decks — and none are janki's business.
Listing them would bury the one actionable line.

Nothing here is ever an error. No Anki, no import yet, several profiles
to choose between, or a collection too old to read are all ordinary
states reported as warnings, because `janki status` is the command people
run *because* something is already confusing.

### [x] M5.W Milestone 5 wrap

Depends on: all M5 tasks
Files: `README.md`.

- README: audio setup (VOICEVOX install/launch, credit line for shared
  decks), `janki refresh` as the headline weekly workflow, retire the
  "no synthesized audio" limitation paragraph.

*Done 2026-08-09.* The README gained a top-level **Audio** section
covering both engines, how to get VOICEVOX running, how to audition
voices, and the per-character terms of use that matter only if a deck is
shared. Three claims were retired as false rather than merely stale:

- "`janki audio` arrives in Milestone 5" and "`janki build` does not yet
  mark records exported, so `--unexported` currently lists everything".
- **"Changing a voice does not make existing audio stale... so re-run
  with `--force`."** This was the most dangerous of the three, because
  it was accurate when written and became *wrong* rather than
  incomplete: the ledger now records engine, voice, rate and style
  settings per clip, so a change re-voices exactly what it affects and
  `--force` is a bigger hammer than the situation calls for. A reader
  following it would have re-rendered a whole collection — through a
  paid API, for the OpenAI half.

Milestone 5 shipped one thing it did not plan (`tts/openai_tts.py`) and
did not ship one thing it did (`tts/azure.py`); both are recorded under
M5.7. The `janki status` notetype detector specified in M5.5 moved to
M5.8, which completed it.

---

## Milestone 6 — Enriching decks janki did not create

The v1.0.0 line is finished: every task in Milestones 1–5 is `[x]`, the
history carries no AI co-author trailers, and `main` is tagged. This
milestone starts from a question the earlier ones never asked — *what
about the decks already in Anki?* — and from what a twenty-record pilot
of that idea measured.

### [x] M6.1 `janki import-anki` — read a deck out of the collection

Files: `src/japanese_anki/importers/anki_deck.py` (new),
`src/japanese_anki/collection.py` (`read_deck_notes`),
`src/japanese_anki/identifiers.py` (`is_kana`), `src/japanese_anki/cli.py`,
`tests/test_import_anki.py` (new).

Reads a deck through the same copy-then-open-read-only path `janki
status` uses, maps Front/Back into records, and writes them to
`data/staging/` — never to the collection, because which line of a
shared deck's HTML is the reading is inference, and inference does not
ship unread here. janki still never writes to Anki.

Measured on the Yotsubato Volume 1 reading pack: **708 records from 716
notes**, the remaining eight being the same word listed twice. The deck's
conventions, each found by looking at what was failing rather than
assumed: `すごい　「すげえ」` (reading then the spoken form, kept as a usage
note), the same variant on its own line (resolved by position — the first
*unbracketed* kana line is the reading), `「げつもく」` with nothing outside
the brackets (there the brackets hold the word), and `あれ？` (punctuation
belongs on the card, not in a pronunciation that becomes half the record
id). Two unbracketed kana lines stay held: a deck glossing in Japanese
looks exactly like one giving a reading.

### [x] M6.2 What the pilot's review gate paid for

Files: `src/japanese_anki/jpdb.py`, `src/japanese_anki/kanji.py`,
`src/japanese_anki/enrich.py`, `src/japanese_anki/qc.py`.

Three defects the gate found on twenty cards, all systemic — each would
have recurred across the remaining 688:

- **A part of speech jpdb names second.** `["adv", "adj-i"]` for 凄い,
  `["adv", "n"]` for 明日. Taking the first recognized code cost an
  い-adjective its conjugation table and made temporal nouns adverbs.
- **A reading split across characters that do not have it.** jpdb hands
  back one reading per character; 明日 arrived as `明[あ] 日[した]`, and
  した is no reading of 日. Checked against KANJIDIC, which janki already
  holds, so it is a lookup rather than an opinion. It caught 一杯 too.
- **A spill that needed a segmentation.** jpdb's parse of the same
  sentence is fetched to verify readings and was then discarded; its word
  boundaries say where the separator goes. No extra call, no model.

### [x] M6.3 Report the records an AI pass wrote nothing for

*Done 2026-08-11. Live and batched passes now report both the count and every
record id whose valid model answer produced no writable change, including when
other records in the same pass did change.*

Depends on: M6.2
Files: `src/japanese_anki/enrich.py`, `src/japanese_anki/cli.py`,
`tests/test_enrich_ai.py`.

`enrich --ai` visited 凄い and よつば, wrote nothing for either, and
reported only `Enriched 17 record(s)`. A rejected sentence *is* reported;
an answer that produces no change is not, so the only way to notice is to
diff the collection. At 708 records that is roughly 75 records quietly
without examples.

- Count targets, and name every target that produced no change, in the
  same shape the rejection warning already uses.
- A test that a record whose answer yields nothing is named on stdout.

### [x] M6.4 A concurrent-write guard for the records file

*Done 2026-08-11. Every read-modify-write command captures the exact records
file it started from and `save_records_json` refuses if that content changed
before the atomic replace. The newer file stays intact and the error names the
safe remedy: let the other command finish, then re-run. Review hardening put an
interprocess lock around the comparison and replace (and the ledger's matching
transaction), closing the remaining window where two writers could both pass
the comparison. The stable lock lives under a private per-user temp or cache
directory, not in tracked `data/`.*

Depends on: —
Files: `src/japanese_anki/io.py`, `src/japanese_anki/ledger.py` (the
existing guard is the model), `tests/test_io.py`.

`enrich --ai` run while `janki audio` was still writing saved
`vocabulary.json` from a copy read before the audio paths landed: 48
clips on disk that no record referenced. The ledger has a guard for
exactly this and refused its own write with a message naming the fix; the
records file has none.

- Same shape as `ledger`'s: remember what was read, refuse to save over a
  file that changed underneath, and say what to re-run.
- A test that two writers over one file lose nothing silently.

### [x] M6.5 `--polish-meanings` at scale

*Done 2026-08-11. Chose Message Batch submit/fetch. Fetch keeps the existing
per-record diff and y/n/q review, but the model work is already complete; `q`
stores only the unreviewed proposal ids in the ledger so the next fetch resumes
without paying again or repeating settled rows. Submission fingerprints each
record's prompt inputs, so edits made while the batch is out cannot be
overwritten by its older answer. A machine-owned recovery journal preserves the
full submission descriptor across a failed initial ledger save and preserves
accepted meanings plus the exact retry set across the records/ledger handoff.
Fetch reconciles that journal before fingerprint checks, so meanings that
already landed cannot turn their own provenance into a stale discarded row.
Review hardening made journal replacement and removal compare-and-swap safe:
two fetches of the same batch cannot overwrite or clear one another's distinct
accepted/retry state.*

Depends on: M6.3
Files: `src/japanese_anki/enrich.py`, `src/japanese_anki/cli.py`.

The pilot's residual errors were all *source gloss quality* — すぐ carried
"almost" (a sense of もうすぐ), 一杯 "a full container" (the counter, not
the な-adjective). That is what this pass is for, and it asks about one
record at a time, which does not fit 688 of them. `--ai` already has
batch mode; this does not.

- Batch submit/fetch for the polish pass, or a staging-file route like
  the one `--ai` takes above fifty records.
- Keep the one-at-a-time path: it is right for a handful, and `q` stopping
  the spend is a real property.

### [~] M6.6 claimed codex/main 2026-08-12 — The remaining 688 Yotsubato records

*Progress 2026-08-12: batch 02 (source rows 21–40) is complete — 20 promoted,
dictionary- and AI-enriched, meaning-polished, voiced, reviewed clean, and
exported. Forty unique source rows now have committed archives, leaving 668.
The review gate corrected five dictionary/source mismatches and removed three
examples that could not reach a card field. A live polish request also exposed
an Anthropic content-filter error escaping as an SDK traceback; request errors
now become clean per-record warnings, so a later row cannot discard proposals
already accepted in the same pass.*

Depends on: M6.3, M6.4, M6.5
Files: `data/staging/anki-yotsubato-volume-1-reading-pack-vocab.yaml`,
`data/decks/yotsuba.yaml`.

The staging file is written and committed; eighteen of its records are
promoted, enriched, voiced, gate-passed and built. Measured per record
from that pilot: ~1.5 jpdb lookups, ~1 Opus call for sentences, ~0.5
Haiku calls for adjudication, ~1 review call, ~2.8 audio clips, and
**about one card in six needing a human** — almost all of it now source
gloss quality rather than janki defects.

- Promote in batches rather than all at once: `janki promote` archives
  the file when nothing is left held, and a 688-record diff is not review.
- Expect the deck to need `exclude_tags`/`include_tags` upkeep as it
  grows — `verbs.yaml` and `yotsuba.yaml` split on the `anki` tag, and a
  record in both pools has to be assigned to one by hand (ある and いる
  already were).

### [x] M6.7 Ship the review hooks with the repository

*Done 2026-08-12. The reviewer now lives at tracked
`scripts/janki-review.sh`, with its project-specific review instructions inline
so a fresh clone does not depend on the ignored `.claude/agents` directory.
Tracked post-commit and pre-push shims are installed by bootstrap (or the
standalone idempotent installer), which migrates the old janki hooks but refuses
to overwrite an unrelated local hook. The per-clone `.claude/hooks/DISABLED`
marker remains the kill switch. Failure-path coverage also exposed and fixed a
watchdog child that could outlive the reviewer and hold output pipes open until
the full timeout.*

Depends on: —
Files: `scripts/` (new home), `.git/hooks/*` (shims), `AGENTS.md`.

`.claude/` is gitignored, so `janki-review.sh` and both git hooks exist
in one clone and nowhere else: a fresh clone gets no gate at all, and the
failure-reporting fix (a review that dies now writes `VERDICT: ERROR`
instead of looking like one still running) travels with neither.

- Move the script somewhere tracked; keep `.git/hooks/*` as thin shims.
- Keep the kill switch: `.claude/hooks/DISABLED` turns both hooks off and
  says so on every commit and push. **Currently in place** — delete that
  file to re-enable.

---

## Milestone 7 — Deck-driven hardening (the repository learns)

The Yotsubato rollout proved the useful half of the loop: a real deck exposes
failures that synthetic unit tests did not, and a systemic failure can become a
deterministic rule instead of costing a human again. What is missing is the
durable protocol around that behavior. Today an agent can fix the current card,
write an ordinary test, or improve a prompt, but nothing requires it to say
which kind of lesson it learned, preserve a minimal reproduction, replay every
past lesson, or prove that an automatic repair cannot damage a different card.

**“Self-healing” has a deliberately narrow meaning in this milestone.** After a
systemic failure has been diagnosed once, a later run must either avoid it,
repair it with a named deterministic transformation whose pre- and
postconditions pass, or stop with an actionable finding. It does not mean model
weight updates, cross-run hidden memory, or allowing a model to rewrite curated
Japanese unattended. The repository is the memory: cases, fixtures, rules,
repair functions, prompt versions, decisions, and tests all travel with it.

Six boundaries apply to every M7 task:

1. **Classify before generalizing.** A source-specific correction (this gloss is
   wrong, this handwritten kana is ambiguous) stays a curated edit. A systemic
   correction (this layout loses the second column, this parser always chooses
   the wrong POS) becomes a finding and a regression case. Agents must not turn
   one ambiguous example into a global rule.
2. **Evidence before automation.** Automatic repair is default-deny. A repair
   needs a stable code, a narrow precondition, an invariant explaining why
   meaning and identity are preserved, a postcondition, an idempotence test, and
   both a positive and a near-miss fixture. Stored curated records are never
   changed merely because a new repair exists.
3. **No model grades itself.** Model-reported row counts, confidence, and
   omissions are useful evidence, not ground truth. Pilot oracles come from a
   human inventory of the source. Semantic judgements remain reviewable.
4. **Keep private source material private by default.** `data/inbox/` remains
   the immutable evidence. A full PDF or photo is not copied into the regression
   corpus merely because it found a bug. Cases use a minimized, cropped,
   redacted, or synthetic fixture; committing source pixels requires an explicit
   redistribution declaration. A source fingerprint and locator preserve the
   link back to private evidence without duplicating it.
5. **Source evidence is not teaching content.** A source excerpt proves where a
   target occurred. It is not automatically a suitable example sentence. A
   model-extracted meaning, part of speech, or example stays provisional until
   the pipeline binds it to dictionary evidence or an explicit curation
   decision. Schema validation, staging review, target-oracle approval, and
   source-context review do not imply approval of every learner-facing field.
6. **The final AI review is a residual check, not the source of truth.** Janki
   must establish identity, provenance, dictionary agreement, field ownership,
   furigana, romaji, register, audio eligibility, and other mechanically
   decidable facts before this review. The final reviewer can find semantic or
   naturalness errors that remain. Give it the canonical study record and a
   compact authority summary, not the full PDF, image, or extraction history.
   It does not reconstruct the source, repair records, establish field
   authority, or replace deterministic validation. A repeated late finding is
   evidence of an earlier pipeline defect and must move to that earlier
   boundary.

Lane map: {M7.1} → {M7.2} → {M7.3} → {M7.4, M7.5} → {M7.6A} →
{M7.6T} → {M7.6B, M7.6C} → {M7.7} → {M7.W}.

### [x] M7.1 The hardening protocol and safety contract

> *Superseded 2026-08-15. The protocol this milestone built — `docs/HARDENING.md`,
> the findings-and-cases corpus, the systemic-defect completion rule — was
> deleted by M8.4, which replaced it with the ordinary loop: a failing test
> first, then the fix. The **safety contract** survives and is enforced in
> code: the repair allow-list, the identity-field prohibition, and the
> approvals the owner keeps. Read `AGENTS.md` for the live version; the body
> below is how it was built, not how it works.*


*Done 2026-08-12. Added the deck-defect workflow and self-healing limits in
`docs/HARDENING.md`. Added the systemic-defect completion rule, exact automatic
repair allowlist, identity boundary, and user-only authority rules to
`AGENTS.md`. M7.1 defines behavior only. The catalog, cases, repair registry,
and live evaluation controls remain in their dependent tasks.*

Depends on: —
Files: new `docs/HARDENING.md`, `AGENTS.md`,
`docs/IMPLEMENTATION_PLAN.md` (task completion note only).

Write the procedure an agent follows whenever building or reviewing a real
deck exposes a defect. This is a behavior contract before it is a CLI:

1. Reproduce the observed behavior and preserve the source fingerprint,
   locator (page/row/region or note/field), pipeline stage, and exact command.
2. Classify the correction as `content-specific` or `systemic`, with a short
   reason. When uncertain, choose content-specific and keep it staged.
3. For a systemic defect, open a finding and make the smallest permissible case
   that still fails: a redistributable minimized fixture or a fingerprinted
   private-inbox reference. Do not put an entire owner-provided work into a
   fixture.
4. Fix the earliest production boundary that can state the rule reliably. A
   prompt change is appropriate for model behavior, but it still needs an
   offline case for downstream handling and a live evaluation case for the
   prompt itself.
5. Add or extend a deterministic validator when the same fact can now be
   stated without judgement. Add a repair only within the safety boundary
   below; otherwise emit a proposal or hold the record.
6. Run the new case, the entire accumulated corpus, and `make gates`. Close the
   finding only when it links to the passing case and the production fix.

Pin the repair boundary in both docs. The exact safe-repair field allowlist is
`furigana`, `romaji`, `examples[*].furigana`, `examples[*].romaji`, `audio`,
`image`, and `frequency_rank`. A repair may update one of these derived or
presentation fields only when its evidence predicate proves the result. The
namespaced `source.raw_fields["janki_repairs"]` provenance entry is a required
side effect, not a repair target. All other record and source fields are
default-deny. This includes `id`, `expression`, `reading`, meanings,
part-of-speech/verb classification, transitivity, pitch and audio-accent data,
Japanese or English example text, conjugations, tags, usage notes, and unknown
source fields. Existing identity (`id`, `expression`, and `reading`) is never a
repair proposal either: it determines the Anki GUID, so a finding there stays
blocked and names the manual migration/history decision it needs. A new
candidate's identity is still reviewed in staging before its first promotion.
For the remaining protected content fields, a proposed change uses M7.5's
fingerprinted, field-scoped staging diff and explicit human acceptance path; an
ordinary `janki promote` must never consume it. A future task may widen the
allowlist only by amending this contract with a concrete invariant and
adversarial fixtures.

Add an AGENTS.md rule that a systemic defect found during a deck task is not
complete when only the current data row is corrected: it must follow this
protocol or be recorded as an open/deferred finding. Conversely, require a
content-specific edit to stay content-specific rather than manufacturing a
general validator from one observation. Add a second owner-authority rule. The
user owns live-eval consent, redistribution approval for owner-provided source
material, human unit-oracle acceptance, manual coverage acceptance,
repair-proposal acceptance, an ambiguous new-identity resolution,
existing-identity migration, `accepted-risk` approval, and baseline acceptance.
An agent must not infer, generate, or set these values to complete a task. It
must not answer an interactive approval prompt as the user.

For `live_eval_consent`, the user must explicitly approve the exact case ID,
source fingerprint, provider, resolved model IDs or model scope, purpose, and
stored reason. Redistribution approval must name the exact artifact fingerprint
and license or other redistribution basis. Unit-oracle acceptance must name the
exact source fingerprint, oracle ID, normalized oracle-content fingerprint,
oracle type, and selection rubric when present. Manual coverage acceptance must
name the exact source and coverage-block fingerprints, every mismatch or
unmeasured disposition accepted, and the reason. A repair-proposal decision
must name the exact proposal-entry fingerprint and answer. An ambiguous
new-identity resolution must name the exact source fingerprint and locator,
expression, chosen reading, and reviewed evidence. An existing-identity
migration must name the exact old/new identity and review-history plan.
`accepted-risk` approval must name the exact finding ID, normalized risk-content
fingerprint, risk, and reason. Baseline acceptance must name the exact report
and acceptance-input-manifest fingerprints and diff. An agent may prepare an
oracle draft, but it cannot approve it. It may record only an exact user
approval. It must not widen its scope or treat a broad task request as one of
these decisions. If approval is absent, the value stays false/missing and the
agent stops or selects a source that does not need that approval.

Tests: documentation and link checks only. Each later task adds its schema and
executable checks at the first boundary that uses the approval.

### [x] M7.2 Findings catalog + `janki harden status`

> *Deleted 2026-08-15 by M8.4. There is no `janki harden`, no `hardening.py`,
> no `quality/` directory. The catalog's durable content — what each finding
> actually pinned — was migrated into `tests/`, and the five gaps that had no
> test are named in M8.4's own record.*


*Done 2026-08-12. Added strict, duplicate-key-safe schemas for the systemic
finding catalog and pilot reports. Added the read-only `janki harden status`
text and deterministic JSON reports. The status reports open and deferred
findings, recurrences, repair false positives, incomplete pilots, correction
counts, and empty source-archetype cells. Accepted-risk approvals bind the
exact owner decision and normalized risk content. Added the seed catalog,
pilot authoring guide, agent rules, and schema and CLI tests. Case and oracle
target resolution remains in M7.3 as planned.*

Depends on: M7.1
Files: new `src/japanese_anki/hardening.py`, new `quality/findings.yaml`,
new `quality/pilots/README.md`, `src/japanese_anki/cli.py`, `AGENTS.md`,
new `tests/test_hardening.py`.

Make the lessons and rollout evidence machine-readable without turning them
into another opaque ledger.

- `quality/findings.yaml` is human-readable and committed. One systemic finding
  has a stable slug, state (`open`, `fixed`, `deferred`, `accepted-risk`),
  pipeline stage, source archetype(s), concise symptom, invariant violated,
  evidence references (content fingerprint plus locator; no absolute paths),
  linked case IDs, and the production fix or deferral rationale. `fixed`
  requires at least one declared case ID and fix reference; existence and the
  reverse link become enforceable when M7.3 defines case discovery.
  `accepted-risk` requires a repository-owner approval and reason. The approval
  binds the finding ID and a normalized risk-content fingerprint. That
  fingerprint covers the pipeline stage, source archetypes, symptom, violated
  invariant, evidence, linked cases, stated risk, and reason. It excludes the
  state and approval block, so it is not circular. A missing approval or any
  later content change makes `accepted-risk` invalid. Unknown keys and malformed
  link IDs are errors rather than being discarded. Content-specific corrections
  are counted in pilot reports but do not bloat this catalog.
- Each `quality/pilots/<id>.yaml` records the source fingerprint and archetype,
  whether redistribution is allowed, its unit-oracle ID, repeatable-eval case
  ID, extracted, omitted, held, promoted, and rejected counts,
  content-specific and systemic correction counts, repair applications,
  false-positive repairs, model and prompt fingerprints, linked finding IDs,
  and whether the deck completed extract → review → promote → enrich → audio →
  final review → build. M7.2 validates the link syntax; M7.3 validates the
  targets once it owns their schemas. These are measurements, not generated
  packages; `dist/` remains disposable.
- `janki harden status [--format text|json]` validates both document shapes,
  reports open/deferred findings, recurrences, repair false positives,
  incomplete pilots, and which source-archetype cells in M7.6 remain empty.
  Case/oracle link resolution and corpus coverage are added in M7.3, after the
  target format exists. Paths in output are repository-relative.
- Parsing is strict and deterministic. Dates and prose do not participate in
  finding identity. Status is read-only; agents edit the reviewed YAML rather
  than gaining a command that silently rewrites decisions and comments.

Tests: valid/invalid documents, unknown fields, path traversal, duplicate IDs,
malformed link IDs, the requirements for each terminal state, missing or stale
accepted-risk approval, deterministic JSON, and a content-specific pilot edit
that is counted without becoming a systemic finding. Do not create a placeholder
case parser merely to make M7.2 appear to resolve links before M7.3.

### [x] M7.3 Minimized case bundles + offline replay

> *Deleted 2026-08-15 by M8.4. `make gates` is `lint test` plus one sample
> deck build; the replay step and the 25 gating cases are gone. The split M8.4
> measured was 18/2/5: eighteen redundant with existing pytest tests, two that
> observed no production behaviour at all, and five carrying real coverage that
> became ordinary tests before the corpus went.*


*Done 2026-08-12. Added strict human-oracle and minimized-case schemas,
fingerprinted and symlink-safe fixture loading, exact owner-approval binding,
reciprocal case/finding/pilot/oracle checks, and six registered offline
production-boundary runners. Added `janki harden replay`, made the full gating
corpus part of `make gates`, and seeded four fixed Yotsubato regressions. The
offline path uses canned structured responses and can hash a private source,
but it does not decode or send private source content.*

Depends on: M7.2
Files: new `quality/cases/README.md`, new `quality/cases/` fixtures,
new `quality/oracles/README.md`, new unit-oracle fixtures,
new `src/japanese_anki/hardening_replay.py`, `src/japanese_anki/cli.py`,
`src/japanese_anki/hardening.py`, `quality/findings.yaml`,
`tests/test_hardening.py`, `tests/test_hardening_replay.py`, `Makefile`.

Give every systemic lesson a portable reproduction that exercises production
code and can run without a network key.

- A human unit oracle lives at `quality/oracles/<oracle-id>.yaml` and pins one
  source SHA-256. An `exhaustive` table/list oracle inventories ordered, unique
  page/section/ordinal unit keys, normalized-context fingerprints, and expected
  dispositions. A `selection` prose oracle inventories the human-chosen target
  identities and locators plus its selection rubric, but never claims the rest
  of the page is accounted for. The oracle contains the M7.1 owner-approval
  record bound to the source fingerprint, oracle ID, oracle type, and normalized
  oracle-content fingerprint. That content fingerprint covers canonical oracle
  data and excludes the approval block, so it is not circular. An unapproved
  oracle is a draft. It cannot bind a case or pilot, satisfy a replay gate,
  authorize promotion, or enter a live baseline. Its manifest schema and safe
  path/hash loader land here so cases, pilots, and status share one definition.
  M7.4 teaches extraction to produce and compare these facts; M7.3 only validates
  and replays canned facts without making a model call.
- One case lives at `quality/cases/<case-id>/case.yaml` beside only the minimal
  artifacts it needs. Its `purpose` is either `regression` or `coverage`: a
  regression case must link one systemic finding; a coverage case must link one
  pilot and its human unit oracle and does not invent a finding merely to satisfy
  the schema. The manifest also names a registered runner (never a shell command
  or import path from YAML), pipeline boundary, source archetype, fixture hashes,
  oracle, and redistribution status.
- A model-boundary case has exactly one source form. `bundled_fixture` names a
  minimized file inside the case and requires `redistributable: true`.
  Owner-provided or source-derived material also requires the M7.1
  redistribution-approval record bound to the exact bundled artifact hash and
  stated redistribution basis. The manifest loader rejects missing or
  mismatched approval. An agent-created synthetic fixture records that basis
  explicitly and must not contain source-derived pixels.
  `private_inbox_ref` names a repository-relative immutable file under
  `data/inbox/`, pins its SHA-256 without copying its pixels, requires
  `redistributable: false`, and carries a separate repository-owner
  `live_eval_consent` record. The record contains the case ID, exact source
  fingerprint, provider, user-approved model IDs or model scope, purpose
  `hardening-eval`, approval date, and the user's reason. Missing consent means
  false. Only the user can grant it under M7.1's owner-authority rule. An agent
  cannot write `true` or widen the model scope on the user's behalf. Consent
  permits sending that fingerprint only to the approved destination and for the
  approved purpose. It does not permit redistribution. A changed source, case
  ID, provider, model scope, or purpose invalidates the consent. Reject absolute
  paths, `..`, symlink escapes, paths outside those two roots, hash mismatches,
  unknown runners, undeclared bundled files, and consent data that does not
  match the case and source.
- Initial runners cover the boundaries real decks have already stressed:
  candidate-response → staging records, staging read/rewrite/promote with fake
  lookup evidence, record validation/QC, and finished-record render/build. Add
  two enrichment runners that the seeded cases require: canned jpdb
  parse/lookup plus KANJIDIC evidence → production dictionary enrichment, and
  canned AI structured response plus sentence-parse evidence → production AI
  enrichment outcome **and command-level changed/no-writable-change reporting**.
  The first owns the POS-precedence and impossible per-character-furigana cases;
  the second owns the no-writable-change case. The validation/QC runner owns the
  missing-separator case. Every runner must call production functions; a
  test-only copy of normalization or reporting logic can agree with the same bug
  and is not a regression test.
- A model-boundary case may keep a canned structured response so routine replay
  proves how janki handles that response. If the failure was the model's
  extraction itself, the case uses one of the two source forms above and a human
  oracle for M7.7. Offline replay does **not** claim the current model would read
  the pixels correctly and never decodes or sends a private visual fixture;
  manifest validation may stream its bytes only to verify the pinned hash.
- Oracles compare structural facts exactly—source-unit accounting, record IDs,
  held rows, diagnostic codes, field changes, and rendered invariants. Do not
  golden-test arbitrary model prose when the contract is semantic.
- `janki harden replay [CASE...] [--format text|json]` runs every fixed case by
  default and exits nonzero on a mismatch. Pytest parametrizes over the same
  discovery API, so `make gates` automatically replays the complete fixed
  corpus. An open finding may have a reproducing red case, but it is explicitly
  marked non-gating until the fix lands; closing it flips that case into the
  permanent gate.
- Extend `janki harden status` with the case/oracle knowledge M7.2 deliberately
  did not guess: validate both directions of finding, pilot, oracle, and case
  links; report dangling or mismatched links and case coverage; and refuse a
  `fixed` finding whose case is absent or still marked non-gating.

Seed the corpus by minimizing at least the already-understood Yotsubato defects:
POS precedence, impossible per-character furigana, missing furigana separator,
and an AI response that produces no writable change. Add their fixed finding
entries and reciprocal case links to `quality/findings.yaml` in the same task;
these are not new fixes, but they prove the bundle and catalog formats together
against known production behavior.

Tests: unit-oracle duplicate keys and malformed hashes; missing or mismatched
oracle approval and draft-oracle gate refusal; both case purposes and source
forms; reciprocal/dangling links; manifest safety; missing-consent default,
consent/source fingerprint binding, and changed-source invalidation;
provider/model/purpose mismatch; missing or mismatched redistribution approval;
hash checks; every runner; a test that every seeded finding names a compatible
runner; stable discovery ordering; open-versus-fixed gate behavior; and a
deliberate production mutant at each seeded boundary that makes its
corresponding case fail. No runner may make a live network call or read a
private visual source.

### [x] M7.4 Extraction accounting and prompt provenance

Depends on: M7.2, M7.3
Files: `src/japanese_anki/extract.py`, `src/japanese_anki/staging.py`,
`src/japanese_anki/promote.py`, `src/japanese_anki/cli.py`,
`src/japanese_anki/validation.py`,
`tests/test_extract.py`, `tests/test_promote.py`,
`tests/test_validation.py`, new hardening cases and unit-oracle fixtures.

Make “the model returned a plausible list” distinguishable from “the source was
accounted for.” This is strict for table/list material and intentionally
different for prose, where selection is judgement.

- Replace the table-mode response's bare candidate list with source units. Each
  unit carries a stable page/section/ordinal key, verbatim context, and exactly
  one disposition: candidate, duplicate, non-vocabulary, or unreadable.
  Candidate units link one-to-one to a candidate; every other unit gives a
  reason. Preserve repeated source rows as separate source units even when one
  is correctly classified `duplicate`; never create two canonical records with
  the same deterministic ID. Auto mode uses this contract for sections it
  identifies as tables; prose candidates retain `inclusion_reason` and are
  explicitly reported as coverage `unmeasured`, never “complete.”
- Use M7.3's `exhaustive` human unit-oracle format for table/list coverage.
  Define its context normalization once in production extraction code and
  compare the exact key/fingerprint/disposition set—not its length—so omitting
  row six while repeating row five cannot pass. A prose `selection` oracle feeds
  M7.7's repeatable quality metrics but does not turn prose extraction into an
  exhaustive promotion gate.
- `janki extract --coverage-oracle FILE` is repeatable for multi-input runs; each
  supplied oracle binds to exactly one prepared source by SHA-256, and duplicate
  or unmatched bindings are errors before the first paid call. A supplied
  oracle must also have a valid owner-approval record for its current normalized
  content fingerprint; a draft oracle is an error before the first paid call.
  Inputs without a supplied oracle follow the explicit `unmeasured` path below
  rather than borrowing another file's count or oracle.
  Persist the oracle ID/fingerprint and exact missing, unexpected, duplicate,
  context-mismatched, candidate, omission, and unreadable unit sets in a
  `coverage` block in staging metadata. Model-reported totals remain diagnostic
  only and can never satisfy or replace the human oracle.
- An oracle mismatch keeps the paid extraction as a staging file but marks
  promotion blocked. A reviewer resolves it by correcting the source units and
  corresponding staging records, rerunning against a corrected oracle, or
  recording the exact M7.1 manual coverage acceptance and reason. Its source and
  coverage-block fingerprints, accepted dispositions, reason, and approval
  evidence survive in `data/staging/done/`. A newly extracted table with no
  oracle is `unmeasured` and needs the same user decision before promotion.
  Prose selection remains reviewable but is not falsely treated as exhaustive.
  `janki promote` refuses an unresolved coverage block or a missing or
  mismatched approval record before mutating records or the ledger. Files
  written before this task remain valid with legacy coverage `unmeasured` and
  are not retroactively blocked.
- Record source SHA-256, mode, provider/model, response-schema version, and
  fingerprints of the system prompt, style guide, and user prompt in staging
  metadata. Never store an API key or absolute source path. This is sufficient
  to explain model drift without treating a model name as the full prompt.
- Give deterministic extraction/validation failures stable diagnostic codes;
  human-facing messages may improve without breaking case identity. Codes are
  additive output, not replacements for the current source filename and row or
  page detail.

Tests: exact complete coverage; an omitted row replaced by a duplicate or
invented row; missing/duplicate/context-mismatched/unreadable units; incorrect
model self-count; source/oracle hash mismatch; multi-input binding before spend;
blocked promotion with no partial mutation; reasoned acceptance/archive;
missing or mismatched coverage approval; legacy staging compatibility; mixed
table/prose reporting; prompt fingerprint changes; and no secrets or absolute
paths in metadata.

### [x] M7.5 Named safe-repair registry + staged proposals

> *Half deleted 2026-08-15. The **registry** survives and is the live
> mechanism — default-deny, seven allowed fields, identity fields refused at
> three points (`repairs.py`). The **staged proposal** half — proposal
> documents, the two-phase journal, the archive, `repair --propose`,
> `promote --accept-proposals`, and the `proposal-only` mode — was deleted by
> d7ee435 after M8.3 removed its only producer. `REPAIR_MODES` is now
> `("ingest-safe", "revoked")`. Everything below about proposals describes
> code that no longer exists.*


Depends on: M7.2, M7.3
Files: new `src/japanese_anki/repairs.py`, `src/japanese_anki/cli.py`,
`src/japanese_anki/io.py`, `src/japanese_anki/staging.py`,
`src/japanese_anki/promote.py`, new `tests/test_repairs.py`,
`tests/test_promote.py`, new hardening cases.

Turn repeated mechanical fixes into constrained production behavior without
giving an agent general write access to curated content.

- A repair declaration has a stable code and version, applicable pipeline
  phase, operating mode (`ingest-safe`, `proposal-only`, or `revoked`), allowed
  fields, declared record-input fields, pure precondition, transformation,
  postcondition, and provenance text. All three callbacks must read only the
  declared record-input fields and named external evidence. The registry
  enforces this rule. It gives each callback the same immutable input projection
  and immutable evidence object. The precondition receives the projected
  before-values. The transformation receives the same values. The postcondition
  receives those before-values and the planned output values, but not the
  complete record. Full-file validation remains a separate step. No callback
  receives an undeclared field, complete record, path, client, or live I/O
  capability. Applying a repair returns a field diff and evidence. It cannot
  silently decline after its precondition matches.
- `janki repair PATH` is check-only and prints applicable repairs, exact diffs,
  and the input revision. `--check CODE... --format json` selects an ordered
  repair set and also emits a **repair-plan fingerprint** over the input
  revision and canonical plan JSON. The JSON contains ordered code/version
  pairs, target record/field names, exact old/new values, evidence, provenance,
  and the fingerprint of the complete intended target-file bytes. Those bytes
  include all repair annotations and serialization changes. Plan construction
  runs every repair postcondition and validates the complete intended file
  before it emits the fingerprint.
  `--apply CODE...` recomputes and prints that same plan in one process. It
  captures the revision before display. It requires interactive confirmation
  and compare-and-swap writes against the captured revision. For a
  non-interactive caller, the same ordered code list and
  `--expected-plan FINGERPRINT` are required. Apply recomputes the complete plan
  and refuses before any edit if its fingerprint differs; matching the input
  revision alone is not sufficient. There is no unbound `--yes` path. After
  confirmation, apply writes the exact planned bytes. It does not call the
  transformations again. It then verifies the output fingerprint. Apply uses
  the atomic write path and records repair code/version in the namespaced
  `source.raw_fields["janki_repairs"]` annotation as canonical JSON.

  Check and apply accept only a regular, record-shaped file under the resolved
  repository's `data/normalized/` or active `data/staging/` root. They bind its
  canonical repository-relative path and require the strict expected schema.
  They reject every other root, proposal-shaped files, archive paths, absolute
  escapes, `..`, symlinks in any path component, non-regular files, and a file
  identity or revision that changes during safe open. These checks apply on
  every read and again before atomic replace. A proposal generator creates its
  output only through the same safe staging-path helper. It never accepts an
  output path from repair data.

  Apply never changes an existing ID. Check-only output may show a
  `proposal-only` repair, but direct apply accepts only a current `ingest-safe`
  declaration whose target fields are inside M7.1's automatic allowlist. A
  `proposal-only` repair can write only a proposal-shaped staging file. A
  `revoked` repair can do neither. No direct-apply confirmation can cross this
  mode or field boundary.
- Only repairs explicitly marked `ingest-safe` may run automatically, and only
  while a new candidate is being normalized before it becomes curated. The M7.1
  protected fields remain default-deny. An existing identity-field disagreement
  becomes a blocking finding, never a repair. A repair that wants to suggest a
  non-identity protected content change writes a **proposal-shaped** staging
  file. Its `proposals:` list contains one entry per record **and field** with
  code/version, the exact old/new values, evidence, and declared basis fields.
  A basis fingerprint covers the record identity, canonical target old value,
  all record fields that produced the proposal, and named external evidence.
  A proposal-entry fingerprint also covers code/version, record ID, target
  field, old/new values, basis fingerprint, and evidence. Duplicate entries for
  the same record/field are invalid. A bundle-level source-file revision is kept
  for audit. It does not make an unrelated field stale. Creating this file never
  touches the source record.
- Ordinary `janki promote` refuses a proposal-shaped file and directs the user
  to `janki promote FILE --accept-proposals`, so the current existing-wins merge
  cannot report a conflict and then archive/delete the only proposal.
  `--accept-proposals` is valid only for that shape: it reloads the normalized
  records and recomputes each basis fingerprint. It refuses a stale proposal.
  It requires the proposal code and version to match the current registered
  declaration exactly. It rejects unknown, changed, or revoked declarations.
  It also rejects identity, undeclared target fields, and undeclared basis
  fields. A rejected declaration stays visible and is marked stale; the
  proposal must be regenerated. The command displays each field diff and asks
  the user `y/n/q` for **each proposal entry**. An agent cannot answer this
  prompt under M7.1. The command collects all decisions before it writes. It
  then acquires the transaction lock set described below. Under the locks, it
  reloads the records, staging proposals, declarations, and external evidence.
  It rechecks every displayed old/new value, proposal-entry and basis
  fingerprint, declaration version and mode, and dependency between accepted
  entries. Any change makes it refuse before the journal or a target file is
  written. It refuses an accepted set when one accepted target changes a basis
  field of another accepted entry; the dependent proposal must be regenerated.
  It applies the remaining accepted fields in one records compare-and-swap
  transaction and validates the full result. Rejected and unreviewed entries
  stay in staging. They stay valid only if no accepted target is in their basis.
  Otherwise, they remain visible but are marked stale. Accepted entries go to
  the archive with repair provenance. This path explicitly carries the
  annotation onto the existing source record. It does not use ordinary
  existing-wins source merging.
- Proposal acceptance uses a small recovery journal with the M6.5 discipline.
  Before the records write, the command computes and journals all intended
  results. The journal stores the records input revision and exact intended
  records bytes and revision. It stores the archive input revision and exact
  intended archive bytes and revision. It also stores the staging input
  revision and exact intended staging bytes and revision. The intended staging
  bytes remove accepted entries and mark every dependent rejected or unreviewed
  entry stale. They preserve unrelated entries and human comments. The journal
  stores each stable proposal-entry fingerprint and the complete accepted
  archive payload. It also binds the exact repository-relative records,
  archive, and staging paths. This payload is sufficient to reconstruct all
  intended files. The journal has a marker, schema version, and transaction
  fingerprint over these targets, revisions, intended bytes, and accepted
  entries. Load rejects unknown fields, path escapes, and fingerprint mismatch.

  The write order is fixed: create the journal, write the exact intended records
  bytes, write the exact intended archive bytes, write the exact intended
  staging bytes, verify all three results, and remove the journal. Journal
  creation expects no unfinished journal for the same targets. A different
  unfinished journal blocks the command. Journal replacement and removal use
  the same lock and compare-and-swap discipline as its target writes. Removal
  is permitted only when the complete journal bytes still match what this
  process read. Each target file write also uses a compare-and-swap guard. An
  existing identical archive entry is a no-op. A different entry with the same
  fingerprint is an error. Computing the complete intended archive before the
  first write prevents lost entries. Fingerprinted upsert prevents duplicates.

  Normal execution and recovery acquire interprocess locks for the journal,
  records, archive, and staging paths in one canonical path order. They hold the
  complete lock set while they inspect a state and advance it. All other janki
  writers use the same per-path locks. Normal execution holds the set from the
  post-prompt revalidation through journal removal. A process crash releases
  the locks but leaves the journal. The compare-and-swap checks remain required;
  the locks do not replace them. Two transactions that share any target cannot
  create journals or advance their states at the same time.

  Recovery checks the records, archive, and staging revisions as one state
  machine. It accepts only the ordered states `(input, input, input)`,
  `(intended, input, input)`, `(intended, intended, input)`, and
  `(intended, intended, intended)`. Equal input and intended revisions count as
  both states. Recovery writes the next exact intended bytes, or verifies the
  final state and removes the journal. Any other revision combination, archive
  conflict, target-path mismatch, or changed journal makes recovery refuse and
  name the journal for manual review. Recovery never recomputes decisions,
  reruns a transformation, or treats a rejected or unreviewed entry as
  accepted.
- No force, generic `prefer-incoming`, or noninteractive confirmation flag
  bypasses either repair boundary. A future batch acceptance path would need the
  same field and proposal-entry fingerprints. It would also need an externally
  supplied approval artifact.
- Seed the registry by wrapping an existing proven mechanical correction, not
  inventing a new semantic rule for the sake of the framework. Furigana
  separator insertion is eligible only when the existing jpdb tokenization and
  reading-evidence predicates agree. If that cannot meet the contract, use a
  narrower derived-field repair and leave furigana as a proposal.
- Every repair ships with: triggering case, at least two non-triggering near
  misses, apply-twice idempotence, no-ID-change assertion, full-file validation,
  and a mutation that proves the pre- or postcondition matters. A single false
  positive in M7 pilots demotes that repair from automatic to proposal-only
  until a narrower precondition lands.

Tests also cover a file changed after check-only preview; unchanged input with a
changed repair version, order, evidence, or diff; a file changed after the
in-process confirmation but before replace; partial failure (original intact);
an outside-root path, parent traversal, symlink component, non-regular file, and
file-identity swap; an attempted undeclared input read; an undeclared write or
changed provenance that changes the intended output; multiple repairs in
deterministic order; canonical provenance; and a semantic proposal that reaches
staging without touching the source record. Undeclared-read tests cover the
precondition, transformation, and postcondition separately. Proposal tests cover
ordinary-promote refusal without pruning, duplicate record/field entries,
target and basis staleness, an accepted target used by another proposal,
unrelated and related partial acceptance, per-field accept/reject/quit,
protected identity fields, source annotation carry, and accepted-entry-only
archive behavior. Inject a crash after the records write, after archive upsert,
and after the staging write. Each rerun must produce one archive entry, the
exact intended records and staging bytes, no lost entry, and no second
application. A dependent rejected proposal must keep its stale marker after
recovery. An unrelated proposal must remain valid. Also test an unknown,
changed, or revoked repair version; a second transaction blocked by an existing
journal; a journal changed before removal; a records, staging, declaration, or
evidence change during the prompts; a `proposal-only` direct-apply attempt; two
transactions with one shared target; and every invalid recovery-state
combination.

### [x] M7.6A Pilot pair — born-digital structure

*Done 2026-08-13. Built a 12-unit native PDF table pilot and a 20-target
mixed-layout PDF pilot through extraction, oracle review, promotion,
dictionary and AI enrichment, safe audio generation, final semantic review,
and deck build. The pilots exposed ten systemic corrections. Each correction
has a fixed finding and a passing offline case. Both approved private-source
coverage cases remain available for M7.7 live evaluation. Generated `.apkg`
files remain outside version control.*

Depends on: M7.4, M7.5
Files: two new immutable source copies placed under `data/inbox/` by the input
pipeline (never hand-edited), corresponding staging archives, curated
records/deck definitions under `data/`, two `quality/pilots/*.yaml`, two unit
oracles and two purpose-`coverage` case bundles, plus new findings/regression
cases/tests exposed by the pilots.

Build two small, genuinely different source slices end to end: one native PDF
table/grid and one multi-column or mixed prose-and-table PDF. Inventory the
source before extraction. Target 10–30 useful records per source (or the whole
page when it contains fewer); the point is layout breadth, not another
hundreds-row rollout.

Complete extract → coverage review → promote → dictionary/AI enrichment →
audio → final review → build for both. Every human correction is classified.
A systemic correction must link an open finding before the fix and a passing
case before this task becomes `[x]`; a deferred systemic defect leaves this
pilot incomplete. Record content-specific corrections and repair outcomes in
the pilot reports. Do not commit `.apkg` files.

Every pilot source also produces one repeatable purpose-`coverage` case even
when it exposes no systemic defect. The case links the pilot's exact human unit
oracle and uses either a minimized redistributable source slice or the exact
fingerprinted `private_inbox_ref` with repository-owner live-eval consent. The
source slice must be the material actually used for that pilot, not a cleaner
substitute created after the fact. A pilot without a source that can be sent on
a later evaluation run remains incomplete; use another suitable source rather
than silently leaving an archetype out of the baseline. This completion rule
does not authorize an agent to grant consent: if the user has not explicitly
approved the exact private fingerprint, the agent must use redistributable
material or leave the task open. Regression cases for systemic findings are
additional and may point at the same minimized evidence.

### [x] M7.6T Trust-boundary repair — source evidence versus study content

> **Partly superseded by M7.6V (2026-08-14).** Every mention below of the
> learner-load bound — rule 4's "bounded learner load", the coverage line
> naming a learner-load hold, and the summary clauses about the batch path and
> `validate` — describes code that no longer exists. The bound was deleted on
> measurement: it held 0 of 155 examples, did not fire on 逡巡, 邂逅 or 憂鬱
> either, and read a segmentation that was not the sentence. This block is left
> as written because it records what M7.6T delivered; it is not a description of
> the current tree. Nothing else here is affected.

*Added 2026-08-13 after the camera pilot. This task blocks more M7.6B and M7.6C
source runs. The camera extraction found the approved identities and source
context, but the first final semantic review still found five content errors
and several quality notes across 12 records. The records were corrected, but
that result showed a design defect: the late AI review had become the main
teaching-content gate. Develop and verify this task with synthetic or offline
fixtures. Do not send private source material or run a paid review while this
task is in progress.*

*Completed 2026-08-13, entirely on synthetic and offline fixtures; no private
source bytes were sent and no paid review ran. Four findings opened with red
non-gating reproductions (`extraction-source-example-promotion`,
`enrichment-provisional-precedence`, `enrichment-reviewed-example-claim`,
`example-teaching-suitability`), then fixed and flipped gating slice by
slice: extraction keeps the excerpt as `raw_fields["example"]` evidence with
`examples` empty; promotion stamps `example_authority` for reviewer-placed
examples and enrichment pins only curated Japanese; extraction marks model
semantics in a value-bound `provisional_fields` and dictionary reconciliation
replaces or holds them with the change in the diff; `qc.example_content_holds`
gives validate and audio one shared teaching-content judgment, the AI pass
bounds learner load from canned-able parser data, and a shipping build reports
local failures before the review store answers. `docs/HARDENING.md` gained the
"Study-content trust boundaries" contract, including the legacy-record and
pre-boundary staging-file migration posture. Full gating replay (32 cases) and
`make gates` pass. The slice-5 local review cycle (ten offline finder angles)
then found and fixed real boundary defects before any live use: the merge now
carries every authority key with the field it describes, example acceptance
became explicit and fingerprint-bound (typed by the reviewer, bound by
promote) with unaccepted and legacy examples preserved-and-unpinned rather
than replaceable, mark-clears persist, the learner-load bound covers the
batch path and is visible to `validate`, reconciliation demands the exact
entry spelling (kana homographs, suru stems), and the fragment/register
checks gained the stem-morphology guards that keep 励ました, そば, and
何について？ out of the gate. The cycle ran four full ten-angle rounds; every
execution-verified finding was fixed and mutation-tested, through the final
round's residuals (the preserve-warning seam, the extract-to-extract merge
direction, the sentinel diagnostic scope, imperative について, the humble
いただきます auxiliary, stacked final particles, and replay known-set
normalization). Accepted, not fixed — all bounded, all recorded here so a
later pass can pick them up: gloss reconciliation issues one
lookup-vocabulary call per marked record on its first enrich (batchable; the
mark clears on settlement); an unresolvable provisional mark (kana homograph,
suru stem) re-parses per run until a person settles it, which is the "hold
until evidence or review" contract; shipping builds validate twice (CLI
ordering gate plus exporter backstop, ~2 ms measured); the learner-load known
set is parameter-threaded rather than derived in the absorb layer; rare
i-column noun endings (にじます) can false-positive the register gate; and
parse-backed register verdicts could eventually replace the surface formula
list. M7.6B and M7.6C can resume; their next paid semantic review is the
milestone measurement of this boundary.* *(Correction 2026-08-15: M7.6B and
M7.6C are cancelled and the paid semantic review was deleted in M8.2. The
sentence stands as history.)*

Depends on: M7.4, M7.5, M7.6A
Files: `src/japanese_anki/extract.py`, `src/japanese_anki/enrich.py`,
`src/japanese_anki/promote.py`, `src/japanese_anki/validation.py`,
`src/japanese_anki/models.py`, `src/japanese_anki/cli.py`, the ledger and
review-state modules, new offline quality cases and tests, and
`docs/HARDENING.md` if the trust contract changes.

The camera pilot exposed these trust-boundary failures:

1. Extraction copied a source excerpt into `examples` and thereby promoted
   evidence of occurrence into learner-facing study content.
2. Enrichment then described the extracted example as reviewed and required
   the model to preserve its exact Japanese, although no person or dictionary
   had accepted it as a teaching example.
3. Extracted meanings and parts of speech entered canonical records without a
   provisional authority state. Dictionary enrichment filled only empty
   fields, so a plausible model gloss could outrank stronger dictionary
   evidence.
4. Local validation checked structure but did not fully check teaching
   suitability. Sentence fragments, false register labels, examples above the
   intended learner level, unexplained collocations, and unsupported usage
   notes could reach audio and remain there until the final AI review.

Implement an explicit three-layer trust model:

- **Source evidence** records what the source contains: locator, excerpt,
  printed reading, image region, and extraction confidence. It can prove an
  identity or give review context. It is not a study example by default.
- **Provisional facts** are model or OCR claims that still need a named
  authority path. A dictionary match, an exact reviewed source assertion, or a
  human curation decision can resolve each claim.
- **Study content** is learner-facing data that passed its field-specific
  authority and validation rules. Only this layer can reach cards and audio.

Provenance that defines field authority must be committed with the durable
record or its committed review artifact. It must not exist only in the
operational ledger. Reuse canonical `source.raw_fields` provenance when it can
represent the contract without ambiguity. Otherwise, make the smallest
reviewed schema change. Do not infer that oracle acceptance, promotion, or a
valid record approves all semantic fields.

Make these production changes:

1. Stop automatic promotion of source excerpts into `examples`. Preserve the
   excerpt as source evidence or review context. Keep support for explicitly
   curated examples already present in staging. An explicit field-level
   acceptance can promote a source excerpt when it is also suitable teaching
   content.
2. Remove the false reviewed state from enrichment. If a record has no
   explicitly curated example, enrichment can generate a new pedagogic
   example. It must preserve an existing curated example unless the user
   requests a change.
3. Mark semantic extraction fields as provisional. For an exact identity
   match, dictionary reconciliation can replace only provisional meanings,
   parts of speech, and related facts, and must show the authority change in a
   reviewable diff. Preserve human-curated fields. Hold or propose a decision
   for near matches, absent entries, conflicting evidence, or a different
   reading.
4. Add a local teaching-example gate. Keep the existing furigana, romaji, and
   token checks. Also apply conservative checks for a complete generated
   example or an intentional utterance, agreement between the target sense and
   predicate or register, and bounded learner load based on parsed words,
   known-vocabulary data, and available frequency data. Hold when the result is
   undecidable. Do not build a broad Japanese grammar parser from the one
   camera source.
5. Keep usage notes optional unless they have an authority path. An empty note
   is better than an unsupported distinction. Do not require one model to write
   a note only so another model can grade it.
6. Apply the same content gate before audio. Do not voice an example that is
   held or fails local validation. Use the normal audio command so content
   addressing, ledger state, and stale-media detection remain authoritative.
7. Keep final AI review read-only and residual. Run it only after all local
   gates pass for a complete release candidate. It can block residual semantic
   or naturalness errors, but it cannot establish authority or mutate a
   record. Its normal request contains the canonical study record and a compact
   authority summary. It does not contain the original PDF or image. A narrow
   source excerpt can be included only when the check concerns a declared
   source-dependent assertion. After a correction, rerun only changed records.
   A second broad review needs explicit repository-owner approval.

Add findings and offline cases before the production fixes for at least the
automatic source-example promotion and provisional-semantic precedence bugs.
Tests must cover:

- source evidence that survives promotion while canonical `examples` stays
  empty, plus a positive case for an explicitly accepted source example;
- oracle approval and ordinary promotion that do not imply approval of an
  example, meaning, part of speech, note, or other learner-facing field;
- exact dictionary reconciliation of provisional fields, with near-match,
  human-curated, different-reading, absent-entry, and conflict cases;
- complete polite and casual examples, a noun fragment, a trailing connective,
  false register labels, an acceptable short utterance, and a learner-load hold
  driven by canned parser data;
- held examples that are not voiced, and an edited voiced example that cannot
  reuse stale media; and
- build readiness that reports local content failures separately from a
  missing final AI result, and a saved AI result that cannot hide a later local
  failure.

Do not rewrite existing curated records only because this task adds provenance.
Add a synthetic reproduction of the camera fragment pattern without private
prose or pixels. Complete linked findings, the full offline replay, `make
gates`, and a dry pipeline run without a live model. *(2026-08-15: M8.4
deleted findings and the replay; `make gates` alone is the check now.)* M7.6B and M7.6C can resume
after these checks pass. The next paid semantic review is a milestone
measurement of the repaired boundary, not part of the development loop.
*(Correction 2026-08-15: both milestones are cancelled and that review was
deleted in M8.2 — history, not a plan.)*

Use these small implementation and commit slices so another agent can resume
without a live source run:

1. Add the open findings, failing offline cases, and synthetic fixtures.
2. Separate source evidence from curated examples and add its provenance.
3. Add provisional semantic authority and dictionary reconciliation.
4. Add the local teaching-content, build-readiness, and pre-audio gates.
5. Update migration and operator documentation, run `make gates` (the full
   replay it names went with M8.4), then do one local code-review cycle until
   it is clean.

Each slice must pass its focused tests before commit. Do not run a paid
advisory review after a slice. Keep the review hooks disabled while the owner
cost pause is in effect. Do not remove or change the disable marker as part of
this task.

### [~] M7.6V Authority realignment — the LLM parses, the dictionaries enrich

> *Correction 2026-08-15: sentences below promise that the paid review is "the
> one this task keeps" and that deleted checks are replaced by "a prompt clause
> plus the paid review." M8.2 deletes the review subsystem; the duty those
> sentences assign to it belongs to the prompt templates (`docs/DESIGN.md`).
> The sentences stand as history rather than being rewritten.*

*Opened 2026-08-14, replacing three discarded drafts of a jpdb-side furigana
repair. Those drafts assumed jpdb should adjudicate the AI's readings; the
owner's decision is that it should not adjudicate anything. jpdb and KANJIDIC
answer questions about words and characters the LLM has already identified,
and the paid review moves to Opus 5 at extra-high effort with a completeness
gate. Measured live against jpdb, the installed SDK, and the real corpus. One
live model call is required (see slice 4); no paid semantic review is.*

Depends on: M7.6T
Files: `janki.toml`, `src/japanese_anki/claude_client.py`,
`src/japanese_anki/qc.py`,
`src/japanese_anki/enrich.py`, `src/japanese_anki/cli.py`,
`src/japanese_anki/config.py`, `docs/ENRICHMENT.md`,
`docs/DESIGN_V2.md`, `README.md`, and new tests. *(`docs/HARDENING.md` and
"new cases" were deleted with the corpus in M8.4.)*

**The line.** The LLM reads the source, segments it, assigns readings *in
context*, and writes the sentences. jpdb answers one question about a word it
is handed — what does the dictionary say, including its frequency rank —
and KANJIDIC one about a character. Neither judges the LLM's output. The paid
review is the only reviewer, and it reviews language, not mechanics.

**The evidence, measured.** jpdb's *sentence* tokenizer returns a wrong lexeme,
not merely a wrong split, on kana-heavy text: `ほんをよむ` resolves 本 to ほる;
`だれとすんでる` becomes とする / でる; `何回もしぬの` reads しぬ as する — three
of five real sentences. Its per-character furigana produced the one recorded
jukujikun break, which is why `enrich.py` already carries a KANJIDIC guard
against jpdb. `verify_example_furigana` flags 38 of 155 examples with **zero**
true positives. Recorded AI furigana *reading* errors: zero. Of 22 blocking
review findings only 11 touch fields the AI writes; 10 of those are
gpt-5.6-sol's, and three of the ten are camera excerpts M7.6T already fixed.
Opus authored 37 records and drew **no blocking finding** on a field it
wrote — but seven note-level ones across five records, including `第1話`
glossed only "Chapter 1" while an example uses it in the episode-of-a-series
sense. That is the same sense-mismatch class this task cites against
gpt-5.6-sol, and unlike the 手 case it is real: 手's meanings already listed
"move" and "tactic", so the finding quoted against it had been answered before
this milestone began. The model change narrows the failure class; it does not
close it, and the reviewer that catches it is the one this task keeps.

**Model change.** Set `enrich_provider = "anthropic"` and
`enrich_model = "claude-opus-5"`. Already wired: `cli.py:1475` selects
`claude_client.parse_call` for any non-codex provider, `config.py` accepts
`"anthropic"` and defaults that provider's model to `claude-opus-5`, and
`--batch-submit` already requires it. `enrich_reasoning_effort` ("ultra") is a
Codex knob already gated on `enrich_provider == "codex"` at `cli.py:1478`;
leave it. `docs/DESIGN_V2.md` and `docs/ENRICHMENT.md` document the Codex
default in prose and must be updated with it.

**Effort must be a parameter, not a constant.** `_request_body` is shared by
**five** passes, and one of them cannot accept `effort`:
`config.adjudicate_model` defaulted to `claude-haiku-4-5-20251001`, which
rejects it with a 400 (it now defaults to `claude-opus-5`, which accepts it) — and `adjudicate_reading` swallows every exception into
`return "unsure", ...`, so an unconditional `effort` would make adjudication
fail forever with no error surface. Give `_request_body` an `effort` parameter
defaulting to `None`, omit the key when unset, and thread it exactly as
`max_tokens` is already threaded. Resolve it from the **model**, not the
provider or the config: `--model` overrides the model alone, so a depth guarded
by anything else guards something the caller did not just override. The anti-drift guarantee is
unaffected — live and batch still build one body for the same call — and
`tests/test_enrich_batch.py:253` proves it.

**Raise the budget the review actually uses.** `max_tokens` bounds thinking
plus response, thinking is on by default on Opus 5, and `"max_tokens"` is
absent from `COMPLETE_STOP_REASONS`, so truncation is a hard failure. There is
no single budget to raise:

| pass | `max_tokens` | site |
|---|---|---|
| adjudication | `DEFAULT_MAX_TOKENS` (was 200) | `enrich.py` |
| patterns | `DEFAULT_MAX_TOKENS` (was 4000) | `patterns.py` |
| review, and its pitch recheck | `DEFAULT_MAX_TOKENS` (was 8000) | `review.py` |
| extract / enrich / polish / batch | 16000 | `DEFAULT_MAX_TOKENS` |

The pass this task exists to improve is the *tightest*, and its own comment
already records that 2000 cut a card off mid-verdict. Raising
`DEFAULT_MAX_TOKENS` alone does nothing for it. Size review's budget from a
measured card and raise `review.py:497` and `:540` in the same change.
*(2026-08-15: moot. M8.2 deleted `review.py`, so the tightest pass in this
table no longer exists and the row above is a record of what was measured,
not a site to edit.)*

*Measured 2026-08-14, extra-high effort, live:* five real cards read at 541,
980, 1017, 2662 and 4106 output tokens — a sevenfold spread leaving 8000 only
1.9x over the peak. Three enrichment calls on the same setting used 519, 638
and 687, so **reading is the expensive pass and writing is not**: enrichment
sits at 4% of its 16000 budget while the review sat at half of its 8000. Both
review budgets now follow `DEFAULT_MAX_TOKENS`; unused budget costs nothing,
and the cap is only ever reached by a card long enough to fail on.

**Streaming: keep the change, fix the reasoning and the call.** The earlier
claim that the SDK refuses a non-streaming call above ~16K is wrong. The guard
is a latency estimate — `3600 * max_tokens / 128_000 > 600`, i.e. **21,333**
tokens — `claude-opus-5` has no entry in `MODEL_NONSTREAMING_TOKENS`, and an
explicit `timeout` suppresses the guard entirely. So streaming is not the only
way to keep one identical budget on both paths; it is the way that does not
hold an idle connection open for minutes, and a bare `ValueError` from that
guard would escape `parse_call`'s `APIError` boundary as a traceback. Two
things the call shape requires:

- `messages.stream()` returns a `MessageStreamManager`, which has only
  `__enter__`/`__exit__`. `get_final_message()` exists on the `MessageStream`
  that `__enter__` yields, so the form is
  `with api.messages.stream(**body) as stream: response = stream.get_final_message()`,
  with **both** statements inside the existing `try`.
- httpx errors raised while iterating the body are not wrapped as `APIError`,
  because the SDK wraps only the initial send. Widen the boundary to catch
  `httpx.HTTPError` as well, or the failure this boundary exists to prevent
  arrives as a traceback.

`_request_body` is unchanged, so the batch path is unaffected — verified that
the Batches API accepts `output_config.effort` and any `max_tokens`. The fakes
in **both** `tests/test_claude_client.py` and `tests/test_enrich_batch.py:253`
need a `stream()` context manager; the latter is the only test enforcing the
shared-body invariant.

**Retiring the sentence oracle also retires the re-check and the adjudicator.**
`verify_example_furigana` has a second production caller —
`enrich.py:1665`, inside `recheck_furigana` — whose verdict is the only thing
that clears a `furigana_unverified` flag and whose `verdict.expected` is what
`adjudicate_reading` shows the adjudicator. Both exist solely to settle
jpdb-versus-AI reading disagreements, which this task stops raising, so both
go: the CLI flag, the refresh stage, `config.adjudicate_model`,
`tests/test_recheck_furigana.py`, and their entries in `README.md` and
`docs/ENRICHMENT.md`. That narrows the routes back from a flagged example to
the human `--accept` alone, which `ai-impossible-character-furigana`'s
invariant still permits — "until review" — but the narrowing must be recorded
in that finding's recurrences rather than left implicit.

Keep, and lift out before deleting anything:

- the `furigana_base`-versus-sentence check, which needs no dictionary and
  catches a model rewriting the sentence inside the furigana field. It is
  currently unreachable behind **two** short-circuits, not one: `impossible`
  and `parse is None` both precede the call, so a card with an impossible
  reading never gets its base check either;
- `impossible_character_furigana`, the notation checks, and
  `example_content_holds`, all unchanged.

**When the parse goes, its disjunct goes with it.** `apply_ai_result` flags an
example when `impossible or parse is None or not verify_example_furigana(...)`.
Once no check consumes a parse, `parse` is always `None`, so leaving that
disjunct in place flags **every** example and `janki audio` refuses the whole
collection — reintroducing, corpus-wide, the silent-sentence defect this task
exists to fix. Delete it in the same change, making
`impossible_character_furigana` the sole source of `UNVERIFIED_KEY`, and record
that change of meaning in a finding. Verified in a scratch tree: with the
disjunct replaced by `if impossible:` alone, replay is 31/32 and
`ai-impossible-character-furigana` still passes on its existing oracle.

**Learner load: deleted, not migrated.** *This paragraph originally designed
the migration — the LLM supplying words and jpdb ranking them, memoized per
word, compared against jpdb's returned `spelling` so 有難う/ありがとう drift
could not count a known word as unknown. None of it was built.* Measured
instead: the bound held 0 of 155 examples and did not fire on 逡巡, 邂逅 or
憂鬱 either, because its limit was rank 20000. It was deleted on the owner's
decision — see slice 6 and the comment on `example-teaching-suitability` in
`quality/findings.yaml`. Nothing below about `_learner_load_excess`,
`known_expressions` or the `example-learner-load` hold describes code that
exists.

**The review completeness gate.** ~~Twenty-four lines of design for a gate on
`command_review`.~~ *(Moot 2026-08-15: M8.2 deleted `janki review`,
`command_review`, `_shipping_records`, `_refuse_unreviewed` and `enrich
--accept`, and M8.3 deleted the flag subsystem the gate was to consult. The
waste it measured — 181 reads for 97 records, 71 of 84 superseded reads
carrying no blocking finding — is why the subsystem went rather than gained a
gate. The placement anchor it gave, "after the `--accept` branch at
`cli.py:4468`", now lands inside `promote --accept-coverage`; do not follow
it.)*

**Coverage the cases do not have.** No gating case supplies
`sentence_responses` or `known_expressions`, and there is no `audio` runner in
`hardening_replay.RUNNERS`, so the audio coupling is not observable by replay
at all — the "flagged is not voiced" half of `ai-impossible-character-furigana`
is proved only by pytest. (The learner-load half of this paragraph is moot: the
bound was deleted rather than re-sourced, so there is no input to change.)
`ai-existing-example-annotations` is the only case in the corpus
where "no parse ⇒ unverified" is observable; re-pinning its oracle deletes the
last trace of that rule, so add a case pinning the surviving rule first.
Oracles are regenerated by hand — `case.yaml` pins each fixture's sha256 and no
`harden` subcommand rewrites them. *(2026-08-15: the corpus, its oracles and
`janki harden` were deleted by M8.4. "Add a case" now means add a test in
`tests/`, and `ai-existing-example-annotations`'s coverage was ported to
`tests/test_enrich_ai.py` there.)*

Use these slices, each a finding plus a reproducing case plus a fix, ordered so
`make gates` is green at every commit boundary:

1. ~~Lift the base-versus-sentence check out from behind both short-circuits.~~
   *(Moot: M8.3 deleted `verify_example_furigana`, `furigana_base`'s checking
   caller, `impossible_character_furigana`, `example_content_holds` and
   `spilled_furigana_groups`. `qc.py` is three notation functions now —
   `furigana_pairs`, `furigana_reading`, `settle_example_romaji` — and
   none of them judges.)*
2. ~~The review completeness gate.~~ *(Moot — see above.)*
3. `effort` as a parameter, `"xhigh"` wherever the model accepts it; the
   streaming call shape and the widened exception boundary; fakes updated in
   both test files.
4. One live `enrich --ai` call at `xhigh` on a named record to read `usage`,
   then raise `DEFAULT_MAX_TOKENS` from it. *(The companion pointer at
   `review.py:497`/`:540` is gone — M8.2 deleted that module.)*
5. Switch `enrich_provider`/`enrich_model`; update `DESIGN_V2.md` and
   `ENRICHMENT.md`.
6. *Done 2026-08-14 by deletion, on the owner's decision.* The learner-load
   bound is gone rather than migrated, and with it `_learner_load_excess`,
   `_LEARNER_LOAD_ALLOWED`, `_LEARNER_LOAD_RANK_LIMIT`, `LEARNER_LOAD_HOLD_KEY`,
   `_known_expressions`, the `load_held` channel and its warning, the
   `known_expressions` parameter through every caller, `qc`'s
   `example-learner-load` hold, and `io`'s carrying of the flag.

   Measured before deciding. It held **0 of 155 examples across 97 records** —
   and it does not fire on genuinely hard vocabulary either: 逡巡 (rank 9100),
   邂逅 (16600) and 憂鬱 (6900) all pass, because the limit was rank 20000, a
   level a model writing from the style guide never reaches. It was not broken
   — canned ranks of 45000, null and 52000 held correctly — it was calibrated
   to a threshold nothing meets.

   Its input was wrong as well, which is what slice 6 was originally to fix:
   reading jpdb's sentence segmentation, it counted `とする` on
   `今、だれとすんでるの？` — a word absent from the sentence — while 住む, the
   word actually inside すんでる, went uncounted. Migrating it would have cost N
   jpdb word lookups per sentence to feed a check that had never held anything.

   **This narrows `example-teaching-suitability`'s invariant**, whose
   learner-load clause is removed. Recorded here and in the commit rather than
   in the finding, because `recurrences` takes only fingerprint/locator pairs
   and `janki harden status` cannot see prose change either way. What replaces
   the bound is the paid review, which judges whether a sentence is too hard as
   a language question rather than by rank — the line this whole task draws.

7. *Done 2026-08-14.* Retired together: the oracle and its now-orphaned
   `FuriganaVerdict`/`_render`/`_comparable`, `recheck_furigana`,
   `adjudicate_reading`, the `parse is None` disjunct, the `jpdb_client`
   plumbing through the whole AI path (live and batch), the CLI flag's jpdb route and `--no-adjudicate`
   (`--accept` was kept, and is now a standalone `enrich` pass), the `recheck` refresh stage,
   `config.adjudicate_model`, and the replay runner's `sentence_responses`
   support, which no case used. `--ai` now needs no jpdb key: `_JPDB_STAGES` is
   `{"jpdb"}` alone.

   Both terms of the flag are pinned, and each needed its own fixture. Dropping
   `rewritten` is caught by nine tests. Dropping `impossible` **survived** at
   first: a record that already has examples takes the merge branch, which
   re-derives the flag from `outcome.impossible_furigana`, so the two routes
   mask each other exactly as slice 6's measurement warned. A record with no
   stored example has no merge to re-derive from, and pins the verdict itself.

   **Two checks were lost**, the second in two shapes. A wrong-but-attested reading that KANJIDIC lists
   is the deliberate one — no local rule can decide it, and it is left to a
   reader. The second was not intended and is worth recording: the oracle was
   the only rule that read a **full-width space** as a separator fault, and
   Anki's filter separates on the ASCII space alone, so `毎日　話[はな]します。`
   drops 毎日 from the reading, the romaji and the sentence audio. Measured:
   `spilled_furigana_groups` catches `お　茶[ちゃ]` — but for the reading, not
   the space (`お茶[ちゃ]` without any space gets the same verdict, and
   `お　茶[おちゃ]` with the same space gets none). Only the subset that also
   spills a reading survives. The same loss covers a group with *no* separator
   before a Han-initial run — `毎日話[はな]します。` — which `spilled_furigana_groups`
   documents as its own known gap. Both are recorded as the open finding
   `furigana-full-width-separator`, and both are pinned as tests asserting the
   wrong behaviour until the separator rule lands.

   The merge branch's aggregation carries both routes and both are now pinned
   too — dropping either the impossible or the rewritten term fails exactly one
   test. `*landed_unverified` beside them is **dead**: it was the sole carrier
   of the `parse is None` flag, and post-slice-7 `unverified` is a subset of
   `impossible ∪ rewritten`, so removing it passes the whole suite. Left in
   place rather than removed at the end of a long change, because taking it out
   also changes `_fill_existing_example_annotations`' return shape; it is the
   one known-dead line in this diff.

   `ai-existing-example-annotations`'s oracle changed and was re-pinned. Its
   example is all-kana with furigana identical to its sentence, so it had been
   flagged solely by `parse is None`; `unverified_ids` and `warning_count` are
   now empty and zero. The input fingerprint is untouched, so the finding's
   evidence still resolves.
8. *Done 2026-08-14, taken ahead of 6 and 7 because it removes a parse
   consumer.* Measured over all 158 examples carrying furigana, including the
   five regenerated with Opus: `repair_from_word_boundaries` changed
   **nothing**. Deleted, along with `_words_of`, which had no other caller. It
   carried no finding and no gating case, so this retired nothing.
   `spilled_furigana_groups` still *reports* the defect and its validation-qc
   case still covers it; what is gone is the silent repair, whose input was
   jpdb's sentence boundaries — measured wrong on three of five colloquial
   sentences. A repair that has never fired, driven by boundaries that are
   sometimes wrong, could only ever introduce the error it was written to
   prevent. `_learner_load_excess` was then one of two consumers of the parse and has since been deleted; `verify_example_furigana` was then the parse's last consumer, and slice 7 retired it.
9. Run `make gates` and one local review cycle. *(2026-08-15: this slice
   originally said "update `docs/HARDENING.md`, full replay" — both were
   deleted by M8.4, and `make gates` no longer runs a replay step. Corrected
   rather than stamped, because this milestone is still open and its slices
   read as instructions. Which of slices 1–5 are done is not recorded here:
   3 and 5 are implemented in the code, 2's subject died with M8.2, and the
   rest are unmarked — so treat the slice list as a description, not a
   checklist, until someone reconciles it.)*

Do not run a paid semantic review as part of this task. The first review after
it lands is the milestone measurement.

### [x] M7.6P Prompts are templates, and the prompt does the work

*Complete 2026-08-18 — all five implementation items landed; independent
architecture, provenance, and CLI/documentation review findings were resolved;
the final `make gates` passed 2,131 tests and built the sample deck.*

`prompts/` contains four rich task templates: three complete source shapes
(`extract-auto`, `extract-table`, and `extract-prose`) and one bare-word shape
(`enrich-bare-word`). `src/japanese_anki/prompts.py` re-reads each file on
every call. The source templates return coverage, complete card candidates,
and document patterns in one answer; the bare-word template proposes meanings,
examples, and an optional usage note in one answer. An empty note is current
when the template finds no useful, certain nuance; default targeting therefore
uses missing meanings or examples, while explicit ids request a revisit. The
former standalone paid `--polish-meanings` and document-reading `patterns`
passes are deleted.
`janki patterns` remains the local list/review surface for patterns emitted by
source extraction.

Substantive instructions now live in the Markdown templates. Python user turns
carry labelled input data, schema descriptions are terse structural labels,
and the Codex-only JSON/no-tools wrapper remains provider transport rather than
Japanese policy. A canonical full-request fingerprint over the provider,
exact style, task template, user turn, normalized transport prompt, and actual
wire response schema binds every live and batch answer. Large bare-word runs
additionally bind proposed field replacements to
the values the reviewer saw, so promotion refuses stale proposals instead of
overwriting later edits. Every new model-review artifact also carries a unique
`review_run_id`, so partial retries share an archive while two completed
invocations never conflate provenance even when their requests are identical.

Files: `prompts/`, `src/japanese_anki/ai_schema.py`, `prompts.py`, `extract.py`,
`enrich.py`, `patterns.py`, `claude_client.py`, `codex_client.py`, `cli.py`,
`ledger.py`, `promote.py`, `staging.py`, the prompt/enrichment/extraction/
promotion tests, `README.md`, `AGENTS.md`, and the enrichment/pattern docs.

*The standing rule, recorded here because this project keeps drifting from it.*
janki asks a model to read Japanese. When the answer is wrong or thin, the fix
is to **expand the prompt**, never to add code that inspects or corrects what
came back. Japanese is contextual: a rule derived from one sentence is wrong for
the next, and a list of exceptions is exactly how the retired jpdb sentence
oracle was built — 38 of 155 examples flagged, no true positive among them.
M7.6V retired it and then, within one session, three replacement rules were
proposed for furigana notation and reverted. The rule this project owns is
enrichment and filtering; the logic it owns is about the *artifact* —
identifiers, fingerprints, field counts, provenance — never about the language.

1. Lift every substantive model instruction out of Python into a template file
   readable without opening the code. User-turn builders carry labelled data,
   schema descriptions carry only terse structural labels, and the Codex
   wrapper carries provider transport. Fingerprint the provider, style, task
   template, user turn, normalized transport prompt, and actual wire response
   schema on every paid answer, including `--ai` batches.
2. The loader is the only new logic, and it is a file read — not a renderer with
   conditionals. A prompt that needs a branch in its *instruction prose* is two
   prompts — extract's three modes are three files. Interpolating a record's
   own data into the user turn stays in Python and is not a branch.
3. Prompt content is a reviewable artifact: a **test** observes the contracts
   a prompt states — `tests/test_enrich_ai.py`'s prompt section already does
   this for the exact-spelling and furigana-notation clauses, and M8.4 moved
   the last of it out of the deleted corpus — so retiring a clause is a
   visible change with a failing test rather than a silent weakening. Assert
   against the assembled prompt, not a bare constant, so the assertions
   survive the move into template files.
4. Audit what remains after M8.3 against the rule: anything anywhere in
   `src/` that judges the model's Japanese is a deletion, full stop — M8.3
   names the known ones, this audit catches stragglers. Artifact structure
   (identifiers, counts, file shape, packaging) stays.
5. Consolidate the paid AI passes into two entry points (DESIGN.md stage 2):
   rich source templates return coverage, cards, and source patterns together;
   the rich bare-word template returns proposed meanings, examples, and usage
   notes. Delete the superseded standalone polish and pattern-reading calls. If
   the cards need more, the applicable template asks for more.
   The archived camera pilot pins a `system_prompt_fingerprint` in its staging
   archive; any prompt change invalidates that recorded provenance — accepted,
   pre-release: the archive is history, not a contract.

Decided 2026-08-15, answering "where do humans edit these": a top-level
`prompts/` directory — not `docs/` (operational inputs, not reading material)
and not `templates/` (that is card HTML) — one **Markdown** file per prompt,
named by pass and input shape:

    prompts/
      README.md           the map: which pass sends which file, and the rules
      style-guide.md      moved from docs/JAPANESE_STYLE_GUIDE.md — it is a prompt
      extract-<mode>.md   one per extraction mode (three today, three files)
      enrich-bare-word.md the bare-word-list shape

Markdown because the model reads it natively and the file is sent
byte-for-byte: no placeholders, no comments, no template syntax — if it is in
the file, the model sees it, which is what keeps the loader a file read rather
than a renderer. Record data is composed into the user turn by Python; that is
data, not instruction. Editing a file changes the next run with no rebuild and
deliberately no caching, so an edit can never be silently stale. The loader
exposes a sha-256 for staging archives and batch records and raises a clean
error naming the full path when a file is missing. `git log prompts/` is the
prompt history a human reads. The pydantic `Field(description=…)` strings stay
in code — they are welded to the schema — and the audit keeps them terse,
with any real instruction moved up into the files. `review.INSTRUCTIONS`
needs no file — it died with the review subsystem in M8.2.

Depends on: M7.6V. Files: a prompt template directory, its loader, the two paid
entry points, `docs/ENRICHMENT.md`, and the prompt-contract tests.

## M8 The design leads

*2026-08-15. `docs/DESIGN.md` is now the leading document; these milestones
remove what it does not justify. The through-line of every one: janki's logic
enriches the card, it never audits the model, and a rules engine for Japanese
is this project's defining anti-pattern.*

*Standing rule, recorded here because it keeps getting forgotten: **this
software is not released.** No user, deck, or file outside this repository
depends on any behaviour here. There are no legacy paths to support, no
deprecation periods, and no backward-compatibility obligations. When an
approach is superseded, the old one is deleted in the same change — code,
config keys, data shims, tests, docs. A milestone that replaces a mechanism is
not done until the mechanism it replaced is gone.*

### [x] M8.1 OpenAI TTS for sentences; VOICEVOX keeps the words

*Complete 2026-08-18 — the sentence-side escape hatch, provider cleanup,
currency reporting, and identity-address safety landed; adversarial review was
resolved through the final gate. `make gates` is green (2,247 tests plus the
sample deck build). No paid call ran.*

**Reversed 2026-08-15, after measuring.** This milestone previously moved word
clips to OpenAI TTS too, on the owner's judgment that it speaks natural
Japanese. Re-examined at the owner's request, the numbers argued the other way
and the owner's decision is that **VOICEVOX stays for words**:

- At the reversal measurement, 89 of 97 records carried a pitch pattern, so
  forcing applied to 92% of that deck, and 0 patterns mismatched their reading
  — the pilot-era gap was closed.
- The split already in production is exactly the hybrid: every word clip is a
  VOICEVOX `.wav` (92 when this was measured, 97 after the accent fallback
  below voiced the rest), all 144 example clips OpenAI `.mp3`.
- The measured claim stands and is the whole point: unforced, 橋/箸/端 all come
  out accent 1 (`tts/openai_tts.py`). An isolated word gives the listener
  nothing *but* the accent, which is why the word clip is the one artifact
  worth forcing — a sentence carries context that disambiguates.
- Keeping costs nothing to execute: the 92 clips stay valid, against a billed
  regeneration for switching.

The honest costs of keeping, recorded so they are not rediscovered as
surprises: word audio needs a local server on `localhost:50021`, and since
`refresh` runs `audio --words --examples` a full refresh needs it up unless
`--no-audio` is passed (the availability check is per-job, so an
examples-only run never asks for it);
and one card carries two voices, currently 麒ヶ島宗麟 (speaker 53) for the word
and `onyx` for the sentence.

*Done in this reversal: the accent safe-refusal is gone.* A record with no
usable pattern used to get no word clip at all unless `--allow-default-accent`
was passed — 5 records silent today. Now every word is voiced, the engine
picking the accent when janki cannot force one, tagged `accent_unverified` in
the ledger and reported by the run. Safe because it is temporary: the word
audio fingerprint covers the exact bare-reading or forced-AquesTalk utterance,
so filling the accent with `enrich --jpdb` makes the guessed clip stale and the
next `janki audio` replaces it — pinned end-to-end, with a control half, because the first draft
of that test passed with the pattern dropped from the fingerprint entirely.
`--allow-default-accent` is deleted: it now opts into nothing. A pattern that
does not fit its reading takes the same fallback rather than staying the last
silent refusal, and still warns.

The sentence-side completion adds sparse, human-owned
`examples[].instructions`. Empty values stay absent from JSON. A clip's value
supplements the configured OpenAI baseline with one blank line between them;
the exact effective text is both the API request and the ledger render profile,
so changing one hint makes only that clip stale. Engines that cannot honor a
hint refuse the whole selected run before synthesis, and `status` explains the
configuration instead of reporting an unexplained stale count. Models that do
not apply instructions are refused, as are unsupported legacy voice/model
pairs, a blank model, invalid UTF-8 request text, and over-limit OpenAI request
fields.

The advertised Azure surface is deleted: no module, config key, dependency,
environment key, or `--provider` choice remains. `_speech_provider` retains the
useful tailored refusal only for a stale hand-written `provider = "azure"`
value.

Clips remain **identity-addressed** (`fp(record.id)` and
`fp(record.id + example.japanese)`), not addressed by text alone — two records
sharing a sentence keep two clips. An edit first makes the existing reference
stale; regeneration writes/repoints to the new address, and only then is the old
file orphaned and pruneable. The
frozen concatenation/48-bit formula can collide, so preflight compares every
selected destination against every other selected destination and every audio
reference in the supplied record universe, using the exact prepared per-clip
provider suffix. It refuses before writing rather than overwriting a clip that
another card still plays. Duplicate sentences with identical instructions
share one paid clip regardless of list order, and a reference-only repair is
persisted even when no new bytes were written.

No billed regeneration and no vocabulary/media migration: legacy
empty-instruction ledger profiles keep their exact settings shape, and the
repository's current 191 records / 343 examples round-trip with zero
`instructions` keys, zero stale clips, and zero address collisions. The 534
current ledger `content_fp` values were mechanically rebound in place from the
old normalized 48-bit digest to a raw, framed SHA-256 over the exact provider
request; filenames and audio bytes did not move.

The adversarial audio review also replaced direct paid writes with a per-clip
transaction. Each result is atomically staged under `audio/.pending`, merged
additively into top-level `pending_audio` before the next provider call, then
the record reference wins its compare-and-swap before canonical media is
published. The canonical audio ledger and superseded-entry cleanup finalize
last. An interrupted batch, concurrent record or ledger edit, or failed final
ledger commit therefore leaves exact request/profile/SHA recovery instead of a
second bill; an exact rerun adopts it, while a relevant edit cannot. Audio
commands serialize media mutation so stale `--prune` snapshots cannot delete a
concurrently committed clip. `status` reports pending recovery separately and
`build` refuses it until the matching audio command finishes.

The stage name frames both the exact request key and bytes SHA, closing the
process-death gap between writing paid bytes and merging their WAL row. Stage
and canonical promotion use no-follow, directory-bound atomic writes; matching
recovery works with the provider offline, corrupt recovery refuses before a
new call unless `--force` explicitly replaces it, and a failed forced re-voice
is adopted by the ordinary rerun. Pending targets participate in the collision
census. The audio-operation and owner-file locks span record CAS through media
publication and ledger finalization, and `build` takes the same operation lock.
Prune records canonical-ledger removal before unlinking bytes and revalidates
all durable owners under those locks.

Depends on: nothing.
Files: `src/japanese_anki/models.py`, `tts/__init__.py`,
`tts/openai_tts.py`, `audio_cmd.py`, `ledger.py`, `status.py`, `cli.py`,
`config.py`, `janki.toml`, `docs/AUDIO.md`, `docs/DATA_MODEL.md`, and the
audio/model/ledger/status tests.

### [x] M8.2 Delete the review subsystem

Decided 2026-08-15: not just the gate — the subsystem. Quality lives in the
templates; a paid pass that audits finished cards is the audit instinct at
its most expensive. Delete `review.py`, the review store and card
fingerprints, readiness, `janki review`, the refresh `review` stage and
`--no-review`, `require_review`, and every consultation of any of it.

`data/review.json` freezes as committed history — 183 paid entries are data,
not a code path — and nothing reads it again. Before the code that displays
them dies, its two human-written accent notes (both on `word:なる:なる`) are
copied into that record's provenance. The four `review-*` replay cases die
here. M7.6V and M7.6P justified their deletions by pointing at "the paid
review this task keeps"; a dated correction under M7.6V's header records
that the templates now carry that duty.

*Done 2026-08-15. `review.py` (890 lines) and `tests/test_review.py` are
gone, with `command_review` and its subparser, `_refuse_unreviewed`,
`_shipping_records`, `_card_design_text`, the `review` refresh stage and the
`--no-review` flag it generated, `require_review`, `review_file`,
`review_model`, and the two replay runners (`review-readiness`,
`semantic-review-recheck`) with their `final-review` boundary entries.*

*Three corrections found in execution. First, the two なる acceptances were
copied into `raw_fields.pitch_accent_note` rather than into a new provenance
field: the record's own `source.raw_fields` is where a fact about where its
pitch came from already lives, and adding a schema field for two sentences of
history would outlive the history. Second, `quality/pilots/m7-mixed-tsumori.yaml`
linked `review-pitch-fact-recheck` in its `finding_ids` — the corpus loader
refused the whole repository until the link was dropped, which is the pilot's
own structural check working; the run's notes now record that the finding
retired here. Third, the repository's `janki.toml` carried `review_model`,
which `test_the_repository_config_loads_without_warnings` caught the moment
the key stopped being known — the unknown-key warning doing its job.*

Depends on: nothing.
Files: `src/japanese_anki/review.py`, `cli.py`, `config.py`, `validation.py`,
`claude_client.py`, `jpdb.py`, `hardening.py`, `hardening_replay.py`,
`exporters/pattern_cards.py`, `importers/anki_deck.py`, `janki.toml`,
`data/normalized/vocabulary.json` (the two notes), `quality/cases/review-*`,
`quality/findings.yaml`, `quality/pilots/m7-mixed-tsumori.yaml`,
`quality/pilots/README.md`, `tests/test_review.py` and six other test files,
`README.md`, `AGENTS.md`, `docs/QUALITY.md`, `docs/HARDENING.md`,
`docs/ENRICHMENT.md`, `docs/DESIGN_V2.md`.

### [x] M8.3 Delete the model-audit logic

The deletions the owner ordered on 2026-08-15, each named here per the
pre-release rule. janki stops checking the model's Japanese anywhere:

- **The furigana flag subsystem**: `qc.impossible_character_furigana` and
  `kanji.assigns_a_known_reading`'s rendaku/sokuon tables — live false
  positive today: 父[とう] in `word:お父さん:おとうさん`, correct furigana
  flagged because KANJIDIC lists only ちち/フ — plus
  `qc.furigana_rewrites_sentence`, `FURIGANA_UNVERIFIED_KEY` with
  `add/prune_example_flags`, `enrich --accept` and `AcceptResult`, the audio
  hold in `audio_cmd.py`, and the flag warnings.
- **The headword rejection**: `qc.example_contains_target`, `qc.target_forms`,
  the one-limited-retry loop and `ai_retry_prompt`, the `rejected` channel.
  Measured: it rejects 食べよう, 食べれば, 食べさせる and 行かなきゃ — the
  casual forms `AI_INSTRUCTIONS` explicitly asks for — because the table
  knows seven forms.
- **The teaching-suitability grammar**: `qc.example_content_holds` with its
  politeness, fragment and conditional tables. Measured: 0 holds on the 155
  real examples, while holding もっと早く行けば and 君のことについて (natural
  casual ellipsis) and missing the identical register fault on ichidan verbs.
  Its validation error and audio refusal go with it.
- **The notation checks**: `qc.spilled_furigana_groups`,
  `qc.stray_furigana_spaces`, and `qc.repair_spilled_punctuation` — which
  silently rewrites the model's furigana. The prompt states the notation
  contract; nothing re-checks it.
- **The pattern-chart checker**: `patterns.check_pattern_rules` and
  `--check` — the dictionary-checks-writer shape M7.6V retired,
  rebuilt for charts. The human `reviewed:` gate is the approval.

Stays, reclassified: `conjugation.py` — the drill decks and the Conjugations
field are stage-3 derivation, now named in DESIGN.md; its audit callers die
above. The promote-time reading check and する predicate stay — a word fact
from a dictionary. `furigana_reading`, `furigana_base`, and
`settle_example_romaji` stays — notation arithmetic that renders and never
judges. Every gating case and finding owned by a deleted check retires in the
same commit.

*Done 2026-08-15, with two scope corrections found in execution. First:
`kanji.assigns_a_known_reading` has a second caller the milestone missed —
`enrich._furigana_for`, on the **--jpdb word path**, where it decides between
two dictionary renderings of a word's furigana (jpdb's per-character split vs
whole-word) so a jukujikun like 明日 is not taught as 明[あ]日[した]. That is
dictionary-vs-dictionary display formatting about a word — stage-3 enrichment —
so it was restored for that caller; only the model-example caller died.
Second: `check_pattern_rules`'s notation *scan* survives as
`patterns.worked_examples_in` — reading which verb ⇨ form pairs a row states
is artifact reading; judging them was the deletion. A reviewed chart's
examples now ship verbatim, and a pattern build no longer reads the collection
at all. Also recorded: the repair **proposal machinery** lost its only
instance (the punctuation repair was the sole proposal-only declaration), so
`promote --accept-proposals` currently has no producer — delete or re-instance
it in M8.4's audit. *(Resolved 2026-08-15: deleted. d7ee435 cut the whole
subsystem — proposal documents, the two-phase journal, the archive, the
dependency and staleness checks, `repair --propose`, `promote
--accept-proposals` and the `proposal-only` mode.)* Roughly 240 tests retired with their subjects; the
replaced local-failure fixtures in `test_review.py`/`test_pattern_cards.py`
now use structural faults (unbalanced brackets), which is what local
validation still owns.*

Depends on: nothing.
Files: `qc.py`, `kanji.py`, `enrich.py`, `models.py`, `validation.py`,
`audio_cmd.py`, `cli.py`, `patterns.py`, `quality/cases/`,
`quality/findings.yaml`, tests throughout.

### [x] M8.4 The corpus becomes tests, and the ceremony goes with it

Most of the corpus's value was ordinary regression testing wearing ceremony —
prose invariants, fingerprint pinning, findings opened for synthetic probes.
Migrate the replay cases that guard artifact behaviour (importers, packaging,
provenance, repairs) into plain pytest. *(The "26 of 37 portable" estimate
written here when the milestone was drafted was superseded by the measurement
below: 25 cases survived M8.2 and M8.3, and the real split is 18/2/5.)* Then delete `hardening.py`,
`hardening_replay.py` (including the dead `render-build` runner no case ever
used), `janki harden`, `quality/`, and `docs/HARDENING.md`.

Governance that outlives the apparatus moves to simple homes: **live-model
consent** — the gate on sending a private source to a billed API — becomes a
plain y/N confirmation on `janki extract` naming the file and the model;
redistribution notes live in the deck docs; staging coverage acceptance
already lives outside the corpus and is untouched. *(Repair proposals were
named here too — that sentence was falsified the same day by d7ee435,
which deleted the proposal machinery outright after M8.3 removed its only
producer. The open decision this milestone recorded is therefore
resolved: deleted, not re-instanced.)* A new
defect in janki's machinery gets a failing test first and a fix second — the
ordinary loop, no YAML. The immutable inbox and provenance rules are design,
not corpus, and are unaffected.

**Measured before deleting anything (2026-08-15).** Each surviving case was
tested by mutating the production code it pins and running pytest *with the
replay excluded* — `tests/test_hardening_replay.py` parametrizes over
`discover_cases`, so the corpus runs inside pytest and every case trivially
catches its own mutation. The honest baseline is 1909, not 2028: that file
(47 tests) and `tests/test_hardening.py` (72) both die with the apparatus.

Of 25 cases: **18 are redundant** with existing pytest tests, each confirmed by
a named non-replay failure. **Two die outright** — `m7-mixed-tsumori-coverage`
and `m7-native-teform-table-coverage` feed an empty candidate list and assert
an empty result, with byte-identical fixtures; they observe no production
behaviour at all, and their content is pilot bookkeeping for a cancelled
programme. **Five carry real coverage** and get tests before the corpus goes:

- `input-external-existing-name-collision` → `tests/test_inputs.py`. The
  refusal is covered; *which files the error names* is not. Passing
  `conflicts + matches` to `_durable_name_collision` leaves pytest green while
  the message names a byte-identical file and claims it "has the same basename
  but different content" — pointing the reader at the wrong file to move aside.
- `model-request-effort-support` → `tests/test_effort_call_sites.py`. The
  *absent* half is parametrized already; the *positive* half is pinned only for
  `claude-opus-5`. Deleting `opus-4-8`, `opus-4-7`, `sonnet-5`, `fable-5` or
  `mythos-5` from `_XHIGH_MODELS`, or trimming `_ADAPTIVE_THINKING_MODELS`,
  leaves pytest green — so every AI pass would quietly stop asking for the
  depth it was configured for. Assert the whole table, testing the key's
  *absence* rather than a `None` value.
- `ai-existing-example-annotations` → `tests/test_enrich_ai.py`. The English
  fill was covered; three other halves were not. Dropping `or
  incoming.furigana`, writing `audio=""` into the replacement, or skipping the
  romaji regeneration each left pytest green — stranding a paid clip, or
  leaving romaji that describes the sentence as it was before the furigana
  arrived.
- `furigana-full-width-separator` → `tests/test_enrich_ai.py`. Replacing the
  whole notation paragraph with "write the furigana however you think best"
  left pytest green. Written now rather than deferred to M7.6P: this prompt
  clause is the *only* remaining protection for that field, the file already
  has a prompt-assertion section, and asserting against
  `AI_INSTRUCTIONS + ai_prompt(record)` survives the move to a template.
- `derived-romaji-repair` → `tests/test_repairs.py`. The repaired value was
  covered; the version, evidence algorithm and provenance sentence stamped
  beside it were not — the audit trail a person reads months later.

*A second round, after review, found that "redundant" had been too generous.
Eight of the eighteen pinned **multi-part** results whose secondary halves
nothing else guarded, and one deletion took three assertions with it that had
nothing to do with oracles — including the only test in the repository checking
that an API key never reaches a committed staging file. Six more tests were
written and mutation-verified: the staging file's provenance block and its
absence of secrets; that an い-adjective is conjugated from its part of speech
(jpdb states no verb class for one, so narrowing `verb_group or
part_of_speech` silently emptied a whole word class's drill forms); that the
disposition lists partition `source_units` rather than sampling a deduplicated
view of them; that `inclusion_reason` survives into `raw_fields`; that only an
explicit yes at the consent prompt sends anything; and the word-deck list named
rather than counted.*

*A third round reviewed the second round's repair and found more: Ctrl-C at
the new consent prompt sent the document (the abort handler was unpinned); the
refusal announced `kept in the inbox` for files it had not copied, because
every branch of `_copy_into_inbox` returns a path under the inbox and the
filter asked containment rather than "did this run write it" —
`PreparedInput.copied` now answers exactly; that message was split across
stdout and stderr, so `2>/dev/null` printed the sentence and dropped its
qualification; and `staging._coverage_fact`, a live promote gate, had no test
at all, before this milestone or after.*

*The lesson worth keeping is not about cases. Across three rounds the tests
held up under mutation almost every time; what kept failing were the
**claims** — a correction that never reached the file because the edit script
raised before writing, a comment crediting the wrong function for a call, a
stamp applied to a milestone that was already closed while the open one kept
its dead instructions. Code gets checked by mutation. Prose only gets checked
by someone reading it, which is the argument for the review round rather than
against it.*

*Done 2026-08-15, with one scope correction found in execution. The
**coverage-oracle apparatus had to go with the corpus**, which the milestone
did not name: `promote.check_coverage` resolved oracles from
`quality/oracles/`, so deleting `quality/` broke it structurally, and
`--coverage-oracle`, `_bind_coverage_oracles`, the approved-inventory prompt
blocks in `extract.prompt_for`, and every oracle comparison in
`extract.coverage_block` had no reachable producer once it went. That is the
pilot programme's question — "did the model return everything a human said was
on this page" — cancelled with the programme. What survives is the part that
needs nobody's approval: the coverage block still records the source units,
their dispositions, both unit counts and any repeated key, and
`promote._verify_coverage_facts` still re-derives it and refuses a staging file
whose numbers no longer follow from its units. The staging schema shrank to
match — `status` accepts only `unmeasured` or `selection`, because `matched`
and `mismatch` were verdicts against an oracle and no producer can reach them,
so continuing to accept them would let a hand-edited file claim a measurement
nothing performs.*

Depends on: M8.2, M8.3.
Files: `hardening.py`, `hardening_replay.py`, `cli.py`, `Makefile`,
`quality/`, `AGENTS.md` (Hardening rules), `docs/HARDENING.md`,
`tests/test_hardening_replay.py`, `tests/test_hardening.py`,
`tests/test_inputs.py`, `tests/test_effort_call_sites.py`.

### [~] M8.5 Real sources through the real pipeline

*Started 2026-08-16. `transitivity` and `imported_from` are done (701000b).
One source is through end to end; five remain.*

**The first real page found two things, and neither was where I expected.**
`Kanji Review 104 Week11.pdf` extracted cleanly — 30 candidates, 0 validation
errors — and then `promote` refused it: coverage is `unmeasured`, and since
M8.4 deleted oracles that is now the only outcome a table extraction can have.
The approval that clears it repeats every source unit's facts: **168 lines of
YAML per page**, hand-written, with the judgment that actually matters living
in about six of them. That is transcription wearing the costume of review, and
it would have been paid five more times.

**Decided by the owner: a model may answer it.** `promote --accept-coverage`
sends `prompts/approve-coverage.md`, the page itself (the same base64 block
extraction was given, re-prepared from the inbox and fingerprint-checked
against what the extraction recorded), and janki's account of the page. The
verdict is written with `authority: model`, the model id and the prompt's
fingerprint. This overturns a standing AGENTS.md rule; the reasoning, and the
line that keeps it from being M8.2's review subsystem returning, are recorded
there.

*The first run of the checker refused the page — and it was my prompt that was
wrong, not the extraction.* It reported the red sentence at the bottom
(体温が38度以上の方は…) as missing from the record. True, and permitted:
`extract-auto.md` asks for exhaustive `source_units` from **lists and tables**
and says prose selection is explicitly not exhaustive, so a sentence has no
entry of its own and words drawn from it cite it as context. My checker
demanded everything on the page, which would have refused every page carrying
running text, forever. Corrected to state both standards; the same page then
approved with a reason naming the distinction. Two prompts disagreeing about a
contract is a failure mode worth remembering — the fix is in the file, and the
test now pins that the checker knows prose is not exhaustive.

Replaces the cancelled pilot program (M7.6B, M7.6C, M7.7, M7.W — stamped
below). The goal survives without the harness: run each collected real source
through the templates, human-review the staging file, promote, build, and
study the deck. What a pilot proved with oracles and coverage cells, use
proves directly; a defect found this way gets a failing test (M8.4's loop),
and a weak card gets a stronger template clause. Also here: the three
hand-seeded records with empty `imported_from` get it filled, so DESIGN.md's
provenance sentence is true without a caveat; and `transitivity` gets a filler
at last — set at import and backfilled by nothing since the hand-paste era
— the gap was recorded in the hand-paste prompt `prompts/ENRICH_VOCABULARY.md`,
deleted 2026-08-15, which is why it is restated here rather than cited — while
jpdb's `vt`/`vi` POS codes have carried the dictionary answer all along. Wire
it into `enrich --jpdb`.

Depends on: M7.6P, M8.1–M8.4.
Files: `data/inbox/` sources already collected, staging archives, deck
definitions under `data/decks/`.

### [ ] M7.6B Pilot pair — scans and camera captures

> *Cancelled 2026-08-15 — superseded by M8.5. The oracle, coverage-cell
> and eval apparatus this milestone requires is deleted by M8.4; the goal
> (prove extraction on real sources) survives as ordinary use.*

*Progress 2026-08-13: inventoried an owner-provided phone photo and added an
approved 14-target camera oracle. The owner also approved the exact private
Anthropic evaluation scope. The first live extraction returned all 14 approved
identities and readings; source-context review and validation passed. Promotion
added all 14 records and archived the reviewed staging file at
`data/staging/done/IMG_4563.jpg.yaml`. Dictionary enrichment filled verified
facts for 13 records. The approved source reading supplied `貴方[あなた]`; jpdb
did not match that spelling and no pitch was guessed. AI enrichment, safe audio,
and semantic review are complete. The first semantic pass exposed M7.6T. After
12 example corrections, all 14 records passed the final semantic review. The
pilot has 13 word clips and 8 example clips; six examples remain safely
unvoiced because their jpdb parses did not meet the audio guard. Build,
coverage-case completion, and the pilot report are paused until M7.6T is done.
The distinct real scan and the ambiguous or unreadable-unit evidence are still
open.*

Depends on: M7.4, M7.5, M7.6T
Files: two new immutable source copies placed under `data/inbox/` by the input
pipeline (never hand-edited), corresponding staging archives, curated
records/deck definitions under `data/`, two `quality/pilots/*.yaml`, two unit
oracles and two purpose-`coverage` case bundles, plus new findings/regression
cases/tests exposed by the pilots.

Build one scanned/skewed or low-contrast page and one phone photo with realistic
perspective, lighting, or background clutter. Use real owner-provided material;
if no suitable licensed/private source is available, leave the task open rather
than generating a clean mock-up and calling the capture path tested. At least
one source should contain a cell the human marks unreadable or ambiguous, so the
hold/omission path is exercised rather than only happy-path OCR.

Use the same size, full-pipeline, classification, case, and no-generated-output
requirements as M7.6A. Minimized regression crops must exclude unrelated page
content and declare whether the pixels may be committed.

### [ ] M7.6C Pilot pair — adversarial visual language

> *Cancelled 2026-08-15 — superseded by M8.5. The oracle, coverage-cell
> and eval apparatus this milestone requires is deleted by M8.4; the goal
> (prove extraction on real sources) survives as ordinary use.*

*Preparation 2026-08-13: inventoried distinct printed-ruby and bilingual-layout
sources. Added approved oracles for 34 exhaustive ruby rows and 19 selected
bilingual worksheet targets. The owner also approved both exact private
Anthropic evaluation scopes. The worksheet keeps `明日` unbound to a reading so
the required owner identity decision can occur in staging. No model call has
occurred.*

Depends on: M7.4, M7.5, M7.6T
Files: two new immutable source copies placed under `data/inbox/` by the input
pipeline (never hand-edited), corresponding staging archives, curated
records/deck definitions under `data/`, two `quality/pilots/*.yaml`, two unit
oracles and two purpose-`coverage` case bundles, plus new findings/regression
cases/tests exposed by the pilots.

Build two sources from the remaining hard cases: vertical text, dense annotated
screenshots, handwritten board/notes, ruby already printed over kanji, or a
layout that interleaves Japanese-only and bilingual material. Choose two that
fill different uncovered cells in `janki harden status`; do not pick two visual
variants of the same table.

Use the same size, full-pipeline, classification, case, and no-generated-output
requirements as M7.6A. At least one pilot must force a human reading decision;
the correct result is a held row followed by review, never a confident guessed
reading minted into an ID. Its staging archive keeps the M7.1 approval record
bound to the exact source, locator, expression, reading, and reviewed evidence.

### [ ] M7.7 Live extraction eval + release scorecard

> *Cancelled 2026-08-15 — superseded by M8.5. The oracle, coverage-cell
> and eval apparatus this milestone requires is deleted by M8.4; the goal
> (prove extraction on real sources) survives as ordinary use.*

Depends on: M7.6A, M7.6T, M7.6B, M7.6C
Files: `src/japanese_anki/hardening_eval.py` (new),
`src/japanese_anki/cli.py`, `Makefile`, `quality/baseline.json` (new),
`tests/test_hardening_eval.py`, `docs/HARDENING.md`.

Separate deterministic regression safety from the question that actually
changes with a model or prompt: can it still read the source?

- `janki harden eval CASE... --model ID` runs only purpose-`coverage` and
  model-boundary regression cases with a human unit oracle. A bundled fixture
  must be redistributable. Before it opens private source bytes, the command
  resolves the provider, model ID, case ID, and purpose. All values and the
  pinned source hash must match the user consent. The private case must also be
  named explicitly on this invocation. A model alias is resolved before the
  comparison; an unapproved new target needs new user consent. Neither an empty
  case list nor a broad `--include-private` switch can send private data.
  Evaluation makes no curated/staging/ledger writes. It emits a scorecard and
  raw structured responses under `dist/`.
- For `exhaustive` oracles compare exact source-unit keys, context fingerprints,
  dispositions, identities, unsupported additions, and holds/uncertainty. For
  prose `selection` oracles compare the inventoried target identities and
  locators plus unsupported additions, but label the rest of the page
  unmeasured. Exact model wording is not a metric. Record provider/model, schema
  and all prompt fingerprints, request date, and token usage so a later
  difference is explainable.
- The initial accepted scorecard, and every release comparison after it, must
  include the purpose-`coverage` case linked from each of the six M7.6 pilot
  reports. Refuse an incomplete or duplicate pilot set; one easy table case
  cannot stand in for an uncovered scan, camera, or adversarial-layout cell.
- `make eval-live` is opt-in and never a dependency of `make gates`; its target
  enumerates the six reviewed case IDs rather than discovering private sources
  broadly. Tests fake the transport. `make gates` continues to run the offline
  corpus and never needs a secret or spends money.
- `quality/baseline.json` is updated only by
  `janki harden eval --accept-baseline REPORT`, which shows the old/new
  scorecard diff and requires confirmation. Before display, it reads the report
  once and validates its strict schema. It captures the report fingerprint and
  current baseline revision. It also builds an acceptance-input manifest. This
  manifest fingerprints every pilot, case, oracle, finding, source reference,
  prompt, schema, and repair-registry value used to decide whether acceptance is
  safe. It also fingerprints each raw result artifact referenced by the report.
  The report's case set must equal the manifest case set. For each case, its
  source, oracle, provider, resolved model, prompt, response-schema, and raw
  result fingerprints must match the manifest. Every oracle must have a valid
  owner approval for its current normalized content. Missing or duplicate
  results are errors. The command recomputes all unit comparisons, refusal
  conditions, and score totals from the detailed case results. It never trusts
  report summary fields. The command puts the manifest in the planned baseline
  and plans the exact new baseline bytes. The confirmation binds the exact
  report fingerprint, manifest fingerprint, and diff.

  Eval writes one self-contained run directory under `dist/hardening/`.
  Baseline acceptance requires a regular report in that root. Every raw-result
  path is a declared relative file in the same run directory. The safe loader
  rejects absolute paths, `..`, symlinks in any component, non-regular files,
  undeclared result files, root escape after resolution, and file-identity or
  revision changes during open. It applies the same checks on the initial read
  and the post-confirmation rehash. No report field can select another input or
  output path.

  After confirmation, the command rehashes the report and every manifest input.
  It uses those reads only to compare fingerprints; it does not replace the
  validated report or planned baseline with new content. It refuses if a
  fingerprint or the baseline revision changed. It writes the planned bytes
  with a compare-and-swap guard. The accepted baseline keeps all manifest
  fingerprints, and status reports later changes as drift. There is no
  noninteractive confirmation bypass. The command verifies all six current
  pilot/case/oracle links and current prompt fingerprints. Refuse acceptance
  when the six-case matrix is incomplete, any fixed systemic case regresses, a
  human-inventoried source unit is missing, duplicated, or context-mismatched, a
  promoted identity is wrong, or an automatic repair has a false positive.
  Semantic/content correction rates are displayed by source archetype rather
  than collapsed into a misleading single accuracy score.
- `janki harden status` incorporates the accepted baseline and reports prompt
  drift (current fingerprints differ without an accepted eval), recurring
  finding codes, source-archetype coverage, and correction/repair trends across
  pilots. Do not claim statistical improvement from six samples; these are
  guardrails and directional measurements.

Tests: fake live responses; explicit private-case selection; source, case,
provider, resolved-model, and purpose consent boundaries; six-pilot matrix
completeness; score calculations; baseline compare/accept refusal conditions;
report-to-manifest mismatch; a forged summary; report, raw result, baseline, or
manifest input changed after display; outside-root, parent-traversal, symlink,
non-regular, undeclared-result, and file-identity-swap paths; prompt drift; and
stable JSON suitable for review in git. An evaluation run writes only under
`dist/`. The separately confirmed acceptance operation has one non-`dist/`
write: the atomic update of `quality/baseline.json`.

### [ ] M7.W Milestone 7 wrap and operating cadence

> *Cancelled 2026-08-15 — superseded by M8.5. The oracle, coverage-cell
> and eval apparatus this milestone requires is deleted by M8.4; the goal
> (prove extraction on real sources) survives as ordinary use.*

Depends on: all M7 tasks
Files: `README.md`, `docs/QUALITY.md`, `docs/IMPORTING.md`,
`docs/PROJECT_PLAN.md`, `docs/HARDENING.md`.

- Add the deck-driven workflow to the main docs: inventory → extract → review →
  classify → capture systemic case → fix/repair → replay → finish deck. Explain
  plainly that agents improve repository behavior, not model weights or hidden
  memory.
- Reconcile PROJECT_PLAN's later phases with this inserted hardening milestone;
  direct local Anki sync remains a separate future capability and does not gain
  write permission as a side effect of M7.
- Publish the six-pilot scorecard by source archetype: oracle coverage, holds,
  content-specific/systemic correction rates, fixed/deferred findings, repair
  applications and false positives, and live-eval prompt/model fingerprints.
- Completion gate: all six pilot reports are complete; every systemic finding
  is fixed with a passing case or carries a user-approved accepted-risk reason;
  the entire fixed corpus replays in `make gates`; the accepted live baseline
  contains the six pilot-linked coverage cases with current source, approved
  oracle, and prompt fingerprints; no inventoried source unit disappeared,
  duplicated, or changed context silently; every promoted identity passed
  review; and every enabled automatic repair is idempotent with zero known false
  positives. These are the evidence that the loop works. More total cards, by
  itself, is not.
