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
4. Definition of done, every task, no exceptions
   (`AGENTS.md` requires all of these):
   - `ruff check .` clean
   - `pytest` green (including the new tests the task specifies)
   - `janki build data/decks/verbs.yaml` succeeds
   - README/docs updated when user-visible behavior changed
   - the checkbox here flipped to `[x]`
5. Never modify `data/inbox/`. Never change `stable_record_id`, GUID
   derivation, or an existing record's `id` (single exception: M3.4's
   malformed-ID re-mint at promote time, specified there). Never insert
   into `FIELD_NAMES` (append-only, and only in task M5.4). Schema
   fields (`pitch_accent`, `audio_accent`, `frequency_rank`,
   `ExampleSentence.audio`) are added **only** in M2.2 — if an earlier
   task needs them, use defensive access, don't add fields early.
6. **No live network calls in tests.** Every client (jpdb, Claude,
   VOICEVOX, Azure) takes an injectable transport; tests use fakes with
   canned responses.
7. Dependencies policy (`AGENTS.md`: prefer stdlib): jpdb/VOICEVOX/Azure
   clients use `urllib.request` (JSON-over-POST is trivial); the
   `anthropic` SDK is justified for AI tasks and lives in the optional
   extras group `ai` created by M3.1. HEIC conversion uses `sips` via
   `subprocess` on macOS; other platforms get a clear error naming
   `pillow-heif` as the workaround (accepted scope: single-user macOS
   tool).

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
- **Fingerprints** (all via `identifiers.short_fingerprint`; formulas
  live in `ledger.py` (M1.3) — never restate them elsewhere). Two
  families, deliberately different:
  - *Filename* fp (stable addresses): word audio `fp(record.id)`;
    example audio `fp(record.id + example.japanese)`.
  - *Content* fp (staleness detection): word audio
    `fp(reading + selected_pitch_pattern)`; example audio
    `fp(example.japanese)`.
- **Staging file shape** (M1.5): a YAML mapping with `records:` (list of
  record dicts — the shape `load_records` already accepts) plus metadata
  keys the loader ignores (`source_file`, `extracted_at`, `model`,
  `review_notes`). Per-record annotations (`hold_reason`,
  `already_known`, `suggested_reading`) are stringified into that
  record's `source.raw_fields` so records round-trip through
  `VocabularyRecord.from_dict` unchanged.
- **Unverified-furigana convention** (writer M4.2, reader M5.3): a
  record-level key `source.raw_fields["furigana_unverified"]` holding a
  comma-joined list of *content fingerprints of the flagged examples'
  `japanese` text*. No per-example schema field.
- **Field-diff output** (shared helper, first built in M2.6, reused by
  M4.2/M4.3): per record, per field, one indented line of the form
  `<field>: <old> -> <new>` under a `<record id>` header.

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

### [ ] M1.1 Curation-safe merge + import CLI update

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

### [ ] M1.6 Config v2

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
  `azure_voice` (`ja-JP-NanamiNeural`), `azure_region` (`westus2`).
  TOML sections: `[paths]`, `[ai]`, `[tts]`.
- Unknown TOML keys/sections emit a warning to stderr naming the key
  and the nearest valid one (today typos are silently ignored).
- Secrets are never read from TOML. Env vars only: `ANTHROPIC_API_KEY`,
  `JPDB_API_KEY`, `AZURE_SPEECH_KEY`.
- Tests: defaults when sections absent; overrides; unknown-key warning.

### [ ] M1.3 Ledger module

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
  `exports` (`dict[deck_file_stem, iso_date]`), and a top-level
  `pending_batches` section (consumed by M4.4).
- API (idempotent, persisted via the atomic writer, file sorted by
  record id): `load(path)`, `record_added`, `record_source_seen`,
  `record_enriched`, `record_audio`, `record_export`, `remove`.
- Query helpers for `status` / `build --only-new`:
  `unexported(deck_stem, ids)`, `missing_audio(records)`,
  `stale_audio(records)` (compare stored `content_fp` against current
  content fingerprints), `missing_enrichment(records)` — **defined
  purely from record content** (empty examples/usage_notes), with
  ledger entries as metadata only, so a jpdb-only pass never hides a
  record from `enrich --ai`.
- Both fingerprint families (see Conventions) are implemented here as
  helpers; `stale_audio` uses defensive access for `pitch_accent` /
  `audio_accent` (`getattr(record, "pitch_accent", [])`) — the schema
  fields arrive in M2.2; do not add them here.
- Dates are ISO `YYYY-MM-DD` strings. `LedgerError` defined here.
- A missing ledger file is an empty ledger, never an error.

### [ ] M1.5 Reading rules + staging module

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

### [ ] M1.4 `janki status`

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
- `--duplicates`, both passes: (a) same expression, different ID;
  (b) same reading, different expression where one expression equals
  the other's reading (kana form) or both share a jpdb `vid` in
  `source.raw_fields`. Output: grouped pairs + a reminder that
  resolution is manual.
- `--rebuild`: reconstruct `sources` from each record's `source`, and
  `audio` entries from files in `media_dir` matching the filename
  fingerprint formulas; print that export state is not reconstructible.
- Record universe = normalized file **plus inline deck notes** (via
  `resolve_deck_records` per deck YAML) until M1.7 migrates them.

### [ ] M1.7 `janki migrate-inline`

Depends on: M1.1, M1.3
Files: `src/japanese_anki/cli.py`, new `src/japanese_anki/migrate.py`
(new module — do not add this to `io.py`),
`tests/test_migrate_inline.py` (new).
Design: DESIGN_V2 "Inline notes migration".

- `janki migrate-inline data/decks/verbs.yaml`: move each inline note
  with an `id` into `vocabulary.json` (merge via M1.1 semantics with
  the inline record's fields as `prefer_incoming` — inline is
  authoritative here), register in the ledger, and rewrite the deck
  YAML with the notes removed and an `include_ids:` list preserving
  exactly the previous deck membership.
- IDs byte-identical before/after (assert in test). Building the deck
  before and after migration produces the same note set (same GUIDs,
  same field values) — assert via the fake-genanki harness in
  `tests/test_anki_builder_contract.py`.

### [ ] M1.8 Import ↔ ledger wiring + shared import summary

Depends on: M1.1, M1.3
Files: `src/japanese_anki/cli.py`, `tests/test_import_ledger.py` (new).

- Wire `command_import_shirabe` to the ledger: `record_added` for
  `added` outcomes, `record_source_seen` for every incoming record —
  this is what makes "later sightings go to the ledger" true from
  Milestone 1, not Milestone 2.
- Extract the import-summary printing + merge + ledger sequence into a
  shared helper in `cli.py` that `import-jpdb` (M2.5) will call — the
  design's "identical pipeline" requirement, made concrete here.

### [ ] M1.W Milestone 1 wrap

Depends on: all M1 tasks
Files: `README.md`.

- README: `status`/ledger section; note the merge-semantics change and
  `--prefer-incoming`; document the needs-reading staging flow.

---

## Milestone 2 — jpdb

Lane map: {M2.1, M2.3, M2.4 in parallel; M2.2 after M1.5} →
{M2.5, M2.6, M2.7} (M2.6 after M2.5 — shared POS/table imports and
`enrich.py` is new in M2.6; M2.5 and M2.7 both small on `cli.py`).

### [ ] M2.1 jpdb API client (+ `jpdb ping`, POS table, /parse contract)

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
    defensively. **Capture step:** before finalizing, capture one real
    `/parse` response (small script or curl with the user's key — a
    documented manual step in the PR) and commit it as the test
    fixture; if the live shape differs from the community one, update
    client + this plan.
  - `furigana_to_anki(segments) -> str`: `[text, reading]` →
    `text[reading]`; plain segments verbatim; a space before each
    bracketed group except at string start (Anki's furigana rule).
    話す → `話[はな]す`. Golden tests: leading-kanji, mid-kanji
    (okurigana), all-kana, multi-kanji compound.
- **JMDict POS mapping lives here** (single owner; M2.5 and M2.6
  import it): `pos_to_verb_group` / `pos_to_part_of_speech` tables —
  `v5*`→godan, `v1`→ichidan, `vs*`→suru, `vk`→kuru, `adj-i`/`adj-na`
  adjectives, nouns et al.
- CLI: register `janki jpdb ping` (prints ok/failure from the client).

### [ ] M2.2 Schema additions

Depends on: M1.1, M1.5 (validation.py contention — land M1.5 first)
Files: `src/japanese_anki/models.py`, `src/japanese_anki/validation.py`,
`tests/test_models.py` (new or extend).
Design: DESIGN_V2 "Schema changes".

- `VocabularyRecord`: `pitch_accent: list[str]` (default `[]`),
  `audio_accent: str` (`""`), `frequency_rank: int | None` (`None`).
  `ExampleSentence`: `audio: str` (`""`). Extend `from_dict` coercion.
- If M1.1's merge enumerates fields, add the new ones there.
- Validation: `pitch_accent` entries match `^[HL]+$`; warn (not error)
  when `len(pattern) != len(reading) + 1` (the particle-slot invariant
  is community-verified only).
- **Do not touch `FIELD_NAMES` or the exporter** — Anki-visible fields
  ship together in M5.4.
- PR note: first `save_records_json` after this rewrites every stored
  record with the new keys (expected one-time diff).

### [ ] M2.3 Romaji converter

Depends on: —
Files: new `src/japanese_anki/romaji.py`, new `tests/test_romaji.py`.
Design: DESIGN_V2 "Division of labor" (romaji row).

- `kana_to_romaji(kana) -> str`, Hepburn, macron-free (`ou`/`uu` long
  vowels — matches Shirabe conventions; document the choice): digraphs
  (きょ→kyo), っ gemination (がっこう→gakkou), ん before b/p/m → m
  (しんぶん→shimbun), ん before vowels/y → n' (きんえん→kin'en),
  katakana accepted (normalized to hiragana first), ー long-vowel marks.
- Pure, table-driven, no deps. Golden tests per rule + a mixed sentence.

### [ ] M2.4 Conjugation tables

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

### [ ] M2.5 `janki import-jpdb`

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
- Merge + ledger + outcome printing via M1.8's shared helper.
- Reading-less kanji entries route through the M1.5 staging rule.

### [ ] M2.6 `janki enrich --jpdb`

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
  2. Stored reading empty (kana-only records only, post-M1.5) or equal
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
- Staging assist: with `--staging FILE`, annotate reading-less rows in
  a needs-reading staging file with `suggested_reading` (from the
  unforced parse) for the human to confirm — delivers the design's
  "/parse proposes a reading" step.
- Show the shared field-diff before save; `--yes` skips. Ledger:
  `record_enriched(kind="jpdb", model="jpdb", fields=...)`.
- Tests: fill-empty, `--force-fields` override, reading-mismatch
  warning path (fake transport scripted for steps 1–3), homograph
  disambiguation (forced-furigana request asserted), no-write on
  populated fields, staging `suggested_reading` annotation.

### [ ] M2.7 `janki import-jpdb-reviews` + deck filter docs

Depends on: M2.1, M1.3
Files: `src/japanese_anki/cli.py` (do **not** fold into
`jpdb_import.py` — M2.5 may be in flight), `tests/test_jpdb_reviews.py`
(new), `README.md`.
Design: DESIGN_V2 "Manual export files" (reviews.json).

- Parse the reviews export (top-level `cards_vocabulary_jp_en` etc.;
  entries `{vid, spelling, reading, reviews:[...]}`). Tag matching
  records `jpdb-known`: by vid in `raw_fields` first, else
  expression+reading. Review counts → ledger source entry. Print
  unmatched entries (count + first few).
- README: document deck YAML filtering (`include_ids`, `include_tags`,
  `exclude_ids`, `exclude_tags` — already implemented in
  `resolve_deck_records`, currently undocumented) with the
  `exclude_tags: [jpdb-known]` recipe.

### [ ] M2.W Milestone 2 wrap

Depends on: all M2 tasks
Files: `README.md`.

- README: jpdb setup (API key location, `JPDB_API_KEY`), sync + enrich
  walkthrough. (Deck-filter docs are M2.7's; don't duplicate.)

---

## Milestone 3 — PDFs and photos

Lane map: {M3.1, M3.2 in parallel} → {M3.3} → {M3.4}.

### [ ] M3.1 AI plumbing (single owner of the Anthropic client)

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

### [ ] M3.2 Input plumbing (formats, HEIC, inbox copy)

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

### [ ] M3.3 `janki extract`

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

### [ ] M3.4 `janki promote`

Depends on: M3.3, M2.1, M1.1, M1.3
Files: new `src/japanese_anki/promote.py`, `src/japanese_anki/cli.py`,
`tests/test_promote.py` (new).
Design: DESIGN_V2 "PDFs and photos > Step 2".

- Reading cross-check per candidate against the **set** of dictionary
  readings for its spelling (unforced `/parse` → vid →
  `lookup_vocabulary` readings incl. alt_sids), three outcomes: pass /
  warn-but-pass (valid non-primary reading) / hold back (in no entry).
  `--skip-reading-check` bypasses (offline).
- Kanji candidates with empty readings are always held back — unless a
  human filled `reading` (or confirmed `suggested_reading` by copying
  it into `reading`). **Malformed-ID re-mint (the one sanctioned ID
  change):** a promoted record whose stored id is `word:<expr>:` gets
  its ID re-minted from expression+reading at promote time — these
  records never entered `vocabulary.json` or Anki, so no history
  exists to orphan.
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

### [ ] M3.W Milestone 3 wrap

Depends on: all M3 tasks
Files: `README.md`.

- README: photo-of-handout → deck walkthrough (extract → review →
  promote), `.[ai]` extra install.

---

## Milestone 4 — AI enrichment

Lane map: {M4.1 (early — only needs M2)} → {M4.2} → {M4.3} → {M4.4}.
M4.2/M4.3/M4.4 all touch `enrich.py` + `cli.py`: strictly serial.

### [ ] M4.1 Example QC functions

Depends on: M2.1, M2.3, M2.4
Files: new `src/japanese_anki/qc.py`, `tests/test_qc.py` (new).
Design: DESIGN_V2 "AI integration > Mechanical QC".

- Pure functions, no CLI:
  `example_contains_target(example, expression, verb_group)` (uses
  M2.4's conjugated forms), `verify_example_furigana(example,
  parse_response)` (compare against jpdb `/parse` of the sentence via
  M2.1's contract; returns verified | mismatch), and
  `regenerate_example_romaji(example)` (M2.3 from verified furigana —
  model-supplied romaji is always discarded).
- Unit tests with canned parse fixtures; no network.

### [ ] M4.2 `janki enrich --ai`

Depends on: M3.1, M4.1, M2.6, M3.4 (the ≥50-record path needs promote)
Files: `src/japanese_anki/enrich.py`, `src/japanese_anki/cli.py`,
`tests/test_enrich_ai.py` (new).
Design: DESIGN_V2 "AI integration".

- Targets: records with empty examples or usage_notes (content-defined,
  per M1.3; explicit `IDS...` overrides). Per record, one structured
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

### [ ] M4.3 `--polish-meanings`

Depends on: M4.2
Files: `src/japanese_anki/enrich.py`, `src/japanese_anki/cli.py`,
`tests/test_enrich_ai.py`.

- Separate explicit operation (meanings are never empty, so fill-empty
  can't reach them): propose improved gloss lists; always print
  old → new per record; per-record confirm (or `--yes`); write + ledger.

### [ ] M4.4 Batch submit/fetch

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

### [ ] M4.W Milestone 4 wrap

Depends on: all M4 tasks
Files: `README.md`, `prompts/ENRICH_VOCABULARY.md`.

- README: enrichment costs table, model configuration. Replace
  `prompts/ENRICH_VOCABULARY.md` body with a pointer at `janki enrich`.

---

## Milestone 5 — Audio

Lane map: {M5.1, M5.2, M5.5 in parallel} → {M5.3} → {M5.4} → {M5.6};
M5.7 anytime after M5.3.

### [ ] M5.1 Pitch conversion + HTML renderer (merge gate: golden tests)

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

### [ ] M5.2 VOICEVOX provider

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

### [ ] M5.5 Notetype-upgrade verification (spike; blocks M5.4)

Depends on: — (start anytime)
Files: new `docs/NOTETYPE_UPGRADE.md` (findings); throwaway scripts in
scratch (not committed).
Design: DESIGN_V2 "Schema changes" (the unproven claim).

- Empirically answer: importing an `.apkg` with the same `model_id` but
  three appended fields into a live Anki collection with review
  history — in-place notetype upgrade with GUID-matched note updates,
  or remap/skip? Method: scratch Anki profile → build current →
  import → review a card → rebuild with appended fields → re-import.
- Document the verified procedure (either "safe: append + reimport" or
  the required in-Anki/scripted notetype migration). M5.4 implements
  whatever this concludes and links it.

### [ ] M5.3 `janki audio`

Depends on: M5.1, M5.2, M2.2, M1.3
Files: new `src/japanese_anki/audio_cmd.py`, `src/japanese_anki/cli.py`,
`tests/test_audio_cmd.py` (new).
Design: DESIGN_V2 "Audio > Mechanics".

- Flags: `--words`, `--examples`, `--provider {voicevox,azure}`
  (overrides `[tts] provider`; azure errors "not implemented" until
  M5.7), `--force`, `--prune`, `--allow-default-accent`, `IDS...`.
  `AudioError` defined here.
- `--words`: records with reading + `select_pattern` result and no
  current word audio (ledger content_fp-aware): M5.1 convert → M5.2
  synthesize → `media_dir/audio/janki-<filename fp>.wav` →
  `record.audio` (media-dir-relative) → ledger `record_audio` with the
  word content fp. Empty pattern → **skip + flag** in the summary;
  `--allow-default-accent` synthesizes without forcing (plain
  audio_query path) and ledger-tags `accent_unverified`.
- `--examples`: per example with furigana **not** listed in the
  record's unverified-furigana key (see Conventions) and no current
  audio: synthesize sentence (no accent forcing), filename/content fps
  per Conventions, set `example.audio`, ledger entry.
- Staleness falls out of content-addressed naming: edited examples
  simply lack audio; `--prune` deletes unreferenced `janki-*` files.
  `--force` regenerates in place (same filenames; Anki media sync
  picks up content changes).

### [ ] M5.4 Exporter + templates (ships all new note fields at once)

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

### [ ] M5.6 `build --only-new` + `janki refresh`

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
  success, `record_export(stem, today)` per included record; plain
  `build` records exports too (idempotent).
- `janki refresh [--deck DECK]`: enrich --jpdb → enrich --ai →
  audio --words --examples → build --only-new, in order, per-stage
  skip flags, one-line summary per stage. Calls the command functions
  directly (no subprocesses).
- While in the exporter: route the `.apkg` write through temp-then-rename
  (genanki needs a real path — `write_to_file(tmp)` then `os.replace`) so
  an interrupted build can't leave a truncated package that a later
  `--only-new`-era run mistakes for a good one.

### [ ] M5.7 Azure sentence-audio provider

Depends on: M5.3
Files: new `src/japanese_anki/tts/azure.py`, `tests/test_azure_tts.py`.

- REST via the M5.2 transport shape (no SDK dep): key header, SSML
  body, voice/region from config, `AZURE_SPEECH_KEY` env.
  `<sub alias="...">` substitution fed from **verified** furigana for
  reading-ambiguous tokens. Fake-transport tests assert SSML shape and
  sub/alias injection. Wire into `audio --provider azure`.

### [ ] M5.W Milestone 5 wrap

Depends on: all M5 tasks
Files: `README.md`.

- README: audio setup (VOICEVOX install/launch, credit line for shared
  decks), `janki refresh` as the headline weekly workflow, retire the
  "no synthesized audio" limitation paragraph.
