# Design v2: Multi-Source, AI-First Pipeline

Status: accepted (2026-08-06), revised after adversarial review.
Supersedes the architecture section of `PROJECT_PLAN.md` where they
disagree. Implementation is tracked in `IMPLEMENTATION_PLAN.md` — its
checkboxes are the source of truth for what is built.

## Goals

Take word lists from multiple sources (Shirabe Jisho, jpdb.io, class-material
PDFs and photos) or raw class material, and quickly build Anki decks that
include kanji, kana, furigana, romaji, English, pitch accent, and generated
Japanese pronunciation audio — with AI as a first-class part of the pipeline
rather than a copy-paste side channel.

Three additions drive this design:

1. **jpdb.io as a data source** — deck sync via its API, per-word dictionary
   enrichment (frequency rank, pitch accent, POS, furigana), and manual
   export files.
2. **PDFs and photos as a data source** — AI extraction from vocab tables,
   prose, and scans, gated by a human-review staging step.
3. **A word ledger** — a simple machine-written database of every word we
   have ever added, with metadata: when it arrived, from where, whether it
   has been enriched, whether audio exists, and which decks it has been
   exported to.

Unchanged principles: the repository is the source of truth, raw inbox files
are immutable, normalization is reproducible, curation lives in readable
JSON/YAML, IDs and Anki GUIDs stay deterministic.

Two v1 limitations are retired, deliberately:

- **Pitch accent** is no longer a non-goal. We still never *guess* it — we
  take it from jpdb's dictionary data, which is real data.
- **Synthesized audio** is no longer out of scope. Japanese TTS with
  explicit reading and accent control is now good enough for study audio.

## Architecture

```text
Shirabe CSV      jpdb API / exports    PDFs & photos (class material)
     |                  |                        |
     |                  |                  janki extract  (Claude, vision)
     |                  |                        |
     |                  |               data/staging/*.yaml
     |                  |                (human review)
     |                  |                        |
     v                  v                        v
  import-shirabe    import-jpdb              promote
     \                  |                       /
      +---------- merge (curation-safe) ------+
                        |
                        v
        data/normalized/vocabulary.json          data/ledger.json
         (canonical card content)              (machine-written state)
                        |
                        v
        janki enrich  (jpdb dictionary data + plain code + Claude API)
        janki audio   (VOICEVOX / Azure TTS -> data/media/audio/)
                        |
                        v
          validation -> deck YAML -> deterministic builder
                        |
                        v
                 dist/*.apkg -> Anki
```

The left-to-right story: *sources produce candidate records; the merge
protects curation; enrichment and audio fill gaps; the ledger records what
happened; builds are reproducible.*

## Division of labor

This table is the design's spine. Facts are looked up, rules are computed,
judgment is generated — never the other way around.

| Work | Who does it | Why |
| --- | --- | --- |
| Readings, furigana, pitch accent, frequency, POS/verb group | **jpdb dictionary data** | Facts. Never generate what you can look up. |
| Romaji (Hepburn, from kana), conjugation tables | **Plain code** | Rule-based. No tokens needed. Long vowels, っ gemination, ん assimilation, ゃゅょ digraphs are a table, not a judgment call. |
| Example sentences, usage notes, meaning polish | **Claude API** | Judgment. This is what the model is for. |
| PDF/photo → candidate records | **Claude API (vision)** | No mechanical parser exists for class handouts. |
| Word/sentence audio | **TTS (VOICEVOX / Azure)** | With reading + accent forced from dictionary data. |

Romaji is called out explicitly because it is in the goal sentence: a
kana→Hepburn converter runs during `enrich` (and on import) whenever
`reading` is present and `romaji` is empty — for records and for example
sentences alike. Example-sentence romaji comes from the same converter fed
by the example's furigana.

## Data sources

### Shirabe (mostly unchanged)

`import-shirabe` keeps its current behavior, subject to the new merge
semantics and the reading rule below. One change: **kanji rows with no
reading no longer enter `vocabulary.json`** — they are routed to a staging
file for review instead (see "Readings are ID-constitutive").

### jpdb.io

jpdb serves three roles: source of word lists, dictionary enricher, and
manual-file import target.

**API client** (`src/japanese_anki/jpdb.py`):

- Base URL `https://jpdb.io/api/v1/`, auth `Authorization: Bearer $JPDB_API_KEY`
  (key from the jpdb settings page; env var only, never in config files).
- Endpoints are JSON-over-POST (`/ping` also answers GET). Responses are
  **column-oriented**: rows are positional arrays in the order of the
  `fields` you requested — the client zips them into dicts immediately.
- Words are addressed by `[vid, sid]` pairs everywhere.
- Retry with backoff on `429 too_many_requests` and `api_unavailable`
  (limits are undocumented; batch requests, request only needed fields).
- The API misspells "occurrences" as `occurences` — use the misspelled keys.
- `janki jpdb ping` validates the key.

**Deck sync** (`janki import-jpdb [--deck NAME | --all-decks]`):

1. `list-user-decks` (fields: id, name, word_count) — pick deck(s).
2. `deck/list-vocabulary` per deck — returns `[vid, sid]` pairs (optionally
   `occurences` counts, worth keeping as ledger metadata for mined decks).
3. `lookup-vocabulary` in batches with fields: `spelling`, `reading`,
   `frequency_rank`, `pitch_accent`, `meanings_chunks`,
   `meanings_part_of_speech`, `part_of_speech`, `card_state`.
4. Map to `VocabularyRecord`: `expression=spelling`, `reading=reading`,
   meanings from gloss chunks, `part_of_speech`/`verb_group` mapped from
   JMDict codes (e.g. `v5k` → godan, `v1` → ichidan, `vs` → suru),
   new fields `pitch_accent` and `frequency_rank` (schema below), romaji
   from the kana converter. `source.type="jpdb"`, `raw_fields` keeps
   vid/sid/deck/card_state. Auto-tag `jpdb` plus the source deck name.

**Dictionary enrichment** for records that came from *other* sources
(`janki enrich --jpdb`): for each record, call `/parse` with the expression.
When the record already has a reading, pass it as forced furigana to
disambiguate homographs — the wire shape is
`furigana: [[position, length, reading]]` spans interpreted under an
explicit `position_length_encoding`, so the client must compute the span
over the whole expression in the chosen encoding. Fill *empty* fields only:
furigana (from parse's token furigana — dictionary-backed, so this does not
violate the "never guess segmentation" rule), pitch accent, frequency rank,
POS/verb group, and romaji via the converter. Mismatches between our stored
reading and jpdb's readings are reported as warnings, never auto-fixed.
**`reading` itself is never filled by enrichment** — see "Readings are
ID-constitutive".

**Manual export files**:

- The JPDB-Export userscript CSV (spelling + furigana reading) — parsed by
  `import-jpdb FILE.csv`. This reuses the CSV alias-mapping machinery, which
  means (a) the generic CSV importer gets factored out of the
  shirabe-named module, and (b) `FIELD_ALIASES` gains `spelling` and the
  userscript's reading header — today neither column would map and the
  import would fail.
- `reviews.json` (jpdb's native "Export vocabulary reviews") —
  `janki import-jpdb-reviews FILE.json` does not create records; it tags
  matching existing records `jpdb-known` and stores review counts in the
  ledger. Matching rule: by `vid` where the record has one in `raw_fields`,
  else by expression+reading; unmatched entries are reported, not dropped.
  Decks then exclude known words with the **existing** deck-YAML filter:
  `exclude_tags: [jpdb-known]` (include/exclude id/tag filters already
  exist in `resolve_deck_records`; they are just undocumented — README
  gains a section).

jpdb does **not** expose JLPT level; if we ever want it, it comes from a
static wordlist, not the API.

### PDFs and photos

Class material arrives in three shapes — structured vocab tables, prose or
readings to mine, and scans — and two file kinds: PDFs and phone photos
(HEIC/JPEG). All go through the same two-step flow because none can be
parsed mechanically.

**Step 1 — AI extraction** (`janki extract FILE... [--mode table|prose]`):

- Accepts `.pdf`, `.jpg`, `.png`, `.heic` (HEIC converted to JPEG on the
  fly via `sips` on macOS / pillow-heif elsewhere). PDFs go to the Claude
  API as `document` content blocks; images as `image` blocks. Scans need no
  OCR step — PDF support is vision-backed, each page is processed as an
  image. Source files are copied into `data/inbox/scans/` for provenance if
  not already under `data/inbox/`.
- Uses **structured outputs** (`client.messages.parse()` with a Pydantic
  schema mirroring the candidate-record shape), so the result is
  schema-valid on normal completion. The command checks `stop_reason`
  before trusting output: `refusal` → clear `ExtractError`; `max_tokens` →
  re-run per-page or raise, never silently truncate.
- The system prompt embeds `docs/JAPANESE_STYLE_GUIDE.md` with a
  `cache_control` breakpoint, so repeated extractions reuse the cached
  prefix.
- `--mode table` transcribes vocab lists faithfully; `--mode prose` mines
  card-worthy vocabulary from running text. Default: the model detects the
  shape per page. In prose mode, the prompt includes the list of known
  expressions so the model skips them.
- Each candidate carries provenance in the schema itself: page number,
  surrounding context line, and a confidence level. (Citations would be
  nicer but are incompatible with structured outputs — self-reported page
  numbers are fine for a human-reviewed step.)
- Output goes to `data/staging/<source-name>.yaml`. `extract` refuses to
  overwrite an existing staging file without `--force` — staging files hold
  un-committed human edits, the one thing in the repo that git can't
  recover.
- After extraction, each candidate is annotated `already_known: true` when
  its ID (or expression+reading) matches an existing record; known
  candidates sort last so review effort goes to the new words.

**Staging file shape** (pinned, so `janki validate` works on it as-is):
a mapping with a `records:` list — the shape `load_records` already
accepts — plus metadata keys (`source_file`, `extracted_at`, `model`,
`review_notes`) that the loader ignores. Validation errors on incomplete
candidates (missing meanings, kanji without reading) are the *expected
review signal*, not breakage.

**Step 2 — human review, then promote** (`janki promote data/staging/X.yaml`):

- The staging file is ordinary YAML: read it, delete bad rows, fix readings.
- `promote` runs a **reading cross-check** against the full set of
  dictionary readings for each spelling (via jpdb lookup of the parse
  results), with three outcomes: reading in the valid set → pass; reading
  in no dictionary entry → held back; valid but non-primary reading (辛い
  as つらい) → warn but pass, because the handout context the extractor saw
  is better evidence than corpus frequency. This catches the most damaging
  extraction error class (wrong reading → wrong ID → wrong audio) without
  punishing correct homograph readings.
- Surviving records merge into `vocabulary.json`; promoted rows move to
  `data/staging/done/<file>`; held-back rows are *rewritten in place* with
  their hold reason inline, so a partially-promoted file always shows
  exactly what still needs attention. The file is removed when empty.

Model choice: extraction defaults to `claude-opus-5` — scans and dense
handouts are where errors are most expensive, since they propagate into
every card. Configurable (`[ai] extract_model`) for cheaper runs on clean
PDFs.

## The ledger

`data/ledger.json` — machine-written, git-committed, deliberately minimal.
The invariant: **`vocabulary.json` holds card content** (human- and
machine-written, human-owned — enrichment and audio paths write there too);
**the ledger holds operational metadata about that content** — no
timestamps or provenance in records, no card content in the ledger.

```json
{
  "version": 1,
  "records": {
    "word:話す:はなす": {
      "added_at": "2026-08-06",
      "sources": [
        {"type": "shirabe", "ref": "shirabe-export.csv", "seen_at": "2026-08-06"},
        {"type": "jpdb", "ref": "deck:Mining", "vid": 1577980, "sid": 1577980, "seen_at": "2026-08-10"}
      ],
      "enriched": [
        {"at": "2026-08-10", "kind": "jpdb", "model": "jpdb", "fields": ["pitch_accent", "frequency_rank"]},
        {"at": "2026-08-11", "kind": "ai", "model": "claude-opus-5", "fields": ["examples", "usage_notes"]}
      ],
      "audio": [
        {"file": "janki-9f8e7d6c5b4a.wav", "of": "word", "provider": "voicevox", "voice": 46,
         "content_fp": "1a2b3c4d5e6f", "at": "2026-08-11"}
      ],
      "exports": {"personal-vocabulary": "2026-08-12"}
    }
  }
}
```

Design points:

- **Keyed by record ID** — the same deterministic ID the Anki GUID derives
  from, so "exported" is tracked at the level Anki actually cares about.
- **`sources` is append-only.** This fixes a real v1 bug: `merge_records`
  overwrites `record.source` on every re-import, erasing provenance. The
  record keeps its *original* source; the ledger keeps the full history.
- **Exports are keyed by deck file stem** (`personal-vocabulary`, not the
  Anki deck name) — stems are unique in the repo, deck names can collide
  when two deck files fall back to the project default name. The value is
  just the last-built date: that is all `build --only-new` consumes.
  Anything richer (build counts, per-build fingerprints) waits until a
  command exists that reads it. `.apkg` checksums are not used — the
  fingerprint of *content* is the useful signal, not the artifact (genanki
  can build hermetically given a fixed timestamp, but that solves a problem
  we don't have).
- **Audio entries carry a `content_fp`** — the fingerprint of what was
  spoken (the reading+accent for word audio, the sentence text for example
  audio). This is what makes stale audio *detectable*: edit an example
  sentence and `status`/`audio` see the mismatch and flag regeneration.
  File naming is covered in the Audio section.
- **Writers**: `import-*` and `promote` (added_at, sources), `enrich`
  (enriched), `audio` (audio), `build` (exports). All writes are
  whole-file, sorted by key, and go through the shared atomic helper
  (`io.atomic_write_text`, built in M1.2) — the same stable-diff pattern
  `save_records_json` and `vocabulary.json` now use.
- **`janki status`** reads it: totals per source, words never exported,
  words missing audio/enrichment/pitch accent, stale audio, and duplicate
  candidates (below). `--format ids` emits plain IDs for scripting.
  `janki status --rebuild` reconstructs what it can (sources from
  `record.source`, audio from files on disk); export state is *not*
  reconstructible — after ledger loss, the next `--only-new` build simply
  includes everything, which is safe (GUIDs make re-export idempotent).

**Duplicate detection** (`janki status --duplicates`) looks for every real
duplicate class, not just the obvious one:

- same expression, different ID (covers both different-reading and
  historical empty-reading records);
- same reading, different expression where one expression is the kana form
  of the other, or both resolve to the same jpdb `vid` — this is the common
  case in practice (Shirabe bookmark わかる vs jpdb-mined 分かる);
- the same non-empty jpdb `vid` under more than one ID regardless of reading
  (vids compared numerically) — a shared vid is the same dictionary word even
  when a reading was hand-corrected or is empty.

Resolution is manual and documented: pick the surviving ID; the loser's
review history is the accepted cost, consistent with the no-re-ID rule.

## Readings are ID-constitutive

`stable_record_id` embeds expression *and* reading; the Anki GUID derives
from the ID. Consequences, now stated as rules rather than left implicit:

1. **A kanji record without a reading never enters `vocabulary.json`.**
   Today the Shirabe importer warns and imports anyway, minting a malformed
   ID (`word:話す:`) that can never be corrected without re-IDing — and
   that later duplicates when the same word arrives with its reading.
   Instead, reading-less kanji rows are diverted to a staging file where
   `/parse` proposes a reading and the human confirms before the ID is
   minted. `validate` gains an error for kanji expressions with empty
   readings / malformed IDs.
2. **Nothing ever auto-fills or auto-corrects `reading` on an existing
   record** — not enrichment, not import merge. Reading disagreements are
   warnings for a human.
3. **No reading normalization in the ID scheme, ever** (e.g.
   katakana→hiragana) — it would re-ID existing records and orphan review
   history. Near-duplicates are surfaced by `status --duplicates` instead.

## Curation-safe merge (prerequisite for everything else)

Today `merge_records` resolves every field as `incoming or existing` —
incoming wins whenever non-empty. That was tolerable when the only writer
was one importer; with enrichment in the pipeline it means **a re-import
silently destroys AI-generated and hand-curated work** (examples, furigana,
usage notes — anything the CSV happens to also carry).

New rules for `merge_records`:

1. **Existing wins.** An import fills only fields that are currently empty.
2. `tags` stays a union. `source` on the record is no longer overwritten —
   first source sticks; later sightings go to the ledger.
3. `--prefer-incoming FIELD[,FIELD]` (on the import commands) opts specific
   fields back into the old behavior for deliberate refreshes.
4. The merge report becomes a per-record outcome map
   (added / filled / unchanged / conflicting); conflicts (both sides
   non-empty and different) are *printed*, not silently resolved.
5. `--replace` prompts for confirmation unless `--yes` (both new; the repo
   is git-versioned, so the guard is against fat-fingering, not data loss).

Known blast radius of this change (Milestone 1 work, not a footnote):
`tests/test_merge.py` encodes incoming-wins today — ironically its
"preserves existing enrichment" test asserts the exact clobbering behavior —
and must be rewritten to assert fill-only semantics plus conflict
reporting. `command_import_shirabe` builds and prints the old counts dict
(including on the `--replace` path) and must move to the outcome map.

`enrich` and `audio` follow the same rule from the other direction: they
fill empty fields only, unless `--force-fields ...` is given, and they show
a diff before saving (`--yes` to skip).

Deck-YAML inline notes keep their replacement semantics (they are explicit,
hand-written overrides — replacement is the point there), with one nuance
to document in `CARD_DESIGN.md`: `_merge_inline_record` replaces every key
wholesale *except* `source`, which is shallow-dict-merged. Making the
contract explicit is enough; unifying it is not worth the churn.

**Inline notes migration**: the only real records in the repo today live
inline in `data/decks/verbs.yaml`, and the new pipeline's writers (enrich,
audio) only ever write to `vocabulary.json` — inline notes would be a
permanent dead zone (no pitch accent, no audio, ever). A one-time
`janki migrate-inline DECK.yaml` moves inline notes into `vocabulary.json`
(IDs preserved, so GUIDs and review history survive) and leaves the deck
YAML as filters/overrides. This lands in Milestone 1, before the ledger
starts tracking records it could never update.

## AI integration

AI runs inside `janki` via the Anthropic Python SDK. `ANTHROPIC_API_KEY`
from the environment (or an `ant auth login` profile — the SDK resolves
it). The `prompts/` directory shrinks to what it is: documentation of
review workflows for a human driving a coding agent, no longer the
enrichment mechanism.

**`janki enrich [--jpdb] [--ai] [IDS...]`**

- `--jpdb`: the mechanical pass described earlier, plus the plain-code
  fills — romaji from kana, conjugation tables from `verb_group`
  (godan/ichidan/suru/くる rule tables; anything irregular beyond those is
  left empty and flagged, never guessed). Cheap, fast, run freely.
- `--ai`: for records still missing examples or usage notes (ledger-aware),
  one structured-output request each: system prompt = style guide (cached;
  1-hour TTL when batching), user content = the record + its jpdb facts +
  a few recently generated examples (so contexts vary across the deck
  instead of producing 「毎日〜します」 two hundred times). Pydantic schema
  covers exactly the enrichable fields, including example romaji/furigana.
- **Mechanical QC before anything is accepted** — AI examples get the same
  distrust AI extraction gets:
  - the example must contain the target expression or one of its
    code-generated conjugated forms, or it is rejected;
  - example furigana is verified against jpdb `/parse` of the sentence;
    mismatches are flagged (model-guessed segmentation must not silently
    drive sentence audio);
  - results fill empty fields only, run through validation, and show a
    diff before save.
- For large runs, results route through the same staging/promote flow PDFs
  use — a 500-record y/n diff is not review; a staging file is.
- **Meaning polish is a separate, explicit operation** —
  `enrich --ai --polish-meanings` — because meanings are never empty (the
  importer always fills them), so fill-empty semantics can never touch
  them. It always shows old→new per record and requires confirmation.
- Default model `claude-opus-5`, configurable (`[ai] enrich_model`).

**Batch mode** (`enrich --ai --batch-submit` / `--batch-fetch`): the
Message Batches API halves token cost but jobs can take hours, so it is
split into two subcommands with the in-flight batch ID and pending record
IDs persisted in the ledger — submit, walk away, fetch later; re-submit
refuses while a batch is pending. Worth it for backfilling a
thousand-word jpdb mining deck; pointless for the weekly ten words (which
run synchronously in seconds for cents).

## Audio

**`janki audio [--words] [--examples] [IDS...]`**

Provider is pluggable behind a small interface; two ship initially.

**VOICEVOX (default; word audio).** Free, local (engine at
`localhost:50021`), community-standard for Anki audio, and — decisive for
flashcards — it can force *both* the reading and the accent-drop mora.
The correct flow matters, because the obvious one fails silently:

1. Convert reading + jpdb pitch pattern to AquesTalk kana notation
   (katakana + `'` accent mark).
2. `POST /accent_phrases?text=<kana>&is_kana=true&speaker=N` — **this is
   the only endpoint that accepts `is_kana`**. (Passing it to
   `/audio_query` is silently ignored — FastAPI drops unknown params and
   returns normally-processed, possibly-wrong audio, the exact 橋/箸
   failure this feature exists to prevent.)
3. Substitute the returned accent phrases into an `AudioQuery`, then
   `POST /synthesis`.
4. A unit test asserts the forced accent survives into the final
   `AudioQuery` — an endpoint mix-up must fail loudly, not hum along.

**Pitch-pattern conversion** (jpdb → AquesTalk) is specified, not
hand-waved, because the naive rule is wrong for the largest accent class:

- jpdb's pattern is one character per *kana* of the reading **plus one
  trailing position for the following particle** (`話す` → `LHHH`).
  Community clients hard-assert `len(pattern) == len(reading) + 1`.
- Regroup kana into **morae** first (ゃゅょ attach to the preceding kana;
  っ and ん count as morae) — accent positions are mora positions.
- Accent nucleus = the mora after which H→L occurs. The particle position
  is what distinguishes odaka (drop at the particle: 橋 `LHL`) from heiban
  (no drop anywhere: 端 `LHH`).
- Heiban has **no** H→L transition; VOICEVOX's kana notation still
  requires exactly one accent mark per phrase, and the engine's own
  convention writes heiban with the mark on the final mora. The converter
  encodes that convention explicitly.
- A golden test set — heiban, atamadaka, nakadaka, odaka, plus a 拗音 word
  (病院) — is a **merge gate** for the audio milestone, not a follow-up.
- Multi-accent words: audio uses `pitch_accent[0]` (jpdb's ordering),
  overridable per record with an `audio_accent` field.
- Empty `pitch_accent`: the record is **skipped and flagged** by default
  (letting the engine guess would violate the never-guess rule for exactly
  the homograph minimal pairs that matter); `--allow-default-accent` opts
  in explicitly and tags the ledger entry `accent_unverified`.
- Community testing says even forced accents are occasionally rendered
  wrong (やり直す is a known case) — spot-check; the ledger makes
  regeneration targeted.

**Azure Speech (optional; sentence audio).** ja-JP neural voices are the
most natural cloud option and Microsoft explicitly models pitch accent;
the 500K-chars/month free tier covers a personal pipeline forever. Used
for example sentences, where a natural adult voice matters more than
forced accent (sentence-level accent is much harder to pin anyway). SSML
`<sub alias>` pins readings inside sentences when needed — fed from the
example's *verified* furigana, never from unchecked model output.

**Mechanics:**

- Files land in `data/media/audio/`, named `janki-<fp>.wav` where the
  fingerprint is over **content, not position**: `fp(record_id)` for word
  audio, `fp(record_id + example.japanese)` for sentence audio. Editing or
  reordering examples therefore *orphans* the old file (a visible
  missing-audio state the tools flag) rather than silently re-binding
  existing cards to the wrong sentence's audio. `status`/`audio` compare
  the ledger's `content_fp` against current record content to catch
  staleness, and `audio --prune` removes orphaned files.
- Fingerprint names also fix a real exporter hazard: media dedup is by
  path but packaging is by basename, so two files both named `audio.mp3`
  would collide inside an `.apkg`. The `janki-` prefix namespaces us in
  the user's global Anki media collection.
- `record.audio` and `ExampleSentence.audio` store paths relative to the
  configured media dir. The exporter resolves media against
  `[paths] media_dir` first, with the current deck-relative resolution
  kept as a fallback for existing decks.
- The exporter's `[sound:...]` passthrough branch (verbatim tag, no file
  packaged) becomes a **warning** — it is a silent-mute trap for a
  pipeline that generates its own audio.
- Word-audio regeneration for unchanged content (`--force`, e.g. new
  voice) keeps the same filename, so Anki media sync picks up new content
  for the same `[sound:]` reference.
- Templates: `{{Audio}}` already renders on recognition-back; production
  back gets it too, and example audio renders next to the example.

## Schema changes

`VocabularyRecord` gains:

- `pitch_accent: list[str]` — jpdb-style patterns (reading kana + particle
  position, e.g. `"LHHH"`), possibly multiple, first entry primary.
- `audio_accent: str` — optional per-record override for audio generation.
- `frequency_rank: int | None` — jpdb corpus rank.

`ExampleSentence` gains `audio: str`.

`SourceReference.raw_fields` stays `dict[str, str]` — PDF provenance
(page, confidence) is stringified into it rather than growing the model.

Dataclass-change side effect, stated accurately: merge counts are *not*
affected (loaded and merged records both get the new defaults, so equality
holds), but the first `save_records_json` after upgrade rewrites every
record with the new keys — a one-time whole-file JSON diff.

**Pitch-accent display is decided now, not punted**: the builder renders
the pattern to inline HTML (span-per-mora with border styling — the
standard Anki approach; no card JavaScript, per the mobile-legibility
rule) into the `PitchAccent` note field at export time. The raw pattern
lives only in the repo. Multi-accent display: all patterns, primary first,
matching the audio-selection rule.

**Anki note-field surface**: new fields (`PitchAccent`, `FrequencyRank`,
`ExampleAudio`) are appended to the end of `FIELD_NAMES`, never inserted —
genanki fields are positional and the model ID does not change with field
edits. **The safe-upgrade claim must be proven, not assumed**: Anki's
importer distinguishes notetypes by a schema hash of field/template names,
and its behavior when an incoming `.apkg` shares a `model_id` but differs
in field count is exactly the kind of thing that can duplicate or skip
notes in a live collection. Before the audio milestone ships, the
append-and-reimport path is tested against a real collection with review
history; if in-place upgrade doesn't hold, the documented procedure becomes
"add the fields to the notetype inside Anki first, then import" or a
scripted notetype migration. All new fields land in **one** release —
field appends are one-way.

Record IDs are untouched: no change to `stable_record_id` or GUID
derivation, ever (see "Readings are ID-constitutive").

## CLI surface

```bash
# existing (unchanged flags shown accurately)
janki inspect FILE.csv
janki import-shirabe FILE.csv [--output PATH] [--replace]
janki validate [PATH]
janki build [DECK_PATH | --all] [--output PATH]
janki preview DECK_PATH

# changed
janki import-shirabe ... [--prefer-incoming F,..] [--yes]   # confirmation on --replace
janki build [DECK | --all] [--only-new]   # DECK = path, or bare name resolved in deck_dir

# new
janki import-jpdb [--deck NAME | --all-decks | FILE.csv] [--prefer-incoming F,..]
janki import-jpdb-reviews reviews.json
janki jpdb ping
janki extract FILE... [--mode table|prose] [--model ID] [--force]
janki promote data/staging/X.yaml [--skip-reading-check]
janki enrich [--jpdb] [--ai] [--polish-meanings] [--batch-submit|--batch-fetch]
             [--force-fields F,..] [--yes] [IDS...]
janki audio [--words] [--examples] [--provider P] [--force] [--prune]
            [--allow-default-accent] [IDS...]
janki status [--unexported] [--missing-audio] [--duplicates] [--rebuild] [--format ids]
janki refresh [--deck DECK]   # the weekly loop, in order (below)
janki migrate-inline DECK.yaml   # one-time, Milestone 1
```

**`janki refresh`** exists because the honest alternative is a six-command
incantation with unstated ordering dependencies (audio needs pitch accent,
which needs `enrich --jpdb`; example audio needs `enrich --ai`; romaji
needs readings). `refresh` runs import-less stages in the right order —
enrich --jpdb → enrich --ai → audio → build --only-new — with per-stage
skips and a summary. Independently, `build --only-new` warns with counts
when included records lack audio, examples, or pitch accent ("12 of 15 new
records have no audio — continue?"), because the quick path silently
shipping bare cards *and* marking them exported (hiding them from future
`--only-new` runs) is the worst-case version of "quickly build."

Implementation notes bound to the current code: new subcommands register in
`build_parser()`. Error handling moves to a common `JankiError` base that
`cli.main()` catches once — new error classes (`JpdbError`, `ExtractError`,
`EnrichError`, `AudioError`, `LedgerError`) subclass it instead of growing
the current except tuple (see IMPLEMENTATION_PLAN.md M1.2). Long-running
commands print progress; everything that writes shows what it wrote.

## Configuration

`janki.toml` additions (all optional, defaults shown):

```toml
[paths]
ledger_file = "data/ledger.json"
staging_dir = "data/staging"
media_dir   = "data/media"
scan_inbox  = "data/inbox/scans"

[ai]
extract_model = "claude-opus-5"
enrich_model  = "claude-opus-5"

[tts]
provider     = "voicevox"          # or "azure"
voicevox_url = "http://localhost:50021"
voicevox_speaker = 46              # int, everywhere (ledger included)
azure_voice  = "ja-JP-NanamiNeural"
azure_region = "westus2"
```

Secrets are environment-only: `ANTHROPIC_API_KEY`, `JPDB_API_KEY`,
`AZURE_SPEECH_KEY`. Config loading starts *warning* on unknown keys and
sections (today typos are silently ignored — with three new sections that
becomes an actual foot-gun).

## Costs (order of magnitude, standard API prices, Aug 2026)

- **Extraction**: a 10-page scanned handout ≈ tens of thousands of input
  tokens on Opus 5 → well under $1 per handout.
- **Enrichment**: ~2K in / ~300 out per record, per 1,000 records:
  Opus 5 ($5/$25 per MTok) ≈ $17.50, half via Batches; Sonnet 5 ($3/$15)
  ≈ $10.50 / $5.25; Haiku 4.5 ($1/$5) ≈ $3.50 / $1.75. At Genki pace
  (tens of words a week) the synchronous cost is cents either way — model
  choice only matters for big backfills.
- **jpdb API**: free with an account.
- **Audio**: VOICEVOX free; Azure free tier (500K chars/month) covers
  sentence audio for any personal volume.

The pattern: pay once per word, cache in the repo, rebuild forever for
free.

## Migration & compatibility

1. Merge-semantics change is behavioral *and* mechanical: imports stop
   overwriting curated fields; `test_merge.py` is rewritten to the new
   contract; `command_import_shirabe`'s count printing moves to the
   outcome map. `--prefer-incoming` restores old behavior per field.
2. Dataclass additions: no merge-count churn; one-time whole-file JSON
   diff on first save. External tooling parsing `vocabulary.json` must
   tolerate new keys.
3. Existing decks build identically until the new note fields ship: GUIDs,
   model IDs, deck IDs unchanged. The note-field append path is
   empirically verified against a live collection before it ships (see
   Schema changes).
4. `janki migrate-inline` moves the current inline `verbs.yaml` records
   into `vocabulary.json` (IDs preserved) so the new pipeline can reach
   them.

## Risks and open items

- **jpdb API stability**: undocumented rate limits, no ToS pages, docs lag
  reality. Mitigations: batching, backoff, defensive parsing of
  `pitch_accent`/`card_state` (formats are community-verified, not
  officially documented), file-based import as the permanent fallback.
- **jpdb pattern encoding**: whether the pattern is strictly
  kana-positional including the particle slot is community-verified only —
  the converter asserts `len(pattern) == len(reading) + 1` and refuses
  (flags) records that don't match, rather than guessing alignment.
- **VOICEVOX accent occasionally wrong even when forced** — spot-check;
  regeneration is targeted via the ledger.
- **Anki notetype upgrade on field append** — treated as unproven until
  tested (see Schema changes); this is the single biggest
  review-history-safety risk in the design.
- **Concurrent writes**: vocabulary.json and ledger.json get atomic
  temp-file+rename writes; true locking is out of scope for a single-user
  tool.

## Milestones

1. **Foundations**: curation-safe merge + per-record outcome map (+ test
   rewrite + CLI count changes); reading-is-ID-constitutive rules in
   importer and validate; ledger module + `status` (+ duplicates);
   atomic writes; config additions + unknown-key warnings; the shared
   `JankiError` base (no per-error registration); `migrate-inline`.
2. **jpdb**: API client, `import-jpdb` (API + CSV alias extension),
   `enrich --jpdb` (+ romaji converter + conjugation tables), reviews
   import + `exclude_tags` documentation, schema additions.
3. **PDF/photos**: `extract` (PDF + image inputs, staging, known-word
   annotation), `promote` with the reading-set cross-check and partial
   promotion.
4. **AI enrichment**: `enrich --ai` with mechanical QC + staging routing;
   `--polish-meanings`; batch submit/fetch.
5. **Audio**: pitch-conversion module with golden test set (merge gate);
   VOICEVOX provider (accent_phrases flow); media_dir resolution +
   passthrough warning; **notetype-upgrade verification against a live
   collection**; template updates; `build --only-new` + `refresh`; Azure
   sentence audio after.

Each milestone is shippable and useful on its own; the order front-loads
the merge fix and ID rules because everything downstream depends on them.
