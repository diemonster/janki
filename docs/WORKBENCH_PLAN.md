# Workbench Plan

The staged delivery plan for janki's browser interface — from a local page
that shows you your own corpus, through the full local workflow, to an
optional hosted edition behind a Cloudflare front door.

This file is a sibling of `docs/IMPLEMENTATION_PLAN.md`, not a replacement.
Everything in that file's **How to work this plan** and **Conventions**
sections applies here unchanged: claim a task by flipping `[ ]` to `[~]` in a
one-line commit on `main`, `make gates` is the definition of done, no live
network calls in tests, prompts stay in files, and `docs/DESIGN.md` wins when
it disagrees with this document.

`docs/DESIGN.md` currently does not describe a browser interface at all. Two
tasks below (W1.0 and H1.0) amend it. Until each lands, the code that depends
on it does not ship.

## Why staged

Every stage ends at a thing you can use for real work that day. No stage
requires the next one to exist. Stop after any stage and janki is still
coherent: the CLI keeps working, the repository stays the source of truth, and
nothing is half-migrated.

| Stage | Ships when you can... | Blocks nothing after it |
| --- | --- | --- |
| **W0** Fixtures | run the whole workflow against fake sources, offline | — |
| **W1** Dashboard | open a browser and see every source's real state | W2–W7 optional |
| **W2** Card review | read, edit, approve and re-identify staged cards in the browser | W3–W7 optional |
| **W3** Intake | drag a PDF in and pay for extraction from the page | W4–W7 optional |
| **W4** Decks + promote | assign a deck and add cards to the library | W5–W7 optional |
| **W5** Finish + build | enrich, voice, preview and build the `.apkg` | W6–W7 optional |
| **W6** Learner polish | hand the tab to someone who has never seen a terminal | W7 optional |
| **W7** Assistant | ask "why is this word waiting?" and get a real answer | — |
| **H1** Store seam | prove the file store is a swappable adapter | H2–H7 optional |
| **H2** Portability | export a corpus bundle and restore it — a real backup | H3–H7 optional |
| **H3** Cloud adapter | run the same contract tests against DO + R2, offline | H4–H7 optional |
| **H4** Front door | read your library from a phone | H5–H7 optional |
| **H5** Jobs | extract and build in the cloud | H6–H7 optional |
| **H6** Gateway | route paid calls through AI Gateway, if it conforms | H7 optional |
| **H7** Operations | back up, fork, move and delete a hosted library | — |

W1–W5 are the product. W6–W7 are polish. H1–H2 are worth doing even if you
never host anything — H2 in particular is the backup story the repository does
not currently have. H3 onward is a real cloud service; see **Before starting H3**.

## What the interface is

Workflow-first, not chat-first. Chat is good at "what does this step do?" and
bad at "which exact twelve sentences did you approve?" The second one has
durable consequences and a bill attached, so it gets a page with the sentences
on it and a button that names the action. An assistant may explain and
navigate (W7); it never holds authority.

The ordinary path is one local browser tab:

> Add a PDF, see what Japanese it teaches, improve the proposed cards, choose
> a deck, approve the exact material you studied, and build the deck.

A learner should not need to know what YAML, a stable ID, or a request
fingerprint is. Those mechanisms stay — they are what makes the collection
trustworthy — but they live under a collapsed **Source and technical details**
panel, after the learning content rather than before it.

### The words on the screen

- **Source material** — the PDF or photo you added.
- **Japanese expression** and **reading** — not "identity fields".
- **Meaning in this lesson** — not "meanings". A card for `あげる = to give`
  does not claim to list every word written or pronounced あげる. It records
  the sense *this source* taught; dictionary facts enrich the same expression
  later. This label is load-bearing and the reason the existing-wins merge is
  comprehensible to a human.
- **Polite example** / **casual example** — Japanese prominent, furigana
  attached to the characters it explains, natural English beside it, romaji
  closed by default.
- **How to use it** — the usage note: particles, register, the contrast that
  stops a predictable English-speaker mistake.
- **Grammar from this lesson** — not "pattern-store entry".
- **Study deck** — not "tag selector".
- **Needs your decision** / **Ready to add** / **Added to your collection** /
  **Ready to build** — not raw validation or ledger states.

Japanese text uses `lang="ja"` with generous line height and a size that
distinguishes small kana and lookalike kanji. Every control works by keyboard
and reports a name to a screen reader. Dark mode and a 500px-wide layout are
completion requirements, not polish.

---

## Milestone W — The local workbench

### [x] W0 Demo corpus fixtures

**Why first.** Every screen below needs material to render, and the real
corpus cannot supply it yet: per M8.5, `teform_song.pdf` returned zero
candidates and eight patterns, and all three paid runs of
`m7-mixed-tsumori-pages-27-34.pdf` returned zero candidates — the `あげる = to
give` callout that the duplicate-word screens are designed around is exactly
what the model has so far failed to produce. Building UI against sources that
do not have cards is how you ship screens nobody has seen populated.

Build a committed, offline fixture corpus under `tests/fixtures/workbench/`:
a small exhaustive vocabulary table; a lesson/dialogue with vocabulary *and*
grammar; a pattern-only chart with zero cards; a same-spelling/different-reading
pair; one stable ID proposed by two live sources with two candidate decks; a
blank and a dictionary-disputed reading; and a truncated provider answer. Each
is a captured or hand-authored response artifact plus the staging, pattern,
record and deck state it produces — never a live call.

- **Depends on:** M7.6P, M8.1–M8.4. **Not M8.5** — the workbench needs that
  code, not that milestone's paid-run evidence.
- **Files:** `tests/fixtures/workbench/` (new: `responses/*.json`,
  `decks/week-a.yaml`, `decks/week-b.yaml`, `README.md`),
  `tests/test_workbench_fixtures.py` (new).
- **Ships when:** `make gates` builds every fixture state from disk, and the
  existing `review-panel` renders the duplicate-word fixture.

**Done 2026-08-20.** Seven scenarios under `responses/`, each a raw dict
validated against the live `extract.candidate_schema()` and driven through the
real `build_records`/`write_staging`/`save_store`/`check_readings`/
`resolve_deck_records` functions (never hand-authored derived state, so the
fixtures cannot drift from the real schema): the exhaustive table (走る/食べる/
飲む); the あげる/もらう lesson with 〜てあげる/〜てもらう grammar; a
zero-candidate pattern-only chart; 一日 read いちにち vs ついたち as the
same-spelling pair; あげる proposed by two sources with two
`include_tags`-based fixture decks
(`decks/week-a.yaml`, `week-b.yaml`) that both legitimately claim the shared
ID, the way `m7-mixed-tsumori.yaml`'s `exclude_ids` already resolves for a
real overlap; 泊まる with a blank reading and 走る with a reading no dictionary
lists, held for `HOLD_MISSING_READING` and `HOLD_UNKNOWN_READING`
respectively via a fake jpdb transport; and a `max_tokens` stop that raises
`extract-response-truncated` and writes nothing. The ships-when render is
`test_duplicate_word_renders_in_the_review_panel`, which opens a panel over
*each* あげる source and proves the same stable ID is reviewable from both.
**Scope note:** no ledger
state — none of these seven reach the audio/build steps that write
`data/ledger.json`, so there is nothing yet for W5 to fixture against; add it
there if W5 needs a committed ledger shape.

### [x] W1.0 DESIGN.md: the workbench and the paid-operation journal

Two sentences DESIGN.md does not yet contain, and both are load-bearing:

1. A local browser surface exists. It is a view and controller over the same
   repository files and the same operations as the CLI; it is not a second
   database of what happened, and it may not weaken an authority gate the CLI
   enforces.
2. A **paid model operation** is journaled durably before dispatch and its
   exact response is persisted as a pending artifact before parsing — the same
   shape as the existing paid-audio write-ahead state already named in
   *Mechanisms the pipeline rests on*. State machine:
   `authorized → dispatching → running → result_captured → committed`, with
   `outcome_unknown`, `failed_before_send`, `canceled_before_send` and
   `expired` as terminal or holding states. A dispatched call whose outcome is
   unknown is never retried automatically.

- **Depends on:** none. **Files:** `docs/DESIGN.md`, `AGENTS.md` if the
  browser surface changes a stated rule.
- **Ships when:** DESIGN.md is amended and reviewed. Nothing in W1–W7 lands
  before this.

**Done 2026-08-21.** DESIGN.md gains a `## Surfaces` section — the workbench
and the CLI as views and controllers over the same repository files and the
same operations, no second database, no weakened authority gate — and
*Mechanisms the pipeline rests on* now extends the paid-audio write-ahead
shape to every paid model call (`extract`, `enrich --ai`,
`promote --accept-coverage`): journaled durably before dispatch, exact
response persisted as a pending artifact before parsing, the
`authorized → dispatching → running → result_captured → committed` machine
with its four terminal/holding states, and no automatic retry of an unknown
outcome. **Scope note:** AGENTS.md unchanged — the browser surface alters no
rule stated there; `review-panel` is already described as a localhost-only
human authority surface, which the amendment generalizes.

### [x] W1.1a Read-only `SourceJourney` projection

The first piece of `src/japanese_anki/application/`: the typed layer both the
CLI and the workbench call, so a page and a command can never disagree about
what a source is waiting for.

A read-only `SourceJourney` projection derives each source's state and one
recommended next action from repository files only — `data/inbox/`, live and
archived staging, the pattern store. Cards and grammar are independent
parallel tracks: pattern review is not a promotion prerequisite, and a
pattern-only chart reaches *Grammar saved — no word cards to build* rather
than being trapped in deck steps. Next-action priority is structural — holds,
then edits, then example review, then coverage, then add — and never
interprets Japanese.

- **Depends on:** W0, W1.0. **Files:** `application/` (new: `__init__.py`,
  `authority.py`, `journey.py`), `review_panel.py`,
  `tests/test_application_journey.py` (new).
- **Ships when:** `SourceJourney` reproduces each W0 fixture's state.

**Done 2026-08-21.** `application/authority.py` holds
`example_authority_state()` and `needs_example_review()` — has a human
approved the exact Japanese sentences currently on this card? — lifted from
where it lived privately in `review_panel.py`; `review_panel` now imports it,
so there is exactly one implementation, and all 2,449 pre-existing tests
passing unchanged is the proof the move was behaviour-preserving.
`application/journey.py` implements nine card-track states — `In corpus —
not yet extracted`, `Some cards held for a reading decision`, `Cards need
edits`, `Examples need review`, `Coverage needs a decision`, `Ready to add`,
`Added — dictionary/audio/build steps remain` (derived from an archived
staging file, so a promoted source whose live staging was pruned stays in the
queue with "Add dictionary facts, audio, and build the deck" as its next
action instead of vanishing), `Grammar saved — no word cards to build` and
`Needs attention — this source's file could not be read` — plus the
independent grammar badge: none / `Grammar needs review` / `Grammar reviewed`
/ `Grammar review state unknown`. The inbox walk consults the pattern store
before calling a file unextracted, so a source read for grammar alone —
reviewed patterns in the store, its zero-record staging archived and pruned —
is never offered a paid re-read of an answer the repository already holds.
`tests/test_application_journey.py` drives the projection off the W0 fixtures
through the real pipeline; of eight mutations run against the implementation,
the three initial survivors — the stale pattern-run guard, the archive
branch, and the live-file guard, which only matters when staging metadata
names a source differently from its filename, as the real Yotsubato pack
does — each exposed a genuinely untested guard and got its own test.
**Scope note:** five planned states are deliberately not implemented because
their machinery does not exist yet: `Waiting for paid extraction consent`,
`Extraction running` and `Recovery needed` all need W3's paid-operation
journal, `Deck needs a decision` needs W4.0's deck model, and `Deck built`
needs W5's ledger export queries. They are absent rather than guessed —
inventing `Extraction running` from a file mtime would lie at exactly the
moment it matters. The projection walks `scan_inbox`, so it reports every
source document in the corpus; a staging file with no inbox counterpart (the
Yotsubato Anki-import pack, `ai-enrichment.yaml`) still appears via its own
staging metadata — intended, not a gap. And the `table_exhaustive` W0 fixture
materializes with coverage `status=unmeasured`, not exhaustive, because the
fixture runs in auto mode. That is useful — it is what exercises `Coverage
needs a decision` — but it means no W0 fixture currently exercises the
exhaustive-coverage path; W4 should add a `mode="table"` materialization if
it needs one.

### [ ] W1.1b Factor the CLI orchestration into the shared services

Factor the orchestration currently inlined in the 5,153-line `cli.py` —
extract, validate, pattern review, promote (`command_promote` alone is 379
lines), the targeted finish steps — into typed services under `application/`.
The CLI becomes a thin caller. Command output and exit codes are unchanged,
proven by the existing tests.

Also add two pure previews with no writes: deck membership for a staged
record, and a promotion preview using the same merge and identity rules as
`promote`.

Not required by W1.2 — the read-only dashboard needs only `SourceJourney` —
but required before W3 and W4, when the browser starts triggering extraction
and promotion.

**Deliberately not here:** the `CorpusStore` seam. It arrives in H1, once
there is a second adapter to justify its shape. Do not design it speculatively.

- **Depends on:** W1.1a. **Files:** `application/`, `cli.py`, `staging.py`,
  `patterns.py`, `promote.py`, `status.py`,
  `tests/test_application_services.py` (new).
- **Ships when:** `janki status` and every existing command still pass their
  tests unchanged.

### [x] W1.2 `janki workbench` — the read-only dashboard

`janki workbench [--no-open]`, bound to `127.0.0.1` on a random port,
server-rendered HTML plus modest local JavaScript. No frontend build
toolchain, no remote assets.

Reuse and generalize `review_panel.py`'s boundary wholesale: unguessable
session token, exact Host and Origin checks, one-session CSRF tokens, CSP and
`no-store` headers, HTML escaping, request size limits, `O_NOFOLLOW` file
capture, sorted lock acquisition, exact-byte snapshots, compare-and-swap
writes, and bounded shutdown. **Private GETs require the live session too**,
not only writes. Source files render in a viewer that cannot inherit workbench
action authority.

The first dashboard is read-only: the source queue with counts per track and
the recommended next action, deep-linking into the existing `review-panel` for
approval. It reads the paid-operation journal and surfaces `Recovery needed`
and `outcome_unknown` without offering a retry it cannot prove is safe.

The dashboard says **Saved on this computer**, never **Backed up**, when
durable files have not been committed or copied elsewhere.

- **Depends on:** W1.1a. **Files:** `workbench/` (new), `review_panel.py`,
  `cli.py`, `tests/test_workbench.py` (new).
- **Ships when:** you open a tab, see all eleven decks' sources with honest
  states, and click through to review one.

**Done 2026-08-21.** The boundary moved first: `localhttp.py` (new) holds what
`review_panel.py` had proven under adversarial review — `LocalOnlyServer`
(non-daemon handler threads, so close waits for an in-flight write),
`LocalOnlyHandler` (the full response-header set on every reply including
errors, the exact single-`Host`/`Origin` check, escaped error pages) and
`bind_loopback` — and `review_panel` now inherits it, so both surfaces share
one implementation; the full pre-existing suite passing unchanged is the
proof the lift was behaviour-preserving. A second copy of a security header
list is a copy that drifts silently. `workbench/server.py` is `janki
workbench [--no-open]` on `127.0.0.1` at a random port, adding the one thing
a long-lived page needs beyond the panel's boundary: an unguessable session
token gates every request, reads included, compared with
`secrets.compare_digest`. Loopback is a network boundary, not a permission —
every local process can reach the port, and this page renders private study
material. The token lives in the URL *path*, not a cookie, because cookies
on `127.0.0.1` ignore the port and would be sent to every other local
server; and a wrong token returns the same 404 body as an unknown page, so a
local process cannot confirm a session exists. `workbench/render.py` is
server-rendered HTML — no script tag, no remote asset, no build toolchain —
on system colours (`Canvas`/`CanvasText`/`Highlight`), so light, dark and
forced-colours modes all work without a theme switcher. Read-only by
construction: there is no write route at all — POST returns 405 — and that
is asserted, not just asserted-about-the-buttons. Approval still runs
through `janki review-panel`, which the page names as a command rather than
links to, since a link this page could follow would be the page taking an
action. `tests/test_workbench.py` covers the boundary — the token gates
reads, a wrong token is indistinguishable from a 404, a foreign Host and a
cross-Origin request are refused, the restrictive headers are on every
response including errors, no script or remote asset, no write route,
non-daemon close — and the rendering: Saved-on-this-computer never
Backed-up, both tracks visible, a filename cannot inject markup, an
unreadable file is shown rather than dropped, no machinery vocabulary, a
disk change visible without a restart. Seven mutations were run; all are
caught. **Scope note:** the task text above says the dashboard "reads the
paid-operation journal and surfaces `Recovery needed` and
`outcome_unknown`". That journal does not exist yet — it is W3's. Nothing
was faked in its place; those states simply do not appear. And one bug was
found by rendering the real corpus rather than the fixtures: the header
counted `Added` as waiting on the person but `Ready to add` as finished,
understating the work ("3 waiting" when 5 were).
`SourceJourney.needs_a_person` now treats every state except a finished
pattern-only source as waiting. The first regression test written for it did
not actually isolate the bug — the source's grammar also needed review, and
that clause rescued the old definition — so the test now marks the grammar
reviewed first.

### [x] W2a Read-only card and grammar view

The card editor shows, in learning order: expression, reading and furigana;
**Meaning in this lesson**; polite and casual examples with furigana, natural
English and register labels; **How to use it**; source page and source
sentence. Pattern content is display-only.

When the expression already exists, show three meaning panels — **Proposed by
this lesson**, **Currently on your card**, **What will remain after adding** —
so the existing-wins merge is visible rather than implied.

- **Depends on:** W1.2. **Files:** `application/detail.py` (new),
  `workbench/render.py`, `workbench/server.py`, `tests/test_workbench.py`.
- **Ships when:** you open a staged source and read every card and its
  grammar in learning order, on a page that can change nothing.

**Done 2026-08-21.** `application/detail.py` holds `source_detail()`,
`SourceDetail` and `CardDetail` — a read-only projection of one opened
source: staged cards, per-card approval state, hold reasons, the grammar set,
and the merge preview. The preview calls `promote.merge_staged_records` — the
real merge — against a copy rather than reimplementing existing-wins, because
a preview that disagrees with what promote does is a lie with a progress bar;
`CardDetail.proposed_meanings_would_be_kept` is the flag the page uses to say
"your card already has meanings, so adding this lesson keeps them".
`workbench/server.py` gains a `/source/<name>` route and `render.py` a
`render_source`: Japanese rendered large with generous leading, furigana
attached, romaji closed by default; the stable ID and source evidence sit in
a collapsed block *after* the learning content. Grammar renders display-only
— no `<input>`, no `<textarea>`. Source lookup is by exact match against the
dashboard's own computed journey list, and the staging path comes from that
journey — a request never names a path. **Scope note:** a path-traversal test
was written with a docstring claiming it would fail if the lookup were
"simplified" into a join against `staging_dir`. That claim was false — the
substitution was made and all tests still passed, because the name-match
refuses traversal names first, so the path handling was never under test. The
discriminating test is `test_a_source_is_opened_by_name_not_by_filename`,
drawn from the real corpus: the Yotsubato pack is `source_file: Yotsubato
Volume 1 Reading Pack Vocab` inside `anki-yotsubato-….yaml`, so opening a
source cannot mean `staging_dir / (name + ".yaml")`. The traversal tests now
state what they do and do not prove.

### [x] W2b Approval writes, and `review-panel` deleted

Approval is exact and narrow. The checkbox reads **Approve these Japanese
example sentences** and repeats the expression and reading; the text beside it
says it does not approve the meaning list, translation, identity or usage
note. Every sentence is shown in full. Changing, adding, removing or
reordering an example voids that card's approval. **Needs help** defers a
sentence without blocking unrelated cards and without asking a model to grant
human authority. Grammar is approved separately, over the complete extracted
set with each template, explanation, example and source page visible. There is
no **Approve all**.

Reuse the existing example fingerprint and exact pattern-set review methods
rather than reimplementing them. A stale page refuses with no partial edit;
monotonic approval writes report precisely if only one file landed.

**Delete `janki review-panel` in the same change** — pre-release, there are no
legacy paths. The loopback boundary is already shared — since W1.2 it lives in
`localhttp.py` — so what moves with the fold is the panel's write machinery:
one-session CSRF tokens, request size limits, `O_NOFOLLOW` file capture,
sorted lock acquisition, exact-byte snapshots and compare-and-swap writes.

- **Depends on:** W2a. **Files:** `workbench/`, `localhttp.py`,
  `review_panel.py` (folded in), `staging.py`, `patterns.py`, `cli.py`,
  `README.md`.
- **Ships when:** you review a real staged source end to end in the browser
  and the CLI review path is gone. This is the whole W2 stage's ship line.

**Done 2026-08-21.** Approval now happens in the workbench. A card's checkbox
reads "Approve these Japanese example sentences for X (Y)", and the sentence
stating what it does *not* cover — not the meanings, the English, the spelling
and reading, or the usage note — is inside the label being clicked, not in
help text. Grammar has its own separate checkbox recording that a person read
the extracted set. There is no Approve all. Two separate secrets guard the
page: the path token gates reading it, and a session CSRF token in the form
body gates writing. They defend different things — one stops another local
process reading the page, the other stops a page the browser was tricked into
submitting — and unlike the deleted panel's, the CSRF token is not one-shot,
because a dashboard approves many sources in a row. The snapshot is the
compare-and-swap: the form carries the exact staging and pattern fingerprints
the page was rendered from, and a form rendered against older bytes is
refused whole with 409 and nothing written. Post/redirect/get, so a reload
cannot resubmit. `janki review-panel` is deleted — the command, its parser,
its own HTTP server and handler, its own CSS and HTML rendering, and its
one-shot CSRF. `review_panel.py` moved to `workbench/review.py` carrying only
the write transaction: exact-byte capture, `O_NOFOLLOW`, sorted lock
acquisition, compare-and-swap writes, and the partial-write reporting. Net
−2,139 / +897 lines. README, `AGENTS.md` and `DESIGN_V2.md` all repointed to
`janki workbench`. Of the panel's 41 tests, 19 covered the write machinery
and moved to `tests/test_workbench_review.py`; 22 covered the deleted
HTTP/HTML surface and went with it. Four gaps that deletion would have left
were closed in `tests/test_workbench.py`: `Origin: null`, duplicate `Host`
headers, a duplicated form field, and two concurrent approvals of one source.
**Scope note:** two findings, both of which are the useful part. First,
concurrency works differently here than in the panel: the workbench opens a
*fresh* panel per request, so the panel's in-process submission lock cannot
serialize two requests the way it did for a single long-lived page. What
protects the file is the advisory lock plus the exact-byte snapshot — the
request that loses the race finds the bytes changed and is refused.
`test_two_concurrent_approvals_land_exactly_once` pins that one lands and one
409s. Second, the write path has two independent guards, and a mutation
proved neither alone is the whole story: removing `submit()`'s early byte
re-check left every test green, because `_bound_replace`'s own
compare-and-swap (`expected_revision`) still refused the write and the file
was not corrupted — so that was redundant defense-in-depth, not an untested
guard. Removing *both* fails four tests. Recorded so a future reader does not
delete the "redundant" check believing it is dead.

### [x] W2c Field edits, removal, undo before save

Grow the review snapshot into safe field edits, card removal, and undo before
save. Round-trip and surgical writers preserve every unedited byte, comment,
provenance block and unrelated pattern entry — the reason `ruamel.yaml`
exists in this repository. Source evidence and machine-owned accounting are
read-only in the form.

- **Depends on:** W2b. **Files:** `workbench/`, `staging.py`, `patterns.py`.
- **Ships when:** you fix a gloss, remove a card and undo it in the tab, and
  every byte you did not edit survives unchanged.

**Done 2026-08-22.** `?edit=1` renders textareas for the human-owned fields,
and saving goes through the same compare-and-swap as approval. Editable is an
**allowlist** — meanings, the usage note, and five example fields. Source
page, source sentence, inclusion reason, confidence and the accounting block
have no control at all: a form that let someone retype the evidence would let
them retype history. An unrecognized form field is refused, not ignored.
Editing and approving are separate views; one form carrying both would let a
stray click approve sentences someone was only correcting. Undo-before-save
is a `type=reset` button — no JavaScript, no round trip — plus a "Leave
without saving" link. Register is a `<select>`, not free text: anything
outside polite/casual makes an example incomplete, so a typo would quietly
degrade a card. Card removal drops one proposed row through
`render_staging_prune` plus the same bound write, and the control says the
deletion cannot be undone and that re-reading the source would be another
paid call, because the row is the model's output and nothing else in the
repository holds it once it leaves live staging. **Scope note:** four
findings, all of which are the useful part. First, a `<select>` destroys
what it cannot represent: a register value outside its options rendered with
nothing selected, so the browser fell back to the first entry and merely
opening the editor and saving erased it — the fix for one silent-degradation
risk had created another. Unrecognized values now get their own option and
survive a round trip. Second, every save was rewriting every card:
`apply_edits` rebuilt examples as a tuple while `VocabularyRecord.examples`
is declared a list and `to_dict` passes the container through, so the
changed-record comparison was always true. Every row got rewritten,
re-folding long scalars the editor does not expose, including
`inclusion_reason`. The guard now is a test that resubmits the entire editor
verbatim and demands byte-identity. Third, the removal buttons would not
have worked in a browser: they were first rendered as a `<form>` nested
inside the edit form, and forms cannot nest, so a browser silently drops the
inner one. They now use the HTML5 `form=` attribute with the forms declared
after the editor's form closes, and a test pins that no `<form>` opens
inside the editor. Fourth, the two YAML writers disagreed: staging files are
created by PyYAML and edited by ruamel, and both wrapped at width 100 but
chose different break points, so a bare load-and-dump with nothing edited
changed 54 lines of a real staging file and left trailing whitespace. That
made `rewrite_staging`'s "byte-for-byte as it was" promise untrue and
churned every promotion's diff, since `promote` calls both `rewrite_staging`
and `prune_staging`. Both writers now read one `STAGING_YAML_WIDTH` constant
and neither folds. The two live staging files were reformatted once, with
every record and metadata value read back and asserted identical; the twelve
archives under `data/staging/done/` were deliberately left alone as
immutable promotion evidence that nothing rewrites.

### [x] W2d Deliberate re-identification

A deliberate **Re-identify this draft word** flow. Re-identification is a
separate flow because changing `とまる` to
`泊まる[とまる]` answers a different question from fixing an English gloss. It
shows old and new, any same-reading or same-spelling neighbours, and the study
history consequence, before saving. It never proposes an identity itself. V1
re-identifies **staged rows only**; migrating a canonical identity is out of
scope.

- **Depends on:** W2c. **Files:** `workbench/`, `application/`, `staging.py`.
- **Ships when:** a staged `とまる` becomes `泊まる[とまる]` in the tab, with
  neighbours and the study-history consequence shown before the save.

**Done 2026-08-22.** `workbench/reidentify.py` plans a change without
writing: the new stable ID, the neighbours it would sit beside — same
identity, same reading, same spelling, in this source and in the collection —
and what happens to review history. The route makes two passes: the first
submission renders the consequences, and only a second carrying the exact
identity the preview computed actually writes. Binding the confirmation to
that string stops the form drifting between the page someone read and the
change they authorised. Because the Anki GUID derives from the record ID, a
card that already shipped becomes a different note under a new identity —
the old note keeps its review history and the new one starts from zero.
`Ledger.ever_exported` (new) answers that, asking about any deck rather than
one, which is what `unexported` is for. A collision refuses: the preview
shows it and offers no confirm button, and a forced confirmation is still
refused. A blank reading cannot become an identity — that is exactly what
the promote gate holds a row back for. It never proposes an identity: the
form is prefilled with what the card already says, and deciding a kana
spelling "should" be particular kanji is reading Japanese, which
`docs/DESIGN.md` reserves for the model and the person. Approval and
sentences are untouched — approval covers the exact Japanese, and that does
not change when the identity does. **Scope note:** five mutations were run
against the route and two initially survived — the CSRF check and the
snapshot check had no tests, because coverage from the approve, edit and
remove routes does not carry over to a new one. Both now have tests. V1
re-identifies staged rows only; migrating a canonical identity has to move
review history and rewrite what the ledger says shipped, and remains out of
scope.

### [ ] W3 Intake and the extraction job

Drag or pick a file; preview its pages; see the permanent filename; save the
immutable copy under `data/inbox/`. A conflicting basename refuses in plain
language: *"A different source called `lesson-3.pdf` is already in your
corpus. Rename this file before continuing."*

**Adding a file to the corpus and sending it to a model are two separate
actions, always.** The consent button names exactly what leaves the computer:

> Send `Genki Lesson 8.pdf` to Claude Opus 5 to propose vocabulary cards and
> grammar — paid API call

with a separate sentence saying the immutable copy stays local, that only this
named source is being sent, that the call consumes Anthropic API credits, and
that Claude Max is a different subscription. Closing a dialog, pressing Enter,
or an unattended request never grants consent. Replacing an existing
extraction is separately confirmed and names the review it invalidates.

**Page selection is request scope.** Whole document, or an exact inclusive
range. The displayed pages, source hash, ordered page numbers, exact provider
input, cost disclosure, provenance and retry identity are one `SourceSelection`
value. Until that contract exists the UI offers **Whole document** only — it
never sends a whole PDF while implying that pages 27–34 left the computer.
Mode defaults to **Choose automatically**; *Vocabulary table* and *Lesson,
dialogue, or exercise* sit under an advanced choice with examples rather than
the internal words `table` and `prose`.

Progress shows human steps — **Preparing pages**, **Reading the source**,
**Checking the answer's shape**, **Saving proposals** — and never invents a
percentage the provider does not report. Before dispatch, journal the one-use
authority, selection, request fingerprint and an operation ID shared with the
CLI. Persist the exact response as a pending artifact **before** parsing, so a
restart finishes without a second call. Lost contact after dispatch is
`outcome_unknown`: inspectable, never auto-retried. A refusal, truncation or
timeout says plainly that nothing was staged.

One mutating job at a time. A browser click calls the shared service with a
bound action token; it never manufactures `--yes`. The CLI uses the same
journal and the same recovery path. Every provider response in tests is fake.

- **Depends on:** W2a–W2d, W1.0. **Files:** `workbench/`, `application/`,
  `extract.py`, `inputs.py`, `ledger.py`, `cli.py`.
- **Ships when:** you drag a PDF in and get staged cards without a terminal,
  and killing the process mid-call leaves a state you can safely resume.

### [ ] W4.0 Decide the deck-assignment model

**Open design question — resolve before W4.1.** The current corpus is *one
deck per handout*: each of the eleven files in `data/decks/` selects on a
source-slug `include_tags` entry, `m7-mixed-tsumori.yaml` carries `exclude_ids`
for the one overlapping record, and `tests/test_anki_builder_contract.py`
asserts the partition over nine word decks. That convention answers "where does
a new PDF's vocabulary go?" with "a new deck named after the handout."

The workbench journey assumes something different: durable thematic decks a
learner picks from ("Week 3"), with a per-deck `intake_tag` that assignment
writes. `intake_tag` exists nowhere in the codebase today. Adopting it is a
deck-schema migration across all eleven files plus the builder and the contract
test — not a clause in a UI task.

Pick one and write it into `docs/DESIGN.md` beside the existing
*"no record belongs to more than one word deck"* sentence:

- **(a) Keep per-handout decks.** Assignment means creating a deck for the new
  source. Cheapest, matches the corpus, and the "two sources propose あげる"
  case stays an `exclude_ids` decision. The picker becomes *name this lesson's
  deck*, and the plan's "Week 3" language is wrong.
- **(b) Adopt `intake_tag` thematic decks.** Every assignable word deck
  declares one nonblank, unique `intake_tag`, present in its includes, absent
  from its exclusions, unable to select another word deck. Migrate the eleven
  files. More work; matches how a learner actually thinks about study decks.

- **Depends on:** W3. **Files:** `docs/DESIGN.md`, `data/decks/*.yaml` if (b),
  `exporters/anki.py`, `tests/test_anki_builder_contract.py`.
- **Ships when:** DESIGN.md names the model and the contract test enforces it.

### [ ] W4.1 Deck assignment, validation and promotion

A deck picker previews where each card lands and checks the real selectors,
exclusions and existing records — the same partition test a build runs. It
never guesses which tag means "put this in that deck"; it shows the exact tag
diff under Technical details. If `あげる` is also waiting in a Yotsubato
import: *"This word is proposed by two sources. Choose which word deck should
teach it; both sources can still remain in its history."* Offer the change as
a reviewable diff. Never delete the other source row or invent an exclusion
silently. A minimal explicit deck creator asks for the learner-facing name and
which card directions to enable (recognition on by default), previews the
result, and runs behind one **Create study deck** button. The workbench never
invents a deck because a file was uploaded.

**Check these cards** runs structural validation and renders each result as an
action beside the relevant field: *"Add a reading"*, *"Review both Japanese
examples"*, *"Choose a study deck"*, *"This reading is not listed by jpdb;
keep it for another decision."* Error codes stay in details, copyable.

Before promotion, a plain preview: cards to add, cards matching an existing
word, cards expected to stay held, source history to record, deck membership
after the merge. **Add 6 cards to your collection** is distinct from validation
and from coverage. Coverage is explained in one sentence — *"Did the extraction
account for the source units it promised to cover?"* — with the accounting
shown and two separately named routes: **I compared the source rows myself**,
which records the owner's scoped reason, and **Ask Claude to check
completeness**, which is a separate paid call over the named source using the
same journal as W3. Neither is described as approval of the Japanese.

External reading checks run in the same order as the CLI. Live and archive
locks, candidate-accounting limits, request lineage, ledger attribution and
retry behaviour remain the one shared promotion transaction. A browser failure
cannot land canonical rows where the same refusal would have stopped the CLI.

- **Depends on:** W4.0. **Files:** `workbench/`, `application/`, `promote.py`,
  `validation.py`, `coverage.py`, `collection.py`.
- **Ships when:** a source goes from staged to promoted entirely in the tab.

### [ ] W5 Targeted finish and build

Carry the exact newly promoted ID set forward as the scope, and offer the
remaining steps in order: **Add dictionary facts** (jpdb POS, furigana, pitch,
frequency) · **Add kanji reference** for newly introduced characters ·
**Create word audio** and **Create example audio**, each naming the local or
paid provider and the clip count · **Preview the cards** · **Build the Anki
deck**, ending on the `.apkg` path with the existing sync-first and Merge
Notetypes import guidance.

Say which steps are local, networked or paid before they start. Reuse the
audio write-ahead recovery transaction and the ledger; an interrupted exact
request never rebills.

Do not hide `refresh`'s corpus-wide behaviour behind a deck-shaped button. The
ordinary path scopes enrichment and audio to the records just added and scopes
build to the chosen deck; whole-collection maintenance stays separate.
`enrich --ai` remains the path for bare records from non-extraction imports,
not a routine second pass over complete extraction cards.

End on a deck summary: new cards, enabled directions, audio coverage, output
path, import steps.

- **Depends on:** W4.1. **Files:** `workbench/`, `application/`, `enrich.py`,
  `kanji.py`, `audio_cmd.py`, `preview.py`, `exporters/anki.py`.
- **Ships when:** PDF to `.apkg` without touching the terminal, YAML or JSON.

### [ ] W6 Learner polish, accessibility and documentation

Replace the README's PDF happy path with a short illustrated workbench guide,
keeping a CLI reference for automation and recovery. Explain "meaning in this
lesson", polite/casual examples, reading holds, grammar review, coverage and
deck ownership without implementation vocabulary. Add a dismissible,
reopenable first-run tour.

First-run setup explains each provider and whether it is local, networked or
paid. A missing key never sends the source. Keys never enter repository files,
browser storage, URLs, logs or error pages; the existing environment-variable
path stays the supported one. **An OS-credential-store entry path and a
double-clickable launcher are explicitly out of scope for v1** — both are
packaging projects, and the current owner already keeps keys in the shell
environment. Reopen them only if a real non-terminal user appears.

Every failure message answers four questions: what happened, what changed,
whether money may have been spent, what to do next. It must never say "nothing
changed" after a partial or uncertain write.

**Completion gate** (replaces the eight-learner study, which becomes optional
future work):

- The owner completes one vocabulary-table source and one lesson/dialogue
  source end to end in the browser — intake, review, deck, promote, enrich,
  audio, build — without editing a repository file or running a command.
- With the same selection, decisions, configuration and fixture response, CLI
  and workbench produce identical durable artifacts, authority marks,
  provenance, merge outcome, ledger, media and deck membership. Build
  comparison is semantic where genanki timestamps prevent byte equality.
- No private source or paid request leaves the machine without a direct action
  naming the file, provider, model, purpose and cost-bearing operation.
- The dashboard fully reconstructs from repository state after a restart, and
  every interrupted paid write exposes the CLI's recovery path.
- Mechanical accessibility pass: full keyboard path with visible focus, an
  accessible name on every control, 500px width, dark mode, Japanese font
  rendering, furigana attachment, long-English wrapping.
- At least one end-to-end journey runs in a real browser. HTTP unit tests
  cannot establish Origin behaviour, PDF isolation, focus order,
  double-submit handling, or the localhost boundary.
- `make gates` stays offline with fake providers.

- **Depends on:** W5. **Files:** `README.md`, `docs/IMPORTING.md`,
  `docs/QUALITY.md`, `workbench/`, browser test.

### [ ] W7 Optional "Ask janki" assistant

Only after W6 passes. The workbench must be fully useful with this disabled,
and its off state makes no request and leaves no broken space in the UI.

Read-only tools only: show a source or card's deterministic state and deck
membership; explain a validation code without judging the Japanese; show
non-sensitive provider/model/time/hash metadata for a past call and link to a
details panel that renders sent content *outside* chat; navigate to the screen
where the learner can act.

There is no assistant tool for editing card or pattern content, paid consent,
`--force`, example approval, pattern review, identity resolution, deck
ownership, coverage acceptance, promotion, audio purchase or deletion. The
PDF, page images and staged card text are not sent to the assistant provider
merely because the drawer is open, or because the source was sent to the
extraction provider. Switching sources carries no context forward. Threads are
convenience context, never provenance or workflow state.

Keys stay server-side. OpenAI API billing is described separately from
ChatGPT, and Anthropic API billing separately from Claude Max.

For an OpenAI-backed build, use a small Responses API drawer or ChatKit's
custom-server integration; do not build on the retiring Agent Builder path.
Widgets are fine for status and navigation; authoritative controls stay
workbench actions handled by janki
(`https://developers.openai.com/api/docs/guides/chatkit`,
`https://developers.openai.com/api/docs/guides/custom-chatkit`).

A general Japanese tutor, if ever offered, is a visibly separate conversation
with no repository tools and no source context. Card improvements still come
from a human edit or from janki's two existing card-writing calls. Adding a
chat UI must not create a third Japanese card-writing path.

- **Depends on:** W6. **Files:** `workbench/`, `config.py`, new assistant
  module, `docs/QUALITY.md`.

### Adversarial coverage for milestone W

Every machinery defect gets a red test and mutation proof. Providers are faked;
no test spends money or sends private bytes. Journeys, using the W0 fixtures:
exhaustive table; the same multi-page source as whole-document and as pages
27–34, proving different request identities and disclosures; lesson with
vocabulary plus grammar; pattern-only chart; empty, invalid and truncated
answers that write nothing; blank and dictionary-disputed readings;
same-spelling/different-reading; one stable ID from two sources and two decks;
a new deck created with explicit direction choices; an edit made while an older
page is open; interruption before dispatch, after dispatch with unknown
outcome, after capture before staging, after archive creation, and during paid
audio finalization; assistant disabled, network-failed, and carrying malicious
text, plus proof that authority-bearing assistant tools do not exist.

Security: malicious filenames and every displayed model or source string;
Host/Origin/CSRF failures; oversized, duplicate and unknown actions; path
traversal; symlink and inode replacement; stale source, staging, pattern and
deck bytes; concurrent CLI and workbench writers; stalled requests; browser
disconnects; terminal partial writes. Each refusal proves which repository
files stayed byte-identical. **The UI may make an operation easier to invoke;
it may not weaken a single authority or provenance gate.**

**Milestone W non-goals:** AnkiConnect sync; hosting; remote or mobile access
to the local repository; automatic deck YAML creation; an exhaustive dictionary
editor; a second model pass auditing Japanese; automatic identity, deck,
approval or promotion decisions; replacing the CLI; migrating a canonical
identity; automatic Git work; editing anything under `data/inbox/`.

---

## Milestone H — The hosted edition

### Before starting H3

H1 and H2 are worth doing on their own merits. H3 onward is a different kind of
commitment, and these questions get written answers in this file before H3 is
claimed:

1. **What does it cost?** Cloudflare Containers require a paid Workers plan.
   Write the expected monthly floor for one user — Workers, Containers, DO,
   R2 storage and egress — beside the value of not needing a laptop open.
2. **Who is it for?** One corpus per account still means accounts. That
   implies signup, account recovery, a privacy policy, and an answer to "who
   else can create an account". If the answer is "only me", say so and use
   Cloudflare Access in front of the Worker — it collapses most of H4.
3. **What kills it?** Name the abandonment criteria now: a cost threshold, a
   Cloudflare limit H3 discovers, or a date. "Optional" is not a criterion.

### [ ] H1.0 DESIGN.md: storage authority

`docs/DESIGN.md` currently says *"The repository is the durable source of
truth"* — and so do `AGENTS.md` and the README. The hosted edition cannot ship
under that sentence. Amend all three together:

- A **local library** is authoritative in its Git repository.
- A **hosted library** is authoritative at one committed Durable Object
  revision together with the immutable R2 objects its manifest references.
  Container disk, unreferenced R2 objects, Workflow state, projections,
  previews and `.apkg` files are never authoritative.
- Every corpus has a lineage ID, an authority mode, an authority epoch and a
  revision, and exactly one authoritative home. **There is no automatic
  two-way sync.** Copy, fork, back up and move are four distinct operations.

Also enumerate the three classes of guarantee, because conflating them is how
a migration quietly loses data:

- **Portable corpus invariants** — stable IDs and ordering; exact
  Japanese/reading/furigana/English values; missing-versus-empty distinctions;
  approval fingerprints; source and request hashes; authority and provenance;
  collision proposals; unknown extensions; media hashes; deck semantics.
- **Local file invariants** — exact byte snapshots, comments, path layout,
  advisory locks, archive and prune behaviour, Git history.
- **Hosted invariants** — entity read-set revisions, SQL transactions,
  immutable object manifests, operation states, authority epochs, backup pins.

Portable fingerprints use a versioned canonical wire encoding and never depend
on SQL row order, locale or collation.

- **Depends on:** W6. **Files:** `docs/DESIGN.md`, `AGENTS.md`, `README.md`.

### [ ] H1 The `CorpusStore` seam, proven on files

Introduce a versioned `CorpusStore` boundary beneath the W1.1a/W1.1b
application services, with `FileCorpusStore` as the only implementation. The contract is
**behavioral, not a key/value wrapper**: snapshot a source journey, save
human-owned draft fields against an expected entity revision, bind exact-example
approval, record paid-call authority, apply a promotion change set,
reserve and finalize audio, export a complete corpus. Japanese decisions live
in prompts and human actions, never in an adapter.

Move the services onto it with **no observable change** to CLI or workbench
files, ordering, fingerprints, recovery or deck output. Build portable contract
tests plus adapter-specific tests, and mutation-test the contract. This is the
task that proves the boundary was extracted rather than a cloud-only parallel
application being designed.

- **Depends on:** H1.0. **Files:** `application/`, `storage/file_store.py`
  (new), `tests/test_corpus_store_contract.py` (new).
- **Ships when:** every existing test passes untouched against the adapter.

### [ ] H2 Export, import and portability

A documented, versioned corpus bundle containing all source bytes, canonical
learning content, review authority, provenance, patterns, decks, media,
ledger and recovery state, and checksums. Round-trip it: export a local
corpus, import into an empty target, prove stable IDs, GUID inputs, source and
request fingerprints, exact approvals, candidate accounting, patterns, ledger
attribution, media hashes, deck membership and a sample deck are equivalent.

Import preserves ordered collision proposals, source raw fields, unknown
extension data and human annotations. A value that cannot round-trip makes
import **refuse**, not normalize. Defend against traversal, duplicate members,
decompression bombs, truncation, substituted objects, unknown schemas and
lossy extensions before any mutation. Bundles contain no secrets. An ordinary
import cannot merge into a live corpus or create two authorities sharing a
lineage and epoch.

**This is the backup story the repository does not currently have.** It is
useful with no cloud at all.

- **Depends on:** H1. **Files:** `storage/`, `application/`, `cli.py`,
  `docs/IMPORTING.md`.
- **Ships when:** `janki export` produces a bundle you would trust as your
  only copy, and importing it reproduces the corpus.

### [ ] H3 Durable Object + R2 adapter, tested offline

`DurableObjectCorpusStore`: versioned SQLite migrations; first-class tables for
sources, expressions and readings, meanings, examples, patterns, decks, exact
approvals, extraction and enrichment runs, candidate accounting, operation
authority, model attribution, media requests and builds. JSON may remain a
versioned wire format or an immutable copy of an exact provider artifact — it
is not the cloud database.

Large bytes always live in R2, private, content-addressed within a corpus
namespace, reached only through short-lived operation-bound access. Three
lifecycle classes with separate prefixes and policies:

- **Authoritative** — preserved sources, exact provider request/response
  artifacts required by provenance, promotion and archive evidence, canonical
  media referenced by the committed manifest.
- **Pending/recovery** — uploads and paid outputs not yet committed, protected
  while an operation, recovery window, export or backup can claim them.
- **Derived cache** — thumbnails, previews, `.apkg` files. Evictable,
  rebuildable, never part of corpus recovery.

**DO SQLite and R2 cannot form one transaction.** Every cross-store operation
follows this protocol: (1) in a DO transaction reserve a one-use operation with
its authority, epoch, entity read set, intended artifact kind and expiry;
(2) write a new immutable R2 object with conditional-create semantics under a
server-derived, corpus-scoped, content-derived key — if the key exists, verify
length, type and hash rather than overwriting; (3) in one DO transaction bind
that manifest to the reserved operation and commit, if epoch and read set still
match; (4) a lost finalization race leaves the unreferenced object for bounded
GC and never overwrites a referenced object; (5) delete a blob only after the
DO proves no source, review, paid request, media row, export, backup, recovery
window, snapshot or in-flight job claims it.

**Read sets, not one global CAS.** Every `CorpusSnapshot` declares the exact
source, records, pattern set, deck configuration, authority grant and artifact
manifests it read, with their revisions and hashes. Unrelated work may advance
while a model runs; only a changed read-set member conflicts, and a conflict
**retains the exact paid output** as a protected recovery artifact for
no-charge restaging. A monotonic global revision exists for audit and quiescent
export, not as a giant compare-and-swap token.

Test everything offline against a local DO/R2 simulation plus the H1 contract
tests. Recheck current per-object and per-DO limits at implementation time
(`https://developers.cloudflare.com/durable-objects/api/sqlite-storage-api/`,
`https://developers.cloudflare.com/r2/reference/consistency/`).

- **Depends on:** H2, and the three questions above answered in writing.
- **Ships when:** the same contract suite that passes on files passes on
  DO + R2, and the cross-store crash matrix passes.

### [ ] H4 The authenticated front door

A Cloudflare Worker as the only public entry point. It authenticates, routes
and renders; it does **not** reimplement promotion or review logic in
TypeScript. A server-owned mapping derives exactly one corpus DO from the
authenticated subject — a Worker never discovers a DO from a browser-supplied
ID. V1: one hosted corpus per account, one owner.

Private sessions, recent-authentication checks for sensitive actions, CSRF,
restrictive browser headers, upload limits, content-type handling, rate limits,
and no analytics on private study content. Direct private R2 upload and
download capabilities with no public object URLs; signed links expire in
minutes.

Ships as a **read-only hosted mirror**: log in from a phone, see your library
and your sources. No jobs yet. That alone is worth having.

Account creation states plainly that hosted PDFs, draft cards, review history,
audio and deck files leave the device, and names the storage provider. Login is
required before an upload; no corpus URL is public.

- **Depends on:** H3. **Ships when:** you read your corpus from a phone.

### [ ] H5 Workflows and the Container runner

Package Python 3.11+ janki and its exact prompt files in a reproducible
Container image. A plain Python Worker is not the janki runtime — Pyodide has
an ephemeral in-memory filesystem and no functional threading or
multiprocessing. Container disk is scratch on every run.

Workflows start Containers, materialize an immutable job snapshot, stream human
progress, store exact outputs, and submit authority- and read-set-bound change
sets through operation-scoped capabilities. The Container receives only the
declared read set and a pending R2 prefix — never a browser ID, a broad bucket
credential, or the whole corpus. It returns a typed `CorpusChangeSet` plus
artifact hashes and never writes authoritative state.

**Workflow payloads carry IDs, hashes and object references — never PDFs,
audio, snapshots, prompts or model answers.** Every job records the code image
digest, schema version, prompt fingerprints, provider, model and request
lineage.

**Consent and job start are not one transaction.** The authority transaction
writes both an operation row and a DO outbox row; an idempotent dispatcher
starts exactly the Workflow named by that operation ID and records its
instance. Outbox replay cannot mint new authority or a differently keyed job.
Workflows retry steps, but a provider call or audio purchase is never
automatically repeated unless the provider's own idempotency contract proves
the original cannot duplicate. A timeout after dispatch is `outcome_unknown`; a
fresh charge needs fresh authority. No lock or DO transaction is held open
across a model call.

Cover extraction, model coverage, jpdb and kanji enrichment, audio, preview and
build as separate jobs. Local-only providers such as a desktop VOICEVOX engine
are clearly unavailable until they get their own hosted design. Provider keys
stay in encrypted server-side secrets; a browser never talks to a provider.

Pick one honest billing model up front — owner-supplied keys stored outside the
corpus, or service-owned keys with a visible allowance — and always say whose
credits a call consumes.

- **Depends on:** H4. **Ships when:** a PDF uploaded from a phone becomes a
  downloadable `.apkg`, and closing the browser mid-job loses nothing.

### [ ] H6 Optional AI Gateway transport

Enable only if a conformance spike proves it preserves janki's exact Anthropic
and OpenAI call shapes. Use provider-native endpoints, never a generic
chat-completions translation. Test document and image content blocks,
structured-output schemas, adaptive-thinking and effort fields, refusal and
stop details, streaming and timeouts, the batch endpoints enrichment uses,
OpenAI sentence TTS, and representative large page selections. A failed case
disables the gateway for that operation; it never silently changes model,
request shape or provider.

Fail-closed defaults, each proven by a deployment test:

- Payload logging off, verified by reading back a gateway log and finding no
  request or response body. **Verify the exact header name against current
  docs** — the plan draft asserted `cf-aig-collect-log-payload`; the documented
  header this project knows is `cf-aig-collect-log`. If the gateway cannot
  enforce metadata-only logging, disable gateway logs entirely and use janki's
  redacted operation metrics. Metadata is a pseudonymous operation ID plus
  provider, model, tokens, cost, status and duration — never filenames, source
  or card text, prompts, readings or meanings.
- `cf-aig-skip-cache: true` on every card-writing, coverage, tutoring and TTS
  request. Extraction's cache is janki's own fingerprinted durable artifact; a
  gateway cache must never make `--force` return an old answer.
- Exactly one attempt, gateway and per-request. Dynamic routes may cap spend
  or rate but cannot fall back to another model or provider. A transport
  timeout enters `outcome_unknown` exactly as a direct provider timeout does.
- BYOK only after choosing owner-key versus service-billed. Unified Billing is
  never an unnoticed fallback; the confirmation names whose credits are used.
- Run tokens are account-scoped, so neither a browser nor a general Container
  receives one. An authenticated Worker-side binding exchanges an
  operation-scoped capability for one pinned request. Gateway names are not
  tenant isolation.

Consent names the full path: *"Send pages 27–34 through Cloudflare AI Gateway
to Anthropic Claude Opus 5."* Disabling payload storage does not mean
Cloudflare never processes the bytes, and the screen says so. Record the
gateway config fingerprint, no-cache and no-retry policy, credential owner and
upstream request identifiers in provenance. Direct-provider mode is a
separately configured transport, never an automatic fallback after the learner
approved the gateway path.

- **Depends on:** H5. **Ships when:** conformance passes and the disabled-log,
  skipped-cache, single-attempt, no-fallback assertions all hold in deployment.

### [ ] H7 Backup, fork, move, delete and operations

An export begins with a short DO transaction capturing a quiescent global
revision and pinning exact R2 object versions in its manifest — it never holds
a transaction across a model call. Running jobs first reach a captured
boundary; `outcome_unknown` operations are included as non-resumable history
and are never restarted under different credentials after import. **A DO
database backup without its pinned R2 manifest is not a corpus backup.**

- **Download backup** — an inert checksummed bundle keeping the original
  lineage, epoch and revision. Downloading it does not create a second
  writable authority.
- **Fork as a separate library** — import into an empty target with a *new*
  lineage. The confirmation warns that preserved stable word IDs preserve Anki
  GUID behaviour, so importing decks from both forks updates the same notes.
- **Move this library** — freeze the source read-only at a named revision,
  export a checked manifest, import into an empty target, advance the authority
  epoch, acknowledge, and only then release the old source. A failed import
  before acknowledgement safely unfreezes; acknowledgement retry is idempotent.

Deletion is two-stage. The confirmation shows the soft-delete interval (30 days
default) and the final purge date. The first transaction tombstones the corpus,
advances its epoch, revokes every capability and outbox entry, and makes it
read-only except for restore, then drains or cancels jobs — a late Workflow
result cannot clear a tombstone or recreate state. Restore inside the window
creates a new epoch. At the purge date a workflow removes DO data and every
unshared R2 object after verifying manifests and pins. Until then the UI says
**scheduled for deletion**, never **permanently deleted**. Downloaded backups
and provider-side retention are outside the action and explained separately.

Plus: structured audit events, redacted logs, per-provider usage and cost
visibility, quotas, restore drills, and a support bundle excluding source and
card text and secrets by default. Schema changes use a verified pre-migration
corpus-plus-R2 backup, a quiesced forward-only migration, validation and a
tested restore — not a dual-schema branch or an assumed SQL rollback.

- **Depends on:** H5 (H6 optional). **Ships when:** you can back up, fork,
  move and delete a hosted library and the UI never lies about which state
  it is in.

### Adversarial coverage for milestone H

- **Two sources of truth.** Backup, fork, failed move, lost acknowledgement,
  completed move. Lineage, epoch, mode and writable home stay unambiguous;
  implicit sync and import into a live target refuse.
- **Ephemeral Container disk.** A restarted Container reconstructs any job
  solely from DO state and R2, producing the same result.
- **Cross-store partial commit.** Crash after R2 upload, before and after the
  DO transaction, during backup, during GC. No referenced or pinned artifact
  is lost; every true orphan is reclaimable.
- **Consent-to-job gap.** Crash after authority and outbox commit before
  Workflow start, and after start before its ID is acknowledged. Replay yields
  one operation and one keyed Workflow.
- **Retry after a paid side effect.** Restart each step at its worst boundary,
  including provider return before capture and capture before commit. One
  authority yields at most one accepted result; unknown completion never
  triggers a second charge; conflicts preserve paid bytes for restaging.
- **Tenant escape.** Substitute another account's corpus, source, object,
  operation and download IDs. Every path refuses without revealing existence.
- **Read-set precision.** An unrelated edit during extraction still lets the
  paid result land; a read-set member change becomes `needs_restage` with no
  overwrite, loss or second call; two devices on one card land one review.
- **Prompt injection.** A PDF instruction cannot call tools, grant authority,
  expose another source, or turn the assistant into a card-writing path.
- **Gateway surprise state.** Payload logs absent, cache skipped, one attempt,
  no dynamic fallback, no accidental Unified Billing, no account-scoped Run
  token reaching a Container or browser.
- **Deletion theater.** Tombstone with a live job, reject its late result,
  restore inside the window, complete a second deletion through purge. The UI
  distinguishes recoverable, scheduled and purged at every point.
- **Migration collision.** Migrate with an in-flight job, crash after the
  pinned backup, restore both SQL and exact R2 versions. No dual-schema path.
- **Wire corruption.** Vary SQL insertion order and locale; round-trip
  Japanese, missing versus empty, unknown extensions, annotations, collision
  order and approval fingerprints without normalization or loss.
- **Hostile bundle.** Refuse traversal, duplicates, decompression bombs,
  truncation, substituted hashes, unknown schemas, secrets and epoch replay
  before mutating the target.
- **Platform pressure.** Fill structured storage near the DO limit, use an
  oversized media corpus, exhaust a quota. The learner gets an export and a
  recovery path — not silent partial loss or a surprise bill.
- **Outage and long job.** A closed browser, a Worker deployment, a Container
  restart or an unavailable provider cannot invent success, forget a paid
  result, or leave the learner unable to tell what to do next.

**Milestone H non-goals:** two-way local/cloud sync; team or classroom
permissions; public source or deck links; automatic redistribution approval;
browser-held provider keys; storing the corpus on Container disk; a Python
Worker port of janki; a second Japanese-review model; making Cloudflare a
prerequisite for local use.

---

## Files this plan touches

Local: `src/japanese_anki/workbench/` (new),
`src/japanese_anki/application/` (new, factored from `cli.py`),
`src/japanese_anki/localhttp.py` (new, the loopback boundary both surfaces
share), `review_panel.py` (folded into the workbench and deleted),
`staging.py`, `patterns.py`, `promote.py`, `status.py`, `validation.py`,
`coverage.py`,
`inputs.py`, `extract.py`, `ledger.py`, `data/decks/*.yaml` (if W4.0 picks
`intake_tag`), `exporters/anki.py`, `tests/fixtures/workbench/` (new),
`tests/test_workbench.py` (new), `README.md`, `docs/DESIGN.md`,
`docs/IMPORTING.md`, `docs/QUALITY.md`.

Hosted: `src/japanese_anki/storage/` (new), a Cloudflare Worker, Workflow and
migration tree (new), a Container image for the existing Python package, store
contract and import/export/security/recovery tests, and hosted learner,
privacy and operations documentation.
