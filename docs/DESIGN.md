# janki design

One page. This document leads: when code, plan, or another document disagrees
with it, this one wins and the other changes.

## What janki is

janki turns Japanese study material — Shirabe Jisho CSV exports, photos,
PDFs — into Anki decks. The repository is the durable source of truth;
generated `.apkg` files are build artifacts.

## The pipeline

Where the code has not yet caught up with a sentence here, the implementation
or workbench plan names the gap. The sentence states the design, not the
present.

**1. Intake.** Sources arrive and are preserved immutably under `data/inbox/`,
with provenance. Nothing ever edits a source.

**2. The AI passes (Claude Opus 5 by default; Codex is optional for immediate
bare-record enrichment).** One rich template per input shape asks
for the complete card: Japanese, natural English glosses, examples, furigana
giving each kanji's *contextual* reading in that sentence, romaji, usage, and
register. There are three external card-writing paths. `extract` reads a preserved
source once and returns candidates, coverage facts, complete card proposals,
and the grammar patterns that source teaches in the same answer; its auto,
table, and prose modes remain three complete source templates. A large job may
be requested as one confirmed finite batch of smaller independent `extract`
calls over explicit source parts, run with bounded parallelism; each child is
still one source read with its own journal entry, capture and staging.
`enrich --ai`
reads a bare record once and returns glosses, examples, and usage together.
`revise` reads only explicitly selected existing cards or deck content plus the
owner's requested change and returns a fingerprinted staging proposal; it never
writes canonical content directly. Janki's ChatKit agent may prepare typed
plans and, after exact owner authority where required, dispatch these same
application operations. Conversation is a repository-wide surface over the
three paths, not a fourth writing path. Extraction and revision share that
transport selection, each with its own plan/consent/journal/recovery/staging
pipeline; configuration selects only who is billed, never a different
workflow. `claude-code` consumes the locally logged-in Claude Pro or Max
subscription allowance, guarded as first-party Pro/Max; `anthropic-api` uses
Anthropic API billing and has to be written down, because a missing CLI, a
logged-out login or any failure is a refusal rather than a fallback. Each
source call is exactly consented to on the transport it names — on its own, or
as one exactly enumerated child of a confirmed batch, where the child keeps its
own source, request, provider and billing identity — and extraction's existing
staging and journal flow is unchanged by the choice.

A subscription reply is captured inside its own operation artifact as a
versioned wrapper holding the original request manifest and channels beside
the exact raw provider bytes, written before anything parses them. `janki
operations --show-reply` exports the complete captured artifact byte for
byte, retaining the manifest and the raw reply that recovery needs; pure
helpers unwrap it for internal viewing and classification. Recovery needs no
prompt file — the request is read back from what was saved — but it does need
the current response contract to match the saved one; a changed schema refuses
and preserves both the request and the reply rather than reinterpreting them.
A stream interrupted before its terminal capture still has its exact frames
and request fingerprint, but not yet that stored manifest, so it is not
replayable across a later contract change.

Development, planning and code-review launches use the tracked
`scripts/claude-subscription.py` launcher. It strips inherited provider overrides
and verifies a Claude Pro/Max subscription under the same environment and
settings context used for dispatch. An unverified login stops the model call;
there is no API fallback. This rule does not change the separately authorized
paid content providers or the owner's subscription extra-usage setting.

An ordinary Janki message is a separate, non-card-writing model turn. Its
`[assistant]` provider is independent from the revision transport, its model is
pinned to `claude-opus-5`, and its default is Claude Code using the owner's
logged-in Pro or Max subscription. Sending the message authorizes exactly that
one journaled turn over its exact bounded repository context. The turn may
answer, request typed read-only repository projections, and return typed
staged-change or action plans. Model output itself is never write authority:
only janki's local broker may validate a closed action type, resolve opaque
repository identities, call an existing application planner, and render any
authority the operation still needs.
Pattern listing and human review remain local operations over extraction
output. Prompts are template files, readable without opening Python, and they
are the quality mechanism: **when the output is wrong or thin, expand the
template.** A bare word list has no source sentences, so its template writes
them; everything else about the card contract is shared.

Extraction's structural accounting is separate from judging the Japanese. One
deterministic staging row represents each stable word ID; if the parsed answer
proposes that ID more than once, the fingerprinted staging metadata preserves
every member of the collision group in response order. The editable `records`
list remains the human keep/correct/re-identify list.

The opt-in `promote --accept-coverage` gate is another paid model operation. It
re-reads a preserved source only to check janki's coverage bookkeeping; it does
not write or judge card content.

janki never writes code that reads Japanese — no rule that decides a reading,
a word boundary, a register, or whether notation matches a sentence. Japanese
is deeply contextual; a rule derived from one sentence is wrong for the next.
**Writing a rules engine for Japanese from scratch is this project's defining
anti-pattern.** The retired jpdb sentence oracle is the cautionary measurement:
38 of 155 examples flagged, zero true positives.

**3. Enrichment.** Facts from outside, and deterministic derivation janki
owns:

- **jpdb** — what a dictionary knows about a word: furigana, romaji, part of
  speech, verb group, transitivity, conjugations, pitch accent, frequency rank. It
  is also the reading witness at the promote gate, where a reading it
  contradicts is held back rather than minted into a permanent record ID.
  **Meanings are not on that list, and a dictionary may not write them even to
  replace a model's guess** — a card's meaning is the sense its source taught,
  and a gloss list keyed on spelling cannot tell おたふく "mumps" from the
  identically-spelled お多福 "homely woman". Only a person or `enrich --ai`,
  both of which read the card, may settle a provisional meaning, and
  `status --unsettled [field]` is how you find the ones still waiting, and
  what settles each — a mark nobody
  can see cannot do the job it exists for.
  The line is drawn at *sense*: everything still granted above is a property of
  the word rather than of the meaning taught, so an entry resolved on matching
  spelling and reading states it correctly for either sense. That leaves a
  narrower hole, and it is deliberate rather than overlooked: for an all-kana
  record spelling and reading are the same string, so a true homograph pair
  passes the guard and can differ in accent or word class. Refusing the fields
  outright would cost every legitimate kana word, and telling which pair is
  which is reading Japanese. So janki keeps writing them and a person decides.
- **KANJIDIC** — kanji meanings, readings and stroke *count*. Stroke **order**
  comes from KanjiVG, a separate source under CC BY-SA 3.0 that a shared deck
  must credit.
- **JPDB kanji pages** — published reading percentages and examples explicitly
  attached to each reading by the provider. These carry JPDB's own labels,
  groups, displayed rounding, bounds, source URLs, retrieval times and response
  fingerprints. They are labelled **JPDB reported usage**; missing percentages,
  corpus scope and denominators remain unknown. These contextual groups are
  separate from KANJIDIC's on/kun inventory. Janki copies supplied word readings
  and ruby segmentation, never assigns a word to a character reading itself.
  Requested missing pages are fetched on demand; saved facts are reused, and
  refresh is explicit. Full HTML stays in a local cache outside the repository;
  extracted facts live in `data/jpdb_readings.json`. Builds are offline.
- **VOICEVOX** — audio for words, with the pitch accent **forced**. It is the
  only engine here that can force one, which is why it voices the words: left
  to guess, an engine renders 橋 and 箸 identically, and telling those apart is
  what a pitch card is for. A word with no usable pattern is still voiced, with
  the engine's own accent and a ledger mark saying so.
- **OpenAI Realtime** — audio for example *sentences*, read naturally by
  `gpt-realtime-1.5`. Cedar, ash, verse and marin form an equal-weight pool;
  a versioned framed SHA-256 of the stable record id picks one, so every
  example on a note keeps the same voice across rebuilds. The reviewed prompt
  asks for a clear learner pace around 75% of normal conversation while
  preserving connected Japanese phrasing. A sentence carries context that
  disambiguates, and no accent data janki has covers a whole sentence. If the
  model misreads one reviewed sentence, a human may add sparse
  `examples[].spoken_japanese`; janki sends that exact nonblank text instead
  of the displayed `japanese`, and never derives it automatically.
  The displayed sentence remains the card text and stable filename identity;
  the effective spoken input participates in the content/request fingerprint
  and audio write-ahead record, so changing it stales only that clip without
  renaming it. Lesson-specific conjugation drills use this same reviewed
  sentence profile and transaction for their authored polite/casual examples.
  A deck-scoped synthetic owner keeps their ledger and filenames distinct from
  the source vocabulary note, and their audio paths live in that deck's
  `drill_examples` entries.
- **Derivation** — mechanics janki owns: romaji transliterated from the
  model's furigana (for a bare word, from its reading), and the conjugation
  tables that fill the Conjugations field and build the drill decks.
  Deterministic grammar *generation* is derivation; judging the model's
  grammar is not.

Enrichment adds facts about *words* and *files*. It never judges sentences.

**4. Compile.** genanki builds the decks. Deterministic **GUIDs** — note ids
are timestamps genanki mints per build, and the GUID is what Anki matches on
make rebuilds update cards instead of duplicating them; the ledger records
what shipped. Word decks are durable thematic study destinations, not
per-source artifacts. Every assignable word deck declares one nonblank, unique
`intake_tag`; assigning a staged record to that deck writes that tag. Its
`intake_tag` appears in its `include_tags`, not its `exclude_tags`, and cannot
select another word deck. No record belongs to more than one word deck — a
convention the deck files keep, checked by a test over the real decks rather
than enforced by the builder, which filters each deck without looking at the
others.

A vocabulary deck is **shared** by default. An explicit **standalone** deck
holds its own scoped records — `standalone:<scope>:<expression>:<reading>` — in
that same canonical store, so its cards and their Anki review progress are
independent of the shared ones. The scope is derived once at creation and saved
in the deck definition; renaming the deck does not rederive it, and no reviewed
shared identity is ever migrated into a scope. A copy is a distinct record, so
one record still belongs to one word deck. The owner chooses shared or
standalone in the Assistant's single, exact deck-creation confirmation.

**Kanji is a distinct content type.** An explicit character target produces one
character note with identity `kanji:<character>` and GUID
`genanki.guid_for(record.id)`, independent of its readings. The curated
`data/kanji_notes.json` store is separate from both vocabulary and refreshable
reference caches. Compounds are contextual examples inside character notes;
they do not mint vocabulary records. Character decks have `kind: kanji` and
accept only character notes. Existing word identities and enabled card sets
are not converted by adding kanji study material.

Recognition is the default character direction: the front shows the character,
and the learner recalls its core meaning. Anki's Show Answer reveals meanings,
strokes and common contextual reading examples together. A separate disclosure
on the answer holds additional readings. Reading practice uses one fixed,
source-bound example; production requires an explicit disambiguating cue.
Only directions supported by the prepared note are offered, with the exact
card count shown before apply. The default character flow is dictionary-only
and has no bare-character audio or paid card-writing call. Character-target
extraction, character revision and authored mnemonics are separate work.

**Card review shows cards in context.** The owner's preferred format for every
card type and creation/revision review is an interactive HTML preview using
the actual proposed fields, templates and CSS, with card navigation, front/answer
flipping and independent disclosures. Janki Assistant links or embeds that
preview as part of its wizard confirmations and proposal reviews, and offers a
read-only action for existing decks. Rendering uses the real exporters and
Anki in a temporary collection, with exact proposed content overlaid there.
Opening the immutable HTML snapshot makes no paid call, changes no canonical
content and grants no approval. `janki preview` provides the same interaction
as a standalone file. `docs/CARD_DESIGN.md` records the interaction demonstrated
by the Week 2 kanji preview.

An extraction batch reviews its children together the same way: the exact
proposed normalized records are overlaid into that one temporary collection and
rendered with the real templates and CSS. Combining them is a reading
convenience only. Without a chosen destination, a temporary vocabulary deck
shows the proposals using the configured card directions; that preview makes
no permanent deck or assignment. It fabricates no multi-source staging document, marks no
Japanese reviewed, accepts no coverage, promotes no canonical content, and
settles no duplicate meaning automatically; each source keeps its own promotion
and coverage rules. Where two sources propose the same identity differently,
the review discloses the conflict rather than hiding it behind first-wins
merging.

## Mechanisms the pipeline rests on

Not stages, but load-bearing: the word database
(`data/normalized/vocabulary.json`) as the durable vocabulary store and
`data/kanji_notes.json` as the separate curated character store; stable
identities (`word:<expression>:<reading>`) so cards dedup and review history
survives rebuilds; a human approving what enters the store (staging review);
a reviewed reading checked against jpdb before it becomes an identity, where
jpdb can resolve it — silence passes; a dictionary is a witness, not a gate;
a billed call over a private source confirmed by the owner, never assumed;
provenance from every record back to its source. A paid Realtime sentence call
is journaled before dispatch; an operation-bound spool fsyncs every exact
WebSocket text frame before JSON inspection, then a terminal stream becomes
the captured envelope before PCM decoding. The operation becomes committed
only while its decoded WAV is staged and ledgered by exact request. After that,
the guarded owning-record or drill-deck write must win before janki publishes
canonical media and finalizes its ordinary audio ledger entry. Either recovery
layer can therefore resume the exact interrupted request without another bill.
Staging completion similarly holds the live review and selected archive locks
through one byte-checked archive/prune transaction, so an exact interrupted
retry cannot duplicate rows or delete a concurrently replaced review.

A paid model call follows the same write-ahead shape as that paid audio
staging: journaled durably before dispatch, its exact response persisted as a
pending artifact before parsing, so neither a crash nor a parse failure can
lose an answer already paid for. `extract`, `revise`, the separate
`promote --accept-coverage` completeness check, OpenAI Realtime sentence
audio, and Assistant-selected Anthropic enrichment do this today. The
Assistant enrichment route always stages its captured answer and never writes
canonical cards. Its Codex option is refused because that transport does not
yet expose the exact raw-reply capture seam this contract requires. The direct
CLI `enrich --ai` route still spends money without journaling, which is a gap
to close rather than a design choice; this journal contract does not describe
that CLI route yet. The journal moves
`authorized → dispatching → running → result_captured → committed`, with
`outcome_unknown`, `failed_before_send`, `canceled_before_send`, and `expired`
as terminal or holding states. A dispatched call whose outcome is unknown is
never retried automatically — a fresh charge requires fresh authority.
`janki operations --show-reply ID` is the only advertised recovery reader. A
complete captured artifact passes through byte-for-byte. For an incomplete
streaming response, it emits a deterministic JSON view preserving every exact
committed UTF-8 frame payload and boundary. It revalidates the binding without
publishing a private reply, adopting a same-name replacement or unjournalled
crash extension, settling the call, or changing the journal. A recorded
artifact name is historical metadata, not proof that reply bytes remain
accessible. Before `operations --end` settles a nonterminal stream, it adopts
the sole structurally valid frame that may have been fsynced just before its
journal-head write was interrupted. An exact provider-specific rerun may then
strengthen `outcome_unknown` to `result_captured` only by sealing terminal
frames already on disk; it never redispatches. Nonempty response frames from a
call that may have been sent require `operations --forget --force` when they
did not become committed output; an ordinary forget cannot silently erase
partial paid output.
An ordinary Assistant turn follows that shape too. Over the Claude Code
transport it fsyncs every exact stream frame before inspecting one, as Realtime
does. Its exact user message, every bounded repository projection disclosed to
the provider, the closed typed action schema, provider/model identity, and
assistant reply become a durable turn under `data/assistant/`; no dynamically
obtained repository byte may enter model context without first joining that
manifest. ChatKit's thread is a view of that repository record, never its only
copy or its authority source. A typed local read is not another paid operation.
Every projection sent to a provider belongs to one freshly manifested Assistant
request, and no typed intent can open an unjournaled nested model call.
Normal capture keeps its terminal operation-bound marker until the
`result_captured` journal write durably records the relative name, pending
directory identity, five-field file snapshot (including ctime), and response
digest, together with the exact terminal marker's direct name, five-field
snapshot, and digest. The private hard link is retired before that snapshot,
so its removal cannot immediately stale the receipt; only after the receipt is
durable may that exact marker be finalized. A missing, replaced, or unreadable
marker falls back to the receipt-bound public answer and is never freshly
adopted or retired. Thus a crash before the journal write leaves WAL
proof, and one after it leaves receipt proof. With neither proof, a lexical
operation-id filename is never freshly adopted, even when its bytes match.

**One journaled call at a time, or one confirmed batch.** The journal refuses
to authorize a new one while any entry is live or holds money nobody has
accounted for — every state
except the terminal ones, plus `outcome_unknown`. `authorized` counts, because
writing the authority and marking it dispatched are two writes and excluding
it leaves a window where two runs both pass. This is enforced under the
journal's own lock at authorization, never merely displayed by a surface: a
check read when a page renders and acted on when a button is clicked is one
two callers can both pass. Nothing lifts the block automatically, because only
a person can say a vanished process is gone. `janki operations --end` records
that — as `canceled_before_send` when nothing was sent, as `outcome_unknown`
when a request left and no answer came back — and `--forget` records that they
have dealt with what it cost. Forgetting first records the exact
directory, inode and content bindings it is authorized to retire in that same
journal entry. A crash retries only those bindings without asking for a new
force decision or adopting a same-name replacement. The lexical pending
directory is part of that authority: if it is missing, symlinked/non-directory,
or stably opens as a different directory inode after a checkout, cleanup
touches no names there, never searches for the moved inode, preserves the
replacement namespace, and closes the now-unreachable binding. Other probe or
I/O failures keep the entry. The entry disappears only after its still-bound
names retire or their old namespace is proven no longer bound. That durable
forget decision means the call's money is accounted for and no longer blocks a
new authorization, while a cleanup-only entry remains listed with its exact
ordinary `--forget` retry.

**The exception is one confirmed extraction batch.** The owner may confirm one
exact finite batch of explicit immutable source parts — prepared source files
in the first increment, since discovering and cropping arbitrary PDF rows is
separate work — and janki runs them as smaller independent calls with bounded
parallelism. No batch logic reads Japanese or corrects a child's answer. One
locked atomic journal write reserves every child operation id at once, each
with its own exact source hash, request fingerprint and model, the fixed batch
membership, and the stored concurrency limit — two by default, four at most.
Ordinary unrelated authorization still refuses while any existing operation is
live or unaccounted for.

Batches reuse the states above. Each child consumes `authorized → dispatching`
exactly once under that same journal lock, which checks its exact batch
membership, fingerprint and concurrency, and the normal advance method cannot
bypass a batch claim. The streaming spool is prepared while the child is still
`authorized`, before the claim; each reply is captured before decoding and
stages independently through the ordinary extraction lifecycle. An in-flight or
`outcome_unknown` child occupies a slot; a complete captured reply does not,
and stays recoverable on its own. An unknown call is never sent again
automatically or presumed dead, while a reserved unsent child may resume under
its original exact authority after fresh binding checks. A failed child
discards nothing a successful sibling produced, and an authentication refusal
stops queued dispatch rather than falling back to the API.

Retrying part of a batch needs one fresh owner confirmation binding the new
exact requests and any discard decision for the failed operations it clears. A
successful child cannot be retried; an in-flight call needs the owner's end
decision first. An `outcome_unknown` call is already ended: its retry
confirmation acknowledges the uncertain prior cost and explicitly discards its
bound evidence. Fresh operation ids are mandatory, and an unchanged
exact request fingerprint is valid for a freshly authorized retry. The
cleanup-then-reserve sequence is resumable from durable recovery state rather
than one atomic transaction, and evidence is never erased silently.
After exact confirmation, a separate `<batch_id>.execution.json` receipt binds
the complete manifest hash, fresh child identities and exact discard decisions.
Resume may finish that confirmed execution after fresh binding checks. A request
manifest alone grants no spending or discard authority; every child still needs
the journal's atomic reservation and one-use dispatch claim.

The batch's committed manifest and execution receipt live under
`data/extraction_batches/` by default, derived beside the configured operations file, not in the
`data/.pending` recovery buffer. It binds each child's expectations and
operation id, its parts' source ancestry and labels, the concurrency limit and
every retry's provenance, and flattens nothing: each page or slice keeps its own
staging document with its own request, coverage, candidate-accounting and
pattern metadata; a reference to the full parent source is documentary, never a
substitute for the sliced input actually sent.

## Surfaces

A local browser workbench and the CLI are the same tool twice: views and
controllers over the same repository files and the same operations. The
workbench is not a second database of what happened — the repository already
is — and it may not weaken an authority gate the CLI enforces: a consent,
review, or approval demanded at the prompt is demanded identically in the tab.

ChatKit uses its self-hosted/custom-backend integration as the conversational
surface inside that workbench, not as the inference provider or a
provider-owned workflow. OpenAI hosts the UI; janki's backend supplies the
conversation through `[assistant].provider`, which defaults to `claude-code`
and the owner's logged-in Claude Pro or Max subscription. `anthropic-api` is
the separately selectable API-billed alternative. These settings do not
inherit from or silently change `[ai].revise_provider`. ChatKit is only the UI;
it holds no repository authority and supplies no inference or workflow state.

**The target design.** The Janki Assistant conversation is the primary
end-to-end study workflow. Inline choices, plans, exact card previews and
counts, content review, consent, progress and status, recovery, and deck
download all belong in the thread: completing a study task requires no
navigation to a CLI command, a separate form, or a wizard page. The CLI
remains a fully supported secondary surface over the same application
operations and the same authority, not a reduced one. This paragraph states
the target rather than present capability; the explicit gaps named below and
elsewhere in this document say where the implementation has not reached it.

Janki is repository-wide for Japanese-library operations. It can inspect and
plan over configured decks, canonical records, staging proposals, pattern and
kanji stores, immutable-source catalogues, operation and ledger projections,
media currency, and package state. An active deck is an optional conversational
focus that narrows context and presentation; it is not a capability grant, and
there is no **Chat only** / **Chat + changes** class. A supported operation may
target any freshly resolved repository object whether or not a deck is active.
An unsafe or malformed individual object refuses when read or planned; it does
not make unrelated library objects unavailable.

Repository-wide does not mean unrestricted machine access. Janki exposes
closed, bounded readers and planners whose browser and model-facing arguments
are opaque resource identities, never host paths. The server resolves those
identities through an allowlisted catalogue, uses containment and no-follow
reads, caps files and bytes, snapshots every disclosed result, and revalidates
it before use. Secrets, Git internals, credentials, and private unfinished paid
reply bytes are not general Assistant context. The Claude Code process receives
neither raw filesystem nor shell, Git, arbitrary subprocess, arbitrary network,
or mutation tools. A model may emit only closed typed read, staged-change, and
action intents. They are untrusted data until janki validates them and invokes
the same application planner the CLI and ordinary workbench use. Unknown or
unsupported intents refuse; the model never improvises a parallel writer.

Sending an ordinary Janki message is the owner's one action authorizing one
journaled Assistant turn over that message and its exact bounded context. The
routine page need not repeat provider, model, or context-provenance prose; the
request manifest and operation journal retain those bindings durably. Normal
deterministic local reads, navigation, status, diffs, and pure plans need no
additional confirmation. A plain instruction may therefore prepare a typed
staged change or application action without a special deck-capability button,
but neither the instruction nor the model's plan executes it.

For explicit kanji targets, the Assistant prepares exact dictionary-based
previews and counts before the one apply confirmation. Bounded preparation
may look up those requested characters and populate the private raw-response
cache under the owner's on-demand lookup preference; it is not a general
network capability. The plan names queried characters and services. Canonical
reference facts, character notes and a new compatible deck are proposed as
exact bytes and written only by the confirmed batch. A build uses those saved
facts, and an interrupted batch resumes from its durable receipt without
inventing another owner decision. The completed package is offered in the
same Assistant thread.

Before a protected effect, Janki renders one exact batch and one confirmation,
not a confirmation ladder. Protected effects are an additional paid call,
disclosure of private source or card bytes not already bound to the ordinary
turn, a destructive action, staging model-authored Japanese the owner has not
seen, or a canonical repository transition. The batch names every target,
every fresh input fingerprint, provider/model/billing identity and exact
request identity knowable before dispatch, explicit replacement or owner-only
decision, recovery consequence, and all writes or removals. One confirmation
may authorize several paid child calls when the batch enumerates them exactly:
each child keeps its own source, request fingerprint, provider/model and
billing identity in that enumeration, and the confirmation buys precisely
those. It never becomes one paid parent call, and the children's answers never
merge into one synthetic provenance. When the result
already exists, the batch also binds its exact output fingerprint and
provenance. A paid call's not-yet-created output is never pretended to exist:
its capture records output provenance after dispatch, and a later
visible-content batch binds those exact bytes. The browser-only one-use
capability is bound to the thread, rendered action, optional focus, and complete
plan. Execution consumes it, reloads configuration, re-plans under the
application service's locks (or revalidates the saved exact dictionary plan),
compares every binding, and still relies on
`OperationJournal.authorize` under its own lock as the final paid-call gate.
The model cannot see, mint, widen, reuse, or consume that capability, answer its
own confirmation, infer an owner review or identity decision, or manufacture a
CLI approval flag.

All protected consequences knowable at one point are consolidated into that
one batch. A paid writing operation whose answer does not exist yet may be
authorized to disclose its exact inputs and stage the unseen answer, but it
must stop there. Once the Japanese is visible and reviewed, its canonical
apply is necessarily a later batch because those exact output bytes could not
have been approved earlier. That later **Apply and finish** batch may include
promotion, selected audio, media/ledger updates, and package build under one
durable receipt; those phases do not each ask again. The receipt is the only
authority an interruption may resume, and a resume never repeats already
captured paid work.

That consolidated later batch is implemented for reviewed rich-conjugation
whole-deck revisions, selected canonical-card revisions targeting vocabulary
decks, and Assistant-selected AI enrichment whose persisted focus resolves to
one vocabulary deck. Each binds the exact visible selection before recording a
resumable review/promotion/audio/build receipt. Source-extraction proposals
still expose their applicable review, assignment, coverage, promotion, audio,
and build operations as separate typed protected actions. Extending the same
receipt-backed **Apply and finish** contract to that flow remains W7 work; the
paragraph above states the design, not every flow's present UI.

An attachment added through Janki is local immutable intake only. Its bytes are
saved through the same intake service under `data/inbox/`; uploading does not
place the source in model context or authorize a paid call. A later protected
batch names the exact stored source, disclosure and extraction request; one
owner-confirmation click dispatches and stages it, with no additional consent
page or phase-by-phase confirmation. OpenAI Realtime audio remains API-backed
and neither Claude transport setting nor a model intent can reroute it.

Extraction batches run through one batch service that the Assistant and the CLI
both call. The Assistant is the primary surface for them: it renders the exact
batch confirmation, per-child progress and status, safe resume, selected retry,
and the combined review of the actual proposed cards as HTML. While a live
batch blocks ordinary paid Assistant chat, those read-only status and review
controls stay available — the owner can watch a batch and read what it already
produced without spending anything.

Code review and Japanese-content approval are separate gates. A machinery
review neither judges nor approves Japanese, and content approval neither
starts nor substitutes for a codebase review.

## What janki's own logic is for

Enrichment, derivation, and artifact structure: identifiers, fingerprints,
field order, packaging, dedup, provenance. janki's logic enriches the card;
it does not audit the model. If a need looks like "check whether the model's
Japanese is right," it is a template problem — ask the template for more.
