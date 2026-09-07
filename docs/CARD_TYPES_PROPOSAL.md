# Card types — a product design for discussion

**Status: proposal.** Nothing here is implemented. Where it disagrees with
`docs/DESIGN.md`, DESIGN.md is right today and this is a request to change it
later. Personal use is the settled scope; that is not re-opened below.

## The problem

Five kanji were handed to janki as a study target and came back as vocabulary
words. Janki has no dedicated character note, so the previous agent answered
"study these characters" by selecting vocabulary words that contain them. Kanji
facts today are **reference material**: `janki kanji` fetches
KANJIDIC2, JMdict examples and KanjiVG strokes into `data/kanji.json`, and
`render_kanji_html` folds a capped block into the `KanjiInfo` field of a *word*
card. A character has no id, note, deck, or direction of its own.

The fix is a product distinction, not a heuristic: **what is being studied** is a
choice the owner makes and janki carries, kept separate from **how it is
reviewed**.

## Content types

| Type | One note is | Identity | Where it stands today |
| --- | --- | --- | --- |
| vocabulary | a word or lexicalized phrase | `word:<expression>:<reading>` | written by `extract`, `enrich`, `revise` |
| grammar / pattern | a pattern a source teaches | pattern record | `extract` emits patterns; one Rule template |
| conjugation | one form drilled over a lesson scope | drill record | rich-conjugation `revise` edits `form_note` / `drill_examples` |
| kanji | a character | **proposed** `kanji:<character>` | a reference block on a word card — no note of its own |

Vocabulary identity is *word* identity: expression plus reading. A character is
not a word, so it needs an identity of its own rather than a seat inside that
one. Readings are content on that note, never part of its identity — 理 with
several readings is still exactly one note.

## Review directions

**Proposed:** each content type takes a per-deck direction set, chosen when the
deck is created and carried on the deck. The wizard offers **only the directions
that are meaningful and supported for the selected type** — not every direction
for every type. Kanji supports the recognition / production / reading set
described below; nothing here proposes adding reading directions to grammar or
conjugation cards. This is not today's universal behaviour: direction masking and
notetype-id participation currently apply to *vocabulary*, while patterns render
a single Rule template and pattern and conjugation decks use explicit model ids.
Making direction handling typed rather than vocabulary-only is part of the
proposal, and each direction means something different per type — "production" on
a word is not "production" on a character.

## Kanji as a first-class type

**Default rule.** *N* kanji named as a study target yield *N* character notes.
Five targets, five notes. Compounds appear as **contextual examples inside those
notes** — they do not become vocabulary notes, do not enter
`data/normalized/vocabulary.json`, and mint no word ids. Adding 理 does not
create, rename, retire, or re-identify any existing `word:` record containing 理,
and migrates no review history. Studying something adds notes; converting
anything is a separate, explicit operation.

### Sample card: 理

```
┌────────── front (recognition) ──────────┐  ┌──────────────── back ─────────────────┐
│                                         │  │ 理   logic, arrangement, reason        │
│                                         │  │ 11 strokes   [stroke-order strip]     │
│                  理                     │  │ り — 料理 りょうり                     │
│                                         │  │ ▸ other dictionary readings (KANJIDIC2)│
│                                         │  │ ▸ source & attribution                 │
└─────────────────────────────────────────┘  └───────────────────────────────────────┘
```

Everything on that back is a supplied fact: the core meanings and the 料理
context are drawn from the existing collection and screenshots — not necessarily
from the current kanji panel, which shows 管理 and 理事 — and strokes and the
collapsed reading inventory come from the lookup. **No Japanese is authored here or by the
card.** A percentage line appears *only* if a frequency source is adopted, and
then labelled by source — e.g. `JPDB reported usage · り 84%`.

Recognition leads with recall of the **core meaning**; the full reading
inventory lives in the disclosure rather than being something to memorise.

**Agreed default (owner-confirmed).** On a kanji-specific card, the thing to
recall before tapping Anki's **Show Answer** is the core meaning. Show Answer
then reveals the back in full: meanings, strokes, and the common contextual
reading examples are all immediately visible — none of them sit behind a further
tap. The **caret disclosure** (`▸`) is a second, separate reveal *within* that
back, and it holds only the additional dictionary readings. Two distinct
mechanisms: Anki's answer reveal shows the card's substance; the caret keeps the
long reading inventory out of the way until asked for. This settles the emphasis
question only — the optional production and contextual-reading directions below
remain optional and off by default.

The current caps (`EXAMPLES_PER_READING = 2`, `MAX_EXAMPLE_ROWS = 4`) were tuned
for a collapsed block on a word card and should be re-decided for a note whose
whole subject is the character.

### Directions on a character note

- **recognition** — character → core meaning. **Default, and the only one on by
  default.**
- **production** *(optional)* — meaning → write the character, self-graded;
  janki has no handwriting input and must not pretend to check one. A bare
  English gloss is often ambiguous, so the prompt carries a **reviewed
  disambiguating cue** — typically the selected context word — chosen when the
  note is written.
- **reading** *(optional)* — one **fixed, selected** contextual example, read in
  context. Not one card per dictionary reading; the enabled examples are chosen,
  not generated per entry. The example comes from a contextual relationship a
  source supplies explicitly — a dictionary entry's own example, or model-written
  content staged for review — which janki consumes as given rather than inferring
  which reading applies where.

Janki **previews the exact card count** in the conversation before anything is
written. Once a deck has review history, the enabled direction set is frozen:
adding a direction later is a separate, explicit change.

### Audio

Audio is owned by the **character note**, not by a vocabulary note that has to be
created first. Default: **no bare-character clip** — 理 has no single
pronunciation, so one arbitrary reading would be presented as *the* reading. When
a contextual example is present, janki reuses a matching clip it already owns by
identity; if none exists, it proposes the extra clips explicitly in the plan.

### Frequency

Frequency here means **reading evidence, source-labelled and verbatim** — not a
whole-character KANJIDIC rank, not a JMdict `nfXX` word-priority band, not a JPDB
word rank.
[READING_FREQUENCY_SOURCES.md](READING_FREQUENCY_SOURCES.md) carries the
comparison. The owner selected **JPDB** published reading percentages for a
personal, on-demand acquisition proof of concept; runtime card integration
remains proposed. Tamaoka was rejected after its export discrepancies could
not be resolved. Store source, retrieval date, metric, scope and
denominator; sort only comparable numeric values *within* one source; keep a
source's own reading groups as its own labels rather than forcing everything into
on/kun; absent evidence renders as absent.

### Where character notes live — recommended, not an owner question

A **dedicated curated character-note store**, separate from both
`data/normalized/vocabulary.json` (a file whose every record is a word stays
that) and the refreshable `data/kanji.json` reference cache, which stays a cache
that can be re-fetched without touching curated notes. The stable record identity
is `kanji:<character>` — derived from the character alone, independent of
readings and of any later re-fetch — and the exported GUID is
`genanki.guid_for(record.id)` over that identity, not the literal string. Schema,
storage and tagging are ours to
settle; the owner does not choose them.

## The wizard

**Owner-confirmed requirement: the whole journey is completed inside the Janki
Assistant.** The chatbot is the primary workflow, not a doorway into one — it
reads like an ordinary conversation with this agent. The choices, card previews
and counts, content staged for review, approvals, progress and status, recovery
after an interruption, and the finished deck's download all appear inline in that
thread. No necessary step sends the owner to a separate CLI command, a form, a
wizard page, or another workbench surface to continue. "Wizard" below names a
**guided conversational progression** — janki tracks what it still needs and asks
for it in order — not a sequence of rigid pages. Janki asks normal conversational
questions, and only for decisions it is genuinely missing: which characters,
which deck, which directions. Anything it can determine or default it *states* so
it can be corrected in one sentence. The CLI remains a **secondary** interface
over the same application operations and the same authority — a second way in,
never a required detour.

Choosing a study type is **ordinary planning**, not an extra permission — janki
states its defaults so they can be corrected in one sentence, and does not add a
consent ladder. The default kanji journey makes **no paid card-writing call**;
every field is a dictionary lookup. Those lookups do disclose the queried
characters to the dictionary services, which the plan says plainly.

Flow, all of it in the one thread: explicit kanji intent **skips the type
question** → source/targets, a **compatible** destination deck, directions →
preview of card types and counts → one consolidated plan → dictionary work
applies directly, while any model-written content stages for visible review →
apply and finish, ending with the built deck offered for download in the
conversation.

> **Owner:** I want to study these five kanji: 理 …
>
> **Janki:** Five kanji, one note per character — compounds shown as examples
> inside them, no new vocabulary notes, existing words untouched. Which deck? I
> can make a new one; your current "Kanji" deck holds word notes, so character
> notes can't go there. Directions default to recognition only.
>
> **Owner:** New deck, "Genki II Kanji". Add production too.
>
> **Janki:** *[preview: the 理 card, both directions, from fetched data]* — 5
> character notes, 10 cards. Create deck `Genki II Kanji`; look up 2 characters
> not yet cached. Dictionary lookups only — no paid card-writing call; the
> characters are sent to the dictionary services. Then build.
>
> **Owner:** *[confirms once]* → apply and finish: apply, audio, build, receipt,
> with progress and then the finished package offered for download in the same
> thread. A thread closed mid-build resumes from that receipt on reopening; the
> owner is never sent to a command line to find out what happened.

A deck named "Kanji" today may well be a vocabulary deck. The wizard offers
**only compatible destinations**, and never implies an existing word deck can
receive or be migrated into character notes.

**Vocabulary intent.** "Study these five kanji" does not reach `extract`,
`enrich --ai`, or any path that mints word records; "…and give me two common
words for each" is a different request that plans both. The model prompt asks for
an **explicit type**, and the broker enforces a **closed set of typed plan and
artifact kinds** — structural contract validation only. No classifier inspects
Japanese or user prose to infer intent.

## Implementation sequence, if accepted

1. **`docs/DESIGN.md`, `docs/CARD_DESIGN.md`** — state the content-type /
   review-direction distinction, and that kanji is a content type. Leads.
2. **Character store + `kanji.py`** — `kanji:<character>` identity, curated store
   beside the `data/kanji.json` cache, source-labelled reading evidence, unmatched
   readings kept. Tests: `test_kanji.py`, `test_cli_kanji.py`.
3. **Notetype and templates** — `exporters/kanji_cards.py` beside
   `pattern_cards.py`; `templates/japanese-study/kanji-*-{front,back}.html`;
   `KNOWN_DECK_KINDS` / `deck_kind` / `deck_notetype` in `exporters/anki.py`.
4. **Typed deck creation** — `application/deck_creation.py` stops hardcoding
   `kind: vocabulary`; `application/assistant_deck_creation.py` and `ai_schema.py`
   gain an explicit study type on `create_deck`, with compatible-destination
   filtering.
5. **Character notes from targets** — extend `application/kanji_addition.py` from
   "characters in these records" to "these exact characters", reusing its
   fingerprint, lock and additive-merge transaction.
6. **The conversational journey** — `prompts/assistant-agent.md`,
   `application/assistant_agent.py`, `workbench/assistant_adapter.py`: the
   questions, preview, consolidated plan, approval, progress, resume and deck
   download all rendered inline in the Assistant thread, and **extending the
   existing apply-and-finish batch** (today it covers several
   revision/enrichment workflows) to cover kanji. This step is where the
   requirement above is *met*, not a description of what chat does today — the
   kanji apply-and-finish batch does not exist yet. Equivalent CLI commands land
   over the same application operations, secondary to the conversation.

**Future, not this milestone:** extracting kanji targets from a *source page*
needs a new complete task template and typed schema inside the existing `extract`
writing path — today's templates and schema demand vocabulary plus two sentences.
`revise` scope for character notes is likewise later. Putting the journey in chat
closes neither gap: until they land, source-extracted kanji targets and
character-note revision are simply **not offered** in the conversation, rather
than started there and finished somewhere else.

**Deleted in the same change** (pre-release, no legacy paths): the "kanji-review
deck backed by canonical vocabulary records" idiom in `prompts/assistant-agent.md`
and its pinned assertion at `tests/test_prompts.py:231`; the `kind: vocabulary`
assumption in `deck_creation._render_deck`. Existing word records, ids, decks and
review history are untouched.

## Open questions

1. **Content** — dictionary reference only, or an optional authored mnemonic per
   character, with the authorship and review cost that carries?
2. **Order** — kanji first, setting the shape for the rest, or does the same
   typed wizard land for grammar/pattern and conjugation in the same milestone?
