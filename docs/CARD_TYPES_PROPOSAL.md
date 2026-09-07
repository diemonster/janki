# Card types and the Assistant journey

**Status: implemented.** This records the owner's
decisions for character cards and their conversational setup.
[DESIGN.md](DESIGN.md) leads the implementation contract.

## What is being studied

The original five kanji targets became vocabulary words because janki only had
word notes and character reference panels. That expanded the requested study
material: 理 became a card for 料理. A dedicated character note makes the study
target explicit.

| Content type | One note represents | Identity |
| --- | --- | --- |
| Vocabulary | A word or lexicalized phrase | `word:<expression>:<reading>` |
| Grammar / pattern | A pattern taught by a source | Pattern record |
| Conjugation | A form practiced over a selected scope | Drill record |
| Kanji | One character | `kanji:<character>` |

Study type is separate from review direction. The Assistant offers directions
supported by the selected type. This change adds character study; it does not
redesign grammar or conjugation workflows.

## Character cards

**Five explicit characters produce five notes and, by default, five recognition
cards.** Compounds appear as contextual examples within a character note.
They do not become vocabulary notes. Readings are content, so acquiring another
reading does not change the character's identity or Anki GUID.

Recognition shows the character on the front and asks the learner to recall its
core meaning. Anki's **Show Answer** reveals meanings, strokes, and common
reading examples together. The separate **caret disclosure** on that answer
holds additional readings; it is not the answer reveal itself.

For 理, the answer uses KANJIDIC meanings and KanjiVG strokes, alongside JPDB's
published `り 84%` and `わ 14%` rows. JPDB explicitly attaches 理由・無理 to
り and 理由 to わ, with the whole-word pronunciations and ruby supplied by the
source. Janki does not infer those relationships. These are source examples,
not a claim that the first listed words are the best learning order.

Optional directions remain off by default:

- **Reading:** one fixed source-bound contextual example, with its pronunciation
  revealed on the answer. It does not create one card per dictionary reading.
- **Production:** an explicit owner-supplied disambiguating cue prompts the
  character. The learner grades their own writing.

The plan shows the exact directions and count. A character deck carries an
explicit note-type identity and direction set; adding kanji notes does not
convert existing word notes or change their review history. Default character
cards have no bare-character audio and require no paid card-writing call.

## Reading evidence and storage

JPDB's percentages are labelled **JPDB reported usage**. Janki preserves the
source's reading groups, order, displayed rounding and bounds. Missing values
stay unknown, including the unpublished corpus scope and denominator. It does
not turn missing percentages into zero or normalize displayed values to 100%.
KANJIDIC's on/kun inventory is separate from these contextual groups.

Requested missing pages are fetched on demand and reused locally. Refresh is
explicit. Full HTML stays in a private cache outside the repository; extracted
facts live in `data/jpdb_readings.json`. The old JMdict word-priority proxy is
removed. The source comparison and rejected Tamaoka experiment remain in
[READING_FREQUENCY_SOURCES.md](READING_FREQUENCY_SOURCES.md).

Curated character notes live in `data/kanji_notes.json`, separate from vocabulary
and refreshable references. Each note stores the exact facts selected when it
was prepared. Refreshing references alone does not silently rewrite curated
cards. Builds use saved facts and need no dictionary network access.

## The journey stays in Janki Assistant

The chatbot is the primary workflow. “Wizard” means a guided conversation, not
a second interface. Explicit kanji intent skips a redundant type question.
Janki asks only for missing decisions and states useful defaults.

1. **Targets and destination:** the owner names the characters and a compatible
   character deck, or a name for a new one. Recognition is the default.
2. **Preparation and preview:** janki acquires requested missing dictionary
   facts into its private cache, then shows the actual proposed note content,
   directions, count and destination. No canonical data changes during this
   preparation.
3. **One confirmation:** one exact plan covers saving reference facts and
   character notes, creating a compatible deck when needed, and building it.
4. **Finish in the thread:** progress, interruption recovery and the completed
   package download remain in the same conversation. A durable receipt resumes
   the confirmed work without repeating dictionary acquisition or inventing a
   new owner decision.

For example, “Make a kanji deck called 201 Week 2 Kanji for
物、特、鳥、料、理” leads to a five-note, five-card preview and one confirmation.
An existing vocabulary deck named “Kanji” is still a vocabulary deck and is
not offered as a character-note destination.

The model supplies a typed action to the application broker. It cannot execute
or confirm its own proposal. Character intent is expressed by the prompt and
typed action; no Japanese classifier decides what the owner meant. The CLI is a
secondary interface over the same preparation, apply and build services.

## Scope of this implementation

This flow uses **explicit character targets and dictionary facts**. Source-page
extraction of character targets, character-note revision, authored mnemonics,
and contextual audio are separate future work. The Assistant does not offer an
unfinished version of those paths. Ordinary Assistant messages still use the
configured conversational provider; dictionary-only card preparation is not a
paid card-writing call.
