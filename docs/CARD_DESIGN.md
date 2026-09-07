# Card Design

## Note versus card

A canonical vocabulary record or character record becomes one Anki note. The
content type determines what is studied; the enabled review directions on that
note generate one or more cards.

The default vocabulary configuration creates:

1. Recognition: Japanese -> meaning and reading.
2. Production: English meaning -> Japanese.

Reading cards are available but disabled by default to avoid redundant reviews.

Kanji decks default to recognition only. Each explicitly selected character
produces one note, identified by `kanji:<character>`, and one recognition card.
Its front shows the character; recall the core meaning. **Show Answer** reveals
meanings, strokes, common JPDB reading groups, their reported percentages, and
provider-bound contextual examples together. A separate caret on that answer
holds additional readings and the KANJIDIC inventory. The caret is not Anki's
answer reveal.

A kanji reading card uses one fixed contextual example. Production needs an
explicit disambiguating cue. These directions are optional and offered only
when the prepared note supports them. The Assistant previews the exact card
count and content before one apply-and-build confirmation. The default flow
has no paid writing or bare-character audio; a character has no single implied
pronunciation.

JPDB's contextual groups remain separate from KANJIDIC on/kun readings. The
caption is **JPDB reported usage**, preserving rounded values, explicit bounds
and unknowns without normalization. Examples and furigana come from the
provider's reading-bound entries, not matching or segmentation rules in Janki.
Vocabulary stroke panels consume the same saved evidence. Builds work offline.

## Stable identity

The record ID normally has this form:

```text
word:<expression>:<reading>
```

The builder uses the ID to derive the Anki note GUID. Do not edit IDs after a
record has entered your real deck unless you intentionally want a new note.

Deck IDs and model IDs also need to remain stable. Vocabulary model IDs derive
from the configured base model ID plus a bit mask representing enabled card
types. Character decks pin their model ID and directions explicitly in the
deck definition. Choose directions before accumulating review history; a later
addition must not silently change the questions attached to existing notes.

## Furigana

Store furigana in Anki's bracket notation:

```text
話[はな]す
日本語[にほんご]
```

The front of a recognition card shows only the expression. The back uses the
furigana filter. If furigana is absent, the template displays the plain reading.

Do not guess segmentation such as where okurigana begins. A whole-word reading
is not enough to derive correct bracket placement for every word.

## Romaji

Romaji remains separate in the canonical record and Anki note, but vocabulary
cards do not render it. The Japanese reading and furigana provide the
learner-facing pronunciation cue.

## Examples

One primary example sentence is shown on the card. Keep it:

- natural;
- short enough to parse during review;
- appropriate for the learner's grammar level;
- useful for distinguishing the word's actual usage.

Additional examples may remain in source data for future templates.

## Dictionary lookup links

Cards include an iOS Shirabe deep link using:

```text
shirabelookup://search?w=<expression>
```

They also include `https://jpdb.io/search?q=<expression>&lang=english` as a web
fallback during desktop review. Both links are intentionally isolated in the
back templates so either can be replaced easily if its current URL contract
differs.

Both take the same `ShirabeQuery` field, which the builder percent-encodes at
export time (`urllib.parse.quote(expression, safe="")`). HTML escaping alone
does not protect query delimiters: `Q&A` rendered as `?w=Q&amp;A`, which the
webview decodes back to `?w=Q&A`, so the app receives `w=Q`. Encoding happens
at export rather than in template JavaScript because the cards carry no script
at all — see the rule below.

## Audio

A record may carry an `audio` path and each example an `audio` of its own;
`janki audio` writes both. The builder packages the file and writes a
`[sound:filename]` field — `Audio` for the word, `ExampleAudio` for the first
example.

Paths are resolved against `[paths] media_dir` first and against the deck file's
own directory second. The fallback exists for decks written before `media_dir`
did; new material should use the media-dir form. Note the two agree for a
`../media/...`-style path, since the directories are siblings — only a
media-dir-relative path tells them apart.

A field holding a verbatim `[sound:...]` tag is passed through **and warned
about**: janki packages no file for it, so the card is silent unless that media
is already in your collection. For a pipeline that generates its own audio, that
is a silent-mute trap rather than a feature.

## Pitch accent

`PitchAccent` holds the diagram M5.1 renders — a line over the high morae and a
drop down the right of the mora the pitch falls from, which is how a Japanese
dictionary draws it. No JavaScript: it is plain spans and CSS, so it renders the
same on AnkiMobile, AnkiDroid and the desktop.

**No card carries JavaScript**, and this is a correctness rule rather than a
stylistic one — AnkiWeb's reviewer strips template scripts, so anything assembled
in script is simply absent there, with no error to notice. A jpdb link built by
an inline `encodeURIComponent` looked right everywhere it was tested and became
a silent jump to an empty search page on AnkiWeb; the query is percent-encoded
at export time instead. `tests/test_card_templates.py` pins this.

Every accepted pattern is drawn, primary first. A word with two accents has two,
and showing one would teach that the other is wrong.

The empty span at the end is the **particle slot**, and it is what makes odaka
visible: the fall happens *after* the word, so a diagram that stops at the last
kana has nowhere to show it and 橋 would look identical to 端. (Their *audio*
does collapse — see `pitch.py` — but the diagram keeps them apart.)

A pattern that does not fit its reading is left out rather than drawn wrong.
`render_pitch_html` refuses it, and a card is the last place to start guessing
at an alignment janki declined to guess at anywhere else.

## How many meanings a card shows

jpdb hands back every sense a word has — する has 17 — and a recognition card is
not a dictionary entry. Cards show the first few, in jpdb's own
roughly-commonest-first order, and say how many they left:

```toml
[cards]
max_meanings = 4    # 0 shows them all; a deck may set its own
```

The record keeps all of them. The cap is a card decision, so `janki status`, a
search, and a human choosing which sense matters all still see the full list.

## Fields are append-only

`FIELD_NAMES` may be **appended to, never inserted into or reordered**. A note's
values are positional, so an insertion shifts every value after it on every note
that already exists. An append is one-way as well: once a release ships, the
field is in every collection that imported the deck.

Adding fields to a notetype Anki already has is safe — it upgrades in place,
matches notes by GUID and keeps review history — **but only if the import has
"Merge Notetypes" ticked, and Anki's default is off.** With it off the import
leaves every existing note on the old notetype and files the new one beside it
as an empty `…+` clone, silently. See [NOTETYPE_UPGRADE.md](NOTETYPE_UPGRADE.md)
for the evidence and the conditions.

`model_id` and the notetype name default to values derived from the enabled card
set, so changing `cards:` mints a *different* notetype and orphans scheduling.
A deck can pin `deck.model_id` / `deck.model_name` to prevent that; pinning is
the sanctioned way to change the card set with review history intact.

## Conjugation drill context

A lesson-specific conjugation deck may add a deck-wide `form_note` and a
`drill_examples` mapping keyed by exact canonical record id. These are
explicitly authored examples of the form being drilled; they are not rewritten
from the record's ordinary vocabulary examples. Such a deck must pin a
nonempty `include_ids` lesson scope. Every included record needs exactly one
complete Japanese/English example labelled `polite` and exactly one labelled
`casual`. Every included record must still exist and be conjugable into the
requested form.
`form_note` never stands alone: when present, it requires that complete
`drill_examples` mapping. Duplicate mapping keys anywhere in a pattern or
conjugation deck are refused before the YAML loader can discard either authored
value.
Every authored example value is text (`furigana`, `audio`, and
`spoken_japanese` may be empty or omitted). `janki audio --deck NAME
--examples` fills `audio` through the same reviewed sentence provider, voice
selection, journal, recovery WAL, ledger, and media packaging used by
vocabulary examples. A sparse human-authored `spoken_japanese` is the same
exact-input override available on a vocabulary example; the displayed
sentence is unchanged.
The example's `furigana` value uses balanced Anki-style `word[reading]`
notation. Any Unicode whitespace terminates a possible ruby base. One ASCII
space immediately before an annotated run is Anki's boundary marker and is
consumed; other whitespace, including a U+3000 full-width space, remains
outside the ruby. Line breaks outside `[reading]` are preserved in the rendered
card; a line break inside the reading is malformed. `form_note` and `english`
are plain text, not furigana markup. Unknown example fields, structured values,
malformed notation, and examples outside that explicit scope are refused
rather than ignored. The rich keys themselves are refused on every
non-conjugation deck.

The rendered explanation, examples, verb group, and canonical `usage_notes`
share the existing `Examples` field of the six-field `Japanese Pattern`
notetype. Each stored drill-example audio path is packaged and rendered beside
its matching sentence. The field order, model id, and
`drill:<form>:<record-id>` GUID stay
unchanged, so a richer rebuild updates installed drill notes and preserves
their scheduling. A rich card's `Source` says the form was computed by janki
while its context came from the deck and record; a plain mechanical drill keeps
the shorter `computed by janki` label. Ordinary rule cards using the same
notetype are unchanged.

This is the standard for new lesson-specific drill decks. A general mechanical
drill may still omit both keys and show only its computed transformation and
verb group. Janki never manufactures a form-specific Japanese sentence by
substituting into a vocabulary example.

## Inline notes and `source`

A deck's inline `notes:` are merged over the normalized record field by field,
and `source` is merged **shallowly**: keys the note supplies win, keys it omits
keep the normalized record's. So an inline note giving only
`source: {row: 12}` keeps the record's `type` and `imported_from` rather than
blanking them.
