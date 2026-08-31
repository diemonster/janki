# Card Design

## Note versus card

A canonical vocabulary record becomes one Anki note. The enabled templates on
that note generate one or more cards.

The default configuration creates:

1. Recognition: Japanese -> meaning and reading.
2. Production: English meaning -> Japanese.

Reading cards are available but disabled by default to avoid redundant reviews.

## Stable identity

The record ID normally has this form:

```text
word:<expression>:<reading>
```

The builder uses the ID to derive the Anki note GUID. Do not edit IDs after a
record has entered your real deck unless you intentionally want a new note.

Deck IDs and model IDs also need to remain stable. The model ID is derived from
the configured base model ID plus a bit mask representing enabled card types.
Changing enabled card types therefore changes the note model. Make that decision
before accumulating significant review history.

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

## Inline notes and `source`

A deck's inline `notes:` are merged over the normalized record field by field,
and `source` is merged **shallowly**: keys the note supplies win, keys it omits
keep the normalized record's. So an inline note giving only
`source: {row: 12}` keeps the record's `type` and `imported_from` rather than
blanking them.
