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

Romaji is hidden inside a disclosure element. It should help with difficult
readings without becoming the first cue the learner sees.

## Examples

One primary example sentence is shown on the card. Keep it:

- natural;
- short enough to parse during review;
- appropriate for the learner's grammar level;
- useful for distinguishing the word's actual usage.

Additional examples may remain in source data for future templates.

## Shirabe link

Cards include an iOS deep link using:

```text
shirabelookup://search?w=<expression>
```

This is intentionally isolated in the template so it can be replaced easily if
Shirabe's current URL scheme differs.

## Audio

A record may contain an `audio` path. The builder packages an existing local file
and writes the appropriate `[sound:filename]` field. Audio is optional and is not
created by the starter project.
