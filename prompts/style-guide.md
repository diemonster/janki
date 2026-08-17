# Japanese Content Style Guide

## Audience

Default material targets a beginner or lower-intermediate learner following a
Genki-style sequence.

## Required fields

Every vocabulary note should have:

- `expression`;
- `reading` when the expression contains kanji;
- at least one concise English meaning;
- a stable `id`.

## Preferred enrichments

- Anki-formatted furigana: a space before every ruby group, punctuation
  included — `今[いま]、 東京[とうきょう]`, never `今[いま]、東京[とうきょう]`,
  because Anki takes everything since the last space as the base and would
  draw the reading over the comma too.
- Hepburn romaji.
- Part of speech.
- Verb group for verbs.
- Transitivity when meaningful.
- One natural example sentence and translation.
- Common conjugations for verbs.
- Usage or nuance notes only when they teach something useful.

## English meanings

Use compact dictionary-style meanings on cards:

```yaml
meanings:
  - "to speak"
  - "to talk"
```

Avoid long paragraphs in the meaning field. Put nuance and restrictions under
`usage_notes`.

## Example sentences

Prefer familiar, concrete content. Useful recurring contexts include software
development, electronic music, family, hiking, travel in Japan, restaurants, and
everyday conversation.

A good example:

```yaml
japanese: "毎日、妻と日本語で話します。"
english: "I speak Japanese with my wife every day."
```

Avoid introducing several unknown grammar points merely to demonstrate one word.

## Verb conjugations

Useful starter forms are:

- dictionary/plain affirmative;
- plain negative;
- plain past;
- plain past negative;
- `て`-form;
- potential;
- passive when common or pedagogically relevant.

Do not manufacture forms for words whose analysis is uncertain.

## Furigana and readings

Use the reading that applies in the exact word, not an exhaustive list of every
possible on'yomi and kun'yomi. Kanji-reference decks can store broader reading
information separately.

## Tags

Use lowercase, stable tags where possible:

```text
shirabe verb godan genki::week11 topic::conversation
```

The builder converts whitespace in individual tags to underscores.
