"""Reading a deck that is already in Anki, so janki can enrich it.

A shared deck arrives as whatever fields its author chose: a `Basic` notetype
with the word on the front and the reading and gloss crammed into one HTML blob
on the back. That is a perfectly good deck to *study* and a dead end to
*improve* — nothing there can gain a pitch accent, an example sentence or a clip
because nothing outside Anki knows what the note says.

This turns those notes into records. What it does **not** do is write to the
collection: the enriched cards leave as an `.apkg` like everything else janki
builds, which is why `collection.py`'s read-only rule survives intact.

**Everything lands in staging.** The mapping below is inference — which line of
an HTML blob is the reading, which is the meaning — and this project does not
ship inference unread. A row it cannot read confidently is held with the reason
attached rather than guessed at, per the same rule the CSV and jpdb importers
follow for a missing reading.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path

from japanese_anki.collection import DeckNote
from japanese_anki.identifiers import (
    contains_kanji,
    is_kana,
    stable_record_id,
)
from japanese_anki.models import SourceReference, VocabularyRecord
from japanese_anki.romaji import kana_to_romaji

__all__ = ["ImportedDeck", "HeldNote", "records_from_notes"]

#: Anki writes block structure as tags; a person reads it as lines. Both of
#: these end a line, and everything else is decoration.
_BREAKS = re.compile(r"(?i)</\s*(div|p|li|tr|h[1-6])\s*>|<\s*br\s*/?>")
_TAGS = re.compile(r"<[^>]+>")
#: Kana, prolonged sound mark, the nakaguro a compound reading uses, and spaces.
#: Anything else in a candidate reading means it is not one.
#: Punctuation and grouping marks are not part of a *pronunciation*: `あれ？` is
#: read あれ, and `ちょっと（まて）` is ちょっとまて. Stripped before the kana test
#: and before the reading is taken, while the expression keeps what the deck
#: wrote — the reading is half the record id, and `word:あれ？:あれ？` would carry
#: a question mark into every GUID derived from it.
_READING_NOISE = re.compile(r"[\s・、,／/（）()！!？?。…‥]+")
#: `すごい　「すげえ」` — the reading, then the colloquial form the book actually
#: prints. A reading pack for a manga is *about* that second form, so it is
#: parsed out and kept rather than being the reason the row is dropped.
_VARIANT = re.compile(r"[「『]([^」』]+)[」』]")


def _lines(value: str) -> list[str]:
    """The visible lines of an HTML field, in order, blanks dropped."""
    broken = _BREAKS.sub("\n", value or "")
    text = html.unescape(_TAGS.sub("", broken))
    # NBSP is what a hand-edited Anki field is full of, and it survives strip().
    lines = [line.replace(" ", " ").strip() for line in text.splitlines()]
    return [line for line in lines if line]


def _looks_like_reading(line: str) -> bool:
    """Is this line a kana reading rather than a gloss?

    Kana only, once separators are removed. An English gloss fails on its first
    letter, and a Japanese *meaning* — a deck that glosses in Japanese — is
    indistinguishable from a reading by shape alone, which is why a note with
    two kana lines is held rather than guessed at.
    """
    stripped = _READING_NOISE.sub("", line)
    return bool(stripped) and is_kana(stripped)


def _split_variant(line: str) -> tuple[str, str, bool]:
    """``(text, variant, was the line nothing but brackets)``.

    `ほら 「ほーら」` is a reading and its spoken form. `「げつもく」` alone is
    ambiguous on its own and decided by position: on the front, or as the only
    kana on the back, it is the word itself; after a reading has been found it
    is that reading's spoken form. The flag is what lets the caller tell those
    apart instead of this function guessing without the context to do it.
    """
    found = _VARIANT.findall(line)
    if not found:
        return line.strip(), "", False
    outside = _VARIANT.sub("", line).strip()
    if not outside:
        return found[0].strip(), "", True
    return outside, found[0].strip(), False


@dataclass(frozen=True, slots=True)
class HeldNote:
    """A note that could not be read into a record, and why."""

    note_id: int
    front: str
    back: str
    reason: str


@dataclass(frozen=True, slots=True)
class ImportedDeck:
    records: tuple[VocabularyRecord, ...]
    held: tuple[HeldNote, ...]

    @property
    def held_note(self) -> str:
        """The `review_notes` block a reviewer reads in the staging file."""
        lines = [
            f"{len(self.held)} note(s) could not be read into a record. They are "
            "listed here rather than dropped; add them by hand if they are worth "
            "keeping.",
            "",
        ]
        lines += [
            f"- [{item.note_id}] {item.front or '(no front)'} — {item.reason}"
            for item in self.held
        ]
        return "\n".join(lines)


def _record(
    note: DeckNote,
    deck: str,
    expression: str,
    reading: str,
    meanings: list[str],
    variants: list[str],
) -> VocabularyRecord:
    # The bracketed form the deck prints beside the reading. Written as a usage
    # note because that is what it is — how the word is said in this book — and
    # because `enrich --ai` fills that field only when it is empty, so a note
    # taken from the source survives the model that would otherwise write one.
    spoken = ""
    if variants:
        spoken = f"Colloquially 「{'」, 「'.join(dict.fromkeys(variants))}」 in this text."
    return VocabularyRecord(
        usage_notes=spoken,
        id=stable_record_id(expression, reading),
        expression=expression,
        reading=reading,
        romaji=kana_to_romaji(reading) if reading and not contains_kanji(reading) else "",
        meanings=meanings,
        tags=["anki"],
        source=SourceReference(
            type="anki",
            imported_from=deck,
            raw_fields={
                # The note as it was, so nothing this parser did not understand
                # is lost — and so a person can find the original card again.
                "anki_note_id": str(note.id),
                "anki_guid": note.guid,
                "anki_notetype": note.notetype,
                "anki_tags": note.tags,
                **{f"anki_{name}": value for name, value in note.fields.items()},
            },
        ),
    )


def records_from_notes(
    notes: list[DeckNote],
    deck: str,
    front_field: str = "",
    back_field: str = "",
) -> ImportedDeck:
    """Records for the notes this can read, and the rest held with a reason.

    The shape it understands is the common one: the word on the front, and on
    the back a kana reading followed by one or more English glosses, each on its
    own line. ``front_field``/``back_field`` name the fields to read when the
    notetype calls them something other than Front and Back.

    Held rather than guessed:

    * no expression, or no gloss — there is no card in that;
    * no kana line on the back, and an expression carrying kanji. The reading is
      half of the record id and the Anki GUID derived from it, so a reading
      invented here is not correctable later;
    * *two or more* kana lines. A deck that glosses in Japanese looks exactly
      like one that gives a reading and a gloss, and picking the first would
      quietly file a meaning as a reading.

    An expression that is already kana needs no reading line: it is its own.
    """
    records: list[VocabularyRecord] = []
    held: list[HeldNote] = []
    seen: dict[str, int] = {}

    for note in notes:
        names = list(note.fields)
        front_name = front_field or (names[0] if names else "")
        back_name = back_field or (names[1] if len(names) > 1 else "")
        front_raw = note.fields.get(front_name, "")
        back_raw = note.fields.get(back_name, "")

        front = _lines(front_raw)
        back = _lines(back_raw)
        if not front:
            held.append(HeldNote(note.id, "", back_raw, "the front is empty"))
            continue
        expression, front_variant, _ = _split_variant(front[0])

        stripped = [_split_variant(line) for line in back]
        kana_lines = [item for item in stripped if _looks_like_reading(item[0])]
        rest = [text for text, _, _ in stripped if not _looks_like_reading(text)]
        # The reading is the first kana line the deck wrote *plainly*. A kana
        # line in brackets is how this deck prints the spoken form — sometimes
        # beside the reading, sometimes on its own line — so treating it as a
        # second candidate reading held back every word that has one.
        plain = [item for item in kana_lines if not item[2]]
        bracketed = [item for item in kana_lines if item[2]]
        variants = [
            v for v in (front_variant, *(var for _, var, _ in kana_lines)) if v
        ] + [text for text, _, _ in bracketed if plain]

        if len(plain) > 1:
            held.append(HeldNote(
                note.id, expression, back_raw,
                f"{len(plain)} unbracketed kana lines on the back, so which is "
                f"the reading is a guess: {', '.join(t for t, _, _ in plain)}",
            ))
            continue

        if plain or bracketed:
            reading = _READING_NOISE.sub("", (plain or bracketed)[0][0])
        elif is_kana(_READING_NOISE.sub("", expression)):
            # Its own reading. もう / もう is how this deck writes a kana word,
            # and `よし！` is one with the exclamation the manga prints — which
            # belongs on the card and not in a pronunciation.
            reading = _READING_NOISE.sub("", expression)
        else:
            held.append(HeldNote(
                note.id, expression, back_raw,
                "no kana reading on the back, and the expression has kanji",
            ))
            continue

        meanings = [line for line in rest if line]
        if not meanings:
            held.append(HeldNote(note.id, expression, back_raw, "no meaning on the back"))
            continue

        record = _record(note, deck, expression, reading, meanings, variants)
        if record.id in seen:
            held.append(HeldNote(
                note.id, expression, back_raw,
                f"the same word as note {seen[record.id]} ({record.id})",
            ))
            continue
        seen[record.id] = note.id
        records.append(record)

    return ImportedDeck(records=tuple(records), held=tuple(held))


def staging_name(deck: str) -> str:
    """`data/staging/` file name for a deck, safe on every filesystem."""
    slug = re.sub(r"[^0-9A-Za-z぀-ヿ一-鿿]+", "-", deck).strip("-").lower()
    return f"anki-{slug or 'deck'}.yaml"


def deck_path(staging_dir: Path, deck: str) -> Path:
    return Path(staging_dir) / staging_name(deck)
