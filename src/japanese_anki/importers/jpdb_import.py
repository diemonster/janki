"""Import vocabulary from jpdb.io — from the API, or from a userscript CSV.

Two paths, one destination:

* **API** — ``list-user-decks`` → ``deck/list-vocabulary`` → batched
  ``lookup-vocabulary``, mapped to records here.
* **CSV** — the JPDB-Export userscript's file, which is just another CSV
  dialect and so goes through :mod:`csv_base` with a jpdb label.

Both end at ``cli.run_import``, so merge semantics, the ledger, staging and
the printed outcome are identical to every other importer's.

**Import writes only what jpdb states.** Dictionary facts (spelling, reading,
meanings, pitch accent, frequency rank, JMDict codes) are copied; conjugations
are computed by :mod:`conjugation` and romaji by :mod:`romaji`, both of which
are rules over data already in hand. Nothing here guesses: an entry jpdb has
no verb class for gets an empty ``verb_group`` and no conjugation table, which
``janki status`` counts and a human fills in.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from japanese_anki import jpdb
from japanese_anki.conjugation import conjugate
from japanese_anki.errors import JankiError
from japanese_anki.identifiers import contains_kanji, stable_record_id
from japanese_anki.importers import csv_base
from japanese_anki.importers.csv_base import FIELD_ALIASES, ImportResult
from japanese_anki.models import SourceReference, VocabularyRecord
from japanese_anki.romaji import kana_to_romaji
from japanese_anki.staging import annotate

__all__ = [
    "JPDB_CSV",
    "JpdbImportError",
    "deck_tag",
    "import_csv",
    "import_deck",
    "record_from_entry",
    "select_decks",
]


class JpdbImportError(JankiError):
    pass


JPDB_CSV = csv_base.csv_format(
    source_type="jpdb",
    aliases=FIELD_ALIASES,
    tags=("jpdb",),
    error=JpdbImportError,
)


def import_csv(path: Path) -> ImportResult:
    """Import a JPDB-Export userscript CSV.

    Nothing jpdb-specific happens here beyond the label and the tag: the
    userscript's columns (``spelling``, its furigana-annotated reading) are
    entries in the shared alias table, so the generic reader handles them.
    """
    return csv_base.import_file(path, JPDB_CSV)


# Anki splits tags on whitespace, so a deck name cannot be one. Everything that
# is not a letter or digit — in any script, which is why this is a unicode
# category test and not an ASCII range — collapses to a single hyphen. `::` is
# Anki's tag-hierarchy separator, so a colon in a deck name would silently nest
# the tag somewhere nobody asked for; it collapses too.
def deck_tag(name: str) -> str:
    """``jpdb:<slug>`` for a deck name, or ``"jpdb"`` if nothing survives.

    Deterministic, because it is how a re-import finds the tag it wrote last
    time. Japanese deck names keep their characters — only separators go.
    """
    normalized = unicodedata.normalize("NFKC", name).strip().lower()
    slug = "".join(
        char if (char.isalnum() or char == "-") else " " for char in normalized
    )
    slug = re.sub(r"[\s-]+", "-", slug).strip("-")
    return f"jpdb:{slug}" if slug else "jpdb"


def _meanings(chunks: Any) -> list[str]:
    """One line per sense, its glosses joined.

    jpdb sends ``meanings_chunks`` as a list of senses, each a list of glosses:
    ``[["to talk", "to speak"], ["to tell", "to explain"]]``. The exporter
    renders one ``<br>`` line per list entry, so keeping senses as entries puts
    each sense on its own line; flattening would spill eleven fragments onto a
    card that should read as three meanings.
    """
    if not isinstance(chunks, Sequence) or isinstance(chunks, str):
        return []
    senses: list[str] = []
    for sense in chunks:
        if isinstance(sense, str):
            glosses = [sense]
        elif isinstance(sense, Sequence):
            glosses = [str(part).strip() for part in sense if str(part).strip()]
        else:
            continue
        line = ", ".join(gloss for gloss in glosses if gloss)
        if line:
            senses.append(line)
    return senses


def _card_state(value: Any) -> str:
    """jpdb's card state, flattened for ``raw_fields``.

    The wire form is a list (``["locked", "new"]``) or ``null`` for a word in
    no deck. ``raw_fields`` is a string map, so the list is joined and null
    becomes empty — an absent key, not the string ``"None"``.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Sequence):
        return ",".join(str(item).strip() for item in value if str(item).strip())
    return str(value)


def record_from_entry(
    entry: Mapping[str, Any], *, deck_name: str, source_ref: str
) -> VocabularyRecord:
    """One ``lookup-vocabulary`` entry as a :class:`VocabularyRecord`.

    The reading is *not* validated here — a caller decides what to do with an
    entry whose reading is unusable, and :func:`import_deck` routes those to
    staging rather than dropping them.
    """
    expression = str(entry.get("spelling", "")).strip()
    reading = str(entry.get("reading", "")).strip()
    # Same rule the CSV path applies, for the same reason: a kana-only word's
    # reading is itself, and this must happen before the ID is minted.
    if not reading and expression and not contains_kanji(expression):
        reading = expression

    codes = entry.get("part_of_speech")
    part_of_speech = jpdb.pos_to_part_of_speech(codes)
    verb_group = jpdb.pos_to_verb_group(codes)
    # `conjugate` takes whichever field says how the word inflects. For a verb
    # that is `verb_group`; for an い-adjective jpdb offers no verb class at
    # all (`adj-i` is not a verb code), so the part of speech is what carries
    # it. Anything else normalizes to no table, which is the safe answer.
    conjugations = conjugate(expression, reading, verb_group or part_of_speech)

    raw_fields = {
        "vid": str(entry.get("vid", "")),
        "sid": str(entry.get("sid", "")),
        "deck": deck_name,
    }
    card_state = _card_state(entry.get("card_state"))
    if card_state:
        raw_fields["card_state"] = card_state

    return VocabularyRecord(
        id=stable_record_id(expression, reading),
        expression=expression,
        reading=reading,
        romaji=kana_to_romaji(reading) if reading and not contains_kanji(reading) else "",
        meanings=_meanings(entry.get("meanings_chunks")),
        part_of_speech=part_of_speech,
        verb_group=verb_group,
        transitivity=jpdb.pos_to_transitivity(codes),
        conjugations=conjugations,
        tags=sorted({"jpdb", deck_tag(deck_name)}),
        pitch_accent=jpdb.accent_patterns(entry.get("pitch_accent")),
        frequency_rank=jpdb.frequency_rank(entry.get("frequency_rank")),
        source=SourceReference(
            type="jpdb",
            imported_from=source_ref,
            raw_fields=raw_fields,
        ),
    )


def select_decks(
    decks: Sequence[Mapping[str, Any]], wanted: Sequence[str]
) -> list[dict[str, Any]]:
    """The requested decks, matched by name; every miss is an error.

    Matching ignores case and surrounding space because a deck name is typed
    at a shell prompt, but nothing further — two decks whose names differ only
    in punctuation are two decks, and quietly picking one would import the
    wrong 500 words.
    """
    by_name = {str(deck.get("name", "")).strip().lower(): dict(deck) for deck in decks}
    chosen: list[dict[str, Any]] = []
    missing: list[str] = []
    for name in wanted:
        deck = by_name.get(name.strip().lower())
        if deck is None:
            missing.append(name)
        elif deck not in chosen:
            chosen.append(deck)
    if missing:
        known = ", ".join(sorted(str(deck.get("name", "")) for deck in decks)) or "none"
        raise JpdbImportError(
            f"No jpdb deck named {', '.join(repr(name) for name in missing)}. "
            f"Decks on this account: {known}"
        )
    return chosen


def import_deck(
    client: jpdb.JpdbClient,
    deck: Mapping[str, Any],
    *,
    batch_size: int = jpdb.DEFAULT_BATCH_SIZE,
    fields: Sequence[str] = jpdb.DEFAULT_LOOKUP_FIELDS,
) -> ImportResult:
    """Every word in one jpdb deck, as records.

    One deck per call because ``source.imported_from`` is the deck name: the
    ledger's source references are identified by every key but ``seen_at``, so
    a record that lives in two decks earns two references, and folding several
    decks into one label would erase which deck a word actually came from.

    Entries whose reading cannot serve as part of an ID are held back on
    ``needs_reading`` rather than imported, in both of the ways that happens —
    an empty reading, and a reading written in kanji. jpdb is a dictionary and
    neither should occur, which is exactly why they are surfaced rather than
    swallowed if they do.
    """
    deck_name = str(deck.get("name", "")).strip()
    deck_id = deck.get("id")
    if not deck_name:
        raise JpdbImportError(f"jpdb deck {deck_id!r} has no name to import under.")

    pairs = client.list_deck_vocabulary(deck_id)
    seen_pairs: set[tuple[int, int]] = set()
    wanted: list[list[int]] = []
    for entry in pairs:
        pair = (int(entry["vid"]), int(entry["sid"]))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        wanted.append([pair[0], pair[1]])

    warnings: list[str] = []
    records: list[VocabularyRecord] = []
    needs_reading: list[VocabularyRecord] = []
    seen_ids: set[str] = set()

    for entry in client.lookup_vocabulary(wanted, fields, batch_size=batch_size):
        record = record_from_entry(entry, deck_name=deck_name, source_ref=deck_name)
        where = f"{deck_name}: vid {entry.get('vid')}"
        if not record.expression:
            warnings.append(f"{where}: no spelling in the jpdb entry; skipped")
            continue
        if record.id in seen_ids:
            warnings.append(
                f"{where}: {record.expression} [{record.reading}] appears twice in "
                "this deck under different ids; kept the first"
            )
            continue
        seen_ids.add(record.id)
        if not record.reading:
            warnings.append(
                f"{where}: {record.expression} contains kanji but jpdb returned no "
                "reading; held back for reading review"
            )
            needs_reading.append(annotate(record, hold_reason="missing reading"))
            continue
        if contains_kanji(record.reading):
            warnings.append(
                f"{where}: the reading jpdb returned for {record.expression} is "
                f"written in kanji ({record.reading}); held back for reading review"
            )
            needs_reading.append(annotate(record, hold_reason="reading contains kanji"))
            continue
        records.append(record)

    return ImportResult(
        records=records,
        warnings=warnings,
        mapping={},
        needs_reading=needs_reading,
    )
