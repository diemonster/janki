"""Fill empty fields on records that came from somewhere other than jpdb.

``janki enrich --jpdb`` is the other half of the jpdb integration:
``import-jpdb`` creates records *from* jpdb, this improves records janki
already has — a Shirabe export, a photographed vocabulary table — using jpdb as
a dictionary.

**Empty fields only, and never ``reading``.** The reading is half of
``stable_record_id``, so writing one would change the record's ID and orphan
whatever review history Anki has against it. That is also why a disagreement
between janki's reading and jpdb's is reported and nothing is written: the two
readings may both be right (a homograph janki holds the other sense of), and
the one thing that must not happen is a dictionary quietly re-IDing a curated
record. ``--force-fields`` widens what may be overwritten, and refuses
``reading`` explicitly for the same reason.

The reading is also what makes the *right* dictionary entry findable. 一日 is
いちにち (one day) or ついたち (the first of the month) depending on which word
you meant, and ``/parse`` picks for itself unless it is told. So the pass runs
``/parse`` twice on purpose:

1. Unforced, to learn what jpdb thinks the expression reads as. Nothing is
   written from this pass when the readings disagree — its job is to give the
   comparison something to compare.
2. Forced with janki's own reading (``jpdb.forced_furigana_span``), but only
   after that reading has been confirmed to be one jpdb actually lists for the
   word. Forcing a reading jpdb has never heard of would get an answer shaped
   like agreement.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from japanese_anki import jpdb
from japanese_anki.conjugation import conjugate
from japanese_anki.errors import JankiError
from japanese_anki.identifiers import contains_kanji
from japanese_anki.io import is_empty
from japanese_anki.models import VocabularyRecord
from japanese_anki.romaji import kana_to_romaji
from japanese_anki.staging import annotate, annotations

__all__ = [
    "ENRICHABLE_FIELDS",
    "EnrichError",
    "EnrichResult",
    "SuggestionResult",
    "enrich_records",
    "format_field_diff",
    "needs_reading",
    "parse_force_fields",
    "suggest_readings",
]


class EnrichError(JankiError):
    pass


#: The fields a jpdb pass may write, in the order a diff lists them.
#:
#: ``reading`` is absent deliberately and permanently — see the module
#: docstring. ``meanings`` is absent too: jpdb's glosses are a dictionary's,
#: and a record that reached janki from a textbook carries the meaning that
#: textbook taught. Overwriting a hole in ``meanings`` is M4.2's call to make
#: with an AI pass that can read the record's examples, not this one's.
ENRICHABLE_FIELDS: tuple[str, ...] = (
    "furigana",
    "romaji",
    "part_of_speech",
    "verb_group",
    "conjugations",
    "pitch_accent",
    "frequency_rank",
)


def parse_force_fields(value: str | None) -> tuple[str, ...]:
    """Parse a ``--force-fields FIELD[,FIELD]`` option value.

    Shared with M4.2's AI pass: which fields an enrichment may overwrite is one
    question, however the values were obtained.
    """
    if not value:
        return ()
    names = [part.strip() for part in value.split(",") if part.strip()]
    if not names:
        # Separators only. The flag was passed and names nothing; enriching as
        # if it were absent would hide the typo behind a pass that looks like
        # it worked.
        raise EnrichError(f"--force-fields got no field names in {value!r}")
    for name in names:
        if name == "reading":
            raise EnrichError(
                "--force-fields cannot take 'reading': it is half of the record ID, "
                "so overwriting it would re-ID the record and orphan its Anki review "
                "history. A reading janki has wrong is fixed by hand, through "
                "data/staging review."
            )
        if name not in ENRICHABLE_FIELDS:
            raise EnrichError(
                f"--force-fields: unknown field '{name}'. "
                f"Valid fields: {', '.join(ENRICHABLE_FIELDS)}"
            )
    return tuple(dict.fromkeys(names))


@dataclass(slots=True)
class EnrichResult:
    """What a pass would write, and what it could not.

    ``records`` is the whole input list with the enriched ones replaced, so a
    caller saves it as-is; ``changes`` maps record ID to ``{field: (old, new)}``
    and is what the field diff renders. ``looked_up`` counts the records that
    cost an API call, which is what separates "nothing needed filling" from
    "jpdb had nothing to add".
    """

    records: list[VocabularyRecord] = field(default_factory=list)
    changes: dict[str, dict[str, tuple[Any, Any]]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    skipped: int = 0
    looked_up: int = 0

    @property
    def changed_fields(self) -> list[str]:
        """Every field this pass wrote, deduplicated — the ledger's ``fields``."""
        return sorted({name for fields in self.changes.values() for name in fields})


def needs_reading(record: VocabularyRecord) -> bool:
    """Is this record one an import held back for reading review?

    Two tests, either of which is enough. ``hold_reason`` is what the importers
    write, but a staging file is also hand-edited, and a row whose reading a
    reviewer half-filled with kanji is still held whatever the annotation says.
    Post-M1.5 a needs-reading file is *not* all reading-less: the second hold
    class mints ``word:<kanji>:<kanji>``, which has a reading.
    """
    if "hold_reason" in annotations(record):
        return True
    return not record.reading or contains_kanji(record.reading)


def _resolved(result: jpdb.ParseResult) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """The parsed tokens that resolved to a dictionary entry, paired with it."""
    pairs = []
    for token in result.tokens:
        entry = result.vocabulary_for(token)
        if entry is not None:
            pairs.append((token, entry))
    return pairs


def _dictionary_entry(
    result: jpdb.ParseResult, expression: str
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """The ``(token, entry)`` for ``expression``, or ``None`` if it is ambiguous.

    A record holds one vocabulary item, so the ordinary answer is one token.
    When jpdb splits the expression into several — a compound it does not list,
    a phrase someone recorded as a word — there is no single entry whose pitch
    accent and frequency rank describe the record, and picking the first would
    file 食べ物's data under 食べ. Preferring an exact spelling match first
    handles the case where one token *is* the whole expression.
    """
    pairs = _resolved(result)
    for token, entry in pairs:
        if str(entry.get("spelling", "")).strip() == expression:
            return token, entry
    return pairs[0] if len(pairs) == 1 else None


def _parse(
    client: jpdb.JpdbClient, expression: str, reading: str = ""
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """``/parse`` one expression, optionally forcing ``reading``."""
    forced = jpdb.forced_furigana_span(expression, reading) if reading else None
    return _dictionary_entry(client.parse(expression, forced_furigana=forced), expression)


def _readings_for(client: jpdb.JpdbClient, entry: Mapping[str, Any]) -> set[str]:
    """Every reading jpdb lists for this word, across all of its senses.

    ``alt_sids`` are the entry's other senses; each carries its own reading,
    and a homograph's *other* reading lives there rather than on the sense
    ``/parse`` happened to pick. Missing ``alt_sids`` is normal — most words
    have one sense — and yields just the parsed entry's own reading.
    """
    vid, sid = entry.get("vid"), entry.get("sid")
    readings: set[str] = set()
    alt_sids: list[int] = []
    for row in client.lookup_vocabulary([[vid, sid]], ("reading", "alt_sids")):
        if reading := str(row.get("reading") or "").strip():
            readings.add(reading)
        raw = row.get("alt_sids")
        for item in raw if isinstance(raw, Sequence) and not isinstance(raw, str) else []:
            if isinstance(item, int) and not isinstance(item, bool) and item != sid:
                alt_sids.append(item)
    if alt_sids:
        pairs = [[vid, alt] for alt in dict.fromkeys(alt_sids)]
        for row in client.lookup_vocabulary(pairs, ("reading",)):
            if reading := str(row.get("reading") or "").strip():
                readings.add(reading)
    return readings


def _proposals(
    record: VocabularyRecord, token: Mapping[str, Any], entry: Mapping[str, Any]
) -> dict[str, Any]:
    """Every value jpdb states, or janki computes, for one record.

    Two sources, deliberately distinguishable. Pitch accent, frequency rank and
    the part-of-speech codes are jpdb's *statements*, copied. Romaji and the
    conjugation table are janki's *rules* run over data it already holds — and
    run over the record's own reading, never jpdb's, so a table cannot describe
    a word the record is not.
    """
    codes = entry.get("part_of_speech")
    part_of_speech = jpdb.pos_to_part_of_speech(codes)
    verb_group = jpdb.pos_to_verb_group(codes)
    kana_reading = record.reading and not contains_kanji(record.reading)
    return {
        "furigana": jpdb.furigana_to_anki(token.get("furigana")),
        "romaji": kana_to_romaji(record.reading) if kana_reading else "",
        "part_of_speech": part_of_speech,
        "verb_group": verb_group,
        # Same rule the importer applies: jpdb has no verb class for an
        # い-adjective, so the part of speech is what carries its inflection.
        "conjugations": conjugate(
            record.expression, record.reading, verb_group or part_of_speech
        ),
        "pitch_accent": jpdb.accent_patterns(entry.get("pitch_accent")),
        "frequency_rank": jpdb.frequency_rank(entry.get("frequency_rank")),
    }


def _wanted(record: VocabularyRecord, force_fields: Sequence[str]) -> list[str]:
    """The fields this pass would write to on this record, if jpdb has them."""
    return [
        name
        for name in ENRICHABLE_FIELDS
        if name in force_fields or is_empty(getattr(record, name))
    ]


def _apply(
    record: VocabularyRecord, proposals: Mapping[str, Any], writable: Sequence[str]
) -> tuple[VocabularyRecord, dict[str, tuple[Any, Any]]]:
    changes: dict[str, tuple[Any, Any]] = {}
    for name in writable:
        new = proposals.get(name)
        old = getattr(record, name)
        # An empty proposal is "jpdb had nothing", not "blank this out" — which
        # matters under --force-fields, where the field being non-empty is
        # exactly the case.
        if is_empty(new) or new == old:
            continue
        changes[name] = (old, new)
    if not changes:
        return record, {}
    return replace(record, **{name: new for name, (_, new) in changes.items()}), changes


def enrich_records(
    client: jpdb.JpdbClient,
    records: Sequence[VocabularyRecord],
    *,
    force_fields: Sequence[str] = (),
    ids: Sequence[str] | None = None,
) -> EnrichResult:
    """Fill empty fields on ``records`` (or just ``ids``) from jpdb.

    Records with nothing to fill never reach the network: a second run over a
    collection the first run finished costs one no API calls, which is what
    makes re-running this cheap enough to be routine.
    """
    result = EnrichResult(records=list(records))
    by_id = {record.id: index for index, record in enumerate(result.records)}
    if ids is None:
        targets = list(by_id)
    else:
        targets = list(dict.fromkeys(ids))
        if missing := [record_id for record_id in targets if record_id not in by_id]:
            raise EnrichError(
                f"No record with id {', '.join(repr(item) for item in missing)} in the "
                "normalized file. Enrichment reads that file only, so an id that "
                "'janki status --format ids' lists but this rejects belongs to an "
                "inline deck note ('janki migrate-inline' moves it) or a staged row "
                "(finish its reading review first)."
            )

    for record_id in targets:
        record = result.records[by_id[record_id]]
        writable = _wanted(record, force_fields)
        if not writable or not record.expression:
            result.skipped += 1
            continue
        result.looked_up += 1
        found = _parse(client, record.expression)
        if found is None:
            result.warnings.append(
                f"{record_id}: jpdb did not parse {record.expression} as one word; "
                "no single dictionary entry to enrich from, so nothing was written"
            )
            continue
        token, entry = found
        jpdb_reading = str(entry.get("reading") or "").strip()
        if record.reading and record.reading != jpdb_reading:
            known = _readings_for(client, entry)
            if record.reading not in known:
                listed = ", ".join(sorted(known)) or jpdb_reading or "none"
                result.warnings.append(
                    f"{record_id}: janki reads {record.expression} as "
                    f"{record.reading}, jpdb lists {listed}. Nothing was written — "
                    "the reading is part of the record ID and is never auto-fixed."
                )
                continue
            # janki's reading is one jpdb knows, so the first parse simply
            # picked the other homograph. Ask again, pinned to this one.
            found = _parse(client, record.expression, record.reading)
            if found is None:
                result.warnings.append(
                    f"{record_id}: jpdb did not parse {record.expression} as one word "
                    f"when given the reading {record.reading}; nothing was written"
                )
                continue
            token, entry = found

        updated, changes = _apply(record, _proposals(record, token, entry), writable)
        if changes:
            result.records[by_id[record_id]] = updated
            result.changes[record_id] = changes
    return result


@dataclass(slots=True)
class SuggestionResult:
    """What a staging pass proposes: the rewritten rows, plus what it proposed.

    ``suggested`` maps record ID to the reading jpdb offered, so the caller can
    report and count without re-deriving which rows changed from the record
    objects — a comparison that would quietly say "changed" for a row re-offered
    the same reading it already carried.
    """

    records: list[VocabularyRecord] = field(default_factory=list)
    suggested: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    held: int = 0


def suggest_readings(
    client: jpdb.JpdbClient, records: Sequence[VocabularyRecord]
) -> SuggestionResult:
    """Annotate held rows with the reading jpdb proposes, for a human to confirm.

    A proposal, never a fix: ``suggested_reading`` is an annotation in
    ``raw_fields`` and the row stays held until someone types the reading into
    the ``reading`` field themselves. That is the whole design of the staging
    review — jpdb is one more opinion about a word whose reading janki could
    not determine, and the ID it would mint is unrecoverable if it is wrong.

    The unforced parse is the only one that makes sense here: forcing would
    need the reading that is missing.
    """
    result = SuggestionResult(records=list(records))
    for index, record in enumerate(result.records):
        if not needs_reading(record) or not record.expression:
            continue
        result.held += 1
        found = _parse(client, record.expression)
        if found is None:
            result.warnings.append(
                f"{record.id}: jpdb did not parse {record.expression} as one word; "
                "no reading suggested"
            )
            continue
        reading = str(found[1].get("reading") or "").strip()
        if not reading:
            result.warnings.append(
                f"{record.id}: jpdb has no reading for {record.expression}"
            )
            continue
        result.records[index] = annotate(record, suggested_reading=reading)
        result.suggested[record.id] = reading
    return result


def _render(value: Any) -> str:
    """One field value on a diff line: readable, and never wrapped."""
    if value is None:
        return "(none)"
    if isinstance(value, str):
        text = value
    elif isinstance(value, Mapping):
        text = ", ".join(f"{key}={item}" for key, item in value.items())
    elif isinstance(value, Sequence):
        text = "; ".join(str(item) for item in value)
    else:
        text = str(value)
    text = " ".join(text.split())
    if not text:
        return "(empty)"
    return text if len(text) <= 60 else f"{text[:57]}..."


def format_field_diff(
    changes: Mapping[str, Mapping[str, tuple[Any, Any]]],
) -> list[str]:
    """The shared per-record field diff, one line per changed field.

    A record ID header, then ``  <field>: <old> -> <new>`` beneath it. Shared
    with M4.2/M4.3 so every command that proposes a write to an existing record
    shows the same thing (IMPLEMENTATION_PLAN, Conventions).
    """
    lines: list[str] = []
    for record_id in sorted(changes):
        lines.append(record_id)
        fields = changes[record_id]
        for name in ENRICHABLE_FIELDS:
            if name in fields:
                old, new = fields[name]
                lines.append(f"  {name}: {_render(old)} -> {_render(new)}")
    return lines
