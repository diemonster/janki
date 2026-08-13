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

import functools
import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, is_dataclass, replace
from dataclasses import fields as dc_fields
from typing import Any

from japanese_anki import claude_client, jpdb, pitch, qc
from japanese_anki.conjugation import conjugate
from japanese_anki.errors import JankiError
from japanese_anki.identifiers import contains_kanji, short_fingerprint
from japanese_anki.io import is_empty
from japanese_anki.ledger import Ledger
from japanese_anki.models import ExampleSentence, VocabularyRecord
from japanese_anki.romaji import kana_to_romaji
from japanese_anki.staging import NON_READING_HOLDS, annotate, annotations

__all__ = [
    "ENRICHABLE_FIELDS",
    "DictionaryReadings",
    "EnrichError",
    "EnrichResult",
    "SuggestionResult",
    "AI_FIELDS",
    "POLISH_FIELDS",
    "STAGING_THRESHOLD",
    "UNVERIFIED_KEY",
    "ai_prompt",
    "ai_retry_prompt",
    "ai_schema",
    "ai_targets",
    "absorb_ai_call",
    "apply_ai_result",
    "apply_batch_results",
    "apply_polish_batch_results",
    "batch_custom_id",
    "batch_key_map",
    "batch_requests",
    "dictionary_readings",
    "enrich_ai",
    "enrich_records",
    "format_field_diff",
    "format_ai_no_changes",
    "needs_reading",
    "parse_force_fields",
    "polish_meanings",
    "polish_batch_requests",
    "polish_call_outcome",
    "polish_prompt",
    "polish_prompt_fingerprint",
    "polish_schema",
    "polish_targets",
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


#: Fields the AI pass may write. Disjoint from :data:`ENRICHABLE_FIELDS` on
#: purpose — a dictionary pass and a writing pass fill different holes, and a
#: record needing one does not need the other.
AI_FIELDS: tuple[str, ...] = ("examples", "usage_notes")

# The order :func:`format_field_diff` prints known fields in: jpdb's pass, then
# the AI pass. Anything outside it still prints, after these.
_DIFF_FIELD_ORDER: tuple[str, ...] = ENRICHABLE_FIELDS + AI_FIELDS


def parse_force_fields(value: str | None, *, ai: bool = False) -> tuple[str, ...]:
    """Parse a ``--force-fields FIELD[,FIELD]`` option value.

    Shared between the two passes, because "which fields may this overwrite" is
    one question however the values were obtained — but the *answer* differs:
    the jpdb pass writes dictionary facts and the AI pass writes prose, and
    naming a field the running pass cannot write is a typo worth catching
    rather than a no-op to shrug at.
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
        allowed = AI_FIELDS if ai else ENRICHABLE_FIELDS
        if name not in allowed:
            other = ENRICHABLE_FIELDS if ai else AI_FIELDS
            hint = (
                f" ('{name}' is a --{'jpdb' if ai else 'ai'} field.)"
                if name in other
                else ""
            )
            raise EnrichError(
                f"--force-fields: unknown field '{name}' for this pass. "
                f"Valid fields: {', '.join(allowed)}.{hint}"
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

    The *value* is tested rather than the key's presence, because one hold class
    is not about the reading at all: ``HOLD_UNVERIFIABLE_ID`` marks a row whose
    reading is fine and whose id could not be checked. Proposing a reading for
    it would spend a call on a settled question and, for a homograph the
    reviewer deliberately chose, suggest the reading they rejected — which typed
    in would mint a different, permanent id.

    Tested against ``NON_READING_HOLDS`` rather than a list of reading holds: a
    staging file is hand-edited, so an unrecognised reason has to keep meaning
    what a reason has always meant here.
    """
    reason = annotations(record).get("hold_reason")
    if reason is not None and reason not in NON_READING_HOLDS:
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


@dataclass(frozen=True, slots=True)
class DictionaryReadings:
    """What jpdb says a spelling can be read as.

    ``primary`` is the reading of the entry ``/parse`` resolved to — the one
    jpdb reaches for unprompted. ``all_readings`` includes it plus every other
    sense's, which is how a valid-but-not-primary reading is told apart from
    one the dictionary has never heard of.
    """

    primary: str
    all_readings: frozenset[str]
    supports_suru_suffix: bool = False


def _supports_suru_suffix(
    expression: str, reading: str, entry: Mapping[str, Any]
) -> bool:
    """Whether jpdb's stem entry proves this exact ``Xする`` identity."""
    entry_spelling = str(entry.get("spelling") or "").strip()
    entry_reading = str(entry.get("reading") or "").strip()
    part_of_speech = entry.get("part_of_speech")
    pos_codes = (
        {
            str(item).strip()
            for item in part_of_speech
            if isinstance(item, str) and item.strip()
        }
        if isinstance(part_of_speech, Sequence)
        and not isinstance(part_of_speech, str)
        else set()
    )
    return (
        expression.endswith("する")
        and reading.endswith("する")
        and bool(expression[:-2])
        and entry_spelling == expression[:-2]
        and entry_reading == reading[:-2]
        and any(code == "vs" or code.startswith("vs-") for code in pos_codes)
    )


def dictionary_readings(
    client: jpdb.JpdbClient, expression: str, reading: str = ""
) -> DictionaryReadings | None:
    """Every reading jpdb lists for ``expression``, or ``None`` if it cannot say.

    Shared with M3.4's promote-time reading check rather than reimplemented
    there: walking ``alt_sids`` to find a homograph's other reading is exactly
    the kind of thing that goes subtly wrong in a second copy, and both callers
    are asking the same question.

    ``None`` means jpdb did not resolve the spelling to a single entry — not
    that the reading is wrong. A caller has to tell those apart, because
    "the dictionary disagrees" and "the dictionary has no opinion" deserve
    different answers.
    """
    found = _parse(client, expression, reading)
    if found is None:
        return None
    entry = found[1]
    entry_reading = str(entry.get("reading") or "").strip()
    return DictionaryReadings(
        primary=entry_reading,
        all_readings=frozenset(_readings_for(client, entry)),
        supports_suru_suffix=_supports_suru_suffix(expression, reading, entry),
    )


def _furigana_for(
    record: VocabularyRecord, token: Mapping[str, Any], kanji_store: Any | None
) -> str:
    """jpdb's furigana for this word, or the whole-word form when its split lies.

    jpdb hands back one reading per *character*: 明日 arrives as
    ``[["明","あ"], ["日","した"]]``. That is right for an ordinary compound and
    wrong for a jukujikun, where the reading belongs to the word — した is no
    reading of 日, and a card built from that split teaches two readings that do
    not exist. Checked against KANJIDIC, which janki already holds in
    `data/kanji.json`, so this is a lookup rather than an opinion.

    Falls back to ``expression[reading]``, which is always true: it says the
    word is read that way and claims nothing about which character contributes
    what. Without a kanji store — nobody has run `janki kanji` — jpdb's split
    stands, because silence is not disagreement.
    """
    written = jpdb.furigana_to_anki(token.get("furigana"))
    segments = token.get("furigana")
    if kanji_store is None or not isinstance(segments, list) or len(segments) < 2:
        return written
    from japanese_anki.kanji import assigns_a_known_reading

    for segment in segments:
        # A bare string is kana the word carries verbatim; only a pair claims a
        # character is read a particular way.
        if not isinstance(segment, list) or len(segment) != 2:
            continue
        text, reading = str(segment[0]), str(segment[1])
        if len(text) != 1:
            continue
        if not assigns_a_known_reading(kanji_store.entries.get(text), reading):
            return f"{record.expression}[{record.reading}]" if record.reading else written
    return written


def _proposals(
    record: VocabularyRecord,
    token: Mapping[str, Any],
    entry: Mapping[str, Any],
    kanji_store: Any | None = None,
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
        "furigana": _furigana_for(record, token, kanji_store),
        "romaji": kana_to_romaji(record.reading) if kana_reading else "",
        "part_of_speech": part_of_speech,
        "verb_group": verb_group,
        # Same rule the importer applies: jpdb has no verb class for an
        # い-adjective, so the part of speech is what carries its inflection.
        "conjugations": conjugate(
            record.expression, record.reading, verb_group or part_of_speech
        ),
        "pitch_accent": _compatible_pitch_patterns(
            record.reading, entry.get("pitch_accent")
        )[0],
        "frequency_rank": jpdb.frequency_rank(entry.get("frequency_rank")),
    }


def _compatible_pitch_patterns(
    reading: str, value: Any
) -> tuple[list[str], list[str]]:
    """Split jpdb patterns into compatible and unusable values."""
    valid: list[str] = []
    invalid: list[str] = []
    for pattern in jpdb.accent_patterns(value):
        try:
            pitch.to_aquestalk(reading, pattern)
        except pitch.PitchError:
            invalid.append(pattern)
        else:
            valid.append(pattern)
    return valid, invalid


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
    kanji_store: Any | None = None,
) -> EnrichResult:
    """Fill empty fields on ``records`` (or just ``ids``) from jpdb.

    ``kanji_store`` is consulted for one thing only: whether jpdb's
    per-character furigana assigns a character a reading it actually has. See
    :func:`_furigana_for`.

    A record with nothing left to fill never reaches the network. Note what that
    does *not* say: a field jpdb has no answer for stays empty, so the record
    stays fillable and is looked up again on every run. A noun has no verb group
    and no conjugation table, and nothing here records "asked, and there was
    nothing" — so re-running over a collection of nouns costs roughly one call
    per word, while re-running over one the dictionary could fully describe
    costs nothing. Memoizing the negative answer would need somewhere to keep
    it that is not the record.
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
        entry_spelling = str(entry.get("spelling", "")).strip()
        if not record.reading and str(entry.get("spelling", "")).strip() != (
            record.expression
        ):
            # A record with a reading proves the entry is the right word by
            # matching it against the entry's reading set below. One without a
            # reading has nothing to prove it with, and jpdb resolves an
            # inflected surface form to its lemma — so 行った would be filled
            # with 行く's pitch accent and frequency rank, which describe a
            # different word.
            result.warnings.append(
                f"{record_id}: jpdb resolved {record.expression} to its entry for "
                f"{str(entry.get('spelling', '')).strip() or 'another word'}, and this "
                "record has no reading to confirm they are the same word. Nothing was "
                "written; fill in the reading first."
            )
            continue
        if (
            record.reading
            and record.reading != jpdb_reading
            and entry_spelling != record.expression
        ):
            # An all-kana expression can tokenize as a different, more common
            # word. Do not use that unrelated entry to reject the reviewed
            # identity. Retry with the stored reading before any dictionary
            # field is trusted. The ordinary exact-spelling homograph path
            # below still checks the entry's declared reading set first.
            pinned = _parse(client, record.expression, record.reading)
            if pinned is not None:
                pinned_token, pinned_entry = pinned
                pinned_reading = str(pinned_entry.get("reading") or "").strip()
                if _supports_suru_suffix(
                    record.expression, record.reading, pinned_entry
                ) or (
                    pinned_reading == record.reading
                    or record.reading in _readings_for(client, pinned_entry)
                ):
                    token, entry = pinned_token, pinned_entry
                    jpdb_reading = pinned_reading
        if record.reading and record.reading != jpdb_reading:
            known = _readings_for(client, entry)
            supports_suru_suffix = _supports_suru_suffix(
                record.expression, record.reading, entry
            )
            if record.reading not in known and not supports_suru_suffix:
                listed = ", ".join(sorted(known)) or jpdb_reading or "none"
                result.warnings.append(
                    f"{record_id}: janki reads {record.expression} as "
                    f"{record.reading}, jpdb lists {listed}. Nothing was written — "
                    "the reading is part of the record ID and is never auto-fixed."
                )
                continue
            if not supports_suru_suffix:
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

        proposals = _proposals(record, token, entry, kanji_store)
        _valid_pitch, invalid_pitch = _compatible_pitch_patterns(
            record.reading, entry.get("pitch_accent")
        )
        if invalid_pitch:
            result.warnings.append(
                f"{record_id}: jpdb pitch pattern(s) {', '.join(invalid_pitch)} "
                f"do not fit reading {record.reading}; pitch accent was not written"
            )
        updated, changes = _apply(record, proposals, writable)
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

    The suggestion comes from the token's **furigana**, not the dictionary
    entry's ``reading``. Those differ exactly when they matter most: a held row
    whose expression is an inflected form (行った, off a photographed vocabulary
    table) resolves to the entry for 行く, whose reading is いく — which does not
    read 行った. A reviewer who trusted that and typed it in would mint
    ``word:行った:いく`` permanently. The furigana over the surface form gives
    いった, and it is the dictionary's segmentation either way.
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
        token, entry = found
        reading = jpdb.furigana_to_reading(token.get("furigana"))
        spelling = str(entry.get("spelling", "")).strip()
        if not reading and spelling == record.expression:
            # No furigana on a token whose entry *is* this word: an all-kana
            # expression, where the entry's reading describes the same surface
            # form and can be trusted.
            reading = str(entry.get("reading") or "").strip()
        if not reading:
            result.warnings.append(
                f"{record.id}: jpdb resolved {record.expression} to its entry for "
                f"{spelling or 'another word'} and stated no furigana for the form "
                "as written, so no reading was suggested — the entry's own reading "
                "would be the wrong word's"
            )
            continue
        result.records[index] = annotate(record, suggested_reading=reading)
        result.suggested[record.id] = reading
    return result


def _render_item(item: Any) -> str:
    """One element of a list field. An :class:`ExampleSentence` renders as its
    Japanese: a dataclass repr on a diff line is furigana, romaji and audio
    paths crowding out the one part a reviewer actually judges."""
    if isinstance(item, ExampleSentence):
        return item.japanese
    return str(item)


def _render(value: Any) -> str:
    """One field value on a diff line: readable, and never wrapped."""
    if value is None:
        return "(none)"
    if isinstance(value, str):
        text = value
    elif isinstance(value, Mapping):
        text = ", ".join(f"{key}={item}" for key, item in value.items())
    elif isinstance(value, Sequence):
        text = "; ".join(_render_item(item) for item in value)
    else:
        text = str(value)
    text = " ".join(text.split())
    if not text:
        return "(empty)"
    return text if len(text) <= 60 else f"{text[:57]}..."


def _elements(value: Any) -> list[Any]:
    """``value`` as a list of elements, or empty for anything without them."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return []


def _hidden_difference(old: Any, new: Any) -> str:
    """Name what changed where the diff line could not show it.

    ``_render`` is lossy on purpose — one line per field, sixty characters, an
    :class:`ExampleSentence` shown as its Japanese. Usually that is the readable
    summary. Sometimes it is a write nobody can see: replacing a curated
    example's English with a model's leaves the Japanese identical, so that
    element reads the same on both sides and a y confirms an overwrite that was
    never displayed. Rather than widening the line for every field, the parts
    rendering hid say what they hid.

    Elements are paired by **what they render as**, not by position. A model
    answering with a different number of examples, or the same ones in another
    order, is the ordinary case — and a positional pairing would let either of
    those switch the check off exactly when the list is being rewritten
    wholesale. An element that renders the same on both sides is the same
    sentence as far as this line is concerned; if the values behind it differ,
    that is the write nobody can see.
    """
    if old == new:
        return ""
    names: list[str] = []
    after = _elements(new)
    for left in _elements(old):
        if not is_dataclass(left):
            continue
        same_text = [
            right
            for right in after
            if type(right) is type(left) and _render_item(right) == _render_item(left)
        ]
        # Gone from the new side, or still there untouched: either way the line
        # is not hiding anything about it.
        if not same_text or any(right == left for right in same_text):
            continue
        names.extend(
            item.name
            for item in dc_fields(left)
            if getattr(left, item.name) != getattr(same_text[0], item.name)
        )
    if names:
        return f"({', '.join(dict.fromkeys(names))} differ)"
    if _render(old) == _render(new):
        # Nothing structured to point at — a truncated string, a list that
        # changed length inside the sixty characters — but the two sides are
        # not the same value, and the line says they look it.
        return "(differs where this line cannot show it)"
    return ""


def format_field_diff(
    changes: Mapping[str, Mapping[str, tuple[Any, Any]]],
) -> list[str]:
    """The shared per-record field diff, one line per changed field.

    A record ID header, then ``  <field>: <old> -> <new>`` beneath it. Shared
    with M4.2/M4.3 so every command that proposes a write to an existing record
    shows the same thing (IMPLEMENTATION_PLAN, Conventions).

    Known fields print in pass order — jpdb's, then the AI pass's — and anything
    else prints after them rather than not at all. A diff that quietly omits a
    changed field is the display version of discarding a row: the user answers
    y to a write they were never shown.
    """
    lines: list[str] = []
    for record_id in sorted(changes):
        lines.append(record_id)
        fields = changes[record_id]
        rest = sorted(name for name in fields if name not in _DIFF_FIELD_ORDER)
        for name in (*_DIFF_FIELD_ORDER, *rest):
            if name in fields:
                old, new = fields[name]
                line = f"  {name}: {_render(old)} -> {_render(new)}"
                if hidden := _hidden_difference(old, new):
                    line = f"{line} {hidden}"
                lines.append(line)
    return lines


# --- the AI pass -------------------------------------------------------------
#
# jpdb fills what a dictionary knows. This fills what it does not: an example
# sentence a beginner can read, and a note about how the word is actually used.
# Both are written rather than looked up, so everything here is arranged around
# not believing the result until a machine check or a human says so.


#: Where a flagged example is recorded (IMPLEMENTATION_PLAN, Conventions):
#: a comma-joined list of content fingerprints of the flagged examples'
#: ``japanese`` text, on the record rather than per example.
UNVERIFIED_KEY = "furigana_unverified"

#: How many other records' examples ride along as variety pressure. Enough to
#: show the model what it has already written this run, few enough that the
#: prompt stays mostly the record in front of it.
VARIETY_EXAMPLES = 3

#: Past this many target records, a monolithic diff stops being review — so the
#: results go to a staging file and through `janki promote` instead
#: (DESIGN_V2: "a 500-record y/n diff is not review; a staging file is").
STAGING_THRESHOLD = 50


@functools.cache
def ai_schema() -> Any:
    """The Pydantic model an AI enrichment response must match.

    Built on demand and cached, for the same reasons :func:`extract` builds its
    own that way: ``pydantic`` arrives with the ``ai`` extra, and a fresh class
    per call would present an identical schema to the API as new every request.
    """
    from pydantic import BaseModel, Field

    class GeneratedExample(BaseModel):
        japanese: str = Field(description="The sentence, in Japanese.")
        # Not `register`: that name shadows an attribute on pydantic's BaseModel
        # and the class emits a warning on every construction. The record field
        # keeps the linguistic term.
        speech_level: str = Field(
            default="polite",
            description=(
                "'polite' for a 〜ます/です sentence, 'casual' for the plain form "
                "a friend would use."
            ),
        )
        furigana: str = Field(
            default="",
            description=(
                "The same sentence in Anki furigana notation — 話[はな]す — with "
                "a space before every bracketed group that follows kana."
            ),
        )
        romaji: str = Field(
            default="", description="Ignored; janki regenerates this from the furigana."
        )
        english: str = Field(default="", description="A natural English translation.")

    class Enrichment(BaseModel):
        examples: list[GeneratedExample] = Field(default_factory=list)
        usage_notes: str = Field(
            default="",
            description=(
                "How the word is actually used: register, common collocations, "
                "what a learner is likely to get wrong. Empty if there is "
                "nothing worth saying."
            ),
        )

    return Enrichment


def ai_targets(
    records: Sequence[VocabularyRecord], ids: Sequence[str] | None = None
) -> list[VocabularyRecord]:
    """The records an AI pass would work on.

    Content-defined by default, via the ledger's own rule: a record with no
    example sentence or no usage notes needs this pass, whatever any previous
    pass recorded about it. Naming ids explicitly overrides that — re-running
    over a record that already has an example is a legitimate thing to ask for,
    and `--force-fields` is what decides whether the answer may replace it.
    """
    if ids is not None:
        wanted = list(dict.fromkeys(ids))
        by_id = {record.id: record for record in records}
        if missing := [item for item in wanted if item not in by_id]:
            raise EnrichError(
                f"No record with id {', '.join(repr(item) for item in missing)} in the "
                "normalized file."
            )
        return [by_id[item] for item in wanted]
    needed = set(Ledger.missing_enrichment(records))
    return [record for record in records if record.id in needed]


def ai_prompt(
    record: VocabularyRecord,
    recent: Sequence[str] = (),
    taught: str = "",
) -> str:
    """The user turn for one record: what janki knows, and what it has seen.

    The dictionary facts go in so the model writes about *this* word rather
    than a homograph — 一日 with its reading attached is a different request
    from 一日 alone. The recent examples go in as variety pressure: asked for
    an example of twenty verbs in a row, a model will write twenty variations
    of 毎日〜ます unless it can see that it already did.

    ``taught`` is the grammar the learner is currently studying, from documents
    they have read and reviewed. It is a preference, not an instruction: a
    sentence forced into a pattern that does not suit the word is worse than one
    in ordinary Japanese, and the block says so.
    """
    lines = [f"Expression: {record.expression}"]
    if record.reading:
        lines.append(f"Reading: {record.reading}")
    if record.meanings:
        lines.append("Meanings: " + "; ".join(record.meanings))
    for label, value in (
        ("Part of speech", record.part_of_speech),
        ("Verb group", record.verb_group),
        ("Transitivity", record.transitivity),
    ):
        if value:
            lines.append(f"{label}: {value}")
    incomplete = [
        example
        for example in record.examples
        if example.needs_ai_annotations()
    ]
    if incomplete:
        lines.append(
            "\nExisting reviewed examples need annotations. Return each listed "
            "Japanese string exactly; do not replace it or add a different "
            "sentence. Fill only its empty English, furigana, and speech_level "
            "values:\n"
            + "\n".join(
                f"- {json.dumps(example.japanese, ensure_ascii=False)}"
                for example in incomplete
            )
        )
    if taught:
        lines.append("\n" + taught)
    if recent:
        lines.append(
            "\nSentences already written in this run — write something "
            "structurally different:\n" + "\n".join(f"- {item}" for item in recent)
        )
    return "\n".join(lines)


def ai_retry_prompt(
    record: VocabularyRecord,
    recent: Sequence[str] = (),
    taught: str = "",
) -> str:
    """A second request after every proposed example missed the headword.

    The accepted forms come from the same deterministic conjugation rules as
    the quality check. The retry therefore narrows the model's choice without
    weakening the check or guessing a new spelling.
    """
    forms = ", ".join(qc.target_forms(record.expression, record.verb_group))
    return (
        ai_prompt(record, recent, taught)
        + "\n\nYour first examples failed the exact headword spelling check.\n"
        + "Use one permitted written target form in each japanese field. "
        + "Keep the kana or kanji spelling shown here; do not substitute a "
        + "different spelling.\n"
        + f"Permitted written target forms: {forms}"
    )


AI_INSTRUCTIONS = """\
Write **two** example sentences for the word, and a usage note if there is
something worth saying.

If the record prompt lists existing reviewed examples that need annotations,
return those exact Japanese strings instead of writing new examples. Fill their
empty English, furigana, and speech_level values. Do not rewrite, replace, or
add a Japanese sentence in that case.

The first sentence is polite (〜ます / 〜です); set its speech_level to
"polite". The second is the same kind of everyday sentence in **casual** plain
form, as a friend would say it; set its speech_level to "casual". Write a different sentence
rather than the same one with the ending swapped — casual speech drops
particles, uses different sentence-final forms (〜の, 〜んだ, 〜よ, 〜ね), and a
mechanical de-politening teaches none of that.

Both must contain the word itself, conjugated if that reads more naturally, and
both must be simple enough for a beginner working through Genki-style grammar.
Use the exact spelling shown in Expression. You can conjugate it, but do not
replace a kana-only expression with kanji or replace its kanji with kana.
Give each sentence's furigana in Anki notation, with a space before every
bracketed group that follows kana. Do not fill in romaji — janki generates that
from the furigana and discards whatever you send.

Say nothing you are not sure of. An empty usage note is a fine answer; an
invented nuance is not."""


@dataclass(slots=True)
class AiOutcome:
    """What the AI pass decided for one record."""

    record: VocabularyRecord
    changes: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    rejected: list[str] = field(default_factory=list)
    unverified: list[str] = field(default_factory=list)


def _words_of(parse: Any) -> list[str]:
    """The spellings jpdb segmented a sentence into, in order.

    Empty for a parse that is missing or shaped unexpectedly: a repair with no
    boundaries to work from must leave the furigana alone, not guess at it.
    """
    tokens = getattr(parse, "tokens", None)
    vocabulary = getattr(parse, "vocabulary", None)
    if not isinstance(tokens, list) or not isinstance(vocabulary, list):
        return []
    words = []
    for token in tokens:
        if not isinstance(token, Mapping):
            continue
        index = token.get("vocabulary_index")
        if not isinstance(index, int) or not 0 <= index < len(vocabulary):
            continue
        entry = vocabulary[index]
        spelling = str(entry.get("spelling", "")) if isinstance(entry, Mapping) else ""
        if spelling:
            words.append(spelling)
    return words


def _fill_existing_example_annotations(
    stored: Sequence[ExampleSentence],
    generated: Sequence[ExampleSentence],
    unverified: Sequence[str],
) -> tuple[list[ExampleSentence], list[str]]:
    """Fill holes only when generated Japanese exactly matches stored text."""
    by_japanese: dict[str, ExampleSentence] = {}
    for example in generated:
        by_japanese.setdefault(example.japanese, example)
    flagged = set(unverified)
    landed_unverified: list[str] = []
    merged: list[ExampleSentence] = []
    for old in stored:
        incoming = by_japanese.get(old.japanese)
        if incoming is None:
            merged.append(old)
            continue
        wrote_furigana = not old.furigana and bool(incoming.furigana)
        updated = replace(
            old,
            furigana=old.furigana or incoming.furigana,
            english=old.english or incoming.english,
            register=(
                old.register
                if old.register in {"polite", "casual"}
                else incoming.register
            ),
        )
        if updated.furigana or not contains_kanji(updated.japanese):
            updated = qc.regenerate_example_romaji(updated)
        if wrote_furigana and incoming.japanese in flagged:
            landed_unverified.append(incoming.japanese)
        merged.append(updated)
    return merged, landed_unverified


def apply_ai_result(
    record: VocabularyRecord,
    parsed: Any,
    *,
    force_fields: Sequence[str] = (),
    parses: Mapping[str, Any] | None = None,
) -> AiOutcome:
    """Put a model's answer through the mechanical checks, then the fill rules.

    Three checks, in the order that matters (M4.1):

    * an example that does not contain the word is **rejected** — it may be a
      fine sentence, but it is not an example of this word;
    * an example whose furigana disagrees with jpdb's parse is **kept and
      flagged**, because the sentence may be right where the segmentation is
      not, and a human deciding that is better than janki throwing away good
      Japanese — but only if the examples land at all: when the fill rules keep
      the record's existing examples, ``unverified`` comes back empty, because
      there is no stored example for the flag to be about;
    * romaji is regenerated from the furigana, always, whatever arrived.

    ``parses`` maps a sentence to its jpdb ``ParseResult``. An absent one is
    not a pass: it means nobody checked, and the example is flagged the same
    way a mismatch is, because "unverified" is exactly what it is.
    """
    outcome = AiOutcome(record=record)
    kept: list[ExampleSentence] = []

    for item in getattr(parsed, "examples", []) or []:
        register = str(getattr(item, "speech_level", "") or "").strip().lower()
        example = ExampleSentence(
            japanese=str(getattr(item, "japanese", "") or "").strip(),
            furigana=str(getattr(item, "furigana", "") or "").strip(),
            english=str(getattr(item, "english", "") or "").strip(),
            # Anything the model does not label is polite: that is what the
            # instructions ask for first and what every example written before
            # the field existed actually is. Guessing "casual" would put a ます
            # sentence in a slot labelled casual, which teaches the opposite of
            # what the label says.
            register=register if register in ("polite", "casual") else "polite",
        )
        if not example.japanese:
            continue
        # Before anything reads the furigana. A model writes 週末[しゅうまつ]、何[なに]
        # without the separator space perhaps half the time, and Anki then draws
        # なに over 、何 while the comma vanishes from the reading the romaji and
        # the sentence audio are built from. The repair adds only the separator,
        # never a reading or a segmentation, so it is safe to run unattended;
        # spills that would need a guess are left for `spilled_furigana_groups`.
        if example.furigana:
            example = replace(
                example, furigana=qc.repair_spilled_punctuation(example.furigana)
            )
            # Then the spills that repair will not touch, using jpdb's own
            # segmentation of this very sentence — already fetched below to
            # verify the readings, and until now thrown away afterwards. The
            # boundary is a dictionary fact; the readings, grouping and
            # punctuation stay the writer's. No extra call, no model.
            example = replace(
                example,
                furigana=qc.repair_from_word_boundaries(
                    example.furigana, _words_of((parses or {}).get(example.japanese))
                ),
            )
        if not qc.example_contains_target(example, record.expression, record.verb_group):
            outcome.rejected.append(example.japanese)
            continue
        parse = (parses or {}).get(example.japanese)
        if parse is None or not qc.verify_example_furigana(example, parse):
            outcome.unverified.append(example.japanese)
        kept.append(qc.regenerate_example_romaji(example))

    unverified = list(outcome.unverified)
    proposals: dict[str, Any] = {
        "examples": kept,
        "usage_notes": str(getattr(parsed, "usage_notes", "") or "").strip(),
    }
    if record.examples and "examples" not in force_fields:
        merged_examples, landed_unverified = _fill_existing_example_annotations(
            record.examples, kept, unverified
        )
        updated = record
        changes: dict[str, tuple[Any, Any]] = {}
        if merged_examples != record.examples:
            changes["examples"] = (record.examples, merged_examples)
            updated = replace(record, examples=merged_examples)
        updated, other_changes = _apply(
            updated,
            proposals,
            [
                name
                for name in AI_FIELDS
                if name != "examples"
                and (name in force_fields or is_empty(getattr(record, name)))
            ],
        )
        changes.update(other_changes)
        outcome.unverified = landed_unverified
    else:
        writable = [
            name
            for name in AI_FIELDS
            if name in force_fields or is_empty(getattr(record, name))
        ]
        updated, changes = _apply(record, proposals, writable)
    if "examples" in changes:
        if outcome.unverified:
            updated = _flag_unverified(updated, outcome.unverified)
    else:
        # The examples were not written — the field was not writable, or the
        # answer matched what is already there. Nothing was flagged, so nothing
        # may be reported as flagged: a reviewer told to check a key would find
        # no key. The rejections still stand; those were the model's sentences
        # either way.
        outcome.unverified = []
    outcome.record = updated
    outcome.changes = changes
    return outcome


def _flag_unverified(record: VocabularyRecord, sentences: Sequence[str]) -> VocabularyRecord:
    """Record which examples nobody verified, by content fingerprint.

    A fingerprint of the sentence rather than its index, because an index stops
    meaning anything the moment a human deletes an example — and this key is
    read much later, by M5.3, deciding whether to speak a sentence whose
    segmentation may be wrong.
    """
    fingerprints = [short_fingerprint(sentence) for sentence in sentences]
    raw_fields = dict(record.source.raw_fields)
    existing = [
        item for item in raw_fields.get(UNVERIFIED_KEY, "").split(",") if item.strip()
    ]
    raw_fields[UNVERIFIED_KEY] = ",".join(dict.fromkeys(existing + fingerprints))
    return replace(record, source=replace(record.source, raw_fields=raw_fields))


@dataclass(slots=True)
class RecheckResult:
    """What a furigana re-check found."""

    records: list[VocabularyRecord] = field(default_factory=list)
    #: ``record id -> [sentence]`` cleared, because jpdb now agrees.
    cleared: dict[str, list[str]] = field(default_factory=dict)
    #: ``record id -> [(sentence, why)]`` still not vouched for.
    differing: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    #: Sentences jpdb could not parse at all — unverified, not fine.
    unparsed: list[str] = field(default_factory=list)
    #: ``record id -> [(sentence, why)]`` cleared because an adjudicator judged
    #: the writer's reading the ordinary one where jpdb disagreed.
    adjudicated: dict[str, list[tuple[str, str]]] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return bool(self.cleared)


ADJUDICATE_INSTRUCTIONS = """\
You settle disagreements about how a Japanese sentence is read.

You are given a sentence, the reading a dictionary parse produced, and the
reading its writer produced. Say which one a native speaker would use for this
sentence in ordinary modern Japanese.

Both are usually defensible; you are judging *ordinary usage*, not possibility.
日本語 is にほんご, not にっぽんご, even though both are attested.

Answer "writer" or "dictionary" — or "unsure", which is a real answer and the
right one whenever the two readings are both ordinary, the word is rare, or the
difference is a proper noun you cannot place. An unsure verdict leaves the
sentence flagged for a human, which costs a re-check; a confident wrong one puts
a reading nobody uses onto a card."""


@functools.cache
def adjudication_schema() -> Any:
    """The shape an adjudication must take."""
    from pydantic import BaseModel, Field

    class Adjudication(BaseModel):
        verdict: str = Field(
            description="'writer', 'dictionary', or 'unsure'."
        )
        why: str = Field(default="", description="One short clause.")

    return Adjudication


def adjudicate_reading(
    sentence: str,
    dictionary_reading: str,
    writer_reading: str,
    *,
    model: str,
    client: Any | None = None,
) -> tuple[str, str]:
    """``(verdict, why)`` for one disagreement. Never raises.

    A model is a poor *source* of readings — non-deterministic, and confidently
    wrong on exactly the rare words where a check matters — but a good judge of
    which of two given readings is the ordinary one, which is a much narrower
    question. So it never proposes a reading, only picks between two that
    already exist, and "unsure" is an answer it is told to give.

    Anything that goes wrong is ``("unsure", …)``: an adjudicator that cannot
    answer must leave the flag alone, not clear it.
    """
    blocks = claude_client.system_blocks(ADJUDICATE_INSTRUCTIONS)
    prompt = (
        f"Sentence: {sentence}\n"
        f"Dictionary reading: {dictionary_reading}\n"
        f"Writer reading: {writer_reading}\n"
        "Which is how this sentence is normally read?"
    )
    try:
        # `adjudication_schema()` is *inside* the guard: it imports pydantic,
        # which ships in the `[ai]` extra, so on a plain `pip install -e .` it
        # raised `ModuleNotFoundError` from the argument list — outside any
        # except, and not a `JankiError`, so it left the interpreter with a
        # traceback. `parse_call` can do the same: with no `ANTHROPIC_API_KEY`
        # the SDK raises a bare `TypeError`. Either way `recheck_furigana` never
        # returned, so `save_records_json` was never reached and every flag jpdb
        # had already cleared in that run — and the calls paid for — was thrown
        # away on the way out.
        call = claude_client.parse_call(
            model, blocks, prompt, adjudication_schema(), client, max_tokens=200
        )
    except Exception as exc:  # noqa: BLE001 — the docstring's contract
        return "unsure", f"the adjudicator could not be reached: {exc}"
    if call.parsed is None:
        return "unsure", f"the adjudicator did not answer ({call.stop_reason})"
    verdict = str(getattr(call.parsed, "verdict", "") or "").strip().lower()
    why = str(getattr(call.parsed, "why", "") or "").strip()
    return (verdict if verdict in {"writer", "dictionary", "unsure"} else "unsure"), why


def recheck_furigana(
    records: Sequence[VocabularyRecord],
    *,
    jpdb_client: jpdb.JpdbClient | None = None,
    ids: Sequence[str] | None = None,
    accept: bool = False,
    adjudicate_model: str = "",
    ai_client: Any | None = None,
) -> RecheckResult:
    """Re-ask jpdb whether each flagged example's furigana is right.

    The flag is written once, when an example is created, and read much later by
    ``janki audio`` deciding whether to speak a sentence. That makes it stale
    twice over: a human can fix the furigana by hand, and the *check itself* can
    improve — as it did when it stopped mistaking jpdb's per-character
    segmentation for disagreement, which had flagged 12 of 17 correct examples.
    Neither an AI pass nor an edit re-asks, so without this the only way to clear
    a flag was to rewrite the sentence and pay for it again.

    Only flagged examples are re-checked, and only ever cleared: an example
    nobody doubted is not put in doubt by a parse that happens to fail today.

    ``accept`` clears the flags of the named records on a *human's* authority
    instead of asking jpdb. It exists because jpdb can be wrong: its parse reads
    日本語 as にっぽんご, and the language is にほんご. Without it the choice was
    to write a reading nobody uses into a card or to leave a correct sentence
    unvoiced forever. It requires ``ids`` — accepting everything unread is not a
    judgment — and the ledger records ``kind="human"`` so the entry says who
    vouched.
    """
    if accept and not ids:
        raise EnrichError(
            "--accept clears a flag on your authority rather than jpdb's, so it "
            "needs the record ids you are vouching for. Accepting everything "
            "unread is not a judgment."
        )
    if not accept and jpdb_client is None:
        raise EnrichError("A furigana re-check needs a jpdb client or --accept.")
    result = RecheckResult(records=list(records))
    known = {record.id for record in result.records}
    wanted = set(ids) if ids else None
    if wanted is not None:
        # Named and not found is a typo, not an empty result. Every other
        # ids-taking pass here refuses the same way; without it
        # `--recheck-furigana word:わかる:わかる` (the id is word:分かる:わかる)
        # reported a clean negative and exited 0.
        missing = sorted(wanted - known)
        if missing:
            raise EnrichError(
                "No record has "
                + ("this id: " if len(missing) == 1 else "these ids: ")
                + ", ".join(missing)
            )
    for index, record in enumerate(result.records):
        if wanted is not None and record.id not in wanted:
            continue
        raw_fields = dict(record.source.raw_fields)
        flagged = {
            item.strip()
            for item in raw_fields.get(UNVERIFIED_KEY, "").split(",")
            if item.strip()
        }
        if not flagged:
            continue
        cleared: list[str] = []
        for example in record.examples:
            sentence = example.japanese.strip()
            fingerprint = short_fingerprint(sentence)
            if not sentence or fingerprint not in flagged:
                continue
            if accept:
                cleared.append(sentence)
                flagged.discard(fingerprint)
                continue
            try:
                parse = jpdb_client.parse(sentence)
            except JankiError:
                # Nothing means unverified rather than fine, the same way the
                # AI pass treats an absent client.
                result.unparsed.append(sentence)
                continue
            verdict = qc.verify_example_furigana(example, parse)
            if verdict:
                cleared.append(sentence)
                flagged.discard(fingerprint)
                continue
            why = "; ".join(verdict.differences[:2])
            if adjudicate_model:
                # jpdb and the writer disagree, and jpdb is not always right —
                # its parse reads 日本語 as にっぽんご. A model is a poor source
                # of readings but a good judge of which of two is ordinary, so
                # it breaks the tie and nothing else.
                call, reason = adjudicate_reading(
                    sentence,
                    verdict.expected,
                    example.furigana or example.japanese,
                    model=adjudicate_model,
                    client=ai_client,
                )
                if call == "writer":
                    cleared.append(sentence)
                    flagged.discard(fingerprint)
                    result.adjudicated.setdefault(record.id, []).append((sentence, reason))
                    continue
                why = f"{why} — adjudicator: {call}" + (f", {reason}" if reason else "")
            result.differing.setdefault(record.id, []).append((sentence, why))
        if not cleared:
            continue
        result.cleared[record.id] = cleared
        if flagged:
            raw_fields[UNVERIFIED_KEY] = ",".join(sorted(flagged))
        else:
            raw_fields.pop(UNVERIFIED_KEY, None)
        result.records[index] = replace(
            record, source=replace(record.source, raw_fields=raw_fields)
        )
    return result


@dataclass(slots=True)
class AiResult:
    """What an AI pass would write, and everything it refused along the way."""

    records: list[VocabularyRecord] = field(default_factory=list)
    changes: dict[str, dict[str, tuple[Any, Any]]] = field(default_factory=dict)
    rejected: dict[str, list[str]] = field(default_factory=dict)
    unverified: dict[str, list[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    no_changes: list[str] = field(default_factory=list)
    looked_up: int = 0

    @property
    def changed_fields(self) -> list[str]:
        return sorted({name for fields in self.changes.values() for name in fields})


def format_ai_no_changes(result: AiResult) -> list[str]:
    """Return the command report for valid answers that wrote no field."""
    if not result.no_changes:
        return []
    return [
        f"No changes for {len(result.no_changes)} of {result.looked_up} "
        "record(s) the AI pass visited:",
        *(f"  {record_id}" for record_id in result.no_changes),
    ]


def _verify_parses(
    jpdb_client: jpdb.JpdbClient | None, sentences: Sequence[str]
) -> dict[str, Any]:
    """jpdb's parse of each sentence, for the furigana check.

    A client that is not there yields nothing, and nothing means unverified
    rather than fine — see :func:`apply_ai_result`. A parse that fails for one
    sentence does the same rather than taking the whole record down: the
    example is still usable, it is just not vouched for.
    """
    parses: dict[str, Any] = {}
    if jpdb_client is None:
        return parses
    for sentence in sentences:
        try:
            parses[sentence] = jpdb_client.parse(sentence)
        except JankiError:
            continue
    return parses


def enrich_ai(
    records: Sequence[VocabularyRecord],
    *,
    model: str,
    style_guide: str,
    force_fields: Sequence[str] = (),
    ids: Sequence[str] | None = None,
    client: Any | None = None,
    parse_call: Any | None = None,
    call_options: Mapping[str, Any] | None = None,
    jpdb_client: jpdb.JpdbClient | None = None,
    taught: str = "",
) -> AiResult:
    """Write examples and usage notes for the records that lack them.

    One initial call per record keeps failures local to that record. If every
    proposed example fails only the exact-headword check and the record still
    has no examples, one limited retry supplies the deterministic forms that
    the check accepts. Refusals and truncated answers are never salvaged.
    """
    result = AiResult(records=list(records))
    positions = {record.id: index for index, record in enumerate(result.records)}
    targets = ai_targets(result.records, ids)
    blocks = claude_client.system_blocks(style_guide, AI_INSTRUCTIONS)
    caller = parse_call or claude_client.parse_call
    options = dict(call_options or {})
    recent: list[str] = []

    for record in targets:
        result.looked_up += 1
        warning_start = len(result.warnings)
        call = caller(
            model,
            blocks,
            ai_prompt(record, recent[-VARIETY_EXAMPLES:], taught),
            ai_schema(),
            client,
            **options,
        )
        absorb_ai_call(
            result,
            record,
            call,
            model=model,
            positions=positions,
            recent=recent,
            force_fields=force_fields,
            jpdb_client=jpdb_client,
        )
        position = positions[record.id]
        current = result.records[position]
        if record.examples or record.id not in result.rejected or current.examples:
            continue

        first_warning_end = len(result.warnings)
        first_rejected = result.rejected.pop(record.id)
        retry = caller(
            model,
            blocks,
            ai_retry_prompt(current, recent[-VARIETY_EXAMPLES:], taught),
            ai_schema(),
            client,
            **options,
        )
        absorb_ai_call(
            result,
            current,
            retry,
            model=model,
            positions=positions,
            recent=recent,
            force_fields=force_fields,
            jpdb_client=jpdb_client,
        )
        retry_wrote_examples = bool(result.records[position].examples)
        retry_reported_rejection = record.id in result.rejected
        if retry_wrote_examples or retry_reported_rejection:
            del result.warnings[warning_start:first_warning_end]
        else:
            result.rejected[record.id] = first_rejected
        if retry_wrote_examples:
            result.no_changes = [
                item for item in result.no_changes if item != record.id
            ]
    return result


def absorb_ai_call(
    result: AiResult,
    record: VocabularyRecord,
    call: claude_client.CallResult,
    *,
    model: str,
    positions: Mapping[str, int],
    recent: list[str],
    force_fields: Sequence[str] = (),
    jpdb_client: jpdb.JpdbClient | None = None,
) -> None:
    """Put one model answer through the checks and fold it into ``result``.

    Public and separate from :func:`enrich_ai` so the batch path (M4.4) runs
    *this* code rather than a second copy of it. A batched answer is the same
    answer, arriving later and cheaper, and the day the two paths' QC diverges
    is the day one of them starts writing sentences the other would have caught.
    """
    parsed, stop_reason, refusal = call
    if stop_reason == "refusal":
        detail = f" ({refusal.category})" if refusal is not None else ""
        result.warnings.append(
            f"{record.id}: {model} declined to write about "
            f"{record.expression}{detail}; nothing written."
        )
        return
    if parsed is None:
        result.warnings.append(
            f"{record.id}: {model} returned nothing usable for "
            f"{record.expression} (stop reason: {stop_reason}); nothing written."
        )
        return

    sentences = [
        text
        for item in getattr(parsed, "examples", []) or []
        if (text := str(getattr(item, "japanese", "") or "").strip())
    ]
    outcome = apply_ai_result(
        record,
        parsed,
        force_fields=force_fields,
        parses=_verify_parses(jpdb_client, sentences),
    )
    if outcome.rejected:
        result.rejected[record.id] = outcome.rejected
        result.warnings.append(
            f"{record.id}: {len(outcome.rejected)} example(s) did not contain "
            f"{record.expression} and were rejected."
        )
    if outcome.unverified:
        result.unverified[record.id] = outcome.unverified
        result.warnings.append(
            f"{record.id}: {len(outcome.unverified)} example(s) have furigana "
            "jpdb did not confirm; kept and flagged for review."
        )
    if outcome.changes:
        result.records[positions[record.id]] = outcome.record
        result.changes[record.id] = {
            **result.changes.get(record.id, {}),
            **outcome.changes,
        }
        recent.extend(
            example.japanese for example in outcome.record.examples if example.japanese
        )
    elif record.id not in result.changes and record.id not in result.no_changes:
        result.no_changes.append(record.id)


# --- polishing meanings ------------------------------------------------------
#
# The one pass that rewrites a field that is already full. Every other pass in
# this module fills holes, which is safe because a hole has no curation in it to
# lose; this one proposes replacing English somebody may have typed, so it is a
# separate flag, it never runs as a side effect of anything else, and the CLI
# confirms it one record at a time.


#: The only field this pass writes. A tuple so it reads like its siblings and
#: so the ledger's ``fields`` list is built the same way.
POLISH_FIELDS: tuple[str, ...] = ("meanings",)


@functools.cache
def polish_schema() -> Any:
    """The Pydantic model a polish response must match. Cached, like the others."""
    from pydantic import BaseModel, Field

    class PolishedMeanings(BaseModel):
        meanings: list[str] = Field(
            default_factory=list,
            description=(
                "The English glosses for this word, best first. Empty if the "
                "existing ones are already right."
            ),
        )

    return PolishedMeanings


POLISH_INSTRUCTIONS = """\
Improve the English glosses for the word you are given.

A gloss list is what a learner reads on the back of a card, so it should be the
few senses that word actually carries, ordered with the most common first, in
the plainest English that is still accurate. Verbs read as "to ..." — "to
speak", not "speaking" or "speech". Drop a gloss that is a restatement of the
one above it, a part-of-speech label, or a dictionary's hedge.

The record's own examples show which sense it was collected for. Keep that sense
first, and do not add a sense the word has only in a register this record is not
about.

Return an empty list if the existing glosses are already right. That is a real
answer and a common one; a rewrite that only moves words around costs a reviewer
their attention for nothing."""


def polish_targets(
    records: Sequence[VocabularyRecord], ids: Sequence[str] | None = None
) -> list[VocabularyRecord]:
    """The records a polish pass would look at.

    No content rule, unlike the other passes: ``meanings`` is never empty, so
    there is no hole to test for and "which records need this" is a judgment
    only the person running it can make. Without ``--ids`` that is every
    record, and the caller says so out loud before spending anything.
    """
    if ids is None:
        return list(records)
    known = {record.id: record for record in records}
    missing = [item for item in ids if item not in known]
    if missing:
        raise EnrichError(
            f"No record with id {missing[0]!r}. Ids come from vocabulary.json; "
            "'janki status' lists them."
        )
    return [known[item] for item in dict.fromkeys(ids)]


def polish_prompt(record: VocabularyRecord) -> str:
    """The user turn for one record: the word, its current glosses, its examples.

    The examples are the point. 「先生に聞く」 and 「音楽を聞く」 are the same
    verb with two glosses a learner needs kept apart, and the sentences the
    record was collected with are the only evidence janki has for which one it
    means.
    """
    lines = [f"Word: {record.expression}"]
    if record.reading:
        lines.append(f"Reading: {record.reading}")
    lines.append("Current meanings: " + "; ".join(record.meanings or ["(none)"]))
    for label, value in (
        ("Part of speech", record.part_of_speech),
        ("Verb group", record.verb_group),
        ("Transitivity", record.transitivity),
    ):
        if value:
            lines.append(f"{label}: {value}")
    sentences = [item.japanese for item in record.examples if item.japanese]
    if sentences:
        lines.append(
            "\nThe sentences this record was collected with — they say which "
            "sense it means:\n" + "\n".join(f"- {item}" for item in sentences)
        )
    if record.usage_notes:
        lines.append(f"\nUsage notes on file: {record.usage_notes}")
    return "\n".join(lines)


def polish_prompt_fingerprint(record: VocabularyRecord) -> str:
    """Identify the exact record content a polish answer was requested for."""
    return hashlib.sha256(polish_prompt(record).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PolishOutcome:
    """One record's turn through the pass: what was proposed, or why nothing was."""

    record: VocabularyRecord
    proposed: VocabularyRecord | None = None
    changes: Mapping[str, tuple[Any, Any]] = field(default_factory=dict)
    warning: str = ""


def apply_polish_result(record: VocabularyRecord, parsed: Any) -> PolishOutcome:
    """Turn a model's gloss list into a proposal, or into nothing.

    Nothing is the common case and is not an error: the instructions ask for an
    empty list when the existing glosses are already right, and a list that
    comes back identical to the one on file is the same answer spelled longer.

    A pass that emptied ``meanings`` would leave a card with a Japanese side and
    no English one, so an answer that reduces to nothing is refused rather than
    written — the record keeps what it has and the caller is told.
    """
    proposed = [
        text
        for item in getattr(parsed, "meanings", []) or []
        if (text := str(item or "").strip())
    ]
    proposed = list(dict.fromkeys(proposed))
    if not proposed:
        return PolishOutcome(record=record)
    if proposed == list(record.meanings):
        return PolishOutcome(record=record)
    return PolishOutcome(
        record=record,
        proposed=replace(record, meanings=proposed),
        changes={"meanings": (list(record.meanings), proposed)},
    )


def polish_call_outcome(
    record: VocabularyRecord, call: claude_client.CallResult, *, model: str
) -> PolishOutcome:
    """Turn either a live or batched model call into one polish outcome."""
    parsed, stop_reason, refusal = call
    if stop_reason == "refusal":
        detail = f" ({refusal.category})" if refusal is not None else ""
        return PolishOutcome(
            record=record,
            warning=(
                f"{record.id}: {model} declined to gloss "
                f"{record.expression}{detail}; left alone."
            ),
        )
    if parsed is None:
        return PolishOutcome(
            record=record,
            warning=(
                f"{record.id}: {model} returned nothing usable for "
                f"{record.expression} (stop reason: {stop_reason}); left alone."
            ),
        )
    return apply_polish_result(record, parsed)


def polish_meanings(
    records: Sequence[VocabularyRecord],
    *,
    model: str,
    style_guide: str,
    ids: Sequence[str] | None = None,
    client: Any | None = None,
) -> Iterator[PolishOutcome]:
    """Propose better glosses, one record at a time, lazily.

    A generator rather than a result object, because this pass is confirmed per
    record and the confirmation is what decides whether the next call is worth
    making. Driving it from the CLI's loop means declining the first proposal
    and walking away costs one call, not one per record in the collection.
    """
    blocks = claude_client.system_blocks(style_guide, POLISH_INSTRUCTIONS)
    for record in polish_targets(records, ids):
        try:
            call = claude_client.parse_call(
                model, blocks, polish_prompt(record), polish_schema(), client
            )
        except claude_client.ClaudeRequestError as exc:
            yield PolishOutcome(
                record=record,
                warning=f"{record.id}: {exc}; left alone.",
            )
            continue
        yield polish_call_outcome(
            record,
            call,
            model=model,
        )


# --- the AI pass, batched ----------------------------------------------------
#
# The Message Batches API is the same request at half price, answered within a
# day instead of within seconds. That trade is worth it for backfilling a
# thousand-word mining deck and pointless for the weekly ten, so it is two
# explicit subcommands rather than a heuristic: submit, walk away, fetch later.
#
# What it cannot have is the variety pressure the live pass uses, because every
# request is built before any answer exists. That is a real difference in output
# and it is why the live path stays the default: a batch gets a style guide and
# the word, and nothing about what it has already written.


#: What a batch's ``custom_id`` is derived from. The API wants a short ASCII
#: identifier and a record id is neither — ``word:話す:はなす`` is the wrong
#: alphabet and an unbounded length — so results come back keyed by a
#: fingerprint of the id, which the fetching side recomputes from the ledger's
#: pending list rather than storing a second copy of.
def batch_custom_id(record_id: str) -> str:
    return f"r{short_fingerprint(record_id)}"


def batch_key_map(record_ids: Sequence[str]) -> dict[str, str]:
    """``custom_id -> record id`` for a set of records.

    A collision here would attach one word's examples to another word, so it is
    refused rather than resolved. It takes a birthday collision across 48 bits
    to happen at all, but "unlikely" is not the standard for silently writing a
    sentence about 話す onto 聞く.
    """
    keys: dict[str, str] = {}
    for record_id in record_ids:
        key = batch_custom_id(record_id)
        if key in keys and keys[key] != record_id:
            raise EnrichError(
                f"Two records share the batch key {key!r}: {keys[key]!r} and "
                f"{record_id!r}. Submitting would risk writing one word's "
                "examples onto the other, so nothing was sent."
            )
        keys[key] = record_id
    return keys


def batch_requests(
    records: Sequence[VocabularyRecord],
    *,
    model: str,
    style_guide: str,
    ids: Sequence[str] | None = None,
    taught: str = "",
) -> tuple[list[dict[str, Any]], list[str]]:
    """The batch entries for every record that needs enriching, and their ids.

    A one-hour cache TTL rather than the default five minutes: the style guide
    leads every request, and a batch's requests are read over a span that a
    five-minute window would not survive.

    ``taught`` carries the reviewed patterns, exactly as the immediate path
    does. Without it the two paths wrote different sentences for the same
    record at different prices — and batch is the one used for bulk, so most of
    a collection would have got the unsteered version.
    """
    targets = ai_targets(records, ids)
    blocks = claude_client.system_blocks(style_guide, AI_INSTRUCTIONS, cache_ttl="1h")
    record_ids = [record.id for record in targets]
    batch_key_map(record_ids)
    requests = [
        claude_client.batch_request(
            batch_custom_id(record.id),
            model,
            blocks,
            ai_prompt(record, taught=taught),
            ai_schema(),
        )
        for record in targets
    ]
    return requests, record_ids


def polish_batch_requests(
    records: Sequence[VocabularyRecord],
    *,
    model: str,
    style_guide: str,
    ids: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str], dict[str, str]]:
    """Anthropic batch entries for a meaning-polish pass."""
    targets = polish_targets(records, ids)
    blocks = claude_client.system_blocks(
        style_guide, POLISH_INSTRUCTIONS, cache_ttl="1h"
    )
    record_ids = [record.id for record in targets]
    prompt_fingerprints = {
        record.id: polish_prompt_fingerprint(record) for record in targets
    }
    batch_key_map(record_ids)
    requests = [
        claude_client.batch_request(
            batch_custom_id(record.id),
            model,
            blocks,
            polish_prompt(record),
            polish_schema(),
        )
        for record in targets
    ]
    return requests, record_ids, prompt_fingerprints


@dataclass(slots=True)
class PolishBatchApplyResult:
    """Reviewable polish proposals and every batch row not represented by one."""

    proposals: dict[str, PolishOutcome] = field(default_factory=dict)
    unchanged: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    invalid: dict[str, str] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)
    settled: list[str] = field(default_factory=list)
    looked_up: int = 0


def apply_polish_batch_results(
    records: Sequence[VocabularyRecord],
    entries: Iterable[claude_client.BatchEntry],
    pending_ids: Sequence[str],
    *,
    model: str,
    prompt_fingerprints: Mapping[str, str],
    only: Sequence[str] = (),
) -> PolishBatchApplyResult:
    """Map a completed polish batch back to records without accepting proposals."""
    result = PolishBatchApplyResult()
    by_id = {record.id: record for record in records}
    keys = batch_key_map(pending_ids)
    candidates = {str(item) for item in only}
    seen: set[str] = set()

    for entry in entries:
        record_id = keys.get(entry.custom_id)
        if record_id is None:
            result.warnings.append(
                f"The batch returned a result keyed {entry.custom_id!r}, which "
                "belongs to no record this batch was submitted for; it was "
                "ignored rather than guessed at."
            )
            continue
        seen.add(record_id)
        if candidates and record_id not in candidates:
            result.settled.append(record_id)
            continue
        record = by_id.get(record_id)
        if record is None:
            result.missing.append(record_id)
            continue
        if prompt_fingerprints.get(record_id) != polish_prompt_fingerprint(record):
            result.stale.append(record_id)
            continue
        if entry.result is None:
            if entry.outcome == "invalid":
                result.invalid[record_id] = entry.detail or "schema validation failed"
            else:
                detail = f": {entry.detail}" if entry.detail else ""
                result.failed[record_id] = f"the batch reported {entry.outcome}{detail}"
            continue
        result.looked_up += 1
        outcome = polish_call_outcome(record, entry.result, model=model)
        if outcome.warning:
            result.warnings.append(outcome.warning)
        elif outcome.proposed is None:
            result.unchanged.append(record_id)
        else:
            result.proposals[record_id] = outcome

    for record_id in pending_ids:
        if record_id in seen or (candidates and record_id not in candidates):
            continue
        if record_id in by_id:
            result.failed[record_id] = "the batch returned no result for it"
        else:
            result.missing.append(record_id)
    return result


@dataclass(slots=True)
class BatchApplyResult:
    """What a fetched batch wrote, plus every record it could not account for."""

    result: AiResult = field(default_factory=AiResult)
    failed: dict[str, str] = field(default_factory=dict)
    #: Rows janki could not parse. Separate from ``failed`` because the answer
    #: still exists — see :func:`apply_batch_results`.
    invalid: dict[str, str] = field(default_factory=dict)
    #: Rows an earlier fetch of this batch already settled, skipped this time.
    settled: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


def apply_batch_results(
    records: Sequence[VocabularyRecord],
    entries: Iterable[claude_client.BatchEntry],
    pending_ids: Sequence[str],
    *,
    model: str,
    force_fields: Sequence[str] = (),
    only: Sequence[str] = (),
    jpdb_client: jpdb.JpdbClient | None = None,
) -> BatchApplyResult:
    """Fold a finished batch into the records, through the live path's checks.

    Every answer goes through :func:`absorb_ai_call`, which is the same function
    the synchronous pass uses — the saving is in how the request was sent, not
    in what is done with the reply.

    Four ways a record can come back with nothing, none of them silent, and one
    of them different in kind from the rest:

    * the batch reported it **errored, expired or was canceled** — terminal, a
      re-fetch returns the same row;
    * the batch **never mentioned it**, which will not change either;
    * its id **no longer names a record**, because the collection moved while
      the batch was out — deliberate curation, on this side;
    * the row succeeded and its answer did **not validate** (``invalid``).

    ``only``, when given, narrows the rows this pass will consider — a held
    batch's second fetch passes the ids whose answers did not parse, and every
    other row is settled by definition, whatever route it took. That matters
    because a held batch is the only one that can be fetched twice, and the gap
    between the two is exactly when a human corrects a sentence the first one
    wrote; re-applying the whole result set under the submitted ``force_fields``
    would replace that correction with the model's original text.

    Only the last of the four is recoverable, which is why it gets its own list
    rather than joining ``failed``. That answer is complete and paid for and lives on
    Anthropic's side for weeks; janki's schema is the only thing rejecting it,
    and a schema can be fixed. A caller that treats it as dead — by clearing
    the batch id — makes it unreachable for a reason that was never the API's.
    """
    outcome = BatchApplyResult(result=AiResult(records=list(records)))
    positions = {record.id: index for index, record in enumerate(outcome.result.records)}
    keys = batch_key_map(pending_ids)
    candidates = {str(item) for item in only}
    recent: list[str] = []
    seen: set[str] = set()

    for entry in entries:
        record_id = keys.get(entry.custom_id)
        if record_id is None:
            outcome.result.warnings.append(
                f"The batch returned a result keyed {entry.custom_id!r}, which "
                "belongs to no record this batch was submitted for; it was "
                "ignored rather than guessed at."
            )
            continue
        seen.add(record_id)
        if candidates and record_id not in candidates:
            outcome.settled.append(record_id)
            continue
        if entry.result is None:
            if entry.outcome != "invalid":
                # Terminal, and the API's reason is the only signal that the API
                # rather than curation is why this word went unenriched. Worth
                # printing whether or not the record is still here — and if it
                # is not, that is worth printing too. The two facts are not in
                # conflict, so the row carries both rather than choosing: it
                # stays out of ``missing`` (nothing there for a later fetch to
                # get) while still saying the collection moved under it, which
                # is otherwise reported by nothing at all.
                detail = f": {entry.detail}" if entry.detail else ""
                where = (
                    "; left untouched"
                    if record_id in positions
                    else "; and the record is no longer in the collection"
                )
                outcome.failed[record_id] = f"{entry.outcome}{detail}{where}"
            elif record_id in positions:
                outcome.invalid[record_id] = entry.detail or "the answer did not parse"
            else:
                # Unreadable *and* deleted. Only this combination goes through
                # the presence test, because only this one would otherwise hold
                # the batch id forever waiting on a word nobody wants.
                outcome.missing.append(record_id)
            continue
        index = positions.get(record_id)
        if index is None:
            outcome.missing.append(record_id)
            continue
        outcome.result.looked_up += 1
        absorb_ai_call(
            outcome.result,
            outcome.result.records[index],
            entry.result,
            model=model,
            positions=positions,
            recent=recent,
            force_fields=force_fields,
            jpdb_client=jpdb_client,
        )

    for record_id in pending_ids:
        if record_id in seen or (candidates and record_id not in candidates):
            continue
        if record_id in positions:
            outcome.failed[record_id] = "the batch returned no result for it"
        else:
            outcome.missing.append(record_id)
    return outcome
