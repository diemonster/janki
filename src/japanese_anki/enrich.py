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
2. Forced with janki's own reading (``jpdb.forced_furigana_span``) whenever the
   first answer disagrees. The returned entry is trusted only when its declared
   primary or alternate readings confirm the stored reading (or the existing
   exact ``Xする`` allowance does). The request hint alone is not evidence.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, is_dataclass, replace
from dataclasses import fields as dc_fields
from typing import Any, Protocol

from japanese_anki import ai_schema as ai_schema_module
from japanese_anki import claude_client, codex_client, jpdb, pitch, prompts, qc
from japanese_anki.conjugation import conjugate
from japanese_anki.errors import JankiError
from japanese_anki.identifiers import (
    contains_kanji,
    normalize_identity_part,
    short_fingerprint,
)
from japanese_anki.io import is_empty
from japanese_anki.ledger import Ledger
from japanese_anki.models import (
    ExampleSentence,
    VocabularyRecord,
    clear_provisional,
    example_accepted,
    provisional_entries,
    split_provisional,
)
from japanese_anki.romaji import kana_to_romaji
from japanese_anki.staging import NON_READING_HOLDS, annotate, annotations

__all__ = [
    "DICTIONARY_MAY_NOT_SETTLE",
    "ENRICHABLE_FIELDS",
    "DictionaryLookup",
    "DictionaryReadings",
    "EnrichError",
    "EnrichResult",
    "SuggestionResult",
    "AI_FIELDS",
    "STAGING_THRESHOLD",
    "AiResult",
    "BatchApplyResult",
    "BatchPlan",
    "ai_prompt",
    "ai_input_fingerprint",
    "ai_request_fingerprint",
    "ai_schema",
    "ai_targets",
    "absorb_ai_call",
    "apply_ai_result",
    "apply_batch_results",
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
    "suggest_readings",
]


class EnrichError(JankiError):
    pass


#: Fields a dictionary may never settle, however exactly the entry matches.
#:
#: ``ENRICHABLE_FIELDS`` below says what a jpdb pass may write into a *hole*.
#: This says what it may not overwrite when the field is already full but
#: provisional — a distinct question, and the one that got answered wrong.
#:
#: A card's meanings are the sense **this source taught**, not everything the
#: word can mean, so a dictionary has no vote on them. The failure that put
#: this here: a medical sheet taught おたふく = "mumps", and reconciliation
#: replaced it with jpdb's entry for the identically-spelled お多福 — "homely
#: woman (esp. one with a small low nose, high flat forehead, and bulging
#: cheeks)" — on a card whose own example sentence reads "My child came down
#: with the mumps." The spelling guard below cannot catch that: both spell
#: おたふく, and only the sense differs. Sixty-three of that sheet's eighty-five
#: cards lost their taught meaning this way, and 135 across the collection.
#:
#: A provisional meaning is settled by a person, or by ``enrich --ai``, which
#: reads the record's own examples. Both can tell homographs apart. A gloss
#: list keyed on spelling cannot.
DICTIONARY_MAY_NOT_SETTLE: frozenset[str] = frozenset({"meanings"})


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
    "transitivity",
    "conjugations",
    "pitch_accent",
    "frequency_rank",
)


#: Fields the AI pass may write. Disjoint from :data:`ENRICHABLE_FIELDS` on
#: purpose — a dictionary pass and a writing pass fill different holes, and a
#: record needing one does not need the other.
AI_FIELDS: tuple[str, ...] = ("meanings", "examples", "usage_notes")

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
    #: ``record id -> [field]`` whose provisional mark was cleared without a
    #: value change (a stale binding, or dictionary confirmation). A separate
    #: channel from ``changes`` because the caller's save gate reads ``changes``
    #: — and a cleared mark that is never persisted comes back to make the same
    #: network calls and print the same warning on every future run.
    cleared: dict[str, list[str]] = field(default_factory=dict)
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


class DictionaryLookup(Protocol):
    """The only two jpdb methods reachable on the promote and enrich path.

    `jpdb.JpdbClient` exposes five; `_parse` calls `parse`, `_readings_for`
    calls `lookup_vocabulary`, and nothing else on this route touches the
    client. Naming that pair as the annotation is what lets a study finish
    hand these callers a recorded fact book instead of a live connection —
    every fact the owner reviewed answers from what was fetched once, before
    the preview, so no later dictionary refresh can change or refuse what was
    approved.

    The defaults are the real module constants rather than ``...`` on purpose:
    a replay client can only key a request on its *effective* arguments if the
    defaults are written down. ``fields`` stays positional-or-keyword because
    `_readings_for` passes it positionally. Nothing isinstance-checks this, so
    it is not ``@runtime_checkable``; `JpdbClient` satisfies it structurally
    with no change at all.
    """

    def parse(
        self,
        text: str,
        *,
        token_fields: Sequence[str] = jpdb.DEFAULT_TOKEN_FIELDS,
        vocabulary_fields: Sequence[str] = jpdb.DEFAULT_VOCABULARY_FIELDS,
        forced_furigana: Sequence[Sequence[Any]] | None = None,
        encoding: str = jpdb.DEFAULT_ENCODING,
    ) -> jpdb.ParseResult: ...

    def lookup_vocabulary(
        self,
        pairs: Iterable[Any],
        fields: Sequence[str] = jpdb.DEFAULT_LOOKUP_FIELDS,
        *,
        batch_size: int = jpdb.DEFAULT_BATCH_SIZE,
    ) -> list[dict[str, Any]]: ...


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
    client: DictionaryLookup, expression: str, reading: str = ""
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """``/parse`` one expression, optionally forcing ``reading``."""
    forced = jpdb.forced_furigana_span(expression, reading) if reading else None
    return _dictionary_entry(client.parse(expression, forced_furigana=forced), expression)


def _readings_for(client: DictionaryLookup, entry: Mapping[str, Any]) -> set[str]:
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
    client: DictionaryLookup, expression: str, reading: str = ""
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
        # The one field with no filler since the hand-paste era: set at import
        # and backfilled by nothing, while jpdb's own vt/vi codes carried the
        # dictionary's answer the whole time. `pos_to_transitivity` returns ""
        # for a word tagged both — する is ["aux-v", "vi", "suf", "vt", "vs"] —
        # so an ambiguous word still shows nothing rather than a coin flip.
        #
        # Gated on the part of speech this pass is about to write, not on the
        # verb class. JMdict tags suru-nouns with vt/vi as well — 仕事 is
        # ["n", "vs", "vi"] — and `pos_to_verb_group` calls that `suru`, so a
        # verb-class test lets a noun through. Transitivity is a property of a
        # verb, and writing it on a record whose own part of speech says "noun"
        # puts a contradiction on a card that shows the field unconditionally
        # and has no validation rule for it.
        # The *derived* label, never the record's own. A stored part of speech
        # is uncontrolled text — copied verbatim from a source column, written
        # by extraction, or typed by hand — so gating on it let "pronoun",
        # "proper noun" and "noun, suru-verb" past a test for exactly "noun"
        # and wrote the contradiction this rule exists to prevent. jpdb's
        # answer is the only input here that cannot be influenced by the file.
        "transitivity": jpdb.transitivity_for(codes, part_of_speech),
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
    """Split jpdb patterns into compatible and unusable values.

    "Compatible" means :func:`pitch.to_aquestalk` renders it, which is a
    stricter test than "the right length": a pattern that fits the reading can
    still name a long vowel with no vowel to repeat.

    A *split*, not a rejection. jpdb often offers several patterns, and the
    usable ones are written while the caller warns about the rest — so an
    unusable pattern does not by itself cost a record its accent. It does when
    every offered pattern is unusable **and** the record had none stored: this
    pass never blanks an accent it did not write, and a record that already has
    one keeps it.
    """
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


def _unusable_detail(reading: str, patterns: Sequence[str]) -> str:
    """Every unusable pattern with *its own* reason.

    One reason for a whole list was the same defect one message for both
    refusal shapes was: `['LHHHHH', 'LHH']` against かんーぱい is one long-vowel
    problem and one length problem, and naming only the first sends a curator
    to check the wrong thing for the second.
    """
    return ", ".join(
        f"{pattern} ({_why_unusable(reading, pattern)})" for pattern in patterns
    ) + f" cannot be used for reading {reading}"


def _what_became_of_the_rest(
    valid: Sequence[str], written: bool, stored: Sequence[str]
) -> str:
    """What the record actually ended up with, in the four ways it can end.

    ``written`` is "``pitch_accent`` appears in the diff", and `_apply` records
    no change when the new value equals the old — so "not written" covers two
    different things and only one of them is about jpdb declining to overwrite.
    A `--force-fields pitch_accent` re-run that lands the same value would
    otherwise be told the record "already carries an accent and jpdb does not
    overwrite one", which is the wrong reason for a field that *was* forced.
    """
    if not valid:
        return "jpdb offered no usable pattern, so none was written"
    if written:
        return f"the {len(valid)} that can was written" if len(valid) == 1 else (
            f"the {len(valid)} that can were written"
        )
    if list(stored) == list(valid):
        return "the rest are usable and are what the record already carried"
    return (
        "the rest are usable but were not written — this record already "
        "carries an accent, and jpdb does not overwrite one"
    )


def _why_unusable(reading: str, pattern: str) -> str:
    """The reason this pattern cannot be spoken, in the words the check gives.

    One message covered both shapes and said "do not fit reading" for both.
    That is right for a length mismatch and wrong for the other: a pattern can
    fit the reading exactly and still be unrenderable, because a long vowel in
    it has no vowel to repeat. Telling a curator the length is wrong when it is
    not sends them to check the one thing that is already correct.
    """
    try:
        pitch.to_aquestalk(reading, pattern)
    except pitch.PitchError as exc:
        detail = str(exc).rstrip(".")
        # Anchored markers, not a bare ": ". Both messages interpolate the
        # pattern and the reading ahead of their real separator, so splitting
        # on the first colon-space cut inside the data: `"L: H"` defeated the
        # bare marker.
        #
        # `"': "` anchors on the closing quote of the *reading*'s repr, which
        # is what the length message interpolates. A reading containing `': `
        # would still mis-split — it is a stored, hand-editable field, so that
        # is not impossible, only unlikely for kana. The failure is cosmetic
        # either way: the caller prints the pattern immediately before this
        # reason, so a mangled explanation sits beside the value it is about,
        # and a message matching neither marker is passed through whole.
        for marker in (" for VOICEVOX: ", "': "):
            head, found, tail = detail.partition(marker)
            if found:
                detail = tail
                break
        # The long-vowel refusal carries its own trailing advice about what the
        # clip will sound like; the caller is about to say that itself.
        return detail.split(". The word is voiced")[0]
    return "it renders — nothing is wrong with it"


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
        # One marker parse for both halves. A marker whose value binding broke
        # is a human edit made after extraction: the mark is now a lie that
        # would authorize overwriting that edit, so it is cleared before
        # anything reads it — and recorded in ``cleared`` so the caller
        # persists the clear rather than repeating it forever.
        active, stale = split_provisional(record)
        if stale:
            record = clear_provisional(record, stale)
            result.records[by_id[record_id]] = record
            result.cleared.setdefault(record_id, []).extend(stale)
            result.warnings.append(
                f"{record_id}: {', '.join(stale)} edited since extraction; the "
                "provisional mark was cleared and the field is kept as curated"
            )
        # Provisional fields are revisited even though they are full: a model
        # claim holds the seat only until dictionary evidence arrives. Without
        # a reading, though, no entry can be confirmed to be this exact word,
        # so the claim is held rather than reconciled against a guess.
        reconcile = [
            name
            for name in active
            if name not in force_fields
            and not is_empty(getattr(record, name))
            and name not in DICTIONARY_MAY_NOT_SETTLE
        ]
        if reconcile and not record.reading:
            result.warnings.append(
                f"{record_id}: {', '.join(reconcile)} stay provisional — the "
                "record has no reading, so no dictionary entry can be confirmed "
                "as this exact word"
            )
            reconcile = []
        # Kept as two lists to their one consumer (`_apply` below): merging
        # them here forced the wrong-spelling branch to un-merge, and the two
        # retractions travelling together was exactly the state a future edit
        # would break.
        wanted = _wanted(record, force_fields)
        if not (wanted or reconcile) or not record.expression:
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
        if record.reading and record.reading != jpdb_reading:
            # A reading can select a separate jpdb vocabulary identity even
            # when both entries have exactly the same spelling. 分/ぶん and
            # 分/ふん, for example, need not be connected through alt_sids.
            # Always make one pinned parse; then inspect its answer rather than
            # treating the forced request itself as confirmation.
            primary_known = (
                _readings_for(client, entry)
                if entry_spelling == record.expression
                else None
            )
            pinned = _parse(client, record.expression, record.reading)
            if pinned is None:
                supports_suru_suffix = _supports_suru_suffix(
                    record.expression, record.reading, entry
                )
                known = (
                    primary_known
                    if primary_known is not None
                    else _readings_for(client, entry)
                )
                if not supports_suru_suffix and record.reading in known:
                    result.warnings.append(
                        f"{record_id}: jpdb did not parse {record.expression} as one "
                        f"word when given the reading {record.reading}; nothing was "
                        "written"
                    )
                    continue
                if not supports_suru_suffix:
                    listed = ", ".join(sorted(known)) or jpdb_reading or "none"
                    result.warnings.append(
                        f"{record_id}: janki reads {record.expression} as "
                        f"{record.reading}, jpdb lists {listed}. Nothing was written — "
                        "the reading is part of the record ID and is never auto-fixed."
                    )
                    continue
            else:
                pinned_token, pinned_entry = pinned
                pinned_reading = str(pinned_entry.get("reading") or "").strip()
                supports_suru_suffix = _supports_suru_suffix(
                    record.expression, record.reading, pinned_entry
                )
                pinned_confirmed = (
                    pinned_reading == record.reading or supports_suru_suffix
                )
                pinned_known: set[str] = set()
                if not pinned_confirmed:
                    same_entry = (
                        pinned_entry.get("vid"),
                        pinned_entry.get("sid"),
                    ) == (entry.get("vid"), entry.get("sid"))
                    pinned_known = (
                        primary_known
                        if same_entry and primary_known is not None
                        else _readings_for(client, pinned_entry)
                    )
                    pinned_confirmed = record.reading in pinned_known
                if not pinned_confirmed:
                    listed = (
                        ", ".join(sorted(pinned_known)) or pinned_reading or "none"
                    )
                    result.warnings.append(
                        f"{record_id}: janki reads {record.expression} as "
                        f"{record.reading}, jpdb lists {listed}. Nothing was written — "
                        "the reading is part of the record ID and is never auto-fixed."
                    )
                    continue
                token, entry = pinned_token, pinned_entry
                jpdb_reading = pinned_reading

        # Reconciliation demands the exact identity, and the reading checks
        # above cannot supply it alone: a kana-written word tokenizes to
        # whichever homograph is more common (あめ the candy resolves to 雨,
        # readings agreeing all the way), and the suru-suffix allowance
        # deliberately accepts the stem's entry for a 〜する record. Both are
        # fine sources for *empty* fields; neither is authority to overwrite a
        # provisional claim about a different spelling.
        # Normalized on both sides: this project's inputs are documented to
        # carry decomposed dakuten, and two spellings that render identically
        # must not read as a homograph mismatch.
        entry_spelling = str(entry.get("spelling", "")).strip()
        if reconcile and normalize_identity_part(entry_spelling) != (
            normalize_identity_part(record.expression)
        ):
            result.warnings.append(
                f"{record_id}: jpdb resolved {record.expression} to its entry "
                f"for {entry_spelling or 'another word'}; provisional "
                f"{', '.join(reconcile)} are settled only against the exact "
                "spelling, so they stay provisional"
            )
            reconcile = []
        proposals = _proposals(record, token, entry, kanji_store)
        _valid_pitch, invalid_pitch = _compatible_pitch_patterns(
            record.reading, entry.get("pitch_accent")
        )
        updated, changes = _apply(record, proposals, [*wanted, *reconcile])
        # After `_apply`, not before. What happens to the usable patterns is
        # `_wanted`'s decision, not this pass's: a record that already carries
        # an accent keeps it and jpdb's is discarded, so a warning written
        # ahead of the write told a curator jpdb's pattern was now on the
        # record while the record still had its own.
        if invalid_pitch:
            result.warnings.append(
                f"{record_id}: jpdb pitch pattern(s) "
                f"{_unusable_detail(record.reading, invalid_pitch)}; "
                + _what_became_of_the_rest(
                    _valid_pitch, "pitch_accent" in changes, updated.pitch_accent
                )
            )
        if "pitch_accent" in changes:
            updated = pitch.bind_source(updated)
        # By this point the entry is the exact identity — the spelling matched
        # and the reading survived the checks above — so a non-empty proposal
        # settles a provisional claim either way: a different value replaced it
        # (visible in the diff), an equal one confirmed it. Only a field jpdb
        # had no answer for stays provisional. A *forced* write to a marked
        # field settles it too: the value is janki's own dictionary write now,
        # and a surviving mark would misreport the next run's stale-clear as a
        # human edit that never happened.
        resolved = [name for name in reconcile if not is_empty(proposals.get(name))]
        forced = [name for name in active if name in changes and name not in resolved]
        if resolved or forced:
            updated = clear_provisional(updated, [*resolved, *forced])
        if confirmed := [name for name in resolved if name not in changes]:
            result.cleared.setdefault(record_id, []).extend(confirmed)
            result.warnings.append(
                f"{record_id}: dictionary evidence confirmed provisional "
                f"{', '.join(confirmed)}; the mark was cleared with no change"
            )
        if unresolved := [name for name in reconcile if name not in resolved]:
            result.warnings.append(
                f"{record_id}: jpdb had no answer for provisional "
                f"{', '.join(unresolved)}; the mark stays until evidence or review"
            )
        if changes:
            result.changes[record_id] = changes
        # Identity, not equality: every no-op path above returns the very
        # object `record` names, and every mutating path built a fresh one.
        if updated is not record:
            result.records[by_id[record_id]] = updated
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
# jpdb fills what a dictionary knows. This fills what it does not: meanings in
# the record's context, example sentences a beginner can read, and a note about
# how the word is actually used.
# The prompt template states the whole contract and nothing audits the answer
# (M8.3) — what follows is fill discipline and derivation.

#: How many other records' examples ride along as variety pressure. Enough to
#: show the model what it has already written this run, few enough that the
#: prompt stays mostly the record in front of it.
VARIETY_EXAMPLES = 3

#: Past this many target records, a monolithic diff stops being review — so the
#: results go to a staging file and through `janki promote` instead
#: (DESIGN_V2: "a 500-record y/n diff is not review; a staging file is").
STAGING_THRESHOLD = 50


def ai_schema() -> Any:
    """The rich-card response shared by source extraction and bare words."""
    return ai_schema_module.rich_card_schema()


def ai_targets(
    records: Sequence[VocabularyRecord], ids: Sequence[str] | None = None
) -> list[VocabularyRecord]:
    """The records an AI pass would work on.

    Content-defined by default: a record with no meaning or example sentence
    needs this pass, whatever any previous pass recorded about it. A usage note
    is deliberately not a completion signal: the rich prompt permits an empty
    note when there is no useful, certain nuance to add.
    Naming ids explicitly overrides that — re-running over a complete record is
    a legitimate thing to ask for, and `--force-fields` decides whether the
    answer may replace existing content.
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


def pinned_examples(record: VocabularyRecord) -> list[ExampleSentence]:
    """The examples the AI prompt pins for annotation, in stored order.

    Pinning an existing sentence is a claim of authority — "someone accepted
    this Japanese, annotate it" — so it is made per sentence, and only for
    accepted ones (:func:`models.example_accepted`). An unaccepted example (a
    machine-era sentence on an extract record that no reviewer's stamp
    covers) gets no mention at all: describing it to the model as content to
    preserve was the camera pilot's false-reviewed failure.

    A named function with one caller — :func:`ai_prompt`, which renders it —
    because the rule it encodes is worth reading on its own. Its previous
    justification was that the replay runner observed the same selection; M8.4
    deleted that runner, and there is no second caller to name in its place.
    """
    return [
        example
        for example in record.examples
        if example.needs_ai_annotations() and example_accepted(record, example)
    ]


def ai_prompt(
    record: VocabularyRecord,
    recent: Sequence[str] = (),
    taught: str = "",
) -> str:
    """The data-only user turn for one record and its run context.

    The dictionary facts go in so the model writes about *this* word rather
    than a homograph — 一日 with its reading attached is a different request
    from 一日 alone. The recent examples go in as variety pressure: asked for
    an example of twenty verbs in a row, a model will write twenty variations
    of 毎日〜ます unless it can see that it already did.

    Instruction prose belongs to the task template under ``prompts/``. This
    function only labels record data, reviewed lesson data, and recent output;
    the template says what the model must do with each block.
    """
    lines = [f"Expression: {record.expression}"]
    if record.reading:
        lines.append(f"Reading: {record.reading}")
    if record.meanings:
        lines.append("Current meanings: " + "; ".join(record.meanings))
    for label, value in (
        ("Part of speech", record.part_of_speech),
        ("Verb group", record.verb_group),
        ("Transitivity", record.transitivity),
    ):
        if value:
            lines.append(f"{label}: {value}")
    accepted = [
        example for example in record.examples if example_accepted(record, example)
    ]
    if accepted:
        lines.append("\nExisting curated examples:")
        lines.extend(
            "- "
            + json.dumps(
                {
                    "japanese": example.japanese,
                    "furigana": example.furigana,
                    "romaji": example.romaji,
                    "english": example.english,
                    "speech_level": example.register,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            for example in accepted
        )
    incomplete = pinned_examples(record)
    lines.append("\nExisting curated examples requiring annotations:")
    if incomplete:
        lines.extend(
            f"- {json.dumps(example.japanese, ensure_ascii=False)}"
            for example in incomplete
        )
    else:
        lines.append("(none)")
    if record.usage_notes:
        lines.append(f"\nCurrent usage notes: {record.usage_notes}")
    if taught:
        lines.append("\n" + taught)
    lines.append("\nRecent examples from this run:")
    if recent:
        lines.extend(f"- {item}" for item in recent)
    else:
        lines.append("(none)")
    return "\n".join(lines)


def ai_input_fingerprint(
    record: VocabularyRecord,
    recent: Sequence[str] = (),
    taught: str = "",
) -> str:
    """Identify the exact record-data turn used to request an answer."""
    return prompts.fingerprint(ai_prompt(record, recent, taught))


def ai_request_fingerprint(
    record: VocabularyRecord,
    *,
    provider: str,
    style_guide: str,
    instructions: str,
    recent: Sequence[str] = (),
    taught: str = "",
) -> str:
    """Identify all prompt channels and the response schema for one call."""
    user_turn = ai_prompt(record, recent, taught)
    if provider == "anthropic":
        wire_schema = claude_client.wire_schema(ai_schema())
        transport_prompt: Any = {
            "system": [style_guide, instructions],
            "user": user_turn,
        }
    elif provider == "codex":
        wire_schema = codex_client.wire_schema(ai_schema())
        transport_prompt = codex_client.wire_prompt(
            style_guide, instructions, user_turn
        )
    else:
        raise EnrichError(f"Unknown AI enrichment provider {provider!r}")
    return prompts.request_fingerprint(
        provider=provider,
        style_guide=style_guide,
        task_template=instructions,
        user_turn=user_turn,
        transport_prompt=transport_prompt,
        schema=wire_schema,
    )


@dataclass(slots=True)
class AiOutcome:
    """What the AI pass decided for one record."""

    record: VocabularyRecord
    changes: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    #: Fields whose provisional mark this answer settled. Its own channel
    #: because a mark can be settled *without* a value change — the model was
    #: shown the claim and restated it — and the caller's save gate reads
    #: ``changes``. Folded in, a confirming answer would look like no answer
    #: at all and the settled record would be dropped on the floor.
    cleared: list[str] = field(default_factory=list)
    #: True when generated examples were discarded while stored ones were
    #: preserved. A generated sentence may now fill an unoccupied labelled slot;
    #: this records only the remainder that had nowhere safe to land, so the
    #: caller's warning reports what this function actually did.
    preserved: bool = False
    #: Romaji the model supplied that does not transliterate its own sentence,
    #: one message per example. Carried rather than printed here because a
    #: disagreement between a sentence and its romaji is a fact about the
    #: answer, and the caller decides whether that reaches a terminal, a
    #: staging file, or both.
    romaji_rejected: list[str] = field(default_factory=list)


def _fill_existing_example_annotations(
    stored: Sequence[ExampleSentence],
    generated: Sequence[ExampleSentence],
) -> list[ExampleSentence]:
    """Fill holes only when generated Japanese exactly matches stored text."""
    by_japanese: dict[str, ExampleSentence] = {}
    for example in generated:
        by_japanese.setdefault(example.japanese, example)
    merged: list[ExampleSentence] = []
    for old in stored:
        incoming = by_japanese.get(old.japanese)
        if incoming is None:
            merged.append(old)
            continue
        updated = replace(
            old,
            furigana=old.furigana or incoming.furigana,
            english=old.english or incoming.english,
            register=(
                old.register
                # Exact, and safe because `ExampleSentence.from_dict` lowercases
                # and strips on every read, so a record reaching here cannot
                # carry "Casual". Normalizing here as well would be worse than
                # redundant: it would *preserve* a mixed-case label, and
                # `needs_ai_annotations` still compares exactly, so the record
                # would be re-sent to the model on every run and never converge.
                if old.register in {"polite", "casual"}
                else incoming.register
            ),
        )
        if updated.furigana or not contains_kanji(updated.japanese):
            updated, _rejected = qc.settle_example_romaji(updated)
        merged.append(updated)
    return merged


def _complete_unoccupied_example_slots(
    record: VocabularyRecord,
    generated: Sequence[ExampleSentence],
) -> tuple[list[ExampleSentence], bool]:
    """Preserve stored sentences, then fill empty labelled card slots.

    Slot occupancy is structural: it reads only the stored ``register`` labels
    and the schema-constrained ``speech_level`` labels decoded onto incoming
    examples. It never inspects a sentence to decide whether it is polite or
    casual. A stored extract example without reviewer authority keeps the
    preserve-only posture it had before this helper existed; only a collection
    of accepted examples may be augmented without ``--force-fields examples``.

    The boolean says whether any generated sentence was discarded. Carrying the
    decision out of this function keeps the CLI warning aligned with partial
    success, where one missing slot fills and an extra answer is still ignored.
    """
    merged = _fill_existing_example_annotations(record.examples, generated)
    stored_texts = {example.japanese for example in record.examples}
    unmatched = [
        example for example in generated if example.japanese not in stored_texts
    ]
    if not all(example_accepted(record, example) for example in record.examples):
        return merged, bool(unmatched)

    occupied = {
        example.register
        for example in merged
        if example.register in {"polite", "casual"}
    }
    used_texts = set(stored_texts)
    discarded = False
    for example in unmatched:
        slot = example.register
        if example.japanese in used_texts or slot not in {"polite", "casual"}:
            discarded = True
            continue
        if slot in occupied:
            discarded = True
            continue
        merged.append(example)
        used_texts.add(example.japanese)
        occupied.add(slot)
    return merged, discarded


def _settle_ai_marks(
    record: VocabularyRecord,
    updated: VocabularyRecord,
    changes: Mapping[str, tuple[Any, Any]],
    writable: Sequence[str],
    proposals: Mapping[str, Any],
) -> tuple[VocabularyRecord, list[str]]:
    """Clear the provisional marks this pass's own answer resolved.

    The AI pass is the one thing `docs/DESIGN.md` allows to settle a
    provisional meaning, because it reads the record — its examples, its usage
    note, the sense the source actually taught. So when it writes that field,
    the value stops being extraction's claim and becomes this pass's write, and
    the mark has to go with it.

    Without this the mark outlives the write and the *next* jpdb run finds a
    fingerprint that no longer matches. It reads that as a human edit and
    reports "edited since extraction; the field is kept as curated" about a
    value janki wrote itself — the exact misreport
    :func:`enrich_records`' forced-write settle exists to prevent, and one that
    then rewrites the collection to say so.

    A model that re-states the value it was shown settles it too. That is
    evidence, not a no-op: a mark means "nobody who can read this card has
    confirmed it", and one just did. Without this clause a restored meaning the
    model agrees with keeps its mark permanently, because no janki operation
    would ever have cause to touch the field again.

    What decides is whether *this pass acted on the field*, not whether the
    mark was still bound to its value. A stale mark on a field this pass then
    overwrote describes a value that is now two writes gone, so keeping it only
    arms the misreport above. A stale mark on a field this pass left alone is
    the real thing — a human edit outranking the dictionary — and stays for the
    jpdb pass to clear with the warning that says so, which is the only notice
    the user gets.
    """
    settled = [
        name
        for name, _fingerprint in provisional_entries(record)
        if name in changes
        or (name in writable and not is_empty(proposals.get(name)))
    ]
    return (clear_provisional(updated, settled) if settled else updated), settled


def apply_ai_result(
    record: VocabularyRecord,
    parsed: Any,
    *,
    force_fields: Sequence[str] = (),
) -> AiOutcome:
    """Fold a model's answer into the record.

    No checks on the Japanese. M8.3 deleted the audit layer that used to sit
    here — headword containment, KANJIDIC reading adjudication, the
    furigana-vs-sentence comparison, the punctuation repair — because janki's
    logic enriches the card and never audits the model (DESIGN.md). The prompt
    template states the contract; the model's answer is the answer.

    What remains is fill discipline and derivation: stored examples are
    preserved unless ``--force-fields examples`` asks otherwise, existing
    annotations win over the model's, the shared schema limits speech level to
    the two card slots, and romaji is checked against the reading rather than
    rebuilt from it — see :func:`qc.settle_example_romaji` for why the rebuild
    had to go.
    """
    outcome = AiOutcome(record=record)
    content = ai_schema_module.adapt_rich_card(parsed)
    kept = list(content.examples)
    proposals: dict[str, Any] = {
        "meanings": list(content.meanings),
        "examples": kept,
        "usage_notes": content.usage_notes,
    }
    # Stored examples are preserved unless the user asked for a replacement
    # (`--force-fields examples`) — including unaccepted machine-era sentences
    # on extract records. Nothing in data can distinguish those from a
    # sentence a person curated before the authority keys existed, and only
    # the user may decide which one they are; the prompt already refuses to
    # pin an unaccepted sentence, so nothing here launders it into curated
    # content either.
    if record.examples and "examples" not in force_fields:
        merged_examples, outcome.preserved = _complete_unoccupied_example_slots(
            record, kept
        )
        updated = record
        changes: dict[str, tuple[Any, Any]] = {}
        if merged_examples != record.examples:
            changes["examples"] = (record.examples, merged_examples)
            updated = replace(record, examples=merged_examples)
        writable = [
            name
            for name in AI_FIELDS
            if name != "examples"
            and (name in force_fields or is_empty(getattr(record, name)))
        ]
        updated, other_changes = _apply(updated, proposals, writable)
        changes.update(other_changes)
    else:
        writable = [
            name
            for name in AI_FIELDS
            if name in force_fields or is_empty(getattr(record, name))
        ]
        updated, changes = _apply(record, proposals, writable)
    updated, outcome.cleared = _settle_ai_marks(
        record, updated, changes, writable, proposals
    )
    outcome.record = updated
    outcome.changes = changes
    outcome.romaji_rejected = list(content.romaji_rejected)
    return outcome


@dataclass(slots=True)
class AiResult:
    """What an AI pass would write, and everything it refused along the way."""

    records: list[VocabularyRecord] = field(default_factory=list)
    changes: dict[str, dict[str, tuple[Any, Any]]] = field(default_factory=dict)
    #: ``record id -> [field]`` whose provisional mark this pass settled with
    #: no value change — the model was shown the claim and restated it.
    #:
    #: The same channel, and for the same reason, as ``EnrichResult.cleared``:
    #: every save gate reads ``changes``, so a settle that rode only on
    #: ``changes`` would never reach disk when the answer agreed. The record
    #: would keep its mark, reappear in ``status --unsettled``, and buy another
    #: paid call the next time somebody piped that list into this pass — for
    #: as long as the model kept agreeing.
    cleared: dict[str, list[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    no_changes: list[str] = field(default_factory=list)
    #: Exact full-call fingerprints, keyed by the record whose answer they
    #: produced. The ledger can therefore say which style guide, task template,
    #: record turn, and schema authored stored enrichment.
    provenance: dict[str, str] = field(default_factory=dict)
    #: Exact data-turn fingerprints. These are also the batch currency tokens:
    #: an answer may land only while its input record still renders this turn.
    input_fingerprints: dict[str, str] = field(default_factory=dict)
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


def enrich_ai(
    records: Sequence[VocabularyRecord],
    *,
    model: str,
    provider: str = "anthropic",
    style_guide: str,
    instructions: str,
    force_fields: Sequence[str] = (),
    ids: Sequence[str] | None = None,
    client: Any | None = None,
    parse_call: Any | None = None,
    call_options: Mapping[str, Any] | None = None,
    taught: str = "",
) -> AiResult:
    """Write meanings, examples, and usage notes for incomplete records.

    One call per record keeps failures local to that record. Refusals and
    truncated answers are never salvaged, and nothing audits the answer —
    the template asked precisely, and the answer is the answer (M8.3).
    """
    result = AiResult(records=list(records))
    positions = {record.id: index for index, record in enumerate(result.records)}
    targets = ai_targets(result.records, ids)
    blocks = claude_client.system_blocks(style_guide, instructions)
    caller = parse_call or claude_client.parse_call
    options = dict(call_options or {})
    recent: list[str] = []

    for record in targets:
        result.looked_up += 1
        recent_examples = recent[-VARIETY_EXAMPLES:]
        user_turn = ai_prompt(record, recent_examples, taught)
        result.input_fingerprints[record.id] = prompts.fingerprint(user_turn)
        result.provenance[record.id] = ai_request_fingerprint(
            record,
            provider=provider,
            style_guide=style_guide,
            instructions=instructions,
            recent=recent_examples,
            taught=taught,
        )
        call = caller(
            model,
            blocks,
            user_turn,
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
        )
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

    outcome = apply_ai_result(
        record,
        parsed,
        force_fields=force_fields,
    )
    if outcome.preserved:
        # Not an audit — a report of janki's own fill discipline. One missing
        # labelled slot may have filled; this flag names only generated answers
        # left over after that structural merge.
        result.warnings.append(
            f"{record.id}: stored examples were preserved; generated sentences "
            "that did not fill an unoccupied polite/casual slot were discarded "
            "— pass --force-fields examples to replace the stored set."
        )
    for rejected in outcome.romaji_rejected:
        # The sentence and its romaji disagree. janki kept the sentence and
        # replaced the romaji with what the reading actually transliterates
        # to, which is the safe half — but the disagreement itself says the
        # answer was not internally consistent, and that is a human's to look
        # at rather than something to repair quietly.
        result.warnings.append(f"{record.id}: {rejected}")
    if outcome.cleared:
        result.cleared.setdefault(record.id, []).extend(outcome.cleared)
    if outcome.changes or outcome.cleared:
        # `or outcome.cleared`: an answer that restates the stored value writes
        # no field but still resolves the authority question, and the object
        # carrying that resolution is `outcome.record`. Keeping only the
        # changed ones dropped it, which left the mark on disk and made the
        # next run ask the model the same question again, at the same price.
        result.records[positions[record.id]] = outcome.record
    if outcome.changes:
        result.changes[record.id] = {
            **result.changes.get(record.id, {}),
            **outcome.changes,
        }
        recent.extend(
            example.japanese for example in outcome.record.examples if example.japanese
        )
    elif record.id not in result.changes and record.id not in result.no_changes:
        result.no_changes.append(record.id)


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


@dataclass(slots=True)
class BatchPlan:
    """Serializable requests and the provenance needed to land them safely."""

    requests: list[dict[str, Any]] = field(default_factory=list)
    record_ids: list[str] = field(default_factory=list)
    request_fingerprints: dict[str, str] = field(default_factory=dict)
    input_fingerprints: dict[str, str] = field(default_factory=dict)


def batch_requests(
    records: Sequence[VocabularyRecord],
    *,
    model: str,
    style_guide: str,
    instructions: str,
    ids: Sequence[str] | None = None,
    taught: str = "",
) -> BatchPlan:
    """Build batch entries together with exact request and input provenance.

    A one-hour cache TTL rather than the default five minutes: the style guide
    leads every request, and a batch's requests are read over a span that a
    five-minute window would not survive.

    ``taught`` carries the reviewed patterns, exactly as the immediate path
    does. Without it the two paths wrote different sentences for the same
    record at different prices — and batch is the one used for bulk, so most of
    a collection would have got the unsteered version.
    """
    targets = ai_targets(records, ids)
    blocks = claude_client.system_blocks(style_guide, instructions, cache_ttl="1h")
    record_ids = [record.id for record in targets]
    batch_key_map(record_ids)
    user_turns = {record.id: ai_prompt(record, taught=taught) for record in targets}
    by_id = {record.id: record for record in targets}
    requests = [
        claude_client.batch_request(
            batch_custom_id(record.id),
            model,
            blocks,
            user_turns[record.id],
            ai_schema(),
            effort=claude_client.effort_for(model),
        )
        for record in targets
    ]
    input_fingerprints = {
        record_id: prompts.fingerprint(user_turn)
        for record_id, user_turn in user_turns.items()
    }
    request_fingerprints = {
        record_id: ai_request_fingerprint(
            by_id[record_id],
            provider="anthropic",
            style_guide=style_guide,
            instructions=instructions,
            taught=taught,
        )
        for record_id in user_turns
    }
    return BatchPlan(
        requests=requests,
        record_ids=record_ids,
        request_fingerprints=request_fingerprints,
        input_fingerprints=input_fingerprints,
    )


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
    #: Rows whose record-data turn no longer matches the submitted request.
    #: Their paid answers are retained by the batch but never applied to newer
    #: curation.
    stale: list[str] = field(default_factory=list)


def apply_batch_results(
    records: Sequence[VocabularyRecord],
    entries: Iterable[claude_client.BatchEntry],
    pending_ids: Sequence[str],
    *,
    model: str,
    input_fingerprints: Mapping[str, str],
    request_fingerprints: Mapping[str, str] | None = None,
    force_fields: Sequence[str] = (),
    only: Sequence[str] = (),
    taught: str = "",
) -> BatchApplyResult:
    """Fold a finished batch into the records, through the live path's checks.

    Every answer goes through :func:`absorb_ai_call`, which is the same function
    the synchronous pass uses — the saving is in how the request was sent, not
    in what is done with the reply.

    Five ways a record can come back with nothing, none of them silent, and two
    of them different in kind from the rest:

    * the batch reported it **errored, expired or was canceled** — terminal, a
      re-fetch returns the same row;
    * the batch **never mentioned it**, which will not change either;
    * its id **no longer names a record**, because the collection moved while
      the batch was out — deliberate curation, on this side;
    * its input **changed after submission** — the old answer is stale and is
      never applied to newer curation;
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
        index = positions.get(record_id)
        if index is not None:
            current_fingerprint = ai_input_fingerprint(
                outcome.result.records[index], taught=taught
            )
            if input_fingerprints.get(record_id) != current_fingerprint:
                outcome.stale.append(record_id)
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
        if index is None:
            outcome.missing.append(record_id)
            continue
        outcome.result.looked_up += 1
        outcome.result.input_fingerprints[record_id] = input_fingerprints[record_id]
        if request_fingerprints and (fingerprint := request_fingerprints.get(record_id)):
            outcome.result.provenance[record_id] = fingerprint
        absorb_ai_call(
            outcome.result,
            outcome.result.records[index],
            entry.result,
            model=model,
            positions=positions,
            recent=recent,
            force_fields=force_fields,
        )

    for record_id in pending_ids:
        if record_id in seen or (candidates and record_id not in candidates):
            continue
        if record_id in positions:
            outcome.failed[record_id] = "the batch returned no result for it"
        else:
            outcome.missing.append(record_id)
    return outcome
