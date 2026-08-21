"""Moving a reviewed staging file into the records janki actually builds from.

This is the gate between "a machine proposed it" and "janki believes it".
Everything upstream — ``extract``, the import hold-backs, M4.2's enrichment —
writes into ``data/staging/``; nothing but this reaches ``vocabulary.json``.

Two checks stand at that gate, and they are about the same thing: the reading.

**The reading is half of the record ID**, so a wrong one is not a typo to fix
later — it is a permanent, uncorrectable identity, and the Anki review history
behind it is orphaned the moment anyone tries. So a candidate whose reading is
missing, or written in kanji, is held back no matter what else is right about
it; and a candidate whose reading jpdb has never heard of is held back too,
because the likeliest explanation is that a human transcribed it wrong.

**The one sanctioned ID change lives here.** A held-back row carries the
malformed id its import minted — ``word:<expr>:`` or ``word:<kanji>:<kanji>`` —
and once a reviewer supplies the reading, that id no longer describes the
record. Promoting it unchanged would write into ``vocabulary.json`` precisely
the id the hold-back existed to prevent, which ``validation`` then errors on
forever. The re-mint is keyed off the id itself — re-mint whenever
``record.id != stable_record_id(expression, reading)`` — which subsumes both
malformed shapes without enumerating them, and is the only test that still
works by promote time, when the kanji reading has already been replaced with
kana. These records have never been in ``vocabulary.json`` or Anki, so there is
no history to orphan.
"""

from __future__ import annotations

import re
from collections.abc import Container, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any

from japanese_anki import enrich, extract, jpdb
from japanese_anki import pitch as pitch_module
from japanese_anki.errors import JankiError
from japanese_anki.identifiers import contains_kanji, stable_record_id
from japanese_anki.io import MergeOutcome, merge_records
from japanese_anki.models import (
    EXAMPLE_AUTHORITY_KEY,
    EXAMPLE_AUTHORITY_STAGING,
    VocabularyRecord,
    set_example_flags,
)
from japanese_anki.staging import (
    CANDIDATE_ACCOUNTING_KEY,
    FIELD_REPLACEMENTS_KEY,
    HOLD_MISSING_READING,
    HOLD_READING_KANJI,
    HOLD_UNKNOWN_READING,
    HOLD_UNVERIFIABLE_ID,
    already_landed_field_replacements,
    annotate,
    authorized_field_replacements,
    field_replacement_block,
    require_resolved_coverage,
)

__all__ = [
    "HOLD_MISSING_READING",
    "HOLD_READING_KANJI",
    "HOLD_UNKNOWN_READING",
    "HOLD_UNVERIFIABLE_ID",
    "FIELD_REPLACEMENTS_KEY",
    "PromoteError",
    "PromoteResult",
    "check_readings",
    "check_coverage",
    "check_candidate_accounting",
    "field_replacement_block",
    "already_landed_staged_fields",
    "merge_staged_records",
    "remint",
]

# The hold vocabulary lives in `staging`, which owns the annotation key these
# are written under and is the one module both this and `enrich` can import —
# `enrich.needs_reading` has to tell a reading hold from an id hold, and this
# module already imports `enrich`. Re-exported here, because this is where they
# are written and where readers have always looked for them.


class PromoteError(JankiError):
    pass


def already_landed_staged_fields(
    existing: Sequence[VocabularyRecord],
    incoming: Sequence[VocabularyRecord],
    meta: dict[str, Any],
) -> dict[str, tuple[str, ...]]:
    """Fields whose staged value proves an earlier records write completed."""
    try:
        return already_landed_field_replacements(meta, existing, incoming)
    except JankiError as exc:
        raise PromoteError(str(exc)) from exc


def merge_staged_records(
    existing: list[VocabularyRecord],
    incoming: list[VocabularyRecord],
    meta: dict[str, Any],
    *,
    validate_incoming: Sequence[VocabularyRecord] | None = None,
) -> tuple[list[VocabularyRecord], dict[str, MergeOutcome]]:
    """Merge reviewed staging, replacing only fingerprint-authorized fields.

    The complete old-value check runs before :func:`merge_records` sees one
    incoming row.  That ordering is the atomicity guarantee for a stale review:
    a concurrent edit to any replacement target refuses the whole merge rather
    than landing the earlier records and discovering the stale one later.

    Metadata with no ``field_replacements`` block takes the ordinary
    existing-wins route.  In particular, schema-v2 extraction staging remains
    promotable unchanged; it was written before rich AI answers could propose
    replacing curated values.
    """
    try:
        # A reading hold narrows what may land, not what review the old-value
        # binding covers.  Validate every row still in the live staging file
        # before merging the promotable subset.  Rows already archived are no
        # longer passed here: their replacements landed under this binding on
        # the earlier partial promotion.
        authorized = authorized_field_replacements(
            meta,
            existing,
            list(validate_incoming) if validate_incoming is not None else incoming,
        )
    except JankiError as exc:
        raise PromoteError(str(exc)) from exc
    return merge_records(
        existing,
        incoming,
        (),
        prefer_incoming_by_id=authorized,
    )


def check_coverage(meta: dict[str, Any]) -> None:
    """Apply the coverage gate before promotion can read clients or write data."""
    try:
        require_resolved_coverage(meta)
    except JankiError as exc:
        raise PromoteError(str(exc)) from exc
    block = meta.get("coverage")
    if not isinstance(block, dict):
        return
    accounting = meta.get(CANDIDATE_ACCOUNTING_KEY)
    if block.get("version") == 2:
        if not isinstance(accounting, Mapping):
            raise PromoteError(
                "[candidate-accounting-invalid] coverage v2 needs candidate_accounting"
            )
        try:
            extract.validate_candidate_accounting_block(accounting)
        except JankiError as exc:
            raise PromoteError(str(exc)) from exc
    elif accounting is not None:
        raise PromoteError(
            "[candidate-accounting-invalid] candidate_accounting is not bound by "
            "coverage v2; it cannot be backfilled onto an older paid artifact"
        )
    _verify_coverage_facts(meta, block)


def check_candidate_accounting(
    meta: Mapping[str, Any],
    live: Sequence[VocabularyRecord],
    archived: Sequence[VocabularyRecord] = (),
    *,
    archived_meta: Mapping[str, Any] | None = None,
) -> tuple[bool, ...]:
    """Authenticate candidate accounting and classify transformed archive retries.

    Reviewed record content, identity, and selection remain editable. The
    immutable block describes what the model proposed, not what the human must
    keep. An exact row in both a same-run done archive and live staging is the
    recoverable boundary where archive writing succeeded but pruning failed.
    """
    coverage = meta.get("coverage")
    coverage_version = (
        coverage.get("version") if isinstance(coverage, Mapping) else None
    )
    accounting = meta.get(CANDIDATE_ACCOUNTING_KEY)
    if coverage_version != 2:
        if accounting is not None:
            raise PromoteError(
                "[candidate-accounting-invalid] candidate_accounting is not bound "
                "by coverage v2; it cannot be backfilled onto an older paid artifact"
            )
    else:
        if not isinstance(accounting, Mapping):
            raise PromoteError(
                "[candidate-accounting-invalid] coverage v2 needs candidate_accounting"
            )
        try:
            extract.validate_candidate_accounting_block(accounting)
        except JankiError as exc:
            raise PromoteError(str(exc)) from exc
        if archived_meta is not None and archived_meta.get(
            CANDIDATE_ACCOUNTING_KEY
        ) != dict(accounting):
            raise PromoteError(
                "[candidate-accounting-archive-divergent] the same-run archive "
                "carries different candidate accounting"
            )

    archived_ids = [record.id for record in archived]
    if len(set(archived_ids)) != len(archived_ids):
        raise PromoteError(
            "[archive-retry-divergent] the same-run archive repeats a canonical "
            "record id"
        )
    live_ids = [record.id for record in live]
    if len(set(live_ids)) != len(live_ids):
        raise PromoteError(
            "[canonical-record-id-collision] reviewed rows converge on one "
            "canonical record id; resolve them explicitly before promotion"
        )
    unmatched_archive = set(range(len(archived)))
    exact_retry: list[bool] = []
    for record in live:
        resolved = _accept_examples(_resolved(record))
        stable = replace(
            resolved,
            id=stable_record_id(resolved.expression, resolved.reading),
        )
        variants = {resolved.id: resolved.to_dict(), stable.id: stable.to_dict()}
        matches = [
            index
            for index in sorted(unmatched_archive)
            if archived[index].to_dict() in variants.values()
        ]
        if len(matches) > 1:
            raise PromoteError(
                "[archive-retry-divergent] one live row matches multiple rows in "
                "the same-run archive"
            )
        match = matches[0] if matches else None
        exact_retry.append(match is not None)
        if match is not None:
            unmatched_archive.remove(match)
            continue
        if set(variants) & set(archived_ids):
            raise PromoteError(
                "[archive-retry-divergent] a live row's unchanged or stable-reminted "
                "id exists in the same-run archive but its reviewed content differs"
            )

    if isinstance(accounting, Mapping):
        accepted_population = len(archived) + exact_retry.count(False)
        parsed_population = accounting["parsed_candidate_count"]
        if accepted_population > parsed_population:
            raise PromoteError(
                "[candidate-accounting-population-exceeded] the same-run done "
                f"archive and live review contain {accepted_population} unique "
                "accepted row(s), but the immutable candidate account proves only "
                f"{parsed_population} parsed proposal(s). Reviewers may delete or "
                "re-identify proposals, but cannot split one proposal into extra "
                "records. Remove the appended row(s); do not edit candidate_accounting."
            )

    return tuple(exact_retry)


def _verify_coverage_facts(meta: dict[str, Any], block: dict[str, Any]) -> None:
    """Re-derive the coverage facts and refuse a staging file that disagrees.

    What survives the M8.4 deletion of the approved-oracle apparatus. The
    oracle answered "did the model return everything a human said was on this
    page" — the pilot programme's question, cancelled with it. This answers a
    question that is still worth asking and needs nobody's approval: do the
    numbers in this file follow from the source units recorded beside them, or
    has one been edited without the other?
    """
    raw_units = block.get("source_units")
    if not isinstance(raw_units, list):
        raise PromoteError(
            "[coverage-block-invalid] coverage source_units must be a list"
        )
    units: list[extract.SourceUnit] = []
    try:
        for raw in raw_units:
            if not isinstance(raw, dict):
                raise TypeError
            page = raw["page"]
            ordinal = raw["ordinal"]
            section = raw["section"]
            context = raw["context"]
            disposition = raw["disposition"]
            reason = raw.get("reason", "")
            if (
                isinstance(page, bool)
                or not isinstance(page, int)
                or page < 1
                or isinstance(ordinal, bool)
                or not isinstance(ordinal, int)
                or ordinal < 1
                or not isinstance(section, str)
                or not re.fullmatch(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", section)
                or not isinstance(context, str)
                or not extract.normalize_context(context)
                or disposition not in extract.SOURCE_UNIT_DISPOSITIONS
                or not isinstance(reason, str)
                or (disposition != "candidate" and not reason.strip())
            ):
                raise ValueError
            units.append(
                extract.SourceUnit(
                    page=page,
                    section=section,
                    ordinal=ordinal,
                    context=context,
                    context_fingerprint=extract.context_fingerprint(context),
                    disposition=disposition,
                    reason=reason.strip(),
                )
            )
        prose_count = int(block.get("prose_candidate_count", 0))
        reported_count = int(block.get("model_reported_unit_count", 0))
    except (KeyError, TypeError, ValueError) as exc:
        raise PromoteError(
            "[coverage-block-invalid] coverage source-unit facts are malformed"
        ) from exc
    if prose_count < 0 or reported_count < 0:
        raise PromoteError(
            "[coverage-block-invalid] coverage counts must be non-negative"
        )
    mode_value = None
    provenance = meta.get("prompt_provenance")
    if isinstance(provenance, dict):
        mode_value = provenance.get("mode")
    mode = None if mode_value in {None, "auto"} else str(mode_value)
    if mode not in {None, *extract.MODES}:
        raise PromoteError("[coverage-block-invalid] coverage has an invalid mode")
    # Coverage only needs to know how many candidates came from prose. The
    # source-unit/candidate link was checked before staging was written.
    candidates = tuple(SimpleNamespace(source_kind="prose") for _ in range(prose_count))
    regenerated = extract.coverage_block(
        extract.ExtractionResult(tuple(candidates), tuple(units), reported_count),
        source_sha256=str(block.get("source_fingerprint", "")),
        mode=mode,
        candidate_accounting=(
            meta.get(CANDIDATE_ACCOUNTING_KEY)
            if block.get("version") == 2
            and isinstance(meta.get(CANDIDATE_ACCOUNTING_KEY), Mapping)
            else None
        ),
    )
    stored_facts = {
        key: value
        for key, value in block.items()
        if key not in {"approval", "coverage_block_fingerprint"}
    }
    regenerated_facts = {
        key: value
        for key, value in regenerated.items()
        if key != "coverage_block_fingerprint"
    }
    if stored_facts != regenerated_facts:
        raise PromoteError(
            "[coverage-facts-stale] coverage facts do not match the stored "
            "source units"
        )


def _unspeakable_patterns(record: VocabularyRecord) -> tuple[list[str], bool]:
    """Every stored pattern that cannot make a forced clip, named by its field,
    and whether the *chosen* one is among them.

    `audio_accent` is checked, not just `pitch_accent`, and it is checked
    first, because :func:`pitch.select_pattern` reads it first — it is the
    pattern that actually reaches the synthesizer when it is set. It is the
    more hand-written of the two: no importer writes it and it is not in
    `ENRICHABLE_FIELDS`, so promote is where a hand-typed one is first seen —
    not the only way it can arrive, since it is mergeable and a hand edit of
    `vocabulary.json` sets it, but the earliest. Reading `pitch_accent` alone
    both missed an
    unspeakable `audio_accent` entirely and blamed a `pitch_accent`
    that nothing was going to use.

    Each pattern carries its field name, and `pitch_accent` entries carry their
    index, because they are edited separately and a curator told "pitch pattern
    LHLL is wrong" on a record whose `pitch_accent` is empty — or whose
    `pitch_accent` has three entries — has been sent to the wrong line of the
    file. `validation.py` labels them `pitch_accent[0]` for the same reason,
    and says why this field is the one that matters: "it is the pattern audio
    generation actually uses when set, so a typo there is the one that reaches
    the synthesizer".

    Comparison is on the canonical form — stripped and upper-cased, as
    `select_pattern` returns it — because `pitch._LEVELS` accepts `h`/`l` as
    well as `H`/`L`. A record whose accent is typed in lower case renders
    perfectly well, and comparing raw strings would report the chosen pattern
    as fine when it is the broken one.
    """
    chosen = pitch_module.select_pattern(record)
    # `audio_accent` first, because `select_pattern` reads it first: when both
    # hold the same pattern the curator must be sent to the field the
    # synthesizer actually uses, not to the one that happens to sort first.
    stored = [("audio_accent", record.audio_accent)]
    stored += [
        (f"pitch_accent[{index}]", pattern)
        for index, pattern in enumerate(record.pitch_accent)
    ]
    unusable: list[str] = []
    seen: set[str] = set()
    for field_name, pattern in stored:
        canonical = pattern.strip().upper()
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        try:
            pitch_module.to_aquestalk(record.reading, pattern)
        except pitch_module.PitchError:
            unusable.append(f"{field_name} {pattern}")
    chosen_is_unusable = bool(
        chosen and any(entry.split(" ", 1)[1].strip().upper() == chosen for entry in unusable)
    )
    return unusable, chosen_is_unusable


@dataclass(slots=True)
class PromoteResult:
    """What a promote pass decided, row by row.

    ``keep`` is one flag per input row in file order — ``True`` for a row that
    stays in the staging file — so the caller can prune the file without
    re-deriving which rows survived.
    """

    promoted: list[VocabularyRecord] = field(default_factory=list)
    held: list[VocabularyRecord] = field(default_factory=list)
    keep: list[bool] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    reminted: dict[str, str] = field(default_factory=dict)


def remint(
    record: VocabularyRecord, already_stored: Container[str] = frozenset()
) -> VocabularyRecord:
    """The record under the id its expression and reading actually mint.

    Keyed off the id rather than the shape of the reading, because by promote
    time a reviewer has replaced a kanji reading with kana — so
    ``contains_kanji(reading)`` is ``False`` in exactly the case the re-mint
    exists for, and only the id still remembers.

    ``already_stored`` is the precondition this function has always claimed and
    never checked: it re-mints because *these records have never been in
    ``vocabulary.json`` or Anki*, so there is no review history to orphan. A
    staged row whose id **is** already in the collection breaks that
    precondition, and re-minting it there does real damage — the merge sees an
    id it has never met, adds a second record, and the curated original keeps
    its Anki history and never receives the change. That state is reachable:
    an id minted from a wrong reading stays put when the reading is corrected,
    because the id is uncorrectable by design, and M4.2's staging route then
    sends such a record back through promote. So an id the collection already
    holds is left exactly as it is.
    """
    if record.id in already_stored:
        return record
    minted = stable_record_id(record.expression, record.reading)
    return record if record.id == minted else replace(record, id=minted)


def _hold(record: VocabularyRecord, reason: str) -> VocabularyRecord:
    return annotate(record, hold_reason=reason)


def check_readings(
    records: Sequence[VocabularyRecord],
    *,
    client: jpdb.JpdbClient | None = None,
    skip_reading_check: bool = False,
    already_stored: Container[str] = frozenset(),
    remint_blocked: bool = False,
) -> PromoteResult:
    """Decide, per record, whether it may become a real record.

    Three outcomes for a record whose reading is usable kana, exactly as
    DESIGN_V2 specifies: the reading is what jpdb reaches for (pass), it is one
    jpdb lists for another sense (pass, with a warning — a homograph is a real
    thing and the reviewer chose it), or no entry lists it at all (held back,
    because a reading nobody recognises is far more likely a transcription slip
    than a discovery).

    ``skip_reading_check`` drops the third check only. The reading still has to
    *be* kana — that rule is about whether an id can exist at all, not about
    whether a dictionary agrees, and no flag turns it off.

    ``already_stored`` is the ids the collection already holds; see
    :func:`remint` for why a re-mint must not touch one of them.
    ``remint_blocked`` says that set could not be completed — an unreadable deck
    — in which case a row whose id would change is **held back** rather than
    promoted. Promoting it under the id it arrived with is not the cautious
    option: that id lands in the store permanently, since ``remint`` is the only
    thing that repairs a stored id and a stored id is exempt from it. Holding
    keeps the row in ``data/staging/``, which is committed, and re-running once
    the deck parses does the right thing.
    """
    result = PromoteResult()
    for record in records:
        reason = _structural_hold(record)
        if reason is not None:
            result.held.append(_hold(record, reason))
            result.keep.append(True)
            continue

        if not skip_reading_check:
            if client is None:
                raise PromoteError(
                    "The reading check needs a jpdb client. Pass one, or use "
                    "--skip-reading-check to promote without it."
                )
            verdict, note = _dictionary_verdict(client, record)
            if note:
                result.warnings.append(note)
            if verdict == "held":
                result.held.append(_hold(record, HOLD_UNKNOWN_READING))
                result.keep.append(True)
                continue

        resolved = _accept_examples(_resolved(record))
        # A near-miss sentinel is an explicit human act about to be silently
        # voided: the value accepts nothing (it matches no content
        # fingerprint), but the reviewer believes they accepted. Name it now,
        # while it is one keystroke to fix, not weeks later as an AI-pass
        # warning about sentences they already reviewed.
        leftover = resolved.source.raw_fields.get(EXAMPLE_AUTHORITY_KEY, "")
        if (
            resolved.source.type == "extract"
            and leftover
            and not _BOUND_AUTHORITY.fullmatch(leftover)
        ):
            result.warnings.append(
                f"{record.id}: example_authority is {leftover!r}, which is "
                f"neither the {EXAMPLE_AUTHORITY_STAGING!r} sentinel nor a "
                "bound acceptance — it accepts nothing. Retype the sentinel "
                "exactly to accept this row's examples."
            )
        # After the structural and reading holds, before the remint one — so a
        # row held for an unverifiable id is still warned about audio it is not
        # getting yet. Left that way deliberately: the row stays in staging and
        # is re-checked next run, and a curator fixing the id wants to know
        # about the accent in the same pass rather than the one after.
        #
        # The last door a pitch pattern can come through. `enrich --jpdb`
        # refuses an unspeakable pattern and `import-jpdb` names one; a staging
        # file carries `pitch_accent` as a first-class field, so a hand-written
        # or held-back row could put one into the collection with nothing said
        # anywhere — and `docs/AUDIO.md` claimed the state was always visible.
        # Named, not held: an accent is not identity, the clip still gets made
        # in the engine's own voice, and holding a whole row over it would be
        # out of proportion.
        unspeakable, chosen_is_unusable = _unspeakable_patterns(resolved)
        if unspeakable:
            result.warnings.append(
                f"{record.id}: {', '.join(unspeakable)} cannot "
                f"be spoken for reading {resolved.reading}"
                + (
                    "; its word audio uses the engine's own accent."
                    if chosen_is_unusable
                    else f"; its word audio uses "
                    f"{pitch_module.select_pattern(resolved)}, which is fine."
                )
            )
        # `remint_blocked` means the set is *incomplete*, not wrong: an id in it
        # was positively proved present, and `remint` would leave that row alone
        # whatever the unreadable deck turns out to hold. Holding it would block
        # a run for a row nothing was ever uncertain about, under a reason that
        # is false for it.
        if (
            remint_blocked
            and record.id not in already_stored
            and record.id != stable_record_id(resolved.expression, resolved.reading)
        ):
            result.held.append(_hold(record, HOLD_UNVERIFIABLE_ID))
            result.keep.append(True)
            continue

        promoted = remint(resolved, already_stored)
        if promoted.id != record.id:
            result.reminted[record.id] = promoted.id
        result.promoted.append(promoted)
        result.keep.append(False)
    return result


def _resolved(record: VocabularyRecord) -> VocabularyRecord:
    """The record with its review annotations cleared.

    A row reaches this gate still carrying whatever the importer or the reading
    assistant wrote on it — ``hold_reason``, ``suggested_reading`` — because a
    reviewer fixes a hold by typing the reading in, not by tidying up the
    annotations, and this module re-tests rather than trusting them. But those
    annotations describe a row *under review*, and this one is about to stop
    being one. Left in place they would be copied into ``vocabulary.json``,
    where a merge keeps the first record's ``source`` forever and every reader
    that treats ``hold_reason`` as "still held" — ``status --staged``,
    ``enrich.needs_reading`` — would go on believing it.
    """
    return annotate(
        record, hold_reason=None, suggested_reading=None, already_known=None
    )


#: What a promote-time acceptance looks like once bound: comma-joined
#: 12-hex content fingerprints (whitespace around commas tolerated, because
#: every reader strips it). Anything else left in the key on an *extract* row
#: after ``_accept_examples`` ran is a value that accepts nothing.
_BOUND_AUTHORITY = re.compile(r"\s*[0-9a-f]{12}(\s*,\s*[0-9a-f]{12})*\s*")


def _accept_examples(record: VocabularyRecord) -> VocabularyRecord:
    """Bind the reviewer's explicit example acceptance to the sentences it saw.

    The acceptance is the reviewer *typing* ``example_authority:
    staging-review`` into the row. The localhost review panel takes the other
    explicit route and writes fingerprints for the exact displayed sentences
    directly. Neither authority is inferred from an example merely being
    present, because presence proves nothing about who wrote it: the large
    AI-enrichment route stages model-generated sentences on extract-type rows,
    and a pre-boundary staging file may still hold machine-copied excerpts.
    Promotion replaces only the manual sentinel with the same fingerprints;
    an already-bound panel decision stays bound to exactly what the reviewer
    saw. A row without either mark promotes its examples unstamped — preserved,
    but never pinned as curated. Non-extract sources are the user's own data
    and need no stamp at all.
    """
    if record.source.type != "extract":
        return record
    if record.source.raw_fields.get(EXAMPLE_AUTHORITY_KEY) != EXAMPLE_AUTHORITY_STAGING:
        # Absent, already fingerprint-bound (a re-promotion), or junk: nothing
        # to bind. An unrecognised value covers no real fingerprint, so it can
        # bless nothing by accident.
        return record
    # Replace-semantics on purpose: the binding states the complete accepted
    # set, and a sentinel with no sentences accepts nothing at all.
    return set_example_flags(
        record,
        EXAMPLE_AUTHORITY_KEY,
        [example.japanese for example in record.examples if example.japanese],
    )


def _structural_hold(record: VocabularyRecord) -> str | None:
    """The hold reason a record's own reading forces, if any.

    Both M1.5 classes, re-tested here rather than trusted from the annotation:
    a staging file is hand-edited, and the question at this gate is what the
    reading *is now*, not what an importer thought it was.
    """
    if not record.expression.strip():
        return "no expression"
    if not record.reading.strip():
        return HOLD_MISSING_READING
    if contains_kanji(record.reading):
        return HOLD_READING_KANJI
    return None


def _dictionary_verdict(
    client: jpdb.JpdbClient, record: VocabularyRecord
) -> tuple[str, str]:
    """``("pass" | "held", warning)`` for one record's reading."""
    # Bind the reviewed reading during tokenization. An unforced kana parse can
    # resolve a visually similar or normalized form to another dictionary
    # entry, which makes the reading check reject the value it was meant to
    # verify.
    readings = enrich.dictionary_readings(client, record.expression, record.reading)
    if readings is None:
        # jpdb has no opinion — it did not resolve the spelling to one entry.
        # That is not disagreement, and holding a record back because the
        # dictionary is silent would punish exactly the uncommon words a
        # textbook is most worth extracting.
        return "pass", (
            f"{record.id}: jpdb did not resolve {record.expression} to a single "
            "entry, so its reading could not be checked; promoted unchecked."
        )
    if record.reading == readings.primary:
        return "pass", ""
    if readings.supports_suru_suffix:
        # jpdb represents many ordinary Xする verbs as the X dictionary entry
        # with the `vs` part-of-speech marker. The exact stem spelling, stem
        # reading, suffix, and POS all have to agree; a loose "ends in する"
        # rule would approve an arbitrary phrase.
        return "pass", ""
    if record.reading in readings.all_readings:
        return "pass", (
            f"{record.id}: {record.expression} is usually read "
            f"{readings.primary}, and this record says {record.reading} — a "
            "reading jpdb does list for it, so it was promoted as a homograph."
        )
    listed = ", ".join(sorted(readings.all_readings)) or "nothing"
    return "held", (
        f"{record.id}: jpdb lists {listed} for {record.expression}, not "
        f"{record.reading}. Held back — the reading is half of the record ID, "
        "so a wrong one cannot be corrected later."
    )


def source_references(records: Iterable[VocabularyRecord]) -> list[tuple[str, str, str]]:
    """``(record id, source type, source ref)`` for the ledger, per record.

    Read off each record rather than fixed to ``"pdf"``: a staging file may hold
    extracted candidates, rows an importer held back, or M4.2's enrichment
    proposals, and every one of them already carries where it came from. Using
    the record's own ``source`` is also what keeps ``status --rebuild`` from
    appending a second, near-duplicate reference to everything promoted.
    """
    return [
        (record.id, record.source.type or "manual", record.source.imported_from)
        for record in records
    ]


def archive_meta(meta: dict[str, Any], promoted: int) -> dict[str, Any]:
    """The metadata the ``done/`` archive keeps from a promoted staging file.

    A reviewer's own ``review_notes`` is kept and the promoted count appended
    to it, never substituted for it: ``read_staging`` carries that key through
    a round trip precisely so a hand-written note survives, and on a fully
    promoted file the archive is the only copy left once the source is deleted.
    """
    kept = dict(meta)
    note = f"Promoted {promoted} record(s) from this file."
    existing = str(kept.get("review_notes") or "").strip()
    kept["review_notes"] = f"{existing}\n\n{note}" if existing else note
    return kept
