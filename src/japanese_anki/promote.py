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
from collections.abc import Container, Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from japanese_anki import enrich, extract, hardening, jpdb
from japanese_anki.errors import JankiError
from japanese_anki.identifiers import contains_kanji, stable_record_id
from japanese_anki.models import VocabularyRecord
from japanese_anki.staging import (
    HOLD_MISSING_READING,
    HOLD_READING_KANJI,
    HOLD_UNKNOWN_READING,
    HOLD_UNVERIFIABLE_ID,
    annotate,
    require_resolved_coverage,
)

__all__ = [
    "HOLD_MISSING_READING",
    "HOLD_READING_KANJI",
    "HOLD_UNKNOWN_READING",
    "HOLD_UNVERIFIABLE_ID",
    "PromoteError",
    "PromoteResult",
    "check_readings",
    "check_coverage",
    "remint",
]

# The hold vocabulary lives in `staging`, which owns the annotation key these
# are written under and is the one module both this and `enrich` can import —
# `enrich.needs_reading` has to tell a reading hold from an id hold, and this
# module already imports `enrich`. Re-exported here, because this is where they
# are written and where readers have always looked for them.


class PromoteError(JankiError):
    pass


def check_coverage(meta: dict[str, Any], root: Path | None = None) -> None:
    """Apply the coverage gate before promotion can read clients or write data."""
    try:
        require_resolved_coverage(meta)
    except JankiError as exc:
        raise PromoteError(str(exc)) from exc
    block = meta.get("coverage")
    if not isinstance(block, dict):
        return
    oracle_id = block.get("oracle_id")
    status = block.get("status")
    if status in {"matched", "mismatch"} and not oracle_id:
        raise PromoteError(
            "[coverage-oracle-missing] exhaustive coverage has no oracle ID"
        )
    if not oracle_id:
        _verify_coverage_facts(meta, block, None, None)
        return
    if root is None:
        raise PromoteError(
            "[coverage-oracle-unverified] repository root is needed to verify the "
            "coverage oracle"
        )
    if not isinstance(oracle_id, str) or not re.fullmatch(
        r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", oracle_id
    ):
        raise PromoteError("[coverage-oracle-invalid] coverage oracle ID is invalid")
    oracle_paths = [
        path
        for suffix in (".yaml", ".yml")
        if (path := root / "quality" / "oracles" / f"{oracle_id}{suffix}").exists()
    ]
    if len(oracle_paths) != 1:
        raise PromoteError(
            f"[coverage-oracle-missing] coverage oracle {oracle_id!r} must have "
            "exactly one .yaml or .yml file"
        )
    try:
        oracle = hardening.load_oracle_file(root, oracle_paths[0])
    except JankiError as exc:
        raise PromoteError(f"[coverage-oracle-invalid] {exc}") from exc
    expected_oracle_fingerprint = hardening.oracle_content_fingerprint(oracle)
    if not oracle.approved:
        raise PromoteError(
            f"[coverage-oracle-unapproved] coverage oracle {oracle.id!r} is a draft"
        )
    if (
        oracle.source_fingerprint != block.get("source_fingerprint")
        or oracle.type != block.get("oracle_type")
        or expected_oracle_fingerprint != block.get("oracle_content_fingerprint")
    ):
        raise PromoteError(
            "[coverage-oracle-stale] coverage does not match the approved oracle's "
            "source, type, and content fingerprint"
        )
    _verify_coverage_facts(meta, block, oracle, expected_oracle_fingerprint)


def _verify_coverage_facts(
    meta: dict[str, Any],
    block: dict[str, Any],
    oracle: hardening.UnitOracle | None,
    oracle_fingerprint: str | None,
) -> None:
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
        oracle=oracle,
        oracle_fingerprint=oracle_fingerprint,
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
            "[coverage-facts-stale] coverage facts do not match the stored source "
            "units and approved oracle"
        )


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

        resolved = _resolved(record)
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
