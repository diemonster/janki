"""What promoting a staging file would do, decided before anything is written.

`WORKBENCH_PLAN.md` W1.1b. Two kinds of thing live here.

**Promote's own orchestration**, lifted out of `cli.py` unchanged: which
durable archive a staging run belongs to and which of its rows are already
there (`archive_for_run`), whether a path is the archive itself
(`inside_archive`), and the gates that refuse a file before anything is
written (`validate_record_archive`, `staged_ai_enrichment`,
`check_pattern_review`). These are not previews — `promote` calls every one of
them — but they are the pieces both the command and the preview need, and a
second copy would answer differently the first time either changed.

`plan_promotion` is the preview. It answers "what would adding this source do"
without doing it: which cards are new, which merge into a card you already
have and therefore keep the meanings already on it, which are held back and
why, and which would be filed under a different identity than the one they
were staged with.

**It is honest about what it cannot know offline**, which `readings_unchecked`
records — see that field for what the gap is and why it is left open.

Nothing in `plan_promotion` writes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from japanese_anki import enrich, jpdb, ledger, patterns, promote, staging
from japanese_anki import status as status_module
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.io import load_records
from japanese_anki.models import VocabularyRecord
from japanese_anki.promote import PromoteError
from japanese_anki.staging import (
    STAGING_SUFFIXES,
    check_rewritable,
    read_staging,
    review_run_id,
    rich_extraction_review_run_id,
    validate_coverage_facts,
)

__all__ = [
    "AiLedgerHandoffIncomplete",
    "HeldCard",
    "LandingCard",
    "PromotionPlan",
    "archive_for_run",
    "archive_run_provenance",
    "check_pattern_review",
    "inside_archive",
    "plan_promotion",
    "unreadable_deck_warning",
    "staged_ai_enrichment",
    "validate_record_archive",
]


def archive_run_provenance(meta: Mapping[str, Any]) -> dict[str, Any]:
    """The model-run identity that decides whether an archive may be appended.

    New model-review files carry a persisted ``review_run_id`` as well as their
    request provenance. Partial promotion retries carry both byte-for-byte
    through the live staging file. A later invocation may reuse the staging
    basename and even the exact request, but its run id is different and its
    rows must not be folded into the earlier archive. Schema-v2 extraction
    staging is intentionally included: without a run id, its smaller
    ``prompt_provenance`` block remains the identity that version knew to record.

    Files older than model provenance retain their historical basename-based
    retry behavior through the explicit ``legacy`` identity.
    """
    run_id = review_run_id(meta)
    if "ai_enrichment" in meta:
        identity = {
            "kind": "ai",
            "ai_enrichment": meta.get("ai_enrichment"),
            "field_replacements": meta.get("field_replacements"),
        }
    elif "prompt_provenance" in meta:
        identity = {
            "kind": "extract",
            "prompt_provenance": meta.get("prompt_provenance"),
        }
    else:
        identity = {"kind": "legacy"}
    if run_id is not None:
        identity["review_run_id"] = run_id
    return identity


def archive_for_run(
    base: Path, meta: Mapping[str, Any]
) -> tuple[Path, list[VocabularyRecord], dict[str, Any] | None]:
    """Choose this run's deterministic archive and any partial rows already there."""
    identity = archive_run_provenance(meta)
    if not base.exists():
        return base, [], None

    previous, previous_meta = read_staging(base)
    if archive_run_provenance(previous_meta) == identity:
        return base, list(previous), previous_meta

    try:
        encoded = json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PromoteError(
            "This staging file has model-run provenance that cannot name a "
            f"durable archive: {exc}. Nothing was promoted."
        ) from exc
    digest = hashlib.sha256(encoded).hexdigest()
    candidate = base.with_name(f"{base.stem}.{digest}{base.suffix}")
    if not candidate.exists():
        return candidate, [], None

    previous, previous_meta = read_staging(candidate)
    if archive_run_provenance(previous_meta) != identity:
        raise PromoteError(
            f"Archive provenance collision at {candidate}; nothing was promoted."
        )
    return candidate, list(previous), previous_meta



def inside_archive(path: Path, archive_dir: Path) -> bool:
    """Is ``path`` the promoted archive, or inside it?

    Identity where the filesystem can answer it, because a lexical comparison
    is wrong on a case-insensitive filesystem: ``staging/Done/lesson.yaml``
    opens the real archive while comparing unequal to ``staging/done/...``, and
    promoting it doubles a committed file that is the only copy of a finished
    review. The lexical test stays as the fallback for a path that does not
    exist yet.
    """
    try:
        if archive_dir.exists() and path.parent.samefile(archive_dir):
            return True
    except OSError:
        pass
    return path.is_relative_to(archive_dir)



def validate_record_archive(
    live_meta: Mapping[str, Any],
    archived: Sequence[VocabularyRecord],
    archived_meta: Mapping[str, Any] | None,
) -> None:
    """Prove a same-run record archive is intact before relying on it."""
    if archived_meta is None:
        if archived:
            raise PromoteError(
                "[record-archive-divergent] archive rows have no metadata"
            )
        return
    if not archived:
        # A zero-row same-run archive is still a completed review. In normal
        # operation it is the pattern-only archive, whose reviewed_pattern_set
        # exists only in the done copy. A later record-bearing live file must
        # not turn absence of rows into absence of an archive and overwrite it.
        # Exact pattern-only retries are routed to their stricter validator by
        # command_promote before the record-archive validator is called.
        raise PromoteError(
            "[record-archive-divergent] the same review run already has a "
            "completed zero-record archive; later record rows cannot be appended "
            "to it. Both the live review and done archive were kept."
        )

    # The done copy is durable paid evidence too. A retry may delete the only
    # live copy, so validate it independently rather than trusting the live
    # block or merely comparing the run-id subset used for archive selection.
    validate_coverage_facts(archived_meta)
    promote.check_coverage(dict(archived_meta))
    promote.check_candidate_accounting(
        live_meta, (), archived, archived_meta=archived_meta
    )

    live_core = {
        key: value for key, value in live_meta.items() if key != "review_notes"
    }
    archive_core = {
        key: value for key, value in archived_meta.items() if key != "review_notes"
    }
    if archive_core != live_core:
        raise PromoteError(
            "[record-archive-divergent] the same-run archive metadata differs "
            "from the live review"
        )
    note = archived_meta.get("review_notes")
    suffix = f"Promoted {len(archived)} record(s) from this file."
    if not isinstance(note, str) or not (
        note.strip() == suffix or note.strip().endswith(f"\n\n{suffix}")
    ):
        raise PromoteError(
            "[record-archive-divergent] the same-run archive does not record "
            "its exact archived row count"
        )


class AiLedgerHandoffIncomplete(Exception):
    """Keep the live review after records landed but AI attribution did not."""



def staged_ai_enrichment(
    meta: Mapping[str, Any],
    record_ids: Sequence[str],
    *,
    archived_ids: Sequence[str] = (),
) -> tuple[str, str, dict[str, tuple[str, tuple[str, ...]]]] | None:
    """Validate AI provenance before a staged review can write anything.

    A partial promote keeps the original top-level metadata while moving some
    rows to the corresponding ``done`` archive.  Provenance may therefore name
    a row absent from the current staging file only when that exact id is
    already in the archive.  An older, completed run under the same basename is
    the opposite shape: archive-only ids need not appear in this run's maps.
    """
    raw = meta.get("ai_enrichment")
    if raw is None:
        if "field_replacements" in meta:
            raise PromoteError(
                "This staging file authorizes field replacements but has no "
                "ai_enrichment provenance. Nothing was promoted."
            )
        return None
    version = raw.get("version") if isinstance(raw, Mapping) else None
    if (
        not isinstance(raw, Mapping)
        or isinstance(version, bool)
        or not isinstance(version, int)
        or version != 1
    ):
        raise PromoteError("ai_enrichment must be a version 1 metadata block")
    raw_model = raw.get("model")
    model = raw_model.strip() if isinstance(raw_model, str) else ""
    raw_provider = raw.get("provider")
    provider = raw_provider.strip() if isinstance(raw_provider, str) else ""
    requests = raw.get("request_fingerprints")
    inputs = raw.get("input_fingerprints")
    fields = raw.get("fields")
    if (
        not model
        or provider not in ledger.AI_PROVIDERS
        or not all(isinstance(value, Mapping) for value in (requests, inputs, fields))
    ):
        raise PromoteError(
            "ai_enrichment needs a provider, model, plus request, input, and field maps"
        )
    replacement_block = meta.get("field_replacements")
    replacement_records = (
        replacement_block.get("records")
        if isinstance(replacement_block, Mapping)
        else None
    )
    if not isinstance(replacement_records, Mapping):
        raise PromoteError(
            "ai_enrichment needs its field_replacements record map. Nothing "
            "was promoted."
        )

    maps = {
        "request_fingerprints": requests,
        "input_fingerprints": inputs,
        "fields": fields,
        "field_replacements": replacement_records,
    }
    id_sets: dict[str, set[str]] = {}
    for label, values in maps.items():
        if any(not isinstance(record_id, str) or not record_id for record_id in values):
            raise PromoteError(f"{label} record ids must be non-empty text")
        id_sets[label] = set(values)
    provenance_ids = id_sets["request_fingerprints"]
    if any(ids != provenance_ids for ids in id_sets.values()):
        raise PromoteError(
            "ai_enrichment request, input, field, and replacement maps must "
            "name identical record ids. Nothing was promoted."
        )

    current_ids = {str(record_id) for record_id in record_ids}
    known_ids = current_ids | {str(record_id) for record_id in archived_ids}
    missing = sorted(current_ids - provenance_ids)
    unknown = sorted(provenance_ids - known_ids)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing current " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise PromoteError(
            "ai_enrichment provenance does not match this staging file or its "
            "done archive: " + "; ".join(details) + ". Nothing was promoted."
        )

    proven: dict[str, tuple[str, tuple[str, ...]]] = {}
    for record_id in sorted(provenance_ids):
        request_fp = requests.get(record_id)
        input_fp = inputs.get(record_id)
        raw_fields = fields.get(record_id)
        replacement_fields = replacement_records.get(record_id)
        if (
            not isinstance(request_fp, str)
            or len(request_fp) != 64
            or any(character not in "0123456789abcdef" for character in request_fp)
            or not isinstance(input_fp, str)
            or len(input_fp) != 64
            or any(character not in "0123456789abcdef" for character in input_fp)
            or not isinstance(raw_fields, list)
            or not raw_fields
            or any(not isinstance(name, str) for name in raw_fields)
            or any(name not in enrich.AI_FIELDS for name in raw_fields)
            or len(set(raw_fields)) != len(raw_fields)
            or not isinstance(replacement_fields, Mapping)
            or any(not isinstance(name, str) for name in replacement_fields)
            or set(raw_fields) != set(replacement_fields)
        ):
            raise PromoteError(
                f"ai_enrichment has incomplete provenance for {record_id}"
            )
        proven[record_id] = (
            request_fp,
            tuple(dict.fromkeys(str(name) for name in raw_fields)),
        )
    return provider, model, proven

@dataclass(frozen=True, slots=True)
class LandingCard:
    """One staged card that would reach the collection, and how."""

    staged: VocabularyRecord
    #: The record as it would be stored: re-minted identity, merged fields.
    landing: VocabularyRecord
    #: The canonical record it would merge into, if there is one.
    existing: VocabularyRecord | None
    #: The id it was staged under, when promoting would change it. Empty
    #: otherwise. A changed id is a different Anki note — worth saying out
    #: loud before it happens rather than after.
    reminted_from: str = ""

    @property
    def is_new(self) -> bool:
        return self.existing is None

    @property
    def keeps_existing_meanings(self) -> bool:
        """Whether the collection's wording wins over this source's.

        The case people are surprised by. `promote` is existing-wins, so a
        word you already have keeps the meaning already on it and this
        lesson's wording becomes source evidence rather than card text.
        """
        if self.existing is None or not self.staged.meanings:
            return False
        return list(self.landing.meanings) != list(self.staged.meanings)


@dataclass(frozen=True, slots=True)
class HeldCard:
    """One staged card that would stay in the staging file, and why."""

    record: VocabularyRecord
    reason: str


@dataclass(frozen=True, slots=True)
class PromotionPlan:
    """What adding this source would do. Computed without writing anything."""

    source: str
    staging_path: Path
    #: A refusal that stops the whole promote before any row is considered —
    #: the coverage gate, an unparseable file. Empty when nothing blocks it.
    blocked: str = ""
    landing: tuple[LandingCard, ...] = ()
    held: tuple[HeldCard, ...] = ()
    #: Rows this run already wrote to its durable archive. Promoting again
    #: removes them from the live review rather than adding them twice.
    already_archived: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    #: Whether the dictionary witness was skipped — load-bearing rather than
    #: decorative.
    #:
    #: The check costs a paid lookup per row, so a plan asked without a client
    #: declines it, and then a row listed in `landing` may still be held when
    #: the real promote asks jpdb whether its reading is one a dictionary
    #: lists. Handed a client, the plan makes the call and this says so, which
    #: is how a caller tells a complete decision from a partial one.
    #:
    #: True on a plan that was blocked before the check could run, and on the
    #: early returns that never reach it — in all of which the witness really
    #: was not consulted.
    readings_unchecked: bool = True

    @property
    def is_blocked(self) -> bool:
        return bool(self.blocked)

    @property
    def adding(self) -> tuple[LandingCard, ...]:
        return tuple(card for card in self.landing if card.is_new)

    @property
    def merging(self) -> tuple[LandingCard, ...]:
        return tuple(card for card in self.landing if not card.is_new)

    @property
    def reminted(self) -> dict[str, str]:
        return {
            card.reminted_from: card.landing.id
            for card in self.landing
            if card.reminted_from
        }


def unreadable_deck_warning(problem: str) -> str:
    """Why a row needing a new id stays put when a deck will not parse.

    One sentence, one place. The command prints it and the plan carries it, and
    two copies of a sentence that explains a *refusal* drift into explaining
    two different refusals.
    """
    return (
        f"{problem}; ids in that deck cannot be checked, so any row needing a "
        "new id stays in the staging file until it parses."
    )


def check_pattern_review(
    config: ProjectConfig, meta: Mapping[str, Any], run_id: str
) -> None:
    """Refuse a zero-record rich extraction whose grammar nobody has reviewed.

    The completion contract for a file that proposed only patterns. Lifted out
    of `_complete_pattern_only_review`'s locked transaction so the preview can
    ask the same question without taking the lock or writing the archive — the
    checks themselves read files and decide, which is all either caller needs
    from them.
    """
    source = meta.get("source_file")
    raw_pattern_set = meta.get("pattern_set")
    if (
        not isinstance(source, str)
        or not source.strip()
        or not isinstance(raw_pattern_set, Mapping)
    ):
        raise PromoteError(
            "[pattern-review-invalid] a rich pattern-only extraction needs "
            "its source_file and pattern_set. Nothing was archived."
        )
    patterns.PatternSet.from_dict(source, dict(raw_pattern_set))
    stored_patterns = patterns.load_store(config.patterns_file).get(source)
    if stored_patterns is None or not stored_patterns.reviewed:
        raise PromoteError(
            f"[patterns-unreviewed] {source} has not been reviewed. Run "
            f"'janki patterns --review {source}' first; nothing was archived."
        )
    provenance = meta.get("prompt_provenance")
    if (
        stored_patterns.review_run_id != run_id
        or not isinstance(provenance, Mapping)
        or stored_patterns.prompt_provenance != dict(provenance)
    ):
        raise PromoteError(
            f"[patterns-review-stale] the reviewed {source} pattern-store "
            "entry belongs to a different extraction run. Nothing was archived."
        )


def _unrewritable(path: Path) -> str:
    return (
        f"{path} is not a staging file janki can rewrite: the promoted archive "
        f"is written under the same name, and that needs "
        f"{' or '.join(STAGING_SUFFIXES)}. Rename it and re-run."
    )


def plan_promotion(
    config: ProjectConfig,
    staging_path: Path,
    *,
    source: str = "",
    client: jpdb.JpdbClient | None = None,
    skip_reading_check: bool | None = None,
) -> PromotionPlan:
    """What `janki promote` would do to this staging file, without doing it.

    Every decision comes from a function promote itself calls, **in the order
    promote calls it**. The order matters even though nothing is written: it
    decides which refusal a person is shown first, and the command puts the
    structural gates ahead of coverage deliberately — sending someone to a
    coverage decision (whose other route is a paid completeness check) for a
    file no acceptance could make promotable is wasted work followed by the
    refusal they should have seen.

    One gap, named rather than hidden: see `PromotionPlan.readings_unchecked`.
    It is a gap only by default. Pass a `client` — as the command does, having
    already decided to spend the lookups — and the dictionary witness runs
    here too, `readings_unchecked` says so, and the plan becomes the whole
    decision rather than most of it. A page refresh passes nothing and gets
    the offline answer; there is one code path either way, which is the point.
    """
    # Handing over a client and still getting the offline answer is the shape
    # a caller would never intend, so it is not spellable by accident: the
    # check follows the client unless someone says otherwise, which is what
    # the command's own `--skip-reading-check` does.
    if skip_reading_check is None:
        skip_reading_check = client is None
    if not skip_reading_check and client is None:
        # Refused here, and unconditionally. `check_readings` raises for this
        # too, but from inside its per-record loop after the structural holds
        # — so a caller could plan a blocked or all-held file, see a plan come
        # back, and crash on the first clean row in production instead.
        raise PromoteError(
            "A reading check needs a dictionary client. Pass one, or leave "
            "skip_reading_check alone to get the offline plan."
        )

    staging_path = staging_path.resolve()
    name = source or staging_path.name

    # Mutable so the refusals reachable *after* the dictionary ran do not
    # report it as skipped. Money was spent by then, and a caller keying "this
    # preview may be incomplete" on the flag would say the wrong thing about a
    # plan that is complete as far as it got.
    consulted = False

    def blocked(reason: object) -> PromotionPlan:
        return PromotionPlan(
            source=name,
            staging_path=staging_path,
            blocked=str(reason),
            readings_unchecked=not consulted,
        )

    archive_base = (config.staging_dir / "done" / staging_path.name).resolve()
    if inside_archive(staging_path, archive_base.parent):
        return blocked(
            f"{staging_path} is inside the promoted archive. Those records are "
            "already in the collection; promoting the archive would only "
            "duplicate it."
        )

    # Every refusal below is one `promote` reaches before it writes a byte, so
    # each is reported rather than raised: "you cannot add this yet, and here
    # is why" is the answer a page needs, and a traceback from a view that
    # changes nothing is not.
    try:
        records, meta = read_staging(staging_path)
        validate_coverage_facts(meta)
        _done, archived, archived_meta = archive_for_run(archive_base, meta)
        if records or archived:
            validate_record_archive(meta, archived, archived_meta)
        retry_flags = promote.check_candidate_accounting(
            meta, records, archived, archived_meta=archived_meta
        )
        promote.check_coverage(meta)
    except JankiError as exc:
        return blocked(exc)

    if not records:
        # An empty live file beside this run's own archive is a completed
        # partial promotion, not a new review: the rows are already in the
        # collection, which is what `already_archived` says.
        if archived:
            return PromotionPlan(
                source=name,
                staging_path=staging_path,
                already_archived=tuple(record.id for record in archived),
            )
        run_id = rich_extraction_review_run_id(meta)
        if run_id is None:
            return PromotionPlan(source=name, staging_path=staging_path)
        # A zero-record rich extraction proposed grammar and nothing else. Its
        # completion contract is a different one, and it refuses a source
        # nobody has reviewed — without this the plan called such a file clean
        # and an Add button keyed on it invoked a promote that refuses.
        try:
            check_pattern_review(config, meta, run_id)
            check_rewritable(staging_path)
        except JankiError as exc:
            return blocked(exc)
        if staging_path.suffix.lower() not in STAGING_SUFFIXES:
            return blocked(_unrewritable(staging_path))
        return PromotionPlan(source=name, staging_path=staging_path)

    # After the zero-record branch, exactly as in the command: a legacy file
    # holding no records is completed without either of these ever being asked,
    # so checking them earlier refuses a file promote finishes cleanly.
    try:
        check_rewritable(staging_path)
    except JankiError as exc:
        return blocked(exc)
    if staging_path.suffix.lower() not in STAGING_SUFFIXES:
        return blocked(_unrewritable(staging_path))

    already = tuple(
        record.id
        for record, is_retry in zip(records, retry_flags, strict=True)
        if is_retry
    )
    work = [
        record
        for record, is_retry in zip(records, retry_flags, strict=True)
        if not is_retry
    ]
    if not work:
        return PromotionPlan(
            source=name, staging_path=staging_path, already_archived=already
        )

    try:
        # Refuses a `field_replacements` block whose provenance does not line
        # up with the rows it claims to have enriched. The merge's own binding
        # check never reads `ai_enrichment`, so nothing downstream catches it.
        staged_ai_enrichment(
            meta,
            [record.id for record in work],
            archived_ids=[record.id for record in archived],
        )
        existing = (
            load_records(config.normalized_file)
            if config.normalized_file.exists()
            else []
        )
        stored_ids, unreadable = status_module.surviving_ids(config, existing)
    except JankiError as exc:
        return blocked(exc)

    # The check and the flag that reports it travel together, so a caller
    # cannot get one without the other.
    consulted = not skip_reading_check
    result = promote.check_readings(
        work,
        client=client,
        skip_reading_check=skip_reading_check,
        already_stored=stored_ids,
        remint_blocked=bool(unreadable),
    )
    warnings = tuple(result.warnings) + tuple(
        unreadable_deck_warning(problem) for problem in unreadable
    )
    held = tuple(
        HeldCard(
            record=record,
            reason=staging.annotations(record).get("hold_reason", ""),
        )
        for record in result.held
    )

    try:
        # The *second* accounting call, the one the command makes after
        # `check_readings`. Not a repeat, and not only a check: its duplicate
        # test is over the ids rows would land *under*, so two rows whose
        # corrected readings mint the same id collide only here — and its
        # Its *return* is deliberately discarded, and that is checked rather
        # than assumed: it flags rows matching this run's archive, but the
        # first call already probes each row's resolved id **and** its stable
        # re-mint (`promote.check_candidate_accounting`), and `remint` mints
        # nothing else. So a row the second call would flag was excluded from
        # `work` by the first. A review suggested the plan could disagree with
        # the command here; building the state it named put the row in
        # `already_archived` before the second call ran.
        promote.check_candidate_accounting(
            meta, result.promoted, archived, archived_meta=archived_meta
        )
    except JankiError as exc:
        return blocked(exc)

    if not result.promoted:
        # Nothing would land, so the command returns before it reads the
        # ledger. Reading it here would block a source on a corrupt ledger that
        # promote never opens — telling someone their work is unusable when it
        # is not.
        return PromotionPlan(
            source=name,
            staging_path=staging_path,
            held=held,
            already_archived=already,
            warnings=warnings,
            readings_unchecked=skip_reading_check,
        )

    try:
        ledger.load(config.ledger_file)
        merged, _outcomes = promote.merge_staged_records(
            list(existing),
            list(result.promoted),
            dict(meta),
            # Every live non-retry row in the form it arrived in, held ones
            # included — the command passes the same. A reading hold narrows
            # what may land, not what review an old-value binding covers, so a
            # stale binding on a held row refuses the whole merge.
            validate_incoming=work,
        )
    except JankiError as exc:
        return blocked(exc)

    merged_by_id = {record.id: record for record in merged}
    by_id = {record.id: record for record in existing}
    landing: list[LandingCard] = []
    promoted_iter = iter(result.promoted)
    for original, stays in zip(work, result.keep, strict=True):
        if stays:
            continue
        transformed = next(promoted_iter)
        landing.append(
            LandingCard(
                staged=original,
                landing=merged_by_id[transformed.id],
                existing=by_id.get(transformed.id),
                # `check_readings` already decided this and says so; deriving
                # it again from the ids is a second copy of the same rule.
                reminted_from=(
                    original.id if original.id in result.reminted else ""
                ),
            )
        )

    return PromotionPlan(
        source=name,
        staging_path=staging_path,
        landing=tuple(landing),
        held=held,
        already_archived=already,
        warnings=warnings,
        readings_unchecked=skip_reading_check,
    )
