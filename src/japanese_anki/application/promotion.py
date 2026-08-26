"""The shared promotion decision and execution service.

`WORKBENCH_PLAN.md` W1.1b. Two kinds of thing live here.

**Promote's own orchestration**, lifted out of `cli.py`: which
durable archive a staging run belongs to and which of its rows are already
there (`archive_for_run`), whether a path is the archive itself
(`inside_archive`), and the gates that refuse a file before anything is
written (`validate_record_archive`, `staged_ai_enrichment`,
`check_pattern_review`). `execute_promotion` is the one mutation path for both
surfaces: it writes canonical records and ledger state, completes pattern-only
reviews, and holds the live/archive locks through archive and pruning.

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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from japanese_anki import enrich, jpdb, ledger, patterns, promote, staging
from japanese_anki import status as status_module
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.io import (
    DataError,
    MergeOutcome,
    RecordsRevision,
    exclusive_path_lock,
    load_records,
    read_bytes_bound,
    records_revision,
    save_records_json,
)
from japanese_anki.models import VocabularyRecord
from japanese_anki.promote import PromoteError
from japanese_anki.staging import (
    STAGING_SUFFIXES,
    check_rewritable,
    prune_staging_under_lock,
    read_staging,
    review_run_id,
    rewrite_staging_under_lock,
    rich_extraction_review_run_id,
    validate_coverage_facts,
    write_staging_under_lock,
)

__all__ = [
    "PromotionExecutionResult",
    "HeldCard",
    "LandingCard",
    "PromotionDecision",
    "PromotionPlan",
    "DECISION_STATES",
    "POST_READING_GATES",
    "archive_for_run",
    "archive_run_provenance",
    "check_pattern_review",
    "record_review_snapshot",
    "staging_wire",
    "inside_archive",
    "decide_promotion",
    "execute_promotion",
    "plan_promotion",
    "project_promotion",
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
    if staging.AI_ENRICHMENT_KEY in meta:
        identity = {
            "kind": "ai",
            staging.AI_ENRICHMENT_KEY: meta.get(staging.AI_ENRICHMENT_KEY),
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
        # execute_promotion before the record-archive validator is called.
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


class _AiLedgerHandoffIncomplete(Exception):
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
    raw = meta.get(staging.AI_ENRICHMENT_KEY)
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
class _PromotionRepositoryBinding:
    """The repository paths one decision was derived from."""

    root: Path
    normalized_file: Path
    deck_dir: Path
    ledger_file: Path
    staging_dir: Path
    patterns_file: Path


def _repository_binding(config: ProjectConfig) -> _PromotionRepositoryBinding:
    return _PromotionRepositoryBinding(
        root=config.root.resolve(),
        normalized_file=config.normalized_file.resolve(),
        deck_dir=config.deck_dir.resolve(),
        ledger_file=config.ledger_file.resolve(),
        staging_dir=config.staging_dir.resolve(),
        patterns_file=config.patterns_file.resolve(),
    )


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


def _pattern_only_archive_meta(
    meta: Mapping[str, Any], reviewed: patterns.PatternSet
) -> dict[str, Any]:
    """Keep the paid answer raw and add the corrected human-reviewed snapshot."""
    archived = dict(meta)
    archived["reviewed_pattern_set"] = reviewed.to_dict()
    note = "Reviewed pattern-only extraction; no records were promoted."
    existing = str(archived.get("review_notes") or "").strip()
    archived["review_notes"] = f"{existing}\n\n{note}" if existing else note
    return archived


def _complete_pattern_only_review(
    config: ProjectConfig,
    path: Path,
    done: Path,
    expected_meta: Mapping[str, Any],
    expected_wire: bytes,
    run_id: str,
) -> tuple[Path, bool]:
    """Archive one reviewed zero-record v3 run as a locked CAS transaction."""
    with exclusive_path_lock(path):
        current_wire = staging_wire(path)
        records, meta = read_staging(path)
        if current_wire != expected_wire or records or dict(meta) != dict(expected_meta):
            raise PromoteError(
                "[staging-review-stale] the live staging file changed while its "
                "pattern review was being completed. The replacement was kept; "
                "nothing was archived."
            )

        # Re-check inside the lock. The first check happened before the CAS so
        # coverage acceptance could run without holding a file lock across a
        # paid call; this one binds the archive to the bytes about to be removed.
        promote.check_coverage(meta)
        if rich_extraction_review_run_id(meta) != run_id:
            raise PromoteError(
                "[staging-review-stale] the rich extraction run changed. "
                "Nothing was archived."
            )

        source = meta.get("source_file")
        raw_pattern_set = meta.get("pattern_set")
        if not isinstance(source, str) or not source.strip() or not isinstance(
            raw_pattern_set, Mapping
        ):
            raise PromoteError(
                "[pattern-review-invalid] a rich pattern-only extraction needs "
                "its source_file and pattern_set. Nothing was archived."
            )
        # Structural parsing only. Human corrections belong in the store and
        # deliberately need not equal this immutable paid proposal.
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

        check_rewritable(path)
        if path.suffix.lower() not in STAGING_SUFFIXES:
            raise PromoteError(_unrewritable(path))
        archived_meta = _pattern_only_archive_meta(meta, stored_patterns)
        archive_base = done
        while True:
            selected, _archived, _archived_meta = archive_for_run(archive_base, meta)
            with exclusive_path_lock(selected):
                # Selection itself reads existing archives. Confirm after the
                # selected path is locked: another completion may have created
                # the base or candidate between those two operations.
                confirmed, _archived, _archived_meta = archive_for_run(
                    archive_base, meta
                )
                if confirmed != selected:
                    continue
                done = selected
                retried = done.exists()
                if retried:
                    archive_records, existing_meta = read_staging(done)
                    if archive_records or existing_meta != archived_meta:
                        raise PromoteError(
                            f"[pattern-archive-divergent] {done} already names this "
                            "review run but is not the exact completed archive. "
                            "Both files were kept; nothing was overwritten."
                        )
                else:
                    done.parent.mkdir(parents=True, exist_ok=True)
                    # `selected` is already locked. The ordinary writer would
                    # acquire this non-reentrant lock again and deadlock.
                    write_staging_under_lock(done, [], archived_meta)
                    archive_records, written_meta = read_staging(done)
                    if archive_records or written_meta != archived_meta:
                        raise PromoteError(
                            f"[pattern-archive-divergent] {done} did not read "
                            "back as the exact completed archive. The live "
                            "review was kept."
                        )

                # Last, while both the live staging path and selected done path
                # remain locked. Neither a failed archive write, a concurrent
                # forced extraction, nor a force-write to the verified archive
                # can remove the only recoverable review artifact in between.
                path.unlink()
                return done, retried


def _finish_record_review(
    path: Path,
    archive_base: Path,
    *,
    expected_wire: bytes,
    expected_meta: Mapping[str, Any],
    expected_archived: Sequence[VocabularyRecord],
    expected_archived_meta: Mapping[str, Any] | None,
    promoted: Sequence[VocabularyRecord],
    retry_records: Sequence[VocabularyRecord],
    keep: Sequence[bool],
    held: Sequence[VocabularyRecord],
    canonical_commit: Callable[[], None] | None = None,
) -> tuple[Path, int]:
    """Commit, archive, and retire rows as one live/done locked transaction.

    The optional canonical commit runs only after both snapshots have been
    revalidated and while the selected done path remains locked. This closes
    the window where a same-run zero-row completion could appear after
    preflight but before vocabulary and ledger writes.
    """
    if len(keep) - sum(keep) != len(promoted) + len(retry_records):
        raise PromoteError(
            "[record-promotion-invalid] row disposition does not match the "
            "archive transaction"
        )

    with exclusive_path_lock(path):
        if staging_wire(path) != expected_wire:
            raise PromoteError(
                "[staging-review-stale] the live staging file changed while "
                "promotion was completing. The replacement was kept."
            )
        current_records, current_meta = read_staging(path)
        if dict(current_meta) != dict(expected_meta) or len(current_records) != len(keep):
            raise PromoteError(
                "[staging-review-stale] the live staging review no longer "
                "matches the validated snapshot. The replacement was kept."
            )

        validate_coverage_facts(current_meta)
        promote.check_coverage(current_meta)
        while True:
            selected, _rows, _meta = archive_for_run(archive_base, current_meta)
            with exclusive_path_lock(selected):
                confirmed, archived, archived_meta = archive_for_run(
                    archive_base, current_meta
                )
                if confirmed != selected:
                    continue
                if list(archived) != list(expected_archived) or (
                    None if archived_meta is None else dict(archived_meta)
                ) != (
                    None
                    if expected_archived_meta is None
                    else dict(expected_archived_meta)
                ):
                    raise PromoteError(
                        "[record-archive-stale] the done archive changed while "
                        "promotion was completing. The live review was kept."
                    )
                validate_record_archive(current_meta, archived, archived_meta)
                retry_flags = promote.check_candidate_accounting(
                    current_meta,
                    retry_records,
                    archived,
                    archived_meta=archived_meta,
                )
                if retry_records and not all(retry_flags):
                    raise PromoteError(
                        "[archive-retry-divergent] rows marked as archive retries "
                        "are not exact promoted rows in the done archive"
                    )
                if any(
                    promote.check_candidate_accounting(
                        current_meta,
                        promoted,
                        archived,
                        archived_meta=archived_meta,
                    )
                ):
                    raise PromoteError(
                        "[archive-retry-divergent] a pending row already exists "
                        "in the done archive"
                    )

                if canonical_commit is not None:
                    canonical_commit()

                done = confirmed
                combined = list(archived) + list(promoted)
                if promoted:
                    done.parent.mkdir(parents=True, exist_ok=True)
                    completed_meta = promote.archive_meta(
                        dict(current_meta), len(combined)
                    )
                    write_staging_under_lock(
                        done, combined, completed_meta, force=True
                    )
                    written, written_meta = read_staging(done)
                    if written != combined or written_meta != completed_meta:
                        raise PromoteError(
                            f"[record-archive-divergent] {done} did not read back "
                            "as the exact completed archive. The live review was kept."
                        )

                removed = prune_staging_under_lock(path, keep)
                if held:
                    rewrite_staging_under_lock(path, held)
                else:
                    path.unlink()
                return done, removed


#: What a decide pass concluded, in one word.
#:
#: `blocked` — a gate refused; `error` carries it.
#: `nothing` — a legacy file holding no records; promote says so and exits 0.
#: `archive_retry` — an empty live file beside this run's own archive, or a
#: file whose every row is already in it: completion, not a new review.
#: `pattern_only` — a zero-record rich extraction whose grammar was reviewed.
#: `nothing_lands` — rows exist and every one is held back.
#: `lands` — rows would reach the collection.
#: Gates reached only *after* the readings pass, so what they refuse depends
#: on which rows survived it.
#:
#: An offline decision cannot settle these: it promotes every row a dictionary
#: would have held, so it judges a superset — and two rows can collide on one
#: landing id offline that a consulted run would never have brought together.
#: A caller that decides offline first has to come back with the client before
#: believing a refusal from here.
POST_READING_GATES = frozenset({"accounting", "merge", "ledger"})

DECISION_STATES = (
    "blocked",
    "nothing",
    "archive_retry",
    "pattern_only",
    "nothing_lands",
    "lands",
)


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    """Everything `promote` works out before it writes a byte.

    The shared middle of the command and the preview. `plan_promotion`
    projects it into something safe to render; `execute_promotion` consumes it
    and performs the transaction for either surface.

    Why a projection rather than one public value: this carries the raw wire
    snapshot, the resolved merge and the compare-and-swap token — which is
    everything a caller would need to replay the write path *around* the gates
    that produced them. `docs/DESIGN.md` says the workbench may not weaken an
    authority gate the CLI enforces, so the boundary between this and
    `PromotionPlan` is a safety boundary and not a matter of taste.

    Constructing this decision writes nothing. The staging snapshot — the
    bytes and the parse — is taken under the writer lock, because the execution
    service's compare-and-swap trusts the two to describe each other. Nothing
    else is locked: the compare-and-swap is the protection for the collection,
    and holding a lock across the dictionary lookups would be worse than the
    race it closed.
    """

    source: str
    staging_path: Path
    state: str
    repository: _PromotionRepositoryBinding
    reading_check: Literal["preview", "explicit_skip", "consulted"]
    #: The refusal itself, not its text. The command re-raises it, so stderr
    #: and the exit code are what they always were, and its *type* survives —
    #: different gates raise different classes, and flattening them changes
    #: which handler catches a refusal.
    error: JankiError | None = None
    #: Which gate refused, when one did.
    #:
    #: Its own field because the type cannot answer this: `check_coverage`
    #: re-wraps whatever it catches into a plain `PromoteError`, so a coverage
    #: refusal, an accounting refusal and an archive-divergence refusal are
    #: indistinguishable by class. The only other discriminator is the
    #: `[coverage-unresolved]`-style tag in the message, and deciding whether
    #: to spend a model call by matching strings is exactly the fragility
    #: `--accept-coverage` must not rest on.
    gate: str = ""
    #: Whether the dictionary was actually consulted before this concluded.
    consulted: bool = False

    #: The staging file exactly as read, for the write path's compare-and-swap.
    wire: bytes = b""
    meta: Mapping[str, Any] = field(default_factory=dict)
    records: tuple[VocabularyRecord, ...] = ()

    #: This run's durable archive, its rows, and its metadata.
    done: Path | None = None
    archived: tuple[VocabularyRecord, ...] = ()
    archived_meta: Mapping[str, Any] | None = None
    #: One flag per input row: already in this run's archive.
    retry_flags: tuple[bool, ...] = ()
    #: The ids this run has already archived — computed once, here, because
    #: two shapes reach `archive_retry` (an empty live file beside a full
    #: archive, and a full file whose every row is already in it) and they
    #: read it from different places. Deriving it in the projection *and*
    #: again in the write path is two chances to disagree about what already
    #: landed.
    already_archived: tuple[str, ...] = ()

    #: The rows this run may still act on, and what the readings pass decided.
    work: tuple[VocabularyRecord, ...] = ()
    readings: promote.PromoteResult | None = None
    #: One flag per promoted row, from the accounting call made *after* the
    #: readings pass. Carried rather than assumed: the first call already
    #: probes each row's resolved id and its stable re-mint, so these are
    #: provably all false by the time they exist — but the write path composes
    #: its prune flags from them, and "provably" is not a thing to build a
    #: staging prune on.
    promoted_retry_flags: tuple[bool, ...] = ()

    #: The collection as it was read, and the token proving it has not moved.
    existing: tuple[VocabularyRecord, ...] = ()
    output_path: Path | None = None
    output_revision: RecordsRevision | None = None
    #: Decks whose ids could not be read, verbatim — the two warning streams
    #: stay separate because the command prints these *before* the readings
    #: warnings and the plan reports them after.
    unreadable_decks: tuple[str, ...] = ()

    ai_provenance: Any = None
    merged: tuple[VocabularyRecord, ...] = ()
    outcomes: Mapping[str, MergeOutcome] = field(default_factory=dict)

    @property
    def is_blocked(self) -> bool:
        return self.state == "blocked"

    @property
    def reading_warnings(self) -> tuple[str, ...]:
        return tuple(self.readings.warnings) if self.readings else ()

    @property
    def deck_warnings(self) -> tuple[str, ...]:
        return tuple(unreadable_deck_warning(one) for one in self.unreadable_decks)


PromotionExecutionState = Literal[
    "nothing",
    "pattern_only",
    "archive_retry",
    "nothing_lands",
    "landed",
    "landed_ledger_incomplete",
    "landed_ai_ledger_incomplete",
]


@dataclass(frozen=True, slots=True)
class PromotionExecutionResult:
    """The durable result of consuming one validated promotion decision.

    It contains facts, not terminal prose. The CLI formats these fields while
    the workbench can render the same committed transaction without parsing a
    command transcript.
    """

    state: PromotionExecutionState
    staging_path: Path
    archive_path: Path | None = None
    output_path: Path | None = None
    promoted: tuple[VocabularyRecord, ...] = ()
    held: tuple[VocabularyRecord, ...] = ()
    retry_records: tuple[VocabularyRecord, ...] = ()
    removed: int = 0
    empty_live_retry: bool = False
    archive_was_retry: bool = False
    reminted: Mapping[str, str] = field(default_factory=dict)
    outcomes: Mapping[str, MergeOutcome] = field(default_factory=dict)
    ledger_added: int = 0
    ledger_sources: int = 0
    ledger_error: ledger.LedgerError | None = None

    @property
    def promoted_ids(self) -> tuple[str, ...]:
        """The exact newly landed scope, in transaction order."""
        return tuple(record.id for record in self.promoted)

    def __post_init__(self) -> None:
        archive_required = self.state in {
            "pattern_only",
            "archive_retry",
            "landed",
            "landed_ledger_incomplete",
        } or (self.state == "nothing_lands" and bool(self.retry_records))
        if (self.archive_path is not None) != archive_required:
            raise ValueError(
                f"Promotion result {self.state!r} has an inconsistent archive path"
            )

        landed = self.state in {
            "landed",
            "landed_ledger_incomplete",
            "landed_ai_ledger_incomplete",
        }
        output_expected = landed or self.state == "nothing_lands"
        if landed != bool(self.promoted) or output_expected != (
            self.output_path is not None
        ):
            raise ValueError(
                f"Promotion result {self.state!r} has an inconsistent landing scope"
            )

        ledger_failed = self.state in {
            "landed_ledger_incomplete",
            "landed_ai_ledger_incomplete",
        }
        if (self.ledger_error is not None) != ledger_failed:
            raise ValueError(
                f"Promotion result {self.state!r} has an inconsistent ledger result"
            )


def staging_wire(path: Path) -> bytes:
    """Read the exact live-review bytes used by pattern-only promotion's CAS."""
    try:
        return read_bytes_bound(path)
    except (DataError, OSError) as exc:
        raise PromoteError(
            f"[staging-review-stale] could not snapshot {path}: "
            f"{getattr(exc, 'strerror', None) or exc}. Nothing was archived."
        ) from exc



def record_review_snapshot(
    path: Path,
) -> tuple[bytes, list[VocabularyRecord], dict[str, Any]]:
    """Read one exact live-review snapshot while its writer lock is held."""
    with exclusive_path_lock(path):
        wire = staging_wire(path)
        try:
            text = wire.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise DataError(f"Could not parse staging file {path}: {exc}") from exc
        records, meta = staging.read_staging_text(text, source=str(path))
    return wire, records, meta



def _unrewritable(path: Path) -> str:
    return (
        f"{path} is not a staging file janki can rewrite: the promoted archive "
        f"is written under the same name, and that needs "
        f"{' or '.join(STAGING_SUFFIXES)}. Rename it and re-run."
    )


def decide_promotion(
    config: ProjectConfig,
    staging_path: Path,
    *,
    source: str = "",
    client: jpdb.JpdbClient | None = None,
    skip_reading_check: bool | None = None,
) -> PromotionDecision:
    """Work out everything `janki promote` decides, and write nothing.

    Every decision comes from a function promote itself calls, **in the order
    promote calls it**. The order matters even though nothing is written: it
    decides which refusal a person is shown first, and the structural gates go
    ahead of coverage deliberately — sending someone to a coverage decision
    (whose other route is a paid completeness check) for a file no acceptance
    could make promotable is wasted work followed by the refusal they should
    have seen.

    **Gate refusals are returned, not raised** — `decision.error` holds the
    exception and `decision.gate` names what refused. The command re-raises it
    so its stderr and exit code are unchanged; a page reports it. A caller
    contract violation still raises, because a bad call is not a fact about
    the staging file and rendering it as "this source cannot be added" would
    say the wrong thing about somebody's work.

    `--accept-coverage` is deliberately outside this: it spends a model call
    and writes an approval into the staging file, so the command runs it
    between two decide passes rather than having a read-only function do it.
    """
    requested_skip = skip_reading_check
    if skip_reading_check is None:
        # Handing over a client and still getting the offline answer is the
        # shape a caller would never intend, so it is not spellable by
        # accident: the check follows the client unless someone says
        # otherwise, which is what `--skip-reading-check` does.
        skip_reading_check = client is None
    if not skip_reading_check and client is None:
        # Refused here, and unconditionally. `check_readings` refuses this too,
        # but from inside its per-record loop after the structural holds — so a
        # caller could decide a blocked or all-held file, see a decision come
        # back, and crash on the first clean row in production instead.
        raise PromoteError(
            "A reading check needs a dictionary client. Pass one, or leave "
            "skip_reading_check alone to get the offline plan."
        )

    reading_check: Literal["preview", "explicit_skip", "consulted"] = (
        "explicit_skip" if requested_skip is True else "preview"
    )
    repository = _repository_binding(config)
    staging_path = staging_path.resolve()
    name = source or staging_path.name
    consulted = False
    # What has been read so far. A refusal carries it too: `--accept-coverage`
    # has to show the model the coverage block of the very file the gate
    # refused, and a blocked decision that dropped its snapshot left the
    # acceptance step with nothing to read — reporting "carries no coverage
    # block" about a file whose block is right there.
    snapshot: dict[str, Any] = {}

    def at(state: str, **fields: Any) -> PromotionDecision:
        # Enforced, not documented: `at` took any string, so a mislabelled
        # state was invisible until Step 2 dispatched on it.
        assert state in DECISION_STATES, state
        return PromotionDecision(
            source=name, staging_path=staging_path, state=state,
            repository=repository, reading_check=reading_check,
            consulted=consulted, **fields,
        )

    def blocked(reason: JankiError, gate: str) -> PromotionDecision:
        return at("blocked", error=reason, gate=gate, **snapshot)

    archive_base = (config.staging_dir / "done" / staging_path.name).resolve()
    if inside_archive(staging_path, archive_base.parent):
        return blocked(
            PromoteError(
                f"{staging_path} is inside the promoted archive. Those records "
                "are already in the collection; promoting the archive would "
                "only duplicate it."
            ),
            "archive",
        )

    try:
        # Both reads under the writer lock, so the bytes the write path's
        # compare-and-swap trusts provably describe the records the gates ran
        # on. Two unlocked reads leave a window where a write lands between
        # them: the benign direction fails safe at write time, but an
        # A-then-back-to-A edit defeats the check entirely, and the secondary
        # revalidation compares only metadata and a row count.
        #
        # It also wraps the read. `read_bytes` raises `OSError`, which is not
        # a `JankiError` and sailed straight out of this function — so a
        # staging file deleted between listing a page and previewing it
        # produced a traceback where the old code returned a refusal.
        wire, records, meta = record_review_snapshot(staging_path)
        snapshot.update(wire=wire, records=tuple(records), meta=meta)
        validate_coverage_facts(meta)
        done, archived, archived_meta = archive_for_run(archive_base, meta)
        snapshot.update(
            done=done, archived=tuple(archived), archived_meta=archived_meta
        )
        if records or archived:
            validate_record_archive(meta, archived, archived_meta)
        retry_flags = promote.check_candidate_accounting(
            meta, records, archived, archived_meta=archived_meta
        )
    except JankiError as exc:
        return blocked(exc, "structure")
    try:
        promote.check_coverage(meta)
    except JankiError as exc:
        # Named separately because this is the one `--accept-coverage` can
        # answer. Every other refusal above means the model call would be
        # spent on a file no acceptance could make promotable.
        return blocked(exc, "coverage")

    common = {
        "wire": wire, "meta": meta, "records": tuple(records),
        "done": done, "archived": tuple(archived), "archived_meta": archived_meta,
        "retry_flags": tuple(retry_flags),
        "already_archived": (
            tuple(record.id for record in archived)
            if not records
            else tuple(
                record.id
                for record, is_retry in zip(records, retry_flags, strict=True)
                if is_retry
            )
        ),
    }

    if not records:
        if archived:
            # An empty live file beside this run's own archive is a completed
            # partial promotion, not a new review.
            return at("archive_retry", **common)
        run_id = rich_extraction_review_run_id(meta)
        if run_id is None:
            return at("nothing", **common)
        # A zero-record rich extraction proposed grammar and nothing else. Its
        # completion contract is a different one, and it refuses a source
        # nobody has reviewed.
        try:
            check_pattern_review(config, meta, run_id)
        except JankiError as exc:
            return blocked(exc, "patterns")
        try:
            check_rewritable(staging_path)
        except JankiError as exc:
            return blocked(exc, "rewritable")
        if staging_path.suffix.lower() not in STAGING_SUFFIXES:
            return blocked(PromoteError(_unrewritable(staging_path)), "rewritable")
        return at("pattern_only", **common)

    # After the zero-record branch, exactly as in the command: a legacy file
    # holding no records is completed without either of these being asked, so
    # checking them earlier refuses a file promote finishes cleanly.
    try:
        check_rewritable(staging_path)
    except JankiError as exc:
        return blocked(exc, "rewritable")
    if staging_path.suffix.lower() not in STAGING_SUFFIXES:
        return blocked(PromoteError(_unrewritable(staging_path)), "rewritable")

    work = [
        record
        for record, is_retry in zip(records, retry_flags, strict=True)
        if not is_retry
    ]
    if not work:
        return at("archive_retry", **common)

    try:
        # Refuses a `field_replacements` block whose provenance does not line
        # up with the rows it claims to have enriched. The merge's own binding
        # check never reads `ai_enrichment`, so nothing downstream catches it.
        ai_provenance = staged_ai_enrichment(
            meta,
            [record.id for record in work],
            archived_ids=[record.id for record in archived],
        )
        output_path = config.normalized_file.resolve()
        # Captured in the same pass that reads the records the decision is
        # made from. Split apart, the token races the read it is meant to
        # prove unchanged.
        output_revision = records_revision(output_path)
        existing = load_records(output_path) if output_path.exists() else []
        stored_ids, unreadable = status_module.surviving_ids(config, existing)
    except JankiError as exc:
        return blocked(exc, "collection")

    consulted = not skip_reading_check
    if consulted:
        reading_check = "consulted"
    readings = promote.check_readings(
        work,
        client=client,
        skip_reading_check=skip_reading_check,
        already_stored=stored_ids,
        remint_blocked=bool(unreadable),
    )
    common |= {
        "work": tuple(work), "readings": readings,
        "existing": tuple(existing), "output_path": output_path,
        "output_revision": output_revision,
        "unreadable_decks": tuple(unreadable),
        "ai_provenance": ai_provenance,
    }

    try:
        # The *second* accounting call, the one made after `check_readings`.
        # Not a repeat: its duplicate test is over the ids rows would land
        # *under*, so two rows whose corrected readings mint the same id
        # collide only here.
        #
        # Its answer is carried rather than discarded. The first call already
        # probes each row's resolved id **and** its stable re-mint, and
        # `remint` mints nothing else, so a row the second call would flag was
        # excluded from `work` by the first — the plan has a test pinning
        # exactly that. The write path still composes its prune flags from
        # these, because a staging prune is not a place to spend an invariant.
        promoted_retry = promote.check_candidate_accounting(
            meta, readings.promoted, archived, archived_meta=archived_meta
        )
    except JankiError as exc:
        return blocked(exc, "accounting")

    common["promoted_retry_flags"] = tuple(promoted_retry)

    if not readings.promoted:
        # Nothing would land, so promote returns before it reads the ledger.
        # Reading it here would block a source on a corrupt ledger promote
        # never opens — telling someone their work is unusable when it is not.
        return at("nothing_lands", **common)

    try:
        ledger.load(config.ledger_file)
    except JankiError as exc:
        return blocked(exc, "ledger")
    try:
        merged, outcomes = promote.merge_staged_records(
            list(existing),
            list(readings.promoted),
            dict(meta),
            # Every live non-retry row in the form it arrived in, held ones
            # included. A reading hold narrows what may land, not what review
            # an old-value binding covers, so a stale binding on a held row
            # refuses the whole merge.
            validate_incoming=work,
        )
    except JankiError as exc:
        return blocked(exc, "merge")

    return at("lands", **common, merged=tuple(merged), outcomes=outcomes)


def _save_execution_ledger(book: ledger.Ledger) -> ledger.LedgerError | None:
    """Save the promotion ledger without hiding already-landed records."""
    try:
        book.save()
    except ledger.LedgerError as exc:
        return exc
    return None


def execute_promotion(
    config: ProjectConfig, decision: PromotionDecision
) -> PromotionExecutionResult:
    """Consume a validated decision through the one promotion transaction.

    All mutation formerly in ``cli.command_promote`` lives here: canonical and
    ledger writes, exact archive retries, the live/archive CAS, held-row
    rewriting, and pattern-only completion. A caller may format the returned
    facts differently; it may not reproduce this writer.
    """
    if decision.repository != _repository_binding(config):
        raise PromoteError(
            "[promotion-config-mismatch] this promotion decision belongs to a "
            "different repository configuration. Nothing was promoted."
        )
    if decision.is_blocked:
        if decision.error is None:
            raise PromoteError("A blocked promotion decision has no refusal")
        raise decision.error

    path = decision.staging_path
    archive_base = (config.staging_dir / "done" / path.name).resolve()
    meta = decision.meta
    archived = decision.archived
    archived_meta = decision.archived_meta
    expected_wire = decision.wire

    if decision.state == "nothing":
        return PromotionExecutionResult(state="nothing", staging_path=path)

    if decision.state == "pattern_only":
        run_id = rich_extraction_review_run_id(meta)
        if run_id is None:
            raise PromoteError(
                "A pattern-only promotion decision has no rich extraction run"
            )
        done, retried = _complete_pattern_only_review(
            config, path, archive_base, meta, expected_wire, run_id
        )
        return PromotionExecutionResult(
            state="pattern_only",
            staging_path=path,
            archive_path=done,
            archive_was_retry=retried,
        )

    if decision.state == "archive_retry":
        # The archive is written before the live review is pruned and deleted.
        # A crash after the prune leaves an empty extraction file; it is
        # completion evidence, not a new pattern-only review.
        empty_live = not decision.records
        done, removed = _finish_record_review(
            path,
            archive_base,
            expected_wire=expected_wire,
            expected_meta=meta,
            expected_archived=archived,
            expected_archived_meta=archived_meta,
            promoted=(),
            retry_records=() if empty_live else list(decision.records),
            keep=() if empty_live else [False] * len(decision.records),
            held=(),
        )
        return PromotionExecutionResult(
            state="archive_retry",
            staging_path=path,
            archive_path=done,
            retry_records=() if empty_live else decision.records,
            removed=removed,
            empty_live_retry=empty_live,
        )

    if decision.reading_check == "preview":
        raise PromoteError(
            "[reading-check-required] an offline promotion preview cannot be "
            "executed. Consult jpdb or explicitly authorize skipping the "
            "reading check. Nothing was promoted."
        )

    result = decision.readings
    if result is None:
        raise PromoteError(
            f"Promotion decision {decision.state!r} has no reading disposition"
        )

    records = decision.records
    raw_archive_retry = decision.retry_flags
    work_records = list(decision.work)
    existing = list(decision.existing)
    output_path = decision.output_path
    output_revision = decision.output_revision
    ai_provenance = decision.ai_provenance
    if output_path is None or output_revision is None:
        raise PromoteError(
            f"Promotion decision {decision.state!r} has no collection snapshot"
        )

    retry_records: list[VocabularyRecord] = [
        record
        for record, is_retry in zip(records, raw_archive_retry, strict=True)
        if is_retry
    ]
    exact_promoted = decision.promoted_retry_flags
    pending_promoted = [
        record
        for record, is_retry in zip(result.promoted, exact_promoted, strict=True)
        if not is_retry
    ]
    retry_records.extend(
        record
        for record, is_retry in zip(result.promoted, exact_promoted, strict=True)
        if is_retry
    )
    promoted_flags = iter(exact_promoted)
    retry_by_work = [
        False if stays else next(promoted_flags) for stays in result.keep
    ]
    work_keep = [
        stays and not is_retry
        for stays, is_retry in zip(result.keep, retry_by_work, strict=True)
    ]
    work_keep_iter = iter(work_keep)
    keep = [
        False if is_retry else next(work_keep_iter)
        for is_retry in raw_archive_retry
    ]
    pending_records = [
        record
        for record, is_retry in zip(work_records, retry_by_work, strict=True)
        if not is_retry
    ]
    pending_reminted: dict[str, str] = {}
    promoted_iter = iter(result.promoted)
    retry_iter = iter(exact_promoted)
    for original, stays in zip(work_records, result.keep, strict=True):
        if stays:
            continue
        transformed = next(promoted_iter)
        is_retry = next(retry_iter)
        if not is_retry and transformed.id != original.id:
            pending_reminted[original.id] = transformed.id

    if not pending_promoted:
        # Held reasons still land in the live review; exact retries are pruned.
        done, removed = _finish_record_review(
            path,
            archive_base,
            expected_wire=expected_wire,
            expected_meta=meta,
            expected_archived=archived,
            expected_archived_meta=archived_meta,
            promoted=(),
            retry_records=retry_records,
            keep=keep,
            held=result.held,
        )
        return PromotionExecutionResult(
            state="nothing_lands",
            staging_path=path,
            archive_path=done if retry_records else None,
            output_path=output_path,
            held=tuple(result.held),
            retry_records=tuple(retry_records),
            removed=removed,
        )

    # Deciding proved the ledger parses; load it here for the object this
    # transaction will mutate and attempt to save.
    book = ledger.load(config.ledger_file)
    merged, outcomes = promote.merge_staged_records(
        existing,
        pending_promoted,
        dict(meta),
        validate_incoming=pending_records,
    )
    already_landed_fields = (
        promote.already_landed_staged_fields(existing, pending_promoted, meta)
        if ai_provenance is not None
        else {}
    )

    added = sum(
        book.record_added(record_id)
        for record_id, outcome in outcomes.items()
        if outcome.label == "added"
    )
    seen = sum(
        book.record_source_seen(record_id, source_type, source_ref)
        for record_id, source_type, source_ref in promote.source_references(
            pending_promoted
        )
    )
    if ai_provenance is not None:
        ai_provider, ai_model, provenance = ai_provenance
        for record_id, outcome in outcomes.items():
            request_fp, proposed_fields = provenance[record_id]
            written_fields = (
                proposed_fields
                if outcome.label == "added"
                else tuple(
                    name
                    for name in proposed_fields
                    if name in outcome.filled_fields
                    or name in already_landed_fields.get(record_id, ())
                )
            )
            if written_fields:
                book.record_enriched(
                    record_id,
                    kind="ai",
                    model=ai_model,
                    provider=ai_provider,
                    fields=written_fields,
                    request_fingerprint=request_fp,
                )
    ledger_error: ledger.LedgerError | None = None

    def commit_canonical_state() -> None:
        nonlocal ledger_error
        save_records_json(output_path, merged, expected=output_revision)
        ledger_error = _save_execution_ledger(book)
        if ledger_error is not None and ai_provenance is not None:
            # The reviewed proposal remains the only recoverable attribution.
            # Raising before archive/prune keeps it live while the outer locks
            # still guarantee no competing completion changed either copy.
            raise _AiLedgerHandoffIncomplete

    # Canonical writes, archive append, and live pruning/deletion share the
    # same live/done transaction. A same-run zero-row completion that wins the
    # lock is refused before `commit_canonical_state`; one that loses cannot
    # appear between validation and the canonical writes.
    try:
        done, removed = _finish_record_review(
            path,
            archive_base,
            expected_wire=expected_wire,
            expected_meta=meta,
            expected_archived=archived,
            expected_archived_meta=archived_meta,
            promoted=pending_promoted,
            retry_records=retry_records,
            keep=keep,
            held=result.held,
            canonical_commit=commit_canonical_state,
        )
    except _AiLedgerHandoffIncomplete:
        return PromotionExecutionResult(
            state="landed_ai_ledger_incomplete",
            staging_path=path,
            output_path=output_path,
            promoted=tuple(pending_promoted),
            held=tuple(result.held),
            retry_records=tuple(retry_records),
            reminted=pending_reminted,
            outcomes=dict(outcomes),
            ledger_added=added,
            ledger_sources=seen,
            ledger_error=ledger_error,
        )

    return PromotionExecutionResult(
        state=("landed" if ledger_error is None else "landed_ledger_incomplete"),
        staging_path=path,
        archive_path=done,
        output_path=output_path,
        promoted=tuple(pending_promoted),
        held=tuple(result.held),
        retry_records=tuple(retry_records),
        removed=removed,
        reminted=pending_reminted,
        outcomes=dict(outcomes),
        ledger_added=added,
        ledger_sources=seen,
        ledger_error=ledger_error,
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

    The public projection of `decide_promotion` — the cards, the holds, the
    refusal, and nothing a caller could replay the write path with.

    One gap, named rather than hidden: see `PromotionPlan.readings_unchecked`.
    It is a gap only by default. Pass a `client` — as the command does, having
    already decided to spend the lookups — and the dictionary witness runs
    here too, and this says so. A page refresh passes nothing and gets the
    offline answer; one code path either way, which is the point.

    Raises rather than reports for one thing only: asking for the dictionary
    check without handing over a client. See `decide_promotion`.
    """
    decision = decide_promotion(
        config, staging_path, source=source, client=client,
        skip_reading_check=skip_reading_check,
    )
    return project_promotion(decision)


def project_promotion(decision: PromotionDecision) -> PromotionPlan:
    """The safe view of a decision."""
    base = {
        "source": decision.source,
        "staging_path": decision.staging_path,
        "readings_unchecked": not decision.consulted,
    }
    if decision.is_blocked:
        return PromotionPlan(**base, blocked=str(decision.error))

    if decision.state in ("nothing", "pattern_only"):
        return PromotionPlan(**base)
    already = decision.already_archived

    readings = decision.readings
    if readings is None:
        return PromotionPlan(**base, already_archived=already)

    held = tuple(
        HeldCard(
            record=record,
            reason=staging.annotations(record).get("hold_reason", ""),
        )
        for record in readings.held
    )
    warnings = decision.reading_warnings + decision.deck_warnings
    if decision.state == "nothing_lands":
        return PromotionPlan(
            **base, held=held, already_archived=already, warnings=warnings
        )

    merged_by_id = {record.id: record for record in decision.merged}
    by_id = {record.id: record for record in decision.existing}
    landing: list[LandingCard] = []
    promoted = iter(readings.promoted)
    for original, stays in zip(decision.work, readings.keep, strict=True):
        if stays:
            continue
        transformed = next(promoted)
        landing.append(
            LandingCard(
                staged=original,
                landing=merged_by_id[transformed.id],
                existing=by_id.get(transformed.id),
                # `check_readings` already decided this and says so; deriving
                # it again from the ids is a second copy of the same rule.
                reminted_from=(
                    original.id if original.id in readings.reminted else ""
                ),
            )
        )
    return PromotionPlan(
        **base,
        landing=tuple(landing),
        held=held,
        already_archived=already,
        warnings=warnings,
    )
