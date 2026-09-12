"""Shared coverage decisions and journaled paid completeness checks.

Coverage asks whether extraction accounted for the source units it promised
to cover. It does not judge the Japanese. Both the command and workbench use
this module so the free gates, exact request identity, paid-call journal and
approval write cannot drift between surfaces.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

from japanese_anki import claude_client, coverage, jpdb, operations, promote, prompts, staging
from japanese_anki.application.extraction import (
    ANSWER_EMPTY,
    ANSWER_SAVED,
    ANSWER_UNAVAILABLE,
    FORGOTTEN,
    DispatchFailure,
    capture_hook,
    classify_dispatch_failure,
    durable_inbox_root,
    settle_dispatch,
)
from japanese_anki.application.promotion import POST_READING_GATES, decide_promotion
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.inputs import PreparedInput, prepare_corpus_input
from japanese_anki.io import exclusive_path_lock, prepare_bound_directory
from japanese_anki.promote import PromoteError

__all__ = [
    "CoverageApplicationError",
    "CoverageDecision",
    "CoveragePreview",
    "CoverageRunError",
    "CoverageRunResult",
    "OwnerCoverageApproval",
    "approve_coverage_as_owner",
    "plan_coverage",
    "plan_model_coverage",
    "prepare_owner_coverage_approval",
    "project_coverage",
    "run_model_coverage",
]


class CoverageApplicationError(JankiError):
    """Coverage cannot be decided or recorded safely."""


class CoverageRunError(CoverageApplicationError):
    """A coverage transaction stopped after its one-use action was consumed."""

    def __init__(
        self,
        message: str,
        *,
        operation_id: str,
        failure: DispatchFailure | None,
        provider_dispatched: bool,
        operation_write: Literal["absent", "present", "unknown"],
        operation_state: str | None,
        approval_write: Literal["not_written", "written", "unknown"],
    ) -> None:
        self.operation_id = operation_id
        self.failure = failure
        self.provider_dispatched = provider_dispatched
        self.operation_write = operation_write
        self.operation_state = operation_state
        self.approval_write = approval_write
        super().__init__(message)


CoverageState = Literal["not_required", "already_resolved", "ready"]


@dataclass(frozen=True, slots=True)
class CoverageDecision:
    """The exact local state behind one coverage action.

    This value can contain the prepared private source and is not a rendering
    model. :func:`project_coverage` is the safe browser/terminal projection.
    Every mutator re-plans and compares this snapshot before acting.
    """

    repository_root: Path
    staging_path: Path
    staging_revision: str
    state: CoverageState
    meta: Mapping[str, Any]
    block: Mapping[str, Any] | None
    source_file: str = ""
    prepared: PreparedInput | None = None
    model: str = ""
    instructions: str = ""
    request: coverage.CoverageRequest | None = None
    reaccept_requested: bool = False
    replace_existing: bool = False
    detail: str = ""

    @property
    def source_sha256(self) -> str:
        return self.prepared.source_sha256 if self.prepared is not None else ""

    @property
    def request_fingerprint(self) -> str:
        return self.request.request_fingerprint if self.request is not None else ""

    @property
    def prompt_fingerprint(self) -> str:
        return self.request.prompt_fingerprint if self.request is not None else ""


@dataclass(frozen=True, slots=True)
class CoveragePreview:
    """Coverage facts safe to render; no provider-ready source bytes."""

    staging_path: Path
    state: CoverageState
    staging_fingerprint: str
    source_file: str
    account: str
    source_units: int
    candidate_units: int
    model: str
    request_fingerprint: str
    prompt_fingerprint: str
    detail: str = ""

    @property
    def can_ask_model(self) -> bool:
        return self.state == "ready" and bool(self.request_fingerprint)


CoverageRunState = Literal["approved", "declined"]


@dataclass(frozen=True, slots=True)
class CoverageRunResult:
    state: CoverageRunState
    operation_id: str
    verdict: coverage.CoverageVerdict
    staging_path: Path


def _durable_source(config: ProjectConfig, name: str) -> Path:
    """Resolve one staging basename to exactly one preserved inbox source."""
    root = durable_inbox_root(config)
    try:
        matches = [
            path
            for path in root.rglob("*")
            if path.is_file() and path.name.casefold() == name.casefold()
        ]
    except OSError as exc:
        raise PromoteError(f"Could not inspect {root} for {name}: {exc}") from exc
    if not matches:
        raise PromoteError(
            f"Could not find {name} under {root}. Coverage is checked against "
            "the page itself, so the source has to still be in the inbox."
        )
    if len(matches) > 1:
        listed = ", ".join(str(path) for path in sorted(matches))
        raise PromoteError(f"More than one {name} under {root}: {listed}.")
    return matches[0]


def _plan_coverage(
    config: ProjectConfig,
    staging_path: Path,
    *,
    model: str | None = None,
    reaccept: bool = False,
    prepare_model: bool,
) -> CoverageDecision:
    """Plan one coverage action without writing or sending.

    Every invariant that can be checked locally runs before the private source
    is prepared and before a journal authorization can be written. In
    particular this re-derives the coverage facts, rather than accepting a
    self-consistent edited fingerprint that no verdict could repair.
    """
    path = Path(staging_path).resolve()
    preflight = decide_promotion(config, path, skip_reading_check=True)
    if (
        preflight.is_blocked
        and preflight.gate != "coverage"
        and preflight.gate not in POST_READING_GATES
    ):
        assert preflight.error is not None
        raise preflight.error
    wire = preflight.wire
    meta = preflight.meta
    revision = hashlib.sha256(wire).hexdigest()
    pending = promote.check_coverage_facts(dict(meta))
    raw_block = meta.get("coverage")
    block = dict(raw_block) if isinstance(raw_block, Mapping) else None
    common = {
        "repository_root": config.root.resolve(),
        "staging_path": path,
        "staging_revision": revision,
        "meta": meta,
        "block": block,
        "reaccept_requested": reaccept,
    }
    if pending is None:
        detail = (
            f"{path.name} carries no coverage block, so there is nothing to accept."
            if block is None
            else f"{path.name} does not need a coverage approval."
        )
        return CoverageDecision(state="not_required", detail=detail, **common)

    assert block is not None
    standing = block.get("approval")
    if standing is not None and staging.coverage_already_resolved(meta) and not reaccept:
        approval = standing if isinstance(standing, Mapping) else {}
        who = approval.get("model") or approval.get("authority") or "an owner"
        detail = (
            f"{path.name} is already approved by {who} on "
            f"{approval.get('approved_at', 'an earlier date')}; nothing was sent."
        )
        return CoverageDecision(
            state="already_resolved",
            source_file=str(meta.get("source_file") or "").strip(),
            detail=detail,
            **common,
        )
    if standing is not None and not reaccept:
        raise CoverageApplicationError(
            f"{path.name} already carries a coverage decision that does not "
            "currently pass. Use the explicit reaccept action to replace it; "
            "nothing was sent."
        )

    # Prove the round-trip writer can preserve this review before spending.
    staging.check_rewritable(path)
    source_name = str(meta.get("source_file") or "").strip()
    if not source_name:
        raise PromoteError(f"{path.name} does not say which source it was read from.")
    origin = _durable_source(config, source_name)
    prepared = prepare_corpus_input(origin, durable_inbox_root(config))
    recorded_sha = str(block.get("source_fingerprint") or "")
    if prepared.source_sha256 != recorded_sha:
        raise PromoteError(
            f"{origin} is not the file this staging was read from: it now "
            f"fingerprints {prepared.source_sha256[:12]}, and the extraction "
            f"recorded {recorded_sha[:12]}. Checking coverage against different "
            "bytes would answer a question about the wrong page."
        )
    chosen_model = (model or config.extract_model) if prepare_model else ""
    instructions = ""
    request = None
    if prepare_model:
        instructions = prompts.load(config.root, "approve-coverage")
        request = coverage.build_request(
            prepared,
            block,
            model=chosen_model,
            instructions=instructions,
        )
    return CoverageDecision(
        state="ready",
        source_file=source_name,
        prepared=prepared,
        model=chosen_model,
        instructions=instructions,
        request=request,
        replace_existing=bool(reaccept and standing is not None),
        **common,
    )


def plan_coverage(
    config: ProjectConfig,
    staging_path: Path,
    *,
    reaccept: bool = False,
) -> CoverageDecision:
    """Plan a local owner decision without loading optional AI dependencies."""
    return _plan_coverage(
        config,
        staging_path,
        reaccept=reaccept,
        prepare_model=False,
    )


def plan_model_coverage(
    config: ProjectConfig,
    staging_path: Path,
    *,
    model: str | None = None,
    reaccept: bool = False,
) -> CoverageDecision:
    """Plan the exact paid request a completeness-check consent names."""
    return _plan_coverage(
        config,
        staging_path,
        model=model,
        reaccept=reaccept,
        prepare_model=True,
    )


def project_coverage(decision: CoverageDecision) -> CoveragePreview:
    block = decision.block or {}
    units = block.get("source_units")
    listed = units if isinstance(units, list) else []
    return CoveragePreview(
        staging_path=decision.staging_path,
        state=decision.state,
        staging_fingerprint=decision.staging_revision,
        source_file=decision.source_file,
        account=(coverage.format_account(dict(block)) if block else ""),
        source_units=len(listed),
        candidate_units=sum(
            1
            for unit in listed
            if isinstance(unit, Mapping) and unit.get("disposition") == "candidate"
        ),
        model=decision.model,
        request_fingerprint=decision.request_fingerprint,
        prompt_fingerprint=decision.prompt_fingerprint,
        detail=decision.detail,
    )


def _fresh_decision(
    config: ProjectConfig, expected: CoverageDecision
) -> CoverageDecision:
    if expected.repository_root != config.root.resolve():
        raise CoverageApplicationError(
            "[coverage-config-mismatch] this coverage decision belongs to a "
            "different repository. Nothing was approved."
        )
    if expected.request is not None:
        fresh = plan_model_coverage(
            config,
            expected.staging_path,
            model=expected.model or None,
            reaccept=expected.reaccept_requested,
        )
    else:
        fresh = plan_coverage(
            config,
            expected.staging_path,
            reaccept=expected.reaccept_requested,
        )
    identity = (
        fresh.state,
        fresh.staging_revision,
        fresh.source_file,
        fresh.source_sha256,
        fresh.model,
        fresh.request_fingerprint,
        fresh.replace_existing,
    )
    expected_identity = (
        expected.state,
        expected.staging_revision,
        expected.source_file,
        expected.source_sha256,
        expected.model,
        expected.request_fingerprint,
        expected.replace_existing,
    )
    if identity != expected_identity:
        raise CoverageApplicationError(
            "[coverage-review-stale] the source, coverage account, prompt or "
            "staging review changed after it was shown. Nothing was approved; "
            "reload and review the current coverage decision."
        )
    return fresh


def _approval_payload(
    decision: CoverageDecision,
    *,
    authority: Literal["repository-owner", "model"],
    reason: str,
    approved_at: str | None = None,
) -> dict[str, Any]:
    """The exact approval record one decision would write.

    ``approved_at`` defaults to today, which is what an owner or model approval
    taken now means. A prepared study finish supplies the date its own
    preparation froze instead: the payload is part of a durable intent whose
    ``expected_after`` digest has to be the same on the day of a crash and the
    day after it, and a payload that reads the clock at apply time is a
    different file every midnight.
    """
    block = decision.block
    if block is None:
        raise CoverageApplicationError("This staging file has no coverage to approve.")
    cleaned = reason.strip()
    if not cleaned:
        raise CoverageApplicationError("A coverage decision needs a non-empty reason.")
    if approved_at is None:
        frozen = date.today().isoformat()
    else:
        try:
            if (
                not isinstance(approved_at, str)
                or date.fromisoformat(approved_at).isoformat() != approved_at
            ):
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise CoverageApplicationError(
                "A frozen coverage approval date must use YYYY-MM-DD."
            ) from exc
        frozen = approved_at
    payload: dict[str, Any] = {
        "authority": authority,
        "source_fingerprint": block["source_fingerprint"],
        # Recomputed from what was actually shown, never copied from a field a
        # hand edit could leave stale.
        "coverage_block_fingerprint": staging.coverage_block_fingerprint(block),
        **staging.coverage_acceptance_requirements(block),
        "reason": cleaned,
        "approved_at": frozen,
    }
    if authority == "model":
        payload.update(
            model=decision.model,
            prompt_fingerprint=decision.prompt_fingerprint,
        )

    # Validate the exact intended record before touching the review file.
    checked_meta = dict(decision.meta)
    checked_block = dict(block)
    checked_block["approval"] = payload
    checked_meta["coverage"] = checked_block
    promote.check_coverage(checked_meta)
    return payload


@dataclass(frozen=True, slots=True)
class OwnerCoverageApproval:
    """One owner coverage decision, frozen before anything is written.

    The payload carries its own ``approved_at``, so the same bytes land on the
    day the owner decided and on the day a resume finishes the write. It is
    not authority by itself: the caller brings the owner's exact decision, and
    this only fixes what recording it would say.
    """

    staging_path: Path
    staging_revision: str
    payload: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "staging_path": str(self.staging_path),
            "staging_revision": self.staging_revision,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> OwnerCoverageApproval:
        try:
            return cls(
                staging_path=Path(str(raw["staging_path"])),
                staging_revision=str(raw["staging_revision"]),
                payload=dict(raw["payload"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CoverageApplicationError(
                f"A recorded coverage approval is unreadable: {exc}"
            ) from exc


def prepare_owner_coverage_approval(
    config: ProjectConfig,
    decision: CoverageDecision,
    *,
    reason: str,
    approved_at: str | None = None,
) -> OwnerCoverageApproval:
    """Build one owner approval payload without writing it.

    Side-effect-free: it re-plans, validates the exact intended record through
    the ordinary promotion gate, and returns the bytes a write would record.
    A study finish composes that payload into its prepared staging after-text
    instead of publishing it on its own.
    """
    fresh = _fresh_decision(config, decision)
    if fresh.state != "ready" or fresh.replace_existing:
        raise CoverageApplicationError(
            "This page does not offer a fresh owner coverage approval."
        )
    payload = _approval_payload(
        fresh,
        authority="repository-owner",
        reason=reason,
        approved_at=approved_at,
    )
    return OwnerCoverageApproval(
        staging_path=fresh.staging_path,
        staging_revision=fresh.staging_revision,
        payload=payload,
    )


def approve_coverage_as_owner(
    config: ProjectConfig,
    decision: CoverageDecision,
    *,
    reason: str,
) -> Path:
    """Record one explicit, scoped repository-owner decision."""
    prepared = prepare_owner_coverage_approval(config, decision, reason=reason)
    return staging.record_coverage_approval(
        prepared.staging_path,
        prepared.payload,
        expected_revision=prepared.staging_revision,
    )


def _failure_message(
    source: str, operation_id: str, exc: BaseException, failure: DispatchFailure
) -> str:
    if failure.outcome == ANSWER_SAVED:
        note = (
            "The exact paid reply was saved; inspect it with "
            f"'janki operations --show-reply {operation_id}'."
        )
    elif failure.outcome == ANSWER_EMPTY:
        note = (
            "The paid reply was saved but carries no answer text; inspect its "
            f"exact bytes with 'janki operations --show-reply {operation_id}'."
        )
    elif failure.outcome == ANSWER_UNAVAILABLE:
        note = "The captured reply is recorded, but its exact bytes are unavailable."
    elif failure.outcome == FORGOTTEN:
        note = (
            "The operation's forget decision is still finishing."
            if failure.cleanup_pending
            else "The operation was already forgotten."
        )
    else:
        note = "This call may have been billed; do not retry until it is settled."
    return f"Coverage check for {source} failed: {exc} {note} Operation {operation_id}."


def _operation_write_state(
    config: ProjectConfig, operation_id: str
) -> Literal["absent", "present", "unknown"]:
    """Conservatively classify a journal write whose caller saw an error."""
    try:
        current = operations.OperationJournal.load(config.operations_file)
    except JankiError:
        return "unknown"
    return "present" if operation_id in current.operations else "absent"


def _operation_state(config: ProjectConfig, operation_id: str) -> str | None:
    try:
        current = operations.OperationJournal.load(config.operations_file)
    except JankiError:
        return None
    entry = current.operations.get(operation_id)
    return entry.state if entry is not None else None


def _require_payable_preflight(
    config: ProjectConfig,
    decision: CoverageDecision,
    *,
    reading_client: jpdb.JpdbClient | None,
    skip_reading_check: bool,
) -> None:
    """Refuse every authoritative promotion gate before a paid dispatch.

    Post-reading gates are only a preview when JPDB was skipped: a dictionary
    may hold the row that created an apparent collision, merge or deck problem.
    Consult before believing one unless the caller explicitly chose to skip
    that witness, in which case the offline result is the promotion result too.
    """
    offline = decide_promotion(
        config,
        decision.staging_path,
        skip_reading_check=True,
    )
    preflight = offline
    consulted = False
    if (
        preflight.is_blocked
        and preflight.gate in POST_READING_GATES
        and not skip_reading_check
    ):
        consulted = True
        witness = reading_client or jpdb.JpdbClient(jpdb.api_key_from_env())
        preflight = decide_promotion(
            config,
            decision.staging_path,
            client=witness,
            skip_reading_check=False,
        )
    if consulted:
        # `decide_promotion` snapshots staging, archive, the exact collection
        # revision, and the deck-derived reading inputs before the blocking
        # dictionary lookup. Re-read them afterwards, before journal
        # authorization: otherwise a change during jpdb would be discovered
        # only by the post-charge approval CAS.
        after_lookup = decide_promotion(
            config,
            decision.staging_path,
            skip_reading_check=True,
        )
        before_state = (
            preflight.wire,
            (
                preflight.archive_revision,
                preflight.archived,
                preflight.archived_meta,
            ),
            preflight.output_revision,
            preflight.stored_ids,
            preflight.unreadable_decks,
            preflight.deck_revision,
            preflight.ledger_revision,
        )
        after_state = (
            after_lookup.wire,
            (
                after_lookup.archive_revision,
                after_lookup.archived,
                after_lookup.archived_meta,
            ),
            after_lookup.output_revision,
            after_lookup.stored_ids,
            after_lookup.unreadable_decks,
            after_lookup.deck_revision,
            after_lookup.ledger_revision,
        )
        if after_state != before_state:
            raise CoverageApplicationError(
                "[coverage-preflight-stale] the staging, archive, collection, "
                "or authoritative promotion input changed during the reading "
                "check. Nothing was sent; reload and try again."
            )
    if preflight.is_blocked and preflight.gate != "coverage":
        assert preflight.error is not None
        raise preflight.error
    if hashlib.sha256(preflight.wire).hexdigest() != decision.staging_revision:
        raise CoverageApplicationError(
            "[coverage-review-stale] the staging review changed during its "
            "promotion preflight. Nothing was sent; reload and try again."
        )


def run_model_coverage(
    config: ProjectConfig,
    decision: CoverageDecision,
    *,
    client: Any | None = None,
    reading_client: jpdb.JpdbClient | None = None,
    skip_reading_check: bool = False,
) -> CoverageRunResult:
    """Dispatch, capture and commit one separately authorized coverage check."""
    fresh = _fresh_decision(config, decision)
    if fresh.state != "ready" or fresh.prepared is None or fresh.request is None:
        raise CoverageApplicationError(
            "This staging file does not currently offer a paid coverage check."
        )
    _require_payable_preflight(
        config,
        fresh,
        reading_client=reading_client,
        skip_reading_check=skip_reading_check,
    )
    paid_client = client if client is not None else claude_client.prepare_paid_client()

    operations.prepare_artifact_store(config.operations_file)
    prepare_bound_directory(fresh.staging_path.parent)
    journal = operations.OperationJournal.load(config.operations_file)
    operation_id = str(uuid.uuid4())
    try:
        journal.authorize(
            operation_id,
            kind="coverage",
            source_file=fresh.source_file,
            source_sha256=fresh.source_sha256,
            request_fp=fresh.request_fingerprint,
            model=fresh.model,
        )
    except Exception as exc:  # noqa: BLE001 - durable write may have landed
        operation_write = _operation_write_state(config, operation_id)
        raise CoverageRunError(
            f"Coverage check for {fresh.source_file} could not durably record "
            f"its authority: {exc} Inspect 'janki operations' before authorizing "
            "another paid call.",
            operation_id=operation_id,
            failure=None,
            provider_dispatched=False,
            operation_write=operation_write,
            operation_state=_operation_state(config, operation_id),
            approval_write="not_written",
        ) from exc
    try:
        journal.advance(operation_id, "dispatching")
    except Exception as exc:  # noqa: BLE001 - durable write may have landed
        operation_write = _operation_write_state(config, operation_id)
        raise CoverageRunError(
            f"Coverage check for {fresh.source_file} could not durably record "
            f"its dispatch boundary: {exc} Inspect operation {operation_id} with "
            "'janki operations' before authorizing another paid call.",
            operation_id=operation_id,
            failure=None,
            provider_dispatched=False,
            operation_write=operation_write,
            operation_state=_operation_state(config, operation_id),
            approval_write="not_written",
        ) from exc
    approval_write: Literal["not_written", "written", "unknown"] = "not_written"
    try:
        verdict = coverage.review_coverage(
            fresh.prepared,
            dict(fresh.block or {}),
            model=fresh.model,
            instructions=fresh.instructions,
            client=paid_client,
            capture=capture_hook(config, journal, operation_id),
        )
        settle_dispatch(config, journal, operation_id, verdict)
        if not verdict.approved:
            return CoverageRunResult(
                state="declined",
                operation_id=operation_id,
                verdict=verdict,
                staging_path=fresh.staging_path,
            )

        reason = verdict.reason
        if fresh.replace_existing:
            reason = f"Re-asked and replaced an earlier approval. {reason}"
        payload = _approval_payload(
            fresh,
            authority="model",
            reason=reason,
        )
        with exclusive_path_lock(fresh.staging_path):
            def persist_approval() -> None:
                nonlocal approval_write
                # A writer error may happen before or after publication. Only
                # a normal return proves the exact approval is durable.
                approval_write = "unknown"
                staging.record_coverage_approval_under_lock(
                    fresh.staging_path,
                    payload,
                    replace_existing=fresh.replace_existing,
                    expected_revision=fresh.staging_revision,
                )
                approval_write = "written"

            journal.commit_result(
                operation_id,
                persist_approval,
            )
    except Exception as exc:  # noqa: BLE001 - classify every failure after dispatch
        try:
            failure = classify_dispatch_failure(
                config, journal, operation_id, exc
            )
        except JankiError as journal_error:
            raise CoverageRunError(
                f"Coverage check for {fresh.source_file} failed after dispatch: "
                f"{exc} Janki could not settle operation {operation_id}: "
                f"{journal_error}. This call may have been billed; do not retry "
                "until you inspect janki operations.",
                operation_id=operation_id,
                failure=None,
                provider_dispatched=True,
                operation_write=_operation_write_state(config, operation_id),
                operation_state=_operation_state(config, operation_id),
                approval_write=approval_write,
            ) from exc
        raise CoverageRunError(
            _failure_message(fresh.source_file, operation_id, exc, failure),
            operation_id=operation_id,
            failure=failure,
            provider_dispatched=True,
            operation_write=_operation_write_state(config, operation_id),
            operation_state=_operation_state(config, operation_id),
            approval_write=approval_write,
        ) from exc
    return CoverageRunResult(
        state="approved",
        operation_id=operation_id,
        verdict=verdict,
        staging_path=fresh.staging_path,
    )
