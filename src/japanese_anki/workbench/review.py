"""The exact-approval write transaction, and nothing else.

This began as `review_panel.py`, a one-shot localhost page. W2b deleted that
page — the workbench renders the review now — and what survived the fold is
the part worth keeping: the transaction that turns a human's tick into durable
authority without letting a stale page, a swapped symlink, or a concurrent
editor lose someone's work.

No HTTP, no HTML, no session, no rendering. The caller proves authority; this
proves the bytes it read are still the bytes it writes over.

It makes no network or model call, and it never edits Japanese or promotes a
row. Its only writes are the two human-review marks:

* exact fingerprints for every nonblank Japanese example on the selected rows;
* ``reviewed = true`` on the matching pattern-store entry.

Both land through a compare-and-swap bound to the exact snapshot the caller
captured, and a write that cannot prove which side of the seam it fell on says
so rather than reporting success.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import os
import stat
import threading
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl

from japanese_anki import patterns, staging
from japanese_anki.application.authority import (
    needs_example_review,
)
from japanese_anki.errors import JankiError
from japanese_anki.io import (
    atomic_write_text_bound,
    exclusive_path_lock,
    read_bytes_bound,
)
from japanese_anki.models import (
    EXAMPLE_AUTHORITY_KEY,
    VocabularyRecord,
    set_example_flags,
)

__all__ = [
    "PanelRequestError",
    "PartialReviewBatchError",
    "PartialReviewError",
    "PreparedReview",
    "PreparedReviewBatch",
    "PreparedReviewPart",
    "ReviewBatchOutcome",
    "ReviewBatchRequest",
    "ReviewOutcome",
    "IndeterminateWriteError",
    "ReviewPanel",
    "apply_prepared_review",
    "apply_prepared_review_batch",
    "apply_prepared_review_batch_under_locks",
    "apply_prepared_review_under_locks",
    "bound_replace",
    "bound_replace_under_lock",
    "prepare_review_batch",
    "recover_prepared_review",
    "recover_prepared_review_batch",
    "recover_prepared_review_batch_under_locks",
    "recover_prepared_review_under_locks",
    "review_batch_path_locks",
    "review_path_locks",
    "review_target_paths",
    "ReviewPanelError",
    "StaleReviewError",
    "parse_review_form",
]


_FORM_SINGLETONS = frozenset(
    {"csrf", "staging_snapshot", "patterns_snapshot", "action", "patterns"}
)
_FORM_FIELDS = _FORM_SINGLETONS | {"record"}

#: The part name a single-panel review carries inside the aggregate writer.
_ONE_PANEL = ""


class ReviewPanelError(JankiError):
    """The requested staging review cannot be displayed or saved safely."""


class StaleReviewError(ReviewPanelError):
    """One of the exact files shown in the browser changed before submit."""


class PanelRequestError(ReviewPanelError):
    """A submitted action was not one the rendered page offered."""


@dataclass(frozen=True, slots=True)
class ReviewOutcome:
    accepted_record_ids: tuple[str, ...] = ()
    pattern_reviewed: bool = False


class PartialReviewError(ReviewPanelError):
    """At least one exact monotonic approval landed before a later failure."""

    def __init__(self, message: str, outcome: ReviewOutcome) -> None:
        super().__init__(message)
        self.outcome = outcome


class IndeterminateWriteError(ReviewPanelError):
    """A final replace failed after the target stopped matching its snapshot."""

    def __init__(self, message: str, *, intended_bytes_are_live: bool) -> None:
        super().__init__(message)
        self.intended_bytes_are_live = intended_bytes_are_live


def _fingerprint(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _bound_replace_digest(
    path: Path, text: str, expected_sha256: str, *, label: str
) -> None:
    """CAS one captured revision after the caller settles the path lock.

    The bound side is named by its digest rather than its bytes so a durable
    prepared review can replay this exact write without carrying a second copy
    of every file it replaces.
    """
    intended = text.encode("utf-8")
    try:
        atomic_write_text_bound(
            path,
            text,
            expected_revision=expected_sha256,
        )
    except Exception as exc:
        try:
            is_link = path.is_symlink()
            live = None if is_link else read_bytes_bound(path)
        except (JankiError, OSError):
            is_link = False
            live = None
        message = str(exc).lower()
        if (
            is_link
            or "bound target changed" in message
            or "refusing to replace non-regular" in message
        ):
            raise StaleReviewError(f"The {label} changed at the final review write") from exc
        if live is not None and _fingerprint(live) == expected_sha256:
            raise ReviewPanelError(
                f"Could not save the {label}; its captured bytes remain unchanged: {exc}"
            ) from exc
        raise IndeterminateWriteError(
            f"The {label} write failed after its exact snapshot stopped matching; "
            "restart the panel and inspect the recorded decisions",
            intended_bytes_are_live=live == intended,
        ) from exc


def _bound_replace_unlocked(
    path: Path, text: str, snapshot: bytes, *, label: str
) -> None:
    """CAS one rendered snapshot after the caller settles the path lock."""
    _bound_replace_digest(path, text, _fingerprint(snapshot), label=label)


def bound_replace(path: Path, text: str, snapshot: bytes, *, label: str) -> None:
    """Join the staging transaction lock, then replace one exact snapshot."""
    with exclusive_path_lock(path):
        _bound_replace_unlocked(path, text, snapshot, label=label)


def bound_replace_under_lock(
    path: Path, text: str, snapshot: bytes, *, label: str
) -> None:
    """Replace one exact snapshot while the caller holds its path lock."""
    _bound_replace_unlocked(path, text, snapshot, label=label)


def _absolute(path: Path) -> Path:
    """An absolute lexical path without following its final symlink."""
    return Path(os.path.abspath(os.fspath(path)))


def _require_regular_non_symlink(
    path: Path, description: str, *, absent_ok: bool = False
) -> None:
    """Refuse anything but a plain file at this exact path.

    ``absent_ok`` for the pattern store, which a project that has never
    extracted anything simply does not have yet. `patterns.load_store` reads a
    missing store as an empty one, and this being stricter meant importing a
    deck into a fresh project could not open a review at all — the same
    over-broad refusal as demanding an extraction lineage from a file that
    never had one. A path that *exists* is still held to the same rule.
    """
    try:
        details = path.lstat()
    except FileNotFoundError:
        if absent_ok:
            return
        raise ReviewPanelError(
            f"{description} must be a direct regular non-symlink file: {path}"
        ) from None
    except OSError as exc:
        raise ReviewPanelError(
            f"{description} must be a direct regular non-symlink file: {path}"
        ) from exc
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
        raise ReviewPanelError(f"{description} must be a direct regular non-symlink file: {path}")


def _capture_regular_bytes(
    path: Path, description: str, *, absent_ok: bool = False
) -> bytes:
    """Capture bytes after recovering any interrupted guarded publication."""
    captured, _absent = _capture_regular_state(path, description, absent_ok=absent_ok)
    return captured


def _capture_regular_state(
    path: Path, description: str, *, absent_ok: bool = False
) -> tuple[bytes, bool]:
    """The captured bytes, and whether the file was absent when they were taken.

    An absent pattern store captures as ``{}`` so the panel can open at all,
    but a *durable* preparation has to keep the two apart: ``{}`` is also a
    store somebody can really write, and a recovery that read the one as the
    other would compare-and-swap against bytes that were never there.
    """
    try:
        return read_bytes_bound(path), False
    except FileNotFoundError:
        if absent_ok:
            return b"{}", True
        raise ReviewPanelError(
            f"{description} must remain a direct regular non-symlink file during capture: {path}"
        ) from None
    except (JankiError, OSError) as exc:
        raise ReviewPanelError(
            f"{description} must remain a direct regular non-symlink file during capture: {path}"
        ) from exc


def _active_staging_path(path: Path, staging_dir: Path) -> Path:
    candidate = _absolute(path)
    active = _absolute(staging_dir)
    if candidate.parent != active or candidate.suffix.lower() not in staging.STAGING_SUFFIXES:
        raise ReviewPanelError(
            f"Review panels accept one direct file in the active staging directory "
            f"{active}, not {candidate}"
        )
    _require_regular_non_symlink(candidate, "An active staging path")
    return candidate


def review_target_paths(
    staging_path: Path, *, staging_dir: Path, patterns_path: Path
) -> tuple[Path, Path]:
    """The two exact paths a review writes, proved to be the right shapes.

    Factored out of :meth:`ReviewPanel.open` so a durable prepared review can
    be re-bound to a *live* configuration before it writes anything. The
    staging file must be a direct regular non-symlink member of the configured
    active directory; the pattern store must be the configured file, absent or
    a direct regular non-symlink. Both names are lexical: resolving them first
    would accept an alias for the file whose bytes the owner actually reviewed.
    """
    active = _active_staging_path(staging_path, staging_dir)
    pattern_path = _absolute(patterns_path)
    _require_regular_non_symlink(pattern_path, "The pattern store", absent_ok=True)
    return active, pattern_path


@contextlib.contextmanager
def _sorted_path_locks(paths: Sequence[Path]):
    """Lock distinct real targets in one process-wide deterministic order."""
    targets = [Path(os.path.realpath(path)) for path in paths]
    if len(set(targets)) != len(targets):
        raise ReviewPanelError("The staging file and pattern store must be distinct")
    with ExitStack() as stack:
        for path in sorted(set(targets), key=os.fspath):
            stack.enter_context(exclusive_path_lock(path))
        yield


def review_path_locks(prepared: PreparedReview):
    """The two review locks, in the one deterministic order every caller takes.

    Exported because a coordinator that already owns these paths applies its
    prepared review through the ``_under_locks`` entrypoints; `exclusive_path_lock`
    is deliberately non-reentrant, so a second acquisition here would deadlock.
    """
    return _sorted_path_locks((Path(prepared.staging.path), Path(prepared.patterns.path)))


def review_batch_path_locks(batch: PreparedReviewBatch):
    """Every path an aggregate review writes, in that same one order."""
    return _sorted_path_locks(
        [Path(part.staging.path) for part in batch.parts] + [Path(batch.patterns.path)]
    )


@dataclass(frozen=True, slots=True)
class PreparedReview:
    """Everything one confirmed staging review would write, computed first.

    Two paths, one payload each. Review and coverage target the **same**
    staging path, so their pure renderings are composed in memory into one
    final after-text rather than published one after the other: an
    intermediate review-only document is a state no owner approved and no
    recovery can classify.
    """

    components: tuple[staging.PreparedComponent, ...]
    record_ids: tuple[str, ...] = ()
    review_patterns: bool = False
    #: record id -> the exact ``example_authority`` wire value this review
    #: binds. The projection applies the same flags; a coordinator compares
    #: the two rather than trusting that they agree.
    expected_authority: Mapping[str, str] = field(default_factory=dict)
    #: The frozen ISO date of the composed coverage approval, or None when
    #: this review records no coverage decision.
    coverage_approved_at: str | None = None

    def __post_init__(self) -> None:
        roles = [component.role for component in self.components]
        if roles != ["staging", "patterns"]:
            raise ReviewPanelError(
                "A prepared review binds exactly one staging and one pattern "
                f"component, in that order, not {roles}"
            )

    @property
    def staging(self) -> staging.PreparedComponent:
        return self.components[0]

    @property
    def patterns(self) -> staging.PreparedComponent:
        return self.components[1]

    @property
    def outcome(self) -> ReviewOutcome:
        """What a complete apply of this intent reports."""
        return ReviewOutcome(self.record_ids, self.review_patterns)

    def as_batch(
        self, *, part_name: str = _ONE_PANEL, source: str = ""
    ) -> PreparedReviewBatch:
        """This one panel's review as the aggregate the writer applies.

        The pattern component is already this review's single final payload,
        because a one-panel review has exactly one part contributing to it.
        """
        return PreparedReviewBatch(
            parts=(
                PreparedReviewPart(
                    part_name=part_name,
                    source=source,
                    staging=self.staging,
                    record_ids=self.record_ids,
                    review_patterns=self.review_patterns,
                    expected_authority=self.expected_authority,
                    coverage_approved_at=self.coverage_approved_at,
                ),
            ),
            patterns=self.patterns,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "components": [component.to_dict() for component in self.components],
            "record_ids": list(self.record_ids),
            "review_patterns": self.review_patterns,
            "expected_authority": dict(self.expected_authority),
            "coverage_approved_at": self.coverage_approved_at,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> PreparedReview:
        try:
            approved_at = raw["coverage_approved_at"]
            return cls(
                components=tuple(
                    staging.PreparedComponent.from_dict(item)
                    for item in raw["components"]
                ),
                record_ids=tuple(str(item) for item in raw["record_ids"]),
                review_patterns=bool(raw["review_patterns"]),
                expected_authority={
                    str(key): str(value)
                    for key, value in dict(raw["expected_authority"]).items()
                },
                coverage_approved_at=(
                    None if approved_at is None else str(approved_at)
                ),
            )
        except (KeyError, TypeError, ValueError, staging.StagingError) as exc:
            raise ReviewPanelError(
                f"A recorded review intent is unreadable: {exc}"
            ) from exc


@dataclass(frozen=True, slots=True)
class ReviewBatchRequest:
    """One part's owner choices, as an aggregate preparation receives them."""

    part_name: str
    staging_path: Path
    record_ids: tuple[str, ...] = ()
    review_patterns: bool = False
    coverage_approval: Mapping[str, Any] | None = None
    #: The staging bytes this choice was taken over, when the caller has them.
    expected_revision: str | None = None


@dataclass(frozen=True, slots=True)
class PreparedReviewPart:
    """One part's staging half of an aggregate review.

    The pattern store is **not** here. §7.6 gives it one final after-payload
    for the whole batch, because every part writes the same file: two parts
    each preparing their own before/after pair for it bind two different
    after-digests over one path, and whichever lands second finds the store at
    neither of its own — so the second part's review can never be applied or
    recovered, and its staging half never lands either.
    """

    part_name: str
    source: str
    staging: staging.PreparedComponent
    record_ids: tuple[str, ...] = ()
    review_patterns: bool = False
    expected_authority: Mapping[str, str] = field(default_factory=dict)
    coverage_approved_at: str | None = None

    def __post_init__(self) -> None:
        if self.staging.role != "staging":
            raise ReviewPanelError(
                f"A prepared review part binds a staging component, not "
                f"{self.staging.role!r}"
            )

    @property
    def outcome_when_complete(self) -> ReviewOutcome:
        """What this part reports once the whole batch has been applied."""
        return ReviewOutcome(self.record_ids, self.review_patterns)

    def to_dict(self) -> dict[str, Any]:
        return {
            "part_name": self.part_name,
            "source": self.source,
            "staging": self.staging.to_dict(),
            "record_ids": list(self.record_ids),
            "review_patterns": self.review_patterns,
            "expected_authority": dict(self.expected_authority),
            "coverage_approved_at": self.coverage_approved_at,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> PreparedReviewPart:
        try:
            approved_at = raw["coverage_approved_at"]
            return cls(
                part_name=str(raw["part_name"]),
                source=str(raw["source"]),
                staging=staging.PreparedComponent.from_dict(raw["staging"]),
                record_ids=tuple(str(item) for item in raw["record_ids"]),
                review_patterns=bool(raw["review_patterns"]),
                expected_authority={
                    str(key): str(value)
                    for key, value in dict(raw["expected_authority"]).items()
                },
                coverage_approved_at=(
                    None if approved_at is None else str(approved_at)
                ),
            )
        except (KeyError, TypeError, ValueError, staging.StagingError) as exc:
            raise ReviewPanelError(
                f"A recorded review intent is unreadable: {exc}"
            ) from exc


@dataclass(frozen=True, slots=True)
class ReviewBatchOutcome:
    """What one aggregate review saved, per part."""

    parts: Mapping[str, ReviewOutcome] = field(default_factory=dict)


class PartialReviewBatchError(ReviewPanelError):
    """At least one part's exact monotonic approval landed before a failure."""

    def __init__(self, message: str, outcome: ReviewBatchOutcome) -> None:
        super().__init__(message)
        self.outcome = outcome


@dataclass(frozen=True, slots=True)
class PreparedReviewBatch:
    """Every part's staging payload over **one** final pattern-store payload.

    One before/after pair per distinct path, which is what §7.6 asks apply and
    recovery to precheck. The single-part case is the ordinary panel submit,
    and it goes through the same writer rather than a second copy of it.
    """

    parts: tuple[PreparedReviewPart, ...]
    patterns: staging.PreparedComponent

    def __post_init__(self) -> None:
        if not self.parts:
            raise ReviewPanelError("A prepared review batch binds at least one part")
        if self.patterns.role != "patterns":
            raise ReviewPanelError(
                f"A prepared review batch binds a pattern component, not "
                f"{self.patterns.role!r}"
            )
        names = [part.part_name for part in self.parts]
        if len(set(names)) != len(names):
            raise ReviewPanelError("Each part in one review batch has its own name")
        paths = [Path(part.staging.path).absolute() for part in self.parts]
        if len(set(paths)) != len(paths):
            raise ReviewPanelError("Each part in one review batch has its own file")
        if Path(self.patterns.path).absolute() in set(paths):
            raise ReviewPanelError(
                "The staging file and pattern store must be distinct"
            )

    @property
    def components(self) -> tuple[staging.PreparedComponent, ...]:
        """Every distinct path this batch binds, in the order it writes them."""
        return (*(part.staging for part in self.parts), self.patterns)

    def part(self, part_name: str) -> PreparedReviewPart | None:
        for item in self.parts:
            if item.part_name == part_name:
                return item
        return None

    @property
    def outcome(self) -> ReviewBatchOutcome:
        """What a complete apply of this intent reports."""
        return ReviewBatchOutcome(
            {part.part_name: part.outcome_when_complete for part in self.parts}
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "parts": [part.to_dict() for part in self.parts],
            "patterns": self.patterns.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> PreparedReviewBatch:
        try:
            return cls(
                parts=tuple(
                    PreparedReviewPart.from_dict(item) for item in raw["parts"]
                ),
                patterns=staging.PreparedComponent.from_dict(raw["patterns"]),
            )
        except (KeyError, TypeError, ValueError, staging.StagingError) as exc:
            raise ReviewPanelError(
                f"A recorded review intent is unreadable: {exc}"
            ) from exc


def _review_component(
    role: str, path: Path, *, before: str | None, after_text: str | None
) -> staging.PreparedComponent:
    """One review component, with absence preserved as its own before-state."""
    return staging.PreparedComponent(
        role=role,
        path=str(path),
        expected_before=before,
        expected_after=(
            before
            if after_text is None
            else _fingerprint(after_text.encode("utf-8"))
        ),
        after_text=after_text,
    )


def _live_component_bytes(path: Path, description: str) -> bytes | None:
    """The bytes at one bound path, or ``None`` when the path is absent."""
    try:
        return read_bytes_bound(path)
    except FileNotFoundError:
        return None
    except (JankiError, OSError) as exc:
        raise ReviewPanelError(
            f"{description} must remain a direct regular non-symlink file during capture: {path}"
        ) from exc


def _component_state(
    component: staging.PreparedComponent, *, description: str, label: str
) -> str:
    """``"pending"``, ``"complete"``, or a refusal naming both bound digests."""
    live = _live_component_bytes(Path(component.path), description)
    observed = None if live is None else _fingerprint(live)
    if observed == component.expected_before:
        return "pending" if component.writes else "complete"
    if observed == component.expected_after:
        return "complete"
    raise StaleReviewError(
        f"The {label} changed after this review page was rendered "
        f"(expected {component.expected_before} before or "
        f"{component.expected_after} after, found {observed})"
    )


def _apply_prepared_review_batch_under_locks(
    batch: PreparedReviewBatch, *, resume: bool
) -> ReviewBatchOutcome:
    """Write one aggregate review's whole component vector, prechecked first.

    Every bound path — each part's staging file and the one pattern store — is
    measured before **any** byte is written, so a stale later target cannot be
    discovered after an earlier one has already landed. ``resume`` allows a
    component already at its after-state, which is how a crash between two of
    the writes finishes from the intent alone; an ordinary submit requires all
    of them to be exactly where it read them.

    The parts are written in the order the batch carries them — §7.2's
    ascending published part name — and the single pattern payload last, so a
    second interruption leaves the same recoverable shape.
    """
    states: dict[str, str] = {}
    for part in batch.parts:
        label = _part_label(batch, part)
        state = _component_state(
            part.staging, description="The active staging file", label=label
        )
        if state == "complete" and part.staging.writes and not resume:
            raise StaleReviewError(
                f"The {label} changed after this review page was rendered"
            )
        states[part.part_name] = state
    patterns_state = _component_state(
        batch.patterns, description="The pattern store", label="pattern store"
    )
    if patterns_state == "complete" and batch.patterns.writes and not resume:
        raise StaleReviewError(
            "The pattern store changed after this review page was rendered"
        )

    saved: dict[str, ReviewOutcome] = {}
    for part in batch.parts:
        if not part.staging.writes:
            saved[part.part_name] = ReviewOutcome()
            continue
        if states[part.part_name] == "pending":
            try:
                _bound_replace_digest(
                    Path(part.staging.path),
                    part.staging.after_text or "",
                    part.staging.expected_before or "",
                    label=_part_label(batch, part),
                )
            except IndeterminateWriteError as exc:
                saved[part.part_name] = ReviewOutcome(
                    part.record_ids if exc.intended_bytes_are_live else (),
                    False,
                )
                raise PartialReviewBatchError(
                    str(exc), ReviewBatchOutcome(saved)
                ) from exc
        saved[part.part_name] = ReviewOutcome(part.record_ids, False)

    # There is a deliberately tiny process-crash window between two requested
    # atomic replaces. A WAL would be disproportionate for independent,
    # idempotent, monotonic review marks. Once the first exact approval lands
    # it is never rolled back: doing so could erase a concurrent human edit
    # made before a later failure. A prepared review closes that window a
    # second way — the persisted intent replays the missing write exactly.
    if batch.patterns.writes and patterns_state == "pending":
        try:
            _bound_replace_digest(
                Path(batch.patterns.path),
                batch.patterns.after_text or "",
                batch.patterns.expected_before or "",
                label="pattern store",
            )
        except IndeterminateWriteError as exc:
            raise PartialReviewBatchError(
                str(exc),
                _with_pattern_marks(batch, saved, reviewed=exc.intended_bytes_are_live),
            ) from exc
        except (StaleReviewError, ReviewPanelError) as exc:
            if any(outcome.accepted_record_ids for outcome in saved.values()):
                raise PartialReviewBatchError(
                    f"The card approval was saved, but the pattern review was not: {exc}",
                    ReviewBatchOutcome(saved),
                ) from exc
            raise
    # One report for both shapes, so `PreparedReviewBatch.outcome` — what a
    # complete apply of this intent says — is the value a complete apply
    # actually returns. A batch whose selected marks were already `true` writes
    # nothing and still reports those parts reviewed; a batch where nobody
    # selected one reports every part `False`, exactly as before.
    return _with_pattern_marks(batch, saved, reviewed=True)


def _part_label(batch: PreparedReviewBatch, part: PreparedReviewPart) -> str:
    """How one part's staging file is named in a refusal.

    An ordinary one-panel submit says "staging file", because naming the part
    would be noise about the only file in the request.
    """
    if len(batch.parts) == 1:
        return "staging file"
    return f"staging file for {part.part_name}"


def _with_pattern_marks(
    batch: PreparedReviewBatch,
    saved: Mapping[str, ReviewOutcome],
    *,
    reviewed: bool,
) -> ReviewBatchOutcome:
    """The saved outcomes, with the shared pattern mark reported per part.

    One store write, and only the parts whose owner selected the mark report
    it: a part that chose nothing never claims a pattern review because some
    other part's choice wrote the file. ``reviewed`` says whether this pass's
    store payload is live, which is why a completed apply passes ``True`` even
    when there was nothing to write — a part that selected an entry already
    marked `true` has its review recorded either way.
    """
    return ReviewBatchOutcome(
        {
            part.part_name: ReviewOutcome(
                saved[part.part_name].accepted_record_ids,
                reviewed and part.review_patterns,
            )
            for part in batch.parts
        }
    )


def prepare_review_batch(
    requests: Sequence[ReviewBatchRequest],
    *,
    staging_dir: Path,
    patterns_path: Path,
    collection_name: str = "",
) -> PreparedReviewBatch:
    """Freeze every part's review over **one** captured pattern store.

    §7.2 and §7.6. The parts are ordered by published part name, every panel is
    opened while all of the paths are locked — so every part sees the same
    store bytes — and the selected marks are composed one after another over
    that one capture. The result is a single final pattern payload every part
    shares, rather than several independently prepared writes to one path that
    bind different after-digests and refuse each other.

    Publishes nothing: a batch that is never applied leaves no trace.
    """
    if not requests:
        raise ReviewPanelError("An aggregate review prepares at least one part")
    ordered = sorted(requests, key=lambda request: request.part_name)
    names = [request.part_name for request in ordered]
    if len(set(names)) != len(names):
        raise ReviewPanelError("Each part in one review batch has its own name")
    resolved = [
        review_target_paths(
            request.staging_path, staging_dir=staging_dir, patterns_path=patterns_path
        )
        for request in ordered
    ]
    pattern_path = resolved[0][1]
    staging_paths = [active for active, _pattern in resolved]
    if len(set(staging_paths)) != len(staging_paths):
        raise ReviewPanelError("Each part in one review batch has its own file")

    with _sorted_path_locks([*staging_paths, pattern_path]):
        panels = [
            ReviewPanel.open_under_lock(
                active,
                staging_dir=staging_dir,
                patterns_path=pattern_path,
                collection_name=collection_name,
            )
            for active in staging_paths
        ]
        for request, panel in zip(ordered, panels, strict=True):
            if request.expected_revision is not None and not hmac.compare_digest(
                panel.staging_fingerprint, request.expected_revision
            ):
                raise StaleReviewError(
                    f"[staging-review-stale] {panel.staging_path.name} changed "
                    "before its review could be prepared; nothing was reviewed."
                )
        first = panels[0]
        for panel in panels[1:]:
            # Read under one set of locks at one moment, so two different
            # captures of the same path would mean the store moved under them.
            if (
                panel.patterns_bytes != first.patterns_bytes
                or panel.patterns_absent != first.patterns_absent
            ):
                raise StaleReviewError(
                    "The pattern store changed while this review was prepared"
                )
        parts = tuple(
            panel.prepare_part(
                part_name=request.part_name,
                record_ids=request.record_ids,
                review_patterns=request.review_patterns,
                coverage_approval=request.coverage_approval,
                retain_marked_patterns=True,
            )
            for request, panel in zip(ordered, panels, strict=True)
        )
        captured = first._captured_pattern_text()
        composed = captured
        # §7.2 exactly: once per selected source entry **whose mark is false**.
        # An already-true entry is left alone — the owner's choice is retained
        # in the part, and it contributes no write — and a second part naming
        # an entry an earlier one already marked has nothing left to change
        # either.
        marked: set[str] = set()
        for part, panel in zip(parts, panels, strict=True):
            if part.review_patterns and not panel.pattern_marked:
                if panel.source in marked:
                    continue
                composed = panel.render_reviewed_pattern_store(composed)
                marked.add(panel.source)
        return PreparedReviewBatch(
            parts=parts,
            patterns=_review_component(
                "patterns",
                pattern_path,
                before=None if first.patterns_absent else first.patterns_fingerprint,
                after_text=None if composed == captured else composed,
            ),
        )


def apply_prepared_review_batch_under_locks(
    batch: PreparedReviewBatch,
) -> ReviewBatchOutcome:
    """Apply one aggregate review while the caller holds every review lock."""
    return _apply_prepared_review_batch_under_locks(batch, resume=False)


def apply_prepared_review_batch(batch: PreparedReviewBatch) -> ReviewBatchOutcome:
    """Take every review lock in order, then apply one aggregate review."""
    with review_batch_path_locks(batch):
        return _apply_prepared_review_batch_under_locks(batch, resume=False)


def recover_prepared_review_batch_under_locks(
    batch: PreparedReviewBatch,
) -> ReviewBatchOutcome:
    """Finish one partially applied aggregate review, locks already held."""
    return _apply_prepared_review_batch_under_locks(batch, resume=True)


def recover_prepared_review_batch(batch: PreparedReviewBatch) -> ReviewBatchOutcome:
    """Finish one partially applied aggregate review from its intent alone."""
    with review_batch_path_locks(batch):
        return _apply_prepared_review_batch_under_locks(batch, resume=True)


def _apply_prepared_review_under_locks(
    prepared: PreparedReview, *, resume: bool
) -> ReviewOutcome:
    """One prepared review through the aggregate writer that owns these paths.

    A single panel is a batch of one. There is one low-level writer, so the
    ordinary submit and a study job's whole reviewed phase precheck, write and
    recover by the same rules — and the one-panel outcomes and refusals are
    exactly what they always were.
    """
    batch = prepared.as_batch()
    try:
        outcome = _apply_prepared_review_batch_under_locks(batch, resume=resume)
    except PartialReviewBatchError as exc:
        raise PartialReviewError(str(exc), exc.outcome.parts[_ONE_PANEL]) from exc
    return outcome.parts[_ONE_PANEL]


def apply_prepared_review_under_locks(prepared: PreparedReview) -> ReviewOutcome:
    """Apply one prepared review while the caller holds both review locks."""
    return _apply_prepared_review_under_locks(prepared, resume=False)


def apply_prepared_review(prepared: PreparedReview) -> ReviewOutcome:
    """Take both review locks in order, then apply one prepared review."""
    with review_path_locks(prepared):
        return _apply_prepared_review_under_locks(prepared, resume=False)


def recover_prepared_review_under_locks(prepared: PreparedReview) -> ReviewOutcome:
    """Finish one partially applied review from its intent, locks already held."""
    return _apply_prepared_review_under_locks(prepared, resume=True)


def recover_prepared_review(prepared: PreparedReview) -> ReviewOutcome:
    """Finish one partially applied review from its persisted intent alone.

    From the intent, and from nothing else: a returned value that was never
    persisted is not evidence, and neither is a writer's state label. Each
    bound path is re-measured against its own before/after pair, the missing
    writes are finished, and a path at neither digest refuses and names both.
    """
    with review_path_locks(prepared):
        return _apply_prepared_review_under_locks(prepared, resume=True)


def _staged_lineage(
    records: Sequence[VocabularyRecord],
    meta: Mapping[str, Any],
) -> tuple[str, str, dict[str, Any], patterns.PatternSet | None]:
    """The rich staging lineage, or the absence of one.

    **Absent is not invalid.** A file `janki extract` wrote carries a review
    run id, prompt provenance and a pattern answer, and every approval binds
    to them: an example approval means "a person accepted the sentence *this
    model proposed*", which is meaningless without the answer it proposed.

    A file that arrived another way — an Anki import, a hand-written review —
    has none of that, and it is not broken for lacking it. Refusing to open it
    at all took the whole page down with the approvals: no editing a typo'd
    reading, no re-identifying a row, on a file whose rows are perfectly
    ordinary. So a missing lineage yields empty values and the caller offers
    less, while a *malformed* one still raises.
    """
    try:
        staging.validate_coverage_facts(meta)
        run_id = staging.rich_extraction_review_run_id(meta)
    except staging.StagingError as exc:
        raise ReviewPanelError(f"The active staging lineage is invalid: {exc}") from exc
    provenance = meta.get("prompt_provenance")
    source_value = meta.get("source_file")
    if not isinstance(source_value, str) or not source_value.strip():
        # The one part every staging file must have: without it nothing can
        # say which source these rows belong to, and the pattern store, the
        # archive name and the dashboard all key on it.
        raise ReviewPanelError("The active staging file names no source")
    if source_value != Path(source_value).name or "/" in source_value or "\\" in source_value:
        # Checked for every file, lineage or not: a path-shaped source name is
        # a key that escapes the store, and that is true however the rows
        # arrived.
        raise ReviewPanelError(
            "The staging source_file must be the exact basename-shaped pattern store key"
        )
    if not isinstance(provenance, Mapping):
        # Either no run at all — an import, or a review somebody wrote by hand
        # — or an `enrich --ai` pass, which carries a run id and an
        # `ai_enrichment` block and never extraction provenance. Both open;
        # what they may *do* differs, and `ReviewPanel.provenance_kind` is
        # where that is decided.
        return source_value, "", {}, None
    source = source_value
    nested = meta.get("pattern_set")
    if not isinstance(nested, dict):
        raise ReviewPanelError(
            "The active staging file has no structurally valid nested staged pattern answer"
        )
    try:
        staged_patterns = patterns.PatternSet.from_dict(source, nested)
    except patterns.PatternError as exc:
        raise ReviewPanelError(f"Invalid nested staged pattern answer: {exc}") from exc
    if staged_patterns.review_run_id != run_id or dict(staged_patterns.prompt_provenance) != dict(
        provenance
    ):
        raise ReviewPanelError(
            "The nested staged pattern answer does not match the staging run/provenance"
        )
    ids = [record.id for record in records]
    if len(ids) != len(set(ids)):
        raise ReviewPanelError(
            "The staging file has duplicate record IDs, so row review actions "
            "cannot identify one row safely"
        )
    return source, run_id, dict(provenance), staged_patterns


def _pattern_warning(
    current: patterns.PatternSet | None,
    *,
    source: str,
    run_id: str,
    provenance: Mapping[str, Any],
) -> str | None:
    if not run_id:
        return (
            "This source did not come from an extraction, so there is no model "
            "answer to review. You can still edit its cards and correct a word "
            "it identified wrongly."
        )
    if current is None:
        return (
            f"The current pattern store has no entry for the exact key {source!r}. "
            "Pattern review is disabled; card review remains available."
        )
    if current.review_run_id != run_id or dict(current.prompt_provenance) != dict(provenance):
        return (
            "The current pattern-store entry does not exactly match this staging "
            "review run (review_run_id and prompt_provenance must both match). "
            "Pattern review is disabled; card review remains available."
        )
    return None


def parse_review_form(body: bytes) -> dict[str, list[str]]:
    """Parse and structurally validate an approval form body.

    Shared because the workbench and the panel must agree on what a malformed
    submission *is*. Everything here is about shape — encoding, unknown keys,
    how many times a field may appear, the literal action word — and none of it
    is about authority: the caller still checks the CSRF token and the exact
    snapshot the form claims to have been rendered from.
    """
    try:
        text = body.decode("utf-8", errors="strict")
        pairs = parse_qsl(
            text,
            keep_blank_values=True,
            strict_parsing=True,
            encoding="utf-8",
            errors="strict",
            max_num_fields=4096,
        )
    except (UnicodeError, ValueError) as exc:
        raise PanelRequestError("The submitted form is malformed") from exc
    unknown = sorted({key for key, _value in pairs} - _FORM_FIELDS)
    if unknown:
        raise PanelRequestError("Unknown form action: " + ", ".join(unknown))
    grouped: dict[str, list[str]] = {}
    for key, value in pairs:
        grouped.setdefault(key, []).append(value)
    for key in _FORM_SINGLETONS:
        count = len(grouped.get(key, []))
        expected = 0 if key == "patterns" else 1
        if count not in ({0, 1} if key == "patterns" else {expected}):
            raise PanelRequestError(
                f"Form action {key!r} must appear "
                + ("at most once" if key == "patterns" else "exactly once")
            )
    if grouped["action"][0] != "save":
        raise PanelRequestError("The form action must be 'save'")
    pattern_values = grouped.get("patterns", [])
    if pattern_values and pattern_values != ["review"]:
        raise PanelRequestError("The pattern action is invalid")
    return grouped


@dataclass(slots=True)
class ReviewPanel:
    staging_path: Path
    staging_dir: Path
    patterns_path: Path
    records: list[VocabularyRecord]
    meta: dict[str, Any]
    store: dict[str, patterns.PatternSet]
    source: str
    run_id: str
    provenance: dict[str, Any]
    #: The pattern answer this staging run was extracted with, or None
    #: when the file did not come from an extraction at all.
    staged_pattern_set: patterns.PatternSet | None
    pattern_set: patterns.PatternSet | None
    pattern_warning: str | None
    staging_bytes: bytes
    patterns_bytes: bytes
    #: The configured collection's filename. Needed to recognise an
    #: `enrich --ai` review written before those files carried a marker: what
    #: they name as their source is the collection itself.
    collection_name: str = ""
    #: Whether ``patterns_bytes`` is the synthetic ``{}`` of a project that
    #: has no pattern store rather than a store that really holds one.
    patterns_absent: bool = False
    _submission_lock: threading.RLock = field(
        default_factory=threading.RLock,
        repr=False,
    )

    @classmethod
    def open(
        cls,
        staging_path: Path,
        *,
        staging_dir: Path,
        patterns_path: Path,
        collection_name: str = "",
    ) -> ReviewPanel:
        active_path, pattern_path = review_target_paths(
            staging_path, staging_dir=staging_dir, patterns_path=patterns_path
        )
        with _sorted_path_locks((active_path, pattern_path)):
            return cls.open_under_lock(
                active_path,
                staging_dir=staging_dir,
                patterns_path=pattern_path,
                collection_name=collection_name,
            )

    @classmethod
    def open_under_lock(
        cls,
        staging_path: Path,
        *,
        staging_dir: Path,
        patterns_path: Path,
        collection_name: str = "",
    ) -> ReviewPanel:
        """Capture a panel while the caller owns its staging and pattern locks."""

        # Recheck after acquiring the locks: a cooperating writer may have
        # replaced either file while this process was waiting.
        active_path, pattern_path = review_target_paths(
            staging_path, staging_dir=staging_dir, patterns_path=patterns_path
        )
        staging_bytes = _capture_regular_bytes(active_path, "The active staging file")
        # An absent store captures as an empty document rather than refusing:
        # there is nothing to compare-and-swap against, and pattern review is
        # unavailable for such a project anyway.
        patterns_bytes, patterns_absent = _capture_regular_state(
            pattern_path,
            "The pattern store",
            absent_ok=True,
        )
        try:
            staging_text = staging_bytes.decode("utf-8", errors="strict")
            patterns_text = patterns_bytes.decode("utf-8", errors="strict")
            records, meta = staging.read_staging_text(
                staging_text,
                source=str(active_path),
            )
            store = patterns.load_store_text(
                patterns_text,
                source=str(pattern_path),
            )
            source, run_id, provenance, staged_pattern_set = _staged_lineage(
                records,
                meta,
            )
        except (UnicodeError, JankiError) as exc:
            if isinstance(exc, ReviewPanelError):
                raise
            raise ReviewPanelError(f"Could not open the review snapshot: {exc}") from exc
        pattern_set = store.get(source)
        pattern_warning = _pattern_warning(
            pattern_set,
            source=source,
            run_id=run_id,
            provenance=provenance,
        )
        return cls(
            staging_path=active_path,
            staging_dir=_absolute(staging_dir),
            patterns_path=pattern_path,
            records=records,
            meta=meta,
            store=store,
            source=source,
            run_id=run_id,
            provenance=provenance,
            staged_pattern_set=staged_pattern_set,
            pattern_set=pattern_set,
            pattern_warning=pattern_warning,
            staging_bytes=staging_bytes,
            patterns_bytes=patterns_bytes,
            collection_name=collection_name,
            patterns_absent=patterns_absent,
        )

    @property
    def staging_fingerprint(self) -> str:
        return _fingerprint(self.staging_bytes)

    @property
    def patterns_fingerprint(self) -> str:
        return _fingerprint(self.patterns_bytes)

    @property
    def reviewable_record_ids(self) -> frozenset[str]:
        return frozenset(
            record.id
            for record in self.records
            if needs_example_review(record)
        )

    @property
    def provenance_kind(self) -> str:
        """Where this review's rows came from, which decides what may be done.

        ``extraction`` — a `janki extract` run: a review run id, prompt
        provenance and the pattern answer that run proposed. Everything is
        available, because every approval has something to bind to.

        ``model-pass`` — an `enrich --ai` review, recognised by
        `staging.is_model_pass`. A paid answer about words the collection
        already holds. A modern one carries provenance maps `promote` checks
        against the file; one written before those existed carries only its
        `source_file` and a model name, and promotes as an ordinary file.
        Either way: editable and removable, and **not** re-identifiable,
        because the answer was produced for the word it was asked about and
        re-keying it would claim the model spoke about a different one.

        ``none`` — an import, or a review written by hand. Ordinary rows and no
        model claims: edit, remove and re-identify freely, approve nothing.
        """
        if bool(self.run_id) and self.staged_pattern_set is not None:
            return "extraction"
        if staging.is_model_pass(self.meta, collection_name=self.collection_name):
            return "model-pass"
        return "none"

    @property
    def has_extraction_lineage(self) -> bool:
        return self.provenance_kind == "extraction"

    @property
    def reidentifiable(self) -> bool:
        """Whether a row here may be told it is a different word.

        Not on a model pass. The request that produced these answers named a
        specific record, and its input fingerprint binds that turn — so moving
        the answer to another identity records that a model said something it
        never said about a word it never saw.
        """
        return self.provenance_kind != "model-pass"

    @property
    def pattern_reviewable(self) -> bool:
        return self.pattern_selectable and not self.pattern_marked

    @property
    def pattern_selectable(self) -> bool:
        """Whether this panel's store entry is one a review may name at all.

        Lineage only. Whether the mark is already `true` is a separate
        question, because §7.2 answers it differently for one panel and for an
        aggregate: the page renders a settled entry display-only, while a batch
        retains the owner's choice and simply has nothing to write for it.
        """
        return self.pattern_warning is None and self.pattern_set is not None

    @property
    def pattern_marked(self) -> bool:
        """Whether the captured store already records this source as reviewed."""
        return self.pattern_set is not None and self.pattern_set.reviewed

    def _validate_actions(
        self,
        record_ids: Sequence[str],
        review_patterns: bool,
        *,
        retain_marked_patterns: bool = False,
    ) -> tuple[str, ...]:
        selected = tuple(record_ids)
        if len(selected) != len(set(selected)):
            raise PanelRequestError("A record review action was submitted more than once")
        unavailable = sorted(set(selected) - self.reviewable_record_ids)
        if unavailable:
            raise PanelRequestError(
                "Record review action is not reviewable on this page: " + ", ".join(unavailable)
            )
        if selected and not self.has_extraction_lineage:
            # Approving an example means accepting the sentence a particular
            # model proposed on a particular run. A file with no run cannot
            # carry that, so the approval would bind to nothing — and a page
            # that never offers the control still has to refuse the request,
            # because the request is what writes.
            raise PanelRequestError(
                "This source did not come from an extraction, so its sentences "
                "have no proposal to approve"
            )
        if review_patterns:
            if not self.pattern_selectable:
                raise PanelRequestError(
                    "Pattern review is unavailable because the current store lineage "
                    "does not exactly match this staging run"
                )
            if self.pattern_marked and not retain_marked_patterns:
                raise PanelRequestError(
                    "The pattern set is already reviewed and display-only"
                )
        return selected

    def validate_actions(
        self,
        record_ids: Sequence[str],
        review_patterns: bool,
        *,
        retain_marked_patterns: bool = False,
    ) -> tuple[str, ...]:
        """Validate one exact review selection without writing it.

        The browser used to be the only caller that needed this distinction:
        it rendered checkboxes and later called :meth:`submit`.  An Assistant
        confirmation has the same two phases, so its application-layer plan
        must be able to prove that the exact selected rows and pattern action
        are currently offerable without manufacturing approval merely by
        planning them.

        ``retain_marked_patterns`` is §7.2's aggregate rule and nothing wider:
        a selected mark that is already `true` is *retained* rather than
        refused, and contributes no write. Only :meth:`prepare_part`'s batch
        caller passes it; the ordinary page keeps its display-only refusal.
        """

        return self._validate_actions(
            record_ids,
            review_patterns,
            retain_marked_patterns=retain_marked_patterns,
        )

    def _reviewed_records(self, selected: Sequence[str]) -> list[VocabularyRecord]:
        """The captured rows with exact example fingerprints on the selected ones.

        Uses only the exact objects and bytes rendered in the browser. The
        advisory locks coordinate janki writers; the expected-revision bound
        replaces are what stop a non-cooperating editor at the final seam.
        """
        by_id = {record.id: index for index, record in enumerate(self.records)}
        updated = list(self.records)
        for record_id in selected:
            record = updated[by_id[record_id]]
            updated[by_id[record_id]] = set_example_flags(
                record,
                EXAMPLE_AUTHORITY_KEY,
                (example.japanese for example in record.examples if example.japanese),
            )
        return updated

    def _reviewed_store(self) -> dict[str, patterns.PatternSet]:
        assert self.pattern_set is not None  # proved by _validate_actions
        updated_store = dict(self.store)
        updated_store[self.source] = replace(self.pattern_set, reviewed=True)
        return updated_store

    def render_reviewed_pattern_store(self, captured_text: str) -> str:
        """This part's owner-selected mark applied to a captured store text.

        Takes the text rather than reading `self.patterns_bytes`, because §7.2
        composes **all** the selected marks over one captured store: part two
        renders over part one's result, so the batch's single final payload
        carries every part's mark and one part's choice cannot erase another's.
        `patterns.render_reviewed_update` refuses an already-true mark, and it
        keeps that refusal: it is the raw renderer, and §7.2's "apply it once
        per selected entry **whose mark is false**" is the caller's rule. The
        batch composition below applies it only where the mark changes.
        """
        try:
            return patterns.render_reviewed_update(
                captured_text,
                self.source,
                source=str(self.patterns_path),
            )
        except (UnicodeError, patterns.PatternError) as exc:
            raise ReviewPanelError(
                f"Could not render the exact pattern review: {exc}"
            ) from exc

    def prepare_part(
        self,
        *,
        part_name: str = "",
        record_ids: Sequence[str],
        review_patterns: bool,
        coverage_approval: Mapping[str, Any] | None = None,
        retain_marked_patterns: bool = False,
    ) -> PreparedReviewPart:
        """This part's staging half of a review, computed without writing.

        The staging payload is whole: the review marks and the owner's exact
        coverage payload composed over the same captured bytes, in that order.
        The pattern store is left to the batch, which owns its one final
        payload — including whether this part's selected mark has anything to
        contribute to it, which is what ``retain_marked_patterns`` allows.
        """
        selected = self.validate_actions(
            record_ids, review_patterns, retain_marked_patterns=retain_marked_patterns
        )
        reviewed = self._reviewed_records(selected)
        staging_text: str | None = None
        if selected or coverage_approval is not None:
            try:
                text = self.staging_bytes.decode("utf-8", errors="strict")
                if selected:
                    # First, and over the exact captured browser bytes: the
                    # surgical renderer's whole promise is that it edits the
                    # snapshot the owner saw, one authority line per reviewed
                    # row, leaving every other byte alone.
                    text = staging.render_example_authority_updates(
                        text,
                        reviewed,
                        source=str(self.staging_path),
                    )
                if coverage_approval is not None:
                    # Then the coverage approval, whose whole-document
                    # round-trip is the writer `record_coverage_approval`
                    # already proves preserves comments and unknown keys. The
                    # two do not commute byte-wise, so the order is part of
                    # the contract rather than a detail.
                    text = staging.render_coverage_approval(
                        text,
                        coverage_approval,
                        source=str(self.staging_path),
                    )
                staging_text = text
            except (UnicodeError, staging.StagingError) as exc:
                raise ReviewPanelError(
                    f"Could not render the exact staging approval: {exc}"
                ) from exc
        by_id = {record.id: record for record in reviewed}
        approved_at = (
            coverage_approval.get("approved_at")
            if coverage_approval is not None
            else None
        )
        return PreparedReviewPart(
            part_name=part_name,
            source=self.source,
            staging=_review_component(
                "staging",
                self.staging_path,
                before=self.staging_fingerprint,
                after_text=staging_text,
            ),
            record_ids=selected,
            review_patterns=review_patterns,
            expected_authority={
                record_id: str(
                    by_id[record_id].source.raw_fields[EXAMPLE_AUTHORITY_KEY]
                )
                for record_id in selected
            },
            coverage_approved_at=None if approved_at is None else str(approved_at),
        )

    def prepare(
        self,
        *,
        record_ids: Sequence[str],
        review_patterns: bool,
        coverage_approval: Mapping[str, Any] | None = None,
    ) -> PreparedReview:
        """The exact bytes this review would write, computed without writing.

        Side-effect-free on every published target: it reads nothing beyond the
        snapshot the panel already captured and publishes nothing another reader
        treats as content.

        Selecting nothing is a legitimate preparation, unlike submitting
        nothing. A study job folds every part it owns, including one whose rows
        were reviewed and whose coverage was accepted on an earlier pass; that
        part still has to be *bound*, so both components come back at the same
        before and after digest — an external edit to either still refuses the
        apply — and applying writes nothing. The alternative was fabricating an
        owner decision nobody made.

        ``coverage_approval`` is the owner's exact payload with its frozen
        approval date, when the same click also records one. It is **composed
        into the same staging after-text** rather than written separately: the
        two pure renderers run over the same captured bytes, so no
        review-only intermediate document is ever published and one
        compare-and-swap covers both decisions. Supplying a payload grants no
        coverage authority — the caller brings one it already holds.
        """
        part = self.prepare_part(
            record_ids=record_ids,
            review_patterns=review_patterns,
            coverage_approval=coverage_approval,
        )
        patterns_text: str | None = None
        if review_patterns:
            patterns_text = self.render_reviewed_pattern_store(
                self._captured_pattern_text()
            )
        return PreparedReview(
            components=(
                part.staging,
                _review_component(
                    "patterns",
                    self.patterns_path,
                    before=None if self.patterns_absent else self.patterns_fingerprint,
                    after_text=patterns_text,
                ),
            ),
            record_ids=part.record_ids,
            review_patterns=part.review_patterns,
            expected_authority=part.expected_authority,
            coverage_approved_at=part.coverage_approved_at,
        )

    def _captured_pattern_text(self) -> str:
        """The captured store bytes as text, or a refusal naming the store."""
        try:
            return self.patterns_bytes.decode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise ReviewPanelError(
                f"Could not render the exact pattern review: {exc}"
            ) from exc

    def _recheck_targets(self) -> None:
        """Prove both review paths are still the shapes this panel captured."""
        try:
            _active_staging_path(self.staging_path, self.staging_dir)
            # Absent stays acceptable here, exactly as at open. Strict, a
            # card approval on a project with no pattern store was refused
            # with "a review target path changed" — nothing had changed,
            # the store never existed, and the approval writes only the
            # staging file.
            _require_regular_non_symlink(
                self.patterns_path, "The pattern store", absent_ok=True
            )
        except ReviewPanelError as exc:
            raise StaleReviewError(
                "A review target path changed after this page was rendered"
            ) from exc

    def submit(self, *, record_ids: Sequence[str], review_patterns: bool) -> ReviewOutcome:
        """Write one review, taking the shared curation guard outermost.

        §6.5 and contracts §7.5: the staging mutation coordination guard comes
        before any other lock on **every** entry that writes staged bytes, and
        this is the ordinary one — the workbench review page and the Assistant's
        confirmed review both land here. A caller already inside the guard uses
        :meth:`submit_under_guard`; `io.exclusive_path_lock` is not re-entrant,
        so taking it twice from one thread deadlocks rather than nesting.
        """
        # Imported at call time, not at module scope: the curation service
        # reads the study job store, which reaches the application aggregate,
        # and this module sits below it in the import graph. Promotion's
        # `_pending_curation` takes the same route for the same reason.
        from japanese_anki.application import study_curation

        selected = self.validate_actions(record_ids, review_patterns)
        if not selected and not review_patterns:
            # Nothing staged changes, so there is nothing for the guard to
            # coordinate. Checked before it is taken, so a no-op review never
            # waits behind a curation.
            return ReviewOutcome()
        with study_curation.staging_curation_guard(self.staging_dir):
            return self.submit_under_guard(
                record_ids=selected, review_patterns=review_patterns
            )

    def submit_under_guard(
        self, *, record_ids: Sequence[str], review_patterns: bool
    ) -> ReviewOutcome:
        """Write one review while the caller already holds the shared guard."""
        selected = self.validate_actions(record_ids, review_patterns)
        if not selected and not review_patterns:
            return ReviewOutcome()
        with self._submission_lock, _sorted_path_locks((self.staging_path, self.patterns_path)):
            self._recheck_targets()
            prepared = self.prepare(
                record_ids=selected,
                review_patterns=review_patterns,
            )
            saved = apply_prepared_review_under_locks(prepared)
            if prepared.staging.after_text is not None:
                self.records = self._reviewed_records(selected)
                self.staging_bytes = prepared.staging.after_text.encode("utf-8")
            if prepared.patterns.after_text is not None:
                self.store = self._reviewed_store()
                self.pattern_set = self.store[self.source]
                self.patterns_bytes = prepared.patterns.after_text.encode("utf-8")
            return saved
