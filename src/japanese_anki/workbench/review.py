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
import os
import stat
import threading
from collections.abc import Iterable, Mapping, Sequence
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
from japanese_anki.io import atomic_write_text_bound, exclusive_path_lock
from japanese_anki.models import (
    EXAMPLE_AUTHORITY_KEY,
    VocabularyRecord,
    set_example_flags,
)

__all__ = [
    "PanelRequestError",
    "PartialReviewError",
    "ReviewOutcome",
    "IndeterminateWriteError",
    "ReviewPanel",
    "bound_replace",
    "ReviewPanelError",
    "StaleReviewError",
    "parse_review_form",
]


_FORM_SINGLETONS = frozenset(
    {"csrf", "staging_snapshot", "patterns_snapshot", "action", "patterns"}
)
_FORM_FIELDS = _FORM_SINGLETONS | {"record"}


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


def bound_replace(path: Path, text: str, snapshot: bytes, *, label: str) -> None:
    """CAS one rendered snapshot and classify a final-seam interference safely."""
    intended = text.encode("utf-8")
    try:
        atomic_write_text_bound(
            path,
            text,
            expected_revision=_fingerprint(snapshot),
        )
    except Exception as exc:
        try:
            is_link = path.is_symlink()
            live = None if is_link else path.read_bytes()
        except OSError:
            is_link = False
            live = None
        message = str(exc).lower()
        if (
            is_link
            or "bound target changed" in message
            or "refusing to replace non-regular" in message
        ):
            raise StaleReviewError(f"The {label} changed at the final review write") from exc
        if live == snapshot:
            raise ReviewPanelError(
                f"Could not save the {label}; its captured bytes remain unchanged: {exc}"
            ) from exc
        raise IndeterminateWriteError(
            f"The {label} write failed after its exact snapshot stopped matching; "
            "restart the panel and inspect the recorded decisions",
            intended_bytes_are_live=live == intended,
        ) from exc


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


def _open_no_follow(path: Path) -> int:
    """Open one exact path without following a swapped final symlink."""
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise ReviewPanelError("This platform cannot safely capture review files")
    return os.open(path, os.O_RDONLY | no_follow | getattr(os, "O_CLOEXEC", 0))


def _read_open_fd(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    while chunk := os.read(descriptor, 64 * 1024):
        chunks.append(chunk)
    return b"".join(chunks)


def _capture_regular_bytes(path: Path, description: str) -> bytes:
    """Capture bytes from a no-follow fd still bound to this exact path."""
    try:
        descriptor = _open_no_follow(path)
    except (OSError, ReviewPanelError) as exc:
        raise ReviewPanelError(
            f"{description} must remain a direct regular non-symlink file during capture: {path}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ReviewPanelError(
                f"{description} must remain a direct regular non-symlink file during capture: "
                f"{path}"
            )
        captured = _read_open_fd(descriptor)
        finished = os.fstat(descriptor)
        try:
            bound = path.lstat()
        except OSError as exc:
            raise ReviewPanelError(f"{description} path changed during capture: {path}") from exc
        fd_identity = (finished.st_dev, finished.st_ino)
        path_identity = (bound.st_dev, bound.st_ino)
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if (
            not stat.S_ISREG(bound.st_mode)
            or stat.S_ISLNK(bound.st_mode)
            or fd_identity != path_identity
            or any(getattr(opened, field) != getattr(finished, field) for field in stable_fields)
            or len(captured) != finished.st_size
        ):
            raise ReviewPanelError(f"{description} path changed during capture: {path}")
        return captured
    finally:
        os.close(descriptor)


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


@contextlib.contextmanager
def _sorted_path_locks(paths: Iterable[Path]):
    """Lock distinct real targets in one process-wide deterministic order."""
    real = {Path(os.path.realpath(path)) for path in paths}
    if len(real) != 2:
        raise ReviewPanelError("The staging file and pattern store must be distinct")
    with ExitStack() as stack:
        for path in sorted(real, key=os.fspath):
            stack.enter_context(exclusive_path_lock(path))
        yield


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
        active_path = _active_staging_path(staging_path, staging_dir)
        pattern_path = _absolute(patterns_path)
        _require_regular_non_symlink(
            pattern_path, "The pattern store", absent_ok=True
        )
        with _sorted_path_locks((active_path, pattern_path)):
            # Recheck after acquiring the lock: a cooperating writer may have
            # replaced the file while this process was waiting.
            _active_staging_path(active_path, staging_dir)
            _require_regular_non_symlink(
                pattern_path, "The pattern store", absent_ok=True
            )
            staging_bytes = _capture_regular_bytes(active_path, "The active staging file")
            # An absent store captures as an empty document rather than
            # refusing: there is nothing to compare-and-swap against, and
            # pattern review is unavailable for such a project anyway.
            patterns_bytes = (
                _capture_regular_bytes(pattern_path, "The pattern store")
                if pattern_path.exists()
                else b"{}"
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
        return (
            self.pattern_warning is None
            and self.pattern_set is not None
            and not self.pattern_set.reviewed
        )

    def _validate_actions(
        self, record_ids: Sequence[str], review_patterns: bool
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
        if review_patterns and not self.pattern_reviewable:
            if self.pattern_warning is not None or self.pattern_set is None:
                raise PanelRequestError(
                    "Pattern review is unavailable because the current store lineage "
                    "does not exactly match this staging run"
                )
            raise PanelRequestError("The pattern set is already reviewed and display-only")
        return selected

    def submit(self, *, record_ids: Sequence[str], review_patterns: bool) -> ReviewOutcome:
        selected = self._validate_actions(record_ids, review_patterns)
        if not selected and not review_patterns:
            return ReviewOutcome()
        with self._submission_lock, _sorted_path_locks((self.staging_path, self.patterns_path)):
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
            live_staging = self.staging_path.read_bytes()
            live_patterns = (
                self.patterns_path.read_bytes()
                if self.patterns_path.exists()
                else b"{}"
            )
            if live_staging != self.staging_bytes:
                raise StaleReviewError(
                    "The staging file changed after this review page was rendered"
                )
            if live_patterns != self.patterns_bytes:
                raise StaleReviewError(
                    "The pattern store changed after this review page was rendered"
                )
            # Every transformation below uses only the exact objects and bytes
            # rendered in the browser. The advisory locks coordinate janki
            # writers; the expected-revision bound replaces below are what stop
            # a non-cooperating editor at the final seam.
            by_id = {record.id: index for index, record in enumerate(self.records)}
            updated = list(self.records)
            for record_id in selected:
                record = updated[by_id[record_id]]
                updated[by_id[record_id]] = set_example_flags(
                    record,
                    EXAMPLE_AUTHORITY_KEY,
                    (example.japanese for example in record.examples if example.japanese),
                )

            try:
                staging_text = (
                    staging.render_example_authority_updates(
                        self.staging_bytes.decode("utf-8", errors="strict"),
                        updated,
                        source=str(self.staging_path),
                    )
                    if selected
                    else None
                )
            except (UnicodeError, staging.StagingError) as exc:
                raise ReviewPanelError(
                    f"Could not render the exact staging approval: {exc}"
                ) from exc
            updated_store = dict(self.store)
            patterns_text: str | None = None
            if review_patterns:
                assert self.pattern_set is not None  # proved by _validate_actions
                updated_store[self.source] = replace(self.pattern_set, reviewed=True)
                try:
                    patterns_text = patterns.render_reviewed_update(
                        self.patterns_bytes.decode("utf-8", errors="strict"),
                        self.source,
                        source=str(self.patterns_path),
                    )
                except (UnicodeError, patterns.PatternError) as exc:
                    raise ReviewPanelError(
                        f"Could not render the exact pattern review: {exc}"
                    ) from exc

            saved = ReviewOutcome()
            if staging_text is not None:
                try:
                    bound_replace(
                        self.staging_path,
                        staging_text,
                        self.staging_bytes,
                        label="staging file",
                    )
                except IndeterminateWriteError as exc:
                    outcome = ReviewOutcome(
                        selected if exc.intended_bytes_are_live else (),
                        False,
                    )
                    raise PartialReviewError(str(exc), outcome) from exc
                saved = ReviewOutcome(selected, False)

            # There is a deliberately tiny process-crash window between two
            # requested atomic replaces. A WAL would be disproportionate for
            # independent, idempotent, monotonic review marks. Once the first
            # exact approval lands it is never rolled back: doing so could
            # erase a concurrent human edit made before a later failure.
            if patterns_text is not None:
                try:
                    bound_replace(
                        self.patterns_path,
                        patterns_text,
                        self.patterns_bytes,
                        label="pattern store",
                    )
                except IndeterminateWriteError as exc:
                    outcome = ReviewOutcome(
                        saved.accepted_record_ids,
                        exc.intended_bytes_are_live,
                    )
                    raise PartialReviewError(str(exc), outcome) from exc
                except (StaleReviewError, ReviewPanelError) as exc:
                    if saved.accepted_record_ids:
                        raise PartialReviewError(
                            f"The card approval was saved, but the pattern review was not: {exc}",
                            saved,
                        ) from exc
                    raise
                saved = ReviewOutcome(saved.accepted_record_ids, True)

            if staging_text is not None:
                self.records = updated
                self.staging_bytes = staging_text.encode("utf-8")
            if patterns_text is not None:
                self.store = updated_store
                self.pattern_set = updated_store[self.source]
                self.patterns_bytes = patterns_text.encode("utf-8")
            return saved
