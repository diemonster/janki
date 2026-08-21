"""A narrow localhost surface for extraction-review attestations.

The panel renders one active staging file and the matching current pattern
store entry.  It does not edit Japanese, promote rows, or make network/model
calls.  Its only writes are the two existing human-review marks:

* exact fingerprints for every nonblank Japanese example shown on selected rows;
* ``reviewed = true`` on the matching pattern-store entry.

The HTTP layer is intentionally small and dependency-free.  All substantive
state checks also live in :class:`ReviewPanel`, so a future CLI can start the
server without becoming another review implementation.
"""

from __future__ import annotations

import contextlib
import hashlib
import html
import os
import re
import secrets
import stat
import threading
from collections.abc import Iterable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl

from japanese_anki import patterns, staging
from japanese_anki.errors import JankiError
from japanese_anki.io import atomic_write_text_bound, exclusive_path_lock
from japanese_anki.models import (
    EXAMPLE_AUTHORITY_KEY,
    EXAMPLE_AUTHORITY_STAGING,
    VocabularyRecord,
    example_accepted,
    set_example_flags,
)

__all__ = [
    "MAX_BODY_BYTES",
    "REQUEST_TIMEOUT_SECONDS",
    "PanelRequestError",
    "PartialReviewError",
    "ReviewOutcome",
    "ReviewPanel",
    "ReviewPanelError",
    "StaleReviewError",
    "make_server",
]


MAX_BODY_BYTES = 64 * 1024
REQUEST_TIMEOUT_SECONDS = 5.0
_BOUND_AUTHORITY = re.compile(r"\s*[0-9a-f]{12}(?:\s*,\s*[0-9a-f]{12})*\s*\Z")
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


class _PanelSecurityError(PanelRequestError):
    """The localhost request did not come from this exact review session."""


@dataclass(frozen=True, slots=True)
class ReviewOutcome:
    accepted_record_ids: tuple[str, ...] = ()
    pattern_reviewed: bool = False


class PartialReviewError(ReviewPanelError):
    """At least one exact monotonic approval landed before a later failure."""

    def __init__(self, message: str, outcome: ReviewOutcome) -> None:
        super().__init__(message)
        self.outcome = outcome


class _IndeterminateWriteError(ReviewPanelError):
    """A final replace failed after the target stopped matching its snapshot."""

    def __init__(self, message: str, *, intended_bytes_are_live: bool) -> None:
        super().__init__(message)
        self.intended_bytes_are_live = intended_bytes_are_live


def _fingerprint(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _bound_replace(path: Path, text: str, snapshot: bytes, *, label: str) -> None:
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
        raise _IndeterminateWriteError(
            f"The {label} write failed after its exact snapshot stopped matching; "
            "restart the panel and inspect the recorded decisions",
            intended_bytes_are_live=live == intended,
        ) from exc


def _absolute(path: Path) -> Path:
    """An absolute lexical path without following its final symlink."""
    return Path(os.path.abspath(os.fspath(path)))


def _require_regular_non_symlink(path: Path, description: str) -> None:
    try:
        details = path.lstat()
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
) -> tuple[str, str, dict[str, Any], patterns.PatternSet]:
    """Validate the rich staging lineage and structurally parse its pattern answer."""
    try:
        staging.validate_coverage_facts(meta)
        run_id = staging.rich_extraction_review_run_id(meta)
    except staging.StagingError as exc:
        raise ReviewPanelError(f"The active staging lineage is invalid: {exc}") from exc
    provenance = meta.get("prompt_provenance")
    source_value = meta.get("source_file")
    if (
        run_id is None
        or not isinstance(provenance, Mapping)
        or not isinstance(source_value, str)
        or not source_value.strip()
    ):
        raise ReviewPanelError("The active staging file has no complete rich-extraction lineage")
    if source_value != Path(source_value).name or "/" in source_value or "\\" in source_value:
        raise ReviewPanelError(
            "The staging source_file must be the exact basename-shaped pattern store key"
        )
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


def _authority_state(record: VocabularyRecord) -> str:
    """``available``, ``stale``, ``existing``, ``ineligible``, or ``invalid``."""
    value = record.source.raw_fields.get(EXAMPLE_AUTHORITY_KEY, "")
    examples = [example for example in record.examples if example.japanese]
    if record.source.type != "extract" or not examples:
        return "ineligible"
    if value:
        if value == EXAMPLE_AUTHORITY_STAGING:
            return "existing"
        if _BOUND_AUTHORITY.fullmatch(value):
            if all(example_accepted(record, example) for example in examples):
                return "existing"
            return "stale"
        return "invalid"
    return "available"


def _escaped(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ", ".join(_escaped(item) for item in value) or "—"
    if isinstance(value, Mapping):
        return ", ".join(f"{_escaped(key)}: {_escaped(item)}" for key, item in value.items()) or "—"
    text = str(value)
    return html.escape(text, quote=True) if text else "—"


def _field(label: str, value: Any, *, japanese: bool = False) -> str:
    language = ' lang="ja"' if japanese else ""
    return (
        f"<div class=field><dt>{html.escape(label)}</dt><dd{language}>{_escaped(value)}</dd></div>"
    )


def _optional_field(label: str, value: Any, *, japanese: bool = False) -> str:
    return "" if value in (None, "", [], (), {}) else _field(label, value, japanese=japanese)


def _record_html(record: VocabularyRecord, state: str) -> str:
    parts = [
        '<article class="card-review">',
        f'<h3 lang="ja">{_escaped(record.expression)} '
        f"<small>{_escaped(record.reading)}</small></h3>",
        "<dl class=fields>",
        _field("Meanings", record.meanings),
        _optional_field("Usage notes", record.usage_notes),
        "</dl>",
        "<section><h4>Examples</h4>",
    ]
    if not record.examples:
        parts.append("<p>None.</p>")
    for number, example in enumerate(record.examples, start=1):
        parts.extend(
            [
                '<article class="example">',
                f"<h5>Example {number}</h5><dl class=fields>",
                _field("Japanese", example.japanese, japanese=True),
                _field("English", example.english),
                _optional_field("Furigana", example.furigana, japanese=True),
                _optional_field("Romaji", example.romaji),
                _optional_field("Register", example.register),
                "</dl></article>",
            ]
        )
    parts.extend(
        [
            "</section><details><summary>More card fields</summary><dl class=fields>",
            _field("ID", record.id),
            _optional_field("Furigana", record.furigana, japanese=True),
            _optional_field("Romaji", record.romaji),
            _optional_field("Part of speech", record.part_of_speech),
            _optional_field("Verb group", record.verb_group),
            _optional_field("Transitivity", record.transitivity),
            _optional_field("Tags", record.tags),
            "</dl></details>",
            "<details><summary>Source evidence</summary><dl class=fields>",
            _optional_field("Page", record.source.raw_fields.get("page")),
            _optional_field("Context", record.source.raw_fields.get("context"), japanese=True),
            _optional_field(
                "Inclusion reason",
                record.source.raw_fields.get("inclusion_reason"),
            ),
        ]
    )
    if record.source.raw_fields.get("hold_reason"):
        parts.append(_field("Hold", record.source.raw_fields["hold_reason"]))
    parts.append("</dl></details>")
    if state in {"available", "stale"}:
        escaped_id = html.escape(record.id, quote=True)
        identity = f"{record.expression} ({record.reading})"
        if state == "stale":
            parts.append(
                '<p class="status problem">Stored example fingerprints do not '
                "cover every current Japanese example. Reviewing this card will "
                "replace them with fingerprints for exactly the examples shown.</p>"
            )
        parts.append(
            '<label class="decision"><input type=checkbox name=record '
            f'value="{escaped_id}"> '
            f"I reviewed every Japanese example above for {_escaped(identity)} and "
            "accept it as teaching content.</label>"
        )
    elif state == "existing":
        parts.append('<p class="status reviewed">Already reviewed</p>')
    elif state == "invalid":
        parts.append(
            '<p class="status problem">The existing example authority is not a '
            "recognized review value. Correct it in the staging file.</p>"
        )
    else:
        parts.append('<p class="status">No reviewable Japanese examples.</p>')
    parts.append("</article>")
    return "".join(parts)


def _pattern_set_html(
    entry: patterns.PatternSet,
    heading: str,
    *,
    open_by_default: bool = False,
) -> str:
    opened = " open" if open_by_default else ""
    parts = [
        f"<details{opened}><summary>{html.escape(heading)}</summary><dl class=fields>",
        _field("Source", entry.source),
        _field("Kind", entry.kind),
        _field("Title", entry.title),
        _field("Review run ID", entry.review_run_id),
        "</dl>",
    ]
    if not entry.patterns:
        parts.append("<p>No patterns proposed.</p>")
    for number, pattern in enumerate(entry.patterns, start=1):
        parts.extend(
            [
                '<article class="pattern">',
                f"<h3>Pattern {number}</h3><dl class=fields>",
                _field("Template", pattern.template, japanese=True),
                _field("Gloss", pattern.gloss),
                _field("Examples", pattern.examples, japanese=True),
                _field("Where", pattern.where),
                "</dl></article>",
            ]
        )
    parts.append("</details>")
    return "".join(parts)


def _pattern_html(
    staged: patterns.PatternSet,
    current: patterns.PatternSet | None,
    warning: str | None,
) -> str:
    parts = [
        '<fieldset class="patterns"><legend>Pattern review</legend>',
        _pattern_set_html(staged, "Staged pattern answer"),
    ]
    if current is None:
        parts.append("<p>No current pattern-store entry.</p>")
    else:
        parts.append(
            _pattern_set_html(
                current,
                "Current pattern-store entry",
                open_by_default=True,
            )
        )
    if warning is not None:
        parts.append(f'<p class="status problem">{html.escape(warning)}</p>')
    elif current is not None and current.reviewed:
        parts.append('<p class="status reviewed">Already reviewed</p>')
    elif current is not None:
        parts.append(
            '<label class="decision"><input type=checkbox name=patterns '
            'value="review"> I reviewed the current pattern-store entry—kind, '
            "title, templates, glosses, examples, and locations—and approve it "
            "for use.</label>"
        )
    parts.append("</fieldset>")
    return "".join(parts)


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
    staged_pattern_set: patterns.PatternSet
    pattern_set: patterns.PatternSet | None
    pattern_warning: str | None
    staging_bytes: bytes
    patterns_bytes: bytes
    csrf_token: str
    _csrf_used: bool = False
    _session_terminal: bool = False
    _submission_inflight: bool = False
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
    ) -> ReviewPanel:
        active_path = _active_staging_path(staging_path, staging_dir)
        pattern_path = _absolute(patterns_path)
        _require_regular_non_symlink(pattern_path, "The pattern store")
        with _sorted_path_locks((active_path, pattern_path)):
            # Recheck after acquiring the lock: a cooperating writer may have
            # replaced the file while this process was waiting.
            _active_staging_path(active_path, staging_dir)
            _require_regular_non_symlink(pattern_path, "The pattern store")
            staging_bytes = _capture_regular_bytes(active_path, "The active staging file")
            patterns_bytes = _capture_regular_bytes(pattern_path, "The pattern store")
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
            csrf_token=secrets.token_urlsafe(32),
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
            if _authority_state(record) in {"available", "stale"}
        )

    @property
    def pattern_reviewable(self) -> bool:
        return (
            self.pattern_warning is None
            and self.pattern_set is not None
            and not self.pattern_set.reviewed
        )

    def render(self) -> str:
        coverage = self.meta.get("coverage")
        coverage_status = coverage.get("status") if isinstance(coverage, Mapping) else "—"
        cards = "".join(_record_html(record, _authority_state(record)) for record in self.records)
        return "".join(
            [
                "<!doctype html><html lang=en><head><meta charset=utf-8>",
                '<meta name=viewport content="width=device-width,initial-scale=1">',
                "<title>janki extraction review</title>",
                '<link rel=stylesheet href="/style.css"></head><body><main>',
                "<h1>Extraction review</h1>",
                '<p class="notice">This page records review attestations only. '
                "If anything needs correction, stop here, request or make the "
                "correction outside this page, then restart the panel.</p>",
                '<details class="summary"><summary>Extraction details</summary><dl class=fields>',
                _field("Staging file", self.staging_path.name),
                _field("Source file", self.source),
                _field("Extracted at", self.meta.get("extracted_at")),
                _field("Model", self.meta.get("model")),
                _field("Mode", self.provenance.get("mode")),
                _field("Review run ID", self.run_id),
                _field("Coverage", coverage_status),
                _field("Record count", len(self.records)),
                "</dl></details><form method=post action=/review autocomplete=off>",
                f'<input type=hidden name=csrf value="{html.escape(self.csrf_token, quote=True)}">',
                f'<input type=hidden name=staging_snapshot value="{self.staging_fingerprint}">',
                f'<input type=hidden name=patterns_snapshot value="{self.patterns_fingerprint}">',
                '<input type=hidden name=action value="save">',
                "<fieldset class=records><legend>Cards and examples</legend>",
                cards or "<p>No candidate rows in this extraction.</p>",
                "</fieldset>",
                _pattern_html(
                    self.staged_pattern_set,
                    self.pattern_set,
                    self.pattern_warning,
                ),
                "<div class=submit><button type=submit>Save review decisions</button>",
                "<p>Unchecked items remain unchanged. This does not promote cards.</p>",
                "</div></form></main></body></html>",
            ]
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
                _require_regular_non_symlink(self.patterns_path, "The pattern store")
            except ReviewPanelError as exc:
                raise StaleReviewError(
                    "A review target path changed after this page was rendered"
                ) from exc
            live_staging = self.staging_path.read_bytes()
            live_patterns = self.patterns_path.read_bytes()
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
                    _bound_replace(
                        self.staging_path,
                        staging_text,
                        self.staging_bytes,
                        label="staging file",
                    )
                except _IndeterminateWriteError as exc:
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
                    _bound_replace(
                        self.patterns_path,
                        patterns_text,
                        self.patterns_bytes,
                        label="pattern store",
                    )
                except _IndeterminateWriteError as exc:
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

    def consume_form(self, body: bytes) -> ReviewOutcome:
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
        if (
            grouped["staging_snapshot"][0] != self.staging_fingerprint
            or grouped["patterns_snapshot"][0] != self.patterns_fingerprint
        ):
            raise _PanelSecurityError("The form snapshot does not belong to this page")
        pattern_values = grouped.get("patterns", [])
        if pattern_values and pattern_values != ["review"]:
            raise PanelRequestError("The pattern action is invalid")
        token = grouped["csrf"][0]
        with self._submission_lock:
            if (
                self._csrf_used
                or self._session_terminal
                or self._submission_inflight
                or not secrets.compare_digest(token, self.csrf_token)
            ):
                raise _PanelSecurityError(
                    "The review form token is invalid, already used, or already submitting"
                )
            self._submission_inflight = True
            try:
                outcome = self.submit(
                    record_ids=grouped.get("record", []),
                    review_patterns=bool(pattern_values),
                )
            except PartialReviewError:
                self._session_terminal = True
                raise
            else:
                self._csrf_used = True
                self._session_terminal = True
                return outcome
            finally:
                self._submission_inflight = False

    def success_html(self, outcome: ReviewOutcome) -> str:
        changes = [
            f"Accepted examples for {_escaped(record_id)}."
            for record_id in outcome.accepted_record_ids
        ]
        if outcome.pattern_reviewed:
            changes.append(f"Marked {_escaped(self.source)} patterns reviewed.")
        if not changes:
            changes.append("No review decisions were selected; nothing changed.")
        return "".join(
            [
                "<!doctype html><html lang=en><head><meta charset=utf-8>",
                '<meta name=viewport content="width=device-width,initial-scale=1">',
                "<title>Review saved</title>",
                '<link rel=stylesheet href="/style.css"></head><body><main>',
                "<h1>Review saved</h1><ul>",
                *(f"<li>{item}</li>" for item in changes),
                "</ul><p>This did not promote any cards.</p></main></body></html>",
            ]
        )

    def partial_html(self, error: PartialReviewError) -> str:
        saved = []
        if error.outcome.accepted_record_ids:
            saved.append(
                "Card example approval saved for: "
                + ", ".join(_escaped(item) for item in error.outcome.accepted_record_ids)
                + "."
            )
        if error.outcome.pattern_reviewed:
            saved.append(f"Pattern review saved for {_escaped(self.source)}.")
        if not saved:
            saved.append("The final write result is indeterminate; inspect both files.")
        return "".join(
            [
                "<!doctype html><html lang=en><head><meta charset=utf-8>",
                '<meta name=viewport content="width=device-width,initial-scale=1">',
                "<title>Partial review result</title></head><body><main>",
                "<h1>Partial review result</h1><ul>",
                *(f"<li>{item}</li>" for item in saved),
                f'</ul><p role="alert">{html.escape(str(error))}</p>',
                "<p>The panel has stopped. Restart it to confirm current decisions.</p>",
                "</main></body></html>",
            ]
        )


_STYLE = """
:root { color-scheme: light dark; font: 16px/1.55 system-ui, sans-serif; }
body { margin: 0; background: Canvas; color: CanvasText; }
main { width: min(72rem, calc(100% - 2rem)); margin: 1.5rem auto 6rem; }
h1, h2, h3, h4, h5 { line-height: 1.2; }
.notice, .status, .decision { padding: .8rem; border: 1px solid GrayText; border-radius: .5rem; }
.card-review, .patterns, .summary {
  margin: 1rem 0; padding: 1rem; border: 1px solid GrayText; border-radius: .75rem;
}
fieldset { min-inline-size: 0; }
legend { padding: 0 .35rem; font-size: 1.35rem; font-weight: 700; }
.example, .pattern {
  margin: .75rem 0; padding: .75rem;
  background: color-mix(in srgb, CanvasText 6%, Canvas); border-radius: .5rem;
}
.fields { display: grid; gap: .35rem 1rem; margin: 0; }
.field { min-width: 0; }
dt { font-weight: 700; }
dd { margin: 0; overflow-wrap: anywhere; white-space: pre-wrap; }
.decision { display: flex; gap: .75rem; align-items: flex-start; cursor: pointer; }
input[type=checkbox] { inline-size: 1.4rem; block-size: 1.4rem; flex: none; }
button { min-height: 44px; padding: .65rem 1rem; font: inherit; font-weight: 700; }
button:focus-visible, input:focus-visible { outline: 3px solid Highlight; outline-offset: 3px; }
.submit {
  position: sticky; bottom: 0; padding: 1rem;
  background: Canvas; border-top: 1px solid GrayText;
}
.reviewed { border-color: green; }
.problem { border-color: red; }
@media (min-width: 48rem) { .fields { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto !important; } }
""".strip()


class _ReviewServer(ThreadingHTTPServer):
    # ThreadingHTTPServer opts into daemon handlers. A CLI cancellation followed
    # by server_close must instead wait until an in-flight approval's bound
    # replace has finished, so the command cannot return while it may still write.
    daemon_threads = False

    panel: ReviewPanel
    expected_host: str


class _ReviewHandler(BaseHTTPRequestHandler):
    server: _ReviewServer
    server_version = "janki"
    sys_version = ""

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(REQUEST_TIMEOUT_SECONDS)

    def log_message(self, _format: str, *args: object) -> None:
        return

    def send_error(
        self,
        code: int,
        message: str | None = None,
        explain: str | None = None,
    ) -> None:
        """Keep unsupported-method and parser errors under the same headers."""
        del explain
        self._error(code, message or "The review request was refused.")

    def _request_is_local(self) -> bool:
        hosts = self.headers.get_all("Host") or []
        if hosts != [self.server.expected_host]:
            return False
        origins = self.headers.get_all("Origin") or []
        return not origins or origins == [f"http://{self.server.expected_host}"]

    def _send(
        self,
        status: int,
        body: str | bytes,
        *,
        content_type: str = "text/html; charset=utf-8",
    ) -> None:
        payload = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; style-src 'self'; form-action 'self'; "
            "base-uri 'none'; frame-ancestors 'none'",
        )
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)
        self.close_connection = True

    def _error(self, status: int, message: str) -> None:
        self._send(status, self._error_html(message))

    @staticmethod
    def _error_html(message: str) -> str:
        return (
            "<!doctype html><html lang=en><head><meta charset=utf-8>"
            "<title>Review error</title></head><body><main><h1>Review error</h1>"
            f'<p role="alert">{html.escape(message)}</p></main></body></html>'
        )

    def _terminal_send(self, status: int, body: str) -> None:
        """Stop a used/stale session even when its client has disconnected."""
        try:
            self._send(status, body)
        except OSError:
            self.close_connection = True
        finally:
            self.server.shutdown()

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if not self._request_is_local():
            self._error(403, "This review page accepts only its exact localhost origin.")
            return
        if self.path == "/":
            self._send(200, self.server.panel.render())
            return
        if self.path == "/style.css":
            self._send(200, _STYLE, content_type="text/css; charset=utf-8")
            return
        self._error(404, "No such review page.")

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if not self._request_is_local():
            self._error(403, "This review page accepts only its exact localhost origin.")
            return
        if self.path != "/review":
            self._error(404, "No such review action.")
            return
        if self.headers.get_all("Transfer-Encoding"):
            self._error(400, "Transfer-encoded review forms are not accepted.")
            return
        lengths = self.headers.get_all("Content-Length") or []
        if len(lengths) != 1 or not lengths[0].isdigit():
            self._error(411, "One exact Content-Length is required.")
            return
        length = int(lengths[0])
        if length > MAX_BODY_BYTES:
            self._error(413, "The review form exceeds the 64 KiB limit.")
            return
        content_types = self.headers.get_all("Content-Type") or []
        if content_types != ["application/x-www-form-urlencoded"]:
            self._error(415, "The review form has the wrong content type.")
            return
        try:
            body = self.rfile.read(length)
        except TimeoutError:
            try:
                self._error(408, "The review form body timed out.")
            except OSError:
                self.close_connection = True
            return
        if len(body) != length:
            self._error(400, "The review form ended before Content-Length.")
            return
        try:
            outcome = self.server.panel.consume_form(body)
        except _PanelSecurityError as exc:
            self._error(403, str(exc))
        except PanelRequestError as exc:
            self._error(400, str(exc))
        except PartialReviewError as exc:
            self._terminal_send(409, self.server.panel.partial_html(exc))
        except StaleReviewError as exc:
            self._terminal_send(
                409,
                self._error_html(
                    f"{exc}. Nothing was written by this attempt. Restart the panel "
                    "to review the current files."
                ),
            )
        except ReviewPanelError as exc:
            self._error(
                500,
                f"{exc}. Nothing was proven written; resolve the error and retry, "
                "or restart the panel to confirm current decisions.",
            )
        else:
            # A successful one-submit review is terminal. Shutting the server
            # down lets the future CLI call return instead of leaving a stale,
            # already-consumed form listening on localhost.
            self._terminal_send(200, self.server.panel.success_html(outcome))


def make_server(panel: ReviewPanel) -> _ReviewServer:
    """Create, but do not start, an ephemeral IPv4-loopback review server."""
    server = _ReviewServer(("127.0.0.1", 0), _ReviewHandler)
    server.panel = panel
    server.expected_host = f"127.0.0.1:{server.server_address[1]}"
    return server
