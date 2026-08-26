"""The localhost workbench: a long-lived page over the repository's own state.

`WORKBENCH_PLAN.md` W1.2. This is the review panel's boundary
(`localhttp.LocalOnlyHandler`) plus the one thing a *long-lived* surface needs
that a one-shot form does not: **a session secret on every request, reads
included.**

Loopback is not a permission. Every process on this machine can connect to a
127.0.0.1 port, and this page renders private study material — the Japanese a
person is learning, the filenames of documents they scanned. The review panel
tolerates that because it lives for one submission and dies. A dashboard left
open all afternoon does not, so an unguessable token sits in the URL path and
every request is compared against it in constant time.

The token is in the *path*, deliberately, not a cookie. Cookies on 127.0.0.1
ignore the port: a cookie set for `127.0.0.1` is sent to every other local
server on every other port, so any unrelated local service would receive this
session's secret. A path segment is scoped to the request it is written on, and
`Referrer-Policy: same-origin` keeps it out of cross-origin referrers.

W2b added the one write route: approving the exact Japanese examples on a
card, and marking a lesson's grammar read. Both go through the same
transaction the deleted one-shot panel used (`workbench.review`), so the
guarantees a browser gets are the ones that were reviewed adversarially
there — an exact-byte snapshot, a compare-and-swap write, and a refusal
that leaves the file untouched rather than half-applied.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
import sys
import webbrowser
from collections.abc import Sequence
from dataclasses import dataclass, field
from email import policy as email_policy
from email.parser import BytesParser
from pathlib import Path
from urllib.parse import parse_qsl, quote, unquote

from japanese_anki import extract, inputs, ledger, operations, staging
from japanese_anki.application import (
    ANSWER_EMPTY,
    ANSWER_SAVED,
    ANSWER_UNAVAILABLE,
    FORGOTTEN,
    ExtractionCompletionError,
    ExtractionConsent,
    SourceJourney,
    authorize_dispatch,
    capture_hook,
    classify_dispatch_failure,
    complete_extraction,
    describe_extraction,
    extraction_replacement_revision,
    plan_corpus_extraction,
    source_detail,
    source_journeys,
)
from japanese_anki.application.assignment import (
    AssignableWordDeck,
    AssignmentError,
    DeckAssignmentPlan,
    ProposalOccurrence,
    assignable_word_decks,
    plan_deck_assignment,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.io import load_records
from japanese_anki.localhttp import (
    MAX_BODY_BYTES,
    LocalOnlyHandler,
    LocalOnlyServer,
    bind_loopback,
)
from japanese_anki.models import VocabularyRecord
from japanese_anki.workbench import edit, reidentify, review
from japanese_anki.workbench.dispatch import (
    DispatchFormError,
    ExtractionAction,
    ExtractionActions,
    parse_dispatch_form,
)
from japanese_anki.workbench.render import (
    STYLE,
    render_consent,
    render_dashboard,
    render_extraction_failure,
    render_extraction_progress_start,
    render_extraction_progress_step,
    render_extraction_success,
    render_reidentify,
    render_source,
)

__all__ = ["WorkbenchSession", "make_server", "serve"]

_SOURCE_PREFIX = "/source/"
_EXTRACT_PREFIX = "/extract/"

#: An uploaded source is orders of magnitude larger than a form. It is
#: still bounded: this is a localhost tool reading a scan or a lesson PDF,
#: and an unbounded read is a way to exhaust memory from another local
#: process that guessed the token.
MAX_UPLOAD_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class _DeckAssignmentChoice:
    """One configured destination as it was truthfully planned for a card."""

    stem: str
    name: str
    intake_tag: str
    plan: DeckAssignmentPlan | None
    refusal: str | None


@dataclass(frozen=True, slots=True)
class _DeckAssignmentOffer:
    """The complete picker rendered for one exact staging-row position."""

    card: int
    record_id: str
    proposals: tuple[ProposalOccurrence, ...]
    choices: tuple[_DeckAssignmentChoice, ...]
    fingerprint: str


def _assignment_claim(choice: _DeckAssignmentChoice) -> dict[str, object]:
    """The exact planned facts a deck button displays and may persist."""
    common: dict[str, object] = {
        "stem": choice.stem,
        "name": choice.name,
        "intake_tag": choice.intake_tag,
        "refusal": choice.refusal,
    }
    plan = choice.plan
    if plan is None:
        return common
    common.update(
        {
            "record_id": plan.record_id,
            "existing_owner": plan.existing_owner,
            "tag_diff": {
                "before": plan.tag_diff.before,
                "after": plan.tag_diff.after,
                "removed": plan.tag_diff.removed,
                "added": plan.tag_diff.added,
            },
            "assigned_record": plan.assigned_record.to_dict(),
            "prospective_record": plan.prospective_record.to_dict(),
            "memberships": [
                {
                    "stem": membership.stem,
                    "name": membership.name,
                    "takes": membership.takes,
                    "refusal": membership.refusal,
                }
                for membership in plan.memberships
            ],
            "proposals": [
                {
                    "imported_from": proposal.imported_from,
                    "row": proposal.row,
                    "source_type": proposal.source_type,
                }
                for proposal in plan.proposals
            ],
        }
    )
    return common


def _assignment_offer(
    config: ProjectConfig,
    record: VocabularyRecord,
    card: int,
    decks: Sequence[AssignableWordDeck],
    *,
    sibling_proposals: Sequence[VocabularyRecord] = (),
) -> _DeckAssignmentOffer:
    """Re-plan every rendered destination and bind their exact claims."""
    proposals = tuple(
        ProposalOccurrence(
            imported_from=proposal.source.imported_from,
            row=proposal.source.row,
            source_type=proposal.source.type,
        )
        for proposal in (record, *sibling_proposals)
    )
    choices: list[_DeckAssignmentChoice] = []
    for deck in decks:
        try:
            plan = plan_deck_assignment(
                config,
                record,
                deck.stem,
                sibling_proposals=sibling_proposals,
            )
        except AssignmentError as exc:
            choices.append(
                _DeckAssignmentChoice(
                    stem=deck.stem,
                    name=deck.name,
                    intake_tag=deck.intake_tag,
                    plan=None,
                    refusal=str(exc),
                )
            )
        else:
            choices.append(
                _DeckAssignmentChoice(
                    stem=deck.stem,
                    name=deck.name,
                    intake_tag=deck.intake_tag,
                    plan=plan,
                    refusal=None,
                )
            )
    claims = {
        "card": card,
        "record_id": record.id,
        "proposals": [
            {
                "imported_from": proposal.imported_from,
                "row": proposal.row,
                "source_type": proposal.source_type,
            }
            for proposal in proposals
        ],
        "choices": [_assignment_claim(choice) for choice in choices],
    }
    encoded = json.dumps(
        claims,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _DeckAssignmentOffer(
        card=card,
        record_id=record.id,
        proposals=proposals,
        choices=tuple(choices),
        fingerprint=hashlib.sha256(encoded).hexdigest(),
    )


def _assignment_offers(
    config: ProjectConfig,
    records: Sequence[VocabularyRecord],
    current_path: Path,
) -> tuple[_DeckAssignmentOffer, ...]:
    decks = assignable_word_decks(config)
    siblings = _staged_sibling_proposals(
        config,
        current_path,
        frozenset(record.id for record in records),
    )
    return tuple(
        _assignment_offer(
            config,
            record,
            card,
            decks,
            sibling_proposals=siblings.get(record.id, ()),
        )
        for card, record in enumerate(records)
    )


def _staged_sibling_proposals(
    config: ProjectConfig,
    current_path: Path,
    record_ids: frozenset[str],
) -> dict[str, tuple[VocabularyRecord, ...]]:
    """Matching rows in every other readable live review.

    A broken sibling cannot be silently omitted: its unreadable rows might
    contain the same stable id, which would make the page's duplicate claim
    and its bound assignment plan incomplete.
    """
    found: dict[str, list[VocabularyRecord]] = {
        record_id: [] for record_id in record_ids
    }
    active = Path(os.path.abspath(config.staging_dir))
    current = Path(os.path.abspath(current_path))
    try:
        candidates = sorted(active.iterdir())
    except OSError as exc:
        raise AssignmentError(
            f"Could not inspect live reviews for duplicate proposals: {exc}"
        ) from exc
    for candidate in candidates:
        candidate = Path(os.path.abspath(candidate))
        if candidate == current or candidate.suffix.lower() not in staging.STAGING_SUFFIXES:
            continue
        try:
            details = os.lstat(candidate)
        except OSError as exc:
            raise AssignmentError(
                f"Could not inspect live review {candidate}: {exc}"
            ) from exc
        if not stat.S_ISREG(details.st_mode):
            raise AssignmentError(
                f"Live review {candidate} is not a direct regular file."
            )
        try:
            records, _meta = staging.read_staging(candidate)
        except JankiError as exc:
            raise AssignmentError(
                f"Could not inspect live review {candidate}: {exc}"
            ) from exc
        for record in records:
            if record.id in found:
                found[record.id].append(record)
    return {record_id: tuple(items) for record_id, items in found.items()}


@dataclass(frozen=True, slots=True)
class WorkbenchSession:
    """One run: config, session secrets, and unspent paid-action capabilities.

    Deliberately holds no cached journey. The dashboard is recomputed from the
    repository on every request, so a `promote` run in another terminal shows
    up on the next refresh — and so nothing this process remembers can claim a
    review or a paid call happened when the files say otherwise.
    """

    config: ProjectConfig
    #: Gates *reading* the page at all. In the URL path.
    token: str
    #: Gates *writing*, carried in the form body. Separate from `token`
    #: because they defend different things: the path token stops another
    #: local process reading the page, and this stops a page the browser
    #: was tricked into submitting. Unlike the one-shot panel's, this one
    #: survives the request — a dashboard approves many sources in a row.
    csrf_token: str
    #: Paid consent is narrower than session CSRF: each value is bound to one
    #: rendered call and consumed once. It is ephemeral authority, never a
    #: second record of whether the operation happened.
    extraction_actions: ExtractionActions = field(
        default_factory=ExtractionActions, repr=False, compare=False
    )

    @classmethod
    def open(cls, config: ProjectConfig) -> WorkbenchSession:
        return cls(
            config=config,
            token=secrets.token_urlsafe(32),
            csrf_token=secrets.token_urlsafe(32),
        )

    def journeys(self) -> tuple[list[SourceJourney], list[str]]:
        return source_journeys(self.config)

    def detail(self, source: str):
        return source_detail(self.config, source)

    def canonical_records(self) -> list[VocabularyRecord]:
        """What the collection already holds, for collision detection."""
        if not self.config.normalized_file.exists():
            return []
        return load_records(self.config.normalized_file)

    def exported_ids(
        self, records: Sequence[VocabularyRecord]
    ) -> frozenset[str]:
        """Which of these have already shipped in a built deck."""
        if not self.config.ledger_file.exists():
            return frozenset()
        try:
            book = ledger.load(self.config.ledger_file)
        except JankiError:
            return frozenset()
        return book.ever_exported(record.id for record in records)

    def source_path(self, name: str) -> Path | None:
        """The corpus file called `name`, or None.

        Walked and matched, never joined: `scan_inbox / name` with a name from
        a request is a path traversal waiting for the one caller that forgets
        to check it. The same discipline the source route follows.
        """
        inbox = self.config.scan_inbox
        try:
            root_details = os.lstat(inbox)
        except OSError:
            return None
        if not stat.S_ISDIR(root_details.st_mode):
            return None
        try:
            paths = sorted(inbox.iterdir())
        except OSError:
            return None
        for path in paths:
            # `startswith(".")` like the journey walk: a route that rendered a
            # source the dashboard deliberately hides is a second, quieter
            # answer to "what is in my corpus".
            try:
                details = os.lstat(path)
            except OSError:
                continue
            if (
                stat.S_ISREG(details.st_mode)
                and not path.name.startswith(".")
                and path.name == name
            ):
                return path
        return None

    def consent(self, name: str, mode: str | None = None):
        """What sending this source would mean, or None if it is not a source."""
        path = self.source_path(name)
        if path is None:
            return None
        return describe_extraction(self.config, path, mode=mode)

    def issue_extraction_action(self, consent: ExtractionConsent) -> str:
        return self.extraction_actions.issue(consent)

    def consume_extraction_action(self, token: str) -> ExtractionAction | None:
        return self.extraction_actions.consume(token)

    def panel(self, source: str) -> review.ReviewPanel | None:
        """A freshly opened review panel for one source, or None.

        Opened per request, never cached. The panel captures the exact
        bytes it read, and those bytes are what its compare-and-swap write
        binds against — a cached panel would bind an approval to a file
        state that may be minutes stale.
        """
        detail = self.detail(source)
        if detail is None or detail.journey.staging_path is None:
            return None
        try:
            return review.ReviewPanel.open(
                detail.journey.staging_path,
                staging_dir=self.config.staging_dir,
                patterns_path=self.config.patterns_file,
                collection_name=self.config.normalized_file.name,
            )
        except JankiError:
            return None


class _WorkbenchServer(LocalOnlyServer):
    session: WorkbenchSession


class _ExtractionProgress:
    """A streamed page whose client may leave without canceling paid work."""

    def __init__(self, handler: _WorkbenchHandler, name: str, token: str) -> None:
        self.handler = handler
        self.name = name
        self.token = token
        self.writable = True

    def _write(self, chunk: str) -> None:
        if not self.writable:
            return
        try:
            self.handler.wfile.write(chunk.encode("utf-8"))
            self.handler.wfile.flush()
        except OSError:
            # Authority was already journaled. Closing a tab cannot be allowed
            # to strand a provider answer the process can still finish saving.
            self.writable = False

    def start(self) -> None:
        try:
            self.handler._start_response(200)
        except OSError:
            self.writable = False
            return
        self._write(render_extraction_progress_start(self.name, token=self.token))

    def step(self, label: str) -> None:
        self._write(render_extraction_progress_step(label))

    def success(self, outcome: object) -> None:
        self._write(render_extraction_success(self.name, outcome, token=self.token))

    def failure(self, message: str, note: str) -> None:
        self._write(render_extraction_failure(message, note))


def _completion_refusal_note(config: ProjectConfig, operation_id: str) -> str:
    """Describe the authoritative state after a final journal refusal."""
    try:
        held = operations.OperationJournal.load(config.operations_file).operations.get(
            operation_id
        )
    except JankiError as exc:
        return (
            "Saving was refused, and janki could not re-read the operation "
            f"journal to describe recovery safely: {exc}"
        )
    if held is None:
        return (
            "Saving was refused after the operation was already forgotten; "
            "no recovery answer remains in janki operations. Reload the source "
            "before deciding whether to make a new paid call."
        )
    if held.cleanup is not None:
        return (
            "Saving was refused after the operation's forget decision won. "
            "Finish its exact recovery-data cleanup with "
            f"'janki operations --forget {operation_id}'."
        )
    return (
        "Saving was refused. The answer remains in janki operations; reload "
        "the source and inspect what is on disk before trying again."
    )


def _completion_error_note(
    config: ProjectConfig,
    operation_id: str,
) -> str:
    """Describe recovery after an untyped completion failure."""
    try:
        held = operations.OperationJournal.load(config.operations_file).operations.get(
            operation_id
        )
    except JankiError as exc:
        return (
            "The answer arrived, but janki could not re-read the operation "
            f"journal to prove where it landed: {exc}"
        )
    if held is None:
        return (
            "The answer arrived, but the operation was already forgotten and "
            "no recovery answer remains in janki operations."
        )
    if held.cleanup is not None:
        return (
            "The answer arrived, but the operation's forget decision won. "
            "Finish its exact recovery-data cleanup with "
            f"'janki operations --forget {operation_id}'."
        )
    return (
        "The answer arrived, but janki could not prove that every proposal "
        "was saved. Reload the source and inspect janki operations."
    )


class _WorkbenchHandler(LocalOnlyHandler):
    server: _WorkbenchServer
    error_title = "Workbench error"

    def _route(self) -> str | None:
        """The path beneath this session's token, or None if it is not ours.

        A wrong or missing token is a 404 with the same wording as any unknown
        page. Distinguishing "no workbench here" from "wrong secret" would let
        a local process confirm a session exists and then sit on the port.
        """
        parts = self.path.split("?", 1)[0].split("/")
        # "/<token>/rest" splits to ["", token, "rest"].
        if len(parts) < 2:
            return None
        offered = unquote(parts[1])
        if not secrets.compare_digest(offered, self.server.session.token):
            return None
        return "/" + "/".join(parts[2:])

    def _added_banner(self) -> tuple[str, bool] | None:
        """The post-redirect-get result of an upload, if this is one."""
        query = self.path.split("?", 1)
        if len(query) != 2:
            return None
        fields = dict(parse_qsl(query[1]))
        if "added" not in fields or "name" not in fields:
            return None
        return fields["name"], fields["added"] == "1"

    def _wants_mode(self) -> str | None:
        """The page kind a person chose on the consent form, if any.

        Only janki's two real modes are honoured; anything else is the
        default, because a mode this code does not recognise must not reach
        `prompt_name` and pick a prompt by accident.
        """
        query = self.path.split("?", 1)
        if len(query) != 2:
            return None
        asked = dict(parse_qsl(query[1])).get("mode", "")
        return asked if asked in extract.MODES else None

    def _wants_edit(self) -> bool:
        query = self.path.split("?", 1)
        return len(query) == 2 and dict(parse_qsl(query[1])).get("edit") == "1"

    def _saved_banner(self) -> tuple[int, bool, int, int, int, int] | None:
        """The post-redirect-get result, if this is one. Display only."""
        query = self.path.split("?", 1)
        if len(query) != 2:
            return None
        fields = dict(parse_qsl(query[1]))
        if "saved" not in fields:
            return None
        try:
            records = int(fields.get("saved", "0"))
        except ValueError:
            return None
        # Clamped: this is a self-reported number in a URL a person can edit,
        # so it may say what it likes — it never means anything was written.
        try:
            edited = int(fields.get("edited", "0"))
        except ValueError:
            edited = 0
        try:
            removed = int(fields.get("removed", "0"))
        except ValueError:
            removed = 0
        try:
            reidentified = int(fields.get("reidentified", "0"))
        except ValueError:
            reidentified = 0
        try:
            assigned = int(fields.get("assigned", "0"))
        except ValueError:
            assigned = 0
        return (
            max(records, 0),
            fields.get("grammar") == "1",
            max(edited, 0),
            max(removed, 0),
            max(reidentified, 0),
            max(assigned, 0),
        )

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if not self._request_is_local():
            self._error(403, "The workbench accepts only its exact localhost origin.")
            return
        route = self._route()
        if route is None:
            self._error(404, "No such workbench page.")
            return
        if route == "/":
            journeys, warnings = self.server.session.journeys()
            self._send(
                200,
                render_dashboard(
                    journeys,
                    warnings=warnings,
                    root=self.server.session.config.root,
                    token=self.server.session.token,
                    csrf=self.server.session.csrf_token,
                    added=self._added_banner(),
                ),
            )
            return
        if route == "/style.css":
            self._send(200, STYLE, content_type="text/css; charset=utf-8")
            return
        if route.startswith(_EXTRACT_PREFIX):
            # Same discipline as the source route: the name is matched against
            # janki's own view of the corpus, and the path comes from that
            # match rather than from the request.
            name = unquote(route[len(_EXTRACT_PREFIX) :])
            asked = self._wants_mode()
            consent = self.server.session.consent(name, asked)
            if consent is None:
                self._error(404, "No such source.")
                return
            dispatch = self.server.session.issue_extraction_action(consent)
            self._send(
                200,
                render_consent(
                    consent,
                    token=self.server.session.token,
                    csrf=self.server.session.csrf_token,
                    dispatch=dispatch,
                ),
            )
            return
        if route.startswith(_SOURCE_PREFIX):
            # The name is matched against the dashboard's own computed list and
            # the staging path comes from that journey — the request never
            # names a path, so it cannot name one outside the corpus. Do not
            # "improve" this into a join against staging_dir.
            encoded_name = route[len(_SOURCE_PREFIX) :]
            if not encoded_name or "/" in encoded_name:
                self._error(404, "No such source.")
                return
            name = unquote(encoded_name)
            detail = self.server.session.detail(name)
            if detail is None:
                self._error(404, "No such source.")
                return
            # The panel is opened here only for the exact bytes it read: those
            # fingerprints go into the form, and the approval is refused later
            # unless the file is still byte-identical to them.
            panel = self.server.session.panel(name)
            assignments: tuple[_DeckAssignmentOffer, ...] = ()
            assignment_error = ""
            if panel is not None:
                # Do not attach row-indexed controls to a detail projection and
                # a write snapshot that saw different ordered records.
                shown = [card.record for card in detail.cards]
                if shown != panel.records:
                    assignment_error = (
                        "This source changed while the page was opening. Reload "
                        "before choosing a study deck."
                    )
                else:
                    try:
                        assignments = _assignment_offers(
                            self.server.session.config,
                            panel.records,
                            panel.staging_path,
                        )
                    except JankiError as exc:
                        assignment_error = str(exc)
            self._send(
                200,
                render_source(
                    detail,
                    token=self.server.session.token,
                    csrf=self.server.session.csrf_token if panel else "",
                    staging_snapshot=panel.staging_fingerprint if panel else "",
                    patterns_snapshot=panel.patterns_fingerprint if panel else "",
                    saved=self._saved_banner(),
                    editing=self._wants_edit(),
                    approvable=bool(panel and panel.has_extraction_lineage),
                    reidentifiable=bool(panel and panel.reidentifiable),
                    assignment_offers=assignments,
                    assignment_error=assignment_error,
                ),
            )
            return
        self._error(404, "No such workbench page.")

    def _upload(self, body: bytes, content_type: str) -> None:
        """Store one uploaded source in the durable inbox. Sends nothing.

        Adding a file to the corpus and sending it to a model are two separate
        actions, always — this route is the first one, and it has no provider
        call anywhere behind it.
        """
        session = self.server.session
        try:
            name, data, fields = _parse_upload(body, content_type)
        except ValueError as exc:
            self._error(400, f"That upload could not be read: {exc}")
            return
        if not secrets.compare_digest(fields.get("csrf", ""), session.csrf_token):
            self._error(403, "That form did not come from this workbench session.")
            return
        try:
            intake = inputs.receive_upload(
                name, data, inbox_root=session.config.scan_inbox
            )
        except JankiError as exc:
            self._error(409, str(exc))
            return
        added = "1" if intake.stored else "0"
        self.send_response(303)
        self.send_header(
            "Location",
            f"/{session.token}/?added={added}&name={quote(intake.path.name, safe='')}",
        )
        self.send_header("Content-Length", "0")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.close_connection = True

    def _read_upload_body(self) -> tuple[bytes, str] | None:
        """The multipart body, or None having already sent the refusal."""
        if self.headers.get_all("Transfer-Encoding"):
            self._error(400, "Transfer-encoded uploads are not accepted.")
            return None
        lengths = self.headers.get_all("Content-Length") or []
        if len(lengths) != 1 or not lengths[0].isdigit():
            self._error(411, "One exact Content-Length is required.")
            return None
        length = int(lengths[0])
        if length > MAX_UPLOAD_BYTES:
            self._error(
                413,
                f"That file is larger than the "
                f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
            )
            return None
        types = self.headers.get_all("Content-Type") or []
        if len(types) != 1 or not types[0].startswith("multipart/form-data"):
            self._error(415, "An upload must be multipart/form-data.")
            return None
        try:
            body = self.rfile.read(length)
        except TimeoutError:
            self._error(408, "The upload timed out.")
            return None
        if len(body) != length:
            self._error(400, "The upload ended before its Content-Length.")
            return None
        return body, types[0]

    def _read_body(self) -> bytes | None:
        """The exact declared body, or None having already sent the refusal.

        Every check the one-shot panel made, for the same reasons: a chunked
        body has no length to bound, two `Content-Length` headers let a proxy
        and this server disagree about where the body ends, and a short read
        means the client vanished mid-submission and its intent is unknown.
        """
        if self.headers.get_all("Transfer-Encoding"):
            self._error(400, "Transfer-encoded forms are not accepted.")
            return None
        lengths = self.headers.get_all("Content-Length") or []
        if len(lengths) != 1 or not lengths[0].isdigit():
            self._error(411, "One exact Content-Length is required.")
            return None
        length = int(lengths[0])
        if length > MAX_BODY_BYTES:
            self._error(413, "That form is larger than the 64 KiB limit.")
            return None
        if (self.headers.get_all("Content-Type") or []) != [
            "application/x-www-form-urlencoded"
        ]:
            self._error(415, "That form has the wrong content type.")
            return None
        try:
            body = self.rfile.read(length)
        except TimeoutError:
            self._error(408, "The form body timed out.")
            return None
        if len(body) != length:
            self._error(400, "The form ended before its Content-Length.")
            return None
        return body

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if not self._request_is_local():
            self._error(403, "The workbench accepts only its exact localhost origin.")
            return
        route = self._route()
        if route is None:
            self._error(404, "No such workbench action.")
            return
        if (
            route != "/add-source"
            and not route.startswith(_SOURCE_PREFIX)
            and not route.startswith(_EXTRACT_PREFIX)
        ):
            self._error(404, "No such workbench action.")
            return
        if route == "/add-source":
            read = self._read_upload_body()
            if read is None:
                return
            self._upload(*read)
            return
        if route.startswith(_EXTRACT_PREFIX):
            name = unquote(route[len(_EXTRACT_PREFIX) :])
            if not name:
                self._error(404, "No such workbench action.")
                return
            body = self._read_body()
            if body is None:
                return
            self._extract(name, body)
            return
        rest = route[len(_SOURCE_PREFIX) :]
        name, _, action = rest.rpartition("/")
        actions = {"approve", "edit", "remove", "reidentify", "assign"}
        if action not in actions or not name:
            self._error(404, "No such workbench action.")
            return
        body = self._read_body()
        if body is None:
            return
        if action == "approve":
            self._approve(unquote(name), body)
        elif action == "edit":
            self._edit(unquote(name), body)
        elif action == "remove":
            self._remove(unquote(name), body)
        elif action == "assign":
            self._assign(unquote(name), body)
        else:
            self._reidentify(unquote(name), body)

    def _extract(self, name: str, body: bytes) -> None:
        """Spend one bound browser consent through the shared extraction path."""
        session = self.server.session
        try:
            submission = parse_dispatch_form(body)
        except DispatchFormError as exc:
            self._error(400, str(exc))
            return
        if not secrets.compare_digest(
            submission.csrf.encode("utf-8"), session.csrf_token.encode("utf-8")
        ):
            self._error(403, "That form did not come from this workbench session.")
            return

        action = session.consume_extraction_action(submission.token)
        if action is None:
            self._error(
                409,
                "This paid action has already been used or is no longer available. "
                "Nothing was sent. Reload the consent page before trying again.",
            )
            return
        if (
            action.name != name
            or action.model != submission.model
            or action.mode != submission.mode
            or action.request_fingerprint != submission.request_fingerprint
            or action.replacement_offered != submission.replacement_offered
        ):
            self._error(
                409,
                "This paid action does not match the call the page described. "
                "Nothing was sent. Reload the consent page before trying again.",
            )
            return
        if action.replacement_offered and not submission.replacement_confirmed:
            self._error(
                409,
                "Confirm that this re-read replaces the review named on the "
                "page. Nothing was sent.",
            )
            return

        source = session.source_path(name)
        if source is None:
            self._error(404, "No such source. Nothing was sent.")
            return
        force = submission.replacement_confirmed
        try:
            # Fresh at click time. GET planned with force solely to describe a
            # collision; no object from that forced snapshot reaches here.
            plan = plan_corpus_extraction(
                session.config,
                source,
                mode=action.mode,
                model=action.model,
                force=force,
            )
        except JankiError as exc:
            self._error(409, f"{exc} Nothing was sent.")
            return
        if len(plan.targets) != 1:
            self._error(409, "That source no longer makes one extraction request.")
            return
        target = plan.targets[0]
        fresh_fingerprint = str(target.provenance["request_fingerprint"])
        if (
            fresh_fingerprint != action.request_fingerprint
            or target.source_sha256 != action.source_sha256
        ):
            self._error(
                409,
                "The extraction request changed after this page was rendered. "
                "Nothing was sent. Reload and review the current call.",
            )
            return

        try:
            fresh_revision = extraction_replacement_revision(session.config, target)
        except JankiError as exc:
            self._error(409, f"{exc} Nothing was sent.")
            return
        expected_revision = action.replacement_revision
        if fresh_revision is None or expected_revision is None:
            if fresh_revision != expected_revision:
                self._error(
                    409,
                    "The review changed after this page was rendered. Nothing was "
                    "sent. Reload and review the current replacement.",
                )
                return
        elif fresh_revision.staging_sha256 != expected_revision.staging_sha256:
            self._error(
                409,
                "The review changed after this page was rendered. Nothing was sent. "
                "Reload and review the current replacement.",
            )
            return
        elif (
            fresh_revision.pattern_entry_sha256
            != expected_revision.pattern_entry_sha256
        ):
            self._error(
                409,
                "The grammar review changed after this page was rendered. Nothing "
                "was sent. Reload and review the current replacement.",
            )
            return

        try:
            journal = operations.OperationJournal.load(session.config.operations_file)
            # The gate, under its own lock. `busy_refusal` was only what the
            # GET happened to display; two old pages can both have seen clear.
            operation_id = authorize_dispatch(journal, target, model=plan.model)
        except JankiError as exc:
            self._error(409, f"{exc} Nothing was sent by this attempt.")
            return

        progress = _ExtractionProgress(self, name, session.token)
        progress.start()
        progress.step("Preparing pages")
        progress.step("Reading the source")
        captured = capture_hook(session.config, journal, operation_id)

        def capture(response: object) -> None:
            captured(response)
            progress.step("Checking the answer's shape")

        try:
            result = extract.extract_candidates(
                target.item,
                model=plan.model,
                style_guide=plan.style_guide,
                system=plan.system,
                mode=plan.mode,
                known=plan.skip_list,
                capture=capture,
            )
        except Exception as exc:  # noqa: BLE001 - settle any dispatched call
            try:
                failure = classify_dispatch_failure(
                    session.config, journal, operation_id, exc
                )
                if failure.outcome == ANSWER_SAVED:
                    note = (
                        "Nothing was staged. The paid answer is recoverable with "
                        f"'janki operations --show-reply {operation_id}'; "
                        f"operation {operation_id}."
                    )
                elif failure.outcome == ANSWER_EMPTY:
                    note = (
                        "Nothing was staged. The paid reply was saved, but it "
                        f"contains no answer; inspect its exact bytes with "
                        f"'janki operations --show-reply {operation_id}'."
                    )
                elif failure.outcome == ANSWER_UNAVAILABLE:
                    note = (
                        "Nothing was staged. The reply was recorded as captured, "
                        "but its exact recovery bytes are unavailable; "
                        f"operation {operation_id}. Inspect janki operations "
                        "before accepting that loss."
                    )
                elif failure.outcome == FORGOTTEN:
                    if failure.cleanup_pending:
                        note = (
                            "Nothing was staged. The operation's forget decision "
                            "is recorded; finish exact cleanup with "
                            f"'janki operations --forget {operation_id}'."
                        )
                    else:
                        note = (
                            "Nothing was staged. The operation was already "
                            "forgotten, so no recovery answer remains in janki "
                            "operations."
                        )
                else:
                    note = (
                        "Nothing was staged. This call may already have been "
                        f"billed; operation {operation_id}. Do not retry until "
                        "you inspect janki operations."
                    )
            except JankiError as journal_error:
                note = (
                    "Nothing was proven staged, and janki could not settle the "
                    f"operation journal: {journal_error}"
                )
            progress.failure(str(exc), note)
            return

        progress.step("Saving proposals")
        try:
            outcome = complete_extraction(
                session.config,
                journal,
                target,
                result,
                operation_id=operation_id,
                known=plan.known,
                mode=plan.mode,
                model=plan.model,
                force=force,
                expected_revision=expected_revision,
            )
        except ExtractionCompletionError as exc:
            progress.failure(
                exc.detail,
                f"The proposals are saved at {exc.staging_path}, including the "
                "embedded pattern set. The separate pattern store was not "
                "updated; reload the source and review that staging file.",
            )
            return
        except operations.OperationError as exc:
            progress.failure(
                str(exc),
                _completion_refusal_note(session.config, operation_id),
            )
            return
        except JankiError as exc:
            progress.failure(
                str(exc),
                _completion_error_note(
                    session.config,
                    operation_id,
                ),
            )
            return
        progress.success(outcome)

    def _approve(self, source: str, body: bytes) -> None:
        session = self.server.session
        try:
            form = review.parse_review_form(body)
        except review.PanelRequestError as exc:
            self._error(400, str(exc))
            return
        # Authority first, and constant-time: a form that cannot prove it came
        # from this session's page is refused before its contents are read for
        # anything else.
        if not secrets.compare_digest(form["csrf"][0], session.csrf_token):
            self._error(403, "That form did not come from this workbench session.")
            return
        panel = session.panel(source)
        if panel is None:
            self._error(404, "No such source.")
            return
        # The snapshot is the compare-and-swap. A form rendered against older
        # bytes is refused whole — never partly applied to what is there now.
        if (
            form["staging_snapshot"][0] != panel.staging_fingerprint
            or form["patterns_snapshot"][0] != panel.patterns_fingerprint
        ):
            self._error(
                409,
                "This source changed after the page was rendered. Nothing was "
                "written. Reload and review the current cards.",
            )
            return
        try:
            outcome = panel.submit(
                record_ids=form.get("record", []),
                review_patterns=bool(form.get("patterns", [])),
            )
        except review.PanelRequestError as exc:
            self._error(400, str(exc))
        except review.StaleReviewError as exc:
            self._error(409, f"{exc}. Nothing was written by this attempt.")
        except review.PartialReviewError as exc:
            # Never "nothing changed": say exactly which file landed.
            self._error(409, str(exc))
        except JankiError as exc:
            self._error(500, f"{exc}. Nothing was proven written.")
        else:
            self._redirect_to_source(
                source,
                saved=len(outcome.accepted_record_ids),
                grammar=outcome.pattern_reviewed,
            )

    def _edit(self, source: str, body: bytes) -> None:
        session = self.server.session
        try:
            pairs = parse_qsl(
                body.decode("utf-8", errors="strict"),
                keep_blank_values=True,
                strict_parsing=True,
                encoding="utf-8",
                errors="strict",
                max_num_fields=4096,
            )
            submission = edit.parse_edit_form(pairs)
        except (UnicodeError, ValueError) as exc:
            self._error(400, f"The submitted form is malformed: {exc}")
            return
        except edit.EditError as exc:
            self._error(400, str(exc))
            return
        if not secrets.compare_digest(submission.csrf, session.csrf_token):
            self._error(403, "That form did not come from this workbench session.")
            return
        panel = session.panel(source)
        if panel is None:
            self._error(404, "No such source.")
            return
        if submission.staging_snapshot != panel.staging_fingerprint:
            self._error(
                409,
                "This source changed after the page was rendered. Nothing was "
                "written. Reload and edit the current cards.",
            )
            return
        try:
            updated = edit.apply_edits(panel.records, submission)
        except edit.EditError as exc:
            self._error(400, str(exc))
            return
        changed = edit.changed_records(panel.records, updated)
        if not changed:
            # Nothing to write, so nothing is written. Rewriting an identical
            # file would still touch its mtime and its Git status for no
            # reason, and would report a save that changed nothing.
            self._redirect_to_source(source, saved=0, grammar=False, edited=0)
            return
        try:
            text = staging.render_staging_update(panel.staging_path, updated)
            review.bound_replace(
                panel.staging_path,
                text,
                panel.staging_bytes,
                label="staging file",
            )
        except review.StaleReviewError as exc:
            # Proven: the compare-and-swap refused before writing anything.
            self._error(409, f"{exc}. Nothing was written by this attempt.")
            return
        except review.IndeterminateWriteError as exc:
            # NOT proven. The write failed after its snapshot stopped matching,
            # so whether it landed is unknown — and "nothing was written" would
            # be a lie at exactly the moment someone needs the truth.
            self._error(
                409,
                f"{exc} Reload this source and check the cards before editing "
                "again.",
            )
            return
        except JankiError as exc:
            self._error(500, f"{exc}. Nothing was proven written.")
            return
        self._redirect_to_source(source, saved=0, grammar=False, edited=len(changed))

    def _remove(self, source: str, body: bytes) -> None:
        """Drop one proposed card from the review file.

        Destructive in a way editing is not: the row is the model's proposal,
        and once it leaves a live staging file nothing else in the repository
        holds it. The page says so before the button is pressed.
        """
        session = self.server.session
        try:
            pairs = parse_qsl(
                body.decode("utf-8", errors="strict"),
                keep_blank_values=True,
                strict_parsing=True,
                encoding="utf-8",
                errors="strict",
                max_num_fields=64,
            )
        except (UnicodeError, ValueError) as exc:
            self._error(400, f"The submitted form is malformed: {exc}")
            return
        fields: dict[str, list[str]] = {}
        for key, value in pairs:
            fields.setdefault(key, []).append(value)
        if sorted(fields) != ["action", "card", "csrf", "staging_snapshot"] or any(
            len(values) != 1 for values in fields.values()
        ):
            self._error(400, "That removal form is not one this page offered.")
            return
        if fields["action"][0] != "remove":
            self._error(400, "The form action must be 'remove'")
            return
        if not secrets.compare_digest(fields["csrf"][0], session.csrf_token):
            self._error(403, "That form did not come from this workbench session.")
            return
        panel = session.panel(source)
        if panel is None:
            self._error(404, "No such source.")
            return
        if fields["staging_snapshot"][0] != panel.staging_fingerprint:
            self._error(
                409,
                "This source changed after the page was rendered. Nothing was "
                "removed. Reload and review the current cards.",
            )
            return
        try:
            index = int(fields["card"][0])
        except ValueError:
            self._error(400, "That removal names no card.")
            return
        if not 0 <= index < len(panel.records):
            self._error(400, "That removal names a card which is not on this page.")
            return
        keep = [position != index for position in range(len(panel.records))]
        try:
            text = staging.render_staging_prune(panel.staging_path, keep)
            if text is None:
                self._redirect_to_source(source, removed=0)
                return
            review.bound_replace(
                panel.staging_path,
                text,
                panel.staging_bytes,
                label="staging file",
            )
        except review.StaleReviewError as exc:
            self._error(409, f"{exc}. Nothing was removed by this attempt.")
            return
        except review.IndeterminateWriteError as exc:
            self._error(
                409,
                f"{exc} Reload this source and check which cards are there.",
            )
            return
        except JankiError as exc:
            self._error(500, f"{exc}. Nothing was proven removed.")
            return
        self._redirect_to_source(source, removed=1)

    def _assign(self, source: str, body: bytes) -> None:
        """Persist one exact, freshly re-planned thematic deck assignment."""
        session = self.server.session
        try:
            pairs = parse_qsl(
                body.decode("utf-8", errors="strict"),
                keep_blank_values=True,
                strict_parsing=True,
                encoding="utf-8",
                errors="strict",
                max_num_fields=64,
            )
        except (UnicodeError, ValueError) as exc:
            self._error(400, f"The submitted form is malformed: {exc}")
            return
        fields: dict[str, list[str]] = {}
        for key, value in pairs:
            fields.setdefault(key, []).append(value)
        expected = {
            "action",
            "card",
            "csrf",
            "destination",
            "plan_fingerprint",
            "staging_snapshot",
        }
        if set(fields) != expected or any(
            len(values) != 1 for values in fields.values()
        ):
            self._error(400, "That deck assignment form is not one this page offered.")
            return
        if fields["action"][0] != "assign":
            self._error(400, "The form action must be 'assign'")
            return
        if not secrets.compare_digest(fields["csrf"][0], session.csrf_token):
            self._error(403, "That form did not come from this workbench session.")
            return

        # Fresh per click: both the exact staging bytes and every real deck
        # selector may have changed since the source page rendered.
        panel = session.panel(source)
        if panel is None:
            self._error(404, "No such source.")
            return
        if fields["staging_snapshot"][0] != panel.staging_fingerprint:
            self._error(
                409,
                "This source changed after the page was rendered. Nothing was "
                "assigned. Reload and review the current card.",
            )
            return
        try:
            card = int(fields["card"][0])
        except ValueError:
            self._error(400, "That assignment names no card.")
            return
        if not 0 <= card < len(panel.records):
            self._error(400, "That assignment names a card which is not on this page.")
            return
        try:
            decks = assignable_word_decks(session.config)
            siblings = _staged_sibling_proposals(
                session.config,
                panel.staging_path,
                frozenset({panel.records[card].id}),
            )
            offer = _assignment_offer(
                session.config,
                panel.records[card],
                card,
                decks,
                sibling_proposals=siblings.get(panel.records[card].id, ()),
            )
        except JankiError as exc:
            self._error(
                409,
                f"The study-deck plan could not be refreshed: {exc} Nothing "
                "was assigned.",
            )
            return
        if not secrets.compare_digest(
            fields["plan_fingerprint"][0], offer.fingerprint
        ):
            self._error(
                409,
                "The study-deck plan changed after this page was rendered. "
                "Nothing was assigned. Reload and review the current tag diff.",
            )
            return
        destination = fields["destination"][0]
        choice = next(
            (candidate for candidate in offer.choices if candidate.stem == destination),
            None,
        )
        if choice is None:
            self._error(400, "That assignment names no configured study deck.")
            return
        if choice.plan is None:
            self._error(
                409,
                f"That study deck cannot take this card: {choice.refusal} "
                "Nothing was assigned.",
            )
            return

        updated = list(panel.records)
        updated[card] = choice.plan.assigned_record
        try:
            text = staging.render_staging_update(panel.staging_path, updated)
            review.bound_replace(
                panel.staging_path,
                text,
                panel.staging_bytes,
                label="staging file",
            )
        except review.StaleReviewError as exc:
            self._error(409, f"{exc}. Nothing was assigned by this attempt.")
            return
        except review.IndeterminateWriteError as exc:
            self._error(
                409,
                f"{exc} Reload this source and check the card's study deck.",
            )
            return
        except JankiError as exc:
            self._error(500, f"{exc}. Nothing was proven assigned.")
            return
        self._redirect_to_source(source, assigned=1)

    def _reidentify(self, source: str, body: bytes) -> None:
        """Two passes over one route: preview, then apply.

        The first submission renders what the change would do — the new
        identity, the neighbours it would sit beside or collide with, and what
        happens to review history. Only a second submission carrying the exact
        identity that preview showed actually writes. Binding the confirmation
        to that string is what stops the form drifting between the page
        someone read and the change they authorised.
        """
        session = self.server.session
        try:
            pairs = parse_qsl(
                body.decode("utf-8", errors="strict"),
                keep_blank_values=True,
                strict_parsing=True,
                encoding="utf-8",
                errors="strict",
                max_num_fields=64,
            )
        except (UnicodeError, ValueError) as exc:
            self._error(400, f"The submitted form is malformed: {exc}")
            return
        fields: dict[str, list[str]] = {}
        for key, value in pairs:
            fields.setdefault(key, []).append(value)
        required = {
            "action", "card", "csrf", "staging_snapshot", "expression", "reading",
        }
        if not required <= set(fields) or any(
            len(values) != 1 for values in fields.values()
        ):
            self._error(400, "That form is not one this page offered.")
            return
        # `confirm` is the only optional field; anything else means the form is
        # not the one this page rendered.
        unknown = set(fields) - required - {"confirm"}
        if unknown:
            self._error(
                400, f"That form carries a field this page does not offer: "
                f"{', '.join(sorted(unknown))}"
            )
            return
        if fields["action"][0] != "reidentify":
            self._error(400, "The form action must be 'reidentify'")
            return
        if not secrets.compare_digest(fields["csrf"][0], session.csrf_token):
            self._error(403, "That form did not come from this workbench session.")
            return
        # One read, not two: `detail` and `panel` each re-open the file, and
        # a plan computed from one while the page is rendered from the other
        # would describe a state nothing was actually looked at.
        panel = session.panel(source)
        if panel is None:
            self._error(404, "No such source.")
            return
        if not panel.reidentifiable:
            # The page does not offer this control here, but the page is not
            # what writes. A model pass answered about the word it was asked
            # about, and moving that answer to another identity would record
            # it saying something it never said.
            self._error(
                400,
                "These proposals came from a paid model pass about words your "
                "collection already holds, so they cannot be moved to a "
                "different word. Remove the card instead.",
            )
            return
        if fields["staging_snapshot"][0] != panel.staging_fingerprint:
            self._error(
                409,
                "This source changed after the page was rendered. Nothing was "
                "changed. Reload and review the current cards.",
            )
            return
        try:
            index = int(fields["card"][0])
        except ValueError:
            self._error(400, "That form names no card.")
            return
        try:
            plan = reidentify.plan_reidentification(
                panel.records,
                index,
                fields["expression"][0],
                fields["reading"][0],
                existing=session.canonical_records(),
                exported_ids=session.exported_ids(panel.records),
            )
        except reidentify.ReidentifyError as exc:
            self._error(400, str(exc))
            return

        confirmed = fields.get("confirm", [""])[0]
        if confirmed != plan.new_id:
            # First pass, or a confirmation that does not match what the
            # preview showed. Either way: show, do not write.
            self._send(
                200,
                render_reidentify(
                    source,
                    plan,
                    token=session.token,
                    csrf=session.csrf_token,
                    staging_snapshot=panel.staging_fingerprint,
                ),
            )
            return
        try:
            updated = reidentify.apply_reidentification(panel.records, plan)
        except reidentify.ReidentifyError as exc:
            self._error(409, str(exc))
            return
        try:
            text = staging.render_staging_update(panel.staging_path, updated)
            review.bound_replace(
                panel.staging_path, text, panel.staging_bytes, label="staging file"
            )
        except review.StaleReviewError as exc:
            self._error(409, f"{exc}. Nothing was changed by this attempt.")
            return
        except review.IndeterminateWriteError as exc:
            self._error(409, f"{exc} Reload this source and check the card.")
            return
        except JankiError as exc:
            self._error(500, f"{exc}. Nothing was proven changed.")
            return
        self._redirect_to_source(source, reidentified=1)

    def _redirect_to_source(
        self,
        source: str,
        *,
        saved: int = 0,
        grammar: bool = False,
        edited: int = 0,
        removed: int = 0,
        reidentified: int = 0,
        assigned: int = 0,
    ) -> None:
        """Post/redirect/get, so a reload cannot resubmit a write."""
        target = (
            f"/{self.server.session.token}/source/{quote(source, safe='')}"
            f"?saved={saved}&grammar={'1' if grammar else '0'}"
            f"&edited={edited}&removed={removed}&reidentified={reidentified}"
            f"&assigned={assigned}"
        )
        self.send_response(303)
        self.send_header("Location", target)
        self.send_header("Content-Length", "0")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.close_connection = True


def _parse_upload(body: bytes, content_type: str) -> tuple[str, bytes, dict[str, str]]:
    """`(filename, file bytes, other fields)` from one multipart submission.

    Built on `email`, which has parsed MIME for decades, rather than on a
    hand-rolled boundary split — the boundary rules have enough corner cases
    (quoting, epilogues, a boundary appearing inside the payload) that a fresh
    implementation is a guess about attacker-shaped input.

    Exactly one file part is accepted. Two would raise the question of which
    one the consent screen later names, and a screen that names the wrong file
    is worse than no upload at all.
    """
    parsed = BytesParser(policy=email_policy.default).parsebytes(
        b"Content-Type: " + content_type.encode("latin-1") + b"\r\n\r\n" + body
    )
    if not parsed.is_multipart():
        raise ValueError("it is not a multipart form")

    files: list[tuple[str, bytes]] = []
    fields: dict[str, str] = {}
    for part in parsed.iter_parts():
        disposition = part.get("Content-Disposition")
        if disposition is None:
            continue
        name = part.get_param("name", header="content-disposition")
        filename = part.get_param("filename", header="content-disposition")
        payload = part.get_payload(decode=True)
        if filename is not None:
            files.append((str(filename), payload or b""))
            continue
        if name is not None:
            fields[str(name)] = (payload or b"").decode("utf-8", errors="replace")
    if not files:
        raise ValueError("it carried no file")
    if len(files) > 1:
        raise ValueError("it carried more than one file; add them one at a time")
    return files[0][0], files[0][1], fields


def make_server(session: WorkbenchSession) -> _WorkbenchServer:
    """Create, but do not start, an ephemeral IPv4-loopback workbench server."""
    server = _WorkbenchServer(("127.0.0.1", 0), _WorkbenchHandler)
    server.session = session
    bind_loopback(server)
    return server


def serve(config: ProjectConfig, *, open_browser: bool = True) -> str:
    """Run the workbench until interrupted, and return the URL it served.

    Unlike the review panel there is no terminal request: the page is a view,
    so the only thing that ends it is the person who started it.
    """
    session = WorkbenchSession.open(config)
    server = make_server(session)
    url = f"http://{server.expected_host}/{session.token}/"
    print(f"Workbench: {url}", flush=True)
    print(
        "That address carries this session's key — it stops working when you "
        "stop the workbench. Ctrl-C to stop.",
        flush=True,
    )
    if open_browser and not webbrowser.open(url):
        print(
            "warning: could not open a browser; copy the URL above",
            file=sys.stderr,
        )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nWorkbench stopped.")
    finally:
        server.server_close()
    return url
