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

import secrets
import sys
import webbrowser
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, quote, unquote

from japanese_anki.application import SourceJourney, source_detail, source_journeys
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.localhttp import (
    MAX_BODY_BYTES,
    LocalOnlyHandler,
    LocalOnlyServer,
    bind_loopback,
)
from japanese_anki.workbench import review
from japanese_anki.workbench.render import STYLE, render_dashboard, render_source

__all__ = ["WorkbenchSession", "make_server", "serve"]

_SOURCE_PREFIX = "/source/"


@dataclass(frozen=True, slots=True)
class WorkbenchSession:
    """One run of the workbench: its config, its secret, and nothing else.

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
            )
        except JankiError:
            return None


class _WorkbenchServer(LocalOnlyServer):
    session: WorkbenchSession


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

    def _saved_banner(self) -> tuple[int, bool] | None:
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
        return max(records, 0), fields.get("grammar") == "1"

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
                ),
            )
            return
        if route == "/style.css":
            self._send(200, STYLE, content_type="text/css; charset=utf-8")
            return
        if route.startswith(_SOURCE_PREFIX):
            # The name is matched against the dashboard's own computed list and
            # the staging path comes from that journey — the request never
            # names a path, so it cannot name one outside the corpus. Do not
            # "improve" this into a join against staging_dir.
            name = unquote(route[len(_SOURCE_PREFIX) :])
            detail = self.server.session.detail(name)
            if detail is None:
                self._error(404, "No such source.")
                return
            # The panel is opened here only for the exact bytes it read: those
            # fingerprints go into the form, and the approval is refused later
            # unless the file is still byte-identical to them.
            panel = self.server.session.panel(name)
            self._send(
                200,
                render_source(
                    detail,
                    token=self.server.session.token,
                    csrf=self.server.session.csrf_token if panel else "",
                    staging_snapshot=panel.staging_fingerprint if panel else "",
                    patterns_snapshot=panel.patterns_fingerprint if panel else "",
                    saved=self._saved_banner(),
                ),
            )
            return
        self._error(404, "No such workbench page.")

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
        if route is None or not route.startswith(_SOURCE_PREFIX):
            self._error(404, "No such workbench action.")
            return
        rest = route[len(_SOURCE_PREFIX) :]
        name, _, action = rest.rpartition("/")
        if action != "approve" or not name:
            self._error(404, "No such workbench action.")
            return
        body = self._read_body()
        if body is None:
            return
        self._approve(unquote(name), body)

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
            self._redirect_to_source(source, outcome)

    def _redirect_to_source(self, source: str, outcome: Any) -> None:
        """Post/redirect/get, so a reload cannot resubmit an approval."""
        target = (
            f"/{self.server.session.token}/source/{quote(source, safe='')}"
            f"?saved={len(outcome.accepted_record_ids)}"
            f"&grammar={'1' if outcome.pattern_reviewed else '0'}"
        )
        self.send_response(303)
        self.send_header("Location", target)
        self.send_header("Content-Length", "0")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.close_connection = True


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
