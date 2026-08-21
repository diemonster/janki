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

Read-only, in this milestone. There is no POST route at all — not a disabled
one, not a guarded one. Approval still happens in `janki review-panel`, which
this page links to; W2 folds that in.
"""

from __future__ import annotations

import secrets
import sys
import webbrowser
from dataclasses import dataclass
from urllib.parse import unquote

from japanese_anki.application import SourceJourney, source_detail, source_journeys
from japanese_anki.config import ProjectConfig
from japanese_anki.localhttp import (
    LocalOnlyHandler,
    LocalOnlyServer,
    bind_loopback,
)
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
    token: str

    @classmethod
    def open(cls, config: ProjectConfig) -> WorkbenchSession:
        return cls(config=config, token=secrets.token_urlsafe(32))

    def journeys(self) -> tuple[list[SourceJourney], list[str]]:
        return source_journeys(self.config)

    def detail(self, source: str):
        return source_detail(self.config, source)


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
            self._send(
                200, render_source(detail, token=self.server.session.token)
            )
            return
        self._error(404, "No such workbench page.")

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        """W1.2 is read-only. Every mutation still goes through the CLI."""
        if not self._request_is_local():
            self._error(403, "The workbench accepts only its exact localhost origin.")
            return
        self._error(405, "The workbench does not change anything yet.")


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
