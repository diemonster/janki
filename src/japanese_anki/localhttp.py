"""The localhost boundary both browser surfaces sit behind.

The review panel established these rules first, under adversarial review: an
ephemeral IPv4-loopback port, exactly one expected `Host`, an `Origin` that is
either absent or that same authority, restrictive response headers on *every*
reply including errors, and non-daemon handler threads so a shutdown waits for
an in-flight write instead of abandoning it.

The workbench needs the identical boundary for a long-lived page, so the rules
live here once rather than being copied. A second copy of a security header
list is a copy that drifts, and the drift is silent: nothing fails a test when
one surface quietly stops sending `X-Frame-Options`.

What this module does *not* decide is authorization. Loopback-only is a
network boundary, not a permission — every process on the machine can reach a
127.0.0.1 port, so a surface that exposes private study content adds its own
session secret on top (see `workbench.server`). The review panel does not,
because it is one-shot: it renders a form, accepts exactly one submission, and
shuts down.
"""

from __future__ import annotations

import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

__all__ = [
    "MAX_BODY_BYTES",
    "REQUEST_TIMEOUT_SECONDS",
    "LocalOnlyHandler",
    "LocalOnlyServer",
    "bind_loopback",
]

#: The largest form body either surface will read. Both are small HTML
#: forms; anything larger is a mistake or an attempt to exhaust memory.
MAX_BODY_BYTES = 64 * 1024

#: How long one request may hold a handler thread. Short: every body this
#: serves is small, and a stalled socket must not pin the process open.
REQUEST_TIMEOUT_SECONDS = 5.0


class LocalOnlyServer(ThreadingHTTPServer):
    # ThreadingHTTPServer opts into daemon handlers. A CLI cancellation followed
    # by server_close must instead wait until an in-flight write has finished,
    # so the command cannot return while it may still be writing to the repo.
    daemon_threads = False

    expected_host: str
    allowed_authorities: frozenset[str]


class LocalOnlyHandler(BaseHTTPRequestHandler):
    """Response headers and the loopback-origin check, for any local surface."""

    server: LocalOnlyServer
    server_version = "janki"
    sys_version = ""

    #: Shown in the `<title>` of a refusal page. Overridden per surface.
    error_title = "janki error"

    #: Per-surface, and read off the *class* rather than a module global so
    #: a test lowering it cannot miss (a module-level patch aimed at the
    #: wrong module silently does nothing, which is how a stalled-body test
    #: once kept passing while measuring the wrong thing).
    request_timeout_seconds = REQUEST_TIMEOUT_SECONDS

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(self.request_timeout_seconds)

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
        self._error(code, message or "The request was refused.")

    def _request_is_local(self) -> bool:
        hosts = self.headers.get_all("Host") or []
        if len(hosts) != 1 or hosts[0] not in self.server.allowed_authorities:
            return False
        origins = self.headers.get_all("Origin") or []
        return not origins or origins == [f"http://{hosts[0]}"]

    def _send(
        self,
        status: int,
        body: str | bytes,
        *,
        content_type: str = "text/html; charset=utf-8",
    ) -> None:
        payload = body.encode("utf-8") if isinstance(body, str) else body
        self._start_response(
            status,
            content_type=content_type,
            content_length=len(payload),
        )
        self.wfile.write(payload)
        self.close_connection = True

    def _start_response(
        self,
        status: int,
        *,
        content_type: str = "text/html; charset=utf-8",
        content_length: int | None = None,
    ) -> None:
        """Send the shared boundary headers for a fixed or streaming body."""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        if content_length is not None:
            self.send_header("Content-Length", str(content_length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        # Chrome serializes a same-origin form POST as Origin: null under
        # no-referrer. Preserve the exact localhost Origin that the boundary
        # check above requires while still suppressing cross-origin referrers.
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; style-src 'self'; form-action 'self'; "
            "base-uri 'none'; frame-ancestors 'none'",
        )
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def _error(self, status: int, message: str) -> None:
        self._send(status, self._error_html(message))

    @classmethod
    def _error_html(cls, message: str) -> str:
        return (
            "<!doctype html><html lang=en><head><meta charset=utf-8>"
            f"<title>{html.escape(cls.error_title)}</title></head><body><main>"
            f"<h1>{html.escape(cls.error_title)}</h1>"
            f'<p role="alert">{html.escape(message)}</p></main></body></html>'
        )


def bind_loopback(server: LocalOnlyServer) -> LocalOnlyServer:
    """Record the one authority pair a bound loopback server will accept."""
    port = server.server_address[1]
    server.expected_host = f"127.0.0.1:{port}"
    server.allowed_authorities = frozenset({server.expected_host, f"localhost:{port}"})
    return server
