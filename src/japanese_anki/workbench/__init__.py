"""The local browser workbench (`WORKBENCH_PLAN.md` milestone W).

W1.2 ships the read-only dashboard. Card editing (W2), intake and paid
extraction (W3), deck assignment and promotion (W4) and the finish steps (W5)
land on top of this same server and the same `application` services the CLI
calls, so the page and the command can never disagree about what happened.
"""

from __future__ import annotations

from japanese_anki.workbench.render import STYLE, render_dashboard, render_source
from japanese_anki.workbench.review import ReviewOutcome, ReviewPanel
from japanese_anki.workbench.server import WorkbenchSession, make_server, serve

__all__ = [
    "STYLE",
    "ReviewOutcome",
    "ReviewPanel",
    "WorkbenchSession",
    "make_server",
    "render_dashboard",
    "render_source",
    "serve",
]
