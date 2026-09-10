"""The owner's layout editor: their columns, janki's identities.

Where a printed source-form layout comes from. A model may ask janki to open
this editor (`open_layout_editor`) and may not author, reorder or alter a
column — it has not seen the page, because intake places no source bytes in
model context. What the owner types here is the exact printed heading they
read, the display label they chose and which published parts the revision
covers; janki mints the opaque column and layout identities and the revision
number, so no machine identifier has to be invented in a text editor and
carried between two commands.

The document is a *view*, like ``assistant_previews`` and the region editor:
an opaque token over bytes held in this process, consuming no capability and
writing nothing by itself. Its one owner control — Save — calls the job's own
compare-and-swap writer through the adapter, which appends the immutable
revision and repoints every bound part in one write. A reversible local save
asks for no second confirmation and buys nothing.

Nothing here reads a heading, matches a printed label to a form, or decides
what a column means. The owner does that language work; this module carries
their answer verbatim.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import secrets
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "EDITOR_UNAVAILABLE_MESSAGE",
    "MAX_COLUMNS",
    "MAX_EDITORS",
    "MAX_EDITOR_BYTES",
    "LayoutEditorDocument",
    "LayoutEditorError",
    "LayoutEditorOffer",
    "LayoutEditorSession",
    "LocalLayoutEditorStore",
    "compose_layout",
    "render_layout_editor",
]

_TOKEN_BYTES = 32
#: A working number of open editors; the oldest is evicted first.
MAX_EDITORS = 8
#: One self-contained document. It carries no page images, so this is a bound
#: on a pathological job rather than a working limit.
MAX_EDITOR_BYTES = 4 * 1024 * 1024
#: A printed table this long is a mistake about the page, not a layout.
MAX_COLUMNS = 64

EDITOR_UNAVAILABLE_MESSAGE = (
    "This layout editor link is unknown or has expired. Janki keeps editors "
    "only in memory while this workbench is running, and a bounded number of "
    "them. Ask Janki to open the layout editor for that job again — rendering "
    "it reads the repository locally, writes nothing and saves nothing."
)


class LayoutEditorError(RuntimeError):
    """A safe refusal shown instead of serving or acting on an editor."""


@dataclass(frozen=True, slots=True)
class LayoutEditorDocument:
    """One immutable editor document and the policy its bytes were built for."""

    html: bytes
    sha256: str
    content_security_policy: str
    job_id: str
    part_count: int
    layout_count: int


@dataclass(frozen=True, slots=True)
class LayoutEditorOffer:
    """One opaque editor binding rendered into the thread as a link."""

    token: str
    url: str
    job_id: str
    part_count: int
    layout_count: int
    bound_count: int
    byte_count: int
    sha256: str


@dataclass(frozen=True, slots=True)
class LayoutEditorSession:
    """What this editor is bound to: one study job, and nothing else."""

    token: str
    document: LayoutEditorDocument
    job_id: str


def _mint(prefix: str) -> str:
    """One opaque local identity. It encodes nothing about the column."""

    return f"{prefix}-{secrets.token_hex(6)}"


def compose_layout(
    submitted: Mapping[str, Any],
    *,
    layouts: Sequence[Mapping[str, Any]],
    mint: Callable[[str], str] = _mint,
) -> dict[str, Any]:
    """Turn one owner's editor save into the frozen layout wire.

    The owner supplies the printed witnesses, the display labels and the order.
    Janki supplies every machine identifier: a column the owner kept keeps the
    identity it already had — that is what makes an edit an edit rather than a
    new table — and a column they added gets a freshly minted opaque one. The
    revision is the next whole number after the highest this job records for
    that layout id, so nobody tracks it by hand.

    ``layouts`` is every revision the job already records, in its frozen wire.
    A retained ``column_id`` has to be one of the base revision's own, so a
    posted identity cannot invent or borrow one.
    """

    base_id = submitted.get("base_layout_id") or ""
    base_revision = submitted.get("base_revision")
    if not isinstance(base_id, str):
        raise LayoutEditorError("The layout this edit starts from is named by id.")
    known: dict[tuple[str, Any], Mapping[str, Any]] = {
        (str(entry.get("layout_id")), entry.get("revision")): entry
        for entry in layouts
        if isinstance(entry, Mapping)
    }
    base: Mapping[str, Any] | None = None
    if base_id:
        base = known.get((base_id, base_revision))
        if base is None:
            raise LayoutEditorError(
                f"This job records no layout {base_id} revision "
                f"{base_revision!r} to edit. Reopen the editor; nothing was "
                "saved."
            )
    retained = {
        str(column.get("column_id"))
        for column in (base.get("columns") or () if base is not None else ())
        if isinstance(column, Mapping)
    }

    raw_columns = submitted.get("columns")
    if not isinstance(raw_columns, list) or not raw_columns:
        raise LayoutEditorError(
            "A layout declares at least one printed column. Nothing was saved."
        )
    if len(raw_columns) > MAX_COLUMNS:
        raise LayoutEditorError(
            f"A layout declares up to {MAX_COLUMNS} printed columns; this one "
            f"declares {len(raw_columns)}. Nothing was saved."
        )
    columns: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ordinal, entry in enumerate(raw_columns, start=1):
        if not isinstance(entry, Mapping):
            raise LayoutEditorError(f"Column {ordinal} is not an object.")
        label = entry.get("display_label")
        witnesses = entry.get("label_witnesses")
        if not isinstance(label, str) or not label:
            raise LayoutEditorError(
                f"Column {ordinal} needs the display label you want on the "
                "card. Two columns may share one; an empty one is not a label."
            )
        if (
            not isinstance(witnesses, list)
            or not witnesses
            or any(not isinstance(item, str) for item in witnesses)
        ):
            raise LayoutEditorError(
                f"Column {ordinal} needs at least one printed heading, exactly "
                "as the page prints it."
            )
        column_id = entry.get("column_id") or ""
        if not isinstance(column_id, str):
            raise LayoutEditorError(f"Column {ordinal}'s identity is not a string.")
        if column_id:
            if column_id not in retained:
                raise LayoutEditorError(
                    f"Column {ordinal} names identity {column_id}, which is not "
                    "one of the columns this edit started from. Janki mints "
                    "these; nothing was saved."
                )
        else:
            column_id = mint("col")
            while column_id in seen or column_id in retained:
                column_id = mint("col")
        if column_id in seen:
            raise LayoutEditorError(
                f"Column {ordinal} reuses identity {column_id}; one identity "
                "names one column. Nothing was saved."
            )
        seen.add(column_id)
        columns.append(
            {
                "column_id": column_id,
                "ordinal": ordinal,
                # Verbatim, in the order they were typed. No trim, no NFC.
                "label_witnesses": list(witnesses),
                "display_label": label,
            }
        )

    layout_id = base_id
    if not layout_id:
        layout_id = mint("layout")
        while any(recorded == layout_id for recorded, _revision in known):
            layout_id = mint("layout")
    revisions = [
        revision
        for recorded, revision in known
        if recorded == layout_id and isinstance(revision, int)
    ]
    return {
        "layout_id": layout_id,
        "revision": (max(revisions) + 1) if revisions else 1,
        "columns": columns,
    }


#: The editor's whole behaviour: load a revision, edit the rows, save one.
#: Deliberately one inline script whose hash is the only ``script-src`` the
#: document is served under, so nothing else in the page can run.
_SCRIPT = """
(function () {
  "use strict";
  const state = JSON.parse(document.getElementById("janki-state").textContent);
  const submit = __JANKI_SUBMIT_URL__;
  let rows = [];
  let baseId = "";
  let baseRevision = null;

  function say(message, kind) {
    const box = document.getElementById("janki-status");
    box.textContent = message;
    box.className = "status " + (kind || "");
  }

  function redraw() {
    const list = document.getElementById("janki-columns");
    list.textContent = "";
    rows.forEach(function (row, index) {
      const item = document.createElement("li");

      const label = document.createElement("input");
      label.type = "text";
      label.className = "label";
      label.value = row.display_label;
      label.placeholder = "Display label";
      label.addEventListener("input", function () {
        row.display_label = label.value;
      });

      const witnesses = document.createElement("textarea");
      witnesses.className = "witnesses";
      witnesses.value = row.label_witnesses.join("\\n");
      witnesses.placeholder = "Printed heading, exactly as the page prints it";
      witnesses.addEventListener("input", function () {
        row.label_witnesses = witnesses.value.split("\\n");
      });

      const up = document.createElement("button");
      up.type = "button";
      up.textContent = "Move up";
      up.disabled = index === 0;
      up.addEventListener("click", function () {
        rows.splice(index - 1, 0, rows.splice(index, 1)[0]);
        redraw();
      });

      const remove = document.createElement("button");
      remove.type = "button";
      remove.textContent = "Remove";
      remove.addEventListener("click", function () {
        rows.splice(index, 1);
        redraw();
      });

      const kept = document.createElement("span");
      kept.className = "note";
      kept.textContent = row.column_id
        ? " keeps identity " + row.column_id
        : " new column — janki mints its identity on save";

      item.appendChild(document.createTextNode("Column " + (index + 1) + " "));
      item.appendChild(label);
      item.appendChild(witnesses);
      item.appendChild(up);
      item.appendChild(remove);
      item.appendChild(kept);
      list.appendChild(item);
    });
  }

  function load(key) {
    const chosen = state.layouts.filter(function (layout) {
      return layout.layout_id + "@" + layout.revision === key;
    })[0];
    if (!chosen) {
      baseId = "";
      baseRevision = null;
      rows = [];
      redraw();
      say("Starting a new layout. Janki mints its identity and revision.", "");
      return;
    }
    baseId = chosen.layout_id;
    baseRevision = chosen.revision;
    rows = chosen.columns.map(function (column) {
      return {
        column_id: column.column_id,
        display_label: column.display_label,
        label_witnesses: column.label_witnesses.slice()
      };
    });
    redraw();
    say("Editing " + baseId + " revision " + baseRevision +
        ". Saving appends the next revision; this one is immutable.", "");
  }

  document.getElementById("janki-base").addEventListener("change", function (event) {
    load(event.target.value);
  });

  document.getElementById("janki-add").addEventListener("click", function () {
    rows.push({ column_id: "", display_label: "", label_witnesses: [""] });
    redraw();
  });

  document.getElementById("janki-save").addEventListener("click", function () {
    const parts = [];
    state.parts.forEach(function (part) {
      const box = document.getElementById("janki-part-" + part.index);
      if (box.checked) { parts.push(part.name); }
    });
    if (!parts.length) {
      say("Choose at least one published part this layout covers.", "bad");
      return;
    }
    const columns = rows.map(function (row) {
      const witnesses = row.label_witnesses.filter(function (value) {
        return value !== "";
      });
      const entry = {
        display_label: row.display_label,
        label_witnesses: witnesses
      };
      if (row.column_id) { entry.column_id = row.column_id; }
      return entry;
    });
    say("Saving…", "");
    const request = new XMLHttpRequest();
    request.open("POST", submit, true);
    request.setRequestHeader("Content-Type", "application/json");
    request.onreadystatechange = function () {
      if (request.readyState !== 4) { return; }
      let payload = {};
      try { payload = JSON.parse(request.responseText); } catch (error) { payload = {}; }
      if (request.status === 200 && payload.ok) {
        state.layouts = payload.layouts;
        state.parts = payload.parts;
        const chooser = document.getElementById("janki-base");
        chooser.textContent = "";
        const fresh = document.createElement("option");
        fresh.value = "";
        fresh.textContent = "New layout";
        chooser.appendChild(fresh);
        state.layouts.forEach(function (layout) {
          const option = document.createElement("option");
          option.value = layout.layout_id + "@" + layout.revision;
          option.textContent = layout.layout_id + " revision " + layout.revision;
          chooser.appendChild(option);
        });
        chooser.value = payload.layout_id + "@" + payload.revision;
        load(chooser.value);
        say(payload.message, "good");
      } else {
        say(payload.error || "Janki refused that save.", "bad");
      }
    };
    request.send(JSON.stringify({
      action: "save",
      base_layout_id: baseId,
      base_revision: baseRevision,
      columns: columns,
      bind: parts
    }));
  });

  redraw();
})();
"""


def _script(submit_url: str) -> str:
    """The script bound to one editor's own submit URL."""

    return _SCRIPT.replace("__JANKI_SUBMIT_URL__", json.dumps(submit_url))


def _policy(script: str) -> str:
    """The exact policy these bytes are served under, script hash included."""

    digest = base64.b64encode(hashlib.sha256(script.encode("utf-8")).digest()).decode(
        "ascii"
    )
    return (
        "default-src 'none'; "
        f"script-src 'sha256-{digest}'; "
        "style-src 'unsafe-inline'; "
        "connect-src 'self'; "
        "base-uri 'none'; "
        "form-action 'none'; "
        "object-src 'none'; "
        "frame-ancestors 'none'"
    )


_STYLE = """
:root { color-scheme: light dark; }
body { font: 15px/1.5 -apple-system, system-ui, sans-serif; margin: 0 auto;
       max-width: 60rem; padding: 1.5rem; }
h1 { font-size: 1.3rem; margin-bottom: 0.2rem; }
.note { color: #555; }
button { font: inherit; padding: 0.3rem 0.7rem; margin-right: 0.4rem; }
ul { padding-left: 1.2rem; }
li { margin-bottom: 0.8rem; }
input.label { font: inherit; width: 14rem; margin-right: 0.5rem; }
textarea.witnesses { font-family: ui-monospace, monospace; width: 22rem;
                     height: 3.4rem; vertical-align: top; margin-right: 0.5rem; }
.status { margin: 0.6rem 0; min-height: 1.4rem; }
.status.bad { color: #b03030; }
.status.good { color: #1d6f30; }
"""


def render_layout_editor(
    *,
    job_id: str,
    parts: Sequence[Mapping[str, Any]],
    layouts: Sequence[Mapping[str, Any]],
    submit_url: str,
) -> LayoutEditorDocument:
    """Build one self-contained editor over this job's parts and revisions."""

    if not parts:
        raise LayoutEditorError(
            f"Study job {job_id[:8]} has published no parts yet, so there is "
            "nothing to bind a printed layout to. Choose and publish the "
            "pages first; opening this editor changed nothing."
        )
    script = _script(submit_url)
    # Serialized before the document, and with `<` escaped: a JSON island
    # inside `<script>` ends at the first `</script>` the bytes contain, and
    # every string in here is the owner's.
    state_json = json.dumps(
        {
            "job_id": job_id,
            "parts": [
                {
                    "index": index,
                    "name": str(part.get("name", "")),
                    "bound": str(part.get("bound", "")),
                }
                for index, part in enumerate(parts)
            ],
            "layouts": [dict(layout) for layout in layouts],
        },
        ensure_ascii=False,
    ).replace("<", "\\u003c")
    part_rows = "\n".join(
        f"""<li><label><input type="checkbox" id="janki-part-{index}"
 {"checked" if part.get("bound") else ""}> {html.escape(str(part.get("name", "")))}</label>
<span class="note">{
    "currently " + html.escape(str(part.get("bound")))
    if part.get("bound")
    else "no layout bound"
}</span></li>"""
        for index, part in enumerate(parts)
    )
    options = "\n".join(
        ['<option value="">New layout</option>']
        + [
            f"""<option value="{html.escape(str(layout.get('layout_id')))}@{
                html.escape(str(layout.get('revision')))
            }">{html.escape(str(layout.get('layout_id')))} revision {
                html.escape(str(layout.get('revision')))
            }</option>"""
            for layout in layouts
        ]
    )
    document = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Printed columns — study job {html.escape(job_id[:8])}</title>
<style>{_STYLE}</style>
</head><body>
<h1>Set up the printed columns for study job {html.escape(job_id[:8])}</h1>
<p class="note">One row per column the page prints, in printed order. Type the
heading exactly as it appears — one per line if the page prints it more than one
way — and the label you want on the card. Two columns may share a label. Janki
mints the column identities and the revision number: nothing here asks you to
invent one, and it reads no heading and matches no label of its own.</p>
<p class="note">Saving appends a new immutable revision and points the parts you
tick at it. It is a local decision: no model call, no cost, and no card or deck
definition changes.</p>
<h2>Start from</h2>
<p><select id="janki-base">{options}</select></p>
<h2>Columns</h2>
<ul id="janki-columns"></ul>
<p><button type="button" id="janki-add">Add a column</button></p>
<h2>Parts this revision covers</h2>
<ul>{part_rows}</ul>
<p><button type="button" id="janki-save">Save this layout revision</button></p>
<div id="janki-status" class="status"></div>
<script type="application/json" id="janki-state">{state_json}</script>
<script>{script}</script>
</body></html>
"""
    payload = document.encode("utf-8")
    if len(payload) > MAX_EDITOR_BYTES:
        limit = MAX_EDITOR_BYTES // (1024 * 1024)
        raise LayoutEditorError(
            f"The layout editor for study job {job_id[:8]} is too large to "
            f"open ({len(payload)} bytes; Janki serves up to {limit} MiB). "
            "Save the revision with `janki study layout` instead."
        )
    return LayoutEditorDocument(
        html=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        content_security_policy=_policy(script),
        job_id=job_id,
        part_count=len(parts),
        layout_count=len(layouts),
    )


@dataclass(slots=True)
class LocalLayoutEditorStore:
    """Hold a bounded number of open layout editors behind opaque tokens.

    ``save_layout`` is the adapter's own owner route. It is held here rather
    than reached through the model's intent path because saving a layout is an
    owner action: no model-emittable field carries a column identity, a printed
    witness, a display label or a revision (contracts §9.1).
    """

    editor_prefix: str
    save_layout: Callable[..., dict[str, Any]]
    _sessions: dict[str, LayoutEditorSession] = field(
        default_factory=dict, init=False, repr=False
    )
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def open(
        self,
        *,
        job_id: str,
        parts: Sequence[Mapping[str, Any]],
        layouts: Sequence[Mapping[str, Any]],
    ) -> LayoutEditorOffer:
        """Mint one token, build the document it will be served as, keep both.

        ``job_id`` comes from the adapter's own owner route, never from a
        posted request body: the session decides which job an editor belongs
        to, exactly as the region editor's session decides which source it may
        name.
        """

        token = secrets.token_urlsafe(_TOKEN_BYTES)
        url = f"{self.editor_prefix}{token}"
        document = render_layout_editor(
            job_id=job_id, parts=parts, layouts=layouts, submit_url=url
        )
        session = LayoutEditorSession(
            token=token, document=document, job_id=job_id
        )
        with self._lock:
            while len(self._sessions) >= MAX_EDITORS:
                self._sessions.pop(next(iter(self._sessions)))
            self._sessions[token] = session
        return LayoutEditorOffer(
            token=token,
            url=url,
            job_id=job_id,
            part_count=document.part_count,
            layout_count=document.layout_count,
            bound_count=sum(1 for part in parts if part.get("bound")),
            byte_count=len(document.html),
            sha256=document.sha256,
        )

    def read(self, token: str) -> LayoutEditorDocument:
        """Return the exact retained document, or refuse. Never consumes it."""

        return self._session(token).document

    def _session(self, token: str) -> LayoutEditorSession:
        with self._lock:
            session = self._sessions.get(token)
        if session is None:
            raise LayoutEditorError(EDITOR_UNAVAILABLE_MESSAGE)
        return session

    def act(self, token: str, request: Any) -> dict[str, Any]:
        """The one owner control from an open editor: save a revision.

        The session decides which study job this editor may write to, so a
        job named in the posted body could describe nothing but the document
        the owner opened — and none is read from it.
        """

        session = self._session(token)
        if not isinstance(request, dict):
            raise LayoutEditorError("That editor request is not an object.")
        if request.get("action") != "save":
            raise LayoutEditorError(
                "A layout editor saves one revision; it does nothing else."
            )
        return self.save_layout(session.job_id, request)
