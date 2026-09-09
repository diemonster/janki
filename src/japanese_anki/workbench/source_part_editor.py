"""The owner's region editor: real pixels, owner controls, nothing inferred.

This is where geometry comes from. A model may ask janki to open this editor
(`open_source_part_editor`) and may not author, widen or alter a rectangle —
it never saw the source, because intake places no bytes in model context. What
the owner draws here is serialized into a recipe, and
``application.source_parts`` mints the token that binds it to those exact
bytes.

The document is a *view*, like ``assistant_previews``: an opaque token over
immutable bytes held in this process, consuming no capability, writing no
receipt and changing no canonical file. Its two owner controls — plan, then
publish — call the application service directly under the exact plan
fingerprint the owner reviewed. Preparing parts is ordinary local work: it adds
no approval gate, discloses nothing to a provider and spends nothing, so there
is no confirmation ladder here and no capability to consume.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import secrets
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "EDITOR_UNAVAILABLE_MESSAGE",
    "MAX_EDITORS",
    "MAX_EDITOR_BYTES",
    "SourcePartEditorDocument",
    "SourcePartEditorError",
    "SourcePartEditorOffer",
    "SourcePartEditorSession",
    "LocalSourcePartEditorStore",
    "render_source_part_editor",
]

_TOKEN_BYTES = 32
#: A working number of open editors; the oldest is evicted first.
MAX_EDITORS = 8
#: One self-contained document with every page inlined. Over this it refuses
#: with an explicit message rather than serving a truncated sheet.
MAX_EDITOR_BYTES = 64 * 1024 * 1024

EDITOR_UNAVAILABLE_MESSAGE = (
    "This source-part editor link is unknown or has expired. Janki keeps "
    "editors only in memory while this workbench is running, and a bounded "
    "number of them. Ask Janki to open the editor for that source again — "
    "rendering it reads the repository locally, writes nothing and publishes "
    "nothing."
)


class SourcePartEditorError(RuntimeError):
    """A safe refusal shown instead of serving or acting on an editor."""


@dataclass(frozen=True, slots=True)
class SourcePartEditorDocument:
    """One immutable editor document and the policy its bytes were built for."""

    html: bytes
    sha256: str
    content_security_policy: str
    source_name: str
    recipe_id: str
    page_count: int


@dataclass(frozen=True, slots=True)
class SourcePartEditorOffer:
    """One opaque editor binding rendered into the thread as a link."""

    token: str
    url: str
    source_name: str
    recipe_id: str
    page_count: int
    byte_count: int
    sha256: str


@dataclass(frozen=True, slots=True)
class SourcePartEditorSession:
    """What this editor is bound to: one source, one parent hash, one recipe."""

    token: str
    document: SourcePartEditorDocument
    source_name: str
    parent_sha256: str


#: The editor's whole behaviour: draw, list, plan, publish. Deliberately one
#: inline script whose hash is the only ``script-src`` the document is served
#: under, so nothing else in the page can run.
_SCRIPT = """
(function () {
  "use strict";
  const state = JSON.parse(document.getElementById("janki-state").textContent);
  const submit = __JANKI_SUBMIT_URL__;
  // An empty submit URL is a saved contact sheet written by
  // `janki source-parts choose`: the same chooser, with no workbench behind
  // it, so it plans and publishes from the terminal instead.
  const live = submit.length > 0;
  const parts = [];
  let planFingerprint = "";

  function say(message, kind) {
    const box = document.getElementById("janki-status");
    box.textContent = message;
    box.className = "status " + (kind || "");
  }

  function recipe() {
    return {
      version: 1,
      recipe_id: state.recipe_id,
      parent_name: state.parent_name,
      parent_sha256: state.parent_sha256,
      render_dpi: Number(document.getElementById("janki-dpi").value),
      parts: parts.map(function (part) {
        const entry = {
          page_index: part.page_index,
          page_rotate: part.page_rotate
        };
        if (part.regions.length) {
          entry.regions = part.regions.map(function (region) {
            return region.map(function (value) {
              return Math.round(value * 1000000) / 1000000;
            });
          });
        }
        return entry;
      })
    };
  }

  function redraw() {
    const list = document.getElementById("janki-parts");
    list.textContent = "";
    parts.forEach(function (part, index) {
      const row = document.createElement("li");
      const label = part.regions.length
        ? part.regions.length + " region(s) of page " + (part.page_index + 1)
        : "the whole of page " + (part.page_index + 1);
      row.appendChild(document.createTextNode(label + " "));
      const remove = document.createElement("button");
      remove.type = "button";
      remove.textContent = "Remove";
      remove.addEventListener("click", function () {
        parts.splice(index, 1);
        redraw();
      });
      row.appendChild(remove);
      list.appendChild(row);
    });
    document.getElementById("janki-recipe").value = JSON.stringify(recipe(), null, 2);
    planFingerprint = "";
    document.getElementById("janki-publish").disabled = true;
    document.getElementById("janki-plan").disabled = !live;
    document.getElementById("janki-plan-result").textContent = "";
  }

  function post(body, done) {
    const request = new XMLHttpRequest();
    request.open("POST", submit, true);
    request.setRequestHeader("Content-Type", "application/json");
    request.onreadystatechange = function () {
      if (request.readyState !== 4) { return; }
      let payload = {};
      try { payload = JSON.parse(request.responseText); } catch (error) { payload = {}; }
      if (request.status === 200 && payload.ok) {
        done(payload);
      } else {
        say(payload.error || "Janki refused that request.", "bad");
      }
    };
    request.send(JSON.stringify(body));
  }

  state.pages.forEach(function (page) {
    const frame = document.getElementById("janki-page-" + page.page_index);
    const overlay = frame.querySelector(".overlay");
    let pending = [];
    let start = null;
    let box = null;

    function normalized(event) {
      const rect = frame.getBoundingClientRect();
      return [
        Math.min(Math.max((event.clientX - rect.left) / rect.width, 0), 1),
        Math.min(Math.max((event.clientY - rect.top) / rect.height, 0), 1)
      ];
    }

    frame.addEventListener("mousedown", function (event) {
      event.preventDefault();
      start = normalized(event);
      box = document.createElement("div");
      box.className = "band pending";
      overlay.appendChild(box);
    });
    frame.addEventListener("mousemove", function (event) {
      if (!start || !box) { return; }
      const now = normalized(event);
      box.style.left = (Math.min(start[0], now[0]) * 100) + "%";
      box.style.top = (Math.min(start[1], now[1]) * 100) + "%";
      box.style.width = (Math.abs(now[0] - start[0]) * 100) + "%";
      box.style.height = (Math.abs(now[1] - start[1]) * 100) + "%";
    });
    frame.addEventListener("mouseup", function (event) {
      if (!start || !box) { return; }
      const now = normalized(event);
      const region = [
        Math.min(start[0], now[0]),
        Math.min(start[1], now[1]),
        Math.max(start[0], now[0]),
        Math.max(start[1], now[1])
      ];
      start = null;
      if (region[2] - region[0] < 0.01 || region[3] - region[1] < 0.01) {
        overlay.removeChild(box);
        box = null;
        say("That region is too small to render; drag a band across the page.", "bad");
        return;
      }
      box.className = "band";
      box = null;
      pending.push(region);
      pending.sort(function (a, b) { return a[1] - b[1]; });
      say(pending.length + " region(s) selected on page " + (page.page_index + 1) +
          ". Add this as a part when the selection is complete.", "");
    });

    frame.parentNode.querySelector(".add-regions").addEventListener("click", function () {
      if (!pending.length) {
        say("Select at least one region first, or add the whole page.", "bad");
        return;
      }
      parts.push({
        page_index: page.page_index,
        page_rotate: page.page_rotate,
        regions: pending
      });
      pending = [];
      overlay.textContent = "";
      redraw();
    });
    frame.parentNode.querySelector(".add-page").addEventListener("click", function () {
      parts.push({
        page_index: page.page_index,
        page_rotate: page.page_rotate,
        regions: []
      });
      redraw();
    });
    frame.parentNode.querySelector(".clear-page").addEventListener("click", function () {
      pending = [];
      overlay.textContent = "";
      say("Cleared the selection on page " + (page.page_index + 1) + ".", "");
    });
  });

  document.getElementById("janki-plan").addEventListener("click", function () {
    if (!live) {
      say("This saved sheet has no workbench behind it. Copy the recipe below " +
          "and run janki source-parts prepare --recipe FILE.", "bad");
      return;
    }
    if (!parts.length) {
      say("Choose at least one page or region first.", "bad");
      return;
    }
    say("Rendering these exact parts…", "");
    post({ action: "plan", recipe: recipe() }, function (payload) {
      planFingerprint = payload.plan_fingerprint;
      const result = document.getElementById("janki-plan-result");
      result.textContent = "";
      payload.parts.forEach(function (part) {
        const cell = document.createElement("figure");
        const image = document.createElement("img");
        image.src = "data:image/png;base64," + part.thumbnail_png_base64;
        image.alt = part.target_name;
        const caption = document.createElement("figcaption");
        caption.textContent = part.target_name + " — " + part.byte_length +
          " bytes — sha256 " + part.sha256.slice(0, 12) + "…";
        cell.appendChild(image);
        cell.appendChild(caption);
        result.appendChild(cell);
      });
      document.getElementById("janki-publish").disabled = false;
      say("Reviewed plan " + payload.plan_fingerprint.slice(0, 12) +
          "…. Publishing puts these exact bytes in your corpus.", "good");
    });
  });

  document.getElementById("janki-publish").addEventListener("click", function () {
    if (!planFingerprint) {
      say("Render the plan first; publishing binds the plan you reviewed.", "bad");
      return;
    }
    say("Publishing…", "");
    post({ action: "publish", plan_fingerprint: planFingerprint }, function (payload) {
      say(payload.message, "good");
      document.getElementById("janki-publish").disabled = true;
    });
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
        "img-src data:; "
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
.page { margin: 1.5rem 0; }
.frame { position: relative; display: inline-block; max-width: 100%; }
.frame img { display: block; max-width: 100%; height: auto; }
.overlay { position: absolute; inset: 0; }
.band { position: absolute; border: 2px solid #c0392b; background: rgba(192,57,43,0.18); }
.band.pending { border-style: dashed; }
button { font: inherit; padding: 0.3rem 0.7rem; margin-right: 0.4rem; }
ul { padding-left: 1.2rem; }
textarea { width: 100%; min-height: 9rem; font-family: ui-monospace, monospace; }
figure { display: inline-block; margin: 0 1rem 1rem 0; vertical-align: top; }
figure img { max-width: 18rem; border: 1px solid #999; }
figcaption { font-size: 0.8rem; max-width: 18rem; word-break: break-all; }
.status { margin: 0.6rem 0; min-height: 1.4rem; }
.status.bad { color: #b03030; }
.status.good { color: #1d6f30; }
"""


def render_source_part_editor(
    sheet: Any,
    *,
    recipe_id: str,
    submit_url: str,
    default_render_dpi: int = 200,
) -> SourcePartEditorDocument:
    """Build one self-contained editor over the real rendered contact sheet."""

    if not sheet.pages:
        raise SourcePartEditorError(
            f"{sheet.parent_name} rendered no pages, so there is nothing to choose."
        )
    script = _script(submit_url)
    # Serialized before the document, and with `<` escaped: a JSON island
    # inside `<script>` ends at the first `</script>` the bytes contain, and a
    # filename is the owner's, not janki's.
    state_json = json.dumps(
        {
            "recipe_id": recipe_id,
            "parent_name": sheet.parent_name,
            "parent_sha256": sheet.parent_sha256,
            "pages": [
                {"page_index": page.page_index, "page_rotate": page.page_rotate}
                for page in sheet.pages
            ],
        }
    ).replace("<", "\\u003c")
    pages = "\n".join(
        f"""<section class="page">
<h2>Page {page.page_index + 1} <span class="note">({page.page_size_pt[0]}×{page.page_size_pt[1]}pt,
/Rotate {page.page_rotate}, shown at {sheet.render_dpi} DPI)</span></h2>
<div class="frame" id="janki-page-{page.page_index}">
<img src="data:image/png;base64,{page.png_base64}" alt="Page {page.page_index + 1}"
 width="{page.width}" height="{page.height}">
<div class="overlay"></div>
</div>
<p><button type="button" class="add-regions">Add selected regions as one part</button>
<button type="button" class="add-page">Add the whole page as one part</button>
<button type="button" class="clear-page">Clear this page's selection</button></p>
</section>"""
        for page in sheet.pages
    )
    document = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Source parts — {html.escape(sheet.parent_name)}</title>
<style>{_STYLE}</style>
</head><body>
<h1>Choose the parts of {html.escape(sheet.parent_name)}</h1>
<p class="note">Drag across a page to select a band; add the bands you chose as
one part, or add the whole page. Janki renders exactly the pixels you choose —
it does not look for tables, rows or columns, and it never edits this source.
Publishing writes new files into your corpus. It makes no model call and costs
nothing.</p>
<p class="note">Renderer {html.escape(sheet.renderer)} {html.escape(sheet.renderer_version)},
encoder {html.escape(sheet.encoder)} {html.escape(sheet.encoder_version)};
parent sha256 {html.escape(sheet.parent_sha256)}.</p>
{pages}
<h2>Parts to publish</h2>
<p><label>Publication DPI <input type="number" id="janki-dpi" min="72" max="600"
 step="1" value="{int(default_render_dpi)}"></label></p>
<ul id="janki-parts"></ul>
<p><button type="button" id="janki-plan">Render these exact parts</button>
<button type="button" id="janki-publish" disabled>Publish them into my corpus</button></p>
<div id="janki-status" class="status"></div>
<div id="janki-plan-result"></div>
<h2>Recipe</h2>
<p class="note">The same geometry, as a file: save it and run
<code>janki source-parts --source {html.escape(sheet.parent_name)} --recipe FILE --publish</code>
to publish exactly these parts from the terminal.</p>
<textarea id="janki-recipe" readonly></textarea>
<script type="application/json" id="janki-state">{state_json}</script>
<script>{script}</script>
</body></html>
"""
    payload = document.encode("utf-8")
    if len(payload) > MAX_EDITOR_BYTES:
        limit = MAX_EDITOR_BYTES // (1024 * 1024)
        raise SourcePartEditorError(
            f"The contact sheet for {sheet.parent_name} is too large to open "
            f"({len(payload)} bytes; Janki serves up to {limit} MiB). Prepare "
            "parts from a smaller document, or use janki source-parts."
        )
    return SourcePartEditorDocument(
        html=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        content_security_policy=_policy(script),
        source_name=sheet.parent_name,
        recipe_id=recipe_id,
        page_count=len(sheet.pages),
    )


@dataclass(slots=True)
class LocalSourcePartEditorStore:
    """Hold a bounded number of open editors behind opaque tokens.

    ``plan_recipe`` and ``publish_plan`` are the adapter's own owner routes.
    They are held here rather than reached through the model's intent path
    because publishing is an owner action: no model-emittable field can carry
    a recipe, a coordinate or a plan fingerprint (contracts §9.1).
    """

    editor_prefix: str
    plan_recipe: Callable[[str, bytes], dict[str, Any]]
    publish_plan: Callable[[str, str], dict[str, Any]]
    _sessions: dict[str, SourcePartEditorSession] = field(
        default_factory=dict, init=False, repr=False
    )
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def open(
        self,
        sheet: Any,
        *,
        recipe_id: str,
        default_render_dpi: int = 200,
    ) -> SourcePartEditorOffer:
        """Mint one token, build the document it will be served as, keep both."""

        token = secrets.token_urlsafe(_TOKEN_BYTES)
        url = f"{self.editor_prefix}{token}"
        document = render_source_part_editor(
            sheet,
            recipe_id=recipe_id,
            submit_url=url,
            default_render_dpi=default_render_dpi,
        )
        session = SourcePartEditorSession(
            token=token,
            document=document,
            source_name=sheet.parent_name,
            parent_sha256=sheet.parent_sha256,
        )
        with self._lock:
            while len(self._sessions) >= MAX_EDITORS:
                self._sessions.pop(next(iter(self._sessions)))
            self._sessions[token] = session
        return SourcePartEditorOffer(
            token=token,
            url=url,
            source_name=session.source_name,
            recipe_id=recipe_id,
            page_count=document.page_count,
            byte_count=len(document.html),
            sha256=document.sha256,
        )

    def read(self, token: str) -> SourcePartEditorDocument:
        """Return the exact retained document, or refuse. Never consumes it."""

        return self._session(token).document

    def _session(self, token: str) -> SourcePartEditorSession:
        with self._lock:
            session = self._sessions.get(token)
        if session is None:
            raise SourcePartEditorError(EDITOR_UNAVAILABLE_MESSAGE)
        return session

    def act(self, token: str, request: Any) -> dict[str, Any]:
        """One owner control from the open editor: render a plan, or publish.

        The session decides which source this editor may name, so a recipe
        posted here can only ever describe the document the owner opened.
        """

        session = self._session(token)
        if not isinstance(request, dict):
            raise SourcePartEditorError("That editor request is not an object.")
        action = request.get("action")
        if action == "plan":
            recipe = request.get("recipe")
            if not isinstance(recipe, dict):
                raise SourcePartEditorError("That plan request carries no recipe.")
            payload = json.dumps(recipe, ensure_ascii=False).encode("utf-8")
            return self.plan_recipe(session.source_name, payload)
        if action == "publish":
            fingerprint = request.get("plan_fingerprint")
            if not isinstance(fingerprint, str) or not fingerprint:
                raise SourcePartEditorError(
                    "Publishing needs the exact plan fingerprint you reviewed."
                )
            return self.publish_plan(session.source_name, fingerprint)
        raise SourcePartEditorError(
            "A source-part editor either renders a plan or publishes one."
        )
