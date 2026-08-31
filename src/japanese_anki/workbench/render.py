"""The workbench, as server-rendered HTML plus one exact intake behavior.

There is no frontend build or remote asset. One CSP-hashed inline script shows
the browser's already-local selected file before the existing upload form may
save it; it has no network permission and no authority over the filename the
server derives. Everything else is ordinary server-rendered HTML. The CSS uses
system colours (`Canvas`, `CanvasText`, `Highlight`) so light mode, dark mode
and forced-colours mode all work without a theme switcher, which
`WORKBENCH_PLAN.md` makes a completion requirement rather than polish.

Every value that reaches this module is escaped. A source filename is
attacker-adjacent input — it arrives from whatever the person dragged in — and
it is rendered beside their own study material.
"""

from __future__ import annotations

import base64
import hashlib
import html
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from japanese_anki.application import (
    NOT_EXTRACTED,
    CardCheckReport,
    CoveragePreview,
    PromotionPlan,
    SourceJourney,
)
from japanese_anki.application.audio import AudioPlan
from japanese_anki.application.build import FinishBuildPlan, FinishDeckBuildPlan
from japanese_anki.application.enrichment import DictionaryEnrichmentDecision
from japanese_anki.application.finish import FinishScope
from japanese_anki.application.kanji_addition import KanjiAdditionPlan
from japanese_anki.credential_safety import redact_environment_credentials
from japanese_anki.enrich import format_field_diff

__all__ = [
    "FAILURE_STYLE",
    "FAILURE_STYLE_SOURCE",
    "STYLE",
    "FailureView",
    "INTAKE_SCRIPT",
    "INTAKE_SCRIPT_SOURCE",
    "render_addition",
    "render_consent",
    "render_card_check",
    "render_dashboard",
    "render_deck_creator",
    "render_extraction_failure",
    "render_extraction_progress_start",
    "render_extraction_progress_step",
    "render_extraction_success",
    "render_failure",
    "render_finish",
    "render_reidentify",
    "render_source",
]

INTAKE_SCRIPT = r'''(() => {
  "use strict";
  const form = document.querySelector("form.add-source");
  if (!form) return;
  const input = form.querySelector("#source-file");
  const preview = form.querySelector("#intake-preview");
  const permanentName = form.querySelector("#permanent-filename");
  const pages = form.querySelector("#intake-pages");
  const note = form.querySelector("#intake-preview-note");
  const save = form.querySelector("button[type=submit]");
  let previewUrl = null;

  const clearPreview = () => {
    if (previewUrl !== null) {
      URL.revokeObjectURL(previewUrl);
      previewUrl = null;
    }
    pages.replaceChildren();
    permanentName.textContent = "";
    note.textContent = "";
    preview.hidden = true;
    save.disabled = true;
  };

  const storedName = (raw) => {
    const parts = raw.replace(/\\/g, "/").split("/");
    return (parts[parts.length - 1] || "").trim().normalize("NFC");
  };

  const showPreview = (file) => {
    clearPreview();
    if (!file) return;
    const name = storedName(file.name);
    const lower = name.toLowerCase();
    const isPdf = lower.endsWith(".pdf");
    const isPhoto = [".jpg", ".jpeg", ".png", ".heic", ".heif"].some(
      (suffix) => lower.endsWith(suffix)
    );
    permanentName.textContent = name;
    preview.hidden = false;
    if (
      !name || name === "." || name === ".." || name.startsWith(".") ||
      name.includes("\u0000")
    ) {
      note.textContent = "Give this file an ordinary visible filename before adding it.";
      return;
    }
    if (!isPdf && !isPhoto) {
      note.textContent = "This file type cannot be previewed or saved by janki.";
      return;
    }

    previewUrl = URL.createObjectURL(file);
    if (isPdf) {
      const frame = document.createElement("iframe");
      const frameUrl = previewUrl;
      frame.setAttribute("sandbox", "");
      frame.setAttribute("referrerpolicy", "no-referrer");
      frame.title = `Page preview of ${name}`;
      frame.addEventListener("load", () => {
        if (previewUrl !== frameUrl) return;
        save.disabled = false;
      }, {once: true});
      frame.src = frameUrl;
      pages.append(frame);
      note.textContent = "Save only if the expected pages are visible and readable. " +
        "If no pages appear, convert or rescan this PDF first.";
    } else {
      const image = document.createElement("img");
      const imageUrl = previewUrl;
      image.alt = `Preview of ${name}`;
      image.addEventListener("load", () => {
        if (previewUrl !== imageUrl) return;
        save.disabled = false;
      }, {once: true});
      image.addEventListener("error", () => {
        if (previewUrl !== imageUrl) return;
        save.disabled = true;
        note.textContent = "This browser cannot preview this photo. " +
          "Convert it to JPEG or PNG before adding it.";
      }, {once: true});
      image.src = imageUrl;
      pages.append(image);
      note.textContent = "Check that this is the right photo before saving it.";
    }
  };

  input.addEventListener("change", () => showPreview(input.files[0]));
  window.addEventListener("pagehide", () => {
    clearPreview();
    input.value = "";
  });
})();'''

INTAKE_SCRIPT_SOURCE = "'sha256-" + base64.b64encode(
    hashlib.sha256(INTAKE_SCRIPT.encode("utf-8")).digest()
).decode("ascii") + "'"


FAILURE_STYLE = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body {
  margin: 0; padding: 1.5rem; overflow-wrap: anywhere;
  font: 1rem/1.6 system-ui, sans-serif;
  background: Canvas; color: CanvasText;
}
main { max-width: 60rem; min-inline-size: 0; margin: 0 auto; }
.failure-facts dt { font-weight: 700; }
.failure-facts dd { margin: 0 0 .75rem; }
summary { cursor: pointer; min-height: 44px; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; }
@media (max-width: 500px) { body { padding: .75rem; } }
:focus-visible { outline: 3px solid Highlight; outline-offset: 3px; }
""".strip()

FAILURE_STYLE_SOURCE = "'sha256-" + base64.b64encode(
    hashlib.sha256(FAILURE_STYLE.encode("utf-8")).digest()
).decode("ascii") + "'"


@dataclass(frozen=True, slots=True)
class FailureView:
    """The four learner-facing truths every failed action must carry."""

    happened: str
    changed: str
    money: str
    next_step: str
    technical_detail: str = ""

    def __post_init__(self) -> None:
        for name in ("happened", "changed", "money", "next_step"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"A failure needs a nonblank {name.replace('_', ' ')}")


STYLE = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body {
  margin: 0; padding: 1.5rem;
  overflow-wrap: anywhere;
  font: 1rem/1.6 system-ui, sans-serif;
  background: Canvas; color: CanvasText;
}
main { max-width: 60rem; min-inline-size: 0; margin: 0 auto; }
:lang(ja), .ja, .furigana {
  font-family: "Hiragino Sans", "Yu Gothic", Meiryo, "Noto Sans JP", system-ui, sans-serif;
}
h1 { font-size: 1.5rem; margin: 0 0 .25rem; }
.saved { color: GrayText; margin: 0 0 2rem; font-size: .9rem; }
.tour {
  border: 1px solid GrayText; border-radius: .5rem;
  margin-block-end: 1.5rem; overflow: hidden;
}
.tour > summary {
  padding: .75rem 1rem; font-size: 1rem; font-weight: 700; color: CanvasText;
}
.tour-body { border-block-start: 1px solid ButtonBorder; padding: 0 1rem 1rem; }
.tour-body h2 { font-size: 1.2rem; }
.tour-body h3 { font-size: 1rem; margin-block-end: .25rem; }
.tour-body ol { padding-inline-start: 1.5rem; }
.tour-body li { margin-block: .5rem; }
.providers { margin: 0; }
.providers > div { margin-block: .65rem; }
.providers dt { font-weight: 700; }
.providers dd { margin-inline-start: 0; }
.source {
  border: 1px solid GrayText; border-radius: .5rem;
  padding: 1rem; margin-block-end: 1rem;
}
.name {
  font-size: 1.1rem; font-weight: 700; margin: 0 0 .5rem;
  overflow-wrap: anywhere;
}
.state { font-weight: 700; }
.badges { display: flex; flex-wrap: wrap; gap: .5rem; margin: .5rem 0; }
.badge {
  border: 1px solid GrayText; border-radius: 999px;
  padding: .1rem .6rem; font-size: .85rem;
}
.badge.grammar { border-color: Highlight; }
.next { margin: .75rem 0 0; }
.next b { font-weight: 700; }
.counts { color: GrayText; font-size: .9rem; margin: .5rem 0 0; }
details { margin-block-start: .75rem; }
summary {
  cursor: pointer; min-height: 44px; display: flex; align-items: center;
  gap: .4rem; font-size: .9rem; color: GrayText;
  border-radius: .25rem;
}
summary:hover { color: CanvasText; text-decoration: underline; }
.decision:hover { border-color: Highlight; }
pre {
  white-space: pre-wrap; overflow-wrap: anywhere;
  background: Canvas; border: 1px solid GrayText;
  padding: .5rem; border-radius: .25rem; margin: .5rem 0 0;
}
.done { color: GrayText; }
.warnings { border: 1px solid red; border-radius: .5rem; padding: 1rem; }
.empty { color: GrayText; }
.add-source { border: 1px solid GrayText; border-radius: .5rem; padding: 1rem;
  margin-block-end: 2rem; display: grid; gap: .5rem; }
.add-source h2 { font-size: 1.1rem; margin: 0; }
input[type=file] { font: inherit; padding: .4rem; }
.intake-preview { border-block-start: 1px solid ButtonBorder; padding-block-start: .75rem; }
.intake-preview[hidden] { display: none; }
.intake-preview h3 { font-size: 1rem; margin: 0 0 .5rem; }
.intake-pages iframe, .intake-pages img {
  display: block; inline-size: 100%; border: 1px solid GrayText;
  border-radius: .35rem; background: Canvas;
}
.intake-pages iframe { block-size: min(70vh, 48rem); }
.intake-pages img { max-block-size: 48rem; object-fit: contain; }
.actions {
  display: flex; flex-wrap: wrap; gap: 1rem;
  margin-block-start: 1rem; padding-block-start: .75rem;
  border-block-start: 1px solid ButtonBorder;
}
.actions > div, .actions > form {
  flex: 1 1 16rem; display: grid; gap: .35rem; align-content: start;
}
/* Content width, not column width: a button stretched across the card reads
   as a disabled input, which is how the removal control came to look inert. */
.actions button { justify-self: start; text-align: start; }
.actions .counts { margin: 0; }
.assignment { border-block-start: 1px solid ButtonBorder; margin-block-start: 1rem;
  padding-block-start: .75rem; }
.assignment-choice { border: 1px solid GrayText; border-radius: .35rem;
  padding: .75rem; margin-block-start: .75rem; }
.assignment-choice h5 { font-size: 1rem; margin: 0 0 .5rem; }
.remove button { border-color: GrayText; background: Canvas; color: CanvasText; }
.edit { display: grid; gap: .25rem; margin: .5rem 0 .75rem; }
.edit label { font-size: .85rem; font-weight: 700; color: GrayText; }
select { font: inherit; padding: .4rem; background: Canvas; color: CanvasText;
  border: 1px solid GrayText; border-radius: .25rem; min-height: 44px; }
textarea { font: inherit; width: 100%; padding: .4rem; resize: vertical;
  background: Canvas; color: CanvasText; border: 1px solid GrayText; border-radius: .25rem; }
textarea[lang=ja] { font-size: 1.15rem; line-height: 1.9; }
.decision { display: flex; gap: .75rem; align-items: flex-start; cursor: pointer;
  padding: .75rem; border: 1px solid GrayText; border-radius: .35rem; margin-top: .75rem; }
input[type=checkbox] { inline-size: 1.4rem; block-size: 1.4rem; flex: none; }
button {
  max-inline-size: 100%; white-space: normal;
  min-height: 44px; padding: .65rem 1rem; font: inherit; font-weight: 700;
  cursor: pointer; border-radius: .35rem;
  background: ButtonFace; color: ButtonText; border: 1px solid ButtonBorder;
  transition: background-color .12s ease, color .12s ease, border-color .12s ease;
}
/* A control that looks the same whether or not the pointer is on it reads as
   decoration — the removal button was reported as inert on exactly that
   basis. Hover inverts the surface, so "this will do something" needs no
   explaining, and the pressed state moves so a click that has landed is
   distinguishable from one that has not. */
button:hover { background: Highlight; color: HighlightText; border-color: Highlight; }
button:active { transform: translateY(1px); }
button:disabled {
  cursor: not-allowed; color: GrayText; border-color: GrayText;
  background: ButtonFace; transform: none;
}
button:disabled:hover { background: ButtonFace; color: GrayText; }
/* A link that leads to a decision is a control, so it carries the same
   affordances as one — including the hover state, without which it reads as
   decoration exactly like the removal button did. `inline-block` because an
   inline anchor ignores the padding that makes it a 44px target. */
a.button {
  display: inline-block; text-decoration: none;
  min-height: 44px; padding: .65rem 1rem; font-weight: 700;
  border-radius: .35rem;
  background: ButtonFace; color: ButtonText; border: 1px solid ButtonBorder;
  transition: background-color .12s ease, color .12s ease, border-color .12s ease;
}
a.button:hover, a.button:focus-visible {
  background: Highlight; color: HighlightText; border-color: Highlight;
}
a.button:active { transform: translateY(1px); }
@media (prefers-reduced-motion: reduce) {
  button, a.button { transition: none; }
  button:active, a.button:active { transform: none; }
}
/* The consent disclosures. Wide line spacing and no bullet crowding: every
   item is a separate thing somebody has to actually read before agreeing,
   and a dense list is one people skip. */
.lead { font-size: 1.1rem; }
/* The irreversible ones. Without this they render in the same neutral grey as
   every other status box, which is the wrong weight for "not recoverable". */
.status.held { border-inline-start: .35rem solid Highlight; padding-inline-start: .6rem; }
.disclosure { margin-block: 1rem; padding-inline-start: 1.25rem; }
.disclosure li { margin-block: .5rem; }
.advanced { border: 1px solid GrayText; border-radius: .35rem; padding: .75rem; }
.implicit-submit-guard { position: absolute; inline-size: 1px; block-size: 1px;
  padding: 0; margin: -1px; overflow: hidden; clip-path: inset(50%);
  white-space: nowrap; border: 0; }
.progress { padding-inline-start: 1.5rem; }
.progress li { margin-block: .65rem; }
.submit { position: sticky; bottom: 0; padding: 1rem; background: Canvas;
  border-top: 1px solid GrayText; }
.card, .pattern {
  border: 1px solid GrayText; border-radius: .5rem;
  padding: 1rem; margin-block-end: 1rem;
}
.card h3 {
  font-size: 1.35rem; margin: 0 0 .5rem;
  padding-block-end: .5rem; border-block-end: 1px solid ButtonBorder;
}
.card h3 small { font-size: .85em; font-weight: 400; color: GrayText; }
ruby { ruby-position: over; }
rt { font-size: .55em; font-weight: 400; color: GrayText; }
.card h4 { font-size: .95rem; margin: 1rem 0 .25rem; }
/* Japanese wants room: generous size and leading so small kana and
   lookalike kanji stay apart (WORKBENCH_PLAN.md, "The words on the screen"). */
.ja { font-size: 1.25rem; line-height: 1.9; margin: .25rem 0; }
.furigana { font-size: 1.05rem; line-height: 2; color: GrayText; margin: .15rem 0; }
.en { margin: .15rem 0; }
.example { border-left: 3px solid GrayText; padding-left: .75rem; margin: .75rem 0; }
.panels { display: grid; gap: .75rem; }
.panels > div { border: 1px dashed GrayText; border-radius: .35rem; padding: .5rem; }
.status { padding: .5rem; border: 1px solid GrayText; border-radius: .35rem; }
.status.reviewed { border-color: green; }
.status.problem { border-color: red; }
.field { margin-block-end: .35rem; }
.fields { margin: 0; }
a { color: LinkText; }
@media (min-width: 48rem) {
  .panels { grid-template-columns: repeat(3, minmax(0, 1fr)); }
}
@media (max-width: 31.25rem) {
  body { padding: .75rem; }
  .source, .card, .pattern, .add-source { padding: .75rem; }
  .actions { gap: .75rem; }
}
:focus-visible { outline: 3px solid Highlight; outline-offset: 3px; }
@media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto !important; } }
""".strip()


def _escaped(value: object) -> str:
    return html.escape(redact_environment_credentials(value))


def _failure_facts(failure: FailureView) -> str:
    rows = (
        ("What happened", failure.happened),
        ("What changed", failure.changed),
        ("Money", failure.money),
        ("What to do next", failure.next_step),
    )
    body = ['<dl class="failure-facts">']
    for label, value in rows:
        body.append(
            f"<div class=field><dt>{_escaped(label)}</dt>"
            f"<dd>{_escaped(value)}</dd></div>"
        )
    body.append("</dl>")
    if failure.technical_detail:
        body.append(
            "<details><summary>Technical details</summary>"
            f"<pre>{_escaped(failure.technical_detail)}</pre></details>"
        )
    return "".join(body)


def render_failure(failure: FailureView, *, title: str = "Workbench error") -> str:
    """A secret-free standalone refusal; it deliberately has no session link."""
    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_escaped(title)}</title><style>{FAILURE_STYLE}</style>"
        "</head><body><main>"
        f"<h1>{_escaped(title)}</h1><section role=alert>"
        f"{_failure_facts(failure)}</section></main></body></html>"
    )


def _source_link(journey: SourceJourney, prefix: str) -> str:
    """Only sources with a live staging file have a page to open."""
    name = _escaped(journey.source)
    if journey.staging_path is None or not prefix:
        return name
    target = f"{prefix}/source/{quote(journey.source, safe='')}"
    return f'<a href="{html.escape(target, quote=True)}">{name}</a>'


def _counts(journey: SourceJourney) -> str:
    if not journey.card_count:
        return ""
    parts = [f"{journey.card_count} cards"]
    if journey.held_count:
        parts.append(f"{journey.held_count} held for a reading decision")
    if journey.example_review_count:
        parts.append(f"{journey.example_review_count} awaiting example review")
    if journey.invalid_count:
        parts.append(f"{journey.invalid_count} needing edits")
    return f'<p class=counts>{_escaped(", ".join(parts))}</p>'


def _source_html(journey: SourceJourney, prefix: str = "") -> str:
    badges = []
    if journey.grammar:
        badges.append(f'<span class="badge grammar">{_escaped(journey.grammar)}</span>')
    parts = [
        '<article class="source">',
        f'<h2 class=name>{_source_link(journey, prefix)}</h2>',
        f'<p class=state>{_escaped(journey.state)}</p>',
    ]
    if badges:
        parts.append(f'<p class=badges>{"".join(badges)}</p>')
    parts.append(_counts(journey))
    parts.append(
        f'<p class=next><b>Next:</b> {_escaped(journey.next_action)}</p>'
    )
    if prefix and journey.finish_receipt_ids:
        count = len(journey.finish_receipt_ids)
        for index, receipt_id in enumerate(journey.finish_receipt_ids, start=1):
            target = f"{prefix}/finish/{receipt_id}"
            suffix = f" — batch {index} of {count}" if count > 1 else ""
            parts.append(
                f'<p class=next><a class=button href="'
                f'{html.escape(target, quote=True)}">Finish dictionary, audio '
                f"and deck build for {_escaped(journey.source)}{suffix}</a></p>"
            )
    if prefix and journey.state == NOT_EXTRACTED:
        # Goes to the page that *describes* the paid call, never straight at
        # one. Adding a file and sending it are two separate actions, and a
        # link that spent money would collapse them into one click.
        target = f"{prefix}/extract/{quote(journey.source, safe='')}"
        parts.append(
            f'<p class=next><a class=button href="{html.escape(target, quote=True)}">'
            "See what reading this would send</a></p>"
        )
    detail = " ".join(filter(None, (journey.detail, journey.grammar_detail)))
    if detail:
        parts.append(
            "<details><summary>Source and technical details</summary>"
            f"<pre>{_escaped(detail)}</pre></details>"
        )
    parts.append("</article>")
    return "".join(parts)


def _dashboard_tour(*, open_by_default: bool) -> str:
    """A native, dismissible guide that needs no browser-side state."""
    opened = " open" if open_by_default else ""
    return (
        f'<details class="tour"{opened}>'
        "<summary>Quick start and provider setup</summary>"
        '<div class="tour-body">'
        "<h2>From lesson to deck</h2>"
        "<ol>"
        "<li><b>Add a PDF or photo.</b> Saving a source keeps a permanent "
        "local copy. It sends nothing.</li>"
        "<li><b>Ask the model to read it.</b> You see and confirm the exact "
        "paid send first. The proposal keeps the meaning in this lesson, one "
        "polite and one casual example, and any grammar the lesson teaches.</li>"
        "<li><b>Make the human decisions.</b> Review the Japanese examples. A "
        "reading hold pauses a card whose spelling-and-reading identity needs "
        "your choice. Grammar review is separate. Coverage asks only whether "
        "the promised source units are accounted for, not whether the Japanese "
        "is good. Each card belongs to one study deck, and you choose which "
        "one.</li>"
        "<li><b>Finish and build.</b> Add dictionary facts, kanji reference and "
        "audio, preview the cards, then build the Anki package.</li>"
        "</ol>"
        "<h3>What can contact another service</h3>"
        '<dl class="providers">'
        "<div><dt>Anthropic · paid network model</dt><dd>Claude reads PDFs and "
        "photos after the separate consent page. Anthropic is also the default "
        "provider for bare-record <code>janki enrich --ai</code>. It uses "
        "<code>ANTHROPIC_API_KEY</code>.</dd></div>"
        "<div><dt>jpdb · networked dictionary</dt><dd>It supplies word facts "
        "and can witness a reading. It is not a model call and uses "
        "<code>JPDB_API_KEY</code>.</dd></div>"
        "<div><dt>KANJIDIC/KanjiVG · networked reference sources</dt><dd>They "
        "supply kanji facts and stroke diagrams. They need no account or API "
        "key.</dd></div>"
        "<div><dt>VOICEVOX · local network engine</dt><dd>It creates word audio "
        "and also reads example sentences unless OpenAI Realtime is selected. "
        "It needs the local engine running, but no account or paid API call.</dd></div>"
        "<div><dt>OpenAI Realtime · optional paid network audio</dt><dd>It "
        "reads example sentences with a stable per-record voice and uses "
        "<code>OPENAI_API_KEY</code>. The button names the model before a paid "
        "call.</dd></div>"
        "<div><dt>Codex · optional network enrichment through its CLI</dt><dd>It is "
        "the alternative bare-record <code>janki enrich --ai</code> provider; set "
        "<code>enrich_provider</code> to <code>codex</code> to select it. It "
        "launches the separately installed, authenticated Codex CLI; that "
        "CLI's login controls access and any billing.</dd></div>"
        "</dl>"
        '<p class="status held"><b>Keys stay in your shell environment.</b> '
        "janki never puts them in the repository, browser storage, URLs, logs, "
        "or error messages. If a required key is missing, janki stops before "
        "contacting that provider.</p>"
        "</div></details>"
    )


def render_dashboard(
    journeys: Sequence[SourceJourney],
    *,
    warnings: Sequence[str] = (),
    recovery: Sequence[str] = (),
    tour_open: bool = False,
    root: Path | None = None,
    token: str = "",
    csrf: str = "",
    added: tuple[str, bool] | None = None,
    promoted: tuple[str, int, int] | None = None,
) -> str:
    """The whole page. `token` prefixes every same-session link."""
    prefix = f"/{html.escape(token, quote=True)}" if token else ""
    waiting = sum(1 for journey in journeys if journey.needs_a_person)
    if journeys:
        heading = (
            f"{len(journeys)} sources, {waiting} waiting on you"
            if waiting
            else f"{len(journeys)} sources, nothing waiting on you"
        )
    else:
        heading = "No sources yet"

    body = [
        "<!doctype html><html lang=en><head><meta charset=utf-8>",
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>janki workbench</title>",
        f'<link rel=stylesheet href="{prefix}/style.css">',
        "</head><body><main>",
        "<h1>Your Japanese sources</h1>",
        # Never "Backed up": these files live on this computer and nowhere
        # else until a person copies or commits them (WORKBENCH_PLAN.md W1.2).
        f"<p class=saved>Saved on this computer{_saved_where(root)}. "
        f"{_escaped(heading)}.</p>",
    ]
    body.append(_dashboard_tour(open_by_default=tour_open))
    if added is not None:
        name, stored = added
        body.append(
            '<p class="status reviewed">'
            + (
                f"Added {_escaped(name)} to your corpus. Nothing has been "
                "sent to a model."
                if stored
                else f"{_escaped(name)} was already in your corpus, so "
                "nothing changed."
            )
            + "</p>"
        )
    if promoted is not None:
        source, landed, held = promoted
        body.append(
            '<p class="status reviewed">Finished adding '
            f"{_escaped(source)}: {_escaped(landed)} card(s) reached your "
            f"collection and {_escaped(held)} remain(s) in review.</p>"
        )
    if csrf:
        body.append(_add_source_form(prefix, csrf))
        body.append(
            f'<p><a class=button href="{prefix}/decks/new">'
            "Create a study deck</a></p>"
        )
    if warnings:
        body.append('<section class="warnings"><h2>Could not be read</h2><ul>')
        body.extend(f"<li>{_escaped(warning)}</li>" for warning in warnings)
        body.append("</ul></section>")
    if recovery:
        body.append(
            '<section class="warnings" role=alert><h2>Recovery and operation notices'
            "</h2><p>The repository currently records provider work that may "
            "need recovery, a decision, or cleanup. Some notices are about "
            "cleanup and do not block another call. Follow the command named "
            "for each state; never retry a request whose outcome is uncertain.</p>"
            f"<pre>{_escaped(chr(10).join(recovery))}</pre></section>"
        )
    if not journeys:
        where = "above" if csrf else "to your inbox folder"
        body.append(
            f'<p class=empty>Add a PDF or photo {where}. After you save it, '
            "the workbench will show the separate reading step.</p>"
        )
    body.extend(_source_html(journey, prefix) for journey in journeys)
    body.append("</main></body></html>")
    return "".join(body)


def render_finish(
    scope: FinishScope,
    kanji_plan: KanjiAdditionPlan,
    *,
    token: str,
    csrf: str,
    dictionary_decision: DictionaryEnrichmentDecision | None = None,
    dictionary_action: str = "",
    banner: str = "",
    word_audio_plan: AudioPlan | None = None,
    word_audio_error: str = "",
    example_audio_plan: AudioPlan | None = None,
    example_audio_error: str = "",
    build_plan: FinishBuildPlan | None = None,
    build_error: str = "",
    preview_stem: str = "",
) -> str:
    """The exact post-promotion steps for one durable receipt."""
    prefix = f"/{html.escape(token, quote=True)}"
    action = f"{prefix}/finish/{scope.receipt_id}"
    deck_count = len(scope.owner_groups)
    body = [
        "<!doctype html><html lang=en><head><meta charset=utf-8>",
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>Finish {_escaped(scope.source_file)} · janki workbench</title>",
        f'<link rel=stylesheet href="{prefix}/style.css">',
        "</head><body><main>",
        f'<p><a href="{prefix}/">&larr; All sources</a></p>',
        f"<h1>Finish {_escaped(scope.source_file)}</h1>",
        f'<p class=saved>{len(scope.record_ids)} promoted card record(s) in '
        f"{deck_count} study deck(s). Dictionary facts, kanji, audio and the "
        "preview stay limited to those exact cards; the final package contains "
        "each complete current study deck.</p>",
    ]
    if banner:
        body.append(f'<p class="status reviewed">{_escaped(banner)}</p>')
    body.append('<section class=source><h2>Study decks</h2><ul>')
    body.extend(
        f"<li>{_escaped(group.stem)} — {len(group.record_ids)} card(s) in this "
        "finish batch</li>"
        for group in scope.owner_groups
    )
    body.append("</ul></section>")

    body.extend(
        [
            '<section class=source><h2>1. Add dictionary facts</h2>',
            "<p><b>Networked · jpdb API · no model call.</b> This can fill "
            "word facts such as part of speech, furigana, pitch accent and "
            "frequency. It never replaces the lesson meaning.</p>",
        ]
    )
    if dictionary_decision is None:
        body.extend(
            [
                f'<form method=post action="{html.escape(action, quote=True)}">',
                _hidden("action", "dictionary-plan"),
                _hidden("csrf", csrf),
                _hidden("scope_fingerprint", scope.fingerprint),
                "<button type=submit>Check dictionary facts with jpdb</button>",
                "</form>",
            ]
        )
    else:
        result = dictionary_decision.result
        body.append(
            f"<p>jpdb looked up {_escaped(result.looked_up)} card(s) and skipped "
            f"{_escaped(result.skipped)} with no empty dictionary fields.</p>"
        )
        if result.warnings:
            body.append("<details><summary>Dictionary notes</summary><ul>")
            body.extend(f"<li>{_escaped(warning)}</li>" for warning in result.warnings)
            body.append("</ul></details>")
        if result.changes or result.cleared:
            lines = format_field_diff(result.changes)
            if lines:
                body.append(
                    "<p>Review the exact proposed changes:</p>"
                    f"<pre>{_escaped(chr(10).join(lines))}</pre>"
                )
            if result.cleared:
                marks = sum(len(names) for names in result.cleared.values())
                body.append(
                    f"<p>This also clears {_escaped(marks)} stale or confirmed "
                    "provisional field mark(s).</p>"
                )
            if dictionary_action:
                body.extend(
                    [
                        f'<form method=post action="{html.escape(action, quote=True)}">',
                        _hidden("action", "dictionary-commit"),
                        _hidden("csrf", csrf),
                        _hidden("scope_fingerprint", scope.fingerprint),
                        _hidden("dictionary_action", dictionary_action),
                        _hidden("plan_fingerprint", dictionary_decision.fingerprint),
                        "<button type=submit>Save these dictionary facts</button>",
                        "</form>",
                    ]
                )
        else:
            body.append(
                '<p class="status reviewed">Nothing new to fill for these cards.</p>'
            )
    body.append("</section>")

    body.extend(
        [
            '<section class=source><h2>2. Add kanji reference</h2>',
            "<p><b>Networked · KANJIDIC/KanjiVG sources · no model call.</b> "
            "This adds meanings, readings, stroke count and available stroke "
            "diagrams for the characters on these cards.</p>",
        ]
    )
    if kanji_plan.characters:
        body.append(
            f'<p class=ja lang=ja>{_escaped(" ".join(kanji_plan.characters))}</p>'
        )
    if kanji_plan.to_fetch:
        body.extend(
            [
                f"<p>{len(kanji_plan.to_fetch)} of {len(kanji_plan.characters)} "
                "character(s) still need a lookup.</p>",
                f'<form method=post action="{html.escape(action, quote=True)}">',
                _hidden("action", "kanji-add"),
                _hidden("csrf", csrf),
                _hidden("scope_fingerprint", scope.fingerprint),
                _hidden("plan_fingerprint", kanji_plan.fingerprint),
                "<button type=submit>Add the missing kanji reference</button>",
                "</form>",
            ]
        )
    else:
        body.append(
            '<p class="status reviewed">The kanji reference is current for '
            "these cards.</p>"
        )
    body.append("</section>")

    def audio_step(
        number: int,
        title: str,
        kind: str,
        plan: AudioPlan | None,
        error: str,
    ) -> None:
        body.append(f'<section class=source><h2>{number}. {_escaped(title)}</h2>')
        if error:
            body.append(
                '<p class="status problem">This audio step cannot be planned: '
                f"{_escaped(error)}</p></section>"
            )
            return
        if plan is None:
            body.append(
                '<p class="status problem">This audio step is unavailable.</p>'
                "</section>"
            )
            return
        counts = plan.word_counts if kind == "words" else plan.example_counts
        provider = plan.word_provider if kind == "words" else plan.example_provider
        if provider is None or counts.total == 0:
            body.append(
                f'<p class="status reviewed">These cards need no {kind} clips.</p>'
                "</section>"
            )
            return
        access = (
            "Paid network provider"
            if provider.access == "paid-network"
            else "Local network provider"
        )
        body.append(
            f"<p><b>{_escaped(access)} · {_escaped(provider.name)}.</b> "
            f"{_escaped(counts.total)} distinct clip(s): "
            f"{_escaped(counts.current)} already current, "
            f"{_escaped(counts.recoverable)} recoverable from an interrupted "
            f"exact request, and {_escaped(counts.provider_required)} require "
            "the provider.</p>"
        )
        if counts.current == counts.total:
            body.append(
                f'<p class="status reviewed">All {kind} audio is current for '
                "this finish batch.</p></section>"
            )
            return
        action_name = "audio-words" if kind == "words" else "audio-examples"
        label = (
            f"Recover {counts.recoverable} {kind} clip(s)"
            if counts.provider_required == 0
            else f"Create {kind} audio"
        )
        if (
            counts.provider_required
            and kind == "examples"
            and provider.access == "paid-network"
        ):
            provider_name = {
                "openai-realtime": "OpenAI",
                "voicevox": "VOICEVOX",
            }.get(provider.name.lower(), provider.name)
            model = provider.settings.get("model", "").strip()
            provider_and_model = " ".join(filter(None, (provider_name, model)))
            label = (
                f"Create example sentence audio for {scope.source_file} with "
                f"{provider_and_model} — paid network call"
            )
        body.extend(
            [
                f'<form method=post action="{html.escape(action, quote=True)}">',
                _hidden("action", action_name),
                _hidden("csrf", csrf),
                _hidden("scope_fingerprint", scope.fingerprint),
                _hidden("plan_fingerprint", plan.fingerprint),
                f"<button type=submit>{_escaped(label)}</button>",
                "</form>",
                "<p class=counts>An exact interrupted request is recovered from "
                "its saved bytes and is not sent or billed again.</p>",
                "</section>",
            ]
        )

    audio_step(3, "Create word audio", "words", word_audio_plan, word_audio_error)
    audio_step(
        4,
        "Create example audio",
        "examples",
        example_audio_plan,
        example_audio_error,
    )

    body.append('<section class=source><h2>5. Preview the cards</h2>')
    if build_error:
        body.append(
            '<p class="status problem">The deck preview cannot be planned: '
            f"{_escaped(build_error)}</p></section>"
        )
    elif build_plan is None:
        body.append(
            '<p class="status problem">The deck preview is unavailable.</p>'
            "</section>"
        )
    else:
        body.append(
            "<p>This preview is limited to the exact cards from this finish "
            "batch and uses each deck's real selection and inline edits.</p><ul>"
        )
        for deck in build_plan.decks:
            preview_url = (
                f"{action}?preview={quote(deck.stem, safe='')}"
            )
            directions = ", ".join(deck.card_types)
            body.append(
                f'<li><a href="{html.escape(preview_url, quote=True)}">Preview '
                f"{_escaped(deck.name or deck.stem)}</a> — "
                f"{_escaped(len(deck.receipt_record_ids))} record(s) from this finish; "
                f"cards: {_escaped(directions)}</li>"
            )
        body.append("</ul>")
        selected: FinishDeckBuildPlan | None = next(
            (deck for deck in build_plan.decks if deck.stem == preview_stem),
            None,
        )
        if selected is not None:
            body.append(
                f"<h3>{_escaped(selected.name or selected.stem)} — exact finish "
                "preview</h3>"
            )
            for record in selected.preview_records:
                body.append('<article class=card>')
                body.append(
                    f'<h3><span lang=ja>{_escaped(record.expression)}</span> '
                    f"<small>{_escaped(record.reading)}</small></h3>"
                )
                body.append(
                    "<p>" + " · ".join(_escaped(item) for item in record.meanings) + "</p>"
                )
                for example in record.examples:
                    register = example.register.strip() or "example"
                    body.append(
                        '<div class=example>'
                        f"<h4>{_escaped(register.title())}</h4>"
                        f'<p class=ja lang=ja>{_escaped(example.japanese)}</p>'
                        f'<p class=en>{_escaped(example.english)}</p>'
                        "</div>"
                    )
                if record.usage_notes:
                    body.append(
                        f"<h4>Usage</h4><p>{_escaped(record.usage_notes)}</p>"
                    )
                body.append("</article>")
        body.append("</section>")

    body.append('<section class=source><h2>6. Build the Anki decks</h2>')
    if build_error:
        body.append(
            '<p class="status problem">The build cannot start: '
            f"{_escaped(build_error)}</p>"
        )
    elif build_plan is not None:
        body.append(
            "<p><b>Local.</b> Each receipted owner is built as its complete "
            "current study deck; unrelated decks are not built.</p><ul>"
        )

        def audio_coverage(
            plan: AudioPlan | None,
            record_ids: tuple[str, ...],
            kind: str,
        ) -> str:
            if plan is None:
                return "unavailable"
            selected = [
                clip
                for clip in plan.clips
                if clip.record_id in set(record_ids) and clip.kind == kind
            ]
            current = sum(clip.state == "current" for clip in selected)
            return f"{current}/{len(selected)} current"

        for deck in build_plan.decks:
            directions = ", ".join(deck.card_types)
            word_coverage = audio_coverage(
                word_audio_plan,
                deck.receipt_record_ids,
                "word",
            )
            example_coverage = audio_coverage(
                example_audio_plan,
                deck.receipt_record_ids,
                "example",
            )
            body.append(
                f"<li>{_escaped(deck.name or deck.stem)} — "
                f"{_escaped(len(deck.receipt_record_ids))} record(s) in this "
                f"batch, {_escaped(len(deck.records))} note(s) in the full deck; "
                f"cards: {_escaped(directions)}; output "
                f"<code>{_escaped(deck.output_path)}</code>; this batch's audio: words "
                f"{_escaped(word_coverage)}, examples "
                f"{_escaped(example_coverage)}</li>"
            )
        body.extend(
            [
                "</ul>",
                f'<form method=post action="{html.escape(action, quote=True)}">',
                _hidden("action", "build"),
                _hidden("csrf", csrf),
                _hidden("scope_fingerprint", scope.fingerprint),
                _hidden("plan_fingerprint", build_plan.fingerprint),
                "<button type=submit>Build these complete study decks</button>",
                "</form>",
                "<h3>Import into Anki</h3><ol>",
                "<li>Sync Anki before importing.</li>",
                "<li>In Anki, choose File → Import and select each output above.</li>",
                '<li>Tick <b>Merge Notetypes</b>, then import.</li>',
                "</ol>",
            ]
        )
    body.append("</section>")
    body.append("</main></body></html>")
    return "".join(body)


def _add_source_form(prefix: str, csrf: str) -> str:
    """Put a source in the corpus. This form sends nothing to any provider.

    Saying so on the control matters more than it looks: the whole intake
    design rests on adding and sending being two separate acts, and a person
    who believes an upload already cost money will hesitate over the wrong
    button for the rest of the workflow.
    """
    action = html.escape(f"{prefix}/add-source", quote=True)
    return (
        f'<form method=post action="{action}" enctype="multipart/form-data" '
        'class="add-source">'
        "<h2>Add source material</h2>"
        f'<input type=hidden name=csrf value="{html.escape(csrf, quote=True)}">'
        '<p class=edit><label for="source-file">Drag a PDF or photo here, or '
        "choose your study material</label>"
        '<input id="source-file" name="file" type=file '
        'accept=".pdf,.jpg,.jpeg,.png,.heic,.heif" required></p>'
        '<section id="intake-preview" class="intake-preview" hidden aria-live=polite>'
        "<h3>Check before saving</h3>"
        '<p>Proposed permanent filename — rechecked when saved: '
        '<strong id="permanent-filename"></strong></p>'
        '<div id="intake-pages" class="intake-pages"></div>'
        '<p id="intake-preview-note" class=counts></p>'
        "</section>"
        "<button type=submit disabled>Save permanent copy</button>"
        "<span class=counts>Nothing is copied until you confirm after the preview. "
        "Saving copies the file into your corpus and sends it nowhere. Reading it "
        "with a model is a separate, paid step you choose afterwards.</span>"
        "</form>"
        f"<script>{INTAKE_SCRIPT}</script>"
    )


def _saved_where(root: Path | None) -> str:
    return f" at {_escaped(root)}" if root is not None else ""


def render_deck_creator(
    *,
    token: str,
    csrf: str,
    plan: Any | None = None,
    plan_fingerprint: str = "",
) -> str:
    """Ask for learner choices, then preview the exact deck file they create."""
    prefix = f"/{html.escape(token, quote=True)}"
    action = html.escape(f"{prefix}/decks/new", quote=True)
    name = "" if plan is None else str(plan.name)
    recognition = plan is None or bool(plan.recognition)
    production = bool(plan is not None and plan.production)
    reading = bool(plan is not None and plan.reading)

    def checkbox(direction: str, label: str, checked: bool) -> str:
        mark = " checked" if checked else ""
        return (
            f'<label class=decision><input type=checkbox name="{direction}"{mark}>'
            f"<span>{_escaped(label)}</span></label>"
        )

    body = [
        "<!doctype html><html lang=en><head><meta charset=utf-8>",
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Create a study deck · janki workbench</title>",
        f'<link rel=stylesheet href="{prefix}/style.css">',
        "</head><body><main><h1>Create a study deck</h1>",
        '<p class=lead>Name the deck you want to see in Anki and choose its '
        "card directions. Nothing is created until the final button.</p>",
        f'<form method=post action="{action}" class=add-source>',
        '<input type=hidden name=action value=preview>',
        f'<input type=hidden name=csrf value="{html.escape(csrf, quote=True)}">',
        '<p class=edit><label for=deck-name>Name in Anki</label>',
        '<input id=deck-name name=name type=text required value="'
        f'{html.escape(name, quote=True)}"></p>',
        checkbox("recognition", "Recognition cards", recognition),
        checkbox("production", "Production cards", production),
        checkbox("reading", "Reading cards", reading),
        "<button type=submit>Preview study deck</button></form>",
    ]
    if plan is not None:
        enabled = "".join(
            f'<input type=hidden name={direction} value=on>'
            for direction, selected in (
                ("recognition", plan.recognition),
                ("production", plan.production),
                ("reading", plan.reading),
            )
            if selected
        )
        body.extend(
            [
                '<section class=source><h2>Study deck preview</h2>',
                f'<p><b>Name in Anki:</b> {_escaped(plan.name)}</p>',
                f'<p><b>Deck file:</b> {_escaped(plan.path)}</p>',
                f'<p><b>Build output:</b> {_escaped(plan.output_path)}</p>',
                '<details><summary>Deck and technical details</summary>',
                f'<pre>{_escaped(plan.yaml_bytes.decode("utf-8"))}</pre></details>',
                f'<form method=post action="{action}" class=submit>',
                '<input type=hidden name=action value=create>',
                f'<input type=hidden name=csrf value="{html.escape(csrf, quote=True)}">',
                f'<input type=hidden name=name value="{html.escape(plan.name, quote=True)}">',
                enabled,
                '<input type=hidden name=plan_fingerprint '
                f'value="{html.escape(plan_fingerprint, quote=True)}">',
                "<button type=submit>Create study deck</button></form></section>",
            ]
        )
    body.append(
        f'<p><a href="{prefix}/">Back to sources</a></p></main></body></html>'
    )
    return "".join(body)


# --- one source, opened ------------------------------------------------------


def _textarea(name: str, label: str, value: str, *, rows: int = 2,
              japanese: bool = False) -> str:
    lang = ' lang="ja"' if japanese else ""
    ident = html.escape(name, quote=True)
    return (
        f'<p class=edit><label for="{ident}">{_escaped(label)}</label>'
        f'<textarea id="{ident}" name="{ident}" rows={rows}{lang}>'
        f"{_escaped(value)}</textarea></p>"
    )


def _register_select(name: str, value: str) -> str:
    """Register is a choice, not free text.

    `ExampleSentence.is_incomplete` treats anything outside polite/casual as
    incomplete, so a typo in a text box would quietly degrade the card. Two
    options and a blank are the whole domain, so offer exactly those.
    """
    ident = html.escape(name, quote=True)
    current = value.strip().lower()
    choices = [("", "— not set —"), ("polite", "Polite"), ("casual", "Casual")]
    if current and current not in {"polite", "casual"}:
        # A value this select cannot represent still has to survive being
        # looked at. Without its own option nothing would be selected, the
        # browser would fall back to the first entry, and merely opening the
        # editor and saving would erase what someone wrote by hand.
        choices.append((current, f"{value.strip()} (kept as written)"))
    options = []
    for option, label in choices:
        selected = " selected" if option == current else ""
        options.append(
            f'<option value="{html.escape(option, quote=True)}"{selected}>'
            f"{_escaped(label)}</option>"
        )
    return (
        f'<p class=edit><label for="{ident}">Register</label>'
        f'<select id="{ident}" name="{ident}">{"".join(options)}</select></p>'
    )


def _example_editor(example: Any, card_index: int, example_index: int) -> str:
    """The five example fields a person may retype.

    Japanese is editable on purpose — correcting a mis-transcribed sentence is
    the point — and doing so voids that card's approval by fingerprint, so a
    tick can never end up covering text nobody read.
    """
    key = f"{card_index}_{example_index}"
    return "".join(
        [
            _textarea(f"ej{key}", "Japanese", example.japanese, japanese=True),
            _textarea(f"ef{key}", "Furigana", example.furigana, japanese=True),
            _textarea(f"ee{key}", "English", example.english),
            _textarea(f"er{key}", "Romaji", example.romaji, rows=1),
            _register_select(f"eg{key}", example.register),
        ]
    )


def _example_html(example: Any, index: int) -> str:
    register = getattr(example, "register", "") or ""
    label = {"polite": "Polite example", "casual": "Casual example"}.get(
        register, f"Example {index}"
    )
    parts = [
        '<article class="example">',
        f"<h4>{_escaped(label)}</h4>",
        # Japanese first and largest: it is the thing being learned. Furigana
        # sits with it, English beside it, romaji closed by default so an early
        # learner can reach it without reading it first.
        f'<p class="ja" lang="ja">{_escaped(example.japanese)}</p>',
    ]
    if example.furigana:
        parts.append(
            f'<p class="furigana" lang="ja">{_escaped(example.furigana)}</p>'
        )
    if example.english:
        parts.append(f'<p class="en">{_escaped(example.english)}</p>')
    if example.romaji:
        parts.append(
            "<details><summary>Romaji</summary>"
            f"<p>{_escaped(example.romaji)}</p></details>"
        )
    parts.append("</article>")
    return "".join(parts)


def _meaning_panels(card: Any) -> str:
    """The existing-wins merge, made visible instead of implied."""
    if card.existing is None:
        return (
            "<h4>Meaning in this lesson</h4>"
            f"<p>{_escaped(', '.join(card.record.meanings))}</p>"
            "<p class=counts>This word is new to your collection.</p>"
        )
    kept = ", ".join(card.merged_meanings) or "—"
    note = (
        ""
        if card.proposed_meanings_would_be_kept
        else "<p class=counts>Your card already has meanings, so adding this "
        "lesson keeps them. This lesson's wording stays in the source "
        "history.</p>"
    )
    return (
        '<div class="panels">'
        "<div><h4>Proposed by this lesson</h4>"
        f"<p>{_escaped(', '.join(card.record.meanings))}</p></div>"
        "<div><h4>Currently on your card</h4>"
        f"<p>{_escaped(', '.join(card.existing.meanings))}</p></div>"
        "<div><h4>What will remain after adding</h4>"
        f"<p>{_escaped(kept)}</p></div>"
        "</div>" + note
    )


def _card_html(
    card: Any,
    *,
    actionable: bool = False,
    editing: bool = False,
    index: int = 0,
    remove_action: str = "",
    reidentify_action: str = "",
    assignment_action: str = "",
) -> str:
    record = card.record
    parts = [
        f'<article class="card" id="card-{index + 1}">',
        # These are two explicit record fields, so attaching the full reading
        # to the full expression makes no segmentation guess about Japanese.
        f'<h3 lang="ja"><ruby>{_escaped(record.expression)}'
        f"<rt>{_escaped(record.reading)}</rt></ruby></h3>",
    ]
    if record.furigana:
        parts.append(f'<p class="furigana" lang="ja">{_escaped(record.furigana)}</p>')
    if card.hold_reason:
        parts.append(
            f'<p class="status problem">Held for a reading decision: '
            f"{_escaped(card.hold_reason)}</p>"
        )
    if editing:
        parts.append(
            _textarea(
                f"m{index}",
                "Meaning in this lesson (one per line)",
                "\n".join(card.record.meanings),
                rows=3,
            )
        )
    else:
        parts.append(_meaning_panels(card))
    if record.examples:
        parts.append("<h4>Japanese examples</h4>")
        if editing:
            for number, example in enumerate(record.examples):
                parts.append(
                    f'<article class="example"><h5>Example {number + 1}</h5>'
                    + _example_editor(example, index, number)
                    + "</article>"
                )
        else:
            parts.extend(
                _example_html(example, number)
                for number, example in enumerate(record.examples, start=1)
            )
    if editing:
        parts.append("<h4>How to use it</h4>")
        parts.append(
            _textarea(f"u{index}", "Usage note", record.usage_notes, rows=3)
        )
    elif record.usage_notes:
        parts.append(f"<h4>How to use it</h4><p>{_escaped(record.usage_notes)}</p>")
    if actionable and card.needs_example_review:
        parts.append(_checkbox_html(card))
    else:
        parts.append(_approval_html(card))
    parts.append(_evidence_html(card))
    if assignment_action:
        parts.append(assignment_action)
    if remove_action or reidentify_action:
        # One row, so the two actions read as siblings. Stacked, the
        # full-width removal button looked like a field rather than a
        # control, and the re-identify one had no rule at all and wrapped
        # into its own caption.
        parts.append(
            '<div class="actions">'
            + reidentify_action
            + remove_action
            + "</div>"
        )
    parts.append("</article>")
    return "".join(parts)


def _tag_diff(values: Sequence[str]) -> str:
    return ", ".join(values) if values else "none"


def _assignment_html(offer: Any) -> str:
    """One card's configured destinations and the exact planned tag diffs."""
    if not offer.choices:
        return (
            '<section class=assignment><h4>Choose a study deck</h4>'
            '<p class="status problem">No assignable word deck is configured.</p>'
            "</section>"
        )
    parts = [
        '<section class=assignment><h4>Choose a study deck</h4>',
        "<p class=counts>Each choice is checked against the configured deck "
        "selectors before it can change this proposal.</p>",
    ]
    proposals = offer.proposals
    if len(proposals) > 1:
        count = "two" if len(proposals) == 2 else str(len(proposals))
        retained = "both sources" if len(proposals) == 2 else "all of those sources"
        parts.append(
            f'<p class="status held">This word is proposed by {count} sources. '
            f"Choose which word deck should teach it; {retained} can still "
            "remain in its history.</p>"
        )
    for choice in offer.choices:
        plan = choice.plan
        parts.extend(
            [
                '<article class="assignment-choice">',
                f"<h5>{_escaped(choice.name)}</h5>",
            ]
        )
        if plan is None:
            parts.append(
                '<button type=button disabled>Cannot assign to this deck</button>'
                f'<p class="status problem">{_escaped(choice.refusal)}</p>'
            )
        else:
            parts.append(
                f'<button type=submit form="assign-{offer.card}" '
                f'name=destination value="{html.escape(choice.stem, quote=True)}">'
                f"Assign to {_escaped(choice.name)}</button>"
            )
            parts.append(
                "<details><summary>Technical details</summary><dl class=fields>"
                '<div class=field><dt>Tags added</dt>'
                f"<dd>{_escaped(_tag_diff(plan.tag_diff.added))}</dd></div>"
                '<div class=field><dt>Tags removed</dt>'
                f"<dd>{_escaped(_tag_diff(plan.tag_diff.removed))}</dd></div>"
                '<div class=field><dt>Tags after assignment</dt>'
                f"<dd>{_escaped(_tag_diff(plan.tag_diff.after))}</dd></div>"
                "</dl></details>"
            )
        parts.append("</article>")
    parts.append("</section>")
    return "".join(parts)


def _assignment_forms(
    offers: Sequence[Any], source_path: str, csrf: str, snapshot: str
) -> str:
    """Hidden owners for the form-associated destination buttons."""
    action = html.escape(source_path + "/assign", quote=True)
    forms = []
    for offer in offers:
        if not any(choice.plan is not None for choice in offer.choices):
            continue
        forms.append(
            f'<form id="assign-{offer.card}" method=post action="{action}" hidden>'
            '<input type=hidden name=action value="assign">'
            f'<input type=hidden name=csrf value="{html.escape(csrf, quote=True)}">'
            '<input type=hidden name=staging_snapshot '
            f'value="{html.escape(snapshot, quote=True)}">'
            f'<input type=hidden name=card value="{offer.card}">'
            '<input type=hidden name=plan_fingerprint '
            f'value="{html.escape(offer.fingerprint, quote=True)}">'
            "</form>"
        )
    return "".join(forms)


def _checkbox_html(card: Any) -> str:
    """The approval control, with its exact scope stated beside it.

    Not in help text, not in a tooltip: the sentence that says this covers the
    sentences and nothing else sits in the label the person is clicking.
    """
    identity = f"{card.record.expression} ({card.record.reading})"
    again = " again" if card.authority == "stale" else ""
    return (
        '<label class="decision">'
        f'<input type=checkbox name=record value="{html.escape(card.record.id, quote=True)}">'
        f"<span>Approve these Japanese example sentences{again} for "
        f'<b lang="ja">{_escaped(identity)}</b>. '
        "This approves the sentences above and nothing else — not the "
        "meanings, the English, the spelling and reading, or the usage note."
        "</span></label>"
    )


def _approval_html(card: Any) -> str:
    """What the approval covers — stated beside it, never only in help text."""
    identity = f"{card.record.expression} ({card.record.reading})"
    if card.authority == "existing":
        return (
            '<p class="status reviewed">Japanese examples approved for '
            f"{_escaped(identity)}.</p>"
        )
    if card.authority == "stale":
        return (
            '<p class="status problem">The approval on this card no longer '
            "covers every sentence shown. It needs approving again.</p>"
        )
    if card.authority == "invalid":
        return (
            '<p class="status problem">This card\'s approval value is not a '
            "recognized one. Correct it before adding.</p>"
        )
    if card.authority == "ineligible":
        return ""
    return (
        '<p class="status">Not yet approved. Approving covers only the exact '
        "Japanese sentences above — not the meanings, the English, the "
        "spelling and reading, or the usage note.</p>"
    )


def _evidence_html(card: Any) -> str:
    raw = card.record.source.raw_fields
    rows = [
        ("Stable ID", card.record.id, False),
        ("Source page", raw.get("page"), False),
        ("Source sentence", raw.get("context"), True),
        ("Why it was included", raw.get("inclusion_reason"), False),
        ("Confidence", raw.get("confidence"), False),
    ]
    shown_rows = []
    for label, value, japanese in rows:
        if value in (None, ""):
            continue
        lang = ' lang="ja"' if japanese else ""
        shown_rows.append(
            f"<div class=field><dt>{_escaped(label)}</dt>"
            f"<dd{lang}>{_escaped(value)}</dd></div>"
        )
    shown = "".join(shown_rows)
    return (
        "<details><summary>Source and technical details</summary>"
        f"<dl class=fields>{shown}</dl></details>"
    )


def _grammar_html(detail: Any, *, actionable: bool = False) -> str:
    pattern_set = detail.pattern_set
    if pattern_set is None or not pattern_set.patterns:
        return ""
    state = "Reviewed" if detail.pattern_reviewed else "Not yet reviewed"
    parts = [
        '<section class="grammar">',
        f"<h2>Grammar from this lesson</h2><p class=state>{_escaped(state)}</p>",
    ]
    # Only when the page actually carries a form. A checkbox on a page that
    # cannot submit is a control that silently does nothing.
    if actionable and not detail.pattern_reviewed and detail.can_review_grammar:
        parts.append(
            '<label class="decision">'
            '<input type=checkbox name=patterns value="review">'
            "<span>I read this lesson's grammar above. Marking it reviewed "
            "records that a person read the extracted set — it does not "
            "approve any word card.</span></label>"
        )
    for pattern in pattern_set.patterns:
        parts.append('<article class="pattern">')
        parts.append(f'<h3 lang="ja">{_escaped(pattern.template)}</h3>')
        if pattern.gloss:
            parts.append(f"<p>{_escaped(pattern.gloss)}</p>")
        for example in pattern.examples:
            parts.append(f'<p class="ja" lang="ja">{_escaped(example)}</p>')
        if pattern.where:
            parts.append(f"<p class=counts>{_escaped(pattern.where)}</p>")
        parts.append("</article>")
    parts.append("</section>")
    return "".join(parts)


def _remove_form(
    card: Any, index: int, source_path: str, csrf: str, snapshot: str
) -> str:
    """Removing a card is its own form, deliberately.

    It sits outside the edit form because it is a different act with a
    different consequence: corrections can be undone before saving, and this
    cannot be undone at all. The row is the model's proposal, and once it
    leaves a live staging file nothing else in the repository holds it — so
    the control says that rather than leaving someone to find out.
    """
    del source_path, csrf, snapshot
    identity = f"{card.record.expression} ({card.record.reading})"
    # `form=` associates this button with a form declared outside the edit
    # form. Forms cannot nest — a browser silently drops an inner one — so a
    # per-card <form> inside the editor would render a button that does
    # nothing at all.
    return (
        '<div class="remove">'
        f'<button type=submit form="rm{index}">'
        f"Remove {_escaped(identity)} from this review</button>"
        "<span class=counts>This deletes the proposal. It cannot be undone, and "
        "re-reading the source would be another paid call.</span>"
        "</div>"
    )


def _remove_forms(
    detail: Any, source_path: str, csrf: str, snapshot: str
) -> str:
    """The removal forms themselves, emitted after the editor's form closes."""
    action = html.escape(source_path + "/remove", quote=True)
    return "".join(
        f'<form id="rm{index}" method=post action="{action}" hidden>'
        '<input type=hidden name=action value="remove">'
        f'<input type=hidden name=csrf value="{html.escape(csrf, quote=True)}">'
        "<input type=hidden name=staging_snapshot "
        f'value="{html.escape(snapshot, quote=True)}">'
        f'<input type=hidden name=card value="{index}">'
        "</form>"
        for index in range(len(detail.cards))
    )


def render_card_check(report: CardCheckReport, *, token: str) -> str:
    """Render learner actions from one exact, read-only staged snapshot."""
    prefix = f"/{html.escape(token, quote=True)}"
    source_path = f"{prefix}/source/{quote(report.source, safe='')}"
    body = [
        "<!doctype html><html lang=en><head><meta charset=utf-8>",
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>Check { _escaped(report.source) } · janki workbench</title>",
        f'<link rel=stylesheet href="{prefix}/style.css">',
        "</head><body><main>",
        f'<p><a href="{html.escape(source_path, quote=True)}">&larr; Back to '
        "this source</a></p>",
        "<h1>Check these cards</h1>",
        '<p class=lead>These checks cover the cards\' structure, review marks, '
        "reading holds and study-deck membership. They do not judge the Japanese.</p>",
    ]
    if report.general:
        body.append('<section class=warnings><h2>Source-wide checks</h2><ul>')
        body.extend(f"<li>{_escaped(detail.reason)}</li>" for detail in report.general)
        body.append("</ul></section>")
    if not report.cards:
        body.append("<p class=empty>This source proposed no word cards.</p>")
    for card in report.cards:
        anchor = f"card-{card.index + 1}"
        identity = f"{card.record.expression} ({card.record.reading})"
        body.extend(
            [
                f'<article class=card id="{anchor}">',
                f'<h2 lang=ja>{_escaped(identity)}</h2>',
            ]
        )
        if not card.actions:
            body.append(
                '<p class="status reviewed">No structural action is waiting '
                "for this card.</p>"
            )
        for action in card.actions:
            edit = action.kind in {"edit", "identity"}
            target = source_path + ("?edit=1" if edit else "") + f"#{anchor}"
            body.extend(
                [
                    '<section class=assignment-choice>',
                    f'<p><a class=button href="{html.escape(target, quote=True)}">'
                    f"{_escaped(action.label)}</a></p>",
                    "<details><summary>Source and technical details</summary>",
                    '<dl class=fields>',
                ]
            )
            for detail in action.details:
                body.extend(
                    [
                        '<div class=field><dt>Code</dt>',
                        f"<dd><code>{_escaped(detail.code)}</code></dd></div>",
                        '<div class=field><dt>Why</dt>',
                        f"<dd>{_escaped(detail.reason)}</dd></div>",
                        '<div class=field><dt>Level</dt>',
                        f"<dd>{_escaped(detail.level)}</dd></div>",
                    ]
                )
            body.append("</dl></details></section>")
        body.append("</article>")
    body.append("</main></body></html>")
    return "".join(body)


def _hidden(name: str, value: object) -> str:
    return (
        f'<input type=hidden name="{html.escape(name, quote=True)}" '
        f'value="{html.escape(str(value), quote=True)}">'
    )


def _coverage_actions(
    coverage: CoveragePreview,
    *,
    model_coverage: CoveragePreview | None,
    source_path: str,
    csrf: str,
    coverage_action: str,
    source_fingerprint: str,
    busy: str,
    model_error: str,
) -> str:
    """The two authorities that can answer one unresolved coverage count."""
    action = html.escape(source_path + "/add", quote=True)
    parts = [
        '<section class=source><h2>Coverage needs a decision</h2>',
        "<p>Did the extraction account for the source units it promised to "
        "cover?</p>",
        f"<p>janki recorded {_escaped(coverage.source_units)} source unit(s), "
        f"including {_escaped(coverage.candidate_units)} card candidate(s).</p>",
        f'<pre>{_escaped(coverage.account)}</pre>',
        '<div class="actions">',
        f'<form method=post action="{action}">',
        "<h3>Compare it yourself</h3>",
        _hidden("action", "owner-coverage"),
        _hidden("csrf", csrf),
        _hidden("staging_fingerprint", coverage.staging_fingerprint),
        '<label class=decision><input type=checkbox name=compared '
        'value=confirmed required><span>I compared the named source with the '
        "coverage account above. This is only a completeness decision; it does "
        "not approve the Japanese.</span></label>",
        '<p class=edit><label for=coverage-reason>Why the account is complete</label>',
        '<textarea id=coverage-reason name=reason rows=3 required></textarea></p>',
        '<button type=submit>I compared the source rows myself</button>',
        "</form>",
        '<div><h3>Ask a model</h3>',
    ]
    if model_error:
        parts.append(f'<p class="status problem">{_escaped(model_error)}</p>')
    elif model_coverage is not None and coverage_action:
        model = model_coverage.model
        parts.extend(
            [
                f'<form method=post action="{action}">',
                _hidden("action", "model-coverage"),
                _hidden("csrf", csrf),
                _hidden("coverage_action", coverage_action),
                _hidden(
                    "staging_fingerprint", model_coverage.staging_fingerprint
                ),
                _hidden("source_file", model_coverage.source_file),
                _hidden("source_fingerprint", source_fingerprint),
            ]
        )
        parts.extend(
            [
                _hidden("model", model),
                _hidden(
                    "request_fingerprint", model_coverage.request_fingerprint
                ),
                _hidden("prompt_fingerprint", model_coverage.prompt_fingerprint),
                f"<p>This sends {_escaped(model_coverage.source_file)} and the "
                f"account above to {_escaped(model)}. It is a separate paid "
                "Anthropic API call. Claude Max is a different subscription. "
                "The verdict checks completeness, not the Japanese.</p>",
            ]
        )
        paid_label = (
            f"Send {_escaped(model_coverage.source_file)} to Anthropic using "
            f"{_escaped(model)} to check source-unit completeness — paid API call"
        )
        if busy:
            parts.extend(
                [
                    f"<button type=submit disabled>{paid_label}</button>",
                    f'<p class="status problem">{_escaped(busy)}</p>',
                ]
            )
        else:
            parts.append(
                f"<button type=submit>{paid_label}</button>"
            )
        parts.append("</form>")
    else:
        parts.append(
            '<p class="status problem">A paid completeness check is not '
            "available for this exact source state. Reload the page.</p>"
        )
    parts.append("</div></div></section>")
    return "".join(parts)


def _promotion_card(card: Any, owner: Any | None) -> str:
    identity = f"{card.landing.expression} ({card.landing.reading})"
    source = card.staged.source
    kind = "New card" if card.is_new else "Matches a card already in your collection"
    owners = owner.owners if owner is not None else ()
    deck = ", ".join(item.name for item in owners) or "no study deck"
    parts = [
        '<article class=card>',
        f'<h3 lang=ja>{_escaped(identity)}</h3>',
        f'<p class=state>{_escaped(kind)}</p>',
        f'<p>Study deck after adding: <b>{_escaped(deck)}</b>.</p>',
        "<p>Source history to record: "
        f"{_escaped(source.type or 'manual')} · "
        f"{_escaped(source.imported_from or 'unnamed source')}.</p>",
    ]
    if not card.is_new:
        parts.append(
            "<p>This keeps the existing card's curated fields and records this "
            "source alongside it.</p>"
        )
    if card.reminted_from:
        parts.append(
            f'<p class="status held">Its stable ID changes from '
            f"{_escaped(card.reminted_from)} to {_escaped(card.landing.id)}.</p>"
        )
    parts.append("</article>")
    return "".join(parts)


def _promotion_form(
    *,
    source_path: str,
    csrf: str,
    staging_fingerprint: str,
    preview_fingerprint: str,
    label: str,
    form_action: str = "promote",
    offline_preview_fingerprint: str = "",
) -> str:
    """The exact offline preview authority carried into one fresh plan."""
    action = html.escape(source_path + "/add", quote=True)
    parts = [
        f'<form method=post action="{action}" class=submit>',
        _hidden("action", form_action),
        _hidden("csrf", csrf),
        _hidden("staging_fingerprint", staging_fingerprint),
    ]
    if form_action == "promote-checked":
        parts.extend(
            [
                _hidden(
                    "offline_preview_fingerprint",
                    offline_preview_fingerprint,
                ),
                _hidden("checked_preview_fingerprint", preview_fingerprint),
            ]
        )
    else:
        parts.append(_hidden("preview_fingerprint", preview_fingerprint))
    parts.extend(
        [
            f"<button type=submit>{_escaped(label)}</button>",
            "</form>",
        ]
    )
    return "".join(parts)


def _promotion_preview(
    plan: PromotionPlan,
    *,
    source_path: str,
    csrf: str,
    staging_fingerprint: str,
    preview_fingerprint: str,
    checked_offline_preview_fingerprint: str = "",
) -> str:
    ownership = {item.record_id: item for item in plan.deck_ownership}
    landing = plan.landing
    parts = [
        '<section class=source><h2>What adding this source will do</h2>',
        f"<p><b>{len(plan.adding)}</b> new card(s); "
        f"<b>{len(plan.merging)}</b> existing card match(es); "
        f"<b>{len(plan.held)}</b> card(s) expected to remain held.</p>",
    ]
    if checked_offline_preview_fingerprint:
        parts.append(
            '<p class="status reviewed">jpdb has now been checked. Review '
            "this exact result, then use the separate add button below.</p>"
        )
    if plan.readings_unchecked and landing:
        parts.append(
            '<p class="status held">The preview has not spent dictionary '
            "lookups. When you add, jpdb is consulted in the same order as the "
            "command and may hold a card whose reading it contradicts.</p>"
        )
    parts.extend(
        _promotion_card(card, ownership.get(card.landing.id)) for card in landing
    )
    for card in plan.held:
        identity = f"{card.record.expression} ({card.record.reading})"
        parts.append(
            '<article class=card><h3 lang=ja>'
            f"{_escaped(identity)}</h3><p class=\"status held\">Stays in review: "
            f"{_escaped(card.reason)}</p></article>"
        )
    if plan.already_archived:
        parts.append(
            "<p>Already recorded by this extraction run: "
            f"{_escaped(', '.join(plan.already_archived))}.</p>"
        )
    if plan.warnings:
        parts.append('<section class=warnings><h3>Warnings</h3><ul>')
        parts.extend(f"<li>{_escaped(item)}</li>" for item in plan.warnings)
        parts.append("</ul></section>")
    label = (
        f"Add {len(landing)} card{'s' if len(landing) != 1 else ''} to your collection"
        if landing
        else "Finish this reviewed source"
    )
    parts.append(
        _promotion_form(
            source_path=source_path,
            csrf=csrf,
            staging_fingerprint=staging_fingerprint,
            preview_fingerprint=preview_fingerprint,
            label=label,
            form_action=(
                "promote-checked"
                if checked_offline_preview_fingerprint
                else "promote"
            ),
            offline_preview_fingerprint=checked_offline_preview_fingerprint,
        )
    )
    parts.append("</section>")
    return "".join(parts)


def render_addition(
    source: str,
    promotion: PromotionPlan,
    *,
    token: str,
    csrf: str,
    staging_fingerprint: str,
    preview_fingerprint: str = "",
    coverage: CoveragePreview | None = None,
    model_coverage: CoveragePreview | None = None,
    coverage_action: str = "",
    source_fingerprint: str = "",
    busy: str = "",
    model_error: str = "",
    reading_check_actionable: bool = False,
    checked_offline_preview_fingerprint: str = "",
) -> str:
    """Render coverage or an exact promotion preview for one staged source."""
    prefix = f"/{html.escape(token, quote=True)}"
    source_path = f"{prefix}/source/{quote(source, safe='')}"
    body = [
        "<!doctype html><html lang=en><head><meta charset=utf-8>",
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>Add {_escaped(source)} · janki workbench</title>",
        f'<link rel=stylesheet href="{prefix}/style.css">',
        "</head><body><main>",
        f'<p><a href="{html.escape(source_path, quote=True)}">&larr; Back to '
        "this source</a></p>",
        "<h1>Add cards to your collection</h1>",
    ]
    if coverage is not None:
        section = _coverage_actions(
            coverage,
            model_coverage=model_coverage,
            source_path=source_path,
            csrf=csrf,
            coverage_action=coverage_action,
            source_fingerprint=source_fingerprint,
            busy=busy,
            model_error=model_error,
        )
        body.append(section)
    elif promotion.is_blocked:
        body.append(
            '<p class="status problem">These cards cannot be added yet: '
            f"{_escaped(promotion.blocked)}</p>"
        )
        if reading_check_actionable:
            body.extend(
                [
                    '<section class=source><h2>A reading check can change this '
                    "provisional result</h2>",
                    "<p>This preview has not spent dictionary lookups. The same "
                    "jpdb check used by the command may hold cards before this "
                    "gate is decided. Nothing is added unless the checked plan "
                    "can proceed.</p>",
                    _promotion_form(
                        source_path=source_path,
                        csrf=csrf,
                        staging_fingerprint=staging_fingerprint,
                        preview_fingerprint=preview_fingerprint,
                        label="Check readings and preview what can be added",
                        form_action="check-readings",
                    ),
                    "</section>",
                ]
            )
    else:
        body.append(
            _promotion_preview(
                promotion,
                source_path=source_path,
                csrf=csrf,
                staging_fingerprint=staging_fingerprint,
                preview_fingerprint=preview_fingerprint,
                checked_offline_preview_fingerprint=(
                    checked_offline_preview_fingerprint
                ),
            )
        )
    body.append("</main></body></html>")
    return "".join(body)


def render_source(
    detail: Any,
    *,
    token: str = "",
    csrf: str = "",
    staging_snapshot: str = "",
    patterns_snapshot: str = "",
    saved: tuple[int, bool, int, int, int, int] | None = None,
    editing: bool = False,
    approvable: bool = True,
    reidentifiable: bool = True,
    assignment_offers: Sequence[Any] = (),
    assignment_error: str = "",
) -> str:
    """One source's cards and grammar.

    Without `csrf` this renders read-only — there is no form at all, so a page
    that cannot prove its session cannot show controls that would fail anyway.

    `approvable` is the same rule one level up: a source with no model run has
    nothing to approve, and the panel refuses such a submission. Offering the
    checkbox anyway would be a control that only ever errors — which is the
    thing this page already refuses to do for grammar review.
    """
    prefix = f"/{html.escape(token, quote=True)}" if token else ""
    journey = detail.journey
    waiting = sum(1 for card in detail.cards if card.needs_example_review)
    # Editing and approving are different acts, so they are different views.
    # One form carrying both would let a stray click approve sentences the
    # person was only correcting.
    editing = editing and bool(csrf) and bool(detail.cards)
    actionable = (
        bool(csrf)
        and approvable
        and not editing
        and (waiting or (not detail.pattern_reviewed and detail.can_review_grammar))
    )
    source_path = f"{prefix}/source/{quote(journey.source, safe='')}"
    body = [
        "<!doctype html><html lang=en><head><meta charset=utf-8>",
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{_escaped(journey.source)} — janki workbench</title>",
        f'<link rel=stylesheet href="{prefix}/style.css">',
        "</head><body><main>",
        f'<p><a href="{prefix}/">&larr; All sources</a></p>',
        f'<h1>{_escaped(journey.source)}</h1>',
        f"<p class=saved>{_escaped(journey.state)}. "
        f"{_escaped(journey.next_action)}.</p>",
    ]
    if saved is not None:
        body.append(_saved_banner(*saved))
    if assignment_error:
        body.append(
            '<p class="status problem">Study-deck choices are unavailable: '
            f"{_escaped(assignment_error)}</p>"
        )
    if detail.cards:
        body.append(
            f'<p><a class=button href="{html.escape(source_path + "/check", quote=True)}">'
            "Check these cards</a></p>"
        )
    if csrf:
        body.append(
            f'<p><a class=button href="{html.escape(source_path + "/add", quote=True)}">'
            "Preview adding these cards</a></p>"
        )
    if waiting:
        body.append(
            f"<p class=counts>{waiting} of {len(detail.cards)} cards still need "
            "their Japanese examples approved.</p>"
        )
    if editing:
        body.append(
            f'<form method=post action="{html.escape(source_path + "/edit", quote=True)}">'
            '<input type=hidden name=action value="edit">'
            f'<input type=hidden name=csrf value="{html.escape(csrf, quote=True)}">'
            "<input type=hidden name=staging_snapshot "
            f'value="{html.escape(staging_snapshot, quote=True)}">'
        )
    elif bool(csrf) and detail.cards:
        body.append(
            f'<p><a href="{html.escape(source_path + "?edit=1", quote=True)}">'
            "Correct these cards</a></p>"
        )
    if actionable:
        action = f"{source_path}/approve"
        body.append(f'<form method=post action="{html.escape(action, quote=True)}">')
        body.append(
            '<input type=hidden name=action value="save">'
            f'<input type=hidden name=csrf value="{html.escape(csrf, quote=True)}">'
            "<input type=hidden name=staging_snapshot "
            f'value="{html.escape(staging_snapshot, quote=True)}">'
            "<input type=hidden name=patterns_snapshot "
            f'value="{html.escape(patterns_snapshot, quote=True)}">'
        )
    if not detail.cards:
        body.append("<p class=empty>This source proposed no word cards.</p>")
    body.extend(
        _card_html(
            card,
            actionable=actionable,
            editing=editing,
            index=index,
            remove_action=(
                _remove_form(
                    card, index, source_path, csrf, staging_snapshot
                )
                if editing
                else ""
            ),
            reidentify_action=(
                _reidentify_form(index)
                if editing and reidentifiable
                else ""
            ),
            assignment_action=(
                _assignment_html(assignment_offers[index])
                if not editing and index < len(assignment_offers)
                else ""
            ),
        )
        for index, card in enumerate(detail.cards)
    )
    body.append(_grammar_html(detail, actionable=actionable))
    if actionable:
        # One button, named for what it does. There is deliberately no
        # "approve all": cards, grammar, coverage and deck ownership are
        # separate decisions and a single control would blur them.
        body.append(
            '<div class="submit"><button type=submit>Save the approvals I '
            "ticked</button></div></form>"
        )
    if editing:
        # `type=reset` is undo-before-save with no JavaScript and no server
        # round trip: it restores every field to what this page was rendered
        # with. Leaving without saving writes nothing at all.
        body.append(
            '<div class="submit"><button type=submit>Save these corrections'
            "</button> <button type=reset>Undo my changes</button> "
            f'<a href="{html.escape(source_path, quote=True)}">Leave without '
            "saving</a></div></form>"
        )
        body.append(_remove_forms(detail, source_path, csrf, staging_snapshot))
        body.append(
            _reidentify_forms(detail, source_path, csrf, staging_snapshot)
        )
    if not editing and csrf and assignment_offers:
        body.append(
            _assignment_forms(
                assignment_offers,
                source_path,
                csrf,
                staging_snapshot,
            )
        )
    body.append("</main></body></html>")
    return "".join(body)


def _saved_banner(
    records: int,
    grammar: bool,
    edited: int = 0,
    removed: int = 0,
    reidentified: int = 0,
    assigned: int = 0,
) -> str:
    saved = []
    if reidentified:
        saved.append("a card's identity")
    if removed:
        saved.append(f"{removed} card{'s' if removed != 1 else ''} removed")
    if assigned:
        saved.append(
            f"a study deck for {assigned} card{'s' if assigned != 1 else ''}"
        )
    if edited:
        saved.append(f"corrections to {edited} card{'s' if edited != 1 else ''}")
    if records:
        saved.append(f"{records} card{'s' if records != 1 else ''}")
    if grammar:
        saved.append("this lesson's grammar")
    if not saved:
        return '<p class="status">Nothing was ticked, so nothing was saved.</p>'
    return (
        f'<p class="status reviewed">Saved: {_escaped(" and ".join(saved))}.</p>'
    )


# --- W2d: re-identification -------------------------------------------------


def _reidentify_form(index: int) -> str:
    """The control that opens the flow, from inside the editor.

    It submits to a route that *shows* before it writes, so this is a button
    that leads to a decision rather than one that makes it.
    """
    return (
        '<div class="reidentify">'
        f'<button type=submit form="ri{index}">'
        "This is a different word than it says</button>"
        "<span class=counts>Changing the expression or reading changes which "
        "word this card is. That is a different question from fixing a gloss, "
        "so it gets its own page.</span>"
        "</div>"
    )


def _reidentify_forms(
    detail: Any, source_path: str, csrf: str, snapshot: str
) -> str:
    """One form per card, declared after the editor's form closes."""
    action = html.escape(source_path + "/reidentify", quote=True)
    parts = []
    for index, card in enumerate(detail.cards):
        record = card.record
        parts.append(
            f'<form id="ri{index}" method=post action="{action}" hidden>'
            '<input type=hidden name=action value="reidentify">'
            f'<input type=hidden name=csrf value="{html.escape(csrf, quote=True)}">'
            "<input type=hidden name=staging_snapshot "
            f'value="{html.escape(snapshot, quote=True)}">'
            f'<input type=hidden name=card value="{index}">'
            "<input type=hidden name=expression "
            f'value="{html.escape(record.expression, quote=True)}">'
            "<input type=hidden name=reading "
            f'value="{html.escape(record.reading, quote=True)}">'
            "</form>"
        )
    return "".join(parts)


def _neighbour_html(neighbour: Any) -> str:
    label = {
        "same-id": "Already this exact word",
        "same-reading": "Same reading, different spelling",
        "same-spelling": "Same spelling, different reading",
    }[neighbour.relation]
    identity = f"{neighbour.expression} ({neighbour.reading})"
    return (
        f"<li><b>{_escaped(label)}</b> — "
        f'<span lang="ja">{_escaped(identity)}</span>'
        f" in {_escaped(neighbour.where)}</li>"
    )


def render_reidentify(
    source: str,
    plan: Any,
    *,
    token: str = "",
    csrf: str = "",
    staging_snapshot: str = "",
) -> str:
    """Show what changing this card's identity would do, before it is done."""
    prefix = f"/{html.escape(token, quote=True)}" if token else ""
    source_path = f"{prefix}/source/{quote(source, safe='')}"
    action = html.escape(source_path + "/reidentify", quote=True)

    body = [
        "<!doctype html><html lang=en><head><meta charset=utf-8>",
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>Re-identify — {_escaped(source)}</title>",
        f'<link rel=stylesheet href="{prefix}/style.css">',
        "</head><body><main>",
        f'<p><a href="{html.escape(source_path, quote=True)}">&larr; Back to '
        f"{_escaped(source)}</a></p>",
        "<h1>Is this a different word?</h1>",
        '<article class="card">',
        "<h4>What this card says now</h4>",
        f'<p class="ja" lang="ja">{_escaped(plan.old_expression)} '
        f"({_escaped(plan.old_reading)})</p>",
        f"<p class=counts>{_escaped(plan.old_id)}</p>",
        "</article>",
        f'<form method=post action="{action}">',
        '<input type=hidden name=action value="reidentify">',
        f'<input type=hidden name=csrf value="{html.escape(csrf, quote=True)}">',
        "<input type=hidden name=staging_snapshot "
        f'value="{html.escape(staging_snapshot, quote=True)}">',
        f'<input type=hidden name=card value="{plan.index}">',
        '<article class="card"><h4>What it should say</h4>',
        # Prefilled with what was typed, never with a suggestion: deciding that
        # a kana spelling "should" be a particular kanji is reading Japanese,
        # which this project reserves for the model and the person.
        _textarea("expression", "Japanese expression", plan.new_expression,
                  rows=1, japanese=True),
        _textarea("reading", "Reading", plan.new_reading, rows=1, japanese=True),
        f"<p class=counts>New identity: {_escaped(plan.new_id)}</p>",
        "</article>",
    ]

    if plan.neighbours:
        body.append(
            '<article class="card"><h4>Words this would sit beside</h4><ul>'
            + "".join(_neighbour_html(n) for n in plan.neighbours)
            + "</ul></article>"
        )

    body.append('<article class="card"><h4>What would happen</h4>')
    for sentence in plan.consequences():
        # A same-file collision is an error; matching the collection is not.
        css = "status problem" if plan.collides_in_source else "status"
        body.append(f'<p class="{css}">{_escaped(sentence)}</p>')
    body.append("</article>")

    if plan.collides_in_source:
        body.append(
            '<div class="submit"><button type=submit>Check a different '
            "identity</button></div>"
        )
    elif plan.is_change:
        body.append(
            "<div class=\"submit\">"
            "<button type=submit name=confirm "
            f'value="{html.escape(plan.new_id, quote=True)}">'
            f"Yes — this card is {_escaped(plan.new_expression)} "
            f"({_escaped(plan.new_reading)})</button> "
            "<button type=submit>Check a different identity</button> "
            f'<a href="{html.escape(source_path, quote=True)}">Leave it as it '
            "is</a></div>"
        )
    else:
        body.append(
            '<div class="submit"><button type=submit>Check a different '
            "identity</button> "
            f'<a href="{html.escape(source_path, quote=True)}">Leave it as it '
            "is</a></div>"
        )
    body.append("</form>")
    body.append("</main></body></html>")
    return "".join(body)

# --- what one paid call would send -------------------------------------------


#: The mode choice, in what a person would recognise about their own handout
#: rather than in `table` and `prose`, which are janki's words for prompts.
_MODES: tuple[tuple[str, str, str], ...] = (
    ("", "Choose automatically", "janki decides from the page. Start here."),
    (
        "table",
        "A vocabulary list",
        "Columns of words with readings and meanings — every row is copied.",
    ),
    (
        "prose",
        "A lesson, dialogue or exercise",
        "Running text — janki picks out the words worth a card.",
    ),
)


def _mode_choice(prefix: str, name: str, selected: str | None) -> str:
    """Advanced, and collapsed: the default is right for almost every page.

    A real GET form with its own submit, not bare radios. Radios that reorder
    nothing and submit nothing are a control that does not control anything —
    and on *this* page the cost of that is specific: somebody selects "A
    vocabulary list", reads the plan beside it, and consents to a run
    described under different instructions from the ones they picked.

    GET because choosing how to *describe* a page changes nothing on disk.
    """
    chosen = selected or ""
    action = html.escape(f"{prefix}/extract/{quote(name, safe='')}", quote=True)
    rows = []
    for value, label, hint in _MODES:
        mark = " checked" if value == chosen else ""
        rows.append(
            "<p class=edit><label>"
            f'<input type=radio name=mode value="{html.escape(value, quote=True)}"'
            f"{mark}> {_escaped(label)}</label>"
            f"<span class=counts>{_escaped(hint)}</span></p>"
        )
    opened = " open" if chosen else ""
    return (
        f"<details class=advanced{opened}><summary>What kind of page is this?"
        f'</summary><form method=get action="{action}">'
        + "".join(rows)
        + "<button type=submit>Describe it as this kind</button>"
        "<span class=counts>This only changes what the plan above says. "
        "Nothing is sent.</span>"
        "</form></details>"
    )


def render_consent(
    consent: Any,
    *,
    token: str = "",
    csrf: str = "",
    dispatch: str = "",
) -> str:
    """The page that asks whether to spend money, and says what on.

    Every sentence here is load-bearing (WORKBENCH_PLAN.md W3). It names the
    one file leaving the computer, who answers, and that the call is billed —
    and it says the copy in the corpus stays put, because "sending" and
    "uploading my library" are the same words to someone who has not thought
    about it. Claude Max gets its own sentence: it is the single most likely
    wrong belief a person arrives with, and the one that turns an informed
    consent into a surprise invoice.
    """
    prefix = f"/{html.escape(token, quote=True)}" if token else ""
    name = _escaped(consent.name)
    body = [
        "<!doctype html><html lang=en><head><meta charset=utf-8>",
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>Send {name} to a model</title>",
        f'<link rel=stylesheet href="{prefix}/style.css">',
        "</head><body><main>",
        f'<p class=saved><a href="{prefix}/">Back to your sources</a></p>',
        f"<h1>Send {name} to a model?</h1>",
    ]

    if consent.refusal:
        body.append(
            f'<p class="status held">{_escaped(consent.refusal)}</p>'
            "</main></body></html>"
        )
        return "".join(body)

    body.append(
        '<p class=lead>Send <strong>' + name + "</strong> to "
        f"<strong>{_escaped(consent.model)}</strong> to propose vocabulary "
        "cards and grammar — <strong>a paid API call</strong>.</p>"
    )
    # Four separate sentences rather than one paragraph: each answers a
    # different wrong belief, and a reader skimming one block absorbs none.
    body.append(
        "<ul class=disclosure>"
        "<li>Only this one file is sent. Nothing else in your corpus leaves "
        "this computer.</li>"
        "<li>Your copy stays where it is. Sending does not move or delete "
        "it.</li>"
        "<li>The whole document is sent — janki cannot yet send only some "
        "pages.</li>"
        "<li>This spends <strong>Anthropic API credits</strong>, billed to "
        "your API account.</li>"
        "<li>A Claude Pro or Max subscription is a different thing and does "
        "not pay for this.</li>"
        + (
            # Prose mode puts every expression already in the collection into
            # the prompt so the model can skip them. "Only this one file"
            # would otherwise be read as covering that too, and a learner has
            # no reason to distinguish their corpus from their collection.
            "<li>Because you chose a lesson or dialogue, the list of words "
            "you already have is sent too, so the model can skip them.</li>"
            if consent.sends_known_words
            else ""
        )
        + "</ul>"
    )

    if consent.replaces is not None:
        # Named, not merely flagged: "this will replace your review" is not a
        # decision anybody can make without knowing which review.
        card_review = (
            f"{consent.replaces_cards} cards, {_escaped(consent.replaces_state)}"
            if consent.replaces_state
            else "a review already on disk"
        )
        held = (
            f"{card_review}; {_escaped(consent.replaces_grammar)}"
            if consent.replaces_grammar
            else card_review
        )
        body.append(
            '<p class="status held">Reading this again replaces what you '
            f"already have for it — {held}. That work is not recoverable "
            "afterwards.</p>"
        )

    if consent.busy:
        body.append(f'<p class="status held">{_escaped(consent.busy)}</p>')

    body.append(_mode_choice(prefix, consent.name, consent.mode))
    body.append(
        '<p class=counts>Nothing has been sent. This page only describes what '
        "sending would do.</p>"
    )
    if consent.sendable and csrf and dispatch and consent.target is not None:
        action = html.escape(
            f"{prefix}/extract/{quote(consent.name, safe='')}", quote=True
        )
        fingerprint = html.escape(
            str(consent.target.provenance["request_fingerprint"]), quote=True
        )
        escaped_csrf = html.escape(csrf, quote=True)
        escaped_mode = html.escape(consent.mode or "", quote=True)
        escaped_model = html.escape(consent.model, quote=True)
        body.append(f'<form method=post action="{action}" class=submit>')
        # HTML implicit submission clicks the first submit button. A disabled
        # default makes an ordinary Enter harmless; the later paid button is
        # still keyboard-focusable for an intentional activation.
        body.append(
            '<button type=submit disabled class="implicit-submit-guard" '
            'aria-hidden=true tabindex=-1>Do not send</button>'
        )
        body.append('<input type=hidden name=action value="extract">')
        body.append(
            f'<input type=hidden name=csrf value="{escaped_csrf}">'
            f'<input type=hidden name=mode value="{escaped_mode}">'
            f'<input type=hidden name=model value="{escaped_model}">'
            f'<input type=hidden name=request_fingerprint value="{fingerprint}">'
            f'<input type=hidden name=replacement value="{1 if consent.replaces else 0}">'
        )
        if consent.replaces is not None:
            review_scope = (
                "the card and grammar reviews named above"
                if consent.replaces_grammar
                else "the card review named above"
            )
            body.append(
                '<label class=decision><input type=checkbox name=replace '
                'value="confirmed" required> <span>I confirm that reading '
                f"this source again replaces {review_scope}, and that work "
                "is not recoverable.</span></label>"
            )
        body.append(
            '<button type=submit name="dispatch" '
            f'value="{html.escape(dispatch, quote=True)}">Send {name} to '
            f"Anthropic using {_escaped(consent.model)} to propose vocabulary "
            "cards and grammar — paid API call</button></form>"
        )
    body.append("</main></body></html>")
    return "".join(body)


def render_extraction_progress_start(name: str, *, token: str) -> str:
    """Open the streamed extraction page; later chunks close it."""
    prefix = f"/{html.escape(token, quote=True)}"
    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Reading {_escaped(name)}</title>"
        f'<link rel=stylesheet href="{prefix}/style.css">'
        "</head><body><main>"
        f"<h1>Reading {_escaped(name)}</h1>"
        '<p class=saved>Keep this page open. Closing it does not cancel a call '
        "that has already been sent.</p>"
        '<ol class=progress aria-live=polite>'
    )


def render_extraction_progress_step(label: str) -> str:
    """One truthful human phase, appended while the POST is running."""
    return f'<li role=status><strong>{_escaped(label)}</strong></li>'


def render_extraction_success(name: str, outcome: Any, *, token: str) -> str:
    """Close a streamed progress page with the shared writer's result."""
    target = f"/{html.escape(token, quote=True)}/source/{quote(name, safe='')}"
    noun = "proposal" if outcome.records == 1 else "proposals"
    note = (
        " The grammar review already on disk was kept."
        if outcome.kept_reviewed_patterns
        else ""
    )
    return (
        "</ol>"
        f'<p class="status reviewed">Saved {outcome.records} card {noun} and '
        f"the grammar from this source.{note}</p>"
        f'<p><a class=button href="{html.escape(target, quote=True)}">'
        "Review proposed cards</a></p>"
        "</main></body></html>"
    )


def render_extraction_failure(failure: FailureView) -> str:
    """Close a streamed page without claiming more than the journal proves."""
    return (
        "</ol>"
        '<section class="status problem" role=alert>'
        f"{_failure_facts(failure)}</section>"
        "</main></body></html>"
    )
