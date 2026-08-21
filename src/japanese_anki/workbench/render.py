"""The dashboard, as server-rendered HTML.

No frontend build, no remote asset, no script tag — the page is a list, and a
list does not need a framework. The CSS uses system colours (`Canvas`,
`CanvasText`, `Highlight`) so light mode, dark mode and forced-colours mode all
work without a theme switcher, which `WORKBENCH_PLAN.md` makes a completion
requirement rather than polish.

Every value that reaches this module is escaped. A source filename is
attacker-adjacent input — it arrives from whatever the person dragged in — and
it is rendered beside their own study material.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote

from japanese_anki.application import SourceJourney

__all__ = ["STYLE", "render_dashboard", "render_source"]

STYLE = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body {
  margin: 0; padding: 1.5rem;
  font: 1rem/1.6 system-ui, sans-serif;
  background: Canvas; color: CanvasText;
}
main { max-width: 60rem; margin: 0 auto; }
h1 { font-size: 1.5rem; margin: 0 0 .25rem; }
.saved { color: GrayText; margin: 0 0 2rem; font-size: .9rem; }
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
summary { cursor: pointer; min-height: 44px; display: flex; align-items: center; }
pre {
  white-space: pre-wrap; overflow-wrap: anywhere;
  background: Canvas; border: 1px solid GrayText;
  padding: .5rem; border-radius: .25rem; margin: .5rem 0 0;
}
.done { color: GrayText; }
.warnings { border: 1px solid red; border-radius: .5rem; padding: 1rem; }
.empty { color: GrayText; }
.decision { display: flex; gap: .75rem; align-items: flex-start; cursor: pointer;
  padding: .75rem; border: 1px solid GrayText; border-radius: .35rem; margin-top: .75rem; }
input[type=checkbox] { inline-size: 1.4rem; block-size: 1.4rem; flex: none; }
button { min-height: 44px; padding: .65rem 1rem; font: inherit; font-weight: 700; }
.submit { position: sticky; bottom: 0; padding: 1rem; background: Canvas;
  border-top: 1px solid GrayText; }
.card, .pattern {
  border: 1px solid GrayText; border-radius: .5rem;
  padding: 1rem; margin-block-end: 1rem;
}
.card h3 { font-size: 1.35rem; margin: 0 0 .25rem; }
.card h3 small { font-size: .85em; font-weight: 400; color: GrayText; }
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
:focus-visible { outline: 3px solid Highlight; outline-offset: 3px; }
@media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto !important; } }
""".strip()


def _escaped(value: object) -> str:
    return html.escape(str(value))


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
        f'<h2 class=name lang="ja">{_source_link(journey, prefix)}</h2>',
        f'<p class=state>{_escaped(journey.state)}</p>',
    ]
    if badges:
        parts.append(f'<p class=badges>{"".join(badges)}</p>')
    parts.append(_counts(journey))
    parts.append(
        f'<p class=next><b>Next:</b> {_escaped(journey.next_action)}</p>'
    )
    detail = " ".join(filter(None, (journey.detail, journey.grammar_detail)))
    if detail:
        parts.append(
            "<details><summary>Source and technical details</summary>"
            f"<pre>{_escaped(detail)}</pre></details>"
        )
    parts.append("</article>")
    return "".join(parts)


def render_dashboard(
    journeys: Sequence[SourceJourney],
    *,
    warnings: Sequence[str] = (),
    root: Path | None = None,
    token: str = "",
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
    if warnings:
        body.append('<section class="warnings"><h2>Could not be read</h2><ul>')
        body.extend(f"<li>{_escaped(warning)}</li>" for warning in warnings)
        body.append("</ul></section>")
    if not journeys:
        body.append(
            '<p class=empty>Add a PDF or photo to your inbox folder, then run '
            "<code>janki extract</code>.</p>"
        )
    body.extend(_source_html(journey, prefix) for journey in journeys)
    body.append("</main></body></html>")
    return "".join(body)


def _saved_where(root: Path | None) -> str:
    return f" at {_escaped(root)}" if root is not None else ""


# --- one source, opened ------------------------------------------------------


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


def _card_html(card: Any, *, actionable: bool = False) -> str:
    record = card.record
    parts = [
        '<article class="card">',
        f'<h3 lang="ja">{_escaped(record.expression)}'
        f" <small>{_escaped(record.reading)}</small></h3>",
    ]
    if record.furigana:
        parts.append(f'<p class="furigana" lang="ja">{_escaped(record.furigana)}</p>')
    if card.hold_reason:
        parts.append(
            f'<p class="status problem">Held for a reading decision: '
            f"{_escaped(card.hold_reason)}</p>"
        )
    parts.append(_meaning_panels(card))
    if record.examples:
        parts.append("<h4>Japanese examples</h4>")
        parts.extend(
            _example_html(example, number)
            for number, example in enumerate(record.examples, start=1)
        )
    if record.usage_notes:
        parts.append(f"<h4>How to use it</h4><p>{_escaped(record.usage_notes)}</p>")
    if actionable and card.needs_example_review:
        parts.append(_checkbox_html(card))
    else:
        parts.append(_approval_html(card))
    parts.append(_evidence_html(card))
    parts.append("</article>")
    return "".join(parts)


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
        ("Stable ID", card.record.id),
        ("Source page", raw.get("page")),
        ("Source sentence", raw.get("context")),
        ("Why it was included", raw.get("inclusion_reason")),
        ("Confidence", raw.get("confidence")),
    ]
    shown = "".join(
        f"<div class=field><dt>{_escaped(label)}</dt>"
        f'<dd lang="ja">{_escaped(value)}</dd></div>'
        for label, value in rows
        if value not in (None, "")
    )
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


def render_source(
    detail: Any,
    *,
    token: str = "",
    csrf: str = "",
    staging_snapshot: str = "",
    patterns_snapshot: str = "",
    saved: tuple[int, bool] | None = None,
) -> str:
    """One source's cards and grammar.

    Without `csrf` this renders read-only — there is no form at all, so a page
    that cannot prove its session cannot show controls that would fail anyway.
    """
    prefix = f"/{html.escape(token, quote=True)}" if token else ""
    journey = detail.journey
    waiting = sum(1 for card in detail.cards if card.needs_example_review)
    actionable = bool(csrf) and (
        waiting or (not detail.pattern_reviewed and detail.can_review_grammar)
    )
    body = [
        "<!doctype html><html lang=en><head><meta charset=utf-8>",
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{_escaped(journey.source)} — janki workbench</title>",
        f'<link rel=stylesheet href="{prefix}/style.css">',
        "</head><body><main>",
        f'<p><a href="{prefix}/">&larr; All sources</a></p>',
        f'<h1 lang="ja">{_escaped(journey.source)}</h1>',
        f"<p class=saved>{_escaped(journey.state)}. "
        f"{_escaped(journey.next_action)}.</p>",
    ]
    if saved is not None:
        body.append(_saved_banner(*saved))
    if waiting:
        body.append(
            f"<p class=counts>{waiting} of {len(detail.cards)} cards still need "
            "their Japanese examples approved.</p>"
        )
    if actionable:
        action = f"{prefix}/source/{quote(journey.source, safe='')}/approve"
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
    body.extend(_card_html(card, actionable=actionable) for card in detail.cards)
    body.append(_grammar_html(detail, actionable=actionable))
    if actionable:
        # One button, named for what it does. There is deliberately no
        # "approve all": cards, grammar, coverage and deck ownership are
        # separate decisions and a single control would blur them.
        body.append(
            '<div class="submit"><button type=submit>Save the approvals I '
            "ticked</button></div></form>"
        )
    body.append("</main></body></html>")
    return "".join(body)


def _saved_banner(records: int, grammar: bool) -> str:
    saved = []
    if records:
        saved.append(f"{records} card{'s' if records != 1 else ''}")
    if grammar:
        saved.append("this lesson's grammar")
    if not saved:
        return '<p class="status">Nothing was ticked, so nothing was saved.</p>'
    return (
        f'<p class="status reviewed">Saved: {_escaped(" and ".join(saved))}.</p>'
    )


def _saved_where(root: Path | None) -> str:
    return f" at {_escaped(root)}" if root is not None else ""
