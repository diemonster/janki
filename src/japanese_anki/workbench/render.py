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

from japanese_anki.application import NOT_EXTRACTED, SourceJourney

__all__ = [
    "STYLE",
    "render_consent",
    "render_dashboard",
    "render_reidentify",
    "render_source",
]

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
.actions {
  display: flex; flex-wrap: wrap; gap: 1rem;
  margin-block-start: 1rem; padding-block-start: .75rem;
  border-block-start: 1px solid ButtonBorder;
}
.actions > div { flex: 1 1 16rem; display: grid; gap: .35rem; align-content: start; }
/* Content width, not column width: a button stretched across the card reads
   as a disabled input, which is how the removal control came to look inert. */
.actions button { justify-self: start; text-align: start; }
.actions .counts { margin: 0; }
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


def render_dashboard(
    journeys: Sequence[SourceJourney],
    *,
    warnings: Sequence[str] = (),
    root: Path | None = None,
    token: str = "",
    csrf: str = "",
    added: tuple[str, bool] | None = None,
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
    if csrf:
        body.append(_add_source_form(prefix, csrf))
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
        '<p class=edit><label for="source-file">A PDF or photo of your study '
        "material</label>"
        '<input id="source-file" name="file" type=file '
        'accept=".pdf,.jpg,.jpeg,.png,.heic,.heif" required></p>'
        "<button type=submit>Add this to my corpus</button>"
        "<span class=counts>This copies the file into your corpus and sends it "
        "nowhere. Reading it with a model is a separate, paid step you choose "
        "afterwards.</span>"
        "</form>"
    )


def _saved_where(root: Path | None) -> str:
    return f" at {_escaped(root)}" if root is not None else ""


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
) -> str:
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


def render_source(
    detail: Any,
    *,
    token: str = "",
    csrf: str = "",
    staging_snapshot: str = "",
    patterns_snapshot: str = "",
    saved: tuple[int, bool, int, int, int] | None = None,
    editing: bool = False,
    approvable: bool = True,
    reidentifiable: bool = True,
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
    body.append("</main></body></html>")
    return "".join(body)


def _saved_banner(
    records: int,
    grammar: bool,
    edited: int = 0,
    removed: int = 0,
    reidentified: int = 0,
) -> str:
    saved = []
    if reidentified:
        saved.append("a card's identity")
    if removed:
        saved.append(f"{removed} card{'s' if removed != 1 else ''} removed")
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


def render_consent(consent: Any, *, token: str = "") -> str:
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
        held = (
            f"{consent.replaces_cards} cards, {_escaped(consent.replaces_state)}"
            if consent.replaces_state
            else "a review already on disk"
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
    if consent.sendable:
        # The page that asks the question must not be the only one without an
        # answer to it. Sending from here arrives in the next step; until it
        # does, the way that exists gets named rather than left to be guessed.
        body.append(
            "<p class=counts>Sending from this page is not built yet. To go "
            f"ahead now, run <code>janki extract {_escaped(consent.name)}"
            "</code> in a terminal.</p>"
        )
    body.append("</main></body></html>")
    return "".join(body)
