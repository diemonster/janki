"""Structural checks on every page the workbench renders.

Written because a real bug shipped past a full HTTP-level suite. W2c first
rendered each card's removal `<form>` *inside* the editor's form. Forms cannot
nest: a browser silently drops the inner one, so every remove button would
have been dead — and not one test noticed, because an HTTP test reads the
bytes rather than the document a browser builds from them.

No browser runs here. What runs is a parser over the exact HTML the workbench
serves, asserting the structural invariants a browser would otherwise enforce
by quietly discarding things: forms do not nest, `form=` points at a form that
exists, ids are unique, every label is attached to a control, and every control
that carries a value has a name to carry it under. Each of those failing means
a control that renders and does nothing.

This is not a substitute for opening the page. It is the part of "does this
work in a browser" that can be checked without one, so that the part which
still needs a person is smaller and sharper.
"""

from __future__ import annotations

from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

import pytest
from test_application_journey import _project, _stage

from japanese_anki.config import ProjectConfig
from japanese_anki.workbench import (
    WorkbenchSession,
    render_dashboard,
    render_reidentify,
    render_source,
)
from japanese_anki.workbench.reidentify import plan_reidentification

#: Elements with no closing tag. A parser that does not know these reports
#: every page as catastrophically unbalanced.
_VOID = frozenset(
    {
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr",
    }
)

#: Elements HTML5 lets you leave unclosed, mapped to the start tags that
#: implicitly close them. Without this the checker rejects perfectly valid
#: markup — `<li>one<li>two` is legal HTML and `HTMLParser` does no implicit
#: closing of its own. A checker that cries wolf on valid input is a checker
#: someone eventually deletes, so it has to model this much of the parse.
_BLOCK = frozenset(
    {
        "address", "article", "aside", "blockquote", "details", "div", "dl",
        "fieldset", "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5",
        "h6", "header", "hr", "main", "nav", "ol", "p", "pre", "section",
        "table", "ul",
    }
)
_CLOSED_BY: dict[str, frozenset[str]] = {
    "li": frozenset({"li"}),
    "dt": frozenset({"dt", "dd"}),
    "dd": frozenset({"dt", "dd"}),
    "p": _BLOCK,
    "option": frozenset({"option", "optgroup"}),
    "optgroup": frozenset({"optgroup"}),
    "tr": frozenset({"tr"}),
    "td": frozenset({"td", "th", "tr"}),
    "th": frozenset({"td", "th", "tr"}),
    "thead": frozenset({"tbody", "tfoot"}),
    "tbody": frozenset({"tbody", "tfoot"}),
    "rt": frozenset({"rt", "rp"}),
    "rp": frozenset({"rt", "rp"}),
}

#: Controls that submit a value and therefore need a name to submit it under.
_NAMED_CONTROLS = frozenset({"input", "select", "textarea"})

#: Input types that legitimately carry no name.
_NAMELESS_INPUT_TYPES = frozenset({"submit", "reset", "button", "image"})


class _Document(HTMLParser):
    """Just enough of a document model to check the invariants above."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.open_tags: list[str] = []
        self.ids: list[str] = []
        self.label_targets: list[str] = []
        self.form_refs: list[str] = []
        self.form_depth = 0
        self.max_form_depth = 0
        self.nameless_controls: list[str] = []
        self.unbalanced: list[str] = []
        self.controls_outside_forms: list[str] = []

    def _close_implicit(self, upcoming: str | None) -> None:
        """Pop elements HTML5 closes for you before `upcoming` opens or closes."""
        while self.open_tags:
            top = self.open_tags[-1]
            if top not in _CLOSED_BY:
                return
            if upcoming is not None and upcoming not in _CLOSED_BY[top]:
                return
            self.open_tags.pop()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key: (value or "") for key, value in attrs}
        self._close_implicit(tag)
        if tag == "form":
            self.form_depth += 1
            self.max_form_depth = max(self.max_form_depth, self.form_depth)
        if "id" in values:
            self.ids.append(values["id"])
        if tag == "label" and "for" in values:
            self.label_targets.append(values["for"])
        if "form" in values and tag in (_NAMED_CONTROLS | {"button"}):
            self.form_refs.append(values["form"])
        if (
            tag in _NAMED_CONTROLS
            and not values.get("name")
            and not (
                tag == "input" and values.get("type") in _NAMELESS_INPUT_TYPES
            )
        ):
            self.nameless_controls.append(f"{tag} {values}")
        # A control outside any form only submits if it names one.
        if (
            tag in (_NAMED_CONTROLS | {"button"})
            and self.form_depth == 0
            and not values.get("form")
        ):
            self.controls_outside_forms.append(f"{tag} {values}")
        if tag not in _VOID:
            self.open_tags.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag == "form":
            self.form_depth -= 1
        if tag in _VOID:
            return
        if tag not in _CLOSED_BY:
            # `</ul>` closes an open `<li>`; `</select>` closes an `<option>`.
            while (
                self.open_tags
                and self.open_tags[-1] != tag
                and self.open_tags[-1] in _CLOSED_BY
            ):
                self.open_tags.pop()
        if not self.open_tags:
            self.unbalanced.append(f"</{tag}> with nothing open")
            return
        if self.open_tags[-1] != tag:
            self.unbalanced.append(f"</{tag}> closed while <{self.open_tags[-1]}> open")
            return
        self.open_tags.pop()


def _check(html: str) -> _Document:
    document = _Document()
    document.feed(html)
    document.close()

    assert not document.unbalanced, document.unbalanced
    assert document.open_tags == [], f"never closed: {document.open_tags}"
    assert document.max_form_depth <= 1, (
        "a <form> is nested inside another; a browser drops the inner one and "
        "every control in it silently stops working"
    )
    duplicates = [name for name, count in Counter(document.ids).items() if count > 1]
    assert not duplicates, f"duplicate id: {duplicates}"
    missing = sorted(set(document.form_refs) - set(document.ids))
    assert not missing, f'form="..." points at no such form: {missing}'
    orphan_labels = sorted(set(document.label_targets) - set(document.ids))
    assert not orphan_labels, f"<label for=...> points at no control: {orphan_labels}"
    assert not document.nameless_controls, (
        f"control submits no value: {document.nameless_controls}"
    )
    assert not document.controls_outside_forms, (
        f"control outside any form and naming none: {document.controls_outside_forms}"
    )
    return document


def _session(tmp_path: Path, scenario: str, name: str) -> WorkbenchSession:
    _stage(tmp_path, scenario, filename=name)
    return WorkbenchSession.open(ProjectConfig.load(tmp_path))


def test_the_dashboard_is_structurally_sound(tmp_path: Path) -> None:
    session = _session(tmp_path, "lesson_with_grammar", "lesson.pdf")
    journeys, warnings = session.journeys()
    _check(render_dashboard(journeys, warnings=warnings, root=tmp_path,
                            token=session.token))


def test_an_empty_dashboard_is_structurally_sound(tmp_path: Path) -> None:
    _project(tmp_path)
    _check(render_dashboard([], root=tmp_path, token="tok"))


@pytest.mark.parametrize(
    "scenario", ["table_exhaustive", "lesson_with_grammar", "reading_holds"]
)
def test_the_read_view_is_structurally_sound(tmp_path: Path, scenario: str) -> None:
    session = _session(tmp_path, scenario, "source.pdf")
    detail = session.detail("source.pdf")
    panel = session.panel("source.pdf")
    assert panel is not None
    _check(
        render_source(
            detail,
            token=session.token,
            csrf=session.csrf_token,
            staging_snapshot=panel.staging_fingerprint,
            patterns_snapshot=panel.patterns_fingerprint,
        )
    )


@pytest.mark.parametrize(
    "scenario", ["table_exhaustive", "lesson_with_grammar", "reading_holds"]
)
def test_the_edit_view_is_structurally_sound(tmp_path: Path, scenario: str) -> None:
    """The view that regressed. Every card carries a removal button and a
    re-identify button, both associated by `form=` with forms declared after
    the editor's form closes."""
    session = _session(tmp_path, scenario, "source.pdf")
    detail = session.detail("source.pdf")
    panel = session.panel("source.pdf")
    assert panel is not None
    html = render_source(
        detail,
        token=session.token,
        csrf=session.csrf_token,
        staging_snapshot=panel.staging_fingerprint,
        patterns_snapshot=panel.patterns_fingerprint,
        editing=True,
    )
    document = _check(html)
    # Every card's two out-of-form buttons really do reach a form.
    assert len(document.form_refs) == 2 * len(detail.cards)


def test_the_reidentify_preview_is_structurally_sound(tmp_path: Path) -> None:
    session = _session(tmp_path, "reading_holds", "source.pdf")
    panel = session.panel("source.pdf")
    assert panel is not None
    plan = plan_reidentification(panel.records, 0, "泊まる", "とまる")
    _check(
        render_reidentify(
            "source.pdf",
            plan,
            token=session.token,
            csrf=session.csrf_token,
            staging_snapshot=panel.staging_fingerprint,
        )
    )


def test_a_page_with_no_session_offers_no_controls(tmp_path: Path) -> None:
    """Rendered without a session there is no form, so there must be no
    control either — a button on a page that cannot submit is a control that
    silently does nothing."""
    session = _session(tmp_path, "table_exhaustive", "source.pdf")
    html = render_source(session.detail("source.pdf"), token=session.token)
    document = _check(html)
    assert document.max_form_depth == 0
    assert "<textarea" not in html
    assert "<button" not in html




# --- the checker's own red tests -------------------------------------------
#
# A checker nobody has seen fail is a checker that might not check anything.
# Every assertion in `_check` gets a shape that must trip it, and every legal
# shape it once rejected gets one that must not.


@pytest.mark.parametrize(
    ("name", "html", "expected"),
    [
        (
            "nested form",
            "<html><body><form><form><button>x</button></form></form></body></html>",
            "nested",
        ),
        (
            "dangling form reference",
            '<html><body><button form="nope">x</button></body></html>',
            "no such form",
        ),
        (
            "duplicate id",
            '<html><body><p id="a"></p><p id="a"></p></body></html>',
            "duplicate id",
        ),
        (
            "label attached to nothing",
            '<html><body><form><label for="ghost">L</label></form></body></html>',
            "points at no control",
        ),
        (
            "control with no name",
            "<html><body><form><textarea></textarea></form></body></html>",
            "submits no value",
        ),
        (
            "control belonging to no form",
            '<html><body><textarea name="x"></textarea></body></html>',
            "outside any form",
        ),
        (
            "crossed tags",
            "<html><body><div><span></div></span></body></html>",
            "closed while",
        ),
        (
            "element never closed",
            "<html><body><div></body></html>",
            "closed while",
        ),
    ],
)
def test_the_checker_catches(name: str, html: str, expected: str) -> None:
    with pytest.raises(AssertionError, match=expected):
        _check(html)


@pytest.mark.parametrize(
    ("name", "html"),
    [
        ("omitted </li>", "<html><body><ul><li>one<li>two</ul></body></html>"),
        ("omitted </p>", "<html><body><p>one<p>two</body></html>"),
        (
            "omitted </option>",
            "<html><body><form><select name=s><option>a<option>b</select>"
            "</form></body></html>",
        ),
        ("omitted </dt> and </dd>", "<html><body><dl><dt>t<dd>d</dl></body></html>"),
        (
            "omitted </td> and </tr>",
            "<html><body><table><tr><td>a<td>b<tr><td>c</table></body></html>",
        ),
    ],
)
def test_the_checker_accepts_legal_omitted_end_tags(name: str, html: str) -> None:
    """HTML5 lets these close themselves and `HTMLParser` does not, so without
    the implicit-close rules the checker rejected valid markup. A checker that
    cries wolf gets deleted, or — worse — gets "satisfied" by rewriting correct
    HTML into something it happens to accept."""
    _check(html)
