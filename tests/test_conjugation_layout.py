"""How a conjugation row lays out on a phone, measured in a real browser.

`tests/test_card_preview.py` asks what the preview *contains*. This file asks
what it *looks like* at a width somebody actually reviews on: it renders the
real recognition answer through the real exporters, opens the exported page in
installed Chrome, and measures the boxes.

The bug this guards: `.conjugation-row` is a two-column flex row, so at 390px
a long English form name ("Short present affirmative (dictionary form)") eats
the row and squeezes a short Japanese value into two or three lines beside it.
The fix is layout only — the label sits above its value on a narrow screen and
the usual spaced columns stay on a desktop.

Nothing here reads Japanese. Every string is a literal fixture value copied
from the report, used as input for geometry and never judged. The long value is
the short one repeated, so nothing can pass this test by refusing to wrap or by
clipping the Japanese: a `nowrap` or a clipped box shows up as overflow.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

pytest.importorskip("genanki")
pytest.importorskip(
    "anki.collection",
    reason="the `anki` library renders these; it is the optional preview extra",
)
sync_api = pytest.importorskip(
    "playwright.sync_api",
    reason="this measures a real browser layout; playwright is the driver",
)

from test_card_preview import (  # noqa: E402
    HANASU,
    _launch_installed_chrome,
    _project,
    _word_deck,
    _words,
)

from japanese_anki.card_preview import (  # noqa: E402
    render_card_preview,
    write_card_preview,
)
from japanese_anki.models import SourceFormsTable  # noqa: E402

#: The exact reported value, and labels long enough to cause the squeeze.
SHORT_VALUE = "いけなかった"
LONG_VALUE = SHORT_VALUE * 14
LONG_LABELS = (
    "Short present affirmative (dictionary form)",
    "Short past negative (nakatta-form)",
    "Potential · short past negative",
)
SHORT_LABEL = "Te-form"
WRAPPING_LABEL = "Repeated value"
CONJUGATIONS = {
    LONG_LABELS[0]: SHORT_VALUE,
    LONG_LABELS[1]: SHORT_VALUE,
    LONG_LABELS[2]: SHORT_VALUE,
    SHORT_LABEL: SHORT_VALUE,
    WRAPPING_LABEL: LONG_VALUE,
}

PHONE = 390
DESKTOP = 1100

#: One measurement pass over the visible answer's conjugation rows. A DOM
#: Range gives the real line boxes of the value's text — a box height alone
#: cannot tell one tall line from two short ones.
MEASURE = """
(answer) => {
  const box = (el) => {
    const r = el.getBoundingClientRect();
    return {left: r.left, right: r.right, top: r.top, bottom: r.bottom,
            width: r.width, height: r.height};
  };
  const lines = (el) => {
    const range = document.createRange();
    range.selectNodeContents(el);
    return Array.from(range.getClientRects())
      .filter((r) => r.width > 0.5 && r.height > 0.5)
      .map((r) => ({left: r.left, right: r.right, top: r.top,
                    bottom: r.bottom, width: r.width}));
  };
  const card = answer.querySelector('.card');
  const doc = document.documentElement;
  return {
    cardBox: box(card),
    cardOverflow: card.scrollWidth - card.clientWidth,
    pageOverflow: doc.scrollWidth - doc.clientWidth,
    rows: Array.from(answer.querySelectorAll('.conjugation-row')).map((row) => {
      const label = row.querySelector('.conjugation-label');
      const value = row.querySelector('.conjugation-value');
      return {
        label: label.textContent,
        value: value.textContent,
        labelBox: box(label),
        valueBox: box(value),
        rowBox: box(row),
        rowOverflow: row.scrollWidth - row.clientWidth,
        valueLines: lines(value),
        labelLines: lines(label),
        valueLineHeight: parseFloat(getComputedStyle(value).lineHeight),
        labelFontSize: parseFloat(getComputedStyle(label).fontSize),
      };
    }),
  };
}
"""


def _measure(page, width: int) -> dict:
    """Open the recognition answer at `width` and measure its rows."""
    page.set_viewport_size({"width": width, "height": 900})
    index = page.evaluate(
        """() => {
          const cards = Array.from(
            document.querySelectorAll('[data-preview-card]'));
          const wanted = cards.findIndex((c) =>
            (c.getAttribute('data-direction') || '').toLowerCase()
              .includes('recognition')
            || (c.getAttribute('data-template') || '').toLowerCase()
              .includes('recognition'));
          return wanted;
        }"""
    )
    assert index >= 0, "the preview drew no recognition card to measure"
    page.select_option("#preview-jump", str(index))
    if page.get_attribute("#preview-flip", "aria-pressed") != "true":
        page.click("#preview-flip")
    answer = page.locator("[data-preview-card]:visible [data-preview-answer]")
    assert answer.count() == 1, "exactly one answer is on screen"
    measured = answer.evaluate(MEASURE)
    labels = [row["label"] for row in measured["rows"]]
    assert labels == list(CONJUGATIONS), f"measured the wrong section: {labels}"
    return measured


def test_a_long_conjugation_label_leaves_its_value_a_full_line_on_a_phone(
    tmp_path: Path,
) -> None:
    """At 390px every supplied value gets its own line and nothing overflows;
    at 1100px the label and value are still two spaced columns.

    Both colour schemes are measured because the card restyles itself under
    `prefers-color-scheme: dark` — the preview carries no separate night-mode
    class, so the browser's `color_scheme` is the whole switch — and a rule
    added inside that media block could change the geometry.
    """
    config = _project(tmp_path)
    record = dataclasses.replace(HANASU, conjugations=dict(CONJUGATIONS))
    _words(tmp_path, [record])
    deck = _word_deck(tmp_path)
    page_path = write_card_preview(render_card_preview(config, deck), tmp_path / "preview.html")

    with sync_api.sync_playwright() as driver:
        browser = _launch_installed_chrome(driver)
        try:
            for scheme in ("light", "dark"):
                context = browser.new_context(
                    viewport={"width": PHONE, "height": 900},
                    color_scheme=scheme,
                )
                page = context.new_page()
                page.goto(page_path.as_uri())
                page.wait_for_selector("#preview-flip")

                phone = _measure(page, PHONE)
                where = f"{scheme} at {PHONE}px"
                assert phone["pageOverflow"] <= 1, f"page scrolls sideways, {where}"
                assert phone["cardOverflow"] <= 1, f"card scrolls sideways, {where}"
                card_right = phone["cardBox"]["right"]
                for row in phone["rows"]:
                    label, value = row["label"], row["value"]
                    assert row["rowOverflow"] <= 1, f"{label!r} row overflows, {where}"
                    for line in row["valueLines"]:
                        assert line["right"] <= card_right + 1, (
                            f"{label!r} value runs past the card, {where}"
                        )
                    if value == SHORT_VALUE:
                        assert len(row["valueLines"]) == 1, (
                            f"{value!r} is broken over {len(row['valueLines'])} "
                            f"lines under {label!r}, {where}"
                        )
                        assert row["valueBox"]["height"] <= row["valueLineHeight"] * 1.5 + 1, (
                            f"{label!r} value is taller than one line, {where}"
                        )
                        # Stacked, not squeezed into a column beside the label.
                        assert row["valueBox"]["top"] >= row["labelBox"]["bottom"] - 1, (
                            f"{label!r} still shares its line with the value, {where}"
                        )
                    else:
                        assert len(row["valueLines"]) >= 2, (
                            f"the long repeated value did not wrap, {where}"
                        )
                    if label == SHORT_LABEL:
                        assert len(row["labelLines"]) == 1, f"{SHORT_LABEL!r} wrapped, {where}"
                        assert row["labelFontSize"] >= 12, (
                            f"{SHORT_LABEL!r} shrank to {row['labelFontSize']}px, {where}"
                        )

                desktop = _measure(page, DESKTOP)
                where = f"{scheme} at {DESKTOP}px"
                assert desktop["pageOverflow"] <= 1, f"page scrolls sideways, {where}"
                assert desktop["cardOverflow"] <= 1, f"card scrolls sideways, {where}"
                for row in desktop["rows"]:
                    label = row["label"]
                    assert row["rowOverflow"] <= 1, f"{label!r} row overflows, {where}"
                    gap = row["valueBox"]["left"] - row["labelBox"]["right"]
                    assert gap >= 8, f"{label!r} lost its column separation ({gap}px), {where}"
                    assert (
                        row["labelBox"]["top"] < row["valueBox"]["bottom"]
                        and row["valueBox"]["top"] < row["labelBox"]["bottom"]
                    ), f"{label!r} and its value left the same row, {where}"
                    if row["value"] == SHORT_VALUE:
                        assert len(row["valueLines"]) == 1, (
                            f"{label!r} value wrapped on a desktop, {where}"
                        )

                context.close()
        finally:
            browser.close()


#: A printed table with one filled cell, one the source printed *blank*, and a
#: declared column this row has no cell for at all. The three states §4.4
#: distinguishes, measured rather than asserted from the HTML.
SOURCE_FORM_COLUMNS = [
    {"id": "col-9f3a71", "label": LONG_LABELS[0]},
    {"id": "col-2b8d04", "label": "Polite"},
    {"id": "col-7ac511", "label": "Negative"},
]
SOURCE_FORM_CELLS = {"col-9f3a71": SHORT_VALUE, "col-2b8d04": ""}
SOURCE_FORMS_TABLE = SourceFormsTable.from_dict(
    {"columns": SOURCE_FORM_COLUMNS, "cells": SOURCE_FORM_CELLS}
)


def _measure_source_forms(page, width: int) -> dict:
    """The same measurement pass, over a card whose rows come from the table."""
    page.set_viewport_size({"width": width, "height": 900})
    index = page.evaluate(
        """() => {
          const cards = Array.from(
            document.querySelectorAll('[data-preview-card]'));
          return cards.findIndex((c) =>
            (c.getAttribute('data-direction') || '').toLowerCase()
              .includes('recognition')
            || (c.getAttribute('data-template') || '').toLowerCase()
              .includes('recognition'));
        }"""
    )
    assert index >= 0, "the preview drew no recognition card to measure"
    page.select_option("#preview-jump", str(index))
    if page.get_attribute("#preview-flip", "aria-pressed") != "true":
        page.click("#preview-flip")
    answer = page.locator("[data-preview-card]:visible [data-preview-answer]")
    assert answer.count() == 1, "exactly one answer is on screen"
    return answer.evaluate(MEASURE)


def test_a_blank_printed_cell_keeps_its_own_row_on_a_phone(tmp_path: Path) -> None:
    """A printed blank is study content, so it has to be visible as a row.

    An empty `.conjugation-value` in a two-column flex row is exactly the shape
    this file exists to catch: without its own row a reader cannot tell that
    the source printed the column and left it empty. The absent third column
    draws nothing at all, which is the different fact beside it.

    Mutant: skip a row whose value is empty in
    `exporters/anki._conjugation_rows_html`, or draw a row for a declared
    column this record has no cell for.
    """
    config = _project(tmp_path)
    record = dataclasses.replace(
        HANASU,
        conjugations={},
        source_forms=SOURCE_FORMS_TABLE,
    )
    _words(tmp_path, [record])
    deck = _word_deck(tmp_path)
    page_path = write_card_preview(
        render_card_preview(config, deck), tmp_path / "preview.html"
    )

    with sync_api.sync_playwright() as driver:
        browser = _launch_installed_chrome(driver)
        try:
            context = browser.new_context(
                viewport={"width": PHONE, "height": 900}, color_scheme="light"
            )
            page = context.new_page()
            page.goto(page_path.as_uri())
            page.wait_for_selector("#preview-flip")

            measured = _measure_source_forms(page, PHONE)

            labels = [row["label"] for row in measured["rows"]]
            # The filled column and the printed blank, in printed order. The
            # column this row has no cell for draws no row.
            assert labels == [LONG_LABELS[0], "Polite"]
            assert measured["pageOverflow"] <= 1
            assert measured["cardOverflow"] <= 1
            blank = next(row for row in measured["rows"] if row["label"] == "Polite")
            filled = next(
                row for row in measured["rows"] if row["label"] == LONG_LABELS[0]
            )
            assert blank["value"] == ""
            # Its own row, on its own line, below the filled one.
            assert blank["rowBox"]["height"] > 0
            assert blank["rowBox"]["top"] >= filled["rowBox"]["bottom"] - 1
            assert len(blank["labelLines"]) >= 1
            assert blank["labelBox"]["height"] > 0
            assert blank["rowOverflow"] <= 1

            context.close()
        finally:
            browser.close()
