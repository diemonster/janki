"""jpdb's published reading percentages, and the words its pages bind to them.

Every page here is written for the test: three synthetic fixtures shaped like
the tables a jpdb kanji page and a jpdb reading page print, and small inline
strings for everything else. The point is the handful of structures this
adapter accepts and refuses, not an inventory of possible HTML, so no captured
page is stored in the repository. Nothing reaches the network — every fetch
drives the transport seam.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from japanese_anki import jpdb_kanji
from japanese_anki.jpdb_kanji import (
    BoundExample,
    CharacterReadings,
    KanjiReadingsError,
    ReadingGroup,
    ReadingUsage,
    fetch_character,
    load_readings,
    parse_kanji_page,
    parse_reading_page,
    save_readings,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "jpdb_readings"

NOMINAL = (FIXTURES / "nominal.html").read_bytes()
BOUNDED = (FIXTURES / "bounded.html").read_bytes()
RI_DETAIL = (FIXTURES / "reading-detail.html").read_bytes()

RI_URL = "https://jpdb.io/kanji/%E7%90%86"
RI_READING_URL = "https://jpdb.io/kanji-reading/%E7%90%86/%E3%82%8A"
WA_READING_URL = "https://jpdb.io/kanji-reading/%E7%90%86/%E3%82%8F"

# The readings the nominal fixture prints, in its source order.
NOMINAL_GROUPS = (
    ReadingGroup(
        source_class="kanji-reading-list-common",
        readings=(
            ReadingUsage("り", "/kanji-reading/理/り", "(84%)", 84, False),
            ReadingUsage("わ", "/kanji-reading/理/わ", "(14%)", 14, False),
        ),
    ),
    ReadingGroup(
        source_class="kanji-reading-list",
        readings=(
            ReadingUsage("ことわり", "/kanji-reading/理/ことわり", None, None, None),
            ReadingUsage("め", "/kanji-reading/理/め", None, None, None),
            ReadingUsage("ことわ", "/kanji-reading/理/ことわ", None, None, None),
        ),
    ),
)


def _cell(inner: str, source_class: str = "kanji-reading-list-common") -> str:
    return (
        '<table class="cross-table"><tr><td>Readings</td>'
        f'<td class="{source_class}">{inner}</td></tr></table>'
    )


def _reading_page(character: str = "理", reading: str = "り", entries: str = "") -> str:
    """A reading page with the identity table this parser requires."""
    return (
        '<html><body><table class="cross-table">'
        f"<tr><td>Kanji</td><td>{character}</td></tr>"
        f"<tr><td>Reading</td><td>{reading}</td></tr>"
        "<tr><td>Frequency</td><td>84%</td></tr></table>"
        '<div class="subsection-used-in"><h6>Used in</h6>'
        f'<div class="subsection">{entries}</div></div></body></html>'
    )


def _used_in(href: str, ruby: str, english: str) -> str:
    return (
        '<div class="used-in"><div class="jp">'
        f'<a class="plain" href="{href}">{ruby}</a></div>'
        f'<div class="en">{english}</div></div>'
    )


WA_PAGE = _reading_page(
    reading="わ",
    entries=_used_in(
        "/vocabulary/1550140/理由/わけ#a",
        '<span class="highlight"><ruby>理<rt>わ</rt></ruby></span><ruby>由<rt>け</rt></ruby>',
        " reason;  pretext;  motive",
    ),
).encode("utf-8")


# --- what a reading table says -----------------------------------------------


def test_nominal_page_reports_both_cells_in_source_order() -> None:
    assert parse_kanji_page(NOMINAL.decode("utf-8")) == NOMINAL_GROUPS


def test_a_kanji_page_on_its_own_binds_no_examples() -> None:
    """The page lists vocabulary, but nothing on it says which reading a listed
    word uses — so nothing on it can bind one. Examples arrive only from the
    reading page each anchor links to."""
    groups = parse_kanji_page(NOMINAL.decode("utf-8"))

    assert [r.examples for group in groups for r in group.readings] == [()] * 5
    assert all(
        r.detail_source_url is None for group in groups for r in group.readings
    ), "and nothing claims a detail page was consulted"


def test_links_outside_the_reading_cells_are_not_reported() -> None:
    # The fixture's two placeholder links stand in for the vocabulary and
    # example links the real page prints elsewhere: outside a reading cell, so
    # not tied to a reading, so not reported.
    page = NOMINAL.decode("utf-8")
    hrefs = [r.href for group in parse_kanji_page(page) for r in group.readings]
    assert len(hrefs) == 5
    assert all(href.startswith("/kanji-reading/") for href in hrefs)
    assert "/placeholder/not-a-reading" in page  # in an unrelated row of the same table
    assert "/placeholder/also-not-a-reading" in page  # and below the table


def test_unrelated_info_rows_do_not_refuse() -> None:
    # This row's cell holds a div and a bare link, which would be malformed
    # inside a reading cell. Only the target cells are judged.
    page = (
        '<table class="cross-table"><tr><td>Type</td><td class="space-between">'
        '<div>Kyōiku (2nd grade)&nbsp;</div><a href="/x" class="what-is-this">?</a></td></tr>'
        "<tr><td>Heisig</td><td>283</td></tr>"
        '<tr><td>Readings</td><td class="kanji-reading-list-common">'
        '<div><a href="/kanji-reading/理/り">り</a><div>(84%)</div></div>'
        "</td></tr></table>"
    )

    assert parse_kanji_page(page) == (
        ReadingGroup(
            source_class="kanji-reading-list-common",
            readings=(ReadingUsage("り", "/kanji-reading/理/り", "(84%)", 84, False),),
        ),
    )


def test_bounded_and_missing_percentages_are_reported_as_such() -> None:
    common, rare = parse_kanji_page(BOUNDED.decode("utf-8"))
    printed = [(r.percent_text, r.percent, r.percent_less_than) for r in common.readings]

    assert printed == [
        ("(<1%)", 1, True),  # the entity is decoded; the bound is not flattened to 1%
        ("(3%)", 3, False),
        (None, None, None),  # a common reading printed without a percentage
    ]
    assert rare.source_class == "kanji-reading-list"
    assert [(r.percent_text, r.percent) for r in rare.readings] == [(None, None)]


@pytest.mark.parametrize(
    ("printed", "expected_text", "expected_percent", "expected_bound"),
    [
        ("(84%)", "(84%)", 84, False),
        ("84%", "84%", 84, False),
        ("(&lt;1%)", "(<1%)", 1, True),
        ("&lt;1%", "<1%", 1, True),
        ("(less than 1%)", "(less than 1%)", 1, True),
    ],
)
def test_accepted_percentage_forms(
    printed: str, expected_text: str, expected_percent: int, expected_bound: bool
) -> None:
    page = _cell(f'<div><a href="/kanji-reading/理/り">り</a><div>{printed}</div></div>')

    (reading,) = parse_kanji_page(page)[0].readings

    assert reading.percent_text == expected_text
    assert (reading.percent, reading.percent_less_than) == (expected_percent, expected_bound)


# --- refusals on a kanji page -------------------------------------------------


@pytest.mark.parametrize(
    "printed",
    ["(84.5%)", "(84)", "()", "(84%", "84 %", "(approx 84%)", "(-3%)", "(８４%)"],
)
def test_unknown_percentage_text_refuses(printed: str) -> None:
    page = _cell(f'<div><a href="/kanji-reading/理/り">り</a><div>{printed}</div></div>')

    with pytest.raises(KanjiReadingsError, match="percentage"):
        parse_kanji_page(page)


@pytest.mark.parametrize(
    "inner",
    [
        "<div></div>",  # wrapper with no link
        '<div><a href="/kanji-reading/理/り">り</a>',  # unclosed wrapper
        '<div><a href="/kanji-reading/理/り">り</a></div>および',  # stray text in the cell
        '<div><a href="/kanji-reading/理/り">り</a>+<div>(84%)</div></div>',  # stray text
        '<div><a href="/kanji-reading/理/り">り</a><a href="/x">わ</a></div>',  # two links
        '<div><a href="/kanji-reading/理/り"><ruby>理<rt>り</rt></ruby></a></div>',  # markup
        '<span><a href="/kanji-reading/理/り">り</a></span>',  # not a wrapper div
        "<div><a>り</a></div>",  # link with no href
        '<div><a href="/x">り</a><div>(84%)</div><div>(14%)</div></div>',  # two percentages
        "",  # a target cell holding nothing
    ],
)
def test_malformed_reading_structure_refuses(inner: str) -> None:
    with pytest.raises(KanjiReadingsError):
        parse_kanji_page(_cell(inner))


def test_page_ending_inside_a_later_reading_cell_refuses() -> None:
    # Nothing inside either cell is malformed: the first cell is complete and
    # the second cell's wrapper opened and closed cleanly. Only the page's end,
    # with that second cell still open, says the list was cut off — so
    # reporting the first cell here would drop the second silently.
    page = (
        '<table class="cross-table">'
        '<tr><td>Readings</td><td class="kanji-reading-list-common">'
        '<div><a href="/kanji-reading/理/り">り</a><div>(84%)</div></div></td></tr>'
        '<tr><td>Readings</td><td class="kanji-reading-list">'
        '<div><a href="/kanji-reading/理/ことわり">ことわり</a></div>'
    )

    with pytest.raises(KanjiReadingsError, match="unclosed reading cell"):
        parse_kanji_page(page)


def test_page_without_a_reading_cell_refuses() -> None:
    page = '<table class="cross-table"><tr><td>Heisig</td><td>283</td></tr></table>'

    with pytest.raises(KanjiReadingsError, match="reading cell"):
        parse_kanji_page(page)


# --- what a reading page binds ------------------------------------------------


def test_a_reading_page_binds_its_own_words_in_its_own_order() -> None:
    page = parse_reading_page(RI_DETAIL.decode("utf-8"))

    assert (page.character, page.reading) == ("理", "り")
    assert [e.written for e in page.examples] == ["理由", "理論物理学", "無理やり"]


def test_the_expression_and_whole_word_reading_come_from_the_url() -> None:
    """jpdb states both in the link it printed, so neither is read out of the
    Japanese. 理由/わけ and 理由/りゆう are the same spelling under two readings,
    and the URL is what tells them apart."""
    page = parse_reading_page(WA_PAGE.decode("utf-8"))

    (example,) = page.examples
    assert (example.written, example.pronounced) == ("理由", "わけ")
    assert example.source_url.startswith("https://jpdb.io/vocabulary/1550140/")


def test_the_two_spellings_of_one_used_in_path_name_one_resource() -> None:
    """jpdb writes a path either with the characters in it or already escaped.
    `%E7%90%86` *is* 理, so escaping its per-cent sign again names a resource
    that does not exist. Both spellings resolve to the one URL, and the fields
    read out of the link are the same word either way."""
    escaped = parse_reading_page(
        _reading_page(
            entries=_used_in(
                "/vocabulary/1550140/%E7%90%86%E7%94%B1/%E3%82%8A%E3%82%86%E3%81%86#a",
                "<ruby>理由<rt>りゆう</rt></ruby>",
                "reason",
            )
        )
    )
    literal = parse_reading_page(
        _reading_page(
            entries=_used_in(
                "/vocabulary/1550140/理由/りゆう#a",
                "<ruby>理由<rt>りゆう</rt></ruby>",
                "reason",
            )
        )
    )

    (example,) = escaped.examples
    assert (example.written, example.pronounced) == ("理由", "りゆう")
    assert example.source_url == (
        "https://jpdb.io/vocabulary/1550140/%E7%90%86%E7%94%B1/%E3%82%8A%E3%82%86%E3%81%86#a"
    )
    assert literal.examples[0].source_url == example.source_url


def test_one_vocabulary_id_under_two_readings_stays_two_words() -> None:
    """1550140 is 理由 read りゆう and 理由 read わけ. The id is not an identity
    this adapter deduplicates on — the reading is part of what was bound."""
    page = parse_reading_page(
        _reading_page(
            entries=_used_in(
                "/vocabulary/1550140/理由/りゆう#a", "<ruby>理由<rt>りゆう</rt></ruby>", "reason"
            )
            + _used_in(
                "/vocabulary/1550140/理由/わけ#a", "<ruby>理由<rt>わけ</rt></ruby>", "reason"
            )
        )
    )

    assert [e.pronounced for e in page.examples] == ["りゆう", "わけ"]


def test_furigana_is_the_supplied_ruby_and_only_that() -> None:
    """理論物理学 carries 理[り] twice and 無理やり carries a <ruby> with no <rt>
    at all. Both come through as the page wrote them: repetition preserved,
    an unannotated run left as plain text, and no alignment of the expression
    against the whole-word reading anywhere."""
    page = parse_reading_page(RI_DETAIL.decode("utf-8"))

    furigana = {e.written: e.furigana for e in page.examples}
    assert furigana["理論物理学"] == "理[り] 論[ろん] 物[ぶつ] 理[り] 学[がく]"
    assert furigana["無理やり"] == "無[む] 理[り]やり"
    assert furigana["理由"] == "理[り] 由[ゆう]"


def test_an_rt_reading_never_lands_in_the_written_form() -> None:
    page = parse_reading_page(
        _reading_page(
            entries=_used_in(
                "/vocabulary/1/料理/りょうり#a",
                "<ruby>料<rt>りょう</rt></ruby><ruby>理<rt>り</rt></ruby>",
                "cooking",
            )
        )
    )

    (example,) = page.examples
    assert example.furigana == "料[りょう] 理[り]"
    assert "りょう" not in example.written


def test_an_rp_fallback_is_ignored_rather_than_read_as_text() -> None:
    page = parse_reading_page(
        _reading_page(
            entries=_used_in(
                "/vocabulary/1/料理/りょうり#a",
                "<ruby>料<rp>(</rp><rt>りょう</rt><rp>)</rp></ruby>"
                "<ruby>理<rp>(</rp><rt>り</rt><rp>)</rp></ruby>",
                "cooking",
            )
        )
    )

    (example,) = page.examples
    assert example.furigana == "料[りょう] 理[り]"


def test_the_english_is_decoded_and_left_whole() -> None:
    """The gloss is the page's entity-decoded text. Senses are not cut down to
    one, reordered, or rewritten."""
    page = parse_reading_page(RI_DETAIL.decode("utf-8"))

    glosses = {e.written: e.gloss for e in page.examples}
    assert glosses["無理やり"] == "forcibly;  against one's will"
    assert glosses["理由"] == "reason;  pretext;  motive"


@pytest.mark.parametrize(
    "href",
    [
        "/vocabulary/1550140/理由#a",  # no reading field
        "/vocabulary/1550140/理由/りゆう/extra#a",  # a field too many
        "/word/1550140/理由/りゆう#a",  # not the vocabulary shape
        "/vocabulary//理由/りゆう#a",  # no id
    ],
)
def test_a_used_in_link_that_is_not_the_vocabulary_shape_refuses(href: str) -> None:
    with pytest.raises(KanjiReadingsError, match="used-in link"):
        parse_reading_page(
            _reading_page(entries=_used_in(href, "<ruby>理由<rt>りゆう</rt></ruby>", "reason"))
        )


def test_an_off_site_used_in_link_is_refused_rather_than_followed() -> None:
    with pytest.raises(KanjiReadingsError, match="not a jpdb site path"):
        parse_reading_page(
            _reading_page(
                entries=_used_in(
                    "https://example.invalid/vocabulary/1/理由/りゆう",
                    "<ruby>理由<rt>りゆう</rt></ruby>",
                    "reason",
                )
            )
        )


@pytest.mark.parametrize(
    "ruby",
    [
        "<ruby>理<rt>り</rt></ruby>由",  # text outside a ruby
        "<ruby>理<rt>り</rt><rt>わ</rt></ruby>",  # two readings over one base
        "<ruby><rt>り</rt></ruby>",  # a reading over nothing
        "<ruby>理<rt>り</rt>",  # unclosed ruby
        "<b>理由</b>",  # markup this parser does not know
        "",  # a link with no ruby at all
    ],
)
def test_malformed_used_in_markup_refuses(ruby: str) -> None:
    with pytest.raises(KanjiReadingsError):
        parse_reading_page(
            _reading_page(entries=_used_in("/vocabulary/1/理由/りゆう#a", ruby, "reason"))
        )


def test_a_page_that_does_not_state_its_identity_refuses() -> None:
    page = (
        '<html><body><table class="cross-table">'
        "<tr><td>Frequency</td><td>84%</td></tr></table></body></html>"
    )

    with pytest.raises(KanjiReadingsError, match="Kanji and Reading"):
        parse_reading_page(page)


def test_a_reading_page_listing_nothing_is_not_a_refusal() -> None:
    """Zero bound words is a fact about that reading, not a broken page."""
    page = parse_reading_page(_reading_page(entries=""))

    assert (page.character, page.reading) == ("理", "り")
    assert page.examples == ()


# --- the raw cache, and what is requested -------------------------------------


def _manifest(character: str, url: str, data: bytes) -> dict[str, object]:
    """A manifest written by the test, so cache reads do not lean on the module."""
    return {
        "schema_version": 1,
        "character": character,
        "source_url": url,
        "fetched_at_utc": "2026-09-07T00:00:00Z",
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _main_paths(cache: Path, character: str) -> tuple[Path, Path]:
    stem = f"{ord(character):x}"  # 理 -> 7406
    return cache / f"{stem}.html", cache / f"{stem}.json"


def _detail_paths(cache: Path, url: str) -> tuple[Path, Path]:
    stem = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return cache / f"{stem}.html", cache / f"{stem}.json"


def _seed(
    paths: tuple[Path, Path],
    character: str,
    url: str,
    data: bytes,
    manifest: dict[str, object] | None = None,
) -> tuple[Path, Path]:
    html_path, manifest_path = paths
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_bytes(data)
    manifest_path.write_text(
        json.dumps(_manifest(character, url, data) if manifest is None else manifest),
        encoding="utf-8",
    )
    return paths


def _seed_ri(cache: Path, *, details: bool = True) -> None:
    _seed(_main_paths(cache, "理"), "理", RI_URL, NOMINAL)
    if details:
        _seed(_detail_paths(cache, RI_READING_URL), "理", RI_READING_URL, RI_DETAIL)
        _seed(_detail_paths(cache, WA_READING_URL), "理", WA_READING_URL, WA_PAGE)


def _transport(pages: dict[str, bytes] | None = None, error: Exception | None = None):
    """A transport that records what it was asked for."""
    calls: list[str] = []

    def send(url: str) -> bytes:
        calls.append(url)
        if error is not None:
            raise error
        if pages is None or url not in pages:
            raise AssertionError(f"unexpected request for {url}")
        return pages[url]

    return send, calls


def _snapshot(cache: Path) -> dict[str, bytes]:
    return {str(path.relative_to(cache)): path.read_bytes() for path in cache.rglob("*")}


def test_a_fully_cached_character_makes_no_request(tmp_path: Path) -> None:
    cache = tmp_path / "jpdb"
    _seed_ri(cache)
    send, calls = _transport()

    readings = fetch_character("理", html_cache=cache, transport=send)

    assert calls == []
    assert readings.source_url == RI_URL
    assert readings.sha256 == hashlib.sha256(NOMINAL).hexdigest()
    assert readings.fetched_at_utc == "2026-09-07T00:00:00Z", "the saved provenance"
    assert (readings.corpus_scope, readings.denominator) == (None, None)


def test_only_the_missing_pages_are_requested(tmp_path: Path) -> None:
    cache = tmp_path / "jpdb"
    _seed(_main_paths(cache, "理"), "理", RI_URL, NOMINAL)
    _seed(_detail_paths(cache, RI_READING_URL), "理", RI_READING_URL, RI_DETAIL)
    send, calls = _transport({WA_READING_URL: WA_PAGE})

    fetch_character("理", html_cache=cache, transport=send)

    assert calls == [WA_READING_URL], "the kanji page and り's page were already here"


def test_an_escaped_reading_link_is_requested_as_the_page_printed_it(
    tmp_path: Path,
) -> None:
    """The same two spellings on the kanji page, where getting it wrong costs a
    request for a page that is not there. The href kept in the facts stays the
    source's own string; only the URL janki asks for is resolved."""
    cache = tmp_path / "jpdb"
    printed = "/kanji-reading/%E7%90%86/%E3%82%8A"
    page = _cell(f'<div><a href="{printed}">り</a><div>(84%)</div></div>')
    _seed(_main_paths(cache, "理"), "理", RI_URL, page.encode("utf-8"))
    send, calls = _transport({RI_READING_URL: RI_DETAIL})

    readings = fetch_character("理", html_cache=cache, transport=send)

    assert calls == [RI_READING_URL]
    (usage,) = readings.groups[0].readings
    assert usage.href == printed, "the source's own href, carried through verbatim"
    assert usage.detail_source_url == RI_READING_URL


def test_a_quantified_reading_carries_its_pages_words_and_provenance(tmp_path: Path) -> None:
    cache = tmp_path / "jpdb"
    _seed_ri(cache)
    send, _ = _transport()

    readings = fetch_character("理", html_cache=cache, transport=send)

    common = readings.groups[0]
    assert common.source_class == "kanji-reading-list-common"
    ri = common.readings[0]
    assert (ri.label, ri.percent_text, ri.percent) == ("り", "(84%)", 84)
    assert [e.written for e in ri.examples] == ["理由", "理論物理学"], "the page's first two"
    assert ri.detail_source_url == RI_READING_URL
    assert ri.detail_sha256 == hashlib.sha256(RI_DETAIL).hexdigest()
    assert ri.detail_fetched_at_utc == "2026-09-07T00:00:00Z"


def test_the_kanji_pages_own_percentage_survives_the_detail_page(tmp_path: Path) -> None:
    """The reading page prints the same figure rounded its own way. It is
    provenance for the words, never a correction to the quantity."""
    cache = tmp_path / "jpdb"
    _seed(_main_paths(cache, "理"), "理", RI_URL, NOMINAL)
    detail = _reading_page(reading="わ", entries="").replace(
        "<td>84%</td>", "<td>99%</td>"
    )
    _seed(_detail_paths(cache, RI_READING_URL), "理", RI_READING_URL, RI_DETAIL)
    _seed(_detail_paths(cache, WA_READING_URL), "理", WA_READING_URL, detail.encode("utf-8"))
    send, _ = _transport()

    readings = fetch_character("理", html_cache=cache, transport=send)

    wa = readings.groups[0].readings[1]
    assert (wa.percent_text, wa.percent) == ("(14%)", 14)


def test_a_quantified_reading_with_no_bound_word_keeps_its_figure(tmp_path: Path) -> None:
    """"84% of uses, and no example on file" is the fact. Dropping the figure
    for want of an example would report a different one."""
    cache = tmp_path / "jpdb"
    _seed(_main_paths(cache, "理"), "理", RI_URL, NOMINAL)
    _seed(
        _detail_paths(cache, RI_READING_URL),
        "理",
        RI_READING_URL,
        _reading_page(entries="").encode("utf-8"),
    )
    _seed(_detail_paths(cache, WA_READING_URL), "理", WA_READING_URL, WA_PAGE)
    send, _ = _transport()

    ri = fetch_character("理", html_cache=cache, transport=send).groups[0].readings[0]

    assert ri.examples == ()
    assert (ri.percent_text, ri.percent) == ("(84%)", 84)
    assert ri.detail_source_url == RI_READING_URL, "asked, and jpdb listed nothing"


def test_the_rare_cell_comes_through_with_nothing_bound_to_it(tmp_path: Path) -> None:
    """Its readings are reported in source order, under the class jpdb gave the
    cell, with no quantity and no words — which is all that cell says."""
    cache = tmp_path / "jpdb"
    _seed_ri(cache)
    send, calls = _transport()

    readings = fetch_character("理", html_cache=cache, transport=send)

    assert calls == []
    rare = readings.groups[1]
    assert rare.source_class == "kanji-reading-list"
    assert [r.label for r in rare.readings] == ["ことわり", "め", "ことわ"]
    assert all(r.percent_text is None for r in rare.readings)
    assert all(r.examples == () and r.detail_source_url is None for r in rare.readings)


def test_a_common_reading_with_no_percentage_is_never_requested(tmp_path: Path) -> None:
    """jpdb prints some readings in the common cell with no figure beside them.
    Nothing quantifies those, so there is nothing to bind words to and no reason
    to spend a request finding out."""
    cache = tmp_path / "jpdb"
    page = _cell(
        '<div><a href="/kanji-reading/理/り">り</a><div>(84%)</div></div>'
        '<div><a href="/kanji-reading/理/わ">わ</a></div>'
    ).encode("utf-8")
    _seed(_main_paths(cache, "理"), "理", RI_URL, page)
    _seed(_detail_paths(cache, RI_READING_URL), "理", RI_READING_URL, RI_DETAIL)
    send, calls = _transport()

    readings = fetch_character("理", html_cache=cache, transport=send)

    assert calls == [], "り's page was cached, and わ's was never wanted"
    wa = readings.groups[0].readings[1]
    assert (wa.percent_text, wa.percent, wa.percent_less_than) == (None, None, None)
    assert wa.examples == () and wa.detail_source_url is None


def test_a_percentage_outside_the_common_cell_is_still_not_followed(
    tmp_path: Path,
) -> None:
    """Both halves of the rule carry weight: the cell jpdb marked common *and* a
    figure printed beside the reading. A quantity in the other cell is not the
    evidence link this adapter follows."""
    cache = tmp_path / "jpdb"
    page = _cell(
        '<div><a href="/kanji-reading/理/ことわり">ことわり</a><div>(2%)</div></div>',
        source_class="kanji-reading-list",
    ).encode("utf-8")
    _seed(_main_paths(cache, "理"), "理", RI_URL, page)
    send, calls = _transport()

    readings = fetch_character("理", html_cache=cache, transport=send)

    assert calls == []
    (reading,) = readings.groups[0].readings
    assert (reading.percent_text, reading.percent) == ("(2%)", 2), "reported as printed"
    assert reading.examples == () and reading.detail_source_url is None


def test_a_detail_page_for_another_reading_refuses(tmp_path: Path) -> None:
    """The page has to say it is this character and this reading, character for
    character. Nothing here decides that two spellings mean the same sound."""
    cache = tmp_path / "jpdb"
    _seed(_main_paths(cache, "理"), "理", RI_URL, NOMINAL)
    send, _ = _transport({RI_READING_URL: _reading_page(reading="わ").encode("utf-8")})

    with pytest.raises(KanjiReadingsError, match="is the page for"):
        fetch_character("理", html_cache=cache, transport=send)

    assert not _detail_paths(cache, RI_READING_URL)[0].exists(), "and it is not cached"


def test_a_detail_page_for_another_character_refuses(tmp_path: Path) -> None:
    cache = tmp_path / "jpdb"
    _seed(_main_paths(cache, "理"), "理", RI_URL, NOMINAL)
    send, _ = _transport({RI_READING_URL: _reading_page(character="王").encode("utf-8")})

    with pytest.raises(KanjiReadingsError, match="is the page for"):
        fetch_character("理", html_cache=cache, transport=send)


def test_a_missing_kanji_page_is_fetched_and_stored_with_its_manifest(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "jpdb"
    send, calls = _transport(
        {RI_URL: NOMINAL, RI_READING_URL: RI_DETAIL, WA_READING_URL: WA_PAGE}
    )

    readings = fetch_character("理", html_cache=cache, transport=send)

    assert calls == [RI_URL, RI_READING_URL, WA_READING_URL]
    html_path, manifest_path = _main_paths(cache, "理")
    assert html_path.read_bytes() == NOMINAL
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(manifest) == {
        "schema_version",
        "character",
        "source_url",
        "fetched_at_utc",
        "sha256",
    }
    assert manifest["character"] == "理"
    assert manifest["source_url"] == RI_URL
    assert manifest["sha256"] == hashlib.sha256(NOMINAL).hexdigest()
    assert manifest["fetched_at_utc"].endswith("Z")
    assert readings.fetched_at_utc == manifest["fetched_at_utc"]


def test_refresh_requests_this_character_again_and_replaces_its_entries(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "jpdb"
    _seed_ri(cache)
    replacement = NOMINAL.replace(b"(84%)", b"(48%)")
    send, calls = _transport(
        {RI_URL: replacement, RI_READING_URL: RI_DETAIL, WA_READING_URL: WA_PAGE}
    )

    readings = fetch_character("理", html_cache=cache, refresh=True, transport=send)

    assert calls == [RI_URL, RI_READING_URL, WA_READING_URL]
    assert readings.groups[0].readings[0].percent_text == "(48%)"
    html_path, manifest_path = _main_paths(cache, "理")
    assert html_path.read_bytes() == replacement
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["sha256"] == hashlib.sha256(replacement).hexdigest()


def test_refresh_leaves_every_other_character_alone(tmp_path: Path) -> None:
    """One character is refreshed by asking for that character. There is no
    sweep, so a second character's saved bytes cannot be spent by this one."""
    cache = tmp_path / "jpdb"
    _seed_ri(cache)
    other_url = "https://jpdb.io/kanji/%E7%8E%8B"
    other = _seed(_main_paths(cache, "王"), "王", other_url, BOUNDED)
    before = (other[0].read_bytes(), other[1].read_bytes())
    send, calls = _transport(
        {RI_URL: NOMINAL, RI_READING_URL: RI_DETAIL, WA_READING_URL: WA_PAGE}
    )

    fetch_character("理", html_cache=cache, refresh=True, transport=send)

    assert other_url not in calls
    assert (other[0].read_bytes(), other[1].read_bytes()) == before


def test_a_failed_request_is_not_retried_and_keeps_the_cache(tmp_path: Path) -> None:
    cache = tmp_path / "jpdb"
    _seed_ri(cache)
    before = _snapshot(cache)
    send, calls = _transport(error=OSError("connection reset"))

    with pytest.raises(KanjiReadingsError, match="failed"):
        fetch_character("理", html_cache=cache, refresh=True, transport=send)

    assert len(calls) == 1, "one GET, no retry"
    assert _snapshot(cache) == before


def test_an_unparseable_response_is_not_cached(tmp_path: Path) -> None:
    cache = tmp_path / "jpdb"
    send, calls = _transport({RI_URL: b"<html><body>service unavailable</body></html>"})

    with pytest.raises(KanjiReadingsError, match="reading cell"):
        fetch_character("理", html_cache=cache, transport=send)

    assert len(calls) == 1
    assert not cache.exists() or _snapshot(cache) == {}


def test_a_failed_detail_request_leaves_the_kanji_page_cached_and_unchanged(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "jpdb"
    _seed(_main_paths(cache, "理"), "理", RI_URL, NOMINAL)
    before = _snapshot(cache)
    send, calls = _transport({RI_READING_URL: b"<html>nope</html>"})

    with pytest.raises(KanjiReadingsError):
        fetch_character("理", html_cache=cache, transport=send)

    assert calls == [RI_READING_URL], "the cached kanji page was not re-requested"
    assert _snapshot(cache) == before


def test_a_cache_entry_whose_hash_does_not_match_refuses(tmp_path: Path) -> None:
    cache = tmp_path / "jpdb"
    _seed_ri(cache)
    html_path, _ = _main_paths(cache, "理")
    html_path.write_bytes(NOMINAL.replace(b"(84%)", b"(48%)"))  # parseable, not the bytes
    send, calls = _transport()

    with pytest.raises(KanjiReadingsError, match="sha256"):
        fetch_character("理", html_cache=cache, transport=send)

    assert calls == [], "a bad entry is refused, not silently re-fetched over"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("character", "王"),
        ("source_url", "https://jpdb.io/kanji/%E7%8E%8B"),
        ("schema_version", 2),
        ("fetched_at_utc", ""),
    ],
)
def test_a_manifest_that_does_not_match_the_request_refuses(
    tmp_path: Path, field: str, value: object
) -> None:
    cache = tmp_path / "jpdb"
    manifest = _manifest("理", RI_URL, NOMINAL)
    manifest[field] = value
    _seed(_main_paths(cache, "理"), "理", RI_URL, NOMINAL, manifest)
    send, calls = _transport()

    with pytest.raises(KanjiReadingsError, match=field):
        fetch_character("理", html_cache=cache, transport=send)

    assert calls == []


def test_a_cached_page_that_is_not_valid_utf8_refuses(tmp_path: Path) -> None:
    cache = tmp_path / "jpdb"
    data = NOMINAL.replace("り".encode(), b"\xe3\x81")  # truncated code point
    _seed(_main_paths(cache, "理"), "理", RI_URL, data)  # the hash matches these bytes
    send, calls = _transport()

    with pytest.raises(KanjiReadingsError, match="not valid UTF-8"):
        fetch_character("理", html_cache=cache, transport=send)

    assert calls == []


def test_an_incomplete_cache_entry_refuses(tmp_path: Path) -> None:
    cache = tmp_path / "jpdb"
    _seed_ri(cache)
    _main_paths(cache, "理")[1].unlink()
    send, calls = _transport()

    with pytest.raises(KanjiReadingsError, match="incomplete cache entry"):
        fetch_character("理", html_cache=cache, transport=send)

    assert calls == []


@pytest.mark.parametrize("character", ["理由", "", "あ", "A"])
def test_a_fetch_target_is_one_kanji(tmp_path: Path, character: str) -> None:
    send, calls = _transport()

    with pytest.raises(KanjiReadingsError, match="single kanji"):
        fetch_character(character, html_cache=tmp_path, transport=send)

    assert calls == []


def test_a_fetch_writes_nothing_but_the_raw_cache(tmp_path: Path) -> None:
    """Canonical facts are the caller's to save. This call returns them."""
    cache = tmp_path / "jpdb"
    send, _ = _transport(
        {RI_URL: NOMINAL, RI_READING_URL: RI_DETAIL, WA_READING_URL: WA_PAGE}
    )

    fetch_character("理", html_cache=cache, transport=send)

    assert [path.name for path in sorted(tmp_path.iterdir())] == ["jpdb"]
    assert all(path.suffix in {".html", ".json"} for path in cache.iterdir())


# --- the facts store ----------------------------------------------------------


def _readings(character: str = "理") -> CharacterReadings:
    return CharacterReadings(
        character=character,
        source_url=f"https://jpdb.io/kanji/{character}",
        fetched_at_utc="2026-09-07T00:00:00Z",
        sha256="0" * 64,
        groups=(
            ReadingGroup(
                source_class="kanji-reading-list-common",
                readings=(
                    ReadingUsage(
                        label="り",
                        href="/kanji-reading/理/り",
                        percent_text="(84%)",
                        percent=84,
                        percent_less_than=False,
                        examples=(
                            BoundExample(
                                written="理由",
                                pronounced="りゆう",
                                gloss="reason;  pretext",
                                furigana="理[り] 由[ゆう]",
                                source_url="https://jpdb.io/vocabulary/1550140/理由/りゆう#a",
                            ),
                        ),
                        detail_source_url="https://jpdb.io/kanji-reading/理/り",
                        detail_fetched_at_utc="2026-09-07T00:00:01Z",
                        detail_sha256="1" * 64,
                    ),
                    ReadingUsage("わ", "/kanji-reading/理/わ", "(<1%)", 1, True),
                ),
            ),
            ReadingGroup(
                source_class="kanji-reading-list",
                readings=(
                    ReadingUsage("ことわり", "/kanji-reading/理/ことわり", None, None, None),
                ),
            ),
        ),
    )


def test_the_store_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "jpdb_readings.json"
    store = {"理": _readings()}

    save_readings(path, store)

    assert load_readings(path) == store


def test_a_missing_store_is_empty_not_an_error(tmp_path: Path) -> None:
    """A build reads saved facts and never fetches, so an unfetched character
    is simply absent."""
    assert load_readings(tmp_path / "nothing.json") == {}


def test_the_wire_names_its_source_and_metric(tmp_path: Path) -> None:
    path = tmp_path / "jpdb_readings.json"

    save_readings(path, {"理": _readings()})

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert list(payload) == ["schema_version", "source", "metric", "characters"]
    assert payload["schema_version"] == 1
    assert payload["source"] == "jpdb"
    assert payload["metric"] == "jpdb_reported_usage"
    entry = payload["characters"]["理"]
    assert entry["corpus_scope"] is None and entry["denominator"] is None


def test_the_map_is_sorted_and_everything_under_it_keeps_source_order(
    tmp_path: Path,
) -> None:
    path = tmp_path / "jpdb_readings.json"

    save_readings(path, {"理": _readings(), "王": _readings("王")})

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert list(payload["characters"]) == sorted(["理", "王"])
    groups = payload["characters"]["理"]["groups"]
    assert [group["source_class"] for group in groups] == [
        "kanji-reading-list-common",
        "kanji-reading-list",
    ]
    assert [r["label"] for r in groups[0]["readings"]] == ["り", "わ"]


def test_an_unknown_quantity_is_saved_as_null_not_zero(tmp_path: Path) -> None:
    path = tmp_path / "jpdb_readings.json"

    save_readings(path, {"理": _readings()})

    rare = json.loads(path.read_text(encoding="utf-8"))["characters"]["理"]["groups"][1]
    assert rare["readings"][0]["percent"] is None
    assert rare["readings"][0]["percent_text"] is None
    assert rare["readings"][0]["percent_less_than"] is None


def test_the_bound_example_wire_carries_every_field(tmp_path: Path) -> None:
    path = tmp_path / "jpdb_readings.json"

    save_readings(path, {"理": _readings()})

    reading = json.loads(path.read_text(encoding="utf-8"))["characters"]["理"]["groups"][0][
        "readings"
    ][0]
    assert reading["examples"] == [
        {
            "written": "理由",
            "pronounced": "りゆう",
            "gloss": "reason;  pretext",
            "furigana": "理[り] 由[ゆう]",
            "source_url": "https://jpdb.io/vocabulary/1550140/理由/りゆう#a",
        }
    ]
    assert reading["detail_source_url"] == "https://jpdb.io/kanji-reading/理/り"
    assert reading["detail_sha256"] == "1" * 64


def test_the_store_is_written_through_the_protected_atomic_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A committed store is replaced whole or not at all; a plain write can
    leave a reader looking at half a file."""
    seen: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        jpdb_kanji,
        "atomic_write_text_bound",
        lambda path, text, **kwargs: seen.append((path, text)),
    )
    path = tmp_path / "jpdb_readings.json"

    save_readings(path, {"理": _readings()})

    assert [name for name, _ in seen] == [path]
    assert not path.exists(), "nothing wrote around the seam"
    assert seen[0][1].endswith("\n")


@pytest.mark.parametrize(
    ("field", "value"),
    [("schema_version", 2), ("source", "kanjiapi"), ("metric", "frequency")],
)
def test_a_store_that_is_not_this_wire_refuses(
    tmp_path: Path, field: str, value: object
) -> None:
    path = tmp_path / "jpdb_readings.json"
    payload = {
        "schema_version": 1,
        "source": "jpdb",
        "metric": "jpdb_reported_usage",
        "characters": {},
        field: value,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(KanjiReadingsError, match=field):
        load_readings(path)


def test_a_store_key_that_disagrees_with_its_entry_refuses(tmp_path: Path) -> None:
    with pytest.raises(KanjiReadingsError, match="王"):
        save_readings(tmp_path / "jpdb_readings.json", {"理": _readings("王")})


def test_a_stored_reading_missing_a_field_refuses(tmp_path: Path) -> None:
    path = tmp_path / "jpdb_readings.json"
    save_readings(path, {"理": _readings()})
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["characters"]["理"]["groups"][0]["readings"][0]["label"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(KanjiReadingsError, match="label"):
        load_readings(path)
