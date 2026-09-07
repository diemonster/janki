"""Tests for the standalone jpdb reading-page POC (``scripts/jpdb_readings_poc.py``).

Every page here is written for the test: two synthetic fixtures shaped like the
reading table a jpdb kanji page prints, and small inline strings for everything
else. The point is the handful of structures this parser accepts and refuses,
not an inventory of possible HTML, so no captured page is stored in the
repository.
"""

import hashlib
import importlib.util
import json
import urllib.parse
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "jpdb_readings"


def _load_poc():
    """Load the script as a standalone module; it is not part of the package."""
    spec = importlib.util.spec_from_file_location(
        "jpdb_readings_poc", REPO_ROOT / "scripts" / "jpdb_readings_poc.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


poc = _load_poc()

NOMINAL = (FIXTURES / "nominal.html").read_bytes()
BOUNDED = (FIXTURES / "bounded.html").read_bytes()

# The readings the nominal fixture prints, in its source order.
NOMINAL_GROUPS = [
    {
        "source_class": "kanji-reading-list-common",
        "readings": [
            {
                "label": "り",
                "href": "/kanji-reading/理/り",
                "percent_text": "(84%)",
                "percent": 84,
                "percent_less_than": False,
            },
            {
                "label": "わ",
                "href": "/kanji-reading/理/わ",
                "percent_text": "(14%)",
                "percent": 14,
                "percent_less_than": False,
            },
        ],
    },
    {
        "source_class": "kanji-reading-list",
        "readings": [
            {
                "label": "ことわり",
                "href": "/kanji-reading/理/ことわり",
                "percent_text": None,
                "percent": None,
                "percent_less_than": None,
            },
            {
                "label": "め",
                "href": "/kanji-reading/理/め",
                "percent_text": None,
                "percent": None,
                "percent_less_than": None,
            },
            {
                "label": "ことわ",
                "href": "/kanji-reading/理/ことわ",
                "percent_text": None,
                "percent": None,
                "percent_less_than": None,
            },
        ],
    },
]


def _manifest(character, data):
    """A manifest written by the test, so cache reads do not lean on the POC."""
    return {
        "schema_version": 1,
        "character": character,
        "source_url": "https://jpdb.io/kanji/" + urllib.parse.quote(character, safe=""),
        "fetched_at_utc": "2026-09-07T00:00:00Z",
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _paths(cache_dir, character):
    stem = f"{ord(character):x}"  # 理 -> 7406
    return cache_dir / f"{stem}.html", cache_dir / f"{stem}.json"


def _seed(cache_dir, character="理", data=NOMINAL, manifest=None):
    cache_dir.mkdir(parents=True, exist_ok=True)
    html_path, manifest_path = _paths(cache_dir, character)
    html_path.write_bytes(data)
    manifest_path.write_text(
        json.dumps(_manifest(character, data) if manifest is None else manifest),
        encoding="utf-8",
    )
    return html_path, manifest_path


class _FakeResponse:
    def __init__(self, data):
        self._data = data

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def read(self):
        return self._data


def _stub_urlopen(monkeypatch, data=NOMINAL, error=None):
    """Replace the module's urlopen and record what it was asked for."""
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append((request.full_url, request.get_method(), timeout))
        if error is not None:
            raise error
        return _FakeResponse(data)

    monkeypatch.setattr(poc, "urlopen", fake_urlopen)
    return calls


def _cell(inner, source_class="kanji-reading-list-common"):
    return (
        '<table class="cross-table"><tr><td>Readings</td>'
        f'<td class="{source_class}">{inner}</td></tr></table>'
    )


# --- what a reading table says -----------------------------------------------


def test_nominal_page_reports_both_cells_in_source_order():
    assert poc.parse_page(NOMINAL.decode("utf-8")) == NOMINAL_GROUPS


def test_links_outside_the_reading_cells_are_not_reported():
    # The fixture's two placeholder links stand in for the vocabulary and
    # example links the real page prints elsewhere: outside a reading cell, so
    # not tied to a reading, so not reported.
    page = NOMINAL.decode("utf-8")
    hrefs = [reading["href"] for group in poc.parse_page(page) for reading in group["readings"]]
    assert len(hrefs) == 5
    assert all(href.startswith("/kanji-reading/") for href in hrefs)
    assert "/placeholder/not-a-reading" in page  # in an unrelated row of the same table
    assert "/placeholder/also-not-a-reading" in page  # and below the table


def test_unrelated_info_rows_do_not_refuse():
    # This row's cell holds a div and a bare link, which would be malformed
    # inside a reading cell. Only the target cells are judged.
    page = (
        '<table class="cross-table"><tr><td>Type</td><td class="space-between">'
        '<div>Kyōiku (2nd grade)&nbsp;</div><a href="/x" class="what-is-this">?</a></td></tr>'
        '<tr><td>Heisig</td><td>283</td></tr>'
        '<tr><td>Readings</td><td class="kanji-reading-list-common">'
        '<div><a href="/kanji-reading/理/り">り</a><div>(84%)</div></div>'
        "</td></tr></table>"
    )
    assert poc.parse_page(page) == [
        {
            "source_class": "kanji-reading-list-common",
            "readings": [
                {
                    "label": "り",
                    "href": "/kanji-reading/理/り",
                    "percent_text": "(84%)",
                    "percent": 84,
                    "percent_less_than": False,
                }
            ],
        }
    ]


def test_bounded_and_missing_percentages_are_reported_as_such():
    common, rare = poc.parse_page(BOUNDED.decode("utf-8"))
    printed = [
        (r["percent_text"], r["percent"], r["percent_less_than"]) for r in common["readings"]
    ]
    assert printed == [
        ("(<1%)", 1, True),  # the entity is decoded; the bound is not flattened to 1%
        ("(3%)", 3, False),
        (None, None, None),  # a common reading printed without a percentage
    ]
    assert rare["source_class"] == "kanji-reading-list"
    assert [(r["percent_text"], r["percent"]) for r in rare["readings"]] == [(None, None)]


@pytest.mark.parametrize(
    "printed, expected_text, expected_percent, expected_bound",
    [
        ("(84%)", "(84%)", 84, False),
        ("84%", "84%", 84, False),
        ("(&lt;1%)", "(<1%)", 1, True),
        ("&lt;1%", "<1%", 1, True),
        ("(less than 1%)", "(less than 1%)", 1, True),
    ],
)
def test_accepted_percentage_forms(printed, expected_text, expected_percent, expected_bound):
    page = _cell(f'<div><a href="/kanji-reading/理/り">り</a><div>{printed}</div></div>')
    (reading,) = poc.parse_page(page)[0]["readings"]
    assert reading["percent_text"] == expected_text
    assert (reading["percent"], reading["percent_less_than"]) == (expected_percent, expected_bound)


# --- refusals ----------------------------------------------------------------


@pytest.mark.parametrize(
    "printed",
    ["(84.5%)", "(84)", "()", "(84%", "84 %", "(approx 84%)", "(-3%)", "(８４%)"],
)
def test_unknown_percentage_text_refuses(printed):
    page = _cell(f'<div><a href="/kanji-reading/理/り">り</a><div>{printed}</div></div>')
    with pytest.raises(poc.PocError, match="percentage"):
        poc.parse_page(page)


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
def test_malformed_reading_structure_refuses(inner):
    with pytest.raises(poc.PocError):
        poc.parse_page(_cell(inner))


def test_page_ending_inside_a_later_reading_cell_refuses():
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
    with pytest.raises(poc.PocError, match="unclosed reading cell"):
        poc.parse_page(page)


def test_page_without_a_reading_cell_refuses():
    page = '<table class="cross-table"><tr><td>Heisig</td><td>283</td></tr></table>'
    with pytest.raises(poc.PocError, match="reading cell"):
        poc.parse_page(page)


# --- the report header -------------------------------------------------------

REPORT_HEADER = {
    "schema_version": 1,
    "source": "jpdb",
    "metric": "jpdb_reported_usage",
    "corpus_scope": None,
    "denominator": None,
}


def test_report_states_source_metric_and_that_corpus_and_denominator_are_unknown():
    report = poc.build_report([])

    assert list(report) == [*REPORT_HEADER, "characters"]
    assert {field: report[field] for field in REPORT_HEADER} == REPORT_HEADER
    assert report["characters"] == []


def test_a_printed_report_carries_that_header_beside_its_characters(tmp_path, capsys):
    _seed(tmp_path / "cache")

    assert poc.main(["理", "--cache-dir", str(tmp_path / "cache")]) == 0

    report = json.loads(capsys.readouterr().out)
    assert list(report) == [*REPORT_HEADER, "characters"]
    assert {field: report[field] for field in REPORT_HEADER} == REPORT_HEADER
    assert [entry["character"] for entry in report["characters"]] == ["理"]


# --- offline, cache, and fetch -----------------------------------------------


def test_offline_run_reads_the_cache_and_makes_no_request(tmp_path, monkeypatch, capsys):
    calls = _stub_urlopen(monkeypatch)
    _seed(tmp_path / "cache")

    assert poc.main(["理", "--cache-dir", str(tmp_path / "cache")]) == 0

    assert calls == []
    (entry,) = json.loads(capsys.readouterr().out)["characters"]
    assert entry["character"] == "理"
    assert entry["provenance"] == _manifest("理", NOMINAL)
    assert entry["groups"] == NOMINAL_GROUPS


def test_offline_missing_cache_refuses_without_requesting(tmp_path, monkeypatch, capsys):
    calls = _stub_urlopen(monkeypatch)
    cache = tmp_path / "cache"
    cache.mkdir()

    assert poc.main(["理", "--cache-dir", str(cache)]) == 2

    assert calls == []
    assert list(cache.iterdir()) == []  # offline never writes
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "no cache entry" in captured.err


def test_fetch_reuses_a_valid_cache_entry_without_requesting(tmp_path, monkeypatch, capsys):
    calls = _stub_urlopen(monkeypatch, data=b"<html>replaced</html>")
    cache = tmp_path / "cache"
    html_path, manifest_path = _seed(cache)
    before = (html_path.read_bytes(), manifest_path.read_bytes())

    assert poc.main(["理", "--cache-dir", str(cache), "--fetch"]) == 0

    assert calls == []
    assert (html_path.read_bytes(), manifest_path.read_bytes()) == before
    assert json.loads(capsys.readouterr().out)["characters"][0]["groups"] == NOMINAL_GROUPS


def test_refresh_requests_again_and_replaces_the_entry(tmp_path, monkeypatch, capsys):
    calls = _stub_urlopen(monkeypatch, data=BOUNDED)
    cache = tmp_path / "cache"
    html_path, manifest_path = _seed(cache)

    assert poc.main(["理", "--cache-dir", str(cache), "--fetch", "--refresh"]) == 0

    assert len(calls) == 1
    assert html_path.read_bytes() == BOUNDED
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["sha256"] == hashlib.sha256(BOUNDED).hexdigest()
    groups = json.loads(capsys.readouterr().out)["characters"][0]["groups"]
    assert [reading["label"] for reading in groups[0]["readings"]] == ["A", "B", "C"]


def test_refresh_requires_fetch(tmp_path, monkeypatch, capsys):
    calls = _stub_urlopen(monkeypatch)
    _seed(tmp_path / "cache")

    assert poc.main(["理", "--cache-dir", str(tmp_path / "cache"), "--refresh"]) == 2

    assert calls == []
    assert "--fetch" in capsys.readouterr().err


def test_fetch_stores_the_exact_bytes_and_a_manifest_for_them(tmp_path, monkeypatch, capsys):
    calls = _stub_urlopen(monkeypatch)
    cache = tmp_path / "cache"

    assert poc.main(["理", "--cache-dir", str(cache), "--fetch"]) == 0

    assert calls == [("https://jpdb.io/kanji/%E7%90%86", "GET", poc.TIMEOUT_SECONDS)]
    assert poc.TIMEOUT_SECONDS > 0
    html_path, manifest_path = _paths(cache, "理")
    assert html_path.read_bytes() == NOMINAL
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(manifest) == {
        "schema_version",
        "character",
        "source_url",
        "fetched_at_utc",
        "sha256",
    }
    assert manifest["schema_version"] == 1
    assert manifest["character"] == "理"
    assert manifest["source_url"] == "https://jpdb.io/kanji/%E7%90%86"
    assert manifest["sha256"] == hashlib.sha256(NOMINAL).hexdigest()
    assert manifest["fetched_at_utc"].endswith("Z")
    assert json.loads(capsys.readouterr().out)["characters"][0]["provenance"] == manifest


def test_only_the_missing_character_is_requested(tmp_path, monkeypatch, capsys):
    calls = _stub_urlopen(monkeypatch, data=BOUNDED)
    cache = tmp_path / "cache"
    _seed(cache, "理")

    assert poc.main(["理", "王", "--cache-dir", str(cache), "--fetch"]) == 0

    assert [call[0] for call in calls] == ["https://jpdb.io/kanji/%E7%8E%8B"]
    report = json.loads(capsys.readouterr().out)
    assert [entry["character"] for entry in report["characters"]] == ["理", "王"]
    assert _paths(cache, "王")[0].read_bytes() == BOUNDED


def test_failed_request_is_not_retried_and_keeps_the_cache(tmp_path, monkeypatch, capsys):
    calls = _stub_urlopen(monkeypatch, error=OSError("connection reset"))
    cache = tmp_path / "cache"
    html_path, manifest_path = _seed(cache)
    before = (html_path.read_bytes(), manifest_path.read_bytes())

    assert poc.main(["理", "--cache-dir", str(cache), "--fetch", "--refresh"]) == 2

    assert len(calls) == 1
    assert (html_path.read_bytes(), manifest_path.read_bytes()) == before
    assert capsys.readouterr().out == ""


def test_unparseable_response_is_not_cached(tmp_path, monkeypatch, capsys):
    calls = _stub_urlopen(monkeypatch, data=b"<html><body>service unavailable</body></html>")
    cache = tmp_path / "cache"

    assert poc.main(["理", "--cache-dir", str(cache), "--fetch"]) == 2

    assert len(calls) == 1
    assert not cache.exists() or list(cache.iterdir()) == []
    assert capsys.readouterr().out == ""


def test_cache_with_a_wrong_hash_refuses(tmp_path, monkeypatch, capsys):
    calls = _stub_urlopen(monkeypatch)
    cache = tmp_path / "cache"
    html_path, _ = _seed(cache)
    html_path.write_bytes(NOMINAL.replace(b"(84%)", b"(48%)"))  # still parseable, not the bytes

    assert poc.main(["理", "--cache-dir", str(cache), "--fetch"]) == 2

    assert calls == []  # a bad entry is refused, not silently re-fetched over
    assert "sha256" in capsys.readouterr().err


@pytest.mark.parametrize(
    "field, value",
    [
        ("character", "王"),
        ("source_url", "https://jpdb.io/kanji/%E7%8E%8B"),
        ("schema_version", 2),
        ("fetched_at_utc", ""),
    ],
)
def test_cache_manifest_that_does_not_match_the_request_refuses(tmp_path, field, value):
    cache = tmp_path / "cache"
    manifest = _manifest("理", NOMINAL)
    manifest[field] = value
    _seed(cache, manifest=manifest)

    with pytest.raises(poc.PocError, match=field):
        poc.read_cache(cache, "理")


def test_cache_that_is_not_valid_utf8_refuses(tmp_path, capsys):
    cache = tmp_path / "cache"
    data = NOMINAL.replace("り".encode(), b"\xe3\x81")  # truncated code point
    _seed(cache, data=data)  # manifest hash matches, so only the decode can refuse

    assert poc.main(["理", "--cache-dir", str(cache)]) == 2

    assert "not valid UTF-8" in capsys.readouterr().err


def test_incomplete_cache_entry_refuses(tmp_path, monkeypatch, capsys):
    calls = _stub_urlopen(monkeypatch)
    cache = tmp_path / "cache"
    _, manifest_path = _seed(cache)
    manifest_path.unlink()

    assert poc.main(["理", "--cache-dir", str(cache), "--fetch"]) == 2

    assert calls == []
    assert "incomplete cache entry" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [["理由"], [""], ["理", "理"]])
def test_arguments_must_be_single_distinct_characters(tmp_path, monkeypatch, argv, capsys):
    calls = _stub_urlopen(monkeypatch)
    _seed(tmp_path / "cache")

    assert poc.main([*argv, "--cache-dir", str(tmp_path / "cache"), "--fetch"]) == 2

    assert calls == []
    assert capsys.readouterr().out == ""


def test_cache_dir_is_required():
    with pytest.raises(SystemExit):
        poc.main(["理"])
