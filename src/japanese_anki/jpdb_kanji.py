"""What jpdb's own kanji pages report about a character's readings.

Two pages, and nothing between them is inferred:

* the **kanji page** (``/kanji/理``) prints a reading table. Each cell carries
  a class — ``kanji-reading-list-common`` for the readings jpdb quantifies,
  ``kanji-reading-list`` for the ones it does not — and each reading is a link
  whose text is the label and, in the common cell, a sibling div holding the
  percentage jpdb printed beside it.
* the **reading page** an entry links to (``/kanji-reading/理/り``) states that
  reading's identity in a two-row table and lists the words jpdb itself files
  under it, each anchor supplying the expression, the whole-word reading, and
  the ruby that says which kana sit over which characters.

Every figure here is jpdb's. Labels, hrefs and percentage strings are carried
through verbatim, source order is preserved, nothing is sorted, merged,
classified as on or kun, renormalised, or reconciled against KANJIDIC. A
percentage is jpdb's own rounded figure rather than a measurement janki made,
and ``<1%`` stays the upper bound it is. A missing percentage is ``None``, never
zero, and the corpus and denominator behind those figures are unknown because
the page does not print them.

**The reading page is the only thing that binds a word to a reading.** janki
never decides that 料理 shows 理's り: it follows the link jpdb printed under
り, checks that the page it lands on says it is 理's り, and keeps the first
words that page lists. No Japanese is read, matched, segmented, or aligned
anywhere in this module — the ruby that becomes Anki furigana notation is the
page's own, character by character, exactly as supplied.

Raw HTML is a private local cache outside the repository; the facts extracted
from it are what a caller saves, and a build reads only those saved facts.
Cached bytes are reused by default and replaced only by an explicit refresh; a
request that fails, decodes wrong, or does not parse leaves the cache exactly
as it was.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from japanese_anki.errors import JankiError
from japanese_anki.identifiers import contains_kanji
from japanese_anki.io import atomic_write_text_bound, read_text_bound
from japanese_anki.jpdb import furigana_to_anki

__all__ = [
    "BoundExample",
    "CharacterReadings",
    "COMMON_CELL_CLASS",
    "KanjiReadingsError",
    "MAX_BOUND_EXAMPLES",
    "METRIC",
    "READING_CELL_CLASSES",
    "ReadingGroup",
    "ReadingPage",
    "ReadingUsage",
    "SCHEMA_VERSION",
    "SOURCE",
    "Transport",
    "fetch_character",
    "kanji_page_url",
    "load_readings",
    "parse_kanji_page",
    "parse_reading_page",
    "parse_readings",
    "save_readings",
    "urllib_transport",
]

SCHEMA_VERSION = 1
SOURCE = "jpdb"

#: What the numbers in this store *are*: the share of a character's uses jpdb
#: itself reports beside a reading. Not a measurement janki made.
METRIC = "jpdb_reported_usage"

SITE = "https://jpdb.io"

#: The cell jpdb prints a percentage in, and the only cell a detail page is
#: ever requested for.
COMMON_CELL_CLASS = "kanji-reading-list-common"
READING_CELL_CLASSES = (COMMON_CELL_CLASS, "kanji-reading-list")

#: How many of a reading page's words to keep. The page for 理's り lists 1487;
#: two is what a card can show, and they are the page's own first two rather
#: than a selection janki ranked.
MAX_BOUND_EXAMPLES = 2

MANIFEST_FIELDS = ("schema_version", "character", "source_url", "fetched_at_utc", "sha256")

TIMEOUT_SECONDS = 30
USER_AGENT = "janki (personal Japanese deck builder)"

#: ``(url) -> bytes``. The seam tests replace, so no test reaches the network.
Transport = Callable[[str], bytes]

# (84%) and its upper-bound sibling (<1%), the two shapes jpdb prints. The
# source's own parentheses are optional here and preserved in `percent_text`.
_PERCENTAGE = re.compile(r"(?:(?P<bound><|less than)\s*)?(?P<value>[0-9]+)%")

#: A per-cent escape a source href already carries, and the only run of an href
#: that :func:`_absolute` passes through untouched.
_PERCENT_ESCAPE = re.compile(r"%[0-9A-Fa-f]{2}")


class KanjiReadingsError(JankiError):
    """A refusal: this is not a page, cache entry, or store janki will use."""


# --- the facts ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BoundExample:
    """One word a reading page files under that reading, as the page gave it.

    ``written`` and ``pronounced`` are the anchor URL's own fields — jpdb states
    the expression and its whole-word reading there, so neither is parsed out of
    Japanese. ``furigana`` is Anki notation built from the anchor's supplied
    ruby, character by character; it is never derived by aligning ``written``
    against ``pronounced``.
    """

    written: str
    pronounced: str
    gloss: str
    furigana: str
    source_url: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "written": self.written,
            "pronounced": self.pronounced,
            "gloss": self.gloss,
            "furigana": self.furigana,
            "source_url": self.source_url,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> BoundExample:
        fields = _object(raw, "an example")
        return cls(
            written=_text(fields, "written", "an example"),
            pronounced=_text(fields, "pronounced", "an example"),
            gloss=_text(fields, "gloss", "an example"),
            furigana=_text(fields, "furigana", "an example"),
            source_url=_text(fields, "source_url", "an example"),
        )


@dataclass(frozen=True, slots=True)
class ReadingUsage:
    """One reading as the kanji page printed it, with what jpdb binds to it.

    ``label``, ``href`` and ``percent_text`` are verbatim source. ``percent``
    and ``percent_less_than`` read that same string so a caller need not, and
    are ``None`` together when jpdb printed no figure at all. The ``detail_*``
    provenance records which page the examples came from even when it supplied
    none, because "asked, and jpdb listed nothing" is a different fact from
    "never asked".
    """

    label: str
    href: str
    percent_text: str | None
    percent: int | None
    percent_less_than: bool | None
    examples: tuple[BoundExample, ...] = ()
    detail_source_url: str | None = None
    detail_fetched_at_utc: str | None = None
    detail_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "href": self.href,
            "percent_text": self.percent_text,
            "percent": self.percent,
            "percent_less_than": self.percent_less_than,
            "examples": [example.to_dict() for example in self.examples],
            "detail_source_url": self.detail_source_url,
            "detail_fetched_at_utc": self.detail_fetched_at_utc,
            "detail_sha256": self.detail_sha256,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> ReadingUsage:
        fields = _object(raw, "a reading")
        return cls(
            label=_text(fields, "label", "a reading"),
            href=_text(fields, "href", "a reading"),
            percent_text=_optional_text(fields, "percent_text", "a reading"),
            percent=_optional_int(fields, "percent", "a reading"),
            percent_less_than=_optional_bool(fields, "percent_less_than", "a reading"),
            examples=tuple(
                BoundExample.from_dict(item)
                for item in _sequence(fields.get("examples"), "a reading's examples")
            ),
            detail_source_url=_optional_text(fields, "detail_source_url", "a reading"),
            detail_fetched_at_utc=_optional_text(fields, "detail_fetched_at_utc", "a reading"),
            detail_sha256=_optional_text(fields, "detail_sha256", "a reading"),
        )


@dataclass(frozen=True, slots=True)
class ReadingGroup:
    """One reading cell, kept apart because its class is what jpdb said it is.

    ``source_class`` is the cell's own class name. It is *not* an on/kun
    classification and must never be rendered as one: it says only whether jpdb
    printed percentages beside these readings.
    """

    source_class: str
    readings: tuple[ReadingUsage, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_class": self.source_class,
            "readings": [reading.to_dict() for reading in self.readings],
        }

    @classmethod
    def from_dict(cls, raw: Any) -> ReadingGroup:
        fields = _object(raw, "a reading group")
        return cls(
            source_class=_text(fields, "source_class", "a reading group"),
            readings=tuple(
                ReadingUsage.from_dict(item)
                for item in _sequence(fields.get("readings"), "a reading group's readings")
            ),
        )


@dataclass(frozen=True, slots=True)
class CharacterReadings:
    """Everything one jpdb kanji page reported, with the response it came from.

    ``corpus_scope`` and ``denominator`` stay ``None``: jpdb prints neither, and
    a reader who needs them has to get them from jpdb rather than from a guess
    made here.
    """

    character: str
    source_url: str
    fetched_at_utc: str
    sha256: str
    groups: tuple[ReadingGroup, ...]
    corpus_scope: str | None = None
    denominator: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "character": self.character,
            "source_url": self.source_url,
            "fetched_at_utc": self.fetched_at_utc,
            "sha256": self.sha256,
            "corpus_scope": self.corpus_scope,
            "denominator": self.denominator,
            "groups": [group.to_dict() for group in self.groups],
        }

    @classmethod
    def from_dict(cls, raw: Any) -> CharacterReadings:
        fields = _object(raw, "a character entry")
        return cls(
            character=_text(fields, "character", "a character entry"),
            source_url=_text(fields, "source_url", "a character entry"),
            fetched_at_utc=_text(fields, "fetched_at_utc", "a character entry"),
            sha256=_text(fields, "sha256", "a character entry"),
            groups=tuple(
                ReadingGroup.from_dict(item)
                for item in _sequence(fields.get("groups"), "a character entry's groups")
            ),
            corpus_scope=_optional_text(fields, "corpus_scope", "a character entry"),
            denominator=_optional_text(fields, "denominator", "a character entry"),
        )


@dataclass(frozen=True, slots=True)
class ReadingPage:
    """A reading page's stated identity and the words it lists, in its order.

    The page prints its own Frequency row. It is not read: the kanji page's
    figure is the one this module reports, and two roundings of the same number
    are still two numbers.
    """

    character: str
    reading: str
    examples: tuple[BoundExample, ...]


# --- reading a stored entry --------------------------------------------------


def _object(raw: Any, what: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise KanjiReadingsError(f"{what} must be a JSON object, not {type(raw).__name__}")
    return raw


def _sequence(raw: Any, what: str) -> Sequence[Any]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise KanjiReadingsError(f"{what} must be a JSON list, not {type(raw).__name__}")
    return raw


def _text(fields: Mapping[str, Any], key: str, what: str) -> str:
    value = fields.get(key)
    if not isinstance(value, str):
        raise KanjiReadingsError(f"{what} needs a string {key}")
    return value


def _optional_text(fields: Mapping[str, Any], key: str, what: str) -> str | None:
    value = fields.get(key)
    if value is None or isinstance(value, str):
        return value
    raise KanjiReadingsError(f"{what}: {key} must be a string or null")


def _optional_int(fields: Mapping[str, Any], key: str, what: str) -> int | None:
    value = fields.get(key)
    if value is None or (isinstance(value, int) and not isinstance(value, bool)):
        return value
    raise KanjiReadingsError(f"{what}: {key} must be a whole number or null")


def _optional_bool(fields: Mapping[str, Any], key: str, what: str) -> bool | None:
    value = fields.get(key)
    if value is None or isinstance(value, bool):
        return value
    raise KanjiReadingsError(f"{what}: {key} must be true, false, or null")


# --- URLs --------------------------------------------------------------------


def kanji_page_url(character: str) -> str:
    """The page URL for ``character``, and its cache entry's source identity."""
    return f"{SITE}/kanji/" + urllib.parse.quote(character, safe="")


def _absolute(href: str, *, where: str) -> str:
    """One of jpdb's own printed hrefs as the absolute URL it names.

    Site-relative only. An href that leaves jpdb — or a protocol-relative one
    that could — is refused rather than followed, so nothing this module fetches
    is chosen by the page rather than by janki.

    jpdb writes 理 in a path either as the character or as ``%E7%90%86``, and
    both spell the same resource. An escape that is already there is left alone;
    everything else — the characters, and a lone per-cent sign that is not an
    escape — is escaped once. The href itself is never rewritten: this is the
    URL to ask for, while what a page printed stays verbatim in the facts.
    """
    if not href.startswith("/") or href.startswith("//"):
        raise KanjiReadingsError(f"{where}: {href!r} is not a jpdb site path")
    parts = []
    cursor = 0
    for escape in _PERCENT_ESCAPE.finditer(href):
        parts.append(urllib.parse.quote(href[cursor:escape.start()], safe="/#"))
        parts.append(escape.group())
        cursor = escape.end()
    parts.append(urllib.parse.quote(href[cursor:], safe="/#"))
    return SITE + "".join(parts)


def _vocabulary_fields(href: str) -> tuple[str, str]:
    """``/vocabulary/1550140/理由/わけ#a`` -> ``("理由", "わけ")``.

    jpdb states both in the URL, so this is a mechanical field split rather than
    anything read out of the Japanese. The same vocabulary id appears under
    several readings — 1550140 is both 理由/りゆう and 理由/わけ — so the id is
    not an identity this module deduplicates on.
    """
    path = urllib.parse.urlsplit(href).path
    parts = path.split("/")
    if len(parts) != 5 or parts[0] != "" or parts[1] != "vocabulary":
        raise KanjiReadingsError(
            f"a used-in link is not /vocabulary/<id>/<expression>/<reading>: {href!r}"
        )
    written = urllib.parse.unquote(parts[3])
    pronounced = urllib.parse.unquote(parts[4])
    if not parts[2] or not written or not pronounced:
        raise KanjiReadingsError(f"a used-in link is missing one of its fields: {href!r}")
    return written, pronounced


# --- the kanji page ----------------------------------------------------------


def _percentage(text: str, where: str) -> tuple[int, bool]:
    """Read one printed percentage as ``(value, is_upper_bound)``."""
    inner = text.strip()
    if inner.startswith("(") and inner.endswith(")"):
        inner = inner[1:-1].strip()
    match = _PERCENTAGE.fullmatch(inner)
    if match is None:
        raise KanjiReadingsError(f"{where}: unrecognised percentage {text!r}")
    return int(match["value"]), match["bound"] is not None


class _ReadingCellParser(HTMLParser):
    """Collect the reading cells of a kanji page and skip everything else.

    Only markup *inside* a target cell is judged, so the unrelated Info rows,
    the vocabulary list and the examples pass by untouched: this is a reader
    for two cells, not a strict HTML validator.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.groups: list[ReadingGroup] = []
        self._source_class: str | None = None  # the cell being read
        self._readings: list[ReadingUsage] = []
        self._reading: dict[str, str | None] | None = None  # the wrapper div
        self._capture: str | None = None  # "label" or "percent" while inside one
        self._chunks: list[str] = []

    @property
    def unfinished(self) -> bool:
        """True when a target cell was opened and never closed."""
        return self._source_class is not None

    def _refuse(self, problem: str) -> None:
        where = self._source_class or "reading cell"
        raise KanjiReadingsError(f"{where}: {problem} (line {self.getpos()[0]})")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "td":
            if self._source_class is not None:
                self._refuse("a new cell begins before this one is closed")
            classes = (attributes.get("class") or "").split()
            matched = [name for name in READING_CELL_CLASSES if name in classes]
            if len(matched) > 1:
                raise KanjiReadingsError(
                    "cell carries several reading classes: " + " ".join(matched)
                )
            if matched:
                self._source_class = matched[0]
                self._readings = []
            return
        if self._source_class is None:
            return
        if self._capture is not None:
            self._refuse(f"<{tag}> inside a reading {self._capture}")
        if self._reading is None:
            if tag != "div":
                self._refuse(f"<{tag}> where a reading wrapper <div> was expected")
            self._reading = {"label": None, "href": None, "percent_text": None}
        elif tag == "a":
            if self._reading["href"] is not None:
                self._refuse("a second link in one reading wrapper")
            href = attributes.get("href")
            if href is None:
                self._refuse("reading link without an href")
            self._reading["href"] = href
            self._capture, self._chunks = "label", []
        elif tag == "div":
            if self._reading["percent_text"] is not None:
                self._refuse("a second percentage in one reading wrapper")
            self._capture, self._chunks = "percent", []
        else:
            self._refuse(f"unexpected <{tag}> in a reading wrapper")

    def handle_endtag(self, tag: str) -> None:
        if tag == "td":
            if self._source_class is None:
                return
            if self._reading is not None or self._capture is not None:
                self._refuse("cell ends inside an unclosed reading wrapper")
            if not self._readings:
                self._refuse("cell holds no readings")
            self.groups.append(
                ReadingGroup(source_class=self._source_class, readings=tuple(self._readings))
            )
            self._source_class = None
            self._readings = []
            return
        if self._source_class is None:
            return
        if self._capture == "label":
            if tag != "a":
                self._refuse(f"</{tag}> inside a reading link")
            self._reading["label"] = "".join(self._chunks)  # type: ignore[index]
            self._capture = None
        elif self._capture == "percent":
            if tag != "div":
                self._refuse(f"</{tag}> inside a percentage")
            self._reading["percent_text"] = "".join(self._chunks)  # type: ignore[index]
            self._capture = None
        elif self._reading is not None:
            if tag != "div":
                self._refuse(f"unexpected </{tag}> in a reading wrapper")
            self._readings.append(self._finish(self._reading))
            self._reading = None
        else:
            self._refuse(f"unexpected </{tag}> in the cell")

    def handle_data(self, data: str) -> None:
        if self._capture is not None:
            self._chunks.append(data)
        elif self._source_class is not None and data.strip():
            self._refuse(f"stray text {data.strip()!r} beside the readings")

    def _finish(self, reading: Mapping[str, str | None]) -> ReadingUsage:
        if reading["label"] is None:
            self._refuse("reading wrapper without a link")
        text = reading["percent_text"]
        percent, bound = (
            (None, None) if text is None else _percentage(text, self._source_class or "")
        )
        return ReadingUsage(
            label=reading["label"] or "",
            href=reading["href"] or "",
            percent_text=text,
            percent=percent,
            percent_less_than=bound,
        )


def parse_kanji_page(text: str) -> tuple[ReadingGroup, ...]:
    """The reading cells of a decoded kanji page, in source order.

    No examples: a kanji page prints a vocabulary list, but nothing on it says
    which reading a listed word uses, so nothing on it can bind one.
    """
    parser = _ReadingCellParser()
    parser.feed(text)
    parser.close()
    if parser.unfinished:
        raise KanjiReadingsError("the page ends inside an unclosed reading cell")
    if not parser.groups:
        raise KanjiReadingsError(
            "no reading cell on the page: " + " or ".join(READING_CELL_CLASSES)
        )
    return tuple(parser.groups)


# --- the reading page --------------------------------------------------------


class _ReadingPageParser(HTMLParser):
    """Read a reading page's identity table and its ``Used in`` entries.

    Like the kanji-page reader, this judges only the structures it is here for.
    Inside one of them — an entry's anchor above all — anything unrecognised is
    refused rather than skipped, because a silently dropped ruby produces
    furigana that looks right and is missing a character's reading.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[tuple[str, ...]] = []
        self.entries: list[tuple[str, list[Any], str]] = []
        self._in_table = False
        self._cells: list[str] = []
        self._cell: list[str] | None = None
        self._divs: list[str | None] = []
        self._in_section = False
        self._entry: dict[str, Any] | None = None
        self._anchor: dict[str, Any] | None = None
        self._ruby: dict[str, Any] | None = None
        self._in_rt = False
        self._in_rp = False
        self._gloss: list[str] | None = None

    def _refuse(self, problem: str) -> None:
        raise KanjiReadingsError(f"{problem} (line {self.getpos()[0]})")

    # -- the identity table --
    def _start_table(self, classes: list[str]) -> None:
        if "cross-table" in classes and not self.rows:
            self._in_table = True

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = (attributes.get("class") or "").split()
        if tag == "table":
            self._start_table(classes)
            return
        if self._in_table and tag == "td":
            self._cell = []
            return
        if self._anchor is not None:
            self._anchor_starttag(tag)
            return
        if tag == "a" and self._divs and self._divs[-1] == "jp":
            href = attributes.get("href")
            if href is None:
                self._refuse("a used-in link has no href")
            if self._entry is not None and self._entry["href"] is not None:
                self._refuse("a second link in one used-in entry")
            self._anchor = {"href": href, "segments": []}
            return
        if tag == "div":
            self._divs.append(self._div_role(classes))

    def _div_role(self, classes: list[str]) -> str | None:
        if "subsection-used-in" in classes:
            self._in_section = True
            return "section"
        if not self._in_section:
            return None
        if "used-in" in classes and self._entry is None:
            self._entry = {"href": None, "segments": None, "gloss": None}
            return "used-in"
        if self._entry is None:
            return None
        if "jp" in classes:
            return "jp"
        if "en" in classes:
            self._gloss = []
            return "en"
        return None

    def _anchor_starttag(self, tag: str) -> None:
        """Markup inside an entry's anchor: ruby, and the highlight wrapper."""
        assert self._anchor is not None
        if tag == "span":
            # The <span class="highlight"> jpdb wraps the searched character's
            # ruby in. It marks which ruby is this reading's; the ruby inside it
            # is kept exactly like any other, in place.
            return
        if tag == "ruby":
            if self._ruby is not None:
                self._refuse("a <ruby> inside a <ruby>")
            self._ruby = {"base": [], "rt": None}
            return
        if self._ruby is None:
            self._refuse(f"unexpected <{tag}> in a used-in link")
        if tag == "rt":
            if self._ruby["rt"] is not None:
                self._refuse("a second <rt> in one <ruby>")
            self._ruby["rt"] = []
            self._in_rt = True
            return
        if tag == "rp":
            # Parenthesis fallback for readers without ruby support. It repeats
            # nothing and belongs to neither the base nor the reading.
            self._in_rp = True
            return
        self._refuse(f"unexpected <{tag}> in a <ruby>")

    def handle_endtag(self, tag: str) -> None:
        if self._in_table:
            if tag == "td" and self._cell is not None:
                self._cells.append("".join(self._cell))
                self._cell = None
                return
            if tag == "tr":
                if self._cells:
                    self.rows.append(tuple(self._cells))
                self._cells = []
                return
            if tag == "table":
                self._in_table = False
                return
        if self._anchor is not None:
            self._anchor_endtag(tag)
            return
        if tag == "div" and self._divs:
            self._close_div(self._divs.pop())

    def _anchor_endtag(self, tag: str) -> None:
        assert self._anchor is not None
        if tag == "rp":
            self._in_rp = False
            return
        if tag == "rt":
            self._in_rt = False
            return
        if tag == "ruby":
            if self._ruby is None:
                self._refuse("</ruby> without a <ruby>")
            base = "".join(self._ruby["base"])
            reading = "".join(self._ruby["rt"] or [])
            if not base:
                self._refuse("a <ruby> with no text under its reading")
            self._anchor["segments"].append((base, reading) if reading else base)
            self._ruby = None
            return
        if tag == "span":
            return
        if tag == "a":
            if self._ruby is not None:
                self._refuse("a used-in link ends inside an unclosed <ruby>")
            if not self._anchor["segments"]:
                self._refuse("a used-in link supplies no ruby")
            assert self._entry is not None
            self._entry["href"] = self._anchor["href"]
            self._entry["segments"] = self._anchor["segments"]
            self._anchor = None
            return
        self._refuse(f"unexpected </{tag}> in a used-in link")

    def _close_div(self, role: str | None) -> None:
        if role == "section":
            self._in_section = False
            return
        if role == "en":
            if self._entry is not None:
                self._entry["gloss"] = "".join(self._gloss or [])
            self._gloss = None
            return
        if role == "used-in":
            entry = self._entry
            self._entry = None
            if entry is None:
                return
            if entry["href"] is None or entry["segments"] is None:
                self._refuse("a used-in entry has no linked word")
            if entry["gloss"] is None:
                self._refuse("a used-in entry has no English")
            self.entries.append((entry["href"], entry["segments"], entry["gloss"]))

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)
            return
        if self._anchor is not None:
            if self._in_rp:
                return
            if self._ruby is None:
                if data.strip():
                    self._refuse(f"stray text {data.strip()!r} in a used-in link")
                return
            if self._in_rt:
                self._ruby["rt"].append(data)
            else:
                self._ruby["base"].append(data)
            return
        if self._gloss is not None:
            self._gloss.append(data)


def parse_reading_page(text: str) -> ReadingPage:
    """A decoded reading page: what it says it is, and the words it lists.

    Every listed word is kept, in the page's order. Which of them a caller uses
    is a source-order selection, not a claim about word frequency.
    """
    parser = _ReadingPageParser()
    parser.feed(text)
    parser.close()
    stated = {row[0].strip(): row[1].strip() for row in parser.rows if len(row) == 2}
    character = stated.get("Kanji", "")
    reading = stated.get("Reading", "")
    if not character or not reading:
        raise KanjiReadingsError(
            "a reading page states its Kanji and Reading in a cross-table; "
            f"this page states {sorted(stated)}"
        )
    examples = []
    for href, segments, gloss in parser.entries:
        written, pronounced = _vocabulary_fields(href)
        examples.append(
            BoundExample(
                written=written,
                pronounced=pronounced,
                gloss=gloss.strip(),
                furigana=furigana_to_anki(segments),
                source_url=_absolute(href, where="a used-in link"),
            )
        )
    return ReadingPage(character=character, reading=reading, examples=tuple(examples))


# --- the raw cache -----------------------------------------------------------


def main_cache_paths(html_cache: Path, character: str) -> tuple[Path, Path]:
    """A kanji page's two names: the hex code point as ``.html`` and ``.json``."""
    stem = Path(html_cache) / f"{ord(character):x}"
    return stem.with_suffix(".html"), stem.with_suffix(".json")


def detail_cache_paths(html_cache: Path, url: str) -> tuple[Path, Path]:
    """A reading page's two names, from its URL: one page, one entry."""
    stem = Path(html_cache) / hashlib.sha256(url.encode("utf-8")).hexdigest()
    return stem.with_suffix(".html"), stem.with_suffix(".json")


def _build_manifest(character: str, url: str, data: bytes, fetched_at_utc: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "character": character,
        "source_url": url,
        "fetched_at_utc": fetched_at_utc,
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _check_manifest(
    manifest: Any, character: str, url: str, data: bytes, path: Path
) -> None:
    """Refuse an entry that is not this page, for this character, byte for byte."""
    if not isinstance(manifest, dict) or set(manifest) != set(MANIFEST_FIELDS):
        raise KanjiReadingsError(
            f"{path}: manifest fields are not exactly {', '.join(MANIFEST_FIELDS)}"
        )
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise KanjiReadingsError(
            f"{path}: schema_version is {manifest['schema_version']!r}, not {SCHEMA_VERSION}"
        )
    if manifest["character"] != character:
        raise KanjiReadingsError(
            f"{path}: character is {manifest['character']!r}, not {character!r}"
        )
    if manifest["source_url"] != url:
        raise KanjiReadingsError(
            f"{path}: source_url is {manifest['source_url']!r}, not {url!r}"
        )
    if not isinstance(manifest["fetched_at_utc"], str) or not manifest["fetched_at_utc"]:
        raise KanjiReadingsError(f"{path}: fetched_at_utc is not a timestamp")
    if manifest["sha256"] != hashlib.sha256(data).hexdigest():
        raise KanjiReadingsError(f"{path}: sha256 does not match the cached page bytes")


def read_cache(
    paths: tuple[Path, Path], character: str, url: str
) -> tuple[bytes, dict[str, Any]] | None:
    """``(bytes, manifest)`` for a complete entry, or ``None`` for no entry.

    An entry that exists but does not check out is refused, never repaired and
    never quietly requested again: only an explicit refresh replaces bytes that
    are already on disk.
    """
    html_path, manifest_path = paths
    missing = [path for path in paths if not path.exists()]
    if len(missing) == 2:
        return None
    if missing:
        raise KanjiReadingsError(
            f"incomplete cache entry for {url}: {missing[0]} is missing"
        )
    try:
        data = html_path.read_bytes()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise KanjiReadingsError(f"cannot read the cache entry for {url}: {exc}") from exc
    _check_manifest(manifest, character, url, data, manifest_path)
    return data, manifest


def urllib_transport(url: str, timeout: float = TIMEOUT_SECONDS) -> bytes:
    """One GET, with a bounded timeout and no retry."""
    request = Request(url, headers={"User-Agent": USER_AGENT}, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            return bytes(response.read())
    except OSError as exc:
        raise KanjiReadingsError(f"request for {url} failed: {exc}") from exc


@dataclass(frozen=True, slots=True)
class _Page:
    """A page in hand, and whether it still has to be written to the cache."""

    data: bytes
    text: str
    manifest: dict[str, Any]
    fresh: bool


def _decode(data: bytes, source: Any) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise KanjiReadingsError(f"{source}: not valid UTF-8 ({exc})") from exc


def _load_page(
    send: Transport,
    character: str,
    url: str,
    paths: tuple[Path, Path],
    *,
    refresh: bool,
) -> _Page:
    """The cached page, or one fresh request. Nothing is written here."""
    cached = None if refresh else read_cache(paths, character, url)
    if cached is not None:
        data, manifest = cached
        return _Page(data=data, text=_decode(data, paths[0]), manifest=manifest, fresh=False)
    try:
        data = send(url)
    except KanjiReadingsError:
        raise
    except OSError as exc:
        raise KanjiReadingsError(f"request for {url} failed: {exc}") from exc
    if not isinstance(data, bytes | bytearray):
        raise KanjiReadingsError(
            f"the transport answered {url} with {type(data).__name__}, not bytes"
        )
    data = bytes(data)
    fetched_at_utc = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return _Page(
        data=data,
        text=_decode(data, url),
        manifest=_build_manifest(character, url, data, fetched_at_utc),
        fresh=True,
    )


def _publish(paths: tuple[Path, Path], page: _Page) -> None:
    """Write a freshly fetched entry. Callers store only after it parsed."""
    if not page.fresh:
        return
    html_path, manifest_path = paths
    try:
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_bytes(page.data)
        manifest_path.write_text(
            json.dumps(page.manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        raise KanjiReadingsError(
            f"cannot write the cache entry for {page.manifest['source_url']}: {exc}"
        ) from exc


def fetch_character(
    character: str,
    *,
    html_cache: Path,
    refresh: bool = False,
    transport: Transport | None = None,
) -> CharacterReadings:
    """What jpdb reports for one character, from the cache or from requests.

    Saved bytes are reused: a character whose kanji page and quantified reading
    pages are all cached makes no request at all. Missing pages are requested
    one at a time, and ``refresh`` re-requests exactly this character's kanji
    page and the reading pages jpdb printed a percentage beside — never a rare
    or unquantified reading, and never another character.

    Nothing canonical is written. The raw responses land in ``html_cache``; the
    facts come back for a caller to save.
    """
    if len(character) != 1 or not contains_kanji(character):
        raise KanjiReadingsError(f"Not a single kanji character: {character!r}")
    send = transport or urllib_transport
    cache = Path(html_cache)

    url = kanji_page_url(character)
    paths = main_cache_paths(cache, character)
    page = _load_page(send, character, url, paths, refresh=refresh)
    groups = parse_kanji_page(page.text)
    _publish(paths, page)

    bound = tuple(
        _bound_group(group, send, cache, character, refresh=refresh) for group in groups
    )
    return CharacterReadings(
        character=character,
        source_url=url,
        fetched_at_utc=page.manifest["fetched_at_utc"],
        sha256=page.manifest["sha256"],
        groups=bound,
    )


def _bound_group(
    group: ReadingGroup,
    send: Transport,
    cache: Path,
    character: str,
    *,
    refresh: bool,
) -> ReadingGroup:
    """The group with its quantified common readings' detail pages followed.

    Both halves of the rule are here, in one place: the cell jpdb marked common,
    and a figure printed beside that reading. Everything else keeps exactly what
    the kanji page said and costs no request — there is nothing to bind words to
    where jpdb quantified nothing.
    """
    common = group.source_class == COMMON_CELL_CLASS
    return ReadingGroup(
        source_class=group.source_class,
        readings=tuple(
            _bound_reading(usage, send, cache, character, refresh=refresh)
            if common and usage.percent is not None
            else usage
            for usage in group.readings
        ),
    )


def _bound_reading(
    usage: ReadingUsage,
    send: Transport,
    cache: Path,
    character: str,
    *,
    refresh: bool,
) -> ReadingUsage:
    """Follow one reading's own page and keep the words it lists there.

    The kanji page's label and percentage stay exactly as they were printed. The
    reading page states the same figure again, rounded its own way; it is
    provenance for the examples and never a correction to the quantity.
    """
    url = _absolute(usage.href, where=f"the link for reading {usage.label!r}")
    paths = detail_cache_paths(cache, url)
    page = _load_page(send, character, url, paths, refresh=refresh)
    detail = parse_reading_page(page.text)
    # Exact identity, not equivalence: the page has to say it is this character
    # and this reading, character for character, or its words are not this
    # reading's evidence.
    if detail.character != character or detail.reading != usage.label:
        raise KanjiReadingsError(
            f"{url} is the page for {detail.character!r}/{detail.reading!r}, "
            f"not {character!r}/{usage.label!r}"
        )
    _publish(paths, page)
    return replace(
        usage,
        examples=detail.examples[:MAX_BOUND_EXAMPLES],
        detail_source_url=url,
        detail_fetched_at_utc=page.manifest["fetched_at_utc"],
        detail_sha256=page.manifest["sha256"],
    )


# --- the facts store ---------------------------------------------------------


def parse_readings(
    text: str | None, *, source: Any
) -> dict[str, CharacterReadings]:
    """Decode the facts store from exactly ``text``, naming ``source`` in errors.

    The parser this module owns, over text the caller already read. A caller
    that hashed one read gets the facts *that* read held rather than whatever
    a later read of the same path would find, so a write and a restore between
    the two cannot substitute facts nobody bound. ``None`` is the missing file
    :func:`load_readings` reports as an empty store.
    """
    if text is None:
        return {}
    try:
        raw = json.loads(text)
    except ValueError as exc:
        raise KanjiReadingsError(f"Could not read {source}: {exc}") from exc
    fields = _object(raw, f"{source}")
    for key, expected in (
        ("schema_version", SCHEMA_VERSION),
        ("source", SOURCE),
        ("metric", METRIC),
    ):
        if fields.get(key) != expected:
            raise KanjiReadingsError(
                f"{source}: {key} is {fields.get(key)!r}, not {expected!r}"
            )
    characters = _object(fields.get("characters"), f"{source}: characters")
    store = {}
    for character, entry in characters.items():
        fields_for = _object(entry, f"{source}: the entry for {character!r}")
        store[str(character)] = CharacterReadings.from_dict(
            {**fields_for, "character": str(character)}
        )
    return store


def load_readings(path: Any) -> dict[str, CharacterReadings]:
    """Read the facts file. A missing one is an empty store, not an error.

    Nothing here reaches the network: a build reads what an explicit fetch
    already saved, and a character that is not in the file is simply absent.
    """
    file = Path(path)
    try:
        text = read_text_bound(file)
    except FileNotFoundError:
        return {}
    except (JankiError, OSError) as exc:
        raise KanjiReadingsError(f"Could not read {file}: {exc}") from exc
    return parse_readings(text, source=file)


def save_readings(path: Any, store: Mapping[str, CharacterReadings]) -> None:
    """Write the facts file, sorted by character, so a re-fetch diffs cleanly.

    Only the *map* is ordered. Groups, readings and examples keep the order jpdb
    printed them in, which is the only order this store claims anything about.
    """
    characters: dict[str, Any] = {}
    for character in sorted(store):
        entry = store[character]
        if entry.character != character:
            raise KanjiReadingsError(
                f"the store key {character!r} holds the entry for {entry.character!r}"
            )
        payload = entry.to_dict()
        payload.pop("character", None)
        characters[character] = payload
    atomic_write_text_bound(
        Path(path),
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "source": SOURCE,
                "metric": METRIC,
                "characters": characters,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
