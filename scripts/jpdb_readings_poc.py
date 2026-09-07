#!/usr/bin/env python3
"""Report the reading links a jpdb kanji page prints, exactly as printed.

Offline by default: every requested character is read out of ``--cache-dir``
and re-parsed. ``--fetch`` makes one GET for a requested character that has no
cache entry; ``--refresh`` (which requires ``--fetch``) re-requests one that
does. There is no retry, no crawling, no background work and no concurrency: a
failed request or an unparseable answer is reported and whatever was cached is
left exactly as it was.

Nothing here interprets Japanese. The anchor's text, its href and the
percentage string jpdb printed beside it are passed through verbatim, each
reading keeps its source order and the class of the cell it came from, and
nothing is normalised, renormalised, sorted, merged, classified, or inferred.
A percentage is jpdb's own rounded figure, not a measurement this POC made, and
``<1%`` is kept as the upper bound it is. The vocabulary and example sections
elsewhere on the page are not tied to any reading, so they are not reported.
Structure inside a target cell that this parser does not recognise is refused
rather than repaired or silently dropped; the rest of the page is skipped
without being judged.
"""

import argparse
import hashlib
import json
import re
import sys
import urllib.parse
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen

SCHEMA_VERSION = 1
BASE_URL = "https://jpdb.io/kanji/"
TIMEOUT_SECONDS = 30
USER_AGENT = "janki-jpdb-readings-poc/1"
READING_CELL_CLASSES = ("kanji-reading-list-common", "kanji-reading-list")
MANIFEST_FIELDS = ("schema_version", "character", "source_url", "fetched_at_utc", "sha256")

# (84%) and its upper-bound sibling (<1%), the two shapes jpdb prints. The
# source's own parentheses are optional here and preserved in `percent_text`.
_PERCENTAGE = re.compile(r"(?:(?P<bound><|less than)\s*)?(?P<value>[0-9]+)%")


class PocError(Exception):
    """A refusal: this is not a page, cache entry, or argument this POC will use."""


def page_url(character):
    """The page URL for ``character``, and the cache entry's source identity."""
    return BASE_URL + urllib.parse.quote(character, safe="")


def _percentage(text, where):
    """Read one printed percentage as ``(value, is_upper_bound)``."""
    inner = text.strip()
    if inner.startswith("(") and inner.endswith(")"):
        inner = inner[1:-1].strip()
    match = _PERCENTAGE.fullmatch(inner)
    if match is None:
        raise PocError(f"{where}: unrecognised percentage {text!r}")
    return int(match["value"]), match["bound"] is not None


class _ReadingCellParser(HTMLParser):
    """Collect the reading cells of a kanji page and skip everything else.

    Only markup *inside* a target cell is judged, so the unrelated Info rows,
    the vocabulary list and the examples pass by untouched: this is a reader
    for two cells, not a strict HTML validator.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.groups = []
        self._group = None  # the cell being read
        self._reading = None  # the wrapper div being read
        self._capture = None  # "label" or "percent" while inside one
        self._chunks = []

    @property
    def unfinished(self):
        """True when a target cell was opened and never closed."""
        return self._group is not None

    def _refuse(self, problem):
        where = self._group["source_class"] if self._group else "reading cell"
        raise PocError(f"{where}: {problem} (line {self.getpos()[0]})")

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "td":
            if self._group is not None:
                self._refuse("a new cell begins before this one is closed")
            classes = (attributes.get("class") or "").split()
            matched = [name for name in READING_CELL_CLASSES if name in classes]
            if len(matched) > 1:
                raise PocError("cell carries several reading classes: " + " ".join(matched))
            if matched:
                self._group = {"source_class": matched[0], "readings": []}
            return
        if self._group is None:
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

    def handle_endtag(self, tag):
        if tag == "td":
            if self._group is None:
                return
            if self._reading is not None or self._capture is not None:
                self._refuse("cell ends inside an unclosed reading wrapper")
            if not self._group["readings"]:
                self._refuse("cell holds no readings")
            self.groups.append(self._group)
            self._group = None
            return
        if self._group is None:
            return
        if self._capture == "label":
            if tag != "a":
                self._refuse(f"</{tag}> inside a reading link")
            self._reading["label"] = "".join(self._chunks)
            self._capture = None
        elif self._capture == "percent":
            if tag != "div":
                self._refuse(f"</{tag}> inside a percentage")
            self._reading["percent_text"] = "".join(self._chunks)
            self._capture = None
        elif self._reading is not None:
            if tag != "div":
                self._refuse(f"unexpected </{tag}> in a reading wrapper")
            self._group["readings"].append(self._finish(self._reading))
            self._reading = None
        else:
            self._refuse(f"unexpected </{tag}> in the cell")

    def handle_data(self, data):
        if self._capture is not None:
            self._chunks.append(data)
        elif self._group is not None and data.strip():
            self._refuse(f"stray text {data.strip()!r} beside the readings")

    def _finish(self, reading):
        if reading["label"] is None:
            self._refuse("reading wrapper without a link")
        text = reading["percent_text"]
        percent, bound = (
            (None, None) if text is None else _percentage(text, self._group["source_class"])
        )
        return {
            "label": reading["label"],
            "href": reading["href"],
            "percent_text": text,
            "percent": percent,
            "percent_less_than": bound,
        }


def parse_page(text):
    """Parse a decoded kanji page into its reading groups, in source order."""
    parser = _ReadingCellParser()
    parser.feed(text)
    parser.close()
    if parser.unfinished:
        raise PocError("the page ends inside an unclosed reading cell")
    if not parser.groups:
        raise PocError("no reading cell on the page: " + " or ".join(READING_CELL_CLASSES))
    return parser.groups


def _read(data, source):
    """Decode and parse ``data``, naming ``source`` in any refusal."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PocError(f"{source}: not valid UTF-8 ({exc})") from exc
    try:
        return parse_page(text)
    except PocError as exc:
        raise PocError(f"{source}: {exc}") from exc


def cache_paths(cache_dir, character):
    """The entry's two names: the hex code point as ``.html`` and ``.json``."""
    stem = Path(cache_dir) / f"{ord(character):x}"
    return stem.with_suffix(".html"), stem.with_suffix(".json")


def build_manifest(character, data, fetched_at_utc):
    return {
        "schema_version": SCHEMA_VERSION,
        "character": character,
        "source_url": page_url(character),
        "fetched_at_utc": fetched_at_utc,
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _check_manifest(manifest, character, data, path):
    """Refuse an entry that is not this character's page, byte for byte."""
    if not isinstance(manifest, dict) or set(manifest) != set(MANIFEST_FIELDS):
        raise PocError(f"{path}: manifest fields are not exactly {', '.join(MANIFEST_FIELDS)}")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise PocError(
            f"{path}: schema_version is {manifest['schema_version']!r}, not {SCHEMA_VERSION}"
        )
    if manifest["character"] != character:
        raise PocError(f"{path}: character is {manifest['character']!r}, not {character!r}")
    if manifest["source_url"] != page_url(character):
        raise PocError(
            f"{path}: source_url is {manifest['source_url']!r}, not {page_url(character)!r}"
        )
    if not isinstance(manifest["fetched_at_utc"], str) or not manifest["fetched_at_utc"]:
        raise PocError(f"{path}: fetched_at_utc is not a timestamp")
    if manifest["sha256"] != hashlib.sha256(data).hexdigest():
        raise PocError(f"{path}: sha256 does not match the cached page bytes")


def read_cache(cache_dir, character):
    """Return ``(bytes, manifest)`` for a complete entry, or ``None`` for no entry.

    An entry that exists but does not check out is refused, never repaired and
    never quietly requested again: only ``--refresh`` replaces cached bytes.
    """
    html_path, manifest_path = cache_paths(cache_dir, character)
    missing = [path for path in (html_path, manifest_path) if not path.exists()]
    if len(missing) == 2:
        return None
    if missing:
        raise PocError(f"incomplete cache entry for {character!r}: {missing[0]} is missing")
    try:
        data = html_path.read_bytes()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PocError(f"cannot read the cache entry for {character!r}: {exc}") from exc
    _check_manifest(manifest, character, data, manifest_path)
    return data, manifest


def fetch_page(character):
    """One GET for ``character``, with a bounded timeout and no retry."""
    url = page_url(character)
    request = Request(url, headers={"User-Agent": USER_AGENT}, method="GET")
    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            data = response.read()
    except OSError as exc:
        raise PocError(f"request for {url} failed: {exc}") from exc
    fetched_at_utc = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return data, build_manifest(character, data, fetched_at_utc)


def store_cache(cache_dir, character, data, manifest):
    """Write the entry. Callers store only after the bytes parsed."""
    html_path, manifest_path = cache_paths(cache_dir, character)
    try:
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_bytes(data)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        raise PocError(f"cannot write the cache entry for {character!r}: {exc}") from exc


def character_entry(character, cache_dir, *, fetch=False, refresh=False):
    """Report one character from its cache entry, or from one fresh request."""
    html_path, _ = cache_paths(cache_dir, character)
    cached = None if refresh else read_cache(cache_dir, character)
    if cached is not None:
        data, manifest = cached
        groups = _read(data, html_path)
    else:
        if not fetch:
            raise PocError(f"no cache entry at {html_path}; pass --fetch to request the page")
        data, manifest = fetch_page(character)
        groups = _read(data, manifest["source_url"])
        store_cache(cache_dir, character, data, manifest)
    return {"character": character, "provenance": manifest, "groups": groups}


def build_report(entries):
    """Wrap the entries in a header saying what the figures below are.

    ``metric`` names the quantity: the usage share jpdb itself reports beside a
    reading. ``corpus_scope`` and ``denominator`` are null because jpdb prints
    neither, and a reader who needs them must get them from jpdb rather than
    from a guess made here.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "source": "jpdb",
        "metric": "jpdb_reported_usage",
        "corpus_scope": None,
        "denominator": None,
        "characters": list(entries),
    }


def _check_arguments(args):
    if args.refresh and not args.fetch:
        raise PocError("--refresh requires --fetch")
    for character in args.characters:
        if len(character) != 1:
            raise PocError(f"{character!r} is not a single character")
    repeated = [c for c in dict.fromkeys(args.characters) if args.characters.count(c) > 1]
    if repeated:
        raise PocError("repeated character argument: " + " ".join(repeated))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("characters", nargs="+", metavar="CHAR", help="one character per argument")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        metavar="DIR",
        help="directory holding <hex code point>.html and .json per character",
    )
    parser.add_argument(
        "--fetch", action="store_true", help="request a page that has no cache entry"
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="with --fetch, request every named character again and replace its entry",
    )
    args = parser.parse_args(argv)
    try:
        _check_arguments(args)
        entries = [
            character_entry(character, args.cache_dir, fetch=args.fetch, refresh=args.refresh)
            for character in args.characters
        ]
    except PocError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    json.dump(build_report(entries), sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
