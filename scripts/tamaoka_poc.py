#!/usr/bin/env python3
"""Read the durable Tamaoka kanji-database export and report its own arithmetic.

Offline by default: the saved CSV snapshot is parsed and echoed as JSON, with
per-side sums of the named left/right slot counts alongside the accumulated
frequencies the export itself reports. Nothing is normalised, converted,
merged, ranked, or repaired -- source strings are passed through verbatim and
any disagreement is reported, not fixed. ``--fetch`` re-runs the saved export
request and stores a new snapshot; it never touches an existing file.
"""

import argparse
import csv
import io
import json
import os
import sys
import urllib.parse
from pathlib import Path
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = REPO_ROOT / "data" / "research" / "tamaoka" / "2026-09-07"
DEFAULT_CSV = SNAPSHOT_DIR / "export.csv"
REQUEST_JSON = SNAPSHOT_DIR / "export-request.json"
EXPORT_URL = "https://www.kanjidatabase.com/php/export.php"
TIMEOUT_SECONDS = 30

READING_FIELDS = {
    "reading_within_joyo": "Reading within Joyo",
    "on_within_joyo": "On within Joyo",
    "kun_within_joyo": "Kun within Joyo",
}
REPORTED_FIELDS = {
    "left": "Acc. Freq. Left Prod.",
    "right": "Acc. Freq. Right Prod.",
}
SIDE_SLOTS = {
    side: [
        (slot, f"{side.capitalize()}{slot}sound", f"{side.capitalize()}{slot}freq")
        for slot in range(1, count + 1)
    ]
    for side, count in (("left", 6), ("right", 7))
}
REQUIRED_FIELDS = [
    "id",
    "Kanji",
    *READING_FIELDS.values(),
    *REPORTED_FIELDS.values(),
    *(
        field
        for slots in SIDE_SLOTS.values()
        for _, sound_field, count_field in slots
        for field in (sound_field, count_field)
    ),
]


class PocError(Exception):
    """A refusal: the input is not an export this POC will read."""


def _count(raw, where, field):
    if raw == "":
        raise PocError(f"{where}: {field} is blank; counts must be non-negative integers")
    if not (raw.isascii() and raw.isdigit()):
        raise PocError(f"{where}: {field} is {raw!r}; counts must be non-negative integers")
    return int(raw)


def _record(index, row):
    where = f"row {index}"
    kanji = row.get("Kanji")
    if kanji:
        where = f"{where} ({kanji})"
    if None in row or any(value is None for value in row.values()):
        raise PocError(f"{where}: field count does not match the header")

    record = {"id": row["id"], "kanji": row["Kanji"]}
    record.update((key, row[field]) for key, field in READING_FIELDS.items())
    for side, slots in SIDE_SLOTS.items():
        entries = []
        named_sum = 0
        for slot, sound_field, count_field in slots:
            sound = row[sound_field]
            count = _count(row[count_field], where, count_field)
            entries.append({"slot": slot, "sound": sound, "count": count})
            if sound:
                named_sum += count
        reported = _count(row[REPORTED_FIELDS[side]], where, REPORTED_FIELDS[side])
        record[side] = {
            "slots": entries,
            "named_sum": named_sum,
            "reported": reported,
            "matches_reported": named_sum == reported,
        }
    return record


def parse_export(text):
    """Parse a decoded semicolon-delimited export into per-kanji records."""
    reader = csv.DictReader(io.StringIO(text, newline=""), delimiter=";")
    try:
        header = reader.fieldnames or []
        duplicates = [name for name in dict.fromkeys(header) if header.count(name) > 1]
        if duplicates:
            raise PocError("export has duplicate column(s): " + ", ".join(duplicates))
        rows = list(reader)
    except csv.Error as exc:
        raise PocError(f"not a readable semicolon-delimited CSV: {exc}") from exc
    missing = [field for field in REQUIRED_FIELDS if field not in header]
    if missing:
        raise PocError("export is missing required column(s): " + ", ".join(missing))
    return [_record(index, row) for index, row in enumerate(rows, start=1)]


def _decode(data, source):
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise PocError(f"{source}: not valid UTF-8 ({exc})") from exc


def load_export(path):
    """Parse the export stored at ``path``."""
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise PocError(f"cannot read {path}: {exc}") from exc
    text = _decode(data, path)
    try:
        return parse_export(text)
    except PocError as exc:
        raise PocError(f"{path}: {exc}") from exc


def fetch_export(destination):
    """Re-run the saved export request and store the response at ``destination``."""
    destination = Path(destination)
    if destination.exists():
        raise PocError(f"refusing to overwrite existing file: {destination}")
    try:
        payload = json.loads(REQUEST_JSON.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PocError(f"cannot read saved request {REQUEST_JSON}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PocError(f"saved request {REQUEST_JSON} is not a JSON object")

    request = Request(
        EXPORT_URL,
        data=urllib.parse.urlencode(payload).encode("utf-8"),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            data = response.read()
    except OSError as exc:
        raise PocError(f"export request to {EXPORT_URL} failed: {exc}") from exc

    records = parse_export(_decode(data, EXPORT_URL))
    try:
        handle = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except OSError as exc:
        raise PocError(f"cannot create {destination}: {exc}") from exc
    with os.fdopen(handle, "wb") as sink:
        sink.write(data)
    return records


def build_report(records):
    return {"frequency_status": "unverified", "usage_percentages": None, "rows": records}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--csv", type=Path, help=f"export to read instead of {DEFAULT_CSV}")
    source.add_argument(
        "--fetch",
        type=Path,
        metavar="NEWPATH",
        help="re-run the saved export request and store the response at NEWPATH",
    )
    args = parser.parse_args(argv)
    try:
        records = (
            fetch_export(args.fetch)
            if args.fetch is not None
            else load_export(args.csv or DEFAULT_CSV)
        )
    except PocError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    json.dump(build_report(records), sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
