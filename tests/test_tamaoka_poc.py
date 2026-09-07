"""Tests for the standalone Tamaoka export POC (``scripts/tamaoka_poc.py``)."""

import csv
import importlib.util
import json
import urllib.parse
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_poc():
    """Load the script as a standalone module; it is not part of the package."""
    spec = importlib.util.spec_from_file_location(
        "tamaoka_poc", REPO_ROOT / "scripts" / "tamaoka_poc.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


poc = _load_poc()

# (left named_sum, right named_sum) computed by hand from the durable snapshot.
SNAPSHOT_NAMED_SUMS = {
    "物": (60, 113),
    "特": (162, 4),
    "鳥": (656, 490),
    "料": (39, 99),
    "理": (163, 205),
    "生": (889, 366),
    "行": (934, 385),
}


@pytest.fixture(scope="module")
def snapshot():
    return {record["kanji"]: record for record in poc.load_export(poc.DEFAULT_CSV)}


def test_named_sums_add_up_the_named_slot_counts(snapshot):
    named_sums = {
        kanji: (record["left"]["named_sum"], record["right"]["named_sum"])
        for kanji, record in snapshot.items()
    }
    assert named_sums == SNAPSHOT_NAMED_SUMS


def test_reported_totals_are_read_per_side_and_compared_directly(snapshot):
    ri = snapshot["理"]
    assert ri["left"]["reported"] == 205212  # not the right-hand 163378
    assert ri["right"]["reported"] == 163378
    assert ri["left"]["matches_reported"] is False
    assert ri["right"]["matches_reported"] is False
    mono = snapshot["物"]
    assert (mono["left"]["reported"], mono["right"]["reported"]) == (130487, 58706)


def test_slots_keep_raw_labels_order_and_empty_slots(snapshot):
    assert snapshot["物"]["left"]["slots"] == [
        {"slot": 1, "sound": "buQ", "count": 34},
        {"slot": 2, "sound": "mono", "count": 17},
        {"slot": 3, "sound": "butu", "count": 6},
        {"slot": 4, "sound": "moQ", "count": 3},
        {"slot": 5, "sound": "", "count": 0},
        {"slot": 6, "sound": "", "count": 0},
    ]
    assert snapshot["鳥"]["right"]["slots"] == [
        {"slot": 1, "sound": "tyoo", "count": 4},
        {"slot": 2, "sound": "tori", "count": 1},
        {"slot": 3, "sound": "dori", "count": 485},
        *({"slot": slot, "sound": "", "count": 0} for slot in range(4, 8)),
    ]


def test_reading_strings_are_copied_verbatim(snapshot):
    # Compare against the raw fields, so any stripping, kana conversion, or
    # splitting on the way through shows up as a mismatch.
    lines = poc.DEFAULT_CSV.read_text(encoding="utf-8-sig").splitlines()
    header = next(csv.reader([lines[0]], delimiter=";"))
    raw = dict(zip(header, next(csv.reader([lines[1]], delimiter=";")), strict=True))
    mono = snapshot["物"]
    assert mono["id"] == raw["id"] == "1777"
    assert mono["reading_within_joyo"] == raw["Reading within Joyo"] == "ブツ、モツ、もの"
    assert mono["on_within_joyo"] == raw["On within Joyo"]  # inner whitespace and all
    assert mono["kun_within_joyo"] == raw["Kun within Joyo"] == "mono"
    assert snapshot["料"]["kun_within_joyo"] == "-"  # the placeholder is not repaired away


def _row(kanji="物", *, left=(), right=(), reported=(0, 0), reading=""):
    row = {"id": "1", "Kanji": kanji}
    for field in poc.READING_FIELDS.values():
        row[field] = reading
    for side, slots, total in (("left", left, reported[0]), ("right", right, reported[1])):
        for slot, sound_field, count_field in poc.SIDE_SLOTS[side]:
            sound, count = slots[slot - 1] if slot <= len(slots) else ("", 0)
            row[sound_field] = sound
            row[count_field] = str(count)
        row[poc.REPORTED_FIELDS[side]] = str(total)
    return row


def _write_csv(path, rows, fields=None):
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fields or poc.REQUIRED_FIELDS), delimiter=";"
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def _valid_rows():
    return [_row(kanji="物") for _ in range(7)]


def test_report_flags_stay_fixed_even_when_every_side_matches(tmp_path):
    rows = [
        _row(kanji="理", left=[("ri", 163)], right=[("ri", 205)], reported=(163, 205))
        for _ in range(7)
    ]
    report = poc.build_report(poc.load_export(_write_csv(tmp_path / "matching.csv", rows)))
    assert report["frequency_status"] == "unverified"
    assert report["usage_percentages"] is None
    assert all(
        record[side]["matches_reported"]
        for record in report["rows"]
        for side in ("left", "right")
    )


def test_bom_and_quoted_semicolons_survive_parsing(tmp_path):
    reading = "ブツ;モツ;もの"
    path = _write_csv(tmp_path / "quoted.csv", [_row(reading=reading) for _ in range(7)])
    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    assert b'"' + reading.encode("utf-8") + b'"' in raw
    assert poc.load_export(path)[0]["reading_within_joyo"] == reading


@pytest.mark.parametrize("count", [1, 6, 8])
def test_any_number_of_rows_is_read_and_every_row_preserved(tmp_path, count):
    # Each row carries its own id and left-hand count, so a dropped, duplicated,
    # or reordered row shows up rather than hiding behind a matching length.
    rows = [
        _row(kanji="物", left=[("buQ", index)], reported=(index, 0))
        for index in range(1, count + 1)
    ]
    for index, row in enumerate(rows, start=1):
        row["id"] = str(index)
    path = _write_csv(tmp_path / f"{count}-rows.csv", rows)

    records = poc.load_export(path)

    assert [record["id"] for record in records] == [str(i) for i in range(1, count + 1)]
    assert [record["left"]["named_sum"] for record in records] == list(range(1, count + 1))


def test_missing_required_column_is_refused(tmp_path):
    fields = [field for field in poc.REQUIRED_FIELDS if field != "Left3sound"]
    rows = [{k: v for k, v in row.items() if k != "Left3sound"} for row in _valid_rows()]
    path = _write_csv(tmp_path / "missing-column.csv", rows, fields)
    with pytest.raises(poc.PocError, match="Left3sound"):
        poc.load_export(path)


def test_short_row_is_refused(tmp_path):
    path = _write_csv(tmp_path / "short-row.csv", _valid_rows())
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    lines[3] = ";".join(lines[3].split(";")[:-4])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    with pytest.raises(poc.PocError) as excinfo:
        poc.load_export(path)
    assert "row 3" in str(excinfo.value)


@pytest.mark.parametrize(
    "field, value",
    [
        ("Left1freq", ""),
        ("Left1freq", "-1"),
        ("Left1freq", "1.5"),
        ("Left1freq", "many"),
        ("Acc. Freq. Right Prod.", ""),
        ("Acc. Freq. Left Prod.", "n/a"),
    ],
)
def test_blank_or_invalid_counts_are_refused(tmp_path, field, value):
    rows = _valid_rows()
    rows[2][field] = value
    path = _write_csv(tmp_path / "counts.csv", rows)
    with pytest.raises(poc.PocError) as excinfo:
        poc.load_export(path)
    message = str(excinfo.value)
    assert field in message
    assert "row 3" in message
    assert str(path) in message


def test_duplicate_column_is_refused(tmp_path):
    # A repeated header silently wins over the first copy in csv.DictReader, so
    # the real 7 would be read as 999. Written with csv.writer because
    # DictWriter would give both copies the same mapping value, and the two
    # copies need different values to show the overwrite.
    fields = [*poc.REQUIRED_FIELDS, "Left1freq"]
    row = _row(kanji="物", left=[("buQ", 7)])
    assert row["Left1freq"] == "7"
    path = tmp_path / "duplicate-column.csv"
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(fields)
        writer.writerow([*(row[field] for field in poc.REQUIRED_FIELDS), "999"])

    with pytest.raises(poc.PocError) as excinfo:
        poc.load_export(path)
    message = str(excinfo.value)
    assert "duplicate" in message.lower()
    assert "Left1freq" in message


class _FakeResponse:
    def __init__(self, data):
        self._data = data

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def read(self):
        return self._data


def _stub_urlopen(monkeypatch, data=b""):
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append((request, timeout))
        return _FakeResponse(data)

    monkeypatch.setattr(poc, "urlopen", fake_urlopen)
    return calls


def test_fetch_posts_the_saved_request_and_stores_the_exact_bytes(tmp_path, monkeypatch):
    body = poc.DEFAULT_CSV.read_bytes()
    calls = _stub_urlopen(monkeypatch, body)
    destination = tmp_path / "new-export.csv"

    records = poc.fetch_export(destination)

    assert len(calls) == 1
    request, timeout = calls[0]
    assert request.full_url == "https://www.kanjidatabase.com/php/export.php"
    assert request.get_method() == "POST"
    saved_request = json.loads(poc.REQUEST_JSON.read_text(encoding="utf-8"))
    assert request.data == urllib.parse.urlencode(saved_request).encode("utf-8")
    assert timeout == poc.TIMEOUT_SECONDS > 0
    assert destination.read_bytes() == body  # BOM and newlines untouched
    assert len(records) == 7


def test_fetch_writes_nothing_when_the_response_is_not_a_valid_export(tmp_path, monkeypatch):
    calls = _stub_urlopen(monkeypatch, b"<html><body>service unavailable</body></html>")
    destination = tmp_path / "new-export.csv"

    with pytest.raises(poc.PocError):
        poc.fetch_export(destination)

    assert len(calls) == 1
    assert not destination.exists()


def test_fetch_refuses_an_existing_destination_without_requesting(tmp_path, monkeypatch):
    calls = _stub_urlopen(monkeypatch, poc.DEFAULT_CSV.read_bytes())
    destination = tmp_path / "new-export.csv"
    destination.write_bytes(b"existing snapshot\n")

    with pytest.raises(poc.PocError):
        poc.fetch_export(destination)

    assert calls == []
    assert destination.read_bytes() == b"existing snapshot\n"


def test_csv_and_fetch_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        poc.main(["--csv", "a.csv", "--fetch", "b.csv"])


def test_main_prints_a_single_report_object(capsys):
    assert poc.main([]) == 0
    report = json.loads(capsys.readouterr().out)
    assert set(report) == {"frequency_status", "usage_percentages", "rows"}
    assert len(report["rows"]) == 7


def test_main_reports_validation_failures_on_stderr(tmp_path, capsys):
    rows = _valid_rows()
    rows[2]["Left1freq"] = "many"
    path = _write_csv(tmp_path / "bad-count.csv", rows)
    assert poc.main(["--csv", str(path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip() != ""
