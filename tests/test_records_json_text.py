"""Saved collections and prospective package collections use identical bytes.

The canonical saver and `application/deck_package` share `io.records_json_text`
so a package plan's collection digest matches what the saver writes. These
tests cover those two callers; other serialization paths have their own
contracts. S6's planned promotion fold can reuse this same helper.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from japanese_anki import io
from japanese_anki.application import deck_package
from japanese_anki.models import VocabularyRecord


def _record(record_id: str, expression: str, **extra: object) -> VocabularyRecord:
    return VocabularyRecord(
        id=record_id, expression=expression, reading=expression, **extra
    )


def test_saved_collection_bytes_come_from_records_json_text(tmp_path: Path) -> None:
    """The saver writes exactly what the projector computes, byte for byte.

    Mutant: give `save_records_json_locked` its own `json.dumps(...)` again
    instead of calling `records_json_text`, with any argument differing (drop
    `indent=2`, or the trailing newline). The digests stop matching.
    """
    records = [_record("b", "二"), _record("a", "一")]
    path = tmp_path / "records.json"

    io.save_records_json(path, records)

    assert path.read_text(encoding="utf-8") == io.records_json_text(records)


def test_records_json_text_sorts_by_id_so_a_digest_is_order_independent(
    tmp_path: Path,
) -> None:
    """Two readings of the same collection hash the same however it was built.

    Mutant: drop the `sorted(...)` in `records_json_text` and serialize
    `records` in the order given. The two orderings then produce different
    bytes, and a fold's carry would stop matching its own checkpoint.
    """
    forwards = [_record("a", "一"), _record("b", "二")]
    backwards = list(reversed(forwards))

    assert io.records_json_text(forwards) == io.records_json_text(backwards)


def test_records_json_text_keeps_japanese_unescaped(tmp_path: Path) -> None:
    """`ensure_ascii=False` is contract, not preference.

    Mutant: drop `ensure_ascii=False`. Every expression becomes a `\\uXXXX`
    escape, the file stops being readable, and every stored digest moves.
    """
    text = io.records_json_text([_record("a", "食べる")])

    assert "食べる" in text
    assert "\\u" not in text


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_records_json_text_refuses_a_non_finite_number(value: str) -> None:
    """`allow_nan=False`, so no bare `NaN`/`Infinity` literal reaches the text.

    RFC 8259 defines none of them, so what a reader makes of a collection
    carrying one depends on the reader. A projected collection is built in
    memory, which is exactly how a bug upstream arrives here.

    The encoder's own `ValueError` comes back unwrapped: `save_records_json`
    has always raised the encoder's exceptions, and `test_io_atomic` pins that.

    Mutant: drop `allow_nan=False` from `records_json_text`. The call returns
    `"frequency_rank": NaN` instead of raising.
    """
    broken = _record("a", "一", frequency_rank=float(value))

    with pytest.raises(ValueError) as refusal:
        io.records_json_text([broken])

    assert "not JSON compliant" in str(refusal.value)


def test_saving_a_non_finite_number_replaces_no_file(tmp_path: Path) -> None:
    """The refusal reaches the writer before it replaces anything.

    Mutant: have `save_records_json_locked` serialize with its own
    `json.dumps` without `allow_nan=False` again. The good collection on disk
    is replaced by one whose `frequency_rank` is the bare literal `Infinity`.
    """
    path = tmp_path / "records.json"
    io.save_records_json(path, [_record("a", "一")])
    good = path.read_text(encoding="utf-8")

    with pytest.raises(ValueError):
        io.save_records_json(path, [_record("a", "一", frequency_rank=float("inf"))])

    assert path.read_text(encoding="utf-8") == good


def test_package_planning_refuses_the_same_records_in_its_own_error(
    tmp_path: Path,
) -> None:
    """The planner keeps its own refusal while sharing the one serializer.

    `_canonical_records_text` delegates, so the bytes agree; it re-raises the
    encoder's `ValueError` as `DeckPackageError` so a caller still learns which
    planning step refused.

    Mutant: have `_canonical_records_text` let the `ValueError` through
    unwrapped. A package planning failure then surfaces as a bare encoder error
    with no indication that package planning was what refused.
    """
    broken = _record("a", "一", frequency_rank=float("nan"))

    with pytest.raises(deck_package.DeckPackageError) as refusal:
        deck_package._canonical_records_text([broken])

    assert "Prospective package records cannot be serialized exactly" in str(
        refusal.value
    )


def test_package_planning_and_the_saver_agree_byte_for_byte(tmp_path: Path) -> None:
    """The digest the package plans over is the digest a save would produce.

    Mutant: give `_canonical_records_text` back its own `json.dumps(...)` with
    any differing argument. `_plan_vocabulary_revision_unlocked` requires
    `revision.text == _canonical_records_text(records)`, so the two spellings
    drifting apart is what strands a package plan against its own collection.
    """
    records = [_record("b", "二"), _record("a", "一")]
    path = tmp_path / "records.json"
    io.save_records_json(path, records)

    assert deck_package._canonical_records_text(records) == path.read_text(
        encoding="utf-8"
    )
    assert json.loads(deck_package._canonical_records_text(records))[0]["id"] == "a"
