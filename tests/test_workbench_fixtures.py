"""W0: an offline demo corpus the workbench milestones build against.

``docs/WORKBENCH_PLAN.md`` task W0. Every file under
``tests/fixtures/workbench/responses/`` is a raw dict shaped like a real
Claude structured-output answer to ``janki extract`` — the same schema
``extract.candidate_schema()`` validates a live response against. This module
loads each one, drives it through the *real* pipeline functions
(``extract.build_records``, ``staging.write_staging``, ``patterns.save_store``,
``promote.check_readings``, ``exporters.anki.resolve_deck_records``,
``workbench.review.ReviewPanel``) exactly as ``cli.command_extract`` would, and
asserts the derived staging/pattern/record/deck state is what each scenario
promises. No network call, no live model: the fixture *is* the model's
answer.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from japanese_anki import extract, patterns, promote
from japanese_anki.exporters.anki import resolve_deck_records
from japanese_anki.identifiers import stable_record_id
from japanese_anki.inputs import PreparedInput
from japanese_anki.jpdb import JpdbClient
from japanese_anki.promote import HOLD_MISSING_READING, HOLD_UNKNOWN_READING
from japanese_anki.staging import CANDIDATE_ACCOUNTING_KEY, new_review_run_id, write_staging
from japanese_anki.workbench.review import ReviewPanel

FIXTURES = Path(__file__).parent / "fixtures" / "workbench"
RESPONSES = FIXTURES / "responses"
DECKS = FIXTURES / "decks"

PDF = b"%PDF-1.7 fake"


def _prepared(tmp_path: Path, name: str) -> PreparedInput:
    path = tmp_path / "inbox" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(PDF)
    return PreparedInput(
        kind="document",
        media_type="application/pdf",
        data_b64="ZmFrZQ==",
        origin_path=path,
    )


def _load(scenario: str) -> object:
    """The scenario's raw dict, validated against the live extraction schema."""
    raw = json.loads((RESPONSES / f"{scenario}.json").read_text(encoding="utf-8"))
    return extract.candidate_schema()(**raw)


def materialize(tmp_path: Path, scenario: str, *, filename: str | None = None) -> dict:
    """Replicate ``cli.command_extract``'s write sequence for one fixture answer.

    Returns the staging path, pattern-store path, built records, and meta —
    the "staging, pattern, record ... state" W0 asks each fixture to produce.
    """
    item = _prepared(tmp_path, filename or f"{scenario}.pdf")
    parsed = _load(scenario)
    normalized = extract.normalize_response(parsed, None, item.origin_path.name)
    provenance = extract.prompt_provenance(
        item,
        model="claude-opus-5",
        style_guide="fixture style guide",
        system="fixture system prompt",
        mode=None,
        source_sha256=extract.source_fingerprint(item.origin_path),
    )
    result = replace(
        normalized,
        pattern_set=patterns.with_prompt_provenance(normalized.pattern_set, provenance),
    )

    built = extract.build_records(result.candidates, item, known_ids=set())
    coverage = extract.coverage_block(
        result,
        source_sha256=provenance["source_sha256"],
        mode=None,
        candidate_accounting=built.candidate_accounting,
    )

    run_id = new_review_run_id()
    run_patterns = replace(result.pattern_set, review_run_id=run_id)
    meta: dict[str, object] = {
        "source_file": item.origin_path.name,
        "extracted_at": "2026-08-20",
        "model": "claude-opus-5",
        "review_run_id": run_id,
        "prompt_provenance": provenance,
        "pattern_set": run_patterns.to_dict(),
        "coverage": coverage,
    }
    held = list(built.unusable_candidates)
    if held:
        meta["review_notes"] = extract.unusable_note(held)
    meta[CANDIDATE_ACCOUNTING_KEY] = built.candidate_accounting

    staging_dir = tmp_path / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    staging_path = staging_dir / f"{item.origin_path.name}.yaml"
    write_staging(staging_path, built.records, meta)

    patterns_path = tmp_path / "patterns.json"
    store = {run_patterns.source: run_patterns}
    patterns_path.parent.mkdir(parents=True, exist_ok=True)
    patterns.save_store(patterns_path, store)

    return {
        "item": item,
        "records": built.records,
        "meta": meta,
        "staging_path": staging_path,
        "patterns_path": patterns_path,
        "store": store,
        "run_patterns": run_patterns,
    }


# --- a small exhaustive vocabulary table ------------------------------------


def test_table_exhaustive_materializes_every_row(tmp_path: Path) -> None:
    state = materialize(tmp_path, "table_exhaustive")

    assert [record.expression for record in state["records"]] == [
        "走る",
        "食べる",
        "飲む",
    ]
    assert state["meta"]["coverage"]["blocking"] is True
    assert state["meta"]["coverage"]["observed_unit_count"] == 3
    # A table candidate must be table-linked to its source unit one-to-one;
    # normalize_response already enforced that when it built the answer.
    assert state["meta"][CANDIDATE_ACCOUNTING_KEY]["parsed_candidate_count"] == 3


# --- a lesson teaching vocabulary and grammar together ----------------------


def test_lesson_with_grammar_has_records_and_an_unreviewed_pattern_set(
    tmp_path: Path,
) -> None:
    state = materialize(tmp_path, "lesson_with_grammar")

    assert {record.expression for record in state["records"]} == {"あげる", "もらう"}
    run_patterns = state["run_patterns"]
    assert [pattern.template for pattern in run_patterns.patterns] == [
        "〜てあげる",
        "〜てもらう",
    ]
    assert run_patterns.kind == "lesson"


# --- a pattern-only chart: zero vocabulary candidates -----------------------


def test_pattern_only_chart_writes_patterns_and_no_records(tmp_path: Path) -> None:
    state = materialize(tmp_path, "pattern_only_chart")

    assert state["records"] == ()
    run_patterns = state["run_patterns"]
    assert len(run_patterns.patterns) == 2
    assert run_patterns.kind == "pattern"
    # The dashboard's "Grammar saved — no word cards to build" state (W1.1)
    # is exactly this shape: patterns present, records empty.


# --- same spelling, different reading: two distinct stable IDs -------------


def test_same_spelling_different_reading_are_distinct_ids(tmp_path: Path) -> None:
    state = materialize(tmp_path, "same_spelling_different_reading")

    ids = [record.id for record in state["records"]]
    assert ids == ["word:一日:いちにち", "word:一日:ついたち"]
    assert len(set(ids)) == 2


def test_same_spelling_different_reading_are_separately_reviewable(
    tmp_path: Path,
) -> None:
    """Two readings of 一日 are two separate cards, and the review transaction
    offers both independently — the case a reviewer must not be able to
    collapse by accident. (How they *render* is the workbench's concern now;
    `tests/test_workbench.py` covers the page.)"""
    state = materialize(tmp_path, "same_spelling_different_reading")

    panel = ReviewPanel.open(
        state["staging_path"],
        staging_dir=state["staging_path"].parent,
        patterns_path=state["patterns_path"],
    )

    assert panel.reviewable_record_ids == {
        "word:一日:いちにち",
        "word:一日:ついたち",
    }


# --- one stable ID proposed by two sources, two candidate decks ------------


def test_shared_word_from_two_sources_collides_on_one_stable_id(
    tmp_path: Path,
) -> None:
    from_a = materialize(tmp_path, "shared_word_source_a", filename="lesson-8.pdf")
    from_b = materialize(
        tmp_path, "shared_word_source_b", filename="yotsubato-vol1.pdf"
    )

    id_a = from_a["records"][0].id
    id_b = from_b["records"][0].id
    assert id_a == id_b == stable_record_id("あげる", "あげる")
    # Different sources, same stable ID: exactly the case W4.1's deck picker
    # has to surface rather than silently pick a winner.
    assert from_a["records"][0].source.imported_from == "lesson-8.pdf"
    assert from_b["records"][0].source.imported_from == "yotsubato-vol1.pdf"


def test_shared_word_would_be_claimed_by_two_candidate_decks() -> None:
    """The fixture decks under decks/ model what happens once each source's
    record is promoted and tagged for its own deck: both decks claim the same
    stable ID via ordinary ``include_tags``, the same ambiguity
    ``m7-mixed-tsumori.yaml``'s ``exclude_ids`` already resolves for a real
    overlap. A future deck picker (W4.1) must make this visible, not silent."""
    _, week_a_records = resolve_deck_records(DECKS / "week-a.yaml")
    _, week_b_records = resolve_deck_records(DECKS / "week-b.yaml")

    shared_id = stable_record_id("あげる", "あげる")
    assert [record.id for record in week_a_records] == [shared_id]
    assert [record.id for record in week_b_records] == [shared_id]


def test_the_duplicate_word_is_reviewable_from_either_source(
    tmp_path: Path,
) -> None:
    """Each source gets its own staging file and its own review transaction,
    and the same stable ID is reviewable from both. Neither can see that the
    other proposed the same word; making that collision visible is W4.1's
    job."""
    from_a = materialize(tmp_path, "shared_word_source_a", filename="lesson-8.pdf")
    from_b = materialize(
        tmp_path, "shared_word_source_b", filename="yotsubato-vol1.pdf"
    )
    shared_id = stable_record_id("あげる", "あげる")

    for state in (from_a, from_b):
        panel = ReviewPanel.open(
            state["staging_path"],
            staging_dir=state["staging_path"].parent,
            patterns_path=state["patterns_path"],
        )
        assert panel.reviewable_record_ids == {shared_id}


# --- a blank reading and a dictionary-disputed reading ----------------------


class _FakeJpdb:
    """Answers /parse and lookup-vocabulary from canned dictionary data, like
    tests/test_promote.py's FakeJpdb."""

    def __init__(self, parses: dict[str, list], senses: dict[tuple, dict]) -> None:
        self.parses = parses
        self.senses = senses

    def __call__(self, url: str, body: dict, headers: dict) -> object:
        endpoint = url.rsplit("/api/v1/", 1)[-1]
        if endpoint == "parse":
            text = body["text"][0]
            row = self.parses.get(text)
            if row is None:
                return 200, {"tokens": [[]], "vocabulary": []}
            return 200, {"tokens": [[[0, None]]], "vocabulary": [row]}
        if endpoint == "lookup-vocabulary":
            fields = body["fields"]
            rows = []
            for vid, sid in body["list"]:
                sense = self.senses.get((vid, sid), {})
                rows.append([sense.get(name) for name in fields])
            return 200, {"vocabulary_info": rows}
        raise AssertionError(f"unexpected request to {url}")


# jpdb's real reading for 走る is はしる; the fixture candidate claims わしる.
HASHIRU = [1290660, 1000920, "走る", "はしる", ["LHH"], 300, ["v5r", "vi"]]
HASHIRU_SENSES = {(1290660, 1000920): {"reading": "はしる", "alt_sids": []}}


def test_blank_and_disputed_reading_are_held_for_different_reasons(
    tmp_path: Path,
) -> None:
    state = materialize(tmp_path, "reading_holds")
    records = {record.expression: record for record in state["records"]}
    assert set(records) == {"泊まる", "走る"}
    assert records["泊まる"].reading == ""
    assert records["走る"].reading == "わしる"

    client = JpdbClient(
        "test-key",
        _FakeJpdb({"走る": HASHIRU}, HASHIRU_SENSES),
        sleep=lambda _s: None,
        jitter=lambda: 0.0,
    )
    result = promote.check_readings(list(records.values()), client=client)

    assert result.promoted == []
    held_reasons = {
        item.expression: item.source.raw_fields["hold_reason"] for item in result.held
    }
    assert held_reasons["泊まる"] == HOLD_MISSING_READING
    assert held_reasons["走る"] == HOLD_UNKNOWN_READING


# --- a truncated provider answer: nothing is written ------------------------


def test_truncated_response_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_parse_call(*_args: object, **_kwargs: object) -> tuple[None, str, None]:
        return None, "max_tokens", None

    monkeypatch.setattr(extract.claude_client, "parse_call", fake_parse_call)
    item = _prepared(tmp_path, "truncated.pdf")

    with pytest.raises(extract.ExtractError) as excinfo:
        extract.extract_candidates(
            item,
            model="claude-opus-5",
            style_guide="fixture style guide",
            system="fixture system prompt",
        )

    assert excinfo.value.code == "extract-response-truncated"
    # "Nothing is written" is structural, not a file-existence check: this
    # test never calls build_records/write_staging, the same as
    # cli.command_extract, which only reaches them after extract_candidates
    # returns successfully. The raise above is the whole proof.
