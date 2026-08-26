"""W1.1: the dashboard's state, derived from repository files alone.

`WORKBENCH_PLAN.md` W1.1 ships when ``SourceJourney`` reproduces each W0
fixture's state, so these tests drive the projection off exactly those
fixtures — materialized through the real extraction pipeline by
``tests/test_workbench_fixtures.materialize`` — rather than off hand-written
staging YAML that could drift from what ``extract`` actually writes.

Everything here is offline. No provider is called and nothing is faked except
the project layout on ``tmp_path``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from test_workbench_fixtures import materialize

from japanese_anki.application import (
    ADDED,
    CARDS_NEED_EDITS,
    COVERAGE_NEEDS_DECISION,
    DECK_NEEDS_DECISION,
    EXAMPLES_NEED_REVIEW,
    GRAMMAR_NEEDS_REVIEW,
    GRAMMAR_NONE,
    GRAMMAR_ONLY,
    GRAMMAR_REVIEWED,
    GRAMMAR_UNKNOWN,
    JOURNEY_STATES,
    NOT_EXTRACTED,
    READING_HOLD,
    READY_TO_ADD,
    STAGING_UNREADABLE,
    source_journeys,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.models import EXAMPLE_AUTHORITY_KEY, EXAMPLE_AUTHORITY_STAGING

CONFIG = """
[paths]
normalized_file = "vocabulary.json"
deck_dir = "decks"
ledger_file = "ledger.json"
media_dir = "media"
staging_dir = "staging"
patterns_file = "patterns.json"
scan_inbox = "inbox"
"""


def _project(tmp_path: Path) -> Path:
    """A project root laid out the way the fixtures materialize into."""
    (tmp_path / "janki.toml").write_text(CONFIG, encoding="utf-8")
    (tmp_path / "vocabulary.json").write_text("[]", encoding="utf-8")
    (tmp_path / "decks").mkdir(exist_ok=True)
    (tmp_path / "inbox").mkdir(exist_ok=True)
    return tmp_path


def _journeys(tmp_path: Path) -> dict[str, Any]:
    config = ProjectConfig.load(_project(tmp_path))
    journeys, warnings = source_journeys(config)
    assert warnings == [], warnings
    return {journey.source: journey for journey in journeys}


def _stage(tmp_path: Path, scenario: str, *, filename: str | None = None) -> Any:
    """Materialize one W0 fixture into this project's staging/pattern paths."""
    _project(tmp_path)
    return materialize(tmp_path, scenario, filename=filename)


# --- the card track ---------------------------------------------------------


def test_exhaustive_table_waits_on_example_review(tmp_path: Path) -> None:
    """Three parsed cards, nobody has approved their sentences yet."""
    _stage(tmp_path, "table_exhaustive", filename="genki-8-vocab.pdf")

    journey = _journeys(tmp_path)["genki-8-vocab.pdf"]

    assert journey.state == EXAMPLES_NEED_REVIEW
    assert journey.card_count == 3
    assert journey.example_review_count == 3
    assert journey.next_action == "Review the Japanese examples on 3 cards"
    # A vocabulary table teaches no grammar; badging it would be noise.
    assert journey.grammar == GRAMMAR_NONE


def test_a_reading_hold_outranks_example_review(tmp_path: Path) -> None:
    """Both cards also need example review, but the reading decision comes
    first: promoting a row whose identity is unsettled mints a permanent
    record ID from a reading nobody confirmed."""
    _stage(tmp_path, "reading_holds", filename="lesson-9.pdf")
    # `materialize` stops before the reading gate, so stamp the holds the way
    # `promote.check_readings` does rather than asserting on an unheld file.
    staging_path = tmp_path / "staging" / "lesson-9.pdf.yaml"
    data = yaml.safe_load(staging_path.read_text(encoding="utf-8"))
    for record in data["records"]:
        record.setdefault("source", {}).setdefault("raw_fields", {})[
            "hold_reason"
        ] = "reading not listed by jpdb"
    staging_path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    journey = _journeys(tmp_path)["lesson-9.pdf"]

    assert journey.state == READING_HOLD
    assert journey.held_count == 2
    assert journey.next_action == "Decide the reading for 2 cards"
    assert "jpdb" in journey.detail


def _approve_examples(tmp_path: Path, staging_name: str) -> None:
    """Stamp the sentinel a reviewer types, which is approval until promote
    replaces it with per-sentence fingerprints."""
    staging_path = tmp_path / "staging" / staging_name
    data = yaml.safe_load(staging_path.read_text(encoding="utf-8"))
    for record in data["records"]:
        record["source"]["raw_fields"][EXAMPLE_AUTHORITY_KEY] = (
            EXAMPLE_AUTHORITY_STAGING
        )
    staging_path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def _assign_to_fixture_deck(
    tmp_path: Path, staging_name: str, *, tag: str = "lesson"
) -> None:
    """Make one real selector the staged cards' exact prospective owner."""
    (tmp_path / "decks" / "lesson.yaml").write_text(
        yaml.safe_dump(
            {
                "deck": {
                    "name": "Lesson deck",
                    "source": "../vocabulary.json",
                    "intake_tag": tag,
                    "include_tags": [tag],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    staging_path = tmp_path / "staging" / staging_name
    data = yaml.safe_load(staging_path.read_text(encoding="utf-8"))
    for record in data["records"]:
        record.setdefault("tags", []).append(tag)
    staging_path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def test_approving_examples_advances_the_table_to_the_coverage_gate(
    tmp_path: Path,
) -> None:
    """The exhaustive table's coverage is `unmeasured`, so `promote` would
    refuse it even with every sentence approved. The dashboard has to show that
    next gate rather than promising **Ready to add** over a refusal."""
    _stage(tmp_path, "table_exhaustive", filename="genki-8-vocab.pdf")
    _approve_examples(tmp_path, "genki-8-vocab.pdf.yaml")

    journey = _journeys(tmp_path)["genki-8-vocab.pdf"]

    assert journey.state == COVERAGE_NEEDS_DECISION
    assert journey.example_review_count == 0
    assert journey.next_action == "Decide whether the extraction covered this source"
    assert "coverage" in journey.detail


def test_a_selection_covered_lesson_still_needs_a_deck(tmp_path: Path) -> None:
    """Approving sentences cannot silently decide where the cards belong."""
    _stage(tmp_path, "lesson_with_grammar", filename="lesson-8.pdf")
    _approve_examples(tmp_path, "lesson-8.pdf.yaml")

    journey = _journeys(tmp_path)["lesson-8.pdf"]

    assert journey.state == DECK_NEEDS_DECISION
    assert journey.example_review_count == 0
    assert journey.deck_decision_count == 2
    assert journey.next_action == "Choose a study deck for 2 cards"


def test_exact_deck_ownership_advances_the_lesson_to_ready(tmp_path: Path) -> None:
    _stage(tmp_path, "lesson_with_grammar", filename="lesson-8.pdf")
    _approve_examples(tmp_path, "lesson-8.pdf.yaml")
    _assign_to_fixture_deck(tmp_path, "lesson-8.pdf.yaml")

    journey = _journeys(tmp_path)["lesson-8.pdf"]

    assert journey.state == READY_TO_ADD
    assert journey.deck_decision_count == 0
    assert journey.next_action == "Add 2 cards to your collection"


def test_an_unreadable_deck_asks_for_repair_not_a_choice(tmp_path: Path) -> None:
    _stage(tmp_path, "lesson_with_grammar", filename="lesson-8.pdf")
    _approve_examples(tmp_path, "lesson-8.pdf.yaml")
    (tmp_path / "decks" / "broken.yaml").write_text(
        "deck: [not, a, mapping]\n",
        encoding="utf-8",
    )

    journey = _journeys(tmp_path)["lesson-8.pdf"]

    assert journey.state == DECK_NEEDS_DECISION
    assert journey.deck_decision_count == 2
    assert journey.next_action == "Repair the study deck configuration"
    assert "broken.yaml" in journey.detail


def test_an_unreadable_collection_does_not_blame_the_deck_config(
    tmp_path: Path,
) -> None:
    _stage(tmp_path, "lesson_with_grammar", filename="lesson-8.pdf")
    _approve_examples(tmp_path, "lesson-8.pdf.yaml")
    (tmp_path / "vocabulary.json").write_text("{not json\n", encoding="utf-8")

    journeys, warnings = source_journeys(ProjectConfig.load(tmp_path))
    assert warnings == []
    journey = {item.source: item for item in journeys}["lesson-8.pdf"]

    assert journey.state == DECK_NEEDS_DECISION
    assert journey.next_action == "Repair the collection or staged cards"
    assert "vocabulary.json" in journey.detail


def test_a_structurally_broken_card_outranks_example_review(tmp_path: Path) -> None:
    """An edit is a different act from an approval, and it comes first."""
    _stage(tmp_path, "table_exhaustive", filename="genki-8-vocab.pdf")
    staging_path = tmp_path / "staging" / "genki-8-vocab.pdf.yaml"
    data = yaml.safe_load(staging_path.read_text(encoding="utf-8"))
    data["records"][0]["meanings"] = []
    staging_path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    journey = _journeys(tmp_path)["genki-8-vocab.pdf"]

    assert journey.state == CARDS_NEED_EDITS
    assert journey.invalid_count == 1
    assert journey.next_action == "Fix 1 card"


# --- the grammar track, which runs in parallel ------------------------------


def test_a_lesson_reports_both_tracks_independently(tmp_path: Path) -> None:
    """Cards waiting on review must not hide that grammar is waiting too."""
    _stage(tmp_path, "lesson_with_grammar", filename="lesson-8.pdf")

    journey = _journeys(tmp_path)["lesson-8.pdf"]

    assert journey.state == EXAMPLES_NEED_REVIEW
    assert journey.card_count == 2
    assert journey.grammar == GRAMMAR_NEEDS_REVIEW


def test_a_pattern_only_chart_is_not_trapped_in_deck_steps(tmp_path: Path) -> None:
    """Zero cards is a finished card track, not an empty queue. The chart's
    remaining work is grammar review, and the next action must say so."""
    _stage(tmp_path, "pattern_only_chart", filename="teform-chart.pdf")

    journey = _journeys(tmp_path)["teform-chart.pdf"]

    assert journey.state == GRAMMAR_ONLY
    assert journey.card_count == 0
    assert journey.grammar == GRAMMAR_NEEDS_REVIEW
    assert journey.next_action == "Review this source's grammar"


def test_a_reviewed_chart_has_nothing_left(tmp_path: Path) -> None:
    _stage(tmp_path, "pattern_only_chart", filename="teform-chart.pdf")
    store_path = tmp_path / "patterns.json"
    store = json.loads(store_path.read_text(encoding="utf-8"))
    for entry in store.values():
        entry["reviewed"] = True
    store_path.write_text(
        json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    journey = _journeys(tmp_path)["teform-chart.pdf"]

    assert journey.state == GRAMMAR_ONLY
    assert journey.grammar == GRAMMAR_REVIEWED
    assert journey.next_action == "Nothing left to do for this source"
    assert journey.needs_a_person is False


def test_grammar_review_does_not_gate_the_card_track(tmp_path: Path) -> None:
    """A lesson whose grammar is reviewed but whose cards are not still
    reports the card work — the two tracks never collapse into one."""
    _stage(tmp_path, "lesson_with_grammar", filename="lesson-8.pdf")
    store_path = tmp_path / "patterns.json"
    store = json.loads(store_path.read_text(encoding="utf-8"))
    for entry in store.values():
        entry["reviewed"] = True
    store_path.write_text(
        json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    journey = _journeys(tmp_path)["lesson-8.pdf"]

    assert journey.grammar == GRAMMAR_REVIEWED
    assert journey.state == EXAMPLES_NEED_REVIEW


# --- sources that have not been read at all ---------------------------------


def test_an_inbox_file_with_no_staging_is_not_yet_extracted(tmp_path: Path) -> None:
    _project(tmp_path)
    (tmp_path / "inbox" / "week-3.pdf").write_bytes(b"%PDF-1.7 fake")

    journey = _journeys(tmp_path)["week-3.pdf"]

    assert journey.state == NOT_EXTRACTED
    assert journey.card_count == 0
    assert journey.staging_path is None


def test_an_extracted_source_is_not_also_listed_as_unread(tmp_path: Path) -> None:
    """The inbox copy and its staging file are the same source, once."""
    _stage(tmp_path, "table_exhaustive", filename="genki-8-vocab.pdf")
    (tmp_path / "inbox" / "genki-8-vocab.pdf").write_bytes(b"%PDF-1.7 fake")

    journeys = _journeys(tmp_path)

    assert [j.state for j in journeys.values()] == [EXAMPLES_NEED_REVIEW]


# --- an unreadable file is a visible state, never a dropped row -------------


def test_an_unreadable_staging_file_is_reported_not_skipped(tmp_path: Path) -> None:
    config_root = _project(tmp_path)
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir(exist_ok=True)
    (staging_dir / "broken.pdf.yaml").write_text("records: [oops\n", encoding="utf-8")

    config = ProjectConfig.load(config_root)
    journeys, warnings = source_journeys(config)

    assert [journey.state for journey in journeys] == [STAGING_UNREADABLE]
    assert journeys[0].detail
    assert warnings and "broken.pdf.yaml" in warnings[0]


# --- ordering -------------------------------------------------------------


def test_the_queue_leads_with_what_is_most_stuck(tmp_path: Path) -> None:
    """Structural priority, so the reader's eye lands on the real blocker."""
    _stage(tmp_path, "table_exhaustive", filename="b-table.pdf")
    _stage(tmp_path, "pattern_only_chart", filename="a-chart.pdf")
    (tmp_path / "inbox" / "c-unread.pdf").write_bytes(b"%PDF-1.7 fake")

    config = ProjectConfig.load(tmp_path)
    journeys, _ = source_journeys(config)

    ordered = [journey.state for journey in journeys]
    assert ordered == sorted(ordered, key=JOURNEY_STATES.index)
    # ...and the unread source outranks the merely-unreviewed one.
    assert ordered.index(NOT_EXTRACTED) < ordered.index(EXAMPLES_NEED_REVIEW)


@pytest.mark.parametrize("state", JOURNEY_STATES)
def test_no_state_leaks_machinery_vocabulary(state: str) -> None:
    """The dashboard speaks the learner's language (WORKBENCH_PLAN.md W6/slice 0):
    no `staging`, `promote`, `tag`, `fingerprint` or `run ID` on a main screen."""
    lowered = state.lower()
    for word in ("staging", "promote", "fingerprint", "run id", "yaml", "json"):
        assert word not in lowered, state


def test_a_stale_pattern_run_is_not_reported_as_reviewed(tmp_path: Path) -> None:
    """The store's `reviewed` flag answers a question about the text *that*
    paid run produced. If a newer extraction replaced the staging file, the old
    approval covers sentences this source no longer teaches, so the badge must
    say it does not know — never **Grammar reviewed**."""
    _stage(tmp_path, "lesson_with_grammar", filename="lesson-8.pdf")
    store_path = tmp_path / "patterns.json"
    store = json.loads(store_path.read_text(encoding="utf-8"))
    for entry in store.values():
        entry["reviewed"] = True
        entry["review_run_id"] = "11111111-1111-4111-8111-111111111111"
    store_path.write_text(
        json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    journey = _journeys(tmp_path)["lesson-8.pdf"]

    assert journey.grammar == GRAMMAR_UNKNOWN
    assert "different extraction run" in journey.grammar_detail


def test_grammar_missing_from_the_store_is_not_silently_fine(tmp_path: Path) -> None:
    """A lesson that taught patterns but has no store entry lost half its paid
    answer. Reporting GRAMMAR_NONE would hide that."""
    _stage(tmp_path, "lesson_with_grammar", filename="lesson-8.pdf")
    (tmp_path / "patterns.json").write_text("{}", encoding="utf-8")

    journey = _journeys(tmp_path)["lesson-8.pdf"]

    assert journey.grammar == GRAMMAR_UNKNOWN
    assert "not in the pattern store" in journey.grammar_detail


def test_a_source_read_for_grammar_alone_is_not_offered_for_re_reading(
    tmp_path: Path,
) -> None:
    """Found against the real corpus: `104 Week 11 Slide.pdf` sits in the inbox
    with six *reviewed* patterns in the store and no staging file — its
    zero-record staging was archived and pruned long ago. Checking staging
    alone called it unread and offered to read it again, which is an offer to
    re-buy an answer the repository already holds."""
    _project(tmp_path)
    (tmp_path / "inbox" / "week-11-slides.pdf").write_bytes(b"%PDF-1.7 fake")
    (tmp_path / "patterns.json").write_text(
        json.dumps(
            {
                "week-11-slides.pdf": {
                    "kind": "lesson",
                    "title": "Week 11",
                    "reviewed": True,
                    "patterns": [
                        {"template": "〜の？", "gloss": "casual question", "examples": []}
                    ],
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    journey = _journeys(tmp_path)["week-11-slides.pdf"]

    assert journey.state != NOT_EXTRACTED
    assert journey.state == GRAMMAR_ONLY
    assert journey.grammar == GRAMMAR_REVIEWED
    assert journey.next_action == "Nothing left to do for this source"


def test_a_source_read_for_unreviewed_grammar_asks_for_review_not_re_reading(
    tmp_path: Path,
) -> None:
    _project(tmp_path)
    (tmp_path / "inbox" / "chart.pdf").write_bytes(b"%PDF-1.7 fake")
    (tmp_path / "patterns.json").write_text(
        json.dumps(
            {
                "chart.pdf": {
                    "kind": "pattern",
                    "title": "chart",
                    "reviewed": False,
                    "patterns": [
                        {"template": "〜ている", "gloss": "ongoing", "examples": []}
                    ],
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    journey = _journeys(tmp_path)["chart.pdf"]

    assert journey.state == GRAMMAR_ONLY
    assert journey.grammar == GRAMMAR_NEEDS_REVIEW
    assert journey.next_action == "Review this source's grammar"


def test_a_promoted_source_stays_visible_instead_of_vanishing(tmp_path: Path) -> None:
    """Found against the real corpus: promotion archives and prunes the staging
    file, so a source that reads only live staging drops off the queue
    entirely. Two of the corpus's five inbox sources were invisible. A source
    disappearing reads as "janki lost my lesson", not "that one is done"."""
    _project(tmp_path)
    (tmp_path / "inbox" / "week-8.pdf").write_bytes(b"%PDF-1.7 fake")
    archive = tmp_path / "staging" / "done"
    archive.mkdir(parents=True, exist_ok=True)
    (archive / "week-8.pdf.yaml").write_text("records: []\n", encoding="utf-8")

    journey = _journeys(tmp_path)["week-8.pdf"]

    assert journey.state == ADDED
    assert journey.next_action == "Add dictionary facts, audio, and build the deck"


def test_re_extracting_a_promoted_source_shows_the_newer_answer(
    tmp_path: Path,
) -> None:
    """An archive plus a live file is a re-extraction. The live answer is the
    current one; reporting `Added` would hide cards waiting for review."""
    _stage(tmp_path, "table_exhaustive", filename="week-8.pdf")
    (tmp_path / "inbox" / "week-8.pdf").write_bytes(b"%PDF-1.7 fake")
    archive = tmp_path / "staging" / "done"
    archive.mkdir(parents=True, exist_ok=True)
    (archive / "week-8.pdf.yaml").write_text("records: []\n", encoding="utf-8")

    journeys = _journeys(tmp_path)

    assert journeys["week-8.pdf"].state == EXAMPLES_NEED_REVIEW
    assert len(journeys) == 1


def test_a_live_file_whose_meta_renames_the_source_is_not_also_called_unread(
    tmp_path: Path,
) -> None:
    """Staging metadata may name a source differently from the file on disk —
    the real corpus's Yotsubato pack is `source_file: Yotsubato Volume 1
    Reading Pack Vocab`, not a filename. The inbox walk therefore cannot rely
    on the accounted-names set alone: it has to see the live staging file, or
    it reports the same source twice, once as unread."""
    _stage(tmp_path, "table_exhaustive", filename="week-8.pdf")
    staging_path = tmp_path / "staging" / "week-8.pdf.yaml"
    data = yaml.safe_load(staging_path.read_text(encoding="utf-8"))
    data["source_file"] = "Week 8 Handout"
    staging_path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    (tmp_path / "inbox" / "week-8.pdf").write_bytes(b"%PDF-1.7 fake")

    journeys = _journeys(tmp_path)

    assert set(journeys) == {"Week 8 Handout"}
    assert NOT_EXTRACTED not in {journey.state for journey in journeys.values()}


def test_ready_to_add_counts_as_waiting_on_the_person(tmp_path: Path) -> None:
    """Found by rendering the real corpus: the header said "3 waiting on you"
    while two sources sat at **Ready to add**, which is precisely a state
    waiting for the person to act. `Added` counted as waiting and
    `Ready to add` did not — the same work, counted two ways."""
    _stage(tmp_path, "lesson_with_grammar", filename="lesson-8.pdf")
    _approve_examples(tmp_path, "lesson-8.pdf.yaml")
    _assign_to_fixture_deck(tmp_path, "lesson-8.pdf.yaml")
    # Grammar reviewed too, so the grammar track cannot be what makes this
    # count as waiting — the `Ready to add` cards have to do it on their own.
    store_path = tmp_path / "patterns.json"
    store = json.loads(store_path.read_text(encoding="utf-8"))
    for entry in store.values():
        entry["reviewed"] = True
    store_path.write_text(
        json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    journey = _journeys(tmp_path)["lesson-8.pdf"]

    assert journey.state == READY_TO_ADD
    assert journey.grammar == GRAMMAR_REVIEWED
    assert journey.needs_a_person is True


def test_a_promoted_source_still_counts_as_waiting(tmp_path: Path) -> None:
    _project(tmp_path)
    (tmp_path / "inbox" / "week-8.pdf").write_bytes(b"%PDF-1.7 fake")
    archive = tmp_path / "staging" / "done"
    archive.mkdir(parents=True, exist_ok=True)
    (archive / "week-8.pdf.yaml").write_text("records: []\n", encoding="utf-8")

    assert _journeys(tmp_path)["week-8.pdf"].needs_a_person is True
