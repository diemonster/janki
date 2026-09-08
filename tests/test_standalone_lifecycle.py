"""A standalone copy's life after assignment: promote, report, import, extract.

The copy is an ordinary canonical record in a reserved namespace, so it travels
the one promotion transaction every other record travels. What changes is that
nothing may quietly reconnect it to the shared word it was copied from — not a
re-mint, not an archive retry, not a duplicate report, not a review import.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from japanese_anki import promote as promote_module
from japanese_anki import status as status_module
from japanese_anki.application.promotion import decide_promotion, execute_promotion
from japanese_anki.config import ProjectConfig
from japanese_anki.extract import known_ids
from japanese_anki.identifiers import IdentityError
from japanese_anki.importers.jpdb_reviews import ReviewEntry, apply_reviews
from japanese_anki.io import load_records, save_records_json
from japanese_anki.models import SourceReference, VocabularyRecord
from japanese_anki.promote import check_readings, remint
from japanese_anki.staging import write_staging
from japanese_anki.workbench.reidentify import plan_reidentification

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

SCOPE = "ab12"
SHARED_ID = "word:話す:はなす"
COPY_ID = f"standalone:{SCOPE}:話す:はなす"


def _record(
    record_id: str = SHARED_ID,
    *,
    expression: str = "話す",
    reading: str = "はなす",
    tags: list[str] | None = None,
    raw_fields: dict[str, str] | None = None,
) -> VocabularyRecord:
    return VocabularyRecord(
        id=record_id,
        expression=expression,
        reading=reading,
        meanings=["to speak"],
        tags=list(tags or []),
        source=SourceReference(
            type="extract",
            imported_from="page-a.pdf",
            row=3,
            raw_fields=dict(raw_fields or {}),
        ),
    )


# --- promote: minting and re-minting inside a scope -----------------------


def test_remint_keeps_a_new_row_inside_its_own_scope() -> None:
    """A scoped row whose id no longer matches its content re-mints in scope."""
    stale = _record(f"standalone:{SCOPE}:話す:はなし")

    minted = remint(stale)

    assert minted.id == COPY_ID
    assert remint(_record("word:話す:はなし")).id == SHARED_ID


def test_remint_leaves_an_already_stored_scoped_identity_alone() -> None:
    stored = _record(COPY_ID, reading="はなし")

    assert remint(stored, {COPY_ID}) is stored
    assert remint(stored, {COPY_ID}).id == COPY_ID


def test_a_scoped_row_is_not_held_when_its_id_already_matches_its_scope() -> None:
    """`remint_blocked` must not hold a row that would never be re-minted."""
    result = check_readings(
        [_record(COPY_ID)],
        skip_reading_check=True,
        already_stored=frozenset(),
        remint_blocked=True,
    )

    assert [record.id for record in result.promoted] == [COPY_ID]
    assert result.held == []


def test_a_scoped_row_whose_identity_would_move_is_still_held() -> None:
    result = check_readings(
        [_record(f"standalone:{SCOPE}:話す:はなし")],
        skip_reading_check=True,
        already_stored=frozenset(),
        remint_blocked=True,
    )

    assert result.promoted == []
    assert [record.id for record in result.held] == [f"standalone:{SCOPE}:話す:はなし"]


def test_an_archive_retry_never_matches_the_shared_variant_of_a_scoped_row() -> None:
    """The stable variant a retry compares against stays inside the scope."""
    meta: dict[str, object] = {}
    live = [_record(f"standalone:{SCOPE}:話す:はなし")]
    shared_archive = [_record(SHARED_ID)]

    assert promote_module._candidate_accounting_retry_targets(
        meta, live, shared_archive
    ) == (None,)

    scoped_archive = [_record(COPY_ID)]
    assert promote_module._candidate_accounting_retry_targets(
        meta, live, scoped_archive
    ) == (COPY_ID,)


# --- promote: the copy lands once, the original is untouched --------------


def _project(tmp_path: Path, decks: dict[str, dict[str, object]]) -> ProjectConfig:
    (tmp_path / "janki.toml").write_text(CONFIG, encoding="utf-8")
    (tmp_path / "vocabulary.json").write_text("[]", encoding="utf-8")
    deck_dir = tmp_path / "decks"
    deck_dir.mkdir()
    for stem, deck in decks.items():
        (deck_dir / f"{stem}.yaml").write_text(
            yaml.safe_dump({"deck": deck}, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    config = ProjectConfig.load(tmp_path)
    config.staging_dir.mkdir(exist_ok=True)
    return config


def _word_deck(name: str, tag: str, **overrides: object) -> dict[str, object]:
    deck: dict[str, object] = {
        "name": name,
        "source": "../vocabulary.json",
        "include_tags": [tag],
        "intake_tag": tag,
    }
    deck.update(overrides)
    return deck


def test_promoting_a_scoped_copy_lands_it_once_and_leaves_the_original_alone(
    tmp_path: Path,
) -> None:
    config = _project(
        tmp_path,
        {
            "lesson": _word_deck("Lesson deck", "lesson"),
            "verbs": _word_deck("Class verbs", "verbs", scope_id=SCOPE),
        },
    )
    original = _record(tags=["lesson"])
    save_records_json(config.normalized_file, [original])
    path = config.staging_dir / "page-a.pdf.yaml"
    write_staging(
        path,
        [_record(COPY_ID, tags=["verbs"], raw_fields={"standalone_copy_from": SHARED_ID})],
        {"source_file": "page-a.pdf"},
    )

    decision = decide_promotion(config, path, skip_reading_check=True)
    execute_promotion(config, decision)

    landed = {record.id: record for record in load_records(config.normalized_file)}
    assert sorted(landed) == [COPY_ID, SHARED_ID]
    assert landed[SHARED_ID].tags == ["lesson"]
    assert landed[SHARED_ID].source.raw_fields == {}
    assert landed[COPY_ID].tags == ["verbs"]
    assert landed[COPY_ID].source.raw_fields["standalone_copy_from"] == SHARED_ID


# --- workbench re-identification ------------------------------------------


def test_reidentifying_a_scoped_row_keeps_it_in_its_scope() -> None:
    staged = [_record(COPY_ID)]

    plan = plan_reidentification(staged, 0, "話す", "はなし")

    assert plan.old_id == COPY_ID
    assert plan.new_id == f"standalone:{SCOPE}:話す:はなし"
    assert plan_reidentification([_record()], 0, "話す", "はなし").new_id == "word:話す:はなし"


def test_reidentifying_a_scoped_row_sees_its_own_scope_neighbours() -> None:
    """A shared word with the same spelling is not this scope's neighbour."""
    staged = [_record(COPY_ID), _record(f"standalone:{SCOPE}:話す:はなし")]

    plan = plan_reidentification(staged, 0, "話す", "はなし", existing=[_record()])

    assert plan.collides_in_source is True


# --- status --duplicates ---------------------------------------------------


def test_duplicates_are_grouped_inside_one_scope_at_a_time() -> None:
    shared = _record()
    copy = _record(COPY_ID)
    other_scope = _record("standalone:cd34:話す:はなす")

    assert status_module.find_duplicates([shared, copy, other_scope]) == []


def test_duplicates_within_one_scope_are_still_reported() -> None:
    first = _record(COPY_ID)
    second = _record(f"standalone:{SCOPE}:話す:はなし")
    shared_first = _record()
    shared_second = _record("word:話す:はなし")

    scoped = status_module.find_duplicates([first, second])
    shared = status_module.find_duplicates([shared_first, shared_second])

    assert [(group.kind, group.key, group.ids) for group in scoped] == [
        ("expression", "話す", sorted([first.id, second.id]))
    ]
    assert [(group.kind, group.key, group.ids) for group in shared] == [
        ("expression", "話す", sorted([shared_first.id, shared_second.id]))
    ]


def test_a_shared_and_a_scoped_record_sharing_a_vid_are_not_one_duplicate() -> None:
    shared = _record(raw_fields={"vid": "1234"})
    copy = _record(COPY_ID, raw_fields={"vid": "1234"})

    assert status_module.find_duplicates([shared, copy]) == []


# --- jpdb review import ----------------------------------------------------


def test_a_standalone_copy_never_receives_a_global_review_import() -> None:
    copy = _record(COPY_ID)
    shared = _record()
    entry = ReviewEntry(vid="", spelling="話す", reading="はなす", reviews=7)

    # The copy is first, so a first-match-wins index would choose it.
    match = apply_reviews([copy, shared], [entry])

    assert set(match.changed) == {SHARED_ID}
    landed = {record.id: record for record in match.records}
    assert landed[COPY_ID].tags == []
    assert landed[COPY_ID].source.raw_fields == {}
    assert landed[SHARED_ID].tags != []


def test_a_standalone_only_collection_matches_no_global_review() -> None:
    match = apply_reviews(
        [_record(COPY_ID)],
        [ReviewEntry(vid="", spelling="話す", reading="はなす", reviews=7)],
    )

    assert match.changed == []


# --- extraction's known-word set ------------------------------------------


def test_known_ids_default_to_the_shared_collection() -> None:
    records = [_record(), _record(COPY_ID, expression="食べる", reading="たべる")]

    assert known_ids(records) == {SHARED_ID}


def test_known_ids_for_one_scope_answer_in_ordinary_extraction_keys() -> None:
    """Candidates are ordinary ids at extraction time, so the keys are too."""
    hand_written = replace(_record(COPY_ID), id=f"standalone:{SCOPE}:legacy")

    ids = known_ids([_record(), _record(COPY_ID), hand_written], scope_id=SCOPE)

    assert ids == {COPY_ID, SHARED_ID, f"standalone:{SCOPE}:legacy"}


def test_known_ids_for_one_scope_exclude_another_scope() -> None:
    records = [_record("standalone:cd34:話す:はなす"), _record(COPY_ID)]

    assert known_ids(records, scope_id=SCOPE) == {COPY_ID, SHARED_ID}
    assert known_ids(records, scope_id="cd34") == {
        "standalone:cd34:話す:はなす",
        SHARED_ID,
    }


def test_known_ids_refuse_an_unusable_scope() -> None:
    """A scope nothing can match would report every word in a source as new."""
    with pytest.raises(IdentityError):
        known_ids([_record()], scope_id="ZZ")
