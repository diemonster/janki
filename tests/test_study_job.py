"""One committed job document: whitelisted choices, append-only everything else.

These tests are about the store's own contracts, not about Japanese and not
about a provider. Every provider below is faked, every path is a temporary
repository, and nothing here reads or writes the owner's collection.

The four things a job document has to get right, and one thing it must never
grow:

1. **Five namespaces, one CAS'd revision.** ``record_choice`` writes
   whitelisted ``choices`` keys and nothing else, so an ordinary choice edit
   cannot rewrite a layout revision a dispatched child bound or remove a
   pending intent.
2. **Append-only intents and outcomes.** No edit, no reorder, no duplicate id,
   and no mutable field inside an immutable entry.
3. **Backlinks by exact id and hash.** A crash between the fsynced intent and
   the appended outcome leaves the artifact discoverable — by the reserved id,
   its recorded hash and, for a batch, the manifest's own embedded ``job_id``.
   Never by "the newest" and never by mtime.
4. **No progress fields.** Every count and state in a status is derived at read
   time from the journal, the batch manifests, staging, the archive, the
   ledger and the publication receipts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from test_revision_provider import FakeClaudeRunner, _which
from test_workbench_fixtures import RESPONSES

from conftest import seed_prompts
from japanese_anki import operations, staging
from japanese_anki.application import (
    extraction,
    extraction_batch,
    source_parts,
    study_job,
)
from japanese_anki.application.study_job import (
    ActionIntent,
    IntentOutcome,
    StudyJobError,
    append_intent,
    append_outcome,
    discover_actions,
    load_study_job,
    open_study_job,
    record_choice,
    study_job_status,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError

SCENARIO = "table_exhaustive"


def _answer() -> dict[str, Any]:
    return json.loads((RESPONSES / f"{SCENARIO}.json").read_text(encoding="utf-8"))


def _stream(answer: Any) -> bytes:
    fixture = Path(__file__).parent / "fixtures" / "claude-code-stream.ndjson"
    lines = []
    for line in fixture.read_bytes().splitlines(keepends=True):
        payload = json.loads(line)
        if payload.get("type") == "result":
            payload["structured_output"] = answer
            line = json.dumps(payload).encode("utf-8") + b"\n"
        lines.append(line)
    return b"".join(lines)


def _project(tmp_path: Path) -> ProjectConfig:
    seed_prompts(tmp_path)
    (tmp_path / "janki.toml").write_text(
        "[ai]\n"
        'extract_provider = "claude-code"\n'
        'extract_model = "claude-opus-5"\n',
        encoding="utf-8",
    )
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "normalized.json").write_text("[]\n", encoding="utf-8")
    return ProjectConfig.load(tmp_path)


def _deck(config: ProjectConfig, stem: str = "201-verbs") -> Path:
    config.deck_dir.mkdir(parents=True, exist_ok=True)
    path = config.deck_dir / f"{stem}.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "deck": {
                    "name": stem,
                    "deck_id": 1_500_000_001,
                    "source": "../normalized.json",
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _source(config: ProjectConfig, name: str, body: bytes = b"page one") -> Path:
    path = config.scan_inbox / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.7 " + body)
    return path


def _job(config: ProjectConfig, *, parent: Path | None = None) -> Any:
    source = parent if parent is not None else _source(config, "verbs.pdf")
    return open_study_job(
        config,
        kind="source_extraction",
        parent_source=source,
        deck_path=_deck(config),
    )


def _runner() -> FakeClaudeRunner:
    return FakeClaudeRunner(reply=_stream(_answer()))


def _batch_plan(
    config: ProjectConfig,
    sources: list[Path],
    *,
    job_id: str = "",
    force: bool = False,
    **options: Any,
) -> Any:
    return extraction_batch.plan_extraction_batch(
        config,
        sources,
        job_id=job_id,
        force=force,
        provider_env={},
        provider_runner=_runner(),
        provider_which=_which,
        **options,
    )


def _batch_intent(plan: Any) -> ActionIntent:
    return ActionIntent(
        intent_id=study_job.new_intent_id(),
        kind="extract_batch",
        decided_at="2026-09-09T10:07:31+00:00",
        reserves={
            "batch_id": plan.batch_id,
            "manifest_sha256": plan.manifest_sha256,
            "child_operation_ids": [
                child.operation_id for child in plan.children
            ],
        },
        bindings={"concurrency_limit": plan.concurrency_limit},
    )


# --- the document itself ------------------------------------------------------


def test_opening_a_job_binds_its_header_and_reserves_every_namespace(
    tmp_path: Path,
) -> None:
    """Five namespaces exist from the first write; four of them start empty."""

    config = _project(tmp_path)
    source = _source(config, "verbs.pdf", b"verbs")
    deck = _deck(config)

    job = open_study_job(
        config,
        kind="source_extraction",
        parent_source=source,
        deck_path=deck,
    )

    assert job.header.kind == "source_extraction"
    assert job.header.parent_source_name == "verbs.pdf"
    assert job.header.parent_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert job.header.deck_path == "data/decks/201-verbs.yaml"
    assert job.header.deck_sha256 == hashlib.sha256(deck.read_bytes()).hexdigest()
    assert job.choices == {}
    assert job.layouts == {}
    assert job.intents == ()
    assert job.outcomes == ()

    path = study_job.study_job_path(config, job.header.job_id)
    assert path.parent.name == "study_jobs"
    assert path.parent.parent == config.operations_file.parent
    wire = json.loads(path.read_text(encoding="utf-8"))
    assert set(wire) == {"version", "header", "choices", "layouts", "intents", "outcomes"}
    # The revision a caller compares and swaps on is the file's own bytes.
    assert job.revision == hashlib.sha256(path.read_bytes()).hexdigest()
    # No progress namespace, and no second log.
    assert "progress" not in wire
    assert "events" not in wire


def test_a_job_document_holds_no_progress_or_events_namespace(tmp_path: Path) -> None:
    """The contract's list is exhaustive; a sixth namespace is not writable."""

    config = _project(tmp_path)
    job = _job(config)
    with pytest.raises(StudyJobError) as error:
        record_choice(
            config,
            job.header.job_id,
            {"events": []},
            expected_revision=job.revision,
        )
    assert "events" in str(error.value)


def test_record_choice_writes_only_whitelisted_choice_keys(tmp_path: Path) -> None:
    """A choice edit is the owner's local preference and nothing more."""

    config = _project(tmp_path)
    job = _job(config)

    saved = record_choice(
        config,
        job.header.job_id,
        {"directions": ["recognition", "production"]},
        expected_revision=job.revision,
    )
    assert saved.choices == {"directions": ["recognition", "production"]}
    assert saved.revision != job.revision

    again = record_choice(
        config,
        job.header.job_id,
        {"part_selections": ["verbs-p1.png"]},
        expected_revision=saved.revision,
    )
    # A second choice does not erase the first.
    assert again.choices == {
        "directions": ["recognition", "production"],
        "part_selections": ["verbs-p1.png"],
    }


@pytest.mark.parametrize("namespace", ["header", "layouts", "intents", "outcomes"])
def test_record_choice_refuses_a_payload_naming_an_immutable_namespace(
    tmp_path: Path, namespace: str
) -> None:
    """The refusal that stops a choice edit rewriting a bound revision."""

    config = _project(tmp_path)
    job = _job(config)
    with pytest.raises(StudyJobError) as error:
        record_choice(
            config,
            job.header.job_id,
            {namespace: {}},
            expected_revision=job.revision,
        )
    message = str(error.value)
    assert namespace in message
    # Named as a namespace, not merely as an unknown key: these four are the
    # ones an ordinary edit must never reach.
    assert "namespace" in message
    assert load_study_job(config, job.header.job_id).revision == job.revision


# --- the owner's five decisions, each with its own validator ------------------
#
# These five keys used to be refused by name because no control recorded them.
# The finish's editors do, so the refusals are gone and each key has an explicit
# validator instead. What is checked here is that none of them is *merely*
# allowlisted: a decision stored in an unvalidated shape is a decision nobody
# can prove was made, and the four provenance values are what make one
# reviewable later.

BINDINGS: dict[str, Any] = {
    "part": "verbs-p1.png",
    "staging_sha256": "a" * 64,
    "rendering_fingerprint": "b" * 64,
}


def _bound(job_id: str, **overrides: Any) -> dict[str, Any]:
    return {**BINDINGS, "job_id": job_id, **overrides}


#: One *complete and exactly valid* payload per per-part decision, beyond the
#: four bindings. Shared by every test below that needs a real saved decision,
#: so a change to one key's closed field set has one place to be stated.
DECISION_PAYLOADS: dict[str, dict[str, Any]] = {
    "review_flags": {"record_ids": []},
    "review_patterns": {"value": True},
    "coverage_reasons": {"reason": "The table has no totals line."},
    "dispositions": {"action": "defer", "reason": "Next lesson covers these."},
}


@pytest.mark.parametrize(("key", "entry"), sorted(DECISION_PAYLOADS.items()))
def test_an_owner_decision_records_the_four_bindings_it_was_taken_over(
    tmp_path: Path, key: str, entry: dict[str, Any]
) -> None:
    """Job, part, that part's staging bytes and the rendering it was decided over.

    Deliberately not the job document's own revision: that is the CAS this write
    already performs, and binding it here would make saving one owner choice
    stale another independent one.
    """

    config = _project(tmp_path)
    job = _job(config)

    saved = record_choice(
        config,
        job.header.job_id,
        {key: {BINDINGS["part"]: _bound(job.header.job_id, **entry)}},
        expected_revision=job.revision,
    )

    stored = saved.choices[key][BINDINGS["part"]]
    assert stored["job_id"] == job.header.job_id
    assert stored["part"] == BINDINGS["part"]
    assert stored["staging_sha256"] == BINDINGS["staging_sha256"]
    assert stored["rendering_fingerprint"] == BINDINGS["rendering_fingerprint"]
    assert stored["saved_at"]
    assert load_study_job(config, job.header.job_id).choices == saved.choices


@pytest.mark.parametrize("key", sorted(DECISION_PAYLOADS))
@pytest.mark.parametrize(
    "missing", ["job_id", "part", "staging_sha256", "rendering_fingerprint"]
)
def test_an_owner_decision_missing_one_binding_writes_nothing(
    tmp_path: Path, key: str, missing: str
) -> None:
    """A decision nobody can place is not stored under a partial provenance.

    The entry is otherwise exactly this key's valid payload, so the refusal can
    only be about the dropped binding — and the message is matched on the whole
    phrase that names it rather than on the bare field name, which four short
    binding names would match almost anywhere.
    """

    config = _project(tmp_path)
    job = _job(config)
    entry = _bound(job.header.job_id, **DECISION_PAYLOADS[key])
    entry.pop(missing)

    with pytest.raises(StudyJobError, match=f"records the exact {missing}"):
        record_choice(
            config,
            job.header.job_id,
            {key: {BINDINGS["part"]: entry}},
            expected_revision=job.revision,
        )
    assert load_study_job(config, job.header.job_id).choices == {}


def test_one_owner_decision_never_stales_another_independent_one(
    tmp_path: Path,
) -> None:
    """Two parts, two choices, two renderings — and both survive the other's save.

    §9.1's rule stated as an artifact fact: the first part's coverage reason is
    still bound to the rendering it was taken over after a second part saves a
    decision over a different one.
    """

    config = _project(tmp_path)
    job = _job(config)
    first = record_choice(
        config,
        job.header.job_id,
        {
            "coverage_reasons": {
                "p1.png": _bound(
                    job.header.job_id,
                    part="p1.png",
                    reason="The source lists no totals.",
                )
            }
        },
        expected_revision=job.revision,
    )
    second = record_choice(
        config,
        job.header.job_id,
        {
            "coverage_reasons": {
                "p2.png": _bound(
                    job.header.job_id,
                    part="p2.png",
                    staging_sha256="c" * 64,
                    rendering_fingerprint="d" * 64,
                    reason="This page is a chart.",
                )
            }
        },
        expected_revision=first.revision,
    )

    reasons = second.choices["coverage_reasons"]
    assert set(reasons) == {"p1.png", "p2.png"}
    assert reasons["p1.png"]["rendering_fingerprint"] == "b" * 64
    assert reasons["p1.png"]["reason"] == "The source lists no totals."
    assert reasons["p2.png"]["rendering_fingerprint"] == "d" * 64


def test_a_coverage_reason_and_a_disposition_reason_are_the_owners_own_words(
    tmp_path: Path,
) -> None:
    """Blank is refused rather than derived, exactly as `coverage` refuses it."""

    config = _project(tmp_path)
    job = _job(config)
    for key, entry in (
        ("coverage_reasons", {"reason": "   "}),
        ("dispositions", {"action": "exclude", "reason": ""}),
    ):
        with pytest.raises(StudyJobError, match="reason"):
            record_choice(
                config,
                job.header.job_id,
                {key: {BINDINGS["part"]: _bound(job.header.job_id, **entry)}},
                expected_revision=job.revision,
            )
    assert load_study_job(config, job.header.job_id).choices == {}


def test_a_disposition_is_exactly_an_exclusion_or_a_deferral(tmp_path: Path) -> None:
    config = _project(tmp_path)
    job = _job(config)
    assert study_job.DISPOSITION_ACTIONS == ("exclude", "defer")

    with pytest.raises(StudyJobError, match="exclude or defer"):
        record_choice(
            config,
            job.header.job_id,
            {
                "dispositions": {
                    BINDINGS["part"]: _bound(
                        job.header.job_id, action="archive", reason="Not this job."
                    )
                }
            },
            expected_revision=job.revision,
        )
    assert load_study_job(config, job.header.job_id).choices == {}


def _two_parts(
    config: ProjectConfig, job: Any, key: str
) -> Any:
    """Save the same per-part decision for `p1.png` and `p2.png`, in two saves.

    Two different renderings on purpose, so a later assertion that the surviving
    part is byte-identical is also an assertion about the rendering it was bound
    to rather than only about its payload.
    """
    first = record_choice(
        config,
        job.header.job_id,
        {key: {"p1.png": _bound(job.header.job_id, part="p1.png", **DECISION_PAYLOADS[key])}},
        expected_revision=job.revision,
    )
    return record_choice(
        config,
        job.header.job_id,
        {
            key: {
                "p2.png": _bound(
                    job.header.job_id,
                    part="p2.png",
                    staging_sha256="c" * 64,
                    rendering_fingerprint="d" * 64,
                    **DECISION_PAYLOADS[key],
                )
            }
        },
        expected_revision=first.revision,
    )


@pytest.mark.parametrize("key", sorted(DECISION_PAYLOADS))
def test_the_owner_withdraws_one_part_and_the_other_stays_byte_identical(
    tmp_path: Path, key: str
) -> None:
    """An explicit `None` withdraws exactly the part it names, through this writer.

    The owner's own control and the owner's own compare-and-swap: the same
    reversible local save that recorded the decision is what takes it back, so a
    saved decision the finish refuses can be cleared without hand-editing the
    job document. Nothing else in the map moves — the surviving part keeps its
    bindings, its rendering and its saved date byte for byte.
    """

    config = _project(tmp_path)
    job = _job(config)
    saved = _two_parts(config, job, key)
    kept = json.dumps(saved.choices[key]["p2.png"], sort_keys=True)

    withdrawn = record_choice(
        config,
        job.header.job_id,
        {key: {"p1.png": None}},
        expected_revision=saved.revision,
    )

    assert set(withdrawn.choices[key]) == {"p2.png"}
    assert json.dumps(withdrawn.choices[key]["p2.png"], sort_keys=True) == kept
    reread = load_study_job(config, job.header.job_id)
    assert reread.choices == withdrawn.choices


def test_a_withdrawal_leaves_every_other_choice_and_pending_intent_alone(
    tmp_path: Path,
) -> None:
    """One key's withdrawal is not a rewrite of the job's other decisions.

    §9.1's independence rule applied to the new direction: withdrawing a coverage
    reason cannot disturb another key's saved decision, the job-wide audio
    preference, or an intent that is still open — the last of which is what stops
    a reversible local edit clearing work a service already reserved.
    """

    config = _project(tmp_path)
    job = _job(config)
    saved = _two_parts(config, job, "coverage_reasons")
    saved = record_choice(
        config,
        job.header.job_id,
        {
            "dispositions": {
                "p1.png": _bound(
                    job.header.job_id,
                    part="p1.png",
                    **DECISION_PAYLOADS["dispositions"],
                )
            }
        },
        expected_revision=saved.revision,
    )
    saved = record_choice(
        config,
        job.header.job_id,
        {"include_example_audio": {"value": False}},
        expected_revision=saved.revision,
    )
    intent = ActionIntent(
        intent_id=study_job.new_intent_id(),
        kind="source_parts",
        decided_at="2026-09-12T10:00:00+00:00",
        reserves={"recipe_id": "r-1", "receipt_sha256": "77aa"},
        bindings={},
    )
    saved = append_intent(
        config, job.header.job_id, intent, expected_revision=saved.revision
    )
    disposition = json.dumps(saved.choices["dispositions"], sort_keys=True)

    withdrawn = record_choice(
        config,
        job.header.job_id,
        {"coverage_reasons": {"p1.png": None}},
        expected_revision=saved.revision,
    )

    assert set(withdrawn.choices["coverage_reasons"]) == {"p2.png"}
    assert json.dumps(withdrawn.choices["dispositions"], sort_keys=True) == disposition
    assert withdrawn.choices["include_example_audio"]["value"] is False
    assert [entry.intent_id for entry in withdrawn.intents] == [intent.intent_id]
    assert withdrawn.closed_intent_ids == frozenset()


def test_withdrawing_a_decision_this_job_does_not_hold_writes_nothing(
    tmp_path: Path,
) -> None:
    """A withdrawal names a stored decision, so there is nothing to no-op over.

    Accepting it would report a withdrawal to an owner whose stale editor was
    looking at a different part — and would spend a revision saying nothing. The
    other part's reason is still there afterwards.
    """

    config = _project(tmp_path)
    job = _job(config)
    saved = record_choice(
        config,
        job.header.job_id,
        {
            "coverage_reasons": {
                "p2.png": _bound(
                    job.header.job_id,
                    part="p2.png",
                    **DECISION_PAYLOADS["coverage_reasons"],
                )
            }
        },
        expected_revision=job.revision,
    )

    with pytest.raises(StudyJobError, match="nothing to withdraw"):
        record_choice(
            config,
            job.header.job_id,
            {"coverage_reasons": {"p1.png": None}},
            expected_revision=saved.revision,
        )
    reread = load_study_job(config, job.header.job_id)
    assert set(reread.choices["coverage_reasons"]) == {"p2.png"}
    assert reread.revision == saved.revision


@pytest.mark.parametrize("key", sorted(DECISION_PAYLOADS))
def test_an_empty_decision_map_is_refused_rather_than_read_as_a_withdrawal(
    tmp_path: Path, key: str
) -> None:
    """Withdrawal is stated per part; an empty map clears nothing implicitly."""

    config = _project(tmp_path)
    job = _job(config)
    saved = _two_parts(config, job, key)

    with pytest.raises(StudyJobError, match="keyed by published part name"):
        record_choice(
            config,
            job.header.job_id,
            {key: {}},
            expected_revision=saved.revision,
        )
    assert set(load_study_job(config, job.header.job_id).choices[key]) == {
        "p1.png",
        "p2.png",
    }


@pytest.mark.parametrize(
    ("key", "stray", "extra"),
    [
        ("dispositions", "record_id", {"record_id": "word:走る:はしる"}),
        ("review_flags", "review_patterns", {"review_patterns": True}),
        ("coverage_reasons", "approved_at", {"approved_at": "2026-09-12"}),
        ("review_patterns", "values", {"values": True}),
    ],
)
def test_a_per_part_decision_refuses_a_field_its_schema_does_not_define(
    tmp_path: Path, key: str, stray: str, extra: dict[str, Any]
) -> None:
    """A misspelled or invented field refuses by name; nothing is dropped.

    The sharp case is `dispositions`: a payload carrying `record_id` instead of
    `record_ids` used to be stored as "no rows selected", which the finish reads
    as the **whole part** — one held row's exclusion silently widened to the
    page. Closing the field set is what makes that a refusal instead, and the
    refusal names the field so the typo is visible.
    """

    config = _project(tmp_path)
    job = _job(config)

    with pytest.raises(StudyJobError, match=stray):
        record_choice(
            config,
            job.header.job_id,
            {
                key: {
                    BINDINGS["part"]: _bound(
                        job.header.job_id, **DECISION_PAYLOADS[key], **extra
                    )
                }
            },
            expected_revision=job.revision,
        )
    assert load_study_job(config, job.header.job_id).choices == {}


@pytest.mark.parametrize("supplied", [False, 0, "", {}, None])
def test_a_present_record_ids_is_validated_as_supplied_not_widened(
    tmp_path: Path, supplied: Any
) -> None:
    """A present selection that is not a list of ids is refused, not emptied.

    `entry.get("record_ids") or []` turned every one of these into "no rows
    selected", which §7.7 reads as the whole part. A value the owner's control
    could not have meant is a refusal: janki does not repair one decision into a
    wider one.
    """

    config = _project(tmp_path)
    job = _job(config)

    with pytest.raises(StudyJobError, match="distinct nonblank record ids"):
        record_choice(
            config,
            job.header.job_id,
            {
                "dispositions": {
                    BINDINGS["part"]: _bound(
                        job.header.job_id,
                        action="exclude",
                        reason="Only this row repeats lesson 9.",
                        record_ids=supplied,
                    )
                }
            },
            expected_revision=job.revision,
        )
    assert load_study_job(config, job.header.job_id).choices == {}


def test_a_disposition_covers_the_whole_part_only_when_it_selects_no_row(
    tmp_path: Path,
) -> None:
    """Absent and explicit `[]` are the two documented whole-part spellings."""

    config = _project(tmp_path)
    job = _job(config)
    absent = record_choice(
        config,
        job.header.job_id,
        {
            "dispositions": {
                "p1.png": _bound(
                    job.header.job_id,
                    part="p1.png",
                    action="defer",
                    reason="I will read this page with lesson 12.",
                )
            }
        },
        expected_revision=job.revision,
    )
    explicit = record_choice(
        config,
        job.header.job_id,
        {
            "dispositions": {
                "p2.png": _bound(
                    job.header.job_id,
                    part="p2.png",
                    action="defer",
                    reason="Same page, stated the long way.",
                    record_ids=[],
                )
            }
        },
        expected_revision=absent.revision,
    )

    assert explicit.choices["dispositions"]["p1.png"]["record_ids"] == []
    assert explicit.choices["dispositions"]["p2.png"]["record_ids"] == []


def test_a_review_selection_states_which_rows_it_flags_and_nothing_else(
    tmp_path: Path,
) -> None:
    """"None of them" is a decision the owner states, not an omitted argument.

    An empty `record_ids` list is accepted and stored; a *missing* key refuses.
    The pattern-set mark is deliberately absent from this entry — it is its own
    `review_patterns` choice, which is the one fact the finish reads, so a review
    selection cannot hold a second copy for the two to disagree over.
    """

    config = _project(tmp_path)
    job = _job(config)
    saved = record_choice(
        config,
        job.header.job_id,
        {"review_flags": {BINDINGS["part"]: _bound(job.header.job_id, record_ids=[])}},
        expected_revision=job.revision,
    )
    stored = saved.choices["review_flags"][BINDINGS["part"]]
    assert stored["record_ids"] == []
    assert set(stored) == {
        "job_id",
        "part",
        "staging_sha256",
        "rendering_fingerprint",
        "record_ids",
        "saved_at",
    }

    with pytest.raises(StudyJobError, match="which rows it flags"):
        record_choice(
            config,
            job.header.job_id,
            {"review_flags": {BINDINGS["part"]: _bound(job.header.job_id)}},
            expected_revision=saved.revision,
        )
    with pytest.raises(StudyJobError, match="distinct nonblank record ids"):
        record_choice(
            config,
            job.header.job_id,
            {
                "review_flags": {
                    BINDINGS["part"]: _bound(
                        job.header.job_id,
                        record_ids=["word:話す:はなす", "word:話す:はなす"],
                    )
                }
            },
            expected_revision=saved.revision,
        )


def test_the_pattern_review_choice_is_the_one_place_that_mark_is_stored(
    tmp_path: Path,
) -> None:
    """One standalone true-or-false fact per part, and no second copy anywhere.

    `False` is the owner saying "not this part's patterns" and is stored as
    stated; a value that is neither refuses rather than being inferred. Saving a
    review selection for the same part adds no pattern field of its own.
    """

    config = _project(tmp_path)
    job = _job(config)
    marked = record_choice(
        config,
        job.header.job_id,
        {"review_patterns": {BINDINGS["part"]: _bound(job.header.job_id, value=True)}},
        expected_revision=job.revision,
    )
    stored = marked.choices["review_patterns"][BINDINGS["part"]]
    assert stored["value"] is True
    assert set(stored) == {
        "job_id",
        "part",
        "staging_sha256",
        "rendering_fingerprint",
        "value",
        "saved_at",
    }

    with_flags = record_choice(
        config,
        job.header.job_id,
        {"review_flags": {BINDINGS["part"]: _bound(job.header.job_id, record_ids=[])}},
        expected_revision=marked.revision,
    )
    assert "review_patterns" not in with_flags.choices["review_flags"][BINDINGS["part"]]
    assert with_flags.choices["review_patterns"][BINDINGS["part"]]["value"] is True

    cleared = record_choice(
        config,
        job.header.job_id,
        {"review_patterns": {BINDINGS["part"]: _bound(job.header.job_id, value=False)}},
        expected_revision=with_flags.revision,
    )
    assert cleared.choices["review_patterns"][BINDINGS["part"]]["value"] is False

    with pytest.raises(StudyJobError, match="never inferred"):
        record_choice(
            config,
            job.header.job_id,
            {
                "review_patterns": {
                    BINDINGS["part"]: _bound(job.header.job_id, value="yes")
                }
            },
            expected_revision=cleared.revision,
        )


def test_the_sentence_audio_choice_binds_the_job_and_its_revision_and_nothing_else(
    tmp_path: Path,
) -> None:
    """Job-wide, so no part, no staging hash, no rendering and no reason.

    A re-rendered staging file cannot stale this preference, which is exactly
    why it must not carry the bindings the per-part decisions carry.
    """

    config = _project(tmp_path)
    job = _job(config)

    saved = record_choice(
        config,
        job.header.job_id,
        {"include_example_audio": {"value": False}},
        expected_revision=job.revision,
    )

    stored = saved.choices["include_example_audio"]
    assert stored["value"] is False
    assert stored["job_id"] == job.header.job_id
    assert set(stored) == {"value", "job_id", "saved_at"}

    # Re-enabling is the same control and the same CAS, not a second key.
    reenabled = record_choice(
        config,
        job.header.job_id,
        {"include_example_audio": {"value": True}},
        expected_revision=saved.revision,
    )
    assert reenabled.choices["include_example_audio"]["value"] is True

    for stray, message in (
        ({"value": False, "part": BINDINGS["part"]}, "records no part"),
        (
            {"value": False, "rendering_fingerprint": "b" * 64},
            "records no rendering_fingerprint",
        ),
        ({"value": False, "reason": "too expensive"}, "records no reason"),
        ({"value": "no"}, "exactly true or false"),
        ({}, "exactly true or false"),
    ):
        with pytest.raises(StudyJobError, match=message):
            record_choice(
                config,
                job.header.job_id,
                {"include_example_audio": stray},
                expected_revision=reenabled.revision,
            )
    with pytest.raises(StudyJobError, match="another study job"):
        record_choice(
            config,
            job.header.job_id,
            {"include_example_audio": {"value": True, "job_id": "someone-else"}},
            expected_revision=reenabled.revision,
        )
    assert (
        load_study_job(config, job.header.job_id).choices["include_example_audio"][
            "value"
        ]
        is True
    )


def test_every_writable_choice_key_has_its_own_validator(tmp_path: Path) -> None:
    """No catch-all: a key in the schema with no checker fails loudly.

    `CHOICE_KEYS` is the closed list, and the dispatch below it is exhaustive by
    construction. A key added to one without the other is the exact mistake this
    asserts against.
    """

    assert set(study_job.CHOICE_KEYS) == {
        "coverage_reasons",
        "directions",
        "dispositions",
        "include_example_audio",
        "part_layout_bindings",
        "part_selections",
        "review_flags",
        "review_patterns",
    }
    config = _project(tmp_path)
    job = _job(config)
    with pytest.raises(StudyJobError, match="not a study job choice"):
        record_choice(
            config,
            job.header.job_id,
            {"include_example_audio_v2": {"value": True}},
            expected_revision=job.revision,
        )


def test_record_choice_refuses_a_layout_binding_to_a_revision_that_does_not_exist(
    tmp_path: Path,
) -> None:
    """`part_layout_bindings` holds references to immutable revisions only."""

    config = _project(tmp_path)
    job = _job(config)
    with pytest.raises(StudyJobError) as error:
        record_choice(
            config,
            job.header.job_id,
            {"part_layout_bindings": {"verbs-p1.png": ["layout-a", 1]}},
            expected_revision=job.revision,
        )
    assert "layout-a" in str(error.value)


def test_record_choice_refuses_an_unknown_key_and_a_stale_revision(
    tmp_path: Path,
) -> None:
    """Compare and swap, and a closed key set. Neither is advisory."""

    config = _project(tmp_path)
    job = _job(config)
    with pytest.raises(StudyJobError):
        record_choice(
            config,
            job.header.job_id,
            {"scope": "anything"},
            expected_revision=job.revision,
        )
    saved = record_choice(
        config,
        job.header.job_id,
        {"directions": ["reading"]},
        expected_revision=job.revision,
    )
    with pytest.raises(StudyJobError) as error:
        record_choice(
            config,
            job.header.job_id,
            {"directions": ["recognition"]},
            expected_revision=job.revision,
        )
    assert "changed" in str(error.value).lower()
    assert load_study_job(config, job.header.job_id).revision == saved.revision


def test_appending_an_intent_is_append_only_and_immutable(tmp_path: Path) -> None:
    """No edit, no reorder, no duplicate id, and no mutable field inside one."""

    config = _project(tmp_path)
    job = _job(config)
    first = ActionIntent(
        intent_id=study_job.new_intent_id(),
        kind="source_parts",
        decided_at="2026-09-09T10:00:00+00:00",
        reserves={"recipe_id": "r-1", "receipt_sha256": "77aa"},
        bindings={"plan_fingerprint": "fp"},
    )
    saved = append_intent(config, job.header.job_id, first, expected_revision=job.revision)
    assert [intent.intent_id for intent in saved.intents] == [first.intent_id]

    with pytest.raises(StudyJobError) as error:
        append_intent(
            config,
            job.header.job_id,
            ActionIntent(
                intent_id=first.intent_id,
                kind="extract_batch",
                decided_at="2026-09-09T10:05:00+00:00",
                reserves={"batch_id": "b-1"},
                bindings={},
            ),
            expected_revision=saved.revision,
        )
    assert first.intent_id in str(error.value)

    # An ordinary choice edit cannot clear or replace it either.
    after_choice = record_choice(
        config,
        job.header.job_id,
        {"directions": ["reading"]},
        expected_revision=saved.revision,
    )
    assert [intent.intent_id for intent in after_choice.intents] == [first.intent_id]
    assert after_choice.intents[0].reserves == {
        "recipe_id": "r-1",
        "receipt_sha256": "77aa",
    }


def test_an_intent_is_closed_by_an_outcome_and_never_by_editing_it(
    tmp_path: Path,
) -> None:
    """One outcome per intent, appended; there is no `superseded_by` field."""

    config = _project(tmp_path)
    job = _job(config)
    intent = ActionIntent(
        intent_id=study_job.new_intent_id(),
        kind="source_parts",
        decided_at="2026-09-09T10:00:00+00:00",
        reserves={"recipe_id": "r-1", "receipt_sha256": "77aa"},
        bindings={},
    )
    saved = append_intent(config, job.header.job_id, intent, expected_revision=job.revision)
    outcome = IntentOutcome(
        intent_id=intent.intent_id,
        state="applied",
        at="2026-09-09T10:01:00+00:00",
        observed=(("data/source_parts/r-1.json", "77aa"),),
        consequences={"published": 2, "reused": 0},
    )
    closed = append_outcome(
        config, job.header.job_id, outcome, expected_revision=saved.revision
    )
    assert [entry.intent_id for entry in closed.outcomes] == [intent.intent_id]
    assert "superseded_by" not in json.dumps(closed.to_dict())

    with pytest.raises(StudyJobError):
        append_outcome(
            config,
            job.header.job_id,
            IntentOutcome(
                intent_id=intent.intent_id,
                state="refused",
                at="2026-09-09T10:02:00+00:00",
            ),
            expected_revision=closed.revision,
        )
    with pytest.raises(StudyJobError):
        append_outcome(
            config,
            job.header.job_id,
            IntentOutcome(
                intent_id="i-not-here",
                state="applied",
                at="2026-09-09T10:02:00+00:00",
            ),
            expected_revision=closed.revision,
        )


def test_a_successor_supersedes_only_an_intent_an_outcome_already_closed(
    tmp_path: Path,
) -> None:
    """A replan, never a rollback: the original and its evidence stay."""

    config = _project(tmp_path)
    job = _job(config)
    first = ActionIntent(
        intent_id=study_job.new_intent_id(),
        kind="source_parts",
        decided_at="2026-09-09T10:00:00+00:00",
        reserves={"recipe_id": "r-1", "receipt_sha256": "77aa"},
        bindings={},
    )
    saved = append_intent(config, job.header.job_id, first, expected_revision=job.revision)
    successor = ActionIntent(
        intent_id=study_job.new_intent_id(),
        kind="source_parts",
        decided_at="2026-09-09T10:10:00+00:00",
        reserves={"recipe_id": "r-2", "receipt_sha256": "88bb"},
        bindings={},
        supersedes=first.intent_id,
    )
    with pytest.raises(StudyJobError) as error:
        append_intent(
            config, job.header.job_id, successor, expected_revision=saved.revision
        )
    assert first.intent_id in str(error.value)

    closed = append_outcome(
        config,
        job.header.job_id,
        IntentOutcome(
            intent_id=first.intent_id,
            state="abandoned",
            at="2026-09-09T10:09:00+00:00",
        ),
        expected_revision=saved.revision,
    )
    replanned = append_intent(
        config, job.header.job_id, successor, expected_revision=closed.revision
    )
    assert [intent.intent_id for intent in replanned.intents] == [
        first.intent_id,
        successor.intent_id,
    ]


def test_an_intent_kind_outside_the_closed_list_refuses_by_name(
    tmp_path: Path,
) -> None:
    """`INTENT_KINDS` is the one kind check, and `"download"` is what it refuses.

    Every kind in the closed list has an owning service — ``curation`` arrived
    with `application/study_curation.py` and ``finish`` with
    `application/study_finish.py` — so the second "is there a writer for this
    kind yet" guard has been removed rather than left as an unreachable safety
    net. This is the refusal that remains, named for what it actually exercises.
    """

    config = _project(tmp_path)
    job = _job(config)
    assert study_job.INTENT_KINDS == (
        "source_parts",
        "extract_batch",
        "retry",
        "curation",
        "finish",
    )
    with pytest.raises(StudyJobError) as error:
        append_intent(
            config,
            job.header.job_id,
            ActionIntent(
                intent_id=study_job.new_intent_id(),
                kind="download",
                decided_at="2026-09-09T10:00:00+00:00",
                reserves={},
                bindings={},
            ),
            expected_revision=job.revision,
        )
    assert "download" in str(error.value)
    assert load_study_job(config, job.header.job_id).intents == ()


def test_a_finish_intent_is_writable_and_resolves_to_its_authority(
    tmp_path: Path,
) -> None:
    """A reserved finish receipt id resolves to the record at that exact path.

    The finish record is named for the SHA-256 of the authority it carries, so
    the reserved id *is* the path. Until the authority exists the intent stays
    open and unresolved rather than being closed or fabricated.
    """

    config = _project(tmp_path)
    job = _job(config)
    receipt_id = "e" * 64
    intent = ActionIntent(
        intent_id=study_job.new_intent_id(),
        kind="finish",
        decided_at="2026-09-12T10:00:00+00:00",
        reserves={"receipt_id": receipt_id},
        bindings={},
    )

    saved = append_intent(
        config, job.header.job_id, intent, expected_revision=job.revision
    )

    assert [entry.kind for entry in saved.intents] == ["finish"]
    status = study_job_status(config, job.header.job_id)
    assert status.open_intents == (intent.intent_id,)
    assert status.unresolved_intents == (intent.intent_id,)
    assert discover_actions(config, job.header.job_id) == ()


def test_a_job_document_grants_no_spending_or_discard_authority(
    tmp_path: Path,
) -> None:
    """Writing every intent this store can write journals nothing at all."""

    config = _project(tmp_path)
    job = _job(config)
    plan = _batch_plan(config, [_source(config, "one.pdf", b"one")])
    append_intent(
        config, job.header.job_id, _batch_intent(plan), expected_revision=job.revision
    )
    assert not config.operations_file.exists()
    assert not (config.operations_file.parent / "extraction_batches").exists()


# --- backlinks that survive a crash ------------------------------------------


def _crashed_batch(
    config: ProjectConfig,
    job: Any,
    sources: list[Path],
    *,
    runner: FakeClaudeRunner | None = None,
) -> tuple[Any, ActionIntent]:
    """Append the intent, dispatch, and never append the outcome."""

    current = load_study_job(config, job.header.job_id)
    plan = _batch_plan(config, sources, job_id=job.header.job_id)
    intent = _batch_intent(plan)
    append_intent(config, job.header.job_id, intent, expected_revision=current.revision)
    provider = runner if runner is not None else _runner()
    extraction_batch.dispatch_extraction_batch(
        config,
        plan,
        provider_env={},
        provider_runner=provider,
        provider_which=_which,
        provider_spawn=provider.spawn,
    )
    return plan, intent


def test_discover_actions_finds_the_exact_reserved_artifact_after_a_crash(
    tmp_path: Path,
) -> None:
    """The reference write did not land; the reserved id and hash still bind it."""

    config = _project(tmp_path)
    job = _job(config)
    sources = [_source(config, "one.pdf", b"one"), _source(config, "two.pdf", b"two")]
    plan, intent = _crashed_batch(config, job, sources)

    # A later, unrelated, jobless batch over the same two sources.
    jobless = _batch_plan(config, sources, force=True)
    runner = _runner()
    extraction_batch.dispatch_extraction_batch(
        config,
        jobless,
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
        provider_spawn=runner.spawn,
    )
    newest = extraction_batch.batch_manifest_path(config, jobless.batch_id)
    assert newest.stat().st_mtime >= extraction_batch.batch_manifest_path(
        config, plan.batch_id
    ).stat().st_mtime

    found = discover_actions(config, job.header.job_id)

    assert len(found) == 1
    assert found[0].intent_id == intent.intent_id
    assert found[0].kind == "extract_batch"
    assert found[0].path == extraction_batch.batch_manifest_path(config, plan.batch_id)
    assert found[0].sha256 == plan.manifest_sha256
    assert found[0].job_id == job.header.job_id
    # The jobless sibling is never a candidate, whatever its modification time.
    assert jobless.batch_id not in {reference.reserves["batch_id"] for reference in found}


def test_discover_actions_refuses_an_artifact_whose_embedded_job_id_differs(
    tmp_path: Path,
) -> None:
    """Same reserved id and same hash is not enough; the batch must say whose."""

    config = _project(tmp_path)
    job = _job(config)
    other = _job(config)
    sources = [_source(config, "one.pdf", b"one")]
    plan = _batch_plan(config, sources, job_id=other.header.job_id)
    intent = _batch_intent(plan)
    append_intent(config, job.header.job_id, intent, expected_revision=job.revision)
    runner = _runner()
    extraction_batch.dispatch_extraction_batch(
        config,
        plan,
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
        provider_spawn=runner.spawn,
    )

    assert discover_actions(config, job.header.job_id) == ()
    assert [
        reference.reserves["batch_id"]
        for reference in discover_actions(config, other.header.job_id)
    ] == []


def test_discover_actions_refuses_a_manifest_whose_bytes_were_edited(
    tmp_path: Path,
) -> None:
    """A hand-edited manifest is never adopted, whatever its name says."""

    config = _project(tmp_path)
    job = _job(config)
    plan, _intent = _crashed_batch(config, job, [_source(config, "one.pdf", b"one")])
    manifest = extraction_batch.batch_manifest_path(config, plan.batch_id)
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["concurrency_limit"] = 4
    manifest.write_text(json.dumps(raw, sort_keys=True) + "\n", encoding="utf-8")

    assert discover_actions(config, job.header.job_id) == ()


def test_discover_actions_returns_only_intents_no_outcome_has_closed(
    tmp_path: Path,
) -> None:
    """Recovery is for the backlink that did not land, not for finished work."""

    config = _project(tmp_path)
    job = _job(config)
    plan, intent = _crashed_batch(config, job, [_source(config, "one.pdf", b"one")])
    assert len(discover_actions(config, job.header.job_id)) == 1

    current = load_study_job(config, job.header.job_id)
    append_outcome(
        config,
        job.header.job_id,
        IntentOutcome(
            intent_id=intent.intent_id,
            state="applied",
            at="2026-09-09T11:00:00+00:00",
            observed=(
                (
                    str(extraction_batch.batch_manifest_path(config, plan.batch_id)),
                    plan.manifest_sha256,
                ),
            ),
        ),
        expected_revision=current.revision,
    )
    assert discover_actions(config, job.header.job_id) == ()


# --- derived status -----------------------------------------------------------


def test_study_job_status_derives_child_states_from_the_journal(
    tmp_path: Path,
) -> None:
    """No stored progress: the journal says what each child did."""

    config = _project(tmp_path)
    job = _job(config)
    sources = [_source(config, "one.pdf", b"one"), _source(config, "two.pdf", b"two")]
    plan, intent = _crashed_batch(config, job, sources)
    # A second batch, over different sources, whose provider refused. A status
    # that reported the ids the intent reserved rather than what the journal
    # says became of them would have to invent this one's states.
    refused, refused_intent = _crashed_batch(
        config,
        job,
        [_source(config, "three.pdf", b"three")],
        runner=FakeClaudeRunner(reply=b'{"type":"result","is_error":true}\n'),
    )

    status = study_job_status(config, job.header.job_id)

    assert status.job_id == job.header.job_id
    assert status.deck_present is True
    assert status.deck_current is True
    assert [entry.batch_id for entry in status.batches] == [
        plan.batch_id,
        refused.batch_id,
    ]
    journal = operations.OperationJournal.load(config.operations_file)
    derived = {
        child.operation_id: journal.operations[child.operation_id].state
        for child in (*plan.children, *refused.children)
    }
    assert {
        child.operation_id: child.state
        for entry in status.batches
        for child in entry.children
    } == derived
    assert set(derived.values()) != {"committed"}, "the fixture must disagree"

    settled, unsettled = status.batches
    assert settled.intent_id == intent.intent_id and settled.missing is False
    assert settled.committed_count == 2 and settled.failed_count == 0
    assert unsettled.intent_id == refused_intent.intent_id
    assert unsettled.committed_count == 0 and unsettled.failed_count == 1
    # The staging fact is measured too, not remembered.
    committed = settled.children[0]
    assert committed.staging_present is True and committed.archive_present is False
    assert status.open_intents == (intent.intent_id, refused_intent.intent_id)


def test_study_job_status_reports_an_intent_whose_artifact_is_missing(
    tmp_path: Path,
) -> None:
    """A reserved id with nothing behind it is stated, never invented."""

    config = _project(tmp_path)
    job = _job(config)
    plan = _batch_plan(config, [_source(config, "one.pdf", b"one")], job_id=job.header.job_id)
    append_intent(
        config, job.header.job_id, _batch_intent(plan), expected_revision=job.revision
    )

    status = study_job_status(config, job.header.job_id)
    assert len(status.batches) == 1
    assert status.batches[0].missing is True
    assert status.batches[0].children == ()
    assert status.batches[0].committed_count == 0


def test_study_job_status_notices_a_deck_that_changed_under_the_job(
    tmp_path: Path,
) -> None:
    """The deck definition is the authority; the job's hash is only a binding."""

    config = _project(tmp_path)
    job = _job(config)
    deck = config.deck_dir / "201-verbs.yaml"
    deck.write_text(deck.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")

    status = study_job_status(config, job.header.job_id)
    assert status.deck_present is True
    assert status.deck_current is False


def test_study_job_status_measures_publication_rather_than_remembering_it(
    tmp_path: Path,
) -> None:
    """A source-part intent's counts come from the corpus, not from the receipt."""

    config = _project(tmp_path)
    job = _job(config)
    receipts = config.operations_file.parent / "source_parts"
    receipts.mkdir(parents=True, exist_ok=True)
    recipe_id = "3f1a0c1e-2b8d-4c6a-9f0e-5d4c3b2a1908"
    part = config.scan_inbox / "verbs-p1.png"
    part.parent.mkdir(parents=True, exist_ok=True)
    part.write_bytes(b"published bytes")
    payload = json.dumps(
        {
            "version": 1,
            "recipe_id": recipe_id,
            "parent_name": "verbs.pdf",
            "parent_sha256": "9f2c",
            "renderer": "pdfium",
            "renderer_version": "1",
            "encoder": "pillow",
            "encoder_version": "1",
            "render_dpi": 200,
            "recipe_sha256": "aa",
            "plan_fingerprint": "fp",
            "created_at": "2026-09-09T10:00:00+00:00",
            "parts": [
                {
                    "ordinal": 1,
                    "target_name": "verbs-p1.png",
                    "sha256": hashlib.sha256(b"published bytes").hexdigest(),
                    "byte_length": 15,
                    "page_index": 0,
                    "page_rotate": 0,
                    "page_size_pt": [612.0, 792.0],
                    "pixel_rect": [0, 0, 100, 100],
                    "regions": [],
                },
                {
                    "ordinal": 2,
                    "target_name": "verbs-p2.png",
                    "sha256": hashlib.sha256(b"never published").hexdigest(),
                    "byte_length": 15,
                    "page_index": 1,
                    "page_rotate": 0,
                    "page_size_pt": [612.0, 792.0],
                    "pixel_rect": [0, 0, 100, 100],
                    "regions": [],
                },
            ],
        },
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ) + "\n"
    (receipts / f"{recipe_id}.json").write_text(payload, encoding="utf-8")

    append_intent(
        config,
        job.header.job_id,
        ActionIntent(
            intent_id=study_job.new_intent_id(),
            kind="source_parts",
            decided_at="2026-09-09T10:00:00+00:00",
            reserves={
                "recipe_id": recipe_id,
                "receipt_sha256": hashlib.sha256(
                    payload.encode("utf-8")
                ).hexdigest(),
            },
            bindings={},
        ),
        expected_revision=job.revision,
    )

    status = study_job_status(config, job.header.job_id)
    assert len(status.parts) == 1
    assert status.parts[0].recipe_id == recipe_id
    assert status.parts[0].part_count == 2
    assert status.parts[0].published_count == 1
    assert status.parts[0].missing is False


def test_resuming_finds_the_interrupted_batch_by_its_reserved_id_and_hash(
    tmp_path: Path,
) -> None:
    """No new approval, no new operation id: the recorded authority finishes."""

    config = _project(tmp_path)
    job = _job(config)
    sources = [_source(config, "one.pdf", b"one")]
    plan, intent = _crashed_batch(config, job, sources)
    before = operations.OperationJournal.load(config.operations_file)

    outcomes = study_job.resume_job_actions(config, job.header.job_id)

    assert len(outcomes) == 1
    assert outcomes[0].batch_id == plan.batch_id
    # Nothing was reserved again and no new identity was minted.
    after = operations.OperationJournal.load(config.operations_file)
    assert set(after.operations) == set(before.operations)

    settled = load_study_job(config, job.header.job_id)
    assert [outcome.intent_id for outcome in settled.outcomes] == [intent.intent_id]
    assert settled.outcomes[0].observed == (
        (
            str(extraction_batch.batch_manifest_path(config, plan.batch_id)),
            plan.manifest_sha256,
        ),
    )
    # Closed now, so there is nothing left to rediscover.
    assert discover_actions(config, job.header.job_id) == ()
    assert study_job.resume_job_actions(config, job.header.job_id) == ()


# --- what a returning dispatch has and has not finished -----------------------
#
# A worker returning is not the same statement as the job's recorded authority
# being spent. `dispatch_extraction_batch` returns normally with a child still
# reserved and unsent after a transport refusal, and with a paid answer on disk
# that no staging write accepted. Both are exactly what the batch's own resume
# finishes under the authority already recorded, so an intent closed on return
# would put them out of this job's reach for good.


def _refuse_transport_for(monkeypatch: pytest.MonkeyPatch, name: str) -> Any:
    """One child's preflight refuses; every sibling's is the real one.

    The shape the batch service already documents: a logged-out subscription
    CLI or a missing binary refuses *before* anything is sent, stops queuing
    more work, and leaves that child holding reserved, unspent authority.
    """

    real = extraction_batch.prepare_extraction_transport

    def refuse(call_plan: Any, target: Any, **kwargs: Any) -> Any:
        if target.name == name:
            raise JankiError("temporary login refusal")
        return real(call_plan, target, **kwargs)

    monkeypatch.setattr(extraction_batch, "prepare_extraction_transport", refuse)
    return real


def _refuse_staging_for(monkeypatch: pytest.MonkeyPatch, stem: str) -> Any:
    """One child's staging write fails after its reply was captured and paid for."""

    real = extraction.write_staging_under_lock

    def refuse(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path.name.startswith(stem):
            raise staging.StagingError("the disk went away")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(extraction, "write_staging_under_lock", refuse)
    return real


def test_a_dispatch_that_left_a_child_reserved_keeps_its_intent_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reserved unsent child stays reachable from the job that reserved it.

    The first source commits and the second's transport preflight refuses, so
    `dispatch_extraction_batch` returns with one child still `authorized` —
    reserved, unsent and unspent. That is the batch's own resume shape, and the
    job's intent has to stay open for it: repeatedly, while the refusal
    repeats, and without a new approval, a new operation id, or a committed
    sibling being sent a second time.
    """

    config = _project(tmp_path)
    job = _job(config)
    sources = [_source(config, "one.pdf", b"one"), _source(config, "two.pdf", b"two")]
    plan = _batch_plan(
        config, sources, job_id=job.header.job_id, concurrency_limit=1
    )
    runner = _runner()
    real_prepare = _refuse_transport_for(monkeypatch, "two.pdf")

    outcome = study_job.dispatch_job_batch(
        config,
        job.header.job_id,
        plan,
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
        provider_spawn=runner.spawn,
    )

    assert [child.state for child in outcome.children] == ["committed", "authorized"]
    reserved = operations.OperationJournal.load(config.operations_file)
    saved = load_study_job(config, job.header.job_id)
    intent = saved.intents[0]
    # The authority the intent recorded is not spent, so nothing closed it.
    assert saved.outcomes == ()
    assert study_job_status(config, job.header.job_id).open_intents == (
        intent.intent_id,
    )
    assert [reference.intent_id for reference in discover_actions(config, job.header.job_id)] == [
        intent.intent_id
    ]

    # Resumed while the same refusal still stands: it continues exactly that
    # recorded authority, refuses again, and stays open for the next attempt.
    again = study_job.resume_job_actions(
        config,
        job.header.job_id,
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
        provider_spawn=runner.spawn,
    )
    assert len(again) == 1
    assert again[0].intent_id == intent.intent_id
    assert again[0].closed is False
    assert again[0].batch.children[1].state == "authorized"
    assert load_study_job(config, job.header.job_id).outcomes == ()
    assert len(runner.spawned) == 1, "the committed sibling was sent again"

    monkeypatch.setattr(
        extraction_batch, "prepare_extraction_transport", real_prepare
    )
    finished = study_job.resume_job_actions(
        config,
        job.header.job_id,
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
        provider_spawn=runner.spawn,
    )

    assert len(finished) == 1
    assert finished[0].closed is True
    assert [child.state for child in finished[0].batch.children] == [
        "committed",
        "committed",
    ]
    # One send per child, ever: the reserved authority was spent once and the
    # committed sibling was never redispatched.
    assert len(runner.spawned) == 2
    after = operations.OperationJournal.load(config.operations_file)
    assert set(after.operations) == set(reserved.operations)
    closed = load_study_job(config, job.header.job_id)
    assert [entry.state for entry in closed.outcomes] == ["applied"]
    assert closed.outcomes[0].intent_id == intent.intent_id
    assert discover_actions(config, job.header.job_id) == ()
    assert study_job.resume_job_actions(config, job.header.job_id) == ()


def test_a_dispatch_that_captured_a_reply_it_could_not_stage_keeps_its_intent_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A paid answer on disk is the job's to finish, for free, from its own resume."""

    config = _project(tmp_path)
    job = _job(config)
    sources = [_source(config, "one.pdf", b"one"), _source(config, "two.pdf", b"two")]
    plan = _batch_plan(
        config, sources, job_id=job.header.job_id, concurrency_limit=1
    )
    runner = _runner()
    real_write = _refuse_staging_for(monkeypatch, "two")

    outcome = study_job.dispatch_job_batch(
        config,
        job.header.job_id,
        plan,
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
        provider_spawn=runner.spawn,
    )

    assert [child.state for child in outcome.children] == [
        "committed",
        "result_captured",
    ]
    saved = load_study_job(config, job.header.job_id)
    assert saved.outcomes == ()
    assert study_job_status(config, job.header.job_id).open_intents == (
        saved.intents[0].intent_id,
    )

    monkeypatch.setattr(extraction, "write_staging_under_lock", real_write)

    def never(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("staging a captured answer consulted the provider")

    resumed = study_job.resume_job_actions(
        config,
        job.header.job_id,
        provider_env={},
        provider_runner=never,
        provider_which=never,
        provider_spawn=never,
    )

    assert len(resumed) == 1
    assert resumed[0].closed is True
    assert [child.state for child in resumed[0].batch.children] == [
        "committed",
        "committed",
    ]
    assert plan.children[1].staging_path.exists()
    assert len(runner.spawned) == 2, "recovering a saved answer is not a send"
    closed = load_study_job(config, job.header.job_id)
    assert [entry.state for entry in closed.outcomes] == ["applied"]
    assert discover_actions(config, job.header.job_id) == ()


# --- a publication whose outcome write was lost -------------------------------


class _PlannedPart:
    """One planned part, in the shape the publication writer reads.

    Hand-built rather than rendered: what these tests are about is the job's
    binding to a publication, not PDFium's output, and a real render would make
    them depend on an optional dependency.
    """

    def __init__(self, ordinal: int, name: str, data: bytes) -> None:
        self.ordinal = ordinal
        self.target_name = name
        self.sha256 = hashlib.sha256(data).hexdigest()
        self.byte_length = len(data)
        self.page_index = ordinal - 1
        self.page_rotate = 0
        self.page_size_pt = (612.0, 792.0)
        self.pixel_rect = (0, 0, 100, 100)
        self.regions: tuple[Any, ...] = ()
        self.thumbnail_png_base64 = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "target_name": self.target_name,
            "sha256": self.sha256,
            "byte_length": self.byte_length,
            "page_index": self.page_index,
            "page_rotate": self.page_rotate,
            "page_size_pt": list(self.page_size_pt),
            "pixel_rect": list(self.pixel_rect),
            "regions": [],
        }


RECIPE_ID = "3f1a0c1e-2b8d-4c6a-9f0e-5d4c3b2a1908"


def _parts_plan(
    *payloads: bytes,
    suffix: str = "",
    render_dpi: int = 200,
    fingerprint: str = "f" * 64,
) -> Any:
    """One plan over this recipe id.

    The keyword arguments are what an *edited* recipe under the same id
    differs in: `plan_source_parts` puts the recipe's own hash in every part
    name, so new geometry renders different bytes under different names at
    whatever dpi the owner asked for.
    """

    parts = tuple(
        _PlannedPart(ordinal, f"verbs-p{ordinal}{suffix}.png", data)
        for ordinal, data in enumerate(payloads, start=1)
    )
    return source_parts.SourcePartsPlan(
        recipe_id=RECIPE_ID,
        parent_name="verbs.pdf",
        parent_sha256="9f2c",
        renderer="pdfium",
        renderer_version="1",
        encoder="pillow",
        encoder_version="1",
        render_dpi=render_dpi,
        plan_fingerprint=fingerprint,
        parts=parts,
        recipe_sha256="a" * 64,
        payloads=tuple(payloads),
    )


def _edited_recipe_plan() -> Any:
    """The same recipe id after the owner moved the boxes and raised the dpi.

    A recipe id is an owner-written UUID rather than a content hash
    (`source_parts.new_recipe_id`), so copying a recipe file and editing it
    reuses the id of a publication that already exists.
    """

    return _parts_plan(
        b"rendered from the new boxes",
        suffix="-r5e5e5e5e",
        render_dpi=300,
        fingerprint="e" * 64,
    )


def _interrupted_publication(
    config: ProjectConfig, job_id: str, plan: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact window §2.4 exists for: the receipt and its parts land, the
    outcome write does not. Killed inside the real service, not simulated."""

    real_append = study_job.append_outcome

    def die(*args: Any, **kwargs: Any) -> Any:
        raise KeyboardInterrupt("the process died before the outcome write landed")

    monkeypatch.setattr(study_job, "append_outcome", die)
    try:
        with pytest.raises(KeyboardInterrupt):
            study_job.publish_job_source_parts(
                config, job_id, plan, publish_token=plan.plan_fingerprint
            )
    finally:
        monkeypatch.setattr(study_job, "append_outcome", real_append)


def test_resume_closes_a_publication_whose_receipt_and_parts_already_landed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parts are in the corpus; nothing may need re-rendering to use them.

    Recovery is by the exact recipe id and receipt hash this job reserved —
    never the newest receipt — and closing it is what makes those published
    parts available to this job's extraction.
    """

    config = _project(tmp_path)
    job = _job(config)
    plan = _parts_plan(b"rendered part one")
    _interrupted_publication(config, job.header.job_id, plan, monkeypatch)

    receipt_path = source_parts.receipt_path(config, RECIPE_ID)
    receipt_sha256 = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    assert (config.scan_inbox / "verbs-p1.png").is_file()
    saved = load_study_job(config, job.header.job_id)
    assert [intent.kind for intent in saved.intents] == ["source_parts"]
    assert saved.outcomes == ()

    actions = study_job.resume_job_actions(config, job.header.job_id)

    assert len(actions) == 1
    assert actions[0].kind == "source_parts"
    assert actions[0].recipe_id == RECIPE_ID
    assert actions[0].closed is True
    closed = load_study_job(config, job.header.job_id)
    assert [entry.state for entry in closed.outcomes] == ["applied"]
    # Closed from the exact reserved receipt and its exact hash.
    assert closed.outcomes[0].observed == ((str(receipt_path), receipt_sha256),)
    assert closed.outcomes[0].intent_id == saved.intents[0].intent_id
    assert discover_actions(config, job.header.job_id) == ()

    # And the parts it published are what this job would now send.
    assert [
        path.name for path, _lineage in study_job.job_part_sources(config, job.header.job_id)
    ] == ["verbs-p1.png"]

    # Publishing the same recipe again is idempotent for the corpus, and one
    # recipe's parts stay one set of files however many of this job's records
    # bound them: a page is not sent twice for having been recorded twice.
    study_job.publish_job_source_parts(
        config, job.header.job_id, plan, publish_token=plan.plan_fingerprint
    )
    twice = load_study_job(config, job.header.job_id)
    assert [entry.state for entry in twice.outcomes] == ["applied", "applied"]
    assert [
        path.name for path, _lineage in study_job.job_part_sources(config, job.header.job_id)
    ] == ["verbs-p1.png"]


def test_resume_keeps_a_partly_published_receipt_open_and_names_what_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing bytes are the publication service's to write, not the job's.

    A receipt whose parts are not all in the corpus is not a finished
    publication. The job says which part is missing, keeps its intent open with
    the evidence, renders nothing and adopts nothing; the owner's own publish
    finishes it — against the same open intent, so nothing is orphaned.
    """

    config = _project(tmp_path)
    job = _job(config)
    plan = _parts_plan(b"rendered part one", b"rendered part two")

    real_publish = source_parts.inputs.publish_derived_part

    def only_the_first(pairs: Any, **kwargs: Any) -> Any:
        return real_publish(list(pairs)[:1], **kwargs)

    monkeypatch.setattr(source_parts.inputs, "publish_derived_part", only_the_first)
    _interrupted_publication(config, job.header.job_id, plan, monkeypatch)
    monkeypatch.setattr(source_parts.inputs, "publish_derived_part", real_publish)

    receipt_path = source_parts.receipt_path(config, RECIPE_ID)
    receipt_bytes = receipt_path.read_bytes()
    assert (config.scan_inbox / "verbs-p1.png").is_file()
    assert not (config.scan_inbox / "verbs-p2.png").exists()

    actions = study_job.resume_job_actions(config, job.header.job_id)

    assert len(actions) == 1
    assert actions[0].closed is False
    assert "verbs-p2.png" in actions[0].detail
    open_intent = load_study_job(config, job.header.job_id).intents[0]
    assert load_study_job(config, job.header.job_id).outcomes == ()
    # Evidence preserved: the receipt is untouched and the published part stays.
    assert receipt_path.read_bytes() == receipt_bytes
    assert (config.scan_inbox / "verbs-p1.png").is_file()
    assert not (config.scan_inbox / "verbs-p2.png").exists()
    with pytest.raises(StudyJobError):
        study_job.job_part_sources(config, job.header.job_id)

    # The owner's own publish is what writes the missing bytes, and it finishes
    # the intent that is already open rather than orphaning it behind a second.
    study_job.publish_job_source_parts(
        config, job.header.job_id, plan, publish_token=plan.plan_fingerprint
    )

    settled = load_study_job(config, job.header.job_id)
    assert [intent.intent_id for intent in settled.intents] == [open_intent.intent_id]
    assert [entry.state for entry in settled.outcomes] == ["applied"]
    assert [
        path.name for path, _lineage in study_job.job_part_sources(config, job.header.job_id)
    ] == ["verbs-p1.png", "verbs-p2.png"]


def test_a_publish_refused_as_a_different_recipe_records_nothing_to_attach(
    tmp_path: Path,
) -> None:
    """An edited recipe under a published id binds this job to nothing.

    The receipt is immutable, so publishing different parts under the same
    owner-written recipe id refuses and puts nothing in the corpus. There is
    therefore nothing for this job to attach to: an intent reserving the
    *existing* receipt would let a later resume close it `applied` and bind
    this job to a publication it never made, and its batch would then be
    planned over the regions the owner replaced.
    """

    config = _project(tmp_path)
    published = _job(config)
    editing = _job(config)
    original = _parts_plan(b"rendered part one")
    study_job.publish_job_source_parts(
        config,
        published.header.job_id,
        original,
        publish_token=original.plan_fingerprint,
    )
    receipt = source_parts.receipt_path(config, RECIPE_ID)
    held = receipt.read_bytes()
    edited = _edited_recipe_plan()

    with pytest.raises(JankiError) as refused:
        study_job.publish_job_source_parts(
            config, editing.header.job_id, edited, publish_token=edited.plan_fingerprint
        )

    assert "records a different recipe" in str(refused.value)
    assert "nothing was published" in str(refused.value)
    # Nothing was published, so no authority was recorded and there is nothing
    # a resume could turn into an applied publication.
    stalled = load_study_job(config, editing.header.job_id)
    assert stalled.intents == () and stalled.outcomes == ()
    assert study_job.resume_job_actions(config, editing.header.job_id) == ()
    assert load_study_job(config, editing.header.job_id).outcomes == ()
    with pytest.raises(StudyJobError) as unpublished:
        study_job.job_part_sources(config, editing.header.job_id)
    assert "published no source parts yet" in str(unpublished.value)
    # And no paid child is planned over parts this job never published.
    with pytest.raises(StudyJobError) as nothing_to_send:
        study_job.plan_job_extraction_batch(
            config,
            editing.header.job_id,
            provider_env={},
            provider_runner=_runner(),
            provider_which=_which,
        )
    assert "published no source parts yet" in str(nothing_to_send.value)
    # The receipt and the corpus are exactly as the refusal found them.
    assert receipt.read_bytes() == held
    assert (config.scan_inbox / "verbs-p1.png").read_bytes() == b"rendered part one"
    assert not (config.scan_inbox / "verbs-p1-r5e5e5e5e.png").exists()

    # The job that did publish it keeps what it had, and another job may still
    # reuse that same receipt by attaching to it — the receipt binds the parts,
    # not a job.
    assert [
        path.name
        for path, _lineage in study_job.job_part_sources(
            config, published.header.job_id
        )
    ] == ["verbs-p1.png"]
    reusing = _job(config)
    reused = study_job.publish_job_source_parts(
        config, reusing.header.job_id, original, publish_token=original.plan_fingerprint
    )
    assert reused.receipt_sha256 == hashlib.sha256(held).hexdigest()
    assert reused.published == () and len(reused.reused) == 1
    assert receipt.read_bytes() == held
    assert [
        path.name
        for path, _lineage in study_job.job_part_sources(config, reusing.header.job_id)
    ] == ["verbs-p1.png"]


def test_a_publish_against_a_receipt_that_is_not_an_object_refuses_as_a_janki_error(
    tmp_path: Path,
) -> None:
    """A corrupted receipt is a refusal on this seam too, not a crash.

    The publish binds the receipt already on disk before it records anything,
    so a receipt truncated or hand-edited down to `[]`, a bare string or
    `null` is read by that binding rather than by the loader whose `Mapping`
    guard already refuses one. Comparing a non-object receipt's fields raises
    `AttributeError`, which is not a `JankiError`: `cli.main` lets it out as a
    traceback and the desk answers a Python message. Every shape has to refuse
    as a `SourcePartsError` in the writer's own words, and publish nothing.

    Mutant: drop the `Mapping` check in `source_parts._proved_receipt`.
    """

    config = _project(tmp_path)
    job = _job(config)
    plan = _parts_plan(b"rendered part one")
    receipt = source_parts.receipt_path(config, RECIPE_ID)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    corrupted: list[tuple[str, str]] = [
        ("an empty list", "[]"),
        ("a list of part names", '["verbs-p1.png"]'),
        ("a bare string", '"verbs-p1.png"'),
        ("a bare number", "200"),
        ("a null receipt", "null"),
    ]
    for description, payload in corrupted:
        receipt.write_text(payload, encoding="utf-8")

        with pytest.raises(source_parts.SourcePartsError) as refused:
            study_job.publish_job_source_parts(
                config, job.header.job_id, plan, publish_token=plan.plan_fingerprint
            )

        # The writer's own error context: which receipt, and that the refusal
        # published nothing.
        assert str(receipt) in str(refused.value), description
        assert "Nothing was published" in str(refused.value), description
        # JankiError is what `cli.main` and the desk's guard are written
        # against; an AttributeError would escape both.
        assert isinstance(refused.value, JankiError), description
        # Nothing moved: not the corrupted receipt, not the corpus, and not
        # this job's log — there is no reservation for a resume to close.
        assert receipt.read_text(encoding="utf-8") == payload, description
        assert not (config.scan_inbox / "verbs-p1.png").exists(), description
        stalled = load_study_job(config, job.header.job_id)
        assert stalled.intents == () and stalled.outcomes == (), description
        assert study_job.resume_job_actions(config, job.header.job_id) == ()

    # And the recipe id is still free: repairing the receipt by deleting the
    # corruption lets the same plan publish exactly as it would have.
    receipt.unlink()
    published = study_job.publish_job_source_parts(
        config, job.header.job_id, plan, publish_token=plan.plan_fingerprint
    )
    assert [path.name for path in published.published] == ["verbs-p1.png"]
    assert load_study_job(config, job.header.job_id).outcomes[-1].state == "applied"


def test_resume_refuses_a_publication_whose_intent_bound_other_parts(
    tmp_path: Path,
) -> None:
    """Attaching is proved against what the intent itself recorded.

    The receipt hash an intent reserved says which file it meant; the part
    names and hashes it bound say what publishing it was supposed to put in
    the corpus. A resume may only close it when both are that receipt's own,
    so a record left behind by a refused divergent recipe is reported with its
    evidence rather than adopted.
    """

    config = _project(tmp_path)
    publisher = _job(config)
    stranded = _job(config)
    original = _parts_plan(b"rendered part one")
    receipt = study_job.publish_job_source_parts(
        config,
        publisher.header.job_id,
        original,
        publish_token=original.plan_fingerprint,
    )
    edited = _edited_recipe_plan()
    job = load_study_job(config, stranded.header.job_id)
    append_intent(
        config,
        stranded.header.job_id,
        ActionIntent(
            intent_id=study_job.new_intent_id(),
            kind="source_parts",
            decided_at="2026-09-09T10:00:00+00:00",
            reserves={
                "recipe_id": RECIPE_ID,
                "receipt_sha256": receipt.receipt_sha256,
            },
            bindings={
                "plan_fingerprint": edited.plan_fingerprint,
                "parent_name": edited.parent_name,
                "parent_sha256": edited.parent_sha256,
                "created_at": "",
                "target_names": [part.target_name for part in edited.parts],
                "part_sha256": [part.sha256 for part in edited.parts],
            },
        ),
        expected_revision=job.revision,
    )

    actions = study_job.resume_job_actions(config, stranded.header.job_id)

    assert len(actions) == 1
    assert actions[0].closed is False
    assert actions[0].recipe_id == RECIPE_ID
    assert "verbs-p1-r5e5e5e5e.png" in actions[0].detail
    assert load_study_job(config, stranded.header.job_id).outcomes == ()
    with pytest.raises(StudyJobError) as unpublished:
        study_job.job_part_sources(config, stranded.header.job_id)
    assert "published no source parts yet" in str(unpublished.value)
    # Evidence intact: the receipt and the parts it does name stay put, and
    # the parts this record named were never in the corpus at all.
    assert source_parts.receipt_path(config, RECIPE_ID).read_bytes() == (
        receipt.path.read_bytes()
    )
    assert (config.scan_inbox / "verbs-p1.png").is_file()
    assert not (config.scan_inbox / "verbs-p1-r5e5e5e5e.png").exists()


def test_dispatching_a_batch_planned_for_another_job_refuses_before_sending(
    tmp_path: Path,
) -> None:
    """A backlink is part of what was rendered, not a label applied later."""

    config = _project(tmp_path)
    mine = _job(config)
    theirs = _job(config)
    plan = _batch_plan(
        config, [_source(config, "one.pdf", b"one")], job_id=theirs.header.job_id
    )

    with pytest.raises(StudyJobError) as error:
        study_job.dispatch_job_batch(config, mine.header.job_id, plan)

    assert theirs.header.job_id in str(error.value)
    assert not config.operations_file.exists()
    assert load_study_job(config, mine.header.job_id).intents == ()


def test_a_supplied_deck_hash_is_bound_without_re_reading_the_file(
    tmp_path: Path,
) -> None:
    """The deck's own writer already returned the bytes it wrote.

    Re-reading the file to hash it is a second observation of a value the
    writer handed back, and a concurrent edit between the write and that read
    binds the wrong hash silently. So a caller that has the exact bytes says
    so, and this store uses what it was given.
    """

    config = _project(tmp_path)
    source = _source(config, "verbs.pdf", b"verbs")
    deck = _deck(config)
    written = b"deck:\n  name: what the writer actually wrote\n"
    supplied = hashlib.sha256(written).hexdigest()
    assert supplied != hashlib.sha256(deck.read_bytes()).hexdigest()

    job = open_study_job(
        config,
        kind="source_extraction",
        parent_source=source,
        deck_path=deck,
        deck_sha256=supplied,
    )

    assert job.header.deck_sha256 == supplied
    # And the divergence from what is on disk now is reported, not hidden.
    assert study_job_status(config, job.header.job_id).deck_current is False
