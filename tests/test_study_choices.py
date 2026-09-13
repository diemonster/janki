"""The one owner-decision service both S6 surfaces call.

`docs/ASSISTANT_STUDY_JOBS_CONTRACTS.md` §9.1 and §9.3. The Assistant's
review/disposition editor and `janki study review|coverage|disposition|finish
--example-audio` are two front doors onto **one** service, so these tests are
about that service's own behaviour: what a read discloses, what a save binds,
and which states it refuses rather than resolving.

Everything runs the real derivations against a temporary repository — the real
batch services, the real effective frontier, the real card renderer, the real
`ReviewPanel` structural reader and the real compare-and-swap writer. The only
thing faked is the extraction subscription CLI, plus refusals standing in for
every outbound transport this module must never use.

Nothing here reads Japanese. The rows a part holds are asserted as identities
and digests; the one assertion that mentions the extraction's expressions is
the one proving they are **not** in what the service projects.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from test_revision_provider import FakeClaudeRunner, _which
from test_workbench_fixtures import RESPONSES

from conftest import seed_prompts
from japanese_anki import card_preview, claude_client, jpdb, jpdb_kanji, kanji, staging
from japanese_anki.application import extraction_batch, study_choices, study_job
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.workbench import edit, review

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENARIO = "table_exhaustive"
SECOND_SCENARIO = "shared_word_source_a"

#: A paid reply that reached the disk and cannot be parsed: the child stays at
#: `result_captured`, which is a real not-settled state rather than an invented
#: one.
UNUSABLE_ANSWER = b"not a stream at all\n"

pytestmark = pytest.mark.skipif(
    card_preview.preview_unavailable() is not None,
    reason=str(card_preview.preview_unavailable()),
)


# --- the offline guarantee ----------------------------------------------------


@pytest.fixture(autouse=True)
def _offline() -> Iterator[None]:
    """No paid client, no dictionary and no reference service, for every test.

    Deliberately its own `MonkeyPatch` rather than the test's: a test that calls
    `monkeypatch.undo()` to drop a seam it installed undoes *every* patch on the
    shared instance, and an offline guarantee must not be switchable off
    half-way through. The two reference lookups and the jpdb dictionary resolve
    `transport or urllib_transport` at call time, so rebinding that module
    attribute stops a request before anything is sent.
    """

    def refuse_model(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("the Anthropic API was reached")

    def refuse_http(url: str, timeout: float = 0.0) -> bytes:
        raise AssertionError(f"this test reached a live service at {url}")

    patch = pytest.MonkeyPatch()
    patch.setattr(claude_client, "prepare_paid_client", refuse_model)
    patch.setattr(claude_client, "parse_call", refuse_model)
    patch.setattr(jpdb, "urllib_transport", refuse_http)
    patch.setattr(kanji, "urllib_transport", refuse_http)
    patch.setattr(jpdb_kanji, "urllib_transport", refuse_http)
    try:
        yield
    finally:
        patch.undo()


# --- the scratch repository ---------------------------------------------------


def _project(tmp_path: Path) -> ProjectConfig:
    """A scratch repository with the checkout's real templates in it.

    `jpdb_html_cache` is named explicitly at a scratch path beside this test's
    own root: the default is the developer's own cache, and a lookup that read
    or published it would make these results depend on earlier real work.
    """

    seed_prompts(tmp_path)
    cache = tmp_path.parent / f"{tmp_path.name}-jpdb-cache"
    cache.mkdir(parents=True, exist_ok=True)
    (tmp_path / "janki.toml").write_text(
        "[paths]\n"
        f'jpdb_html_cache = "{cache}"\n'
        'normalized_file = "vocabulary.json"\n'
        'deck_dir = "decks"\n'
        'media_dir = "media"\n'
        'template_dir = "templates/japanese-study"\n'
        'dist_dir = "dist"\n'
        'staging_dir = "staging"\n'
        'scan_inbox = "inbox"\n'
        'patterns_file = "patterns.json"\n'
        "[ai]\n"
        'extract_provider = "claude-code"\n'
        'extract_model = "claude-opus-5"\n',
        encoding="utf-8",
    )
    shutil.copytree(PROJECT_ROOT / "templates", tmp_path / "templates")
    (tmp_path / "decks").mkdir()
    (tmp_path / "media").mkdir()
    (tmp_path / "vocabulary.json").write_text("[]", encoding="utf-8")
    return ProjectConfig.load(tmp_path)


def _deck(config: ProjectConfig, stem: str = "lesson") -> Path:
    path = config.deck_dir / f"{stem}.yaml"
    path.write_text(
        "deck:\n"
        f"  name: {stem.title()} deck\n"
        "  deck_id: 1500000007\n"
        "  source: ../vocabulary.json\n"
        "  include_tags: [lesson-intake]\n"
        "  intake_tag: lesson-intake\n"
        "  cards:\n"
        "    recognition: true\n"
        "    production: true\n",
        encoding="utf-8",
    )
    return path


def _source(config: ProjectConfig, name: str) -> Path:
    path = config.scan_inbox / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.7 " + name.encode("ascii"))
    return path


def _answer(scenario: str) -> dict[str, Any]:
    return json.loads((RESPONSES / f"{scenario}.json").read_text(encoding="utf-8"))


def _proposed_content(scenario: str) -> tuple[str, ...]:
    """The card content one extraction proposed: sentences and their glosses.

    Deliberately *not* the record identities. An identity such as
    `word:走る:はしる` is `identifiers.py`'s structural key and is what every
    store in this repository is already keyed by; what must never leave the
    owner's local page is the content those keys point at.
    """

    found: list[str] = []
    for entry in _answer(scenario).get("candidates") or ():
        found.extend(entry.get("meanings") or ())
        found.extend(
            example["japanese"] for example in entry.get("examples") or ()
        )
    return tuple(found)


def _stream(scenario: str) -> bytes:
    """One faked subscription reply carrying that scenario's answer."""

    fixture = Path(__file__).parent / "fixtures" / "claude-code-stream.ndjson"
    lines = []
    for line in fixture.read_bytes().splitlines(keepends=True):
        payload = json.loads(line)
        if payload.get("type") == "result":
            payload["structured_output"] = _answer(scenario)
            line = json.dumps(payload).encode("utf-8") + b"\n"
        lines.append(line)
    return b"".join(lines)


class _ScriptedProvider:
    """One faked subscription whose reply depends on which call this is.

    The batches here run at concurrency one, so call order is the plan's own
    child order and a scripted list is exact rather than racy.
    """

    def __init__(self, replies: list[bytes]) -> None:
        self.runner = FakeClaudeRunner(reply=b"")
        self.replies = list(replies)
        self.spawned = 0

    def __call__(self, command: list[str], **kwargs: Any) -> Any:
        return self.runner(command, **kwargs)

    def spawn(self, command: list[str], **kwargs: Any) -> Any:
        self.runner.reply = self.replies[min(self.spawned, len(self.replies) - 1)]
        self.spawned += 1
        return self.runner.spawn(command, **kwargs)


def _batch(
    config: ProjectConfig,
    job_id: str,
    sources: list[Path],
    replies: list[bytes],
    *,
    force: bool = False,
) -> Any:
    provider = _ScriptedProvider(replies)
    plan = extraction_batch.plan_extraction_batch(
        config,
        sources,
        job_id=job_id,
        destination_deck=_deck(config),
        concurrency_limit=1,
        force=force,
        provider_env={},
        provider_runner=provider,
        provider_which=_which,
    )
    study_job.dispatch_job_batch(
        config,
        job_id,
        plan,
        provider_env={},
        provider_runner=provider,
        provider_which=_which,
        provider_spawn=provider.spawn,
    )
    return plan


def _job(config: ProjectConfig) -> str:
    deck = _deck(config)
    job = study_job.open_study_job(
        config,
        kind="source_extraction",
        parent_source=_source(config, "parent.pdf"),
        deck_path=deck,
    )
    return job.header.job_id


def _settled_job(
    config: ProjectConfig,
    *,
    names: tuple[str, ...] = ("one.pdf",),
    replies: list[bytes] | None = None,
) -> str:
    """One study job whose extraction really ran, through the real services."""

    job_id = _job(config)
    _batch(
        config,
        job_id,
        [_source(config, name) for name in names],
        replies if replies is not None else [_stream(SCENARIO)],
    )
    return job_id


def _digest_tree(config: ProjectConfig) -> dict[str, str]:
    found: dict[str, str] = {}
    for path in sorted(config.root.rglob("*")):
        if path.is_file():
            found[str(path.relative_to(config.root))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return found


def _revision(config: ProjectConfig, job_id: str) -> str:
    return study_job.load_study_job(config, job_id).revision


def _stored(config: ProjectConfig, job_id: str, key: str) -> dict[str, Any]:
    held = study_job.load_study_job(config, job_id).choices.get(key)
    return dict(held) if isinstance(held, dict) else {}


def _rows(path: Path) -> tuple[str, ...]:
    records, _meta = staging.read_staging_text(
        path.read_text(encoding="utf-8"), source=str(path)
    )
    return tuple(record.id for record in records)


def _staging_path(config: ProjectConfig, job_id: str, part: str = "one.pdf") -> Path:
    """That part's current staging document, from the owning frontier."""

    return next(
        attempt.staging_path
        for attempt in study_job.job_effective_frontier(config, job_id).effective
        if attempt.ref.source_name == part
    )


def _panel(config: ProjectConfig, path: Path) -> review.ReviewPanel:
    """The owning reader over one part, opened exactly as the service opens it."""

    return review.ReviewPanel.open(
        path,
        staging_dir=config.staging_dir,
        patterns_path=config.patterns_file,
        collection_name=config.normalized_file.name,
    )


# --- the read -----------------------------------------------------------------


def test_a_read_is_a_snapshot_of_the_owning_derivations_and_writes_nothing(
    tmp_path: Path,
) -> None:
    """One read: the exact revision, the real page, and each part's identities.

    Every value is the owning derivation's own — the job document's revision,
    `render_job_preview`'s fingerprint, the frontier's part name and staging
    document, and `ReviewPanel`'s structural row ids. Nothing is recomputed
    here, and nothing is written.
    """

    config = _project(tmp_path)
    job_id = _settled_job(config)
    before = _digest_tree(config)

    decisions = study_choices.read_job_decisions(config, job_id)

    assert decisions.job_id == job_id
    assert decisions.revision == _revision(config, job_id)
    assert (
        decisions.rendering_fingerprint
        == study_job.render_job_preview(config, job_id).rendering_fingerprint
    )
    assert decisions.rendering_fingerprint == decisions.preview.preview.sha256
    # The real interactive page, carried rather than re-rendered: the owner's
    # local view of their own cards.
    assert decisions.preview.preview.html
    frontier = study_job.job_effective_frontier(config, job_id)
    assert [part.part for part in decisions.parts] == [
        attempt.ref.source_name for attempt in frontier.effective
    ]
    (part,) = decisions.parts
    (attempt,) = frontier.effective
    assert part.ref == attempt.ref
    assert part.staging_path == attempt.staging_path
    assert part.staging_sha256 == hashlib.sha256(
        part.staging_path.read_bytes()
    ).hexdigest()
    assert part.record_ids == _rows(part.staging_path)
    assert part.record_ids
    # Freshly extracted: every row still awaits its first approval, so the rows
    # a review may flag are all of them.
    assert part.reviewable_record_ids == part.record_ids
    assert part.pattern_selectable is True
    assert part.pattern_warning is None
    assert dict(part.choices) == {}
    assert decisions.pending == ()
    # Nothing saved yet, so the effective audio choice is the documented
    # default rather than an implicit false.
    assert decisions.include_example_audio is True
    assert decisions.audio_choice_saved is False
    assert _digest_tree(config) == before


def test_the_part_projection_carries_identities_and_no_proposed_content(
    tmp_path: Path,
) -> None:
    """Row content is the owner's local page, never this service's projection.

    The preview really does hold the extraction's sentences — that is the point
    of a local owner view — and the per-part projection a surface or a model
    context would read holds none of them. The identities it does carry are the
    staging document's own, read structurally and minted by nothing here.
    """

    config = _project(tmp_path)
    job_id = _settled_job(config)

    decisions = study_choices.read_job_decisions(config, job_id)

    (part,) = decisions.parts
    assert part.record_ids == _rows(part.staging_path)
    projected = repr(decisions.parts)
    for content in _proposed_content(SCENARIO):
        assert content not in projected
    # The page is the owner's local view of these exact rows drawn as real
    # cards; the projection beside it is identities and digests.
    assert {card.record_id for card in decisions.preview.preview.cards} == set(
        part.record_ids
    )
    assert decisions.preview.preview.html


def test_a_part_still_in_flight_is_named_pending_rather_than_reviewable(
    tmp_path: Path,
) -> None:
    """Partial review works; finish readiness stays with the finish.

    A part whose one effective attempt is not `.complete` cannot be decided
    over — there is no settled document to bind a decision to — so it is named
    with its own state instead of being quietly absent or silently reviewable.
    """

    config = _project(tmp_path)
    job_id = _settled_job(
        config,
        names=("one.pdf", "two.pdf"),
        replies=[_stream(SCENARIO), UNUSABLE_ANSWER],
    )

    decisions = study_choices.read_job_decisions(config, job_id)

    assert [part.part for part in decisions.parts] == ["one.pdf"]
    assert [attempt.ref.source_name for attempt in decisions.pending] == ["two.pdf"]
    (pending,) = decisions.pending
    assert pending.complete is False
    assert pending.state != "committed"


def _drop_pattern_entry(config: ProjectConfig, part: str) -> None:
    """Fixture mutation: a committed child whose grammar half never landed.

    `bookkeeping_complete` is the pattern store holding an entry for this source
    name, so removing it reproduces the real state a child leaves behind when
    its staging write committed and its separate pattern write did not.
    """

    store = json.loads(config.patterns_file.read_text(encoding="utf-8"))
    del store[part]
    config.patterns_file.write_text(
        json.dumps(store, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def test_a_committed_part_whose_grammar_half_is_missing_is_not_reviewable(
    tmp_path: Path,
) -> None:
    """`complete` is committed *and* bookkeeping complete, and a decision needs both.

    Settled is not enough: a part whose pattern write never finished would have
    its review selection and pattern mark bound to half a part, and the finish
    would then be asked to promote from it.
    """

    config = _project(tmp_path)
    job_id = _settled_job(config)
    _drop_pattern_entry(config, "one.pdf")

    decisions = study_choices.read_job_decisions(config, job_id)

    assert decisions.parts == ()
    (pending,) = decisions.pending
    assert pending.ref.source_name == "one.pdf"
    assert pending.state == "committed"
    assert pending.settled is True
    assert pending.complete is False
    with pytest.raises(JankiError):
        study_choices.save_coverage_reason(
            config,
            job_id,
            "one.pdf",
            expected_revision=decisions.revision,
            rendering_fingerprint=decisions.rendering_fingerprint,
            reason="I checked this page against the source myself.",
        )


def test_two_effective_attempts_at_one_part_refuse_rather_than_taking_the_newer(
    tmp_path: Path,
) -> None:
    """Which of two current attempts a decision binds is not settled by order.

    A forced second batch over the same source plans no retirement, so the job
    really holds two effective attempts at one part, both complete and both
    writing the same staging document. Choosing the newer one would bind an
    owner decision to an attempt they never chose.
    """

    config = _project(tmp_path)
    job_id = _settled_job(config)
    first = study_job.job_effective_frontier(config, job_id).effective[0]
    second = _batch(
        config,
        job_id,
        [config.scan_inbox / "one.pdf"],
        [_stream(SECOND_SCENARIO)],
        force=True,
    )
    frontier = study_job.job_effective_frontier(config, job_id)
    assert [attempt.ref.source_name for attempt in frontier.effective] == [
        "one.pdf",
        "one.pdf",
    ]

    with pytest.raises(JankiError) as caught:
        study_choices.read_job_decisions(config, job_id)

    assert "one.pdf" in str(caught.value)
    assert first.ref.batch_id in str(caught.value)
    assert second.batch_id in str(caught.value)

    # The same refusal on the way in to a save, before anything is written.
    with pytest.raises(JankiError):
        study_choices.save_coverage_reason(
            config,
            job_id,
            "one.pdf",
            expected_revision=_revision(config, job_id),
            rendering_fingerprint="f" * 64,
            reason="I checked this page myself.",
        )
    assert _stored(config, job_id, "coverage_reasons") == {}


def test_a_job_the_owning_preview_refuses_surfaces_that_refusal(
    tmp_path: Path,
) -> None:
    """An honest refusal from the renderer, not an invented empty page."""

    config = _project(tmp_path)
    job_id = _settled_job(config, replies=[UNUSABLE_ANSWER])

    with pytest.raises(JankiError) as owning:
        study_job.render_job_preview(config, job_id)
    with pytest.raises(JankiError) as surfaced:
        study_choices.read_job_decisions(config, job_id)

    assert str(owning.value) in str(surfaced.value)


# --- current and stale display ------------------------------------------------


def _save_review(
    config: ProjectConfig,
    job_id: str,
    part: str,
    *,
    record_ids: list[str] | None = None,
    patterns: bool | None = True,
) -> Any:
    decisions = study_choices.read_job_decisions(config, job_id)
    held = next(item for item in decisions.parts if item.part == part)
    return study_choices.save_review(
        config,
        job_id,
        part,
        expected_revision=decisions.revision,
        rendering_fingerprint=decisions.rendering_fingerprint,
        record_ids=list(held.record_ids[:1]) if record_ids is None else record_ids,
        patterns=patterns,
    )


def test_a_saved_choice_reads_current_until_its_staging_document_moves(
    tmp_path: Path,
) -> None:
    """Staleness is disclosed by the read, not discovered at the finish.

    The trailing comment below is a fixture mutation of the scratch
    repository's own staging file: it changes the document's bytes without
    changing a card field, which is exactly the binding that must go stale
    while the rendering the decision was taken over does not.
    """

    config = _project(tmp_path)
    job_id = _settled_job(config)
    _save_review(config, job_id, "one.pdf")

    current = study_choices.read_job_decisions(config, job_id)
    (part,) = current.parts
    assert part.choices["review_flags"].stale is False
    assert part.choices["review_patterns"].stale is False

    with part.staging_path.open("a", encoding="utf-8") as stream:
        stream.write("# an owner's own note about this page\n")

    moved = study_choices.read_job_decisions(config, job_id)
    (after,) = moved.parts
    assert after.staging_sha256 != part.staging_sha256
    assert after.choices["review_flags"].stale is True
    assert after.choices["review_patterns"].stale is True
    # The decision itself is preserved exactly as it was saved; only its
    # standing changed.
    assert dict(after.choices["review_flags"].entry) == dict(
        part.choices["review_flags"].entry
    )
    # A comment is not a card field, so the page the decision was taken over is
    # unchanged: the two bindings really are independent.
    assert moved.rendering_fingerprint == current.rendering_fingerprint


def test_a_save_refuses_a_revision_that_moved_and_writes_nothing(
    tmp_path: Path,
) -> None:
    """The displayed revision is the compare-and-swap, through `record_choice`."""

    config = _project(tmp_path)
    job_id = _settled_job(config)
    decisions = study_choices.read_job_decisions(config, job_id)
    study_choices.save_audio_preference(
        config, job_id, expected_revision=decisions.revision, include_example_audio=False
    )

    with pytest.raises(JankiError):
        study_choices.save_coverage_reason(
            config,
            job_id,
            "one.pdf",
            expected_revision=decisions.revision,
            rendering_fingerprint=decisions.rendering_fingerprint,
            reason="I checked this page myself.",
        )

    assert _stored(config, job_id, "coverage_reasons") == {}


def test_a_save_never_substitutes_a_fresh_fingerprint_for_the_displayed_one(
    tmp_path: Path,
) -> None:
    """The owner decided over a page; a different page is a fresh decision.

    The refusal names both values. What it must never do is save the decision
    against the fingerprint it just computed, which would record the owner as
    having decided over a rendering they never saw.
    """

    config = _project(tmp_path)
    job_id = _settled_job(config)
    decisions = study_choices.read_job_decisions(config, job_id)
    somewhere_else = "0" * 64

    with pytest.raises(JankiError) as caught:
        study_choices.save_coverage_reason(
            config,
            job_id,
            "one.pdf",
            expected_revision=decisions.revision,
            rendering_fingerprint=somewhere_else,
            reason="I checked this page myself.",
        )

    assert somewhere_else in str(caught.value)
    assert decisions.rendering_fingerprint in str(caught.value)
    assert _stored(config, job_id, "coverage_reasons") == {}


def test_a_save_binds_the_supplied_fingerprint_and_the_current_staging_bytes(
    tmp_path: Path,
) -> None:
    """All four §9.1 bindings, resolved fresh and recorded as the owner saw them."""

    config = _project(tmp_path)
    job_id = _settled_job(config)
    decisions = study_choices.read_job_decisions(config, job_id)
    (part,) = decisions.parts

    study_choices.save_coverage_reason(
        config,
        job_id,
        "one.pdf",
        expected_revision=decisions.revision,
        rendering_fingerprint=decisions.rendering_fingerprint,
        reason="I checked this page against the source myself.",
    )

    entry = _stored(config, job_id, "coverage_reasons")["one.pdf"]
    assert entry["job_id"] == job_id
    assert entry["part"] == "one.pdf"
    assert entry["staging_sha256"] == part.staging_sha256
    assert entry["rendering_fingerprint"] == decisions.rendering_fingerprint


@pytest.mark.parametrize(
    ("saver", "extra"),
    [
        ("save_review", {"record_ids": []}),
        ("save_coverage_reason", {"reason": "I checked this myself."}),
        ("save_disposition", {"action": "exclude", "reason": "Out of scope."}),
    ],
)
def test_every_save_refuses_a_part_this_job_does_not_currently_hold(
    tmp_path: Path, saver: str, extra: dict[str, Any]
) -> None:
    """One part resolution behind all three carriers, and it names the parts."""

    config = _project(tmp_path)
    job_id = _settled_job(config)
    decisions = study_choices.read_job_decisions(config, job_id)

    with pytest.raises(JankiError) as caught:
        getattr(study_choices, saver)(
            config,
            job_id,
            "three.pdf",
            expected_revision=decisions.revision,
            rendering_fingerprint=decisions.rendering_fingerprint,
            **extra,
        )

    assert "three.pdf" in str(caught.value)
    assert "one.pdf" in str(caught.value)


def test_a_selection_naming_a_row_this_part_no_longer_holds_refuses(
    tmp_path: Path,
) -> None:
    """Structural identity, checked against the displayed part's own rows."""

    config = _project(tmp_path)
    job_id = _settled_job(config)
    decisions = study_choices.read_job_decisions(config, job_id)
    (part,) = decisions.parts

    with pytest.raises(JankiError) as caught:
        study_choices.save_review(
            config,
            job_id,
            "one.pdf",
            expected_revision=decisions.revision,
            rendering_fingerprint=decisions.rendering_fingerprint,
            record_ids=[part.record_ids[0], "not-a-row-here"],
            patterns=True,
        )

    assert "not-a-row-here" in str(caught.value)
    assert _stored(config, job_id, "review_flags") == {}


# --- review: rows, and the standalone pattern choice --------------------------


def test_a_review_states_its_rows_even_when_that_is_none_of_them(
    tmp_path: Path,
) -> None:
    """An empty selection is a decision the owner stated, and is stored."""

    config = _project(tmp_path)
    job_id = _settled_job(config)

    _save_review(config, job_id, "one.pdf", record_ids=[])

    assert _stored(config, job_id, "review_flags")["one.pdf"]["record_ids"] == []


def test_a_review_saves_its_rows_and_its_pattern_choice_in_one_compare_and_swap(
    tmp_path: Path,
) -> None:
    """Two independent choices, one write: §9.1's editor saving once."""

    config = _project(tmp_path)
    job_id = _settled_job(config)
    decisions = study_choices.read_job_decisions(config, job_id)
    (part,) = decisions.parts

    saved = study_choices.save_review(
        config,
        job_id,
        "one.pdf",
        expected_revision=decisions.revision,
        rendering_fingerprint=decisions.rendering_fingerprint,
        record_ids=list(part.record_ids[:2]),
        patterns=False,
    )

    assert saved.revision != decisions.revision
    held = study_job.load_study_job(config, job_id)
    assert held.revision == saved.revision
    assert held.choices["review_flags"]["one.pdf"]["record_ids"] == list(
        part.record_ids[:2]
    )
    # `False` is the owner saying "not this part's patterns", stored as stated
    # and in its own standalone key — the review selection holds no copy.
    assert held.choices["review_patterns"]["one.pdf"]["value"] is False
    assert "value" not in held.choices["review_flags"]["one.pdf"]


def test_a_part_carrying_a_pattern_set_refuses_a_review_that_omits_the_choice(
    tmp_path: Path,
) -> None:
    """Never inferred: an omitted pattern decision is not a `false`."""

    config = _project(tmp_path)
    job_id = _settled_job(config)
    decisions = study_choices.read_job_decisions(config, job_id)

    with pytest.raises(JankiError) as caught:
        study_choices.save_review(
            config,
            job_id,
            "one.pdf",
            expected_revision=decisions.revision,
            rendering_fingerprint=decisions.rendering_fingerprint,
            record_ids=[],
        )

    assert "one.pdf" in str(caught.value)
    assert _stored(config, job_id, "review_flags") == {}


#: A canonical UUIDv4 the pattern store accepts and no extraction ever minted.
#: `patterns._review_run_id` refuses anything else, and a store it cannot parse
#: has no entry names at all — which would take the part off the page for a
#: different reason than the one under test.
ANOTHER_RUN_ID = "11111111-1111-4111-8111-111111111111"


def _unselectable_patterns(config: ProjectConfig, part: str) -> None:
    """Fixture mutation: leave the store entry present, and no longer this run's.

    `bookkeeping_complete` is the store holding an entry for this source name,
    so removing the entry would take the part off the page entirely. Changing
    its `review_run_id` keeps the part complete and current while the entry
    stops being one a review of *this* staging run may name — which is the
    state §9.3's "neither is accepted for a part without one" describes.
    """

    store = json.loads(config.patterns_file.read_text(encoding="utf-8"))
    store[part]["review_run_id"] = ANOTHER_RUN_ID
    config.patterns_file.write_text(
        json.dumps(store, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def test_a_part_with_no_selectable_pattern_set_takes_no_pattern_choice(
    tmp_path: Path,
) -> None:
    """The control is absent, so supplying one refuses and omitting it saves."""

    config = _project(tmp_path)
    job_id = _settled_job(config)
    _unselectable_patterns(config, "one.pdf")

    decisions = study_choices.read_job_decisions(config, job_id)
    (part,) = decisions.parts
    assert part.pattern_selectable is False

    with pytest.raises(JankiError):
        study_choices.save_review(
            config,
            job_id,
            "one.pdf",
            expected_revision=decisions.revision,
            rendering_fingerprint=decisions.rendering_fingerprint,
            record_ids=[],
            patterns=True,
        )
    assert _stored(config, job_id, "review_flags") == {}

    study_choices.save_review(
        config,
        job_id,
        "one.pdf",
        expected_revision=decisions.revision,
        rendering_fingerprint=decisions.rendering_fingerprint,
        record_ids=[],
    )
    assert _stored(config, job_id, "review_flags")["one.pdf"]["record_ids"] == []
    assert _stored(config, job_id, "review_patterns") == {}


# --- which rows this page offers for review -----------------------------------


def _approve_in_the_workbench(config: ProjectConfig, path: Path, row: str) -> None:
    """Not a fixture mutation: the real workbench approval of one row.

    The two surfaces are two front doors onto the same staging document, so a
    row approved here is an ordinary state this service has to read rather than
    an invented one. `ReviewPanel.submit` writes the exact example fingerprints.
    """

    _panel(config, path).submit(record_ids=[row], review_patterns=False)


def _owner_edit_removes_the_sentences(
    config: ProjectConfig, path: Path, index: int
) -> None:
    """Not a fixture mutation either: the workbench's own edit write path.

    `server.py` parses a form into this exact submission, applies it with
    `edit.apply_edits`, renders with `staging.render_staging_update` and lands it
    with `review.bound_replace`. Blanking a row's Japanese is what an owner
    deleting a proposed sentence does, and it leaves the row present.
    """

    panel = _panel(config, path)
    record = panel.records[index]
    submission = edit.EditSubmission(
        csrf="",
        staging_snapshot=panel.staging_fingerprint,
        card_fields={},
        example_fields={
            (index, position, "japanese"): ""
            for position, _example in enumerate(record.examples)
        },
    )
    updated = edit.apply_edits(panel.records, submission)
    review.bound_replace(
        panel.staging_path,
        staging.render_staging_update(
            panel.staging_bytes, updated, source=str(panel.staging_path)
        ),
        panel.staging_bytes,
        label="staging file",
    )


def test_a_row_already_approved_in_the_workbench_is_not_offered_for_review(
    tmp_path: Path,
) -> None:
    """The control refuses what its one consumer refuses, before anything lands.

    `ReviewPanel.reviewable_record_ids` is what the finish's `_validate_actions`
    accepts. Storing a row outside it would hold a decision that refuses the
    *whole job* later, at a control the owner is no longer looking at.
    """

    config = _project(tmp_path)
    job_id = _settled_job(config)
    path = _staging_path(config, job_id)
    approved = _rows(path)[0]
    _approve_in_the_workbench(config, path, approved)

    decisions = study_choices.read_job_decisions(config, job_id)
    (part,) = decisions.parts
    assert approved in part.record_ids
    assert approved not in part.reviewable_record_ids
    # The owning reader's own set, and this part's own document order.
    assert set(part.reviewable_record_ids) == set(
        _panel(config, path).reviewable_record_ids
    )
    assert part.reviewable_record_ids == tuple(
        row for row in part.record_ids if row != approved
    )
    before = _digest_tree(config)

    with pytest.raises(JankiError) as caught:
        study_choices.save_review(
            config,
            job_id,
            "one.pdf",
            expected_revision=decisions.revision,
            rendering_fingerprint=decisions.rendering_fingerprint,
            record_ids=[approved],
            patterns=True,
        )

    assert approved in str(caught.value)
    assert _stored(config, job_id, "review_flags") == {}
    assert _stored(config, job_id, "review_patterns") == {}
    assert _digest_tree(config) == before


def test_a_row_whose_sentences_an_owner_edit_removed_is_not_offered_for_review(
    tmp_path: Path,
) -> None:
    """The other ordinary way a held row stops being reviewable.

    Nothing here decides what makes a row reviewable — the panel does, and it
    answers `ineligible` for a row with no Japanese example left to approve.
    """

    config = _project(tmp_path)
    job_id = _settled_job(config)
    path = _staging_path(config, job_id)
    edited = _rows(path)[1]
    _owner_edit_removes_the_sentences(config, path, 1)

    decisions = study_choices.read_job_decisions(config, job_id)
    (part,) = decisions.parts
    assert edited in part.record_ids
    assert edited not in part.reviewable_record_ids
    before = _digest_tree(config)

    with pytest.raises(JankiError) as caught:
        study_choices.save_review(
            config,
            job_id,
            "one.pdf",
            expected_revision=decisions.revision,
            rendering_fingerprint=decisions.rendering_fingerprint,
            record_ids=list(part.record_ids),
            patterns=True,
        )

    assert edited in str(caught.value)
    assert _stored(config, job_id, "review_flags") == {}
    assert _digest_tree(config) == before


def test_a_row_this_page_no_longer_offers_for_review_is_still_a_disposition_target(
    tmp_path: Path,
) -> None:
    """§7.7 disposes the rows a part *holds*, reviewable or not.

    An owner excluding or deferring an already-approved or edited row is an
    ordinary decision, and the finish reads held and excluded ids rather than
    reviewable ones — so narrowing the review selection must not narrow this.
    """

    config = _project(tmp_path)
    job_id = _settled_job(config)
    path = _staging_path(config, job_id)
    rows = _rows(path)
    _approve_in_the_workbench(config, path, rows[0])
    _owner_edit_removes_the_sentences(config, path, 1)

    decisions = study_choices.read_job_decisions(config, job_id)
    (part,) = decisions.parts
    assert part.record_ids == rows
    assert part.reviewable_record_ids == (rows[2],)

    study_choices.save_disposition(
        config,
        job_id,
        "one.pdf",
        expected_revision=decisions.revision,
        rendering_fingerprint=decisions.rendering_fingerprint,
        action="defer",
        reason="These two wait for the next lesson.",
        record_ids=[rows[0], rows[1]],
    )

    assert _stored(config, job_id, "dispositions")["one.pdf"]["record_ids"] == [
        rows[0],
        rows[1],
    ]


def test_a_review_saves_the_rows_this_page_still_offers(tmp_path: Path) -> None:
    """The narrowing refuses what the consumer refuses and nothing besides."""

    config = _project(tmp_path)
    job_id = _settled_job(config)
    path = _staging_path(config, job_id)
    rows = _rows(path)
    _approve_in_the_workbench(config, path, rows[0])

    decisions = study_choices.read_job_decisions(config, job_id)
    (part,) = decisions.parts
    assert part.reviewable_record_ids == (rows[1], rows[2])

    study_choices.save_review(
        config,
        job_id,
        "one.pdf",
        expected_revision=decisions.revision,
        rendering_fingerprint=decisions.rendering_fingerprint,
        record_ids=list(part.reviewable_record_ids),
        patterns=True,
    )

    assert _stored(config, job_id, "review_flags")["one.pdf"]["record_ids"] == [
        rows[1],
        rows[2],
    ]


# --- the owning pattern control, carried rather than re-described -------------


def test_the_owning_pattern_warning_is_carried_word_for_word(tmp_path: Path) -> None:
    """An editor can say *why* the control is absent, in the panel's own words.

    Carried, not composed: the sentence is `ReviewPanel.pattern_warning`, and
    for a part that reaches `parts` the reachable one is the lineage mismatch.
    """

    config = _project(tmp_path)
    job_id = _settled_job(config)
    path = _staging_path(config, job_id)

    (fresh,) = study_choices.read_job_decisions(config, job_id).parts
    assert fresh.pattern_warning is None
    assert fresh.pattern_selectable is True

    _unselectable_patterns(config, "one.pdf")

    (drifted,) = study_choices.read_job_decisions(config, job_id).parts
    panel = _panel(config, path)
    assert panel.pattern_warning is not None
    assert drifted.pattern_warning == panel.pattern_warning
    assert drifted.pattern_selectable is False
    assert (
        "The current pattern-store entry does not exactly match this staging "
        "review run" in drifted.pattern_warning
    )


def test_an_already_marked_pattern_set_stays_selectable_and_takes_an_explicit_bool(
    tmp_path: Path,
) -> None:
    """Selectable is lineage and presence; the mark being set is another question.

    §7.2 answers it differently for one page and for an aggregate, and the
    finish's batch passes `retain_marked_patterns=True`. Gating this service on
    `pattern_reviewable` instead would refuse the owner's own stated decision.
    """

    config = _project(tmp_path)
    job_id = _settled_job(config)
    path = _staging_path(config, job_id)
    _panel(config, path).submit(record_ids=[], review_patterns=True)

    marked = _panel(config, path)
    assert marked.pattern_marked is True
    assert marked.pattern_reviewable is False
    assert marked.pattern_selectable is True

    decisions = study_choices.read_job_decisions(config, job_id)
    (part,) = decisions.parts
    assert part.pattern_selectable is True
    assert part.pattern_warning is None

    study_choices.save_review(
        config,
        job_id,
        "one.pdf",
        expected_revision=decisions.revision,
        rendering_fingerprint=decisions.rendering_fingerprint,
        record_ids=[],
        patterns=True,
    )

    assert _stored(config, job_id, "review_patterns")["one.pdf"]["value"] is True


# --- literal reasons, and whole-part or row dispositions ----------------------


LITERAL_REASON = "  Two lines,\nand the second one.  "


def test_a_coverage_reason_is_stored_as_the_owner_typed_it(tmp_path: Path) -> None:
    """No generated default, and no stripping or normalization of the text."""

    config = _project(tmp_path)
    job_id = _settled_job(config)
    decisions = study_choices.read_job_decisions(config, job_id)

    study_choices.save_coverage_reason(
        config,
        job_id,
        "one.pdf",
        expected_revision=decisions.revision,
        rendering_fingerprint=decisions.rendering_fingerprint,
        reason=LITERAL_REASON,
    )

    assert _stored(config, job_id, "coverage_reasons")["one.pdf"]["reason"] == (
        LITERAL_REASON
    )


def test_a_blank_coverage_reason_refuses_rather_than_being_written(
    tmp_path: Path,
) -> None:
    """janki never derives one, and a whitespace-only reason is not one."""

    config = _project(tmp_path)
    job_id = _settled_job(config)
    decisions = study_choices.read_job_decisions(config, job_id)

    with pytest.raises(JankiError):
        study_choices.save_coverage_reason(
            config,
            job_id,
            "one.pdf",
            expected_revision=decisions.revision,
            rendering_fingerprint=decisions.rendering_fingerprint,
            reason="   \n ",
        )

    assert _stored(config, job_id, "coverage_reasons") == {}


@pytest.mark.parametrize("action", ["exclude", "defer"])
def test_a_disposition_with_no_rows_named_is_the_whole_part(
    tmp_path: Path, action: str
) -> None:
    """§7.7's documented whole-part spelling, for both of the two actions."""

    config = _project(tmp_path)
    job_id = _settled_job(config)
    decisions = study_choices.read_job_decisions(config, job_id)

    study_choices.save_disposition(
        config,
        job_id,
        "one.pdf",
        expected_revision=decisions.revision,
        rendering_fingerprint=decisions.rendering_fingerprint,
        action=action,
        reason=LITERAL_REASON,
    )

    entry = _stored(config, job_id, "dispositions")["one.pdf"]
    assert entry["action"] == action
    assert entry["record_ids"] == []
    assert entry["reason"] == LITERAL_REASON


def test_a_disposition_may_name_exactly_the_rows_it_holds_back(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)
    job_id = _settled_job(config)
    decisions = study_choices.read_job_decisions(config, job_id)
    (part,) = decisions.parts

    study_choices.save_disposition(
        config,
        job_id,
        "one.pdf",
        expected_revision=decisions.revision,
        rendering_fingerprint=decisions.rendering_fingerprint,
        action="defer",
        reason="This row needs the next lesson first.",
        record_ids=[part.record_ids[0]],
    )

    entry = _stored(config, job_id, "dispositions")["one.pdf"]
    assert entry["record_ids"] == [part.record_ids[0]]


@pytest.mark.parametrize(
    "records", [["", "x"], "a-record-id", [None]], ids=["blank", "string", "null"]
)
def test_an_unreadable_row_selection_refuses_instead_of_widening_to_the_part(
    tmp_path: Path, records: Any
) -> None:
    """Widening one held row's exclusion to the page is not a repair."""

    config = _project(tmp_path)
    job_id = _settled_job(config)
    decisions = study_choices.read_job_decisions(config, job_id)

    with pytest.raises(JankiError):
        study_choices.save_disposition(
            config,
            job_id,
            "one.pdf",
            expected_revision=decisions.revision,
            rendering_fingerprint=decisions.rendering_fingerprint,
            action="exclude",
            reason="Out of this job's scope.",
            record_ids=records,
        )

    assert _stored(config, job_id, "dispositions") == {}


def test_a_disposition_action_outside_the_two_refuses(tmp_path: Path) -> None:
    config = _project(tmp_path)
    job_id = _settled_job(config)
    decisions = study_choices.read_job_decisions(config, job_id)

    with pytest.raises(JankiError):
        study_choices.save_disposition(
            config,
            job_id,
            "one.pdf",
            expected_revision=decisions.revision,
            rendering_fingerprint=decisions.rendering_fingerprint,
            action="promote",
            reason="Out of this job's scope.",
        )

    assert _stored(config, job_id, "dispositions") == {}


# --- withdrawal ---------------------------------------------------------------


def test_a_withdrawal_drops_only_the_choice_it_names(tmp_path: Path) -> None:
    """The same control and the same compare-and-swap, with no delete pathway."""

    config = _project(tmp_path)
    job_id = _settled_job(config)
    _save_review(config, job_id, "one.pdf")
    decisions = study_choices.read_job_decisions(config, job_id)
    study_choices.save_coverage_reason(
        config,
        job_id,
        "one.pdf",
        expected_revision=decisions.revision,
        rendering_fingerprint=decisions.rendering_fingerprint,
        reason=LITERAL_REASON,
    )
    before_flags = _stored(config, job_id, "review_flags")
    before_reasons = _stored(config, job_id, "coverage_reasons")

    study_choices.withdraw_choice(
        config,
        job_id,
        "one.pdf",
        "review_patterns",
        expected_revision=_revision(config, job_id),
    )

    assert _stored(config, job_id, "review_patterns") == {}
    assert _stored(config, job_id, "review_flags") == before_flags
    assert _stored(config, job_id, "coverage_reasons") == before_reasons


def test_withdrawing_a_choice_this_job_does_not_hold_refuses(tmp_path: Path) -> None:
    config = _project(tmp_path)
    job_id = _settled_job(config)

    with pytest.raises(JankiError):
        study_choices.withdraw_choice(
            config,
            job_id,
            "one.pdf",
            "dispositions",
            expected_revision=_revision(config, job_id),
        )


def test_withdrawal_covers_the_four_per_part_keys_and_nothing_else(
    tmp_path: Path,
) -> None:
    """The job-wide audio preference is not a per-part choice to withdraw."""

    config = _project(tmp_path)
    job_id = _settled_job(config)

    assert study_choices.PART_CHOICE_KEYS == (
        "review_flags",
        "review_patterns",
        "coverage_reasons",
        "dispositions",
    )
    with pytest.raises(JankiError) as caught:
        study_choices.withdraw_choice(
            config,
            job_id,
            "one.pdf",
            "include_example_audio",
            expected_revision=_revision(config, job_id),
        )

    assert "include_example_audio" in str(caught.value)


# --- independence, and the job-wide audio preference --------------------------


def test_saving_the_audio_preference_never_stales_a_saved_content_decision(
    tmp_path: Path,
) -> None:
    """A job's CAS revision advancing is not a change to what was decided.

    The provenance fingerprint excludes the choice maps and the job revision by
    the renderer's own contract, so an independent save cannot invalidate an
    earlier one.
    """

    config = _project(tmp_path)
    job_id = _settled_job(config)
    _save_review(config, job_id, "one.pdf")
    before = study_choices.read_job_decisions(config, job_id)

    study_choices.save_audio_preference(
        config,
        job_id,
        expected_revision=before.revision,
        include_example_audio=False,
    )

    after = study_choices.read_job_decisions(config, job_id)
    assert after.revision != before.revision
    assert after.rendering_fingerprint == before.rendering_fingerprint
    (part,) = after.parts
    assert part.choices["review_flags"].stale is False
    assert part.choices["review_patterns"].stale is False


def test_the_audio_preference_binds_the_job_and_its_revision_and_nothing_else(
    tmp_path: Path,
) -> None:
    """Job-wide: no part, no staging hash, no rendering, no written reason."""

    config = _project(tmp_path)
    job_id = _settled_job(config)
    before = _digest_tree(config)

    study_choices.save_audio_preference(
        config,
        job_id,
        expected_revision=_revision(config, job_id),
        include_example_audio=False,
    )

    entry = study_job.load_study_job(config, job_id).choices["include_example_audio"]
    assert entry["value"] is False
    assert entry["job_id"] == job_id
    assert set(entry) == {"value", "job_id", "saved_at"}
    # One document changed, and it is the job's own: no staging review mark, no
    # coverage verdict, no canonical byte and no provider request.
    after = _digest_tree(config)
    changed = {
        name for name in {*before, *after} if before.get(name) != after.get(name)
    }
    assert changed == {
        str(study_job.study_job_path(config, job_id).relative_to(config.root))
    }


def test_the_audio_preference_defaults_to_included_and_true_re_enables_it(
    tmp_path: Path,
) -> None:
    """A missing choice includes sentence audio; it is never an implicit false."""

    config = _project(tmp_path)
    job_id = _settled_job(config)

    unset = study_choices.read_job_decisions(config, job_id)
    assert unset.include_example_audio is True
    assert unset.audio_choice_saved is False

    study_choices.save_audio_preference(
        config, job_id, expected_revision=unset.revision, include_example_audio=False
    )
    opted_out = study_choices.read_job_decisions(config, job_id)
    assert opted_out.include_example_audio is False
    assert opted_out.audio_choice_saved is True

    study_choices.save_audio_preference(
        config, job_id, expected_revision=opted_out.revision, include_example_audio=True
    )
    restored = study_choices.read_job_decisions(config, job_id)
    assert restored.include_example_audio is True
    assert restored.audio_choice_saved is True


@pytest.mark.parametrize("value", [None, 1, "true"], ids=["none", "int", "text"])
def test_the_audio_preference_is_exactly_true_or_false(
    tmp_path: Path, value: Any
) -> None:
    config = _project(tmp_path)
    job_id = _settled_job(config)

    with pytest.raises(JankiError):
        study_choices.save_audio_preference(
            config,
            job_id,
            expected_revision=_revision(config, job_id),
            include_example_audio=value,
        )

    assert "include_example_audio" not in study_job.load_study_job(
        config, job_id
    ).choices


def test_a_saved_choice_survives_another_parts_save_untouched(
    tmp_path: Path,
) -> None:
    """§9.1's rule, proved across two parts of one job."""

    config = _project(tmp_path)
    job_id = _settled_job(config, names=("one.pdf", "two.pdf"))
    _save_review(config, job_id, "one.pdf", record_ids=[])
    first = _stored(config, job_id, "review_flags")["one.pdf"]

    _save_review(config, job_id, "two.pdf", record_ids=[])

    held = _stored(config, job_id, "review_flags")
    assert held["one.pdf"] == first
    assert set(held) == {"one.pdf", "two.pdf"}
    decisions = study_choices.read_job_decisions(config, job_id)
    assert all(
        part.choices["review_flags"].stale is False for part in decisions.parts
    )


def test_the_read_carries_each_saved_choice_exactly_as_it_is_stored(
    tmp_path: Path,
) -> None:
    """A surface displays what was saved, not a re-derivation of it."""

    config = _project(tmp_path)
    job_id = _settled_job(config)
    _save_review(config, job_id, "one.pdf")

    (part,) = study_choices.read_job_decisions(config, job_id).parts

    stored = study_job.load_study_job(config, job_id).choices
    for key in ("review_flags", "review_patterns"):
        assert part.choices[key].key == key
        assert dict(part.choices[key].entry) == dict(stored[key]["one.pdf"])
    assert set(part.choices) == {"review_flags", "review_patterns"}


def test_a_frontier_attempt_is_read_through_the_owning_derivation(
    tmp_path: Path,
) -> None:
    """Earlier successful parts and a retried part are both still current.

    A retry of one part does not move its sibling: the job is not one batch, so
    a decision on part one stays bound to the first batch's attempt while part
    two's is the retry's.
    """

    config = _project(tmp_path)
    job_id = _settled_job(
        config,
        names=("one.pdf", "two.pdf"),
        replies=[_stream(SCENARIO), UNUSABLE_ANSWER],
    )
    first = study_job.load_study_job(config, job_id)
    batch_id = next(
        intent.reserves["batch_id"]
        for intent in first.intents
        if intent.kind == "extract_batch"
    )
    provider = _ScriptedProvider([_stream(SECOND_SCENARIO)])
    retry = study_job.plan_job_batch_retry(
        config,
        job_id,
        batch_id,
        [2],
        provider_env={},
        provider_runner=provider,
        provider_which=_which,
    )
    study_job.dispatch_job_batch(
        config,
        job_id,
        retry,
        provider_env={},
        provider_runner=provider,
        provider_which=_which,
        provider_spawn=provider.spawn,
    )

    decisions = study_choices.read_job_decisions(config, job_id)

    assert [part.part for part in decisions.parts] == ["one.pdf", "two.pdf"]
    assert decisions.parts[0].ref.batch_id == batch_id
    assert decisions.parts[1].ref.batch_id == retry.batch_id
    assert decisions.pending == ()
    assert decisions.parts[0].staging_path != decisions.parts[1].staging_path


def test_the_read_refuses_a_job_id_this_repository_does_not_hold(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)
    _settled_job(config)

    with pytest.raises(JankiError):
        study_choices.read_job_decisions(config, "not-a-job")
