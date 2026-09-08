"""One batch's saved proposals, drawn as the cards a deck would really hold.

These are real renders: Anki's own engine, the repository's own templates, and
each child's own staging document exactly as that child wrote it. There is no
combined staging file and no combined metadata anywhere in here — a batch is
several independent extractions that a person reviews together, and inventing
one merged document would lose which source proposed what.

The provider is faked, as everywhere else. What is not faked is the projection:
the scoped identities and intake tags a standalone destination would produce
come from the assignment service, because those rules belong to it.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from test_revision_provider import FakeClaudeRunner, _which
from test_workbench_fixtures import RESPONSES

from conftest import seed_prompts
from japanese_anki import card_preview, claude_client, staging
from japanese_anki.application import assignment, extraction_batch
from japanese_anki.application.extraction_batch import (
    dispatch_extraction_batch,
    plan_extraction_batch,
    render_extraction_batch_preview,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.identifiers import record_scope_id
from japanese_anki.io import load_records, save_records_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENARIO = "table_exhaustive"
SCOPE = "a1b2c3d4"

pytestmark = pytest.mark.skipif(
    card_preview.preview_unavailable() is not None,
    reason=str(card_preview.preview_unavailable()),
)


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


def _no_api(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the Anthropic API was reached")

    monkeypatch.setattr(claude_client, "prepare_paid_client", refuse)
    monkeypatch.setattr(claude_client, "parse_call", refuse)


def _project(tmp_path: Path) -> ProjectConfig:
    """A scratch repository with the repository's real templates in it."""
    seed_prompts(tmp_path)
    (tmp_path / "janki.toml").write_text(
        "[paths]\n"
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


def _deck(config: ProjectConfig, *, scope_id: str = "", name: str = "lesson") -> Path:
    path = config.deck_dir / f"{name}.yaml"
    lines = [
        "deck:",
        f"  name: {name.title()} deck",
        "  source: ../vocabulary.json",
        "  include_tags: [lesson-intake]",
        "  intake_tag: lesson-intake",
        "  cards:",
        "    recognition: true",
        "    production: true",
    ]
    if scope_id:
        lines.append(f"  scope_id: {scope_id}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _source(config: ProjectConfig, name: str, body: bytes) -> Path:
    path = config.scan_inbox / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.7 " + body)
    return path


def _committed_batch(
    config: ProjectConfig,
    *,
    names: tuple[str, ...] = ("one.pdf",),
    destination_deck: Path | None = None,
) -> Any:
    """Two real dispatches through the faked subscription, and their staging."""
    runner = FakeClaudeRunner(reply=_stream(_answer()))
    sources = [
        _source(config, name, name.encode("ascii")) for name in names
    ]
    plan = plan_extraction_batch(
        config,
        sources,
        destination_deck=destination_deck,
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
    )
    outcome = dispatch_extraction_batch(
        config,
        plan,
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
        provider_spawn=runner.spawn,
    )
    assert outcome.committed_count == len(names)
    return plan


def _staged(path: Path) -> list[Any]:
    text = path.read_text(encoding="utf-8")
    records, _meta = staging.read_staging_text(text, source=str(path))
    return records


def _digest_tree(config: ProjectConfig) -> dict[str, str]:
    """Every file under the project, by exact bytes."""
    found: dict[str, str] = {}
    for path in sorted(config.root.rglob("*")):
        if path.is_file():
            found[str(path.relative_to(config.root))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return found


def test_a_standalone_destination_shows_the_identities_it_would_really_take(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scoped ids and intake tags come from assignment, not from here.

    A standalone deck holds its own copies under its own scope. Which id that
    is, and which tag marks intake, are the assignment service's rules; a
    preview that re-derived them would be a second definition and would drift.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    deck = _deck(config, scope_id=SCOPE)
    plan = _committed_batch(config, destination_deck=deck)

    staged = _staged(plan.children[0].staging_path)
    expected = [
        attempts[0].plan.assigned_record
        for attempts in assignment.plan_deck_assignments(
            config, staged, destination_stems=[deck.stem]
        )
    ]

    review = render_extraction_batch_preview(config, plan.batch_id)

    drawn = {card.record_id for card in review.preview.cards}
    assert drawn == {record.id for record in expected}
    # Scoped, and not the shared identity the staging file holds.
    assert all(record_scope_id(record.id) == SCOPE for record in expected)
    assert drawn.isdisjoint({record.id for record in staged})
    assert all("lesson-intake" in record.tags for record in expected)
    assert review.preview.new_note_count == len(expected)
    assert review.child_indices == (1,)
    assert review.conflicts == ()


def test_a_shared_destination_draws_the_proposed_fields_not_the_canonical_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A proposal is what is being reviewed, so it is what must be drawn."""
    _no_api(monkeypatch)
    config = _project(tmp_path)
    deck = _deck(config)
    plan = _committed_batch(config, destination_deck=deck)

    staged = _staged(plan.children[0].staging_path)
    # The collection already holds this identity, with older content.
    save_records_json(
        config.normalized_file,
        [replace(staged[0], meanings=["STALE-CANONICAL-MEANING"], tags=["lesson-intake"])],
    )

    review = render_extraction_batch_preview(config, plan.batch_id)

    assert b"STALE-CANONICAL-MEANING" not in review.preview.html
    assert staged[0].meanings[0].encode("utf-8") in review.preview.html
    # The unrelated canonical rows are untouched by the overlay.
    assert [record.id for record in load_records(config.normalized_file)] == [
        staged[0].id
    ]


def test_two_children_proposing_one_identity_disclose_the_exact_difference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two sources, one identity, two different answers: say so, choose neither."""
    _no_api(monkeypatch)
    config = _project(tmp_path)
    deck = _deck(config)
    plan = _committed_batch(config, names=("one.pdf", "two.pdf"), destination_deck=deck)

    second = plan.children[1].staging_path
    text = second.read_text(encoding="utf-8")
    records, meta = staging.read_staging_text(text, source=str(second))
    subject = records[0]
    staging.write_staging(
        second,
        [replace(subject, meanings=["PAGE-TWO-GLOSS"]), *records[1:]],
        meta,
        force=True,
    )

    review = render_extraction_batch_preview(config, plan.batch_id)

    assert review.child_indices == (1, 2)
    disclosed = [line for line in review.conflicts if subject.id in line]
    assert disclosed, review.conflicts
    detail = disclosed[0]
    assert isinstance(detail, str)
    assert "meanings" in detail
    assert "PAGE-TWO-GLOSS" in detail
    assert subject.meanings[0] in detail
    assert "one.pdf" in detail and "two.pdf" in detail
    # The card drawn for that identity is labelled as one source's proposal
    # rather than presented as a settled merge of the two.
    assert b"proposed twice" in review.preview.html
    # Each child keeps its own document and its own accounting.
    assert json.loads(
        json.dumps(staging.read_staging_text(
            plan.children[0].staging_path.read_text(encoding="utf-8"),
            source=str(plan.children[0].staging_path),
        )[1]["source_file"])
    ) == "one.pdf"


def test_the_default_review_needs_no_destination_and_creates_no_deck(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reviewing saved successes must not require choosing a destination first."""
    _no_api(monkeypatch)
    config = _project(tmp_path)
    plan = _committed_batch(config)
    before = _digest_tree(config)

    review = render_extraction_batch_preview(config, plan.batch_id)

    staged = _staged(plan.children[0].staging_path)
    assert review.preview.note_count == len(staged)
    assert review.preview.new_note_count == len(staged)
    assert review.preview.cards
    assert list(config.deck_dir.iterdir()) == []
    assert _digest_tree(config) == before


def test_a_combined_review_writes_nothing_anywhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read-only, including the canonical collection it is projected against."""
    _no_api(monkeypatch)
    config = _project(tmp_path)
    deck = _deck(config, scope_id=SCOPE)
    plan = _committed_batch(config, names=("one.pdf", "two.pdf"), destination_deck=deck)
    before = _digest_tree(config)

    render_extraction_batch_preview(config, plan.batch_id)
    render_extraction_batch_preview(config, plan.batch_id, deck_path=deck)

    assert _digest_tree(config) == before


def _batch_plan(
    config: ProjectConfig,
    runner: FakeClaudeRunner,
    names: tuple[str, ...],
    *,
    destination_deck: Path | None = None,
    force: bool = False,
) -> Any:
    sources = [config.scan_inbox / name for name in names]
    return plan_extraction_batch(
        config,
        sources,
        destination_deck=destination_deck,
        force=force,
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
    )


def test_staging_that_belongs_to_another_source_is_never_shown_as_this_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A path is where a document lives, not proof of what wrote it."""
    _no_api(monkeypatch)
    config = _project(tmp_path)
    deck = _deck(config)
    plan = _committed_batch(config, names=("one.pdf", "two.pdf"), destination_deck=deck)
    before = _digest_tree(config)

    # The second child's document, sitting at the first child's path.
    plan.children[0].staging_path.write_bytes(
        plan.children[1].staging_path.read_bytes()
    )

    with pytest.raises(JankiError) as caught:
        render_extraction_batch_preview(config, plan.batch_id)

    assert "one.pdf" in str(caught.value)
    unchanged = _digest_tree(config)
    assert unchanged.keys() == before.keys()
    assert unchanged[str(config.normalized_file.relative_to(config.root))] == before[
        str(config.normalized_file.relative_to(config.root))
    ]


def test_only_journal_proven_successes_are_drawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A leftover review from an earlier run is not this batch's proposal.

    A forced re-read leaves the old staging document in place until the new
    answer replaces it. A child that never sent has proposed nothing, and
    showing yesterday's cards as this batch's work would be a lie a reviewer
    would act on.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    deck = _deck(config)
    runner = FakeClaudeRunner(reply=_stream(_answer()))
    first = _committed_batch(
        config, names=("one.pdf", "two.pdf"), destination_deck=deck
    )

    # Make the older document recognizable, keeping its own metadata exactly.
    stale = first.children[1].staging_path
    records, meta = staging.read_staging_text(
        stale.read_text(encoding="utf-8"), source=str(stale)
    )
    staging.write_staging(
        stale,
        [replace(records[0], meanings=["OLD-REPLACED-GLOSS"]), *records[1:]],
        meta,
        force=True,
    )

    second = _batch_plan(
        config, runner, ("one.pdf", "two.pdf"), destination_deck=deck, force=True
    )
    real_prepare = extraction_batch.prepare_extraction_transport

    def refuse_second(call_plan: Any, target: Any, **kwargs: Any) -> Any:
        if target.name == "two.pdf":
            raise JankiError("temporary login refusal")
        return real_prepare(call_plan, target, **kwargs)

    monkeypatch.setattr(
        extraction_batch, "prepare_extraction_transport", refuse_second
    )
    outcome = dispatch_extraction_batch(
        config,
        second,
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
        provider_spawn=runner.spawn,
    )
    monkeypatch.undo()
    _no_api(monkeypatch)
    assert [child.state for child in outcome.children] == ["committed", "authorized"]
    assert stale.exists()

    review = render_extraction_batch_preview(config, second.batch_id)

    assert review.child_indices == (1,)
    assert review.conflicts == ()
    assert b"OLD-REPLACED-GLOSS" not in review.preview.html
    # The genuine sibling's cards are still drawn.
    assert review.preview.cards
    assert {card.record_id for card in review.preview.cards} == {
        record.id for record in _staged(second.children[0].staging_path)
    }
