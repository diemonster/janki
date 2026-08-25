"""Which archives may be read as "what a source taught", and which may not.

``data/staging/done/`` looks like one uniform record of everything janki has
read, and it is not. Most archives are a model reading a document, and their
rows are the best surviving statement of what that page taught. An
`enrich --ai` pass is not: it answers about words the collection already held,
so its rows are model output about janki's own data with no page behind them.

That distinction has already cost something. When `enrich --jpdb` overwrote
taught meanings with dictionary glosses, the repair read the archives to
recover what each source said — the right instinct, and it worked. But
``data/staging/done/ai-enrichment.yaml`` snapshots ninety records *as they
stood while damaged*, and sixty-eight of its rows still carry the bad glosses.
A sweep treating every archive alike would have restored the damage it was
written to undo.

The corpus assertions below are the guard. `is_model_pass` is a rule about
metadata; only running it over the shipped files proves the rule catches the
file it was written for.
"""

from __future__ import annotations

import pytest

from conftest import REPO_ROOT
from japanese_anki.config import ProjectConfig
from japanese_anki.staging import AI_ENRICHMENT_KEY, is_model_pass, read_staging

ARCHIVES = REPO_ROOT / "data" / "staging" / "done"
DANGEROUS = "ai-enrichment.yaml"

needs_corpus = pytest.mark.skipif(
    not ARCHIVES.is_dir(), reason="no corpus in this checkout"
)


def _collection_name() -> str:
    """What this repository actually calls its collection.

    Read from the config rather than written as a literal. The archive's
    ``source_file`` is a frozen historical string, so the rule only fires
    while the live name still matches it — and a rename that quietly disarmed
    the guard must fail this test rather than pass it.
    """
    return ProjectConfig.load(REPO_ROOT).normalized_file.name


# --- the shipped corpus, which is where this actually bites -----------------


@needs_corpus
def test_the_enrichment_snapshot_is_never_read_as_a_source() -> None:
    """The concrete file, by name. It holds sixty-eight rows of the very
    damage the archives were read to repair, and it is the reason this rule
    exists rather than a comment asking people to be careful."""
    _records, meta = read_staging(ARCHIVES / DANGEROUS)

    assert is_model_pass(meta, collection_name=_collection_name()) is True


@needs_corpus
def test_no_source_reading_is_mistaken_for_a_pass() -> None:
    """The guard has to be narrow. Setting aside a real source reading would
    lose the only surviving record of what that page taught — the same damage
    from the other direction.

    Membership rather than a census: promoting another `enrich --ai` run adds
    a digest-suffixed archive beside this one, and a count would then fail on
    correct behaviour.
    """
    collection = _collection_name()
    for path in sorted(ARCHIVES.glob("*.yaml")):
        _records, meta = read_staging(path)
        taught = not is_model_pass(meta, collection_name=collection)
        assert taught == (path.name != DANGEROUS), path.name


@needs_corpus
def test_every_set_aside_archive_says_why_it_was_set_aside() -> None:
    """A property rather than a name list, so this keeps holding as archives
    arrive. Whatever is excluded must carry one of the two signals — not be
    excluded because a filename looked wrong."""
    collection = _collection_name()
    for path in sorted(ARCHIVES.glob("*.yaml")):
        _records, meta = read_staging(path)
        if not is_model_pass(meta, collection_name=collection):
            continue
        assert AI_ENRICHMENT_KEY in meta or (
            str(meta.get("source_file") or "").strip() == collection
            and meta.get("model")
        ), path.name


# --- the two signals, because the marker arrived after the files ------------


def test_a_modern_enrichment_review_is_recognised_by_its_block() -> None:
    """What `enrich --ai` writes today. Presence, not truthiness: a pass that
    recorded an empty block is still a pass."""
    assert is_model_pass({AI_ENRICHMENT_KEY: {"fields": {}}}) is True
    assert is_model_pass({AI_ENRICHMENT_KEY: {}}) is True


def test_an_older_enrichment_review_is_recognised_by_its_source() -> None:
    """What it wrote before the block existed — the shape of the file actually
    sitting in this repository. Naming the collection as your source is what
    it means to be a pass over what janki already had."""
    meta = {"source_file": "vocabulary.json", "model": "claude-opus-5"}

    assert is_model_pass(meta, collection_name="vocabulary.json") is True


def test_a_review_written_by_hand_against_the_collection_is_not_a_model_pass() -> None:
    """The narrowing that keeps the rule honest. A person correcting words the
    collection already holds names it the same way — and calling that a paid
    model pass would refuse re-identification, the repair its rows are most
    likely to need, with a sentence untrue of it. No importer writes `model`
    and no hand edit does either."""
    meta = {"source_file": "vocabulary.json"}

    assert is_model_pass(meta, collection_name="vocabulary.json") is False


def test_a_project_that_calls_its_collection_something_else_is_not_special_cased() -> None:
    """The rule reads the configured name. Hardcoding `vocabulary.json` would
    be right about this repository and silently wrong about the next one."""
    meta = {"source_file": "words.json", "model": "claude-opus-5"}

    assert is_model_pass(meta, collection_name="words.json") is True
    assert is_model_pass(meta, collection_name="vocabulary.json") is False


def test_a_source_reading_is_not_mistaken_for_a_pass() -> None:
    meta = {"source_file": "lesson.pdf", "model": "claude-opus-5"}

    assert is_model_pass(meta, collection_name="vocabulary.json") is False


def test_an_import_is_evidence_about_the_pack_it_came_from() -> None:
    """An Anki pack is a source somebody read, even with no model behind it."""
    meta = {"source_file": "Yotsubato Volume 1 Reading Pack Vocab"}

    assert is_model_pass(meta, collection_name="vocabulary.json") is False


def test_a_caller_with_no_collection_name_gets_no_guess() -> None:
    """Only the modern signal, never a guess about the older one — and blank
    metadata must not match a blank collection name."""
    assert is_model_pass({"source_file": "vocabulary.json", "model": "m"}) is False
    assert is_model_pass({}) is False
    # The one that needs the explicit `collection_name` guard rather than the
    # `model` narrowing: a staging file may carry no source at all, and a
    # blank name must not match a caller that could not supply one.
    assert is_model_pass({"source_file": "", "model": "m"}, collection_name="") is False
    assert is_model_pass({"model": "m"}) is False
