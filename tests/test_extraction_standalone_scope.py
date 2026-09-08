"""Known-word suppression when a source is extracted for one standalone deck.

A standalone deck holds its own copies of words, under its own scope. That
makes "which words does janki already have" a different question per
destination, and extraction is where it is asked: the shared collection must
not be told to skip a standalone deck's private copies, and a standalone deck
must not be told to skip the shared collection's words — those are exactly the
words it is being built to copy.

Nothing here reads Japanese. The expressions are ASCII placeholders because
every rule under test is about identity namespaces, prompt inputs, and consent
bindings, none of which look inside the word.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_extract import PDF, FakeCall, candidate, ok

from conftest import seed_prompts
from japanese_anki.application import extraction as extraction_application
from japanese_anki.application.extraction import (
    ExtractionDispatchError,
    ExtractionDispatchExpectation,
    describe_extraction,
    destination_deck_facts,
    dispatch_extraction,
    plan_extraction,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.identifiers import stable_record_id
from japanese_anki.inputs import prepare_inputs
from japanese_anki.models import VocabularyRecord
from japanese_anki.workbench import assistant_adapter
from japanese_anki.workbench.assistant import AssistantDeckChoice, RevisionRefusal

SCOPE = "a" * 64
OTHER_SCOPE = "b" * 64

SHARED_RECORD = VocabularyRecord(
    id="word:shared:sharedreading",
    expression="shared",
    reading="sharedreading",
)
SAME_SCOPE_RECORD = VocabularyRecord(
    id=f"standalone:{SCOPE}:mine:minereading",
    expression="mine",
    reading="minereading",
)
OTHER_SCOPE_RECORD = VocabularyRecord(
    id=f"standalone:{OTHER_SCOPE}:theirs:theirsreading",
    expression="theirs",
    reading="theirsreading",
)
COLLECTION = (SHARED_RECORD, SAME_SCOPE_RECORD, OTHER_SCOPE_RECORD)


def _project(
    tmp_path: Path,
    records: tuple[VocabularyRecord, ...] = (),
) -> Path:
    (tmp_path / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "vocabulary.json"\n'
        'ledger_file = "ledger.json"\n'
        'staging_dir = "staging"\n'
        'scan_inbox = "inbox"\n'
        # These fixtures drive the explicit Anthropic API extraction path:
        # a faked ``parse_call``. The default transport is the owner's
        # subscription, and choosing it here would probe a real login.
        "[ai]\n"
        'extract_provider = "anthropic-api"\n',
        encoding="utf-8",
    )
    seed_prompts(tmp_path)
    (tmp_path / "vocabulary.json").write_text(
        json.dumps([record.to_dict() for record in records], ensure_ascii=False),
        encoding="utf-8",
    )
    inbox = tmp_path / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    (inbox / "lesson.pdf").write_bytes(PDF)
    return tmp_path


def _deck(root: Path, scope_id: str = "", *, kind: str = "") -> Path:
    path = root / "data" / "decks" / "standalone.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "deck:",
        "  name: Standalone Verbs",
        "  deck_id: 1234567890",
        "  include_tags: [standalone-verbs]",
        "  intake_tag: standalone-verbs",
    ]
    if scope_id:
        lines.append(f"  scope_id: {scope_id}")
    if kind:
        lines.append(f"  kind: {kind}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _source(root: Path) -> Path:
    return root / "inbox" / "lesson.pdf"


def _plan(root: Path, **overrides: Any):
    config = ProjectConfig.load(root)
    return plan_extraction(
        config,
        prepare_inputs([_source(root)], root / "inbox"),
        mode="prose",
        model="claude-opus-5",
        style_guide="style",
        system="system",
        **overrides,
    )


def _expectation(
    config: ProjectConfig,
    consent: Any,
    source: Path,
    **overrides: Any,
) -> ExtractionDispatchExpectation:
    target = consent.target
    assert target is not None
    return ExtractionDispatchExpectation(
        provider="anthropic-api",
        source=source,
        model=consent.model,
        mode=consent.mode,
        source_sha256=target.source_sha256,
        request_fingerprint=str(target.provenance["request_fingerprint"]),
        replacement_revision=consent.replacement_revision,
        replacement_confirmed=False,
        staging_path=target.staging_path,
        patterns_path=target.patterns_path,
        operations_path=config.operations_file,
        **overrides,
    )


def _adapter(
    config: ProjectConfig,
    deck_path: Path | None = None,
) -> assistant_adapter.RevisionAssistantAdapter:
    if deck_path is None:
        return assistant_adapter.RevisionAssistantAdapter(
            config=config,
            deck_choices=(),
            _targets=(),
        )
    scope = deck_path.absolute().relative_to(config.root.absolute()).as_posix()
    choice = AssistantDeckChoice(
        deck_id="standalone-deck-id",
        label="Standalone Verbs",
        scope=scope,
        chat_supported=True,
        revision_supported=False,
    )
    return assistant_adapter.RevisionAssistantAdapter(
        config=config,
        deck_choices=(choice,),
        _targets=(
            assistant_adapter._RevisionDeckTarget(
                choice=choice,
                path=deck_path,
                record_ids=None,
            ),
        ),
    )


def test_shared_planning_skips_the_collection_without_standalone_copies(
    tmp_path: Path,
) -> None:
    """The shared default is the behaviour janki has always had, minus the
    words that are not in the shared collection at all.

    A standalone copy lives under its deck's scope. Sending its expression in
    the prose skip list — or counting its identity as already known — would
    suppress a word the shared collection does not have, which is the one
    outcome no reviewer could recover from without re-paying for the source.
    """
    root = _project(tmp_path, COLLECTION)

    plan = _plan(root)

    assert plan.scope_id == ""
    assert plan.skip_list == ("shared",)
    assert SHARED_RECORD.id in plan.known
    assert stable_record_id("shared", "sharedreading") in plan.known
    assert SAME_SCOPE_RECORD.id not in plan.known
    assert OTHER_SCOPE_RECORD.id not in plan.known
    assert stable_record_id("mine", "minereading") not in plan.known
    assert stable_record_id("theirs", "theirsreading") not in plan.known


def test_scoped_planning_keeps_shared_words_and_knows_its_own_copies(
    tmp_path: Path,
) -> None:
    """The mirror image, and the reason the filter runs in both directions.

    A standalone deck is built by copying words the shared collection already
    has, so a shared word must stay extractable. Its *own* copies are the only
    ones it already has — recognized through the ordinary candidate key, because
    extraction mints ordinary ids and a separate assignment step is what puts a
    card under the scope.
    """
    root = _project(tmp_path, COLLECTION)

    plan = _plan(root, scope_id=SCOPE)

    assert plan.scope_id == SCOPE
    assert plan.skip_list == ("mine",)
    # The candidate key an extraction would actually mint, plus the exact
    # stored scoped id. Only the first can ever match a fresh candidate.
    assert stable_record_id("mine", "minereading") in plan.known
    assert SAME_SCOPE_RECORD.id in plan.known
    assert SHARED_RECORD.id not in plan.known
    assert stable_record_id("shared", "sharedreading") not in plan.known
    assert OTHER_SCOPE_RECORD.id not in plan.known
    assert stable_record_id("theirs", "theirsreading") not in plan.known


def test_the_scope_reaches_describe_and_the_paid_dispatch_replan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Consent, the rendered request identity, and the fresh replan are one
    scope or they are nothing.

    The dispatch does not send what consent described; it re-plans and compares
    the request fingerprint. So a dispatch that *succeeds* against a scoped
    expectation is the proof that the replan asked the same scoped question —
    a shared replan here computes the shared skip list and refuses.
    """
    root = _project(tmp_path, COLLECTION)
    config = ProjectConfig.load(root)
    source = _source(root)

    shared = describe_extraction(config, source, mode="prose")
    scoped = describe_extraction(config, source, mode="prose", scope_id=SCOPE)

    assert scoped.scope_id == SCOPE
    assert shared.scope_id == ""
    assert shared.target is not None and scoped.target is not None
    assert (
        scoped.target.provenance["request_fingerprint"]
        != shared.target.provenance["request_fingerprint"]
    )

    fake = FakeCall(ok(candidate()))
    monkeypatch.setattr("japanese_anki.extract.claude_client.parse_call", fake)

    outcome = dispatch_extraction(
        config,
        _expectation(config, scoped, source, scope_id=SCOPE),
        client=object(),
    )

    assert outcome.target == config.staging_dir / "lesson.pdf.yaml"
    assert len(fake.calls) == 1


def test_a_rewritten_destination_deck_refuses_before_any_paid_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The deck bytes are part of the consent, not a display detail.

    Deliberately an empty collection, so every namespace's skip list is empty
    and the scoped request fingerprint equals the shared one. Nothing about the
    prompt changed; the destination did, and that alone has to stop the call
    before authority is written.
    """
    root = _project(tmp_path)
    config = ProjectConfig.load(root)
    source = _source(root)
    deck = _deck(root, SCOPE)

    scoped = describe_extraction(config, source, mode="prose", scope_id=SCOPE)
    shared = describe_extraction(config, source, mode="prose")
    assert shared.target is not None and scoped.target is not None
    assert (
        scoped.target.provenance["request_fingerprint"]
        == shared.target.provenance["request_fingerprint"]
    )

    expected = _expectation(
        config,
        scoped,
        source,
        scope_id=SCOPE,
        destination_deck=deck,
        destination_deck_sha256=hashlib.sha256(deck.read_bytes()).hexdigest(),
    )
    fake = FakeCall(ok(candidate()))
    monkeypatch.setattr("japanese_anki.extract.claude_client.parse_call", fake)
    deck.write_text(
        deck.read_text(encoding="utf-8") + "  # a note added after the plan\n",
        encoding="utf-8",
    )

    with pytest.raises(ExtractionDispatchError, match="destination deck changed") as caught:
        dispatch_extraction(config, expected, client=object())

    assert caught.value.phase == "binding"
    assert not caught.value.provider_dispatched
    assert fake.calls == []
    assert not config.operations_file.exists()


def test_a_destination_deck_whose_scope_moved_refuses_by_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scope is checked as a scope, not merely as bytes.

    Same empty collection, same request fingerprint, and the deck file is now a
    different collection's deck. Refusing here says which binding broke, so a
    reader is not told a comment edit and a re-scoped deck are the same event.
    """
    root = _project(tmp_path)
    config = ProjectConfig.load(root)
    source = _source(root)
    deck = _deck(root, SCOPE)

    scoped = describe_extraction(config, source, mode="prose", scope_id=SCOPE)
    expected = _expectation(
        config,
        scoped,
        source,
        scope_id=SCOPE,
        destination_deck=deck,
        destination_deck_sha256=hashlib.sha256(deck.read_bytes()).hexdigest(),
    )
    fake = FakeCall(ok(candidate()))
    monkeypatch.setattr("japanese_anki.extract.claude_client.parse_call", fake)
    _deck(root, OTHER_SCOPE)

    with pytest.raises(
        ExtractionDispatchError, match="destination deck scope changed"
    ) as caught:
        dispatch_extraction(config, expected, client=object())

    assert caught.value.phase == "binding"
    assert fake.calls == []
    assert not config.operations_file.exists()


def test_the_assistant_binds_the_selected_deck_and_names_its_known_pool(
    tmp_path: Path,
) -> None:
    """Selection is explicit and server-resolved, and the owner reads it.

    The scope comes from the deck the thread already selected, resolved through
    the startup allowlist — never from prose in the message. What the batch
    then says is the whole point: which deck it is for, and which pool of
    already-known words is being suppressed.
    """
    root = _project(tmp_path, COLLECTION)
    config = ProjectConfig.load(root)
    deck = _deck(root, SCOPE)
    adapter = _adapter(config, deck)
    deck_scope = adapter.deck_choices[0].scope

    plan = adapter.prepare_source_extraction(
        source_path=_source(root),
        deck_scope=deck_scope,
    )

    effects = " ".join(plan.effects)
    assert "Standalone Verbs" in effects
    assert deck_scope in effects
    assert SCOPE in effects
    assert "already in that deck" in effects
    assert "does not assign" in " ".join(plan.disclosures)

    expectation = adapter._extraction_expectations[plan.preparation_id]
    assert expectation.scope_id == SCOPE
    assert expectation.destination_deck == deck.resolve()
    assert (
        expectation.destination_deck_sha256
        == hashlib.sha256(deck.read_bytes()).hexdigest()
    )


def test_the_assistant_without_a_selected_deck_stays_on_the_shared_collection(
    tmp_path: Path,
) -> None:
    """No selection is not an inferred selection. It is the shared collection,
    said out loud, with no destination bound into the consent."""
    root = _project(tmp_path, COLLECTION)
    config = ProjectConfig.load(root)
    adapter = _adapter(config)

    plan = adapter.prepare_source_extraction(source_path=_source(root))

    assert "shared collection" in " ".join(plan.effects)
    expectation = adapter._extraction_expectations[plan.preparation_id]
    assert expectation.scope_id == ""
    assert expectation.destination_deck is None
    assert expectation.destination_deck_sha256 == ""


def test_the_deck_scope_and_hash_are_read_from_one_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scope and a hash of two different file versions are worse than either.

    The dispatch guard compares both against what was rendered, so a helper
    that reads the deck twice can answer with the old scope and the new bytes:
    the scope check passes on a file that has already moved collections, and
    the refusal that follows names the wrong thing. One read, parsed and hashed.
    """
    root = _project(tmp_path)
    _deck(root, OTHER_SCOPE)
    replacement = (root / "data" / "decks" / "standalone.yaml").read_bytes()
    deck = _deck(root, SCOPE)

    original = extraction_application.read_bytes_bound
    reads: list[Path] = []

    def racing_read(path: Path, *args: Any, **kwargs: Any) -> bytes:
        reads.append(Path(path))
        if Path(path) == deck:
            deck.write_bytes(replacement)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(extraction_application, "read_bytes_bound", racing_read)

    scope, deck_sha256 = destination_deck_facts(deck)

    assert scope == OTHER_SCOPE
    assert deck_sha256 == hashlib.sha256(replacement).hexdigest()
    assert reads.count(deck) == 1


def test_the_typed_source_intent_prepares_with_its_selected_deck_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The model may ask to extract a source; it may not choose the collection.

    The typed intent carries no scope of its own — the branch refuses any
    option it did not resolve itself — so the destination is the focus the
    thread already selected, passed through by janki rather than read out of
    the answer.
    """
    root = _project(tmp_path, COLLECTION)
    config = ProjectConfig.load(root)
    deck = _deck(root, SCOPE)
    adapter = _adapter(config, deck)
    deck_scope = adapter.deck_choices[0].scope
    monkeypatch.setattr(
        assistant_adapter.assistant_context.AssistantContextBroker,
        "source_path",
        lambda self, resource_id: _source(root),
    )
    intent = SimpleNamespace(
        kind="extract_source",
        resource_ids=("source-proof",),
        record_ids=(),
        options={},
        instruction="Extract this source",
    )
    result = SimpleNamespace(
        action_intents=(intent,),
        answer="Prepare this source",
    )

    reply = adapter._prepare_agent_intent(config, result=result, deck_scope=deck_scope)

    [expectation] = adapter._extraction_expectations.values()
    assert expectation.scope_id == SCOPE
    assert expectation.destination_deck == deck.resolve()
    assert SCOPE in " ".join(reply.action.effects)


def test_an_uploaded_source_uses_the_threads_verified_deck_selection(
    tmp_path: Path,
) -> None:
    """The attachment path has a selection already: the thread's explicit deck
    focus, verified against the startup catalogue before any of this runs.

    It is forwarded, not re-derived, and never taken from the text typed
    alongside the upload. The chat callbacks stay fake here, so selecting a
    deck and uploading a file make no model call.
    """
    from test_workbench_assistant import (
        SESSION_TOKEN,
        _action_request,
        _create_source_extraction_plan,
        _deck_selector_action,
        _FakeRevisions,
        _post,
        _start_deck_selector,
        create_assistant_sidecar,
    )

    root = _project(tmp_path, COLLECTION)
    config = ProjectConfig.load(root)
    deck = _deck(root, SCOPE)
    adapter = _adapter(config, deck)
    choice = adapter.deck_choices[0]
    callbacks = _FakeRevisions()
    forwarded: list[str] = []

    def prepare(*, source_path: Path, deck_scope: str = "") -> Any:
        forwarded.append(deck_scope)
        return adapter.prepare_source_extraction(
            source_path=source_path,
            deck_scope=deck_scope,
        )

    callbacks.prepare_source_extraction = prepare
    sidecar = create_assistant_sidecar(
        callbacks,
        deck_choices=adapter.deck_choices,
        session_token=SESSION_TOKEN,
        inbox_root=root / "inbox",
    )
    sidecar.start()
    try:
        thread_id, selector, _events = _start_deck_selector(sidecar)
        status, _headers, _body = _post(
            sidecar,
            _action_request(
                thread_id,
                selector,
                _deck_selector_action(selector, choice.deck_id),
            ),
        )
        assert status == 200
        _create_source_extraction_plan(sidecar, PDF, thread_id=thread_id)
    finally:
        sidecar.close()

    assert forwarded == [choice.scope]
    [expectation] = adapter._extraction_expectations.values()
    assert expectation.scope_id == SCOPE
    assert expectation.destination_deck == deck.resolve()


def test_the_assistant_refuses_a_destination_that_cannot_hold_vocabulary(
    tmp_path: Path,
) -> None:
    """A conjugation deck is a real selected deck and a wrong destination for
    proposed vocabulary cards. Refuse the preparation rather than silently
    extracting for the shared collection instead."""
    root = _project(tmp_path, COLLECTION)
    config = ProjectConfig.load(root)
    deck = _deck(root, SCOPE, kind="conjugation")
    adapter = _adapter(config, deck)

    with pytest.raises(RevisionRefusal, match="vocabulary"):
        adapter.prepare_source_extraction(
            source_path=_source(root),
            deck_scope=adapter.deck_choices[0].scope,
        )
