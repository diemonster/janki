"""The paid revision path stops at a fingerprinted staging proposal."""

from __future__ import annotations

import contextlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from conftest import seed_prompts
from japanese_anki import operations
from japanese_anki.application import revision, revision_apply, revision_provider
from japanese_anki.claude_client import CallResult
from japanese_anki.config import ProjectConfig

SELECTED = "word:話す:はなす"
OTHER = "word:遊ぶ:あそぶ"


def _record(
    record_id: str,
    expression: str,
    reading: str,
    potential: str,
    meaning: str,
) -> dict[str, Any]:
    return {
        "id": record_id,
        "expression": expression,
        "reading": reading,
        "meanings": [meaning],
        "part_of_speech": "godan verb",
        "verb_group": "godan",
        "transitivity": "intransitive",
        "conjugations": {"potential": potential},
        "usage_notes": "Canonical detail that this narrow pass must not send.",
        "source": {
            "type": "manual",
            "imported_from": "private-source.csv",
        },
    }


def _project(tmp_path: Path) -> tuple[ProjectConfig, Path]:
    (tmp_path / "janki.toml").write_text(
        "[ai]\n"
        "extract_model = \"do-not-use-for-revision\"\n"
        "revise_model = \"claude-opus-5\"\n",
        encoding="utf-8",
    )
    seed_prompts(tmp_path)
    normalized = tmp_path / "data" / "normalized"
    normalized.mkdir(parents=True)
    (normalized / "vocabulary.json").write_text(
        json.dumps(
            [
                _record(SELECTED, "話す", "はなす", "話せる", "to speak"),
                _record(OTHER, "遊ぶ", "あそぶ", "遊べる", "to play"),
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    decks = tmp_path / "data" / "decks"
    decks.mkdir(parents=True)
    deck = decks / "potential.yaml"
    deck.write_text(
        "deck:\n"
        "  kind: conjugation\n"
        "  form: potential\n"
        "  name: Potential practice\n"
        "  deck_id: 19\n"
        "  model_id: 20\n"
        "  form_note: Existing potential note.\n"
        "  include_ids:\n"
        f"    - {SELECTED}\n"
        f"    - {OTHER}\n"
        "  drill_examples:\n"
        f"    \"{SELECTED}\":\n"
        "      - japanese: 日本語が話せます。\n"
        "        furigana: 日本語[にほんご]が 話[はな]せます。\n"
        "        english: I can speak Japanese.\n"
        "        register: polite\n"
        "        audio: audio/selected.wav\n"
        "      - japanese: 話せる？\n"
        "        furigana: 話[はな]せる？\n"
        "        english: Can you speak?\n"
        "        register: casual\n"
        f"    \"{OTHER}\":\n"
        "      - japanese: 遊べます。\n"
        "        furigana: 遊[あそ]べます。\n"
        "        english: I can play.\n"
        "        register: polite\n"
        "        audio: audio/unselected-secret.wav\n"
        "      - japanese: 遊べる？\n"
        "        furigana: 遊[あそ]べる？\n"
        "        english: Can you play?\n"
        "        register: casual\n",
        encoding="utf-8",
    )
    return ProjectConfig.load(tmp_path), deck


def _answer(plan: revision.RevisionPlan, *, record_id: str = SELECTED) -> Any:
    return plan.schema.model_validate(
        {
            "form_note": "Potential says what someone can do.",
            "cards": [
                {
                    "record_id": record_id,
                    "examples": [
                        {
                            "japanese": "日本語が話せます。",
                            "speech_level": "polite",
                            "furigana": "日本語[にほんご]が 話[はな]せます。",
                            "english": "I can speak Japanese.",
                        },
                        {
                            "japanese": "英語も話せる？",
                            "speech_level": "casual",
                            "furigana": "英語[えいご]も 話[はな]せる？",
                            "english": "Can you speak English too?",
                        },
                    ],
                }
            ],
        }
    )


def _api_response(answer: Any) -> dict[str, Any]:
    return {
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": answer.model_dump_json()}],
    }


def _capturing_call(
    config: ProjectConfig,
    plan: revision.RevisionPlan,
    events: list[str],
):
    def call(*_args: Any, capture=None, **_kwargs: Any) -> CallResult:
        [entry] = operations.OperationJournal.load(
            config.operations_file
        ).operations.values()
        events.append(entry.state)
        assert entry.kind == "revise"
        assert entry.source_file == plan.deck_relative_path
        assert entry.source_sha256 == plan.deck_sha256
        assert entry.request_fp == plan.request_fingerprint
        assert capture is not None
        capture(_api_response(_answer(plan)))
        [captured] = operations.OperationJournal.load(
            config.operations_file
        ).operations.values()
        events.append(captured.state)
        manifest = json.loads(plan.staging_path.read_text(encoding="utf-8"))
        assert manifest["state"] == "request"
        assert manifest["request"]["owner_instruction"] == plan.owner_instruction
        assert manifest["request"]["user_turn"] == plan.user_turn
        events.append("parsed")
        return CallResult(_answer(plan), "end_turn", None)

    return call


def test_plan_is_side_effect_free_and_sends_only_selected_content(tmp_path: Path) -> None:
    config, deck = _project(tmp_path)
    original = deck.read_bytes()

    plan = revision.plan_revision(
        config,
        deck,
        [SELECTED],
        "Keep the meaning, but make the casual example more conversational.",
    )

    assert deck.read_bytes() == original
    assert not config.operations_file.exists()
    assert not config.staging_dir.exists()
    assert plan.selected_record_ids == (SELECTED,)
    assert plan.model == "claude-opus-5"
    assert plan.owner_instruction.endswith("conversational.")
    assert SELECTED in plan.user_turn
    assert "audio/selected.wav" not in plan.user_turn
    assert OTHER not in plan.user_turn
    assert "unselected-secret" not in plan.user_turn
    assert "private-source.csv" not in plan.user_turn
    assert "Canonical detail" not in plan.user_turn
    sent_context = plan.canonical_context[SELECTED]
    assert set(sent_context) == {
        "id",
        "expression",
        "reading",
        "meanings",
        "part_of_speech",
        "verb_group",
        "transitivity",
        "target_conjugation",
    }


@pytest.mark.parametrize("override", ["provider", "model"])
def test_revision_transport_and_model_come_only_from_project_config(
    tmp_path: Path,
    override: str,
) -> None:
    config, deck = _project(tmp_path)

    with pytest.raises(TypeError, match=f"unexpected keyword argument '{override}'"):
        revision.plan_revision(
            config,
            deck,
            [SELECTED],
            "Revise it.",
            **{override: "unconfirmed-override"},
        )

    assert not config.operations_file.exists()


@pytest.mark.parametrize(
    ("change", "request_changes", "plan_changes"),
    [
        ("instruction", True, True),
        ("selected_context", True, True),
        ("task_prompt", True, True),
        ("deck_whitespace", False, True),
        ("staging_state", False, True),
    ],
)
def test_fingerprints_bind_every_request_and_local_plan_channel(
    tmp_path: Path,
    change: str,
    request_changes: bool,
    plan_changes: bool,
) -> None:
    config, deck = _project(tmp_path)
    instruction = "Revise this selected card."
    original = revision.plan_revision(config, deck, [SELECTED], instruction)

    if change == "instruction":
        instruction = "Revise this selected card differently."
    elif change == "selected_context":
        records = json.loads(config.normalized_file.read_text(encoding="utf-8"))
        records[0]["meanings"] = ["to converse"]
        config.normalized_file.write_text(
            json.dumps(records, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    elif change == "task_prompt":
        prompt = tmp_path / "prompts" / "revise-conjugation-deck.md"
        prompt.write_text(prompt.read_text(encoding="utf-8") + "\nBe concise.\n")
    elif change == "deck_whitespace":
        deck.write_bytes(deck.read_bytes() + b"\n")
    elif change == "staging_state":
        original.staging_path.parent.mkdir(parents=True)
        original.staging_path.write_text('{"occupied":true}\n', encoding="utf-8")

    changed = revision.plan_revision(
        config,
        deck,
        [SELECTED],
        instruction,
    )

    assert (changed.request_fingerprint != original.request_fingerprint) is request_changes
    assert (changed.plan_fingerprint != original.plan_fingerprint) is plan_changes


@pytest.mark.parametrize(
    "payload",
    [
        {
            "form_note": "note",
            "cards": [
                {
                    "record_id": SELECTED,
                    "examples": [
                        {
                            "japanese": "話せます。",
                            "speech_level": "polite",
                            "furigana": "話[はな]せます。",
                            "english": "I can speak.",
                        }
                    ],
                }
            ],
        },
        {
            "form_note": "note",
            "cards": [
                {
                    "record_id": SELECTED,
                    "examples": [
                        {
                            "japanese": "話せる。",
                            "speech_level": "casual",
                            "furigana": "話[はな]せる。",
                            "english": "I can speak.",
                        },
                        {
                            "japanese": "話せる？",
                            "speech_level": "casual",
                            "furigana": "話[はな]せる？",
                            "english": "Can you speak?",
                        },
                    ],
                }
            ],
        },
        {
            "form_note": "note",
            "cards": [
                {
                    "record_id": SELECTED,
                    "examples": [
                        {
                            "japanese": "話せます。",
                            "speech_level": "polite",
                            "furigana": "話[はな]せます。",
                            "english": "I can speak.",
                            "audio": "forbidden.wav",
                        },
                        {
                            "japanese": "話せる？",
                            "speech_level": "casual",
                            "furigana": "話[はな]せる？",
                            "english": "Can you speak?",
                        },
                    ],
                }
            ],
        },
    ],
)
def test_schema_requires_complete_polite_and_casual_content_only(
    tmp_path: Path,
    payload: dict[str, Any],
) -> None:
    config, deck = _project(tmp_path)
    plan = revision.plan_revision(config, deck, [SELECTED], "Revise it.")

    with pytest.raises(ValidationError):
        plan.schema.model_validate(payload)


@pytest.mark.parametrize("selected", [[], [""], [SELECTED, SELECTED]])
def test_plan_requires_an_exact_nonempty_unique_selection(
    tmp_path: Path, selected: list[str]
) -> None:
    config, deck = _project(tmp_path)

    with pytest.raises(revision.RevisionApplicationError, match="at least one|nonblank|only once"):
        revision.plan_revision(config, deck, selected, "Revise it.")

    assert not config.operations_file.exists()


def test_plan_refuses_a_selected_record_the_deck_cannot_currently_conjugate(
    tmp_path: Path,
) -> None:
    config, deck = _project(tmp_path)
    records = json.loads(config.normalized_file.read_text(encoding="utf-8"))
    records[0]["verb_group"] = "unknown"
    records[0]["part_of_speech"] = "verb"
    # A stale stored answer must not substitute for the build's current rules.
    assert records[0]["conjugations"]["potential"] == "話せる"
    config.normalized_file.write_text(
        json.dumps(records, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    with pytest.raises(revision.RevisionApplicationError, match="cannot currently"):
        revision.plan_revision(config, deck, [SELECTED], "Revise it.")

    assert not config.operations_file.exists()


def test_stale_request_and_plan_are_refused_before_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, deck = _project(tmp_path)
    plan = revision.plan_revision(config, deck, [SELECTED], "Revise it.")
    deck.write_text(
        deck.read_text(encoding="utf-8").replace("Can you speak?", "Can you talk?"),
        encoding="utf-8",
    )
    called = False

    def prepare(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal called
        called = True

    provider = revision_provider.provider_for(plan.provider)
    monkeypatch.setattr(type(provider), "prepare", prepare)

    with pytest.raises(revision.RevisionApplicationError, match="request-stale"):
        revision.run_revision(config, plan)

    assert not called
    assert not config.operations_file.exists()

    fresh = revision.plan_revision(config, deck, [SELECTED], "Revise it.")
    deck.write_bytes(deck.read_bytes() + b"\n")
    with pytest.raises(revision.RevisionApplicationError, match="plan-stale"):
        revision.run_revision(config, fresh)
    assert not called
    assert not config.operations_file.exists()


def test_switching_revision_provider_invalidates_confirmation_before_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, deck = _project(tmp_path)
    plan = revision.plan_revision(config, deck, [SELECTED], "Revise it.")
    switched = replace(config, revise_provider="claude-code")
    changed_provider_plan = replace(
        plan.provider_plan,
        provider="claude-code",
        request_fingerprint="f" * 64,
    )

    def changed_plan_provider(name: str, **_kwargs: Any) -> Any:
        assert name == "claude-code"
        return changed_provider_plan

    monkeypatch.setattr(
        revision.revision_provider,
        "plan_provider",
        changed_plan_provider,
    )

    with pytest.raises(revision.RevisionApplicationError, match="request-stale"):
        revision.run_revision(switched, plan)

    assert not config.operations_file.exists()
    assert not plan.staging_path.exists()


def test_client_is_prepared_before_journal_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, deck = _project(tmp_path)
    plan = revision.plan_revision(config, deck, [SELECTED], "Revise it.")

    def prepare(*_args: Any, **_kwargs: Any) -> Any:
        assert not config.operations_file.exists()
        raise revision.RevisionApplicationError("credential preflight failed")

    provider = revision_provider.provider_for(plan.provider)
    monkeypatch.setattr(type(provider), "prepare", prepare)

    with pytest.raises(revision.RevisionApplicationError, match="credential preflight"):
        revision.run_revision(config, plan)

    assert not config.operations_file.exists()
    assert not plan.staging_path.exists()


@pytest.mark.parametrize("changed", ["deck", "canonical", "task_prompt"])
def test_client_preparation_cannot_stale_bindings_before_authority(
    tmp_path: Path,
    changed: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, deck = _project(tmp_path)
    plan = revision.plan_revision(config, deck, [SELECTED], "Revise it.")
    called = False

    def prepare(
        _provider: Any,
        provider_plan: revision_provider.RevisionProviderPlan,
        **_kwargs: Any,
    ) -> revision_provider.PreparedRevisionProvider:
        if changed == "deck":
            deck.write_text(
                deck.read_text(encoding="utf-8").replace(
                    "Can you speak?", "Can you still speak?"
                ),
                encoding="utf-8",
            )
        elif changed == "canonical":
            records = json.loads(config.normalized_file.read_text(encoding="utf-8"))
            records[0]["meanings"] = ["to converse"]
            config.normalized_file.write_text(
                json.dumps(records, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        else:
            task = tmp_path / "prompts" / "revise-conjugation-deck.md"
            task.write_text(task.read_text(encoding="utf-8") + "\nChanged.\n")
        return revision_provider.PreparedRevisionProvider(
            plan=provider_plan,
            resource=object(),
        )

    def call(*_args: Any, **_kwargs: Any) -> CallResult:
        nonlocal called
        called = True
        return CallResult(_answer(plan), "end_turn", None)

    provider = revision_provider.provider_for(plan.provider)
    monkeypatch.setattr(type(provider), "prepare", prepare)

    with pytest.raises(revision.RevisionApplicationError, match="request-stale"):
        revision.run_revision(config, plan, api_call=call)

    assert not called
    assert not config.operations_file.exists()
    assert not plan.staging_path.exists()


def test_binding_locks_span_authority_and_request_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, deck = _project(tmp_path)
    plan = revision.plan_revision(config, deck, [SELECTED], "Revise it.")
    real_lock = revision.exclusive_path_lock
    real_authorize = operations.OperationJournal.authorize
    real_write = revision.atomic_write_text_bound
    active: set[Path] = set()
    acquired: list[Path] = []
    bound = {
        path.absolute()
        for path in (
            deck,
            config.normalized_file,
            plan.staging_path,
            config.root / "prompts" / "style-guide.md",
            config.root / "prompts" / "revise-conjugation-deck.md",
        )
    }

    @contextlib.contextmanager
    def tracking_lock(path: Path):
        absolute = path.absolute()
        with real_lock(path):
            active.add(absolute)
            acquired.append(absolute)
            try:
                yield
            finally:
                active.remove(absolute)

    def observing_authorize(self: operations.OperationJournal, *args: Any, **kwargs: Any):
        assert active == bound
        assert acquired == sorted(bound, key=str)
        return real_authorize(self, *args, **kwargs)

    def observing_write(path: Path, text: str, **kwargs: Any) -> None:
        if json.loads(text).get("state") == "request":
            assert active == bound
        real_write(path, text, **kwargs)

    def call(*_args: Any, capture=None, **_kwargs: Any) -> CallResult:
        assert not active
        assert capture is not None
        capture(_api_response(_answer(plan)))
        return CallResult(_answer(plan), "end_turn", None)

    monkeypatch.setattr(revision, "exclusive_path_lock", tracking_lock)
    monkeypatch.setattr(operations.OperationJournal, "authorize", observing_authorize)
    monkeypatch.setattr(revision, "atomic_write_text_bound", observing_write)

    revision.run_revision(config, plan, client=object(), api_call=call)


def test_manifest_failure_cancels_authority_before_send(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, deck = _project(tmp_path)
    plan = revision.plan_revision(config, deck, [SELECTED], "Revise it.")
    called = False

    def fail_manifest(*_args: Any, **_kwargs: Any) -> None:
        raise revision.RevisionApplicationError("manifest disk failed")

    def call(*_args: Any, **_kwargs: Any) -> CallResult:
        nonlocal called
        called = True
        return CallResult(_answer(plan), "end_turn", None)

    monkeypatch.setattr(revision, "atomic_write_text_bound", fail_manifest)

    with pytest.raises(revision.RevisionRunError, match="canceled before send"):
        revision.run_revision(config, plan, client=object(), api_call=call)

    assert not called
    [entry] = operations.OperationJournal.load(config.operations_file).operations.values()
    assert entry.kind == "revise"
    assert entry.state == "canceled_before_send"
    assert entry.money_may_have_been_spent is False


def test_request_manifest_is_durable_while_operation_is_only_authorized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, deck = _project(tmp_path)
    plan = revision.plan_revision(config, deck, [SELECTED], "Revise it.")
    real_write = revision.atomic_write_text_bound
    observed: list[str] = []

    def observing_write(path: Path, text: str, **kwargs: Any) -> None:
        value = json.loads(text)
        if value.get("state") == "request":
            [entry] = operations.OperationJournal.load(
                config.operations_file
            ).operations.values()
            observed.append(entry.state)
        real_write(path, text, **kwargs)

    monkeypatch.setattr(revision, "atomic_write_text_bound", observing_write)

    revision.run_revision(
        config,
        plan,
        client=object(),
        api_call=_capturing_call(config, plan, []),
    )

    assert observed == ["authorized"]


def test_paid_reply_is_authorized_captured_staged_then_committed(tmp_path: Path) -> None:
    config, deck = _project(tmp_path)
    original_deck = deck.read_bytes()
    plan = revision.plan_revision(config, deck, [SELECTED], "Revise it.")
    events: list[str] = []

    result = revision.run_revision(
        config,
        plan,
        client=object(),
        api_call=_capturing_call(config, plan, events),
    )

    assert events == ["dispatching", "result_captured", "parsed"]
    assert result.staging_path == plan.staging_path
    staged = json.loads(plan.staging_path.read_text(encoding="utf-8"))
    assert staged["kind"] == "conjugation_deck_revision"
    assert staged["state"] == "proposed"
    assert staged["operation_id"] == result.operation_id
    assert staged["target"] == {
        "deck_path": "data/decks/potential.yaml",
        "deck_sha256": plan.deck_sha256,
        "form": "potential",
        "selected_record_ids": [SELECTED],
        "staging_path": plan.staging_path.relative_to(config.root).as_posix(),
    }
    assert staged["request"]["owner_instruction"] == "Revise it."
    assert (
        staged["request"]["provider_plan"]["request_fingerprint"]
        == plan.request_fingerprint
    )
    assert staged["before"]["drill_examples"][SELECTED][0]["audio"] == (
        "audio/selected.wav"
    )
    assert set(staged["proposal"]["drill_examples"][SELECTED][0]) == {
        "japanese",
        "furigana",
        "english",
        "register",
    }
    assert deck.read_bytes() == original_deck
    [entry] = operations.OperationJournal.load(config.operations_file).operations.values()
    assert entry.operation_id == result.operation_id
    assert entry.kind == "revise"
    assert entry.state == "committed"


def test_staged_revision_can_be_planned_for_owner_apply(tmp_path: Path) -> None:
    config, deck = _project(tmp_path)
    plan = revision.plan_revision(config, deck, [SELECTED], "Revise it.")
    result = revision.run_revision(
        config,
        plan,
        client=object(),
        api_call=_capturing_call(config, plan, []),
    )

    apply_plan = revision_apply.plan_revision_apply(config, result.staging_path)

    assert apply_plan.selected_record_ids == (SELECTED,)
    assert apply_plan.state == "proposed"


def test_claude_subscription_revision_composes_through_apply_and_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, deck = _project(tmp_path)
    config = replace(config, revise_provider="claude-code")
    real_plan_provider = revision_provider.plan_provider
    answer: dict[str, Any] | None = None

    def which(name: str, **_kwargs: Any) -> str | None:
        assert name == "claude"
        return "/fake/claude"

    def runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        if "--version" in command:
            stdout = b"2.1.246\n"
        elif "auth" in command:
            stdout = json.dumps(
                {
                    "loggedIn": True,
                    "authMethod": "claude.ai",
                    "apiProvider": "firstParty",
                    "subscriptionType": "max",
                }
            ).encode("utf-8")
        else:
            assert answer is not None
            assert kwargs["input"]
            stdout = json.dumps(
                {
                    "type": "result",
                    "subtype": "success",
                    "is_error": False,
                    "structured_output": answer,
                },
                ensure_ascii=False,
            ).encode("utf-8")
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr=b"")

    def fake_plan_provider(name: str, **kwargs: Any) -> Any:
        return real_plan_provider(name, **kwargs, runner=runner, which=which)

    monkeypatch.setattr(revision_provider, "plan_provider", fake_plan_provider)
    plan = revision.plan_revision(config, deck, [SELECTED], "Revise it.")
    answer = _answer(plan).model_dump(mode="json")

    staged = revision.run_revision(
        config,
        plan,
        provider_runner=runner,
        provider_which=which,
    )
    apply_plan = revision_apply.plan_revision_apply(config, staged.staging_path)
    applied = revision_apply.execute_revision_apply(config, apply_plan)

    assert applied.archive_path.exists()
    assert not staged.staging_path.exists()
    assert "Potential says what someone can do." in deck.read_text(encoding="utf-8")


def test_multi_record_revision_applies_in_non_sorted_deck_order(tmp_path: Path) -> None:
    config, deck = _project(tmp_path)
    deck.write_text(
        deck.read_text(encoding="utf-8").replace(
            f"    - {SELECTED}\n    - {OTHER}\n",
            f"    - {OTHER}\n    - {SELECTED}\n",
        ),
        encoding="utf-8",
    )
    selected = (OTHER, SELECTED)
    plan = revision.plan_revision(config, deck, selected, "Revise both cards.")
    answer = plan.schema.model_validate(
        {
            "form_note": "Potential says what someone can do.",
            "cards": [
                {
                    "record_id": OTHER,
                    "examples": [
                        {
                            "japanese": "遊べます。",
                            "speech_level": "polite",
                            "furigana": "遊[あそ]べます。",
                            "english": "I can play.",
                        },
                        {
                            "japanese": "遊べる？",
                            "speech_level": "casual",
                            "furigana": "遊[あそ]べる？",
                            "english": "Can you play?",
                        },
                    ],
                },
                {
                    "record_id": SELECTED,
                    "examples": [
                        {
                            "japanese": "話せます。",
                            "speech_level": "polite",
                            "furigana": "話[はな]せます。",
                            "english": "I can speak.",
                        },
                        {
                            "japanese": "話せる？",
                            "speech_level": "casual",
                            "furigana": "話[はな]せる？",
                            "english": "Can you speak?",
                        },
                    ],
                },
            ],
        }
    )

    def call(*_args: Any, capture=None, **_kwargs: Any) -> CallResult:
        assert capture is not None
        capture(_api_response(answer))
        return CallResult(answer, "end_turn", None)

    staged = revision.run_revision(config, plan, client=object(), api_call=call)
    manifest = json.loads(staged.staging_path.read_text(encoding="utf-8"))
    assert tuple(manifest["canonical_context"]) != selected

    apply_plan = revision_apply.plan_revision_apply(config, staged.staging_path)
    applied = revision_apply.execute_revision_apply(config, apply_plan)

    assert apply_plan.selected_record_ids == selected
    assert applied.archive_path.exists()
    assert not staged.staging_path.exists()


def test_progress_reports_capture_parse_and_publication_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, deck = _project(tmp_path)
    plan = revision.plan_revision(config, deck, [SELECTED], "Revise it.")
    events: list[str] = []
    real_fresh_plan = revision._fresh_plan

    def fresh_plan(*args: Any, **kwargs: Any) -> revision.RevisionPlan:
        assert events == ["Reading the source"]
        return real_fresh_plan(*args, **kwargs)

    monkeypatch.setattr(revision, "_fresh_plan", fresh_plan)

    def progress(label: str) -> None:
        if label in {"Checking the answer's shape", "Saving proposals"}:
            [entry] = operations.OperationJournal.load(
                config.operations_file
            ).operations.values()
            assert entry.state == "result_captured"
            manifest = json.loads(plan.staging_path.read_text(encoding="utf-8"))
            assert manifest["state"] == "request"
        events.append(label)

    def call(*_args: Any, capture=None, **_kwargs: Any) -> CallResult:
        assert capture is not None
        events.append("provider reply received")
        capture(_api_response(_answer(plan)))
        events.append("parsed")
        return CallResult(_answer(plan), "end_turn", None)

    revision.run_revision(
        config,
        plan,
        client=object(),
        api_call=call,
        progress=progress,
    )

    assert events == [
        "Reading the source",
        "provider reply received",
        "Checking the answer's shape",
        "parsed",
        "Saving proposals",
    ]
    assert json.loads(plan.staging_path.read_text(encoding="utf-8"))["state"] == (
        "proposed"
    )


def test_wrong_response_scope_is_captured_but_never_staged(tmp_path: Path) -> None:
    config, deck = _project(tmp_path)
    plan = revision.plan_revision(config, deck, [SELECTED], "Revise it.")

    def call(*_args: Any, capture=None, **_kwargs: Any) -> CallResult:
        assert capture is not None
        capture(_api_response(_answer(plan, record_id=OTHER)))
        return CallResult(_answer(plan, record_id=OTHER), "end_turn", None)

    with pytest.raises(revision.RevisionRunError, match="every selected record"):
        revision.run_revision(config, plan, client=object(), api_call=call)

    request_manifest = json.loads(plan.staging_path.read_text(encoding="utf-8"))
    assert request_manifest["state"] == "request"
    [entry] = operations.OperationJournal.load(config.operations_file).operations.values()
    assert entry.kind == "revise"
    assert entry.state == "result_captured"


def test_uncaptured_dispatch_failure_is_outcome_unknown(tmp_path: Path) -> None:
    config, deck = _project(tmp_path)
    plan = revision.plan_revision(config, deck, [SELECTED], "Revise it.")

    def call(*_args: Any, **_kwargs: Any) -> CallResult:
        raise OSError("connection vanished")

    with pytest.raises(revision.RevisionRunError, match="connection vanished"):
        revision.run_revision(config, plan, client=object(), api_call=call)

    [entry] = operations.OperationJournal.load(config.operations_file).operations.values()
    assert entry.kind == "revise"
    assert entry.state == "outcome_unknown"
    request_manifest = json.loads(plan.staging_path.read_text(encoding="utf-8"))
    assert request_manifest["state"] == "request"


def test_parsed_value_without_exact_capture_is_never_staged(tmp_path: Path) -> None:
    config, deck = _project(tmp_path)
    plan = revision.plan_revision(config, deck, [SELECTED], "Revise it.")

    def call(*_args: Any, **_kwargs: Any) -> CallResult:
        return CallResult(_answer(plan), "end_turn", None)

    with pytest.raises(revision.RevisionRunError, match="no reply to capture"):
        revision.run_revision(config, plan, client=object(), api_call=call)

    manifest = json.loads(plan.staging_path.read_text(encoding="utf-8"))
    assert manifest["state"] == "request"
    [entry] = operations.OperationJournal.load(config.operations_file).operations.values()
    assert entry.state == "outcome_unknown"


def test_captured_reply_can_be_recovered_without_redispatch(tmp_path: Path) -> None:
    config, deck = _project(tmp_path)
    plan = revision.plan_revision(config, deck, [SELECTED], "Revise it.")
    answer_text = _answer(plan).model_dump_json()

    def call(*_args: Any, capture=None, **_kwargs: Any) -> CallResult:
        assert capture is not None
        capture(
            {
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": answer_text}],
            }
        )
        raise revision.RevisionApplicationError("local parser crashed")

    with pytest.raises(revision.RevisionRunError, match="local parser crashed") as failed:
        revision.run_revision(config, plan, client=object(), api_call=call)

    request_manifest = json.loads(plan.staging_path.read_text(encoding="utf-8"))
    assert request_manifest["state"] == "request"
    [captured] = operations.OperationJournal.load(
        config.operations_file
    ).operations.values()
    assert captured.state == "result_captured"

    progress: list[str] = []
    recovered = revision.recover_revision(
        config,
        failed.value.operation_id,
        progress=progress.append,
    )

    assert recovered.operation_id == failed.value.operation_id
    assert progress == [
        "Reading the source",
        "Checking the answer's shape",
        "Saving proposals",
    ]
    proposed = json.loads(plan.staging_path.read_text(encoding="utf-8"))
    assert proposed["state"] == "proposed"
    assert proposed["proposal"]["drill_examples"][SELECTED][1]["register"] == (
        "casual"
    )
    [committed] = operations.OperationJournal.load(
        config.operations_file
    ).operations.values()
    assert committed.state == "committed"


def test_proposed_manifest_recovers_when_the_committed_journal_write_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, deck = _project(tmp_path)
    plan = revision.plan_revision(config, deck, [SELECTED], "Revise it.")
    real_move = operations.OperationJournal._move_under_lock
    failed_commit = False
    answer_text = _answer(plan).model_dump_json()

    def call(*_args: Any, capture=None, **_kwargs: Any) -> CallResult:
        assert capture is not None
        capture(
            {
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": answer_text}],
            }
        )
        return CallResult(_answer(plan), "end_turn", None)

    def fail_first_commit(
        self: operations.OperationJournal,
        current: operations.OperationJournal,
        operation_id: str,
        state: str,
        *,
        artifact: operations.ArtifactReceipt | None = None,
        detail: str = "",
    ) -> operations.Operation:
        nonlocal failed_commit
        if state == "committed" and not failed_commit:
            failed_commit = True
            raise operations.OperationError("journal committed write failed")
        return real_move(
            self,
            current,
            operation_id,
            state,
            artifact=artifact,
            detail=detail,
        )

    monkeypatch.setattr(operations.OperationJournal, "_move_under_lock", fail_first_commit)

    with pytest.raises(revision.RevisionRunError, match="journal committed write failed") as failed:
        revision.run_revision(
            config,
            plan,
            client=object(),
            api_call=call,
        )

    stranded = json.loads(plan.staging_path.read_text(encoding="utf-8"))
    assert stranded["state"] == "proposed"
    [captured] = operations.OperationJournal.load(
        config.operations_file
    ).operations.values()
    assert captured.state == "result_captured"

    original_proposal = plan.staging_path.read_text(encoding="utf-8")
    plan.staging_path.write_text("\n" + original_proposal, encoding="utf-8")
    with pytest.raises(revision.RevisionApplicationError, match="canonical manifest"):
        revision.recover_revision(config, failed.value.operation_id)
    plan.staging_path.write_text(original_proposal, encoding="utf-8")

    tampered = json.loads(original_proposal)
    tampered["proposal"]["drill_examples"][SELECTED][0]["english"] = "Tampered."
    plan.staging_path.write_text(
        json.dumps(tampered, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(revision.RevisionApplicationError, match="exact captured reply"):
        revision.recover_revision(config, failed.value.operation_id)
    [still_captured] = operations.OperationJournal.load(
        config.operations_file
    ).operations.values()
    assert still_captured.state == "result_captured"
    plan.staging_path.write_text(original_proposal, encoding="utf-8")

    recovered = revision.recover_revision(config, failed.value.operation_id)

    assert recovered.operation_id == failed.value.operation_id
    assert plan.staging_path.read_text(encoding="utf-8") == revision._manifest_text(
        stranded
    )
    [committed] = operations.OperationJournal.load(
        config.operations_file
    ).operations.values()
    assert committed.state == "committed"


def test_recovery_refuses_a_manifest_that_no_longer_binds_the_request(
    tmp_path: Path,
) -> None:
    config, deck = _project(tmp_path)
    plan = revision.plan_revision(config, deck, [SELECTED], "Revise it.")
    answer_text = _answer(plan).model_dump_json()

    def call(*_args: Any, capture=None, **_kwargs: Any) -> CallResult:
        assert capture is not None
        capture(
            {
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": answer_text}],
            }
        )
        raise revision.RevisionApplicationError("stop after capture")

    with pytest.raises(revision.RevisionRunError) as failed:
        revision.run_revision(config, plan, client=object(), api_call=call)

    manifest = json.loads(plan.staging_path.read_text(encoding="utf-8"))
    manifest["request"]["owner_instruction"] = "A widened instruction."
    plan.staging_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(revision.RevisionApplicationError, match="do not agree|fingerprint"):
        revision.recover_revision(config, failed.value.operation_id)

    [captured] = operations.OperationJournal.load(
        config.operations_file
    ).operations.values()
    assert captured.state == "result_captured"
