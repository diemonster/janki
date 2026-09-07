"""Explicit kanji targets are one conversation: preview, confirm, download.

The flow under test is the whole one: five characters produce one exact
preview of the actual note sides, one confirmation writes the notes and any
new deck and builds the package, and the finished deck is offered in the same
thread from its durable receipt.
"""

from __future__ import annotations

import hashlib
import http.client
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_application_assistant_kanji_notes import _example, _note, _usage

from japanese_anki import kanji_notes
from japanese_anki.application import character_notes, deck_package, kanji_finish
from japanese_anki.config import ProjectConfig
from japanese_anki.workbench import assistant_adapter, assistant_packages
from japanese_anki.workbench.assistant import (
    RESUME_KANJI_MESSAGE,
    RevisionConfirmation,
    RevisionRefusal,
)
from japanese_anki.workbench.assistant_http import create_assistant_sidecar

_FIVE = ("物", "特", "鳥", "料", "理")
_NOTES = (
    _note(
        "物",
        ("thing", "object"),
        stroke_count=8,
        readings=(
            _usage(
                "物",
                "ぶつ",
                "(56%)",
                56,
                examples=(_example("動物", "どうぶつ", "animal"),),
            ),
        ),
    ),
    _note("特", ("special",)),
    _note("鳥", ("bird",)),
    _note("料", ("fee",)),
    _note("理", ("logic",)),
)
_OWNER_MESSAGE = "I want to study these five kanji: 物特鳥料理 in Genki II Kanji."
_INSTRUCTION = "Add character notes for these five kanji."


def _config(tmp_path: Path) -> ProjectConfig:
    (tmp_path / "janki.toml").write_text(
        "[assistant]\nenabled = true\n\n"
        "[paths]\n"
        'normalized_file = "data/normalized/vocabulary.json"\n'
        'deck_dir = "data/decks"\n'
        'dist_dir = "dist"\n'
        'staging_dir = "data/staging"\n'
        'kanji_notes_file = "data/kanji_notes.json"\n',
        encoding="utf-8",
    )
    normalized = tmp_path / "data/normalized/vocabulary.json"
    normalized.parent.mkdir(parents=True)
    normalized.write_text("[]\n", encoding="utf-8")
    return ProjectConfig.load(tmp_path)


def _adapter(config: ProjectConfig) -> assistant_adapter.RevisionAssistantAdapter:
    return assistant_adapter.RevisionAssistantAdapter(
        config=config,
        deck_choices=(),
        _targets=(),
    )


def _intent(**options: Any) -> Any:
    values: dict[str, Any] = {
        "kanji_characters": list(_FIVE),
        "deck_name": "Genki II Kanji",
    }
    values.update(options)
    return assistant_adapter.assistant_agent.AgentActionIntent(
        kind="add_kanji_notes",
        resource_ids=(),
        record_ids=(),
        instruction=_INSTRUCTION,
        options_json=json.dumps(
            {key: value for key, value in values.items() if value is not None},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def _result(intent: Any) -> Any:
    return SimpleNamespace(
        answer="Here are the five character notes.",
        action_intents=(intent,),
    )


@dataclass(frozen=True)
class _Plan:
    """A stand-in for the pinned CharacterNotesPlan; its notes are real."""

    deck_path: Path
    output_path: Path
    deck_name: str
    characters: tuple[str, ...]
    directions: tuple[str, ...]
    notes: tuple[kanji_notes.CharacterNote, ...]
    deck_record_ids: tuple[str, ...]
    fingerprint: str = "b" * 64

    @property
    def record_ids(self) -> tuple[str, ...]:
        return tuple(note.id for note in self.notes)

    @property
    def note_count(self) -> int:
        return len(self.notes)

    @property
    def card_count(self) -> int:
        return len(self.notes) * len(self.directions)

    @property
    def deck_note_count(self) -> int:
        return len(self.deck_record_ids)

    @property
    def deck_card_count(self) -> int:
        return len(self.deck_record_ids) * len(self.directions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "deck_path": self.deck_path.as_posix(),
            "output_path": self.output_path.as_posix(),
            "deck_name": self.deck_name,
            "characters": list(self.characters),
            "directions": list(self.directions),
            "deck_record_ids": list(self.deck_record_ids),
            "fingerprint": self.fingerprint,
            "notes": [
                {"character": note.character, **note.to_dict()} for note in self.notes
            ],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> _Plan:
        return cls(
            deck_path=Path(value["deck_path"]),
            output_path=Path(value["output_path"]),
            deck_name=value["deck_name"],
            characters=tuple(value["characters"]),
            directions=tuple(value["directions"]),
            notes=tuple(
                kanji_notes.CharacterNote.from_dict(str(item["character"]), item)
                for item in value["notes"]
            ),
            deck_record_ids=tuple(value["deck_record_ids"]),
            fingerprint=value["fingerprint"],
        )


def _install_preparation(
    monkeypatch: pytest.MonkeyPatch,
    config: ProjectConfig,
    *,
    deck_stem: str = "genki-ii-kanji",
    characters: tuple[str, ...] = _FIVE,
    directions: tuple[str, ...] = ("recognition",),
    notes: tuple[kanji_notes.CharacterNote, ...] | None = None,
    deck_record_ids: tuple[str, ...] | None = None,
    calls: list[dict[str, Any]] | None = None,
) -> _Plan:
    prepared = _NOTES if notes is None else notes
    plan = _Plan(
        deck_path=config.deck_dir / f"{deck_stem}.yaml",
        output_path=config.dist_dir / f"{deck_stem}.apkg",
        deck_name="Genki II Kanji",
        characters=characters,
        directions=directions,
        notes=prepared,
        deck_record_ids=(
            tuple(note.id for note in prepared)
            if deck_record_ids is None
            else deck_record_ids
        ),
    )

    def prepare(_config: ProjectConfig, targets: Any, **options: Any) -> _Plan:
        if calls is not None:
            calls.append({"characters": tuple(targets), **options})
        return plan

    monkeypatch.setattr(character_notes, "prepare_character_notes", prepare)
    monkeypatch.setattr(character_notes, "CharacterNotesPlan", _Plan)
    return plan


def _install_apply_and_build(
    monkeypatch: pytest.MonkeyPatch,
    config: ProjectConfig,
    *,
    fail_build: bool = False,
) -> dict[str, Any]:
    package = config.dist_dir / "genki-ii-kanji.apkg"
    package.parent.mkdir(parents=True, exist_ok=True)
    package.write_bytes(b"one exact anki package")
    seen: dict[str, Any] = {"applied": [], "built": []}

    def execute_notes(
        _config: ProjectConfig,
        plan: Any,
        *,
        expected_fingerprint: str,
    ) -> Any:
        seen["applied"].append(expected_fingerprint)
        return SimpleNamespace(
            deck_path=plan.deck_path,
            output_path=plan.output_path,
            record_ids=tuple(f"kanji:{item}" for item in plan.characters),
            note_count=plan.note_count,
            card_count=plan.card_count,
            changed=True,
        )

    def plan_package(inner: ProjectConfig, deck_path: Path, **_kwargs: Any) -> Any:
        return SimpleNamespace(
            deck_path=Path(deck_path).absolute(),
            output_path=(inner.dist_dir / "genki-ii-kanji.apkg").absolute(),
            fingerprint="c" * 64,
        )

    def execute_package(_config: ProjectConfig, plan: Any) -> Any:
        if fail_build:
            raise OSError("the package could not be published")
        seen["built"].append(plan.output_path)
        return SimpleNamespace(
            output_path=plan.output_path,
            package_sha256=hashlib.sha256(package.read_bytes()).hexdigest(),
            note_count=5,
            card_count=5,
        )

    monkeypatch.setattr(character_notes, "execute_character_notes", execute_notes)
    monkeypatch.setattr(deck_package, "plan_deck_package", plan_package)
    monkeypatch.setattr(deck_package, "execute_deck_package", execute_package)
    # Stands in for the core's applied-state check while that call lands; the
    # ordering it guards is covered in tests/test_application_kanji_finish.py.
    monkeypatch.setattr(
        character_notes,
        "verify_character_notes_applied",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    seen["package"] = package
    return seen


def _vocabulary_deck(config: ProjectConfig) -> Path:
    config.deck_dir.mkdir(parents=True, exist_ok=True)
    path = config.deck_dir / "week-two.yaml"
    path.write_text(
        "deck:\n"
        "  name: Week Two\n"
        "  deck_id: 1500000001\n"
        "  source: ../normalized/vocabulary.json\n"
        "  intake_tag: deck:week-two\n"
        "  include_tags: [deck:week-two]\n",
        encoding="utf-8",
    )
    return path


def _kanji_deck(config: ProjectConfig) -> Path:
    config.deck_dir.mkdir(parents=True, exist_ok=True)
    path = config.deck_dir / "genki-ii-kanji.yaml"
    path.write_text(
        "deck:\n"
        "  name: Genki II Kanji\n"
        "  kind: kanji\n"
        "  deck_id: 1500000002\n"
        # Pinned, as `deck_creation` writes it: a character notetype id is what
        # an existing collection matches its notes against.
        "  model_id: 1500000102\n"
        "  include_ids: [kanji:説]\n",
        encoding="utf-8",
    )
    kanji_notes.save_notes(
        config.kanji_notes_file, {"説": _note("説", ("explanation",))}
    )
    return path


def _resource_id(config: ProjectConfig, deck: Path) -> str:
    broker = assistant_adapter.assistant_context.AssistantContextBroker(config)
    return broker.resource_id_for_deck(deck)


def _prepare(
    adapter: assistant_adapter.RevisionAssistantAdapter,
    config: ProjectConfig,
    intent: Any,
    *,
    owner_message: str = _OWNER_MESSAGE,
    owner_history: tuple[str, ...] = (),
) -> Any:
    return adapter._prepare_agent_intent(
        config,
        result=_result(intent),
        deck_scope="",
        owner_message=owner_message,
        owner_history=owner_history,
    )


def test_explicit_targets_preview_the_actual_note_sides_and_exact_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No type question, one confirmation, and the real card content shown."""

    config = _config(tmp_path)
    adapter = _adapter(config)
    _install_preparation(monkeypatch, config)

    reply = _prepare(adapter, config, _intent())

    action = reply.action
    assert action is not None
    assert action.confirm_label == "Add these exact character notes"
    assert action.progress_label == "Preparing finish"
    assert action.target == "data/decks/genki-ii-kanji.yaml"
    assert action.effects[0] == (
        "Add 5 character note(s) — 物、特、鳥、料、理 — to Genki II Kanji"
    )
    assert "Card directions: recognition" in action.effects
    assert "That makes 5 new card(s)." in action.effects
    assert "Create the deck Genki II Kanji and build it" in action.effects
    assert "Recognition front — 物" in action.effects
    assert (
        "Recognition back — Meanings: thing, object · Strokes: 8 · "
        "jpdb reported usage · ぶつ (56%) · 動物 どうぶつ — animal"
        in action.effects
    )
    assert "Recognition front — 理" in action.effects
    assert any(
        disclosure.startswith("Janki looked 物、特、鳥、料、理 up in its dictionary")
        for disclosure in action.disclosures
    )
    assert any("no model call" in disclosure for disclosure in action.disclosures)
    assert list(adapter._agent_plans) == [action.request_fingerprint]


def test_an_existing_character_deck_is_an_accepted_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    deck = _kanji_deck(config)
    adapter = _adapter(config)
    _install_preparation(monkeypatch, config)

    reply = _prepare(
        adapter,
        config,
        _intent(deck_name=None, destination_resource_id=_resource_id(config, deck)),
    )

    action = reply.action
    assert action is not None
    assert (
        "Add to the existing deck Genki II Kanji and rebuild it" in action.effects
    )


def test_a_vocabulary_deck_is_refused_as_a_character_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adding kanji study material never converts an existing note type."""

    config = _config(tmp_path)
    deck = _vocabulary_deck(config)
    adapter = _adapter(config)
    monkeypatch.setattr(
        character_notes,
        "prepare_character_notes",
        lambda *_args, **_kwargs: pytest.fail(
            "an incompatible destination must refuse before preparation"
        ),
    )

    with pytest.raises(RevisionRefusal, match="Character notes need a character deck"):
        _prepare(
            adapter,
            config,
            _intent(deck_name=None, destination_resource_id=_resource_id(config, deck)),
        )


@pytest.mark.parametrize(
    ("options", "owner_message", "message"),
    [
        ({"study_type": "vocabulary"}, _OWNER_MESSAGE, "no character-note flow"),
        (
            {"kanji_characters": ["料理"]},
            "I want to study 料理 as kanji.",
            "exactly one character",
        ),
        (
            {"destination_resource_id": "resource_deck"},
            _OWNER_MESSAGE,
            "not both and not neither",
        ),
    ],
    ids=(
        "unknown-study-type",
        "multi-character-target",
        "two-destinations",
    ),
)
def test_unsupported_or_owner_only_choices_refuse_before_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    options: dict[str, Any],
    owner_message: str,
    message: str,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    monkeypatch.setattr(
        character_notes,
        "prepare_character_notes",
        lambda *_args, **_kwargs: pytest.fail(
            "an unsupported request must refuse before preparation"
        ),
    )

    with pytest.raises(RevisionRefusal, match=message):
        _prepare(adapter, config, _intent(**options), owner_message=owner_message)


def test_one_confirmation_applies_builds_and_offers_the_package_inline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    _install_preparation(monkeypatch, config)
    seen = _install_apply_and_build(monkeypatch, config)
    store = adapter.bind_package_downloads("http://127.0.0.1:9/token/packages/")
    reply = _prepare(adapter, config, _intent())
    action = reply.action
    assert action is not None
    phases: list[str] = []

    execution = adapter.consume_replan_and_execute(
        RevisionConfirmation(
            capability="one-use",
            deck_scope="",
            instruction=_INSTRUCTION,
            expected_fingerprint=action.request_fingerprint,
            target=action.target,
        ),
        progress=phases.append,
    )

    assert seen["applied"] == ["b" * 64]
    assert len(seen["built"]) == 1
    assert execution.complete is True
    assert "Added 5 character note(s) — 物、特、鳥、料、理 — to Genki II Kanji" in (
        execution.message
    )
    assert "[Download genki-ii-kanji.apkg](http://127.0.0.1:9/token/packages/" in (
        execution.message
    )
    assert "Writing character notes" in phases
    assert "Building Anki package" in phases
    token = execution.message.split("/token/packages/")[1].split(")")[0]
    assert store.read(token) == ("genki-ii-kanji.apkg", b"one exact anki package")


def test_a_stale_confirmation_refuses_and_writes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    _install_preparation(monkeypatch, config)
    _install_apply_and_build(monkeypatch, config)
    reply = _prepare(adapter, config, _intent())
    action = reply.action
    assert action is not None
    monkeypatch.setattr(
        character_notes,
        "execute_character_notes",
        lambda *_args, **_kwargs: pytest.fail("a stale confirmation must not apply"),
    )

    with pytest.raises(RevisionRefusal, match="missing, stale, already consumed"):
        adapter.consume_replan_and_execute(
            RevisionConfirmation(
                capability="one-use",
                deck_scope="",
                instruction=_INSTRUCTION,
                expected_fingerprint="f" * 64,
                target=action.target,
            ),
            progress=lambda _label: None,
        )


def test_a_restarted_backend_resumes_the_saved_receipt_without_asking_again(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The receipt on disk is the only thing that survives; it is enough.

    The first process stops after the notes land and before the package is
    built. Everything that held the owner's confirmation in memory — the
    prepared action, its capability, the adapter, the package store — is then
    discarded, and a brand-new backend has to find the work, finish it, and
    hand back the deck without preparing, re-fetching, re-applying, or asking
    for consent a second time.
    """

    config = _config(tmp_path)
    adapter = _adapter(config)
    _install_preparation(monkeypatch, config)
    first = _install_apply_and_build(monkeypatch, config, fail_build=True)
    adapter.bind_package_downloads("http://127.0.0.1:9/first/packages/")
    reply = _prepare(adapter, config, _intent())
    assert reply.action is not None
    fingerprint = reply.action.request_fingerprint
    with pytest.raises(RevisionRefusal):
        adapter.consume_replan_and_execute(
            RevisionConfirmation(
                capability="one-use",
                deck_scope="",
                instruction=_INSTRUCTION,
                expected_fingerprint=fingerprint,
                target=reply.action.target,
            ),
            progress=lambda _label: None,
        )
    assert first["applied"] == ["b" * 64]
    assert first["built"] == []

    # The process is gone: no prepared action, no capability, no adapter, no
    # config object, and no download store carries over.
    del adapter, reply, first
    monkeypatch.setattr(
        character_notes,
        "prepare_character_notes",
        lambda *_args, **_kwargs: pytest.fail("a resume must not prepare or fetch"),
    )
    restarted_config = ProjectConfig.load(tmp_path)
    restarted = _adapter(restarted_config)
    second = _install_apply_and_build(monkeypatch, restarted_config)
    monkeypatch.setattr(
        character_notes,
        "execute_character_notes",
        lambda *_args, **_kwargs: pytest.fail("a resume must not re-apply the notes"),
    )
    store = restarted.bind_package_downloads("http://127.0.0.1:9/second/packages/")
    assert restarted._agent_plans == {}

    choices = restarted.list_kanji_finish_choices()

    assert len(choices) == 1
    assert choices[0].receipt_id == fingerprint
    assert choices[0].state == "applied"
    assert choices[0].deck_name == "Genki II Kanji"
    assert choices[0].characters == "物、特、鳥、料、理"
    assert [option.action for option in choices[0].actions] == ["resume"]
    phases: list[str] = []

    resumed = restarted.resume_kanji_finish(
        receipt_id=choices[0].receipt_id,
        action="resume",
        progress=phases.append,
    )

    assert resumed.complete is True
    assert resumed.receipt_id == fingerprint
    assert resumed.state == "complete"
    assert second["applied"] == []
    assert len(second["built"]) == 1
    assert "Building Anki package" in phases
    assert "[Download genki-ii-kanji.apkg](http://127.0.0.1:9/second/packages/" in (
        resumed.message
    )
    token = resumed.message.split("/second/packages/")[1].split(")")[0]
    assert store.read(token) == ("genki-ii-kanji.apkg", b"one exact anki package")


def test_a_completed_receipt_still_offers_its_package_after_a_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    _install_preparation(monkeypatch, config)
    _install_apply_and_build(monkeypatch, config)
    reply = _prepare(adapter, config, _intent())
    assert reply.action is not None
    plan = adapter._agent_plans[reply.action.request_fingerprint].plan
    completed = kanji_finish.execute_kanji_finish(config, plan)

    del adapter, plan, reply
    restarted = _adapter(ProjectConfig.load(tmp_path))
    restarted.bind_package_downloads("http://127.0.0.1:9/second/packages/")
    monkeypatch.setattr(
        character_notes,
        "execute_character_notes",
        lambda *_args, **_kwargs: pytest.fail("a download must not re-apply"),
    )
    monkeypatch.setattr(
        deck_package,
        "execute_deck_package",
        lambda *_args, **_kwargs: pytest.fail("a download must not rebuild"),
    )

    choices = restarted.list_kanji_finish_choices()
    offered = restarted.resume_kanji_finish(
        receipt_id=completed.receipt_id,
        action="download",
        progress=lambda _label: None,
    )

    assert [option.action for option in choices[0].actions] == ["download"]
    assert choices[0].state == "complete"
    assert offered.complete is True
    assert "[Download genki-ii-kanji.apkg](http://127.0.0.1:9/second/packages/" in (
        offered.message
    )


def test_the_store_only_serves_bytes_a_complete_receipt_still_proves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    _install_preparation(monkeypatch, config)
    seen = _install_apply_and_build(monkeypatch, config)
    store = adapter.bind_package_downloads("http://127.0.0.1:9/token/packages/")
    reply = _prepare(adapter, config, _intent())
    assert reply.action is not None
    plan = adapter._agent_plans[reply.action.request_fingerprint].plan
    result = kanji_finish.execute_kanji_finish(config, plan)

    offer = store.offer(result.receipt_id)

    assert offer.filename == "genki-ii-kanji.apkg"
    assert offer.byte_count == len(b"one exact anki package")
    seen["package"].write_bytes(b"a different package")
    with pytest.raises(
        assistant_packages.AssistantPackageError,
        match="changed after it was finished",
    ):
        store.read(offer.token)
    with pytest.raises(
        assistant_packages.AssistantPackageError, match="unknown or expired"
    ):
        store.read("not-a-token")


def test_an_unfinished_receipt_offers_no_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    _install_preparation(monkeypatch, config)
    _install_apply_and_build(monkeypatch, config)
    store = adapter.bind_package_downloads("http://127.0.0.1:9/token/packages/")
    reply = _prepare(adapter, config, _intent())
    assert reply.action is not None
    plan = adapter._agent_plans[reply.action.request_fingerprint].plan
    def interrupted(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("the package could not be published")

    monkeypatch.setattr(deck_package, "execute_deck_package", interrupted)
    with pytest.raises(kanji_finish.KanjiFinishError):
        kanji_finish.execute_kanji_finish(config, plan)

    with pytest.raises(
        assistant_packages.AssistantPackageError, match="state applied"
    ):
        store.offer(plan.fingerprint)


def test_the_isolated_origin_serves_the_finished_package_once_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    _install_preparation(monkeypatch, config)
    _install_apply_and_build(monkeypatch, config)
    reply = _prepare(adapter, config, _intent())
    assert reply.action is not None
    plan = adapter._agent_plans[reply.action.request_fingerprint].plan
    sidecar = create_assistant_sidecar(
        adapter,
        deck_choices=(),
        session_token="assistant-session-token-000000000000",
    )
    sidecar.start()
    try:
        connection = http.client.HTTPConnection(
            "127.0.0.1",
            sidecar.server.server_address[1],
            timeout=3,
        )
        connection.request("GET", sidecar.server.script_path)
        script = connection.getresponse().read().decode("utf-8")
        connection.close()

        # The recovery entry has to be reachable without a model turn.
        assert RESUME_KANJI_MESSAGE in script

        result = kanji_finish.execute_kanji_finish(config, plan)
        offer = adapter._offer_package(result.receipt_id)
        assert offer is not None
        connection = http.client.HTTPConnection(
            "127.0.0.1",
            sidecar.server.server_address[1],
            timeout=3,
        )
        connection.request(
            "GET", f"{sidecar.server.download_path_prefix}{offer.token}"
        )
        response = connection.getresponse()
        body = response.read()
        disposition = response.getheader("Content-Disposition")
        connection.close()

        assert response.status == 200
        assert body == b"one exact anki package"
        assert disposition == 'attachment; filename="genki-ii-kanji.apkg"'

        connection = http.client.HTTPConnection(
            "127.0.0.1",
            sidecar.server.server_address[1],
            timeout=3,
        )
        connection.request(
            "GET", f"{sidecar.server.download_path_prefix}unknown-token"
        )
        refused = connection.getresponse()
        refused.read()
        connection.close()

        assert refused.status == 404
    finally:
        sidecar.close()


def test_targets_and_destination_may_arrive_in_different_turns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """"These five kanji" then "Genki II Kanji" is one ordinary conversation.

    Requiring every target, the deck name and each direction to appear in the
    latest message makes the obvious two-turn exchange impossible: the owner
    names the characters, Janki asks where they should go, and the reply that
    answers it contains no characters at all. Planning a preview is not write
    authority; the confirmation after it is.
    """

    config = _config(tmp_path)
    adapter = _adapter(config)
    _install_preparation(monkeypatch, config)

    reply = _prepare(
        adapter,
        config,
        _intent(),
        owner_message="Genki II Kanji",
        owner_history=("I want to study these five kanji: 物特鳥料理",),
    )

    assert reply.action is not None


def test_a_direction_chosen_in_an_earlier_turn_needs_no_english_token_now(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    _install_preparation(
        monkeypatch,
        config,
        characters=("物",),
        directions=("recognition", "reading"),
        notes=(_NOTES[0],),
    )

    reply = _prepare(
        adapter,
        config,
        _intent(
            kanji_characters=["物"],
            card_directions=["recognition", "reading"],
        ),
        owner_message="Yes, that is right.",
        owner_history=("Add 物 with recognition and reading cards.",),
    )

    assert reply.action is not None


def test_an_owner_cue_from_an_earlier_turn_is_still_the_owners_cue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    cued = _note("理", ("logic",), production_cue="the one in 料理")
    _install_preparation(
        monkeypatch,
        config,
        characters=("理",),
        directions=("production",),
        notes=(cued,),
    )

    reply = _prepare(
        adapter,
        config,
        _intent(
            kanji_characters=["理"],
            card_directions=["production"],
            production_cues=["理=the one in 料理"],
        ),
        owner_message="Go ahead.",
        owner_history=("Use the cue: the one in 料理",),
    )

    assert reply.action is not None


def test_a_cue_only_the_assistant_ever_said_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Authoring a cue is writing study content, so a model may not do it.

    Everything else in this flow is prompt-led planning the owner confirms
    afterwards. A production cue is the exception: it is content on the card,
    and the only provenance that counts is the owner's own words.
    """

    config = _config(tmp_path)
    adapter = _adapter(config)
    monkeypatch.setattr(
        character_notes,
        "prepare_character_notes",
        lambda *_args, **_kwargs: pytest.fail("an invented cue must refuse first"),
    )

    with pytest.raises(RevisionRefusal, match="production cue"):
        _prepare(
            adapter,
            config,
            _intent(
                kanji_characters=["理"],
                card_directions=["production"],
                production_cues=["理=the one in 料理"],
            ),
            owner_message="Go ahead.",
            # Only Janki ever proposed this wording.
            owner_history=("Add 理 as a production card.",),
        )


def test_a_two_turn_chat_carries_the_owners_earlier_answer_into_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The plumbing, not just the helper: chat history reaches preparation."""

    config = _config(tmp_path)
    adapter = _adapter(config)
    _install_preparation(monkeypatch, config)
    monkeypatch.setattr(
        assistant_adapter,
        "_agent_context",
        lambda *_args, **_kwargs: SimpleNamespace(resource_ids=()),
    )
    monkeypatch.setattr(
        assistant_adapter.assistant_agent,
        "plan_agent",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        assistant_adapter.assistant_agent,
        "run_agent",
        lambda *_args, **_kwargs: _result(_intent()),
    )

    reply = adapter.chat(
        deck_scope="",
        history=(
            ("user", "I want to study these five kanji: 物特鳥料理"),
            ("assistant", "Which deck should they go in?"),
        ),
        message="Genki II Kanji",
        progress=lambda _label: None,
        preview=lambda _text: None,
    )

    assert reply.action is not None, reply.text


def test_the_preview_counts_one_added_note_against_the_whole_deck(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    _install_preparation(
        monkeypatch,
        config,
        characters=("理",),
        notes=(_note("理", ("logic", "reason")),),
        deck_record_ids=(
            "kanji:物",
            "kanji:特",
            "kanji:鳥",
            "kanji:料",
            "kanji:説",
            "kanji:理",
        ),
    )

    reply = _prepare(
        adapter,
        config,
        _intent(kanji_characters=["理"]),
        owner_message="Add 理 to Genki II Kanji.",
    )

    action = reply.action
    assert action is not None
    assert "Add 1 character note" in action.effects[0]
    assert any(
        "6 notes" in effect and "6 cards" in effect for effect in action.effects
    ), action.effects


def test_the_routine_preview_shows_learner_text_not_internal_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hashes and store paths stay bound in the projection, not on the card.

    The owner is deciding whether to make five cards, not auditing a digest.
    The exact bindings are still recorded — this only keeps them off the
    routine confirmation.
    """

    config = _config(tmp_path)
    adapter = _adapter(config)
    _install_preparation(monkeypatch, config)

    reply = _prepare(adapter, config, _intent())

    action = reply.action
    assert action is not None
    rendered = "\n".join((*action.effects, *action.disclosures))
    assert "SHA-256" not in rendered
    assert "data/kanji_notes.json" not in rendered
    assert "fingerprint" not in rendered.casefold()
    assert action.request_fingerprint not in rendered
    assert "物" in rendered
    assert "Genki II Kanji" in rendered
    # The bindings are still there, in the record that is kept.
    plan = adapter._agent_plans[action.request_fingerprint].plan
    assert plan.projection["inputs"]["plan_sha256"]
    assert plan.projection["writes"]["character_notes"] == "data/kanji_notes.json"


def test_the_completion_message_reads_like_a_finished_deck_not_a_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What landed and where to get it; the bindings stay on the receipt."""

    config = _config(tmp_path)
    adapter = _adapter(config)
    _install_preparation(monkeypatch, config)
    _install_apply_and_build(monkeypatch, config)
    adapter.bind_package_downloads("http://127.0.0.1:9/token/packages/")
    reply = _prepare(adapter, config, _intent())
    action = reply.action
    assert action is not None

    execution = adapter.consume_replan_and_execute(
        RevisionConfirmation(
            capability="one-use",
            deck_scope="",
            instruction=_INSTRUCTION,
            expected_fingerprint=action.request_fingerprint,
            target=action.target,
        ),
        progress=lambda _label: None,
    )

    assert "Genki II Kanji" in execution.message
    assert "物、特、鳥、料、理" in execution.message
    assert "Download" in execution.message
    assert "SHA-256" not in execution.message
    assert action.request_fingerprint not in execution.message
    assert "Finish receipt" not in execution.message
    # The receipt still records every binding this message leaves out.
    receipt = kanji_finish.list_kanji_finishes(config)[0]
    assert receipt.package_sha256
    assert receipt.receipt_id == action.request_fingerprint


def test_an_existing_character_deck_discloses_only_the_notes_it_ships(
    tmp_path: Path,
) -> None:
    """A deck's context is the deck, not every character in the store.

    The store is shared by every character deck. Disclosing all of it under
    one deck's resource would tell the model that deck ships notes it does
    not, and grow with every unrelated addition.
    """

    config = _config(tmp_path)
    kanji_notes.save_notes(
        config.kanji_notes_file,
        {
            "理": _note("理", ("logic",)),
            "説": _note("説", ("explanation",)),
        },
    )
    config.deck_dir.mkdir(parents=True, exist_ok=True)
    deck = config.deck_dir / "genki-ii-kanji.yaml"
    deck.write_text(
        "deck:\n"
        "  name: Genki II Kanji\n"
        "  kind: kanji\n"
        "  deck_id: 1500000002\n"
        "  model_id: 1500000102\n"
        "  include_ids: [kanji:理]\n",
        encoding="utf-8",
    )
    broker = assistant_adapter.assistant_context.AssistantContextBroker(config)

    context = broker.deck_context(broker.resource_id_for_deck(deck))

    disclosed = json.loads(context.disclosure.wire)["data"]["character_notes"]
    assert [item["character"] for item in disclosed] == ["理"]


def test_each_character_deck_discloses_the_store_it_actually_reads(
    tmp_path: Path,
) -> None:
    """A deck's `source:` decides which notes it ships, so it decides this too.

    Filtering the project's default store by a deck's ids answers the wrong
    question for a deck configured to read another store: it would disclose
    notes that deck does not ship, and a count a build of it never reports.
    """

    config = _config(tmp_path)
    kanji_notes.save_notes(
        config.kanji_notes_file,
        {"理": _note("理", ("logic",)), "説": _note("説", ("explanation",))},
    )
    alternate = config.root / "data/genki-kanji-notes.json"
    kanji_notes.save_notes(
        alternate,
        {"物": _note("物", ("thing",)), "鳥": _note("鳥", ("bird",))},
    )
    config.deck_dir.mkdir(parents=True, exist_ok=True)
    default_deck = config.deck_dir / "default-store-kanji.yaml"
    default_deck.write_text(
        "deck:\n"
        "  name: Default store kanji\n"
        "  kind: kanji\n"
        "  deck_id: 1500000002\n"
        "  model_id: 1500000102\n"
        "  include_ids: [kanji:説]\n",
        encoding="utf-8",
    )
    alternate_deck = config.deck_dir / "alternate-store-kanji.yaml"
    alternate_deck.write_text(
        "deck:\n"
        "  name: Alternate store kanji\n"
        "  kind: kanji\n"
        "  deck_id: 1500000003\n"
        "  model_id: 1500000103\n"
        "  source: ../genki-kanji-notes.json\n"
        "  include_ids: [kanji:物, kanji:鳥]\n",
        encoding="utf-8",
    )
    broker = assistant_adapter.assistant_context.AssistantContextBroker(config)

    default_context = broker.deck_context(broker.resource_id_for_deck(default_deck))
    alternate_context = broker.deck_context(
        broker.resource_id_for_deck(alternate_deck)
    )

    default_data = json.loads(default_context.disclosure.wire)["data"]
    alternate_data = json.loads(alternate_context.disclosure.wire)["data"]
    assert [item["character"] for item in default_data["character_notes"]] == ["説"]
    assert [item["character"] for item in alternate_data["character_notes"]] == [
        "物",
        "鳥",
    ]
    assert default_context.disclosure.item_count == 1
    assert alternate_context.disclosure.item_count == 2


def test_a_symlinked_character_store_refuses_before_it_is_opened(
    tmp_path: Path,
) -> None:
    """The store a character deck reads passes the same guard as any source."""

    config = _config(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside-notes.json"
    kanji_notes.save_notes(outside, {"理": _note("理", ("logic",))})
    config.kanji_notes_file.parent.mkdir(parents=True, exist_ok=True)
    config.kanji_notes_file.symlink_to(outside)
    config.deck_dir.mkdir(parents=True, exist_ok=True)
    deck = config.deck_dir / "genki-ii-kanji.yaml"
    deck.write_text(
        "deck:\n"
        "  name: Genki II Kanji\n"
        "  kind: kanji\n"
        "  deck_id: 1500000002\n"
        "  model_id: 1500000102\n"
        "  include_ids: [kanji:理]\n",
        encoding="utf-8",
    )
    broker = assistant_adapter.assistant_context.AssistantContextBroker(config)

    with pytest.raises(
        assistant_adapter.assistant_context.AssistantContextError, match="symlink"
    ):
        broker.deck_context(broker.resource_id_for_deck(deck))
