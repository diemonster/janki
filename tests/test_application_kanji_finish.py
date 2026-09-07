"""One durable receipt owns the confirmed character-note apply and build.

The interesting property is recovery: a batch interrupted after the notes
landed must finish from its receipt without writing them twice, without
preparing again, and without a new owner decision.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from test_application_assistant_kanji_notes import _note

from japanese_anki import kanji_notes
from japanese_anki.application import (
    assistant_kanji_notes,
    character_notes,
    deck_package,
    kanji_finish,
)
from japanese_anki.config import ProjectConfig


def _project(tmp_path: Path) -> ProjectConfig:
    (tmp_path / "janki.toml").write_text(
        "[paths]\n"
        'deck_dir = "data/decks"\n'
        'dist_dir = "dist"\n'
        'staging_dir = "data/staging"\n'
        'kanji_notes_file = "data/kanji_notes.json"\n',
        encoding="utf-8",
    )
    return ProjectConfig.load(tmp_path)


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


@dataclass(frozen=True)
class _Applied:
    deck_path: Path
    output_path: Path
    record_ids: tuple[str, ...]
    note_count: int
    card_count: int
    changed: bool = True


@dataclass(frozen=True)
class _BuildPlan:
    deck_path: Path
    output_path: Path
    fingerprint: str = "c" * 64


@dataclass(frozen=True)
class _Built:
    output_path: Path
    package_sha256: str = "d" * 64
    note_count: int = 5
    card_count: int = 5


def _install_verifier(
    monkeypatch: pytest.MonkeyPatch,
    verified: list[Any],
    *,
    refuse: str = "",
) -> None:
    """Stand in for the core's exact applied-state check.

    The real one reads the applied files, which these tests never write: the
    apply is stubbed too. What is under test here is the *ordering* — plan the
    package, verify, then publish — and that a refusal stops the build.
    ``test_the_core_exposes_the_applied_state_verifier`` holds the call itself
    to the core's real API.
    """

    def verify(_config: ProjectConfig, plan: Any) -> None:
        verified.append(plan)
        if refuse:
            raise character_notes.CharacterNotesError(refuse)

    monkeypatch.setattr(character_notes, "verify_character_notes_applied", verify)


def test_the_core_exposes_the_applied_state_verifier() -> None:
    """The build's after-state check is a core call, not a local re-derivation.

    ``_build`` orders package planning, this verification, and publication so
    an edit between them refuses. Standing that call in elsewhere is fine;
    shipping without the real one would mean the ordering guards nothing.
    """

    assert callable(character_notes.verify_character_notes_applied)


_FIVE = ("物", "特", "鳥", "料", "理")


def _service_plan(
    config: ProjectConfig,
    *,
    characters: tuple[str, ...] = _FIVE,
    deck_record_ids: tuple[str, ...] | None = None,
) -> _Plan:
    notes = tuple(
        _note(character, (f"{character} meaning",)) for character in characters
    )
    return _Plan(
        deck_path=config.deck_dir / "genki-ii-kanji.yaml",
        output_path=config.dist_dir / "genki-ii-kanji.apkg",
        deck_name="Genki II Kanji",
        characters=characters,
        directions=("recognition",),
        notes=notes,
        deck_record_ids=(
            tuple(note.id for note in notes)
            if deck_record_ids is None
            else deck_record_ids
        ),
    )


def _finish_plan(
    config: ProjectConfig,
    monkeypatch: pytest.MonkeyPatch,
    *,
    characters: tuple[str, ...] = _FIVE,
    deck_record_ids: tuple[str, ...] | None = None,
) -> kanji_finish.KanjiFinishPlan:
    service = _service_plan(
        config, characters=characters, deck_record_ids=deck_record_ids
    )
    monkeypatch.setattr(
        character_notes,
        "prepare_character_notes",
        lambda *_args, **_kwargs: service,
    )
    monkeypatch.setattr(character_notes, "CharacterNotesPlan", _Plan)
    notes_plan = assistant_kanji_notes.plan_kanji_notes(
        config,
        assistant_kanji_notes.AssistantKanjiNotesRequest(
            characters=characters,
            deck_path=None,
            deck_name="Genki II Kanji",
            directions=("recognition",),
            refresh_readings=False,
            production_cues=(),
            instruction="Add character notes for these five kanji.",
        ),
    )
    return kanji_finish.plan_kanji_finish(config, notes_plan)


def _install_apply(
    monkeypatch: pytest.MonkeyPatch,
    config: ProjectConfig,
    applied: list[str],
) -> None:
    def execute(
        _config: ProjectConfig,
        plan: Any,
        *,
        expected_fingerprint: str,
    ) -> _Applied:
        applied.append(expected_fingerprint)
        return _Applied(
            deck_path=config.deck_dir / "genki-ii-kanji.yaml",
            output_path=config.dist_dir / "genki-ii-kanji.apkg",
            record_ids=tuple(f"kanji:{character}" for character in plan.characters),
            note_count=plan.note_count,
            card_count=plan.card_count,
        )

    monkeypatch.setattr(character_notes, "execute_character_notes", execute)


def _install_build(
    monkeypatch: pytest.MonkeyPatch,
    built: list[Path],
    *,
    fail: bool = False,
) -> None:
    def plan_package(config: ProjectConfig, deck_path: Path, **_kwargs: Any) -> _BuildPlan:
        return _BuildPlan(
            deck_path=Path(deck_path).absolute(),
            output_path=(config.dist_dir / "genki-ii-kanji.apkg").absolute(),
        )

    def execute_package(_config: ProjectConfig, plan: _BuildPlan) -> _Built:
        if fail:
            raise OSError("the package could not be published")
        built.append(plan.output_path)
        return _Built(output_path=plan.output_path)

    monkeypatch.setattr(deck_package, "plan_deck_package", plan_package)
    monkeypatch.setattr(deck_package, "execute_deck_package", execute_package)


def test_one_confirmation_applies_the_notes_and_builds_under_one_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _project(tmp_path)
    plan = _finish_plan(config, monkeypatch)
    applied: list[str] = []
    built: list[Path] = []
    _install_apply(monkeypatch, config, applied)
    _install_build(monkeypatch, built)
    _install_verifier(monkeypatch, [])
    phases: list[str] = []

    result = kanji_finish.execute_kanji_finish(
        config,
        plan,
        progress=phases.append,
    )

    assert applied == ["b" * 64]
    assert len(built) == 1
    assert result.succeeded
    assert result.receipt_id == plan.fingerprint
    assert result.package_sha256 == "d" * 64
    assert result.card_count == 5
    assert result.record_ids == tuple(f"kanji:{item}" for item in _FIVE)
    assert phases == [
        "Preparing finish",
        "Writing character notes",
        "Saving finish receipt",
        "Building Anki package",
        "Saving finish receipt",
    ]
    record = json.loads(plan.record_path.read_text(encoding="utf-8"))
    assert record["state"] == "complete"
    assert record["kind"] == "kanji_finish"
    assert plan.paid_provider_calls == 0


def test_an_interrupted_build_resumes_without_writing_the_notes_again(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The receipt, not a fresh owner decision, owns interrupted work."""

    config = _project(tmp_path)
    plan = _finish_plan(config, monkeypatch)
    applied: list[str] = []
    built: list[Path] = []
    _install_apply(monkeypatch, config, applied)
    _install_build(monkeypatch, built, fail=True)
    _install_verifier(monkeypatch, [])

    with pytest.raises(kanji_finish.KanjiFinishError, match="Could not build"):
        kanji_finish.execute_kanji_finish(config, plan)

    assert applied == ["b" * 64]
    assert kanji_finish.inspect_kanji_finish(config, plan.fingerprint).state == "applied"

    monkeypatch.setattr(
        character_notes,
        "execute_character_notes",
        lambda *_args, **_kwargs: pytest.fail("a resume must not write the notes again"),
    )
    monkeypatch.setattr(
        character_notes,
        "prepare_character_notes",
        lambda *_args, **_kwargs: pytest.fail("a resume must not prepare again"),
    )
    _install_build(monkeypatch, built)

    resumed = kanji_finish.resume_kanji_finish(config, plan.fingerprint)

    assert resumed.succeeded
    assert resumed.receipt_id == plan.fingerprint
    assert len(built) == 1


def test_reconfirming_the_identical_batch_reuses_its_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reopened thread that confirms the same batch repeats no work."""

    config = _project(tmp_path)
    plan = _finish_plan(config, monkeypatch)
    applied: list[str] = []
    built: list[Path] = []
    _install_apply(monkeypatch, config, applied)
    _install_build(monkeypatch, built)
    _install_verifier(monkeypatch, [])
    first = kanji_finish.execute_kanji_finish(config, plan)

    again = kanji_finish.execute_kanji_finish(config, plan)

    assert applied == ["b" * 64]
    assert len(built) == 1
    assert again.receipt_id == first.receipt_id
    assert again.package_sha256 == first.package_sha256


def test_a_saved_plan_whose_fingerprint_changed_refuses_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _project(tmp_path)
    plan = _finish_plan(config, monkeypatch)

    class _Drifted(_Plan):
        @classmethod
        def from_dict(cls, value: dict[str, Any]) -> _Plan:
            restored = _Plan.from_dict(value)
            return _Plan(
                deck_path=restored.deck_path,
                output_path=restored.output_path,
                deck_name=restored.deck_name,
                characters=restored.characters,
                directions=restored.directions,
                notes=restored.notes,
                deck_record_ids=restored.deck_record_ids,
                fingerprint="e" * 64,
            )

    monkeypatch.setattr(character_notes, "CharacterNotesPlan", _Drifted)
    monkeypatch.setattr(
        character_notes,
        "execute_character_notes",
        lambda *_args, **_kwargs: pytest.fail("a drifted plan must not be applied"),
    )

    with pytest.raises(
        kanji_finish.KanjiFinishError,
        match="no longer matches its authorized",
    ):
        kanji_finish.execute_kanji_finish(config, plan)

    assert kanji_finish.inspect_kanji_finish(config, plan.fingerprint).state == (
        "authorized"
    )


def test_a_deck_that_now_builds_elsewhere_refuses_rather_than_publishing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _project(tmp_path)
    plan = _finish_plan(config, monkeypatch)
    applied: list[str] = []
    _install_apply(monkeypatch, config, applied)
    _install_verifier(monkeypatch, [])
    monkeypatch.setattr(
        deck_package,
        "plan_deck_package",
        lambda config, deck_path, **_kwargs: _BuildPlan(
            deck_path=Path(deck_path).absolute(),
            output_path=(config.dist_dir / "somewhere-else.apkg").absolute(),
        ),
    )
    monkeypatch.setattr(
        deck_package,
        "execute_deck_package",
        lambda *_args, **_kwargs: pytest.fail("an unbound package must not publish"),
    )

    with pytest.raises(
        kanji_finish.KanjiFinishError,
        match="different package than the confirmed batch",
    ):
        kanji_finish.execute_kanji_finish(config, plan)


def test_the_build_verifies_the_applied_state_before_publishing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A package plan captured before the notes changed must not publish.

    The build reads the deck and its store, so between planning the package
    and writing it the applied state could have moved. The core's exact
    check runs inside that window, against the same saved plan the owner
    confirmed, and the package executor's own captured inputs close it.
    """

    config = _project(tmp_path)
    plan = _finish_plan(config, monkeypatch)
    applied: list[str] = []
    built: list[Path] = []
    _install_apply(monkeypatch, config, applied)
    _install_build(monkeypatch, built)
    verified: list[Any] = []
    _install_verifier(
        monkeypatch,
        verified,
        refuse="the character notes on disk are not the ones this batch applied",
    )

    with pytest.raises(kanji_finish.KanjiFinishError, match="not the ones this batch"):
        kanji_finish.execute_kanji_finish(config, plan)

    assert len(verified) == 1
    assert verified[0].fingerprint == "b" * 64
    assert built == []
    assert kanji_finish.inspect_kanji_finish(config, plan.fingerprint).state == "applied"


def test_a_resume_verifies_and_builds_without_preparing_or_reapplying(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _project(tmp_path)
    plan = _finish_plan(config, monkeypatch)
    applied: list[str] = []
    built: list[Path] = []
    _install_apply(monkeypatch, config, applied)
    _install_build(monkeypatch, built, fail=True)
    _install_verifier(monkeypatch, [])

    with pytest.raises(kanji_finish.KanjiFinishError):
        kanji_finish.execute_kanji_finish(config, plan)

    monkeypatch.setattr(
        character_notes,
        "prepare_character_notes",
        lambda *_args, **_kwargs: pytest.fail("a resume must not prepare"),
    )
    monkeypatch.setattr(
        character_notes,
        "execute_character_notes",
        lambda *_args, **_kwargs: pytest.fail("a resume must not re-apply"),
    )
    _install_build(monkeypatch, built)
    verified: list[Any] = []
    _install_verifier(monkeypatch, verified)

    resumed = kanji_finish.resume_kanji_finish(config, plan.fingerprint)

    assert resumed.succeeded
    assert applied == ["b" * 64]
    assert len(built) == 1
    assert [item.fingerprint for item in verified] == ["b" * 64]


def test_the_package_is_checked_against_the_whole_deck_not_the_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adding one note to five builds six; a check against one would refuse."""

    config = _project(tmp_path)
    plan = _finish_plan(
        config,
        monkeypatch,
        characters=("理",),
        deck_record_ids=(
            "kanji:物",
            "kanji:特",
            "kanji:鳥",
            "kanji:料",
            "kanji:説",
            "kanji:理",
        ),
    )
    _install_apply(monkeypatch, config, [])
    _install_verifier(monkeypatch, [])

    def plan_package(inner: ProjectConfig, deck_path: Path, **_kwargs: Any) -> Any:
        return _BuildPlan(
            deck_path=Path(deck_path).absolute(),
            output_path=(inner.dist_dir / "genki-ii-kanji.apkg").absolute(),
        )

    def execute_package(_config: ProjectConfig, package: Any) -> Any:
        return _Built(
            output_path=package.output_path,
            note_count=6,
            card_count=6,
        )

    monkeypatch.setattr(deck_package, "plan_deck_package", plan_package)
    monkeypatch.setattr(deck_package, "execute_deck_package", execute_package)

    result = kanji_finish.execute_kanji_finish(config, plan)

    assert result.succeeded
    assert result.note_count == 6
    assert result.card_count == 6
    assert plan.note_count == 1
    assert plan.deck_note_count == 6


def test_a_package_holding_a_different_count_than_the_deck_refuses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _project(tmp_path)
    plan = _finish_plan(config, monkeypatch)
    _install_apply(monkeypatch, config, [])
    _install_verifier(monkeypatch, [])

    def plan_package(inner: ProjectConfig, deck_path: Path, **_kwargs: Any) -> Any:
        return _BuildPlan(
            deck_path=Path(deck_path).absolute(),
            output_path=(inner.dist_dir / "genki-ii-kanji.apkg").absolute(),
        )

    monkeypatch.setattr(deck_package, "plan_deck_package", plan_package)
    monkeypatch.setattr(
        deck_package,
        "execute_deck_package",
        lambda _config, package: _Built(
            output_path=package.output_path,
            note_count=4,
            card_count=4,
        ),
    )

    with pytest.raises(kanji_finish.KanjiFinishError, match="4 note"):
        kanji_finish.execute_kanji_finish(config, plan)


# --- finding the interrupted batch again -------------------------------------

#: The listing's cap, written out here rather than imported: a test that read
#: the limit out of the module under test would pass against any limit at all.
LIST_LIMIT = 20


def _authority(instruction: str) -> dict[str, Any]:
    return {
        "version": 1,
        "instruction": instruction,
        "notes": {
            "target": {"deck_name": "Genki II Kanji", "characters": ["理"]}
        },
        "plan": {"sha256": "e" * 64, "fingerprint": "f" * 64, "wire": "{}"},
        "build": {
            "deck_path": "data/decks/genki-ii-kanji.yaml",
            "output_path": "dist/genki-ii-kanji.apkg",
            "deck_state": "existing",
            "deck_note_count": 1,
            "deck_card_count": 1,
        },
    }


def _receipt_id(instruction: str) -> str:
    """The id the module derives from one authority, so a test can order by it."""

    wire = json.dumps(
        _authority(instruction),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(wire.encode("utf-8")).hexdigest()


def _write_receipt(
    config: ProjectConfig,
    instruction: str,
    *,
    state: str,
    updated_at: str,
) -> str:
    """One durable receipt on disk, in a state the module recognises."""

    authority = _authority(instruction)
    receipt_id = _receipt_id(instruction)
    record = {
        "schema_version": 1,
        "kind": "kanji_finish",
        "receipt_id": receipt_id,
        "state": state,
        "authority": authority,
        "authorized_at": "2026-09-01T00:00:00+00:00",
        "updated_at": updated_at,
        "apply_receipt": (
            {"record_ids": ["kanji:理"]} if state in {"applied", "complete"} else None
        ),
        "build_receipt": (
            {"note_count": 1, "card_count": 1, "package_sha256": "a" * 64}
            if state == "complete"
            else None
        ),
    }
    directory = config.staging_dir / "done" / "kanji"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"kanji-finish-{receipt_id}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return receipt_id


def test_unfinished_work_is_listed_however_many_completed_receipts_precede_it(
    tmp_path: Path,
) -> None:
    """The list exists to find the interrupted batch, so it never sorts by name.

    Receipt filenames carry a content hash, which orders them arbitrarily. A
    cap applied to that order hides whatever happens to sort late — and after
    twenty completed batches that is permanent: the owner's one recorded
    authority to resume becomes unreachable from the surface built to offer it.
    """

    config = _project(tmp_path)
    instructions = sorted(
        (f"Add character notes, batch {index}." for index in range(25)),
        key=_receipt_id,
    )
    # The one receipt filename sorting would drop, and the only unfinished one.
    hidden = instructions[-1]
    completed = instructions[:-1]
    assert len(completed) > LIST_LIMIT
    for index, instruction in enumerate(completed):
        _write_receipt(
            config,
            instruction,
            state="complete",
            updated_at=f"2026-09-02T00:{index:02d}:00+00:00",
        )
    _write_receipt(
        config, hidden, state="applied", updated_at="2026-09-01T00:00:00+00:00"
    )

    found = kanji_finish.list_kanji_finishes(config)

    assert len(found) == LIST_LIMIT
    # Unfinished first, even though it is both the oldest and the last by name.
    assert found[0].receipt_id == _receipt_id(hidden)
    assert found[0].state == "applied"
    assert not found[0].succeeded
    # Then the completed ones, most recently updated first.
    assert [item.receipt_id for item in found[1:]] == [
        _receipt_id(instruction) for instruction in reversed(completed[-19:])
    ]


def test_an_unreadable_sibling_does_not_hide_the_receipt_beside_it(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)
    receipt_id = _write_receipt(
        config,
        "Add character notes for these five kanji.",
        state="authorized",
        updated_at="2026-09-02T00:00:00+00:00",
    )
    directory = config.staging_dir / "done" / "kanji"
    (directory / f"kanji-finish-{'b' * 64}.json").write_text(
        "{not json at all", encoding="utf-8"
    )

    found = kanji_finish.list_kanji_finishes(config)

    assert [item.receipt_id for item in found] == [receipt_id]


@pytest.mark.parametrize("limit", [0, -1, -20])
def test_a_non_positive_limit_asks_for_no_receipts(tmp_path: Path, limit: int) -> None:
    """Rather than slicing from the end, which returns an arbitrary subset."""

    config = _project(tmp_path)
    for index in range(3):
        _write_receipt(
            config,
            f"Add character notes, batch {index}.",
            state="authorized",
            updated_at=f"2026-09-02T00:{index:02d}:00+00:00",
        )

    assert kanji_finish.list_kanji_finishes(config, limit=limit) == ()


def test_a_limit_that_is_not_a_whole_number_refuses(tmp_path: Path) -> None:
    config = _project(tmp_path)

    with pytest.raises(kanji_finish.KanjiFinishError, match="integer"):
        kanji_finish.list_kanji_finishes(config, limit="20")  # type: ignore[arg-type]
