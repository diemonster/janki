"""Reading an answer somebody already paid for, out of the bytes that were kept.

Every capture here is synthetic and every provider is faked: nothing in this
file makes a call, and nothing in it is about what Claude answers. What is
under test is janki's own half — that the closed list of envelope shapes is
exactly four, that a shape outside it refuses with a pointer instead of being
repaired, that a valid successful terminal always wins, that two proposals in
one capture are the owner's choice rather than the reader's, and that every
binding the ordinary batch recovery revalidates is revalidated here too.

The template is the first remedy, not this reader: the shared decoder-prevention
paragraph in the three extraction prompts is what stops a wrapper object or a
JSON-string argument being produced at all. Its presence is asserted here
because that paragraph exists for this failure.
"""

from __future__ import annotations

import copy
import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from test_revision_provider import FakeClaudeRunner, _which
from test_workbench_fixtures import RESPONSES

from conftest import seed_prompts
from japanese_anki import card_preview, claude_client, extract, operations, prompts
from japanese_anki.application import capture_recovery, extraction, extraction_batch
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENARIO = "table_exhaustive"
STREAM = Path(__file__).parent / "fixtures" / "claude-code-stream.ndjson"

needs_preview = pytest.mark.skipif(
    card_preview.preview_unavailable() is not None,
    reason=str(card_preview.preview_unavailable()),
)


# --- synthetic captures ---------------------------------------------------------


def _answer(**overrides: Any) -> dict[str, Any]:
    """One fixture answer in the shape the extraction schema validates."""
    value = json.loads((RESPONSES / f"{SCENARIO}.json").read_text(encoding="utf-8"))
    value.update(overrides)
    return value


def _frames() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in STREAM.read_bytes().splitlines()
        if line.strip()
    ]


def _tool_frame(frames: Sequence[dict[str, Any]]) -> int:
    for index, frame in enumerate(frames):
        if frame.get("type") != "assistant":
            continue
        for block in frame["message"]["content"]:
            if block.get("type") == "tool_use":
                return index
    raise AssertionError("the stream fixture has no assembled tool_use block")


def _stream(
    *,
    tool_arguments: Sequence[Any] = (),
    terminal: Any | None = None,
    tool_name: str = "StructuredOutput",
) -> bytes:
    """A Claude Code capture carrying exactly the tool arguments asked for.

    The fixture's own assembled ``tool_use`` frame is the template: replacing
    it keeps every surrounding frame — the partial ``stream_event`` deltas
    included — exactly as a real capture holds them.
    """
    frames = _frames()
    index = _tool_frame(frames)
    template = frames[index]
    replacements = []
    for position, argument in enumerate(tool_arguments, start=1):
        frame = copy.deepcopy(template)
        block = frame["message"]["content"][0]
        block["input"] = argument
        block["id"] = f"toolu_{position:02d}"
        block["name"] = tool_name
        replacements.append(frame)
    frames[index : index + 1] = replacements
    for frame in frames:
        if frame.get("type") == "result":
            if terminal is None:
                frame.pop("structured_output", None)
            else:
                frame["structured_output"] = terminal
    return b"".join(
        json.dumps(frame, ensure_ascii=False).encode("utf-8") + b"\n"
        for frame in frames
    )


# --- a scratch repository -------------------------------------------------------


def _no_api(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the Anthropic API was reached")

    monkeypatch.setattr(claude_client, "prepare_paid_client", refuse)
    monkeypatch.setattr(claude_client, "parse_call", refuse)


def _project(tmp_path: Path) -> ProjectConfig:
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
        'operations_file = "operations.json"\n'
        "[ai]\n"
        'extract_provider = "claude-code"\n'
        'extract_model = "claude-opus-5"\n',
        encoding="utf-8",
    )
    shutil.copytree(
        PROJECT_ROOT / "templates", tmp_path / "templates", dirs_exist_ok=True
    )
    (tmp_path / "decks").mkdir(exist_ok=True)
    (tmp_path / "media").mkdir(exist_ok=True)
    (tmp_path / "vocabulary.json").write_text("[]", encoding="utf-8")
    return ProjectConfig.load(tmp_path)


def _source(config: ProjectConfig, name: str = "one.pdf", body: bytes = b"one") -> Path:
    path = config.scan_inbox / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.7 " + body)
    return path


def _deck(config: ProjectConfig, name: str = "lesson") -> Path:
    path = config.deck_dir / f"{name}.yaml"
    path.write_text(
        "deck:\n"
        f"  name: {name.title()} deck\n"
        "  source: ../vocabulary.json\n"
        "  include_tags: [lesson-intake]\n"
        "  intake_tag: lesson-intake\n"
        "  cards:\n"
        "    recognition: true\n"
        "    production: true\n",
        encoding="utf-8",
    )
    return path


def _captured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    reply: bytes,
    destination_deck: Path | None = None,
) -> tuple[ProjectConfig, extraction_batch.ExtractionBatchPlan, str]:
    """Run one real batch whose single child's reply cannot be decoded.

    The capture hook fires before anything parses, so a reply the ordinary
    decoder refuses still leaves the operation at ``result_captured`` with the
    exact paid bytes on disk. That is the state this whole module is about.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    source = _source(config)
    runner = FakeClaudeRunner(reply=reply)
    plan = extraction_batch.plan_extraction_batch(
        config,
        [source],
        destination_deck=destination_deck,
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
    )
    extraction_batch.dispatch_extraction_batch(
        config,
        plan,
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
        provider_spawn=runner.spawn,
    )
    return config, plan, plan.children[0].operation_id


def _state(config: ProjectConfig, operation_id: str) -> str:
    journal = operations.OperationJournal.load(config.operations_file)
    return journal.operations[operation_id].state


# --- the four shapes, and nothing else ------------------------------------------


SHAPES = {
    "result_structured_output": lambda answer: None,
    "tool_use_input": lambda answer: answer,
    "tool_use_input_wrapped": lambda answer: {"input": answer},
    "tool_use_input_json_string": lambda answer: json.dumps(
        answer, ensure_ascii=False
    ),
}


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_every_enumerated_envelope_shape_stages_through_the_ordinary_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shape: str
) -> None:
    """Each of the four lossless shapes reaches staging, and says which it was.

    Mutant: drop shape 3's sole-key check so ``{"input": ..., "extra": ...}``
    is accepted.
    """
    answer = _answer()
    if shape == "result_structured_output":
        # A valid terminal decodes normally, so the way to leave one captured
        # is to make the staging write fail once, exactly as a full disk does.
        real_write = extraction.write_staging_under_lock
        calls = {"count": 0}

        def refuse_first(path: Path, *args: Any, **kwargs: Any) -> Any:
            calls["count"] += 1
            if calls["count"] == 1:
                raise JankiError("the disk went away")
            return real_write(path, *args, **kwargs)

        monkeypatch.setattr(extraction, "write_staging_under_lock", refuse_first)
        config, _plan, operation_id = _captured(
            tmp_path, monkeypatch, reply=_stream(terminal=answer)
        )
        monkeypatch.setattr(extraction, "write_staging_under_lock", real_write)
    else:
        config, _plan, operation_id = _captured(
            tmp_path,
            monkeypatch,
            reply=_stream(tool_arguments=[SHAPES[shape](answer)]),
        )

    assert _state(config, operation_id) == "result_captured"
    plan = capture_recovery.inspect_capture_proposals(config, operation_id)
    assert plan.valid_group_count == 1
    assert [
        location.envelope_shape
        for group in plan.groups
        for location in group.locations
    ] == [shape]

    outcome = capture_recovery.stage_capture_proposal(config, operation_id)

    assert _state(config, operation_id) == "committed"
    assert outcome.target.exists()
    _records, meta = __import__(
        "japanese_anki.staging", fromlist=["read_staging"]
    ).read_staging(outcome.target)
    recorded = meta["capture_recovery"]
    assert recorded["envelope_shape"] == shape
    assert recorded["capture_sha256"] == plan.capture_sha256
    assert recorded["proposal_sha256"] == plan.groups[0].proposal_sha256
    assert recorded["locations"] == [
        {
            "frame_index": recorded["frame_index"],
            "block_index": recorded["block_index"],
            "tool_use_id": recorded["tool_use_id"],
            "json_pointer": recorded["json_pointer"],
            "envelope_shape": shape,
        }
    ]


def test_a_wrapper_carrying_another_key_beside_input_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Shape 3 is *sole* key ``input``. Any other key set is outside the list.

    Mutant: drop shape 3's sole-key check.
    """
    config, _plan, operation_id = _captured(
        tmp_path,
        monkeypatch,
        reply=_stream(
            tool_arguments=[{"input": _answer(), "extra": "commentary"}]
        ),
    )

    plan = capture_recovery.inspect_capture_proposals(config, operation_id)

    assert plan.valid_group_count == 0
    with pytest.raises(JankiError) as caught:
        capture_recovery.stage_capture_proposal(config, operation_id)
    assert "/message/content/0/input" in str(caught.value)
    assert _state(config, operation_id) == "result_captured"


def test_a_json_string_argument_with_trailing_garbage_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Shape 4 is a *total* parse. A valid prefix is not a valid answer.

    Mutant: use ``json.JSONDecoder().raw_decode`` and ignore the tail.
    """
    payload = json.dumps(_answer(), ensure_ascii=False) + " trailing"
    config, _plan, operation_id = _captured(
        tmp_path, monkeypatch, reply=_stream(tool_arguments=[payload])
    )
    before = config.operations_file.read_bytes()

    plan = capture_recovery.inspect_capture_proposals(config, operation_id)

    assert plan.valid_group_count == 0
    assert "/message/content/0/input" in plan.groups[0].schema_verdict
    with pytest.raises(JankiError):
        capture_recovery.stage_capture_proposal(config, operation_id)
    assert config.operations_file.read_bytes() == before


@pytest.mark.parametrize(
    ("kind", "mutate"),
    (
        ("extra property", lambda value: {**value, "invented": "field"}),
        (
            "missing source_kind",
            lambda value: {
                **value,
                "candidates": [
                    {
                        key: entry
                        for key, entry in candidate.items()
                        if key != "source_kind"
                    }
                    for candidate in value["candidates"]
                ],
            },
        ),
    ),
)
def test_an_unknown_field_or_missing_fact_refuses_with_its_exact_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    mutate: Any,
) -> None:
    """No unknown-field dropping and no fact filling. A pointer, and a refusal.

    Mutant: strip unknown keys before validating against
    ``extract.candidate_schema()``.
    """
    config, _plan, operation_id = _captured(
        tmp_path, monkeypatch, reply=_stream(tool_arguments=[mutate(_answer())])
    )

    plan = capture_recovery.inspect_capture_proposals(config, operation_id)

    assert plan.valid_group_count == 0, kind
    verdict = plan.groups[0].schema_verdict
    assert "/message/content/0/input" in verdict
    with pytest.raises(JankiError) as caught:
        capture_recovery.stage_capture_proposal(config, operation_id)
    assert "/message/content/0/input" in str(caught.value)
    assert not (config.staging_dir / "one.pdf.yaml").exists()


def test_a_partial_streaming_tool_block_is_not_a_proposal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The assembled block is the answer; a delta is not one yet.

    Mutant: enumerate ``stream_event`` content blocks as proposals.
    """
    config, _plan, operation_id = _captured(
        tmp_path, monkeypatch, reply=_stream(tool_arguments=[])
    )

    plan = capture_recovery.inspect_capture_proposals(config, operation_id)

    assert plan.groups == ()
    assert plan.valid_group_count == 0


def test_an_incomplete_final_frame_refuses_and_preserves_the_reply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A capture read partially is a capture repaired. It is read totally.

    Mutant: skip a frame that is not complete JSON and read the rest.
    """
    truncated = _stream(tool_arguments=[_answer()])[:-40]
    config, _plan, operation_id = _captured(tmp_path, monkeypatch, reply=truncated)
    before = config.operations_file.read_bytes()

    with pytest.raises(JankiError) as caught:
        capture_recovery.inspect_capture_proposals(config, operation_id)

    message = str(caught.value)
    assert "not complete JSON" in message
    assert "preserved exactly" in message
    assert config.operations_file.read_bytes() == before
    assert _state(config, operation_id) == "result_captured"


def test_a_call_to_another_tool_is_not_a_structured_output_proposal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the structured-output tool's argument can be the answer.

    Mutant: accept any ``tool_use`` block's input as a proposal.
    """
    config, _plan, operation_id = _captured(
        tmp_path,
        monkeypatch,
        reply=_stream(tool_arguments=[_answer()], tool_name="Bash"),
    )

    plan = capture_recovery.inspect_capture_proposals(config, operation_id)

    assert plan.groups == ()
    with pytest.raises(JankiError) as caught:
        capture_recovery.stage_capture_proposal(config, operation_id)
    assert "structured-output tool" in str(caught.value)


# --- who wins, and who chooses --------------------------------------------------


def test_a_valid_successful_terminal_wins_over_a_differing_tool_argument(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The settled answer is the answer, even beside a valid earlier argument.

    Mutant: prefer the last or largest structurally valid proposal.
    """
    terminal = _answer(document_title="Terminal answer")
    earlier = _answer(document_title="Earlier tool argument")
    real_write = extraction.write_staging_under_lock
    calls = {"count": 0}

    def refuse_first(path: Path, *args: Any, **kwargs: Any) -> Any:
        calls["count"] += 1
        if calls["count"] == 1:
            raise JankiError("the disk went away")
        return real_write(path, *args, **kwargs)

    monkeypatch.setattr(extraction, "write_staging_under_lock", refuse_first)
    config, _plan, operation_id = _captured(
        tmp_path,
        monkeypatch,
        reply=_stream(tool_arguments=[earlier], terminal=terminal),
    )
    monkeypatch.setattr(extraction, "write_staging_under_lock", real_write)

    plan = capture_recovery.inspect_capture_proposals(config, operation_id)
    assert plan.valid_group_count == 2
    authoritative = plan.authoritative_location
    assert authoritative is not None
    assert authoritative.envelope_shape == "result_structured_output"

    # The salvage path refuses even when the owner names the other location.
    salvage = plan.select(
        next(
            group.proposal_sha256
            for group in plan.groups
            if all(
                location.envelope_shape != "result_structured_output"
                for location in group.locations
            )
        )
    )
    with pytest.raises(JankiError) as caught:
        capture_recovery.stage_capture_proposal(config, operation_id, salvage)
    assert "terminal" in str(caught.value)

    outcome = capture_recovery.stage_capture_proposal(config, operation_id)

    staging_module = __import__(
        "japanese_anki.staging", fromlist=["read_staging"]
    )
    _records, meta = staging_module.read_staging(outcome.target)
    assert meta["capture_recovery"]["envelope_shape"] == "result_structured_output"
    assert meta["pattern_set"]["title"] == "Terminal answer"


def _settled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    tool_arguments: Sequence[Any],
    terminal: Any,
) -> tuple[ProjectConfig, str]:
    """A capture that settled normally and is still sitting at ``result_captured``.

    A valid terminal decodes on its own, so the way to leave one captured is to
    make the staging write fail once, exactly as a full disk does.
    """
    real_write = extraction.write_staging_under_lock
    calls = {"count": 0}

    def refuse_first(path: Path, *args: Any, **kwargs: Any) -> Any:
        calls["count"] += 1
        if calls["count"] == 1:
            raise JankiError("the disk went away")
        return real_write(path, *args, **kwargs)

    monkeypatch.setattr(extraction, "write_staging_under_lock", refuse_first)
    config, _plan, operation_id = _captured(
        tmp_path,
        monkeypatch,
        reply=_stream(tool_arguments=tool_arguments, terminal=terminal),
    )
    monkeypatch.setattr(extraction, "write_staging_under_lock", real_write)
    return config, operation_id


def test_naming_the_terminals_other_location_recovers_the_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One answer in two blocks is one answer, and the terminal is where it lands.

    A real settled capture carries the assembled tool argument *and* the result
    frame's ``structured_output``, character for character the same object — so
    an owner naming the hash `inspect-capture` printed is naming the terminal's
    own content, at the location the listing shows first. That is the ordinary
    authoritative path, not a competing tool argument.

    Mutant: compare the selection's pointer with the terminal's rather than its
    content group.
    """
    answer = _answer()
    config, operation_id = _settled(
        tmp_path, monkeypatch, tool_arguments=[answer], terminal=answer
    )

    plan = capture_recovery.inspect_capture_proposals(config, operation_id)
    assert len(plan.groups) == 1
    pointers = [
        location.json_pointer for location in plan.groups[0].locations
    ]
    assert pointers == ["/message/content/0/input", "/structured_output"]

    outcome = capture_recovery.stage_capture_proposal(
        config,
        operation_id,
        plan.select(
            plan.groups[0].proposal_sha256,
            json_pointer="/message/content/0/input",
        ),
    )

    staging_module = __import__(
        "japanese_anki.staging", fromlist=["read_staging"]
    )
    _records, meta = staging_module.read_staging(outcome.target)
    recorded = meta["capture_recovery"]
    assert recorded["envelope_shape"] == "result_structured_output"
    assert recorded["json_pointer"] == "/structured_output"
    assert recorded["selected_by"] == "successful-terminal"
    # Every location of the group survives: provenance never claims one block
    # when the capture held the same answer in two.
    assert [entry["json_pointer"] for entry in recorded["locations"]] == pointers


def test_matching_the_terminals_content_excuses_no_part_of_the_tuple(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Identical bytes are not a location, on the authoritative path either.

    Mutant: in the terminal branch, compare ``proposal_sha256`` alone and skip
    the other seven components of the owner's tuple.
    """
    from dataclasses import replace

    answer = _answer()
    config, operation_id = _settled(
        tmp_path, monkeypatch, tool_arguments=[answer], terminal=answer
    )
    plan = capture_recovery.inspect_capture_proposals(config, operation_id)
    good = plan.select(
        plan.groups[0].proposal_sha256, json_pointer="/message/content/0/input"
    )

    for field, value in (
        ("operation_id", "1b9f0f2c-6d4e-4b71-9a03-5c8e2d1f4a77"),
        ("capture_sha256", "f" * 64),
        ("response_schema_fingerprint", "f" * 64),
        ("frame_index", good.frame_index + 1),
        ("block_index", good.block_index + 1),
        ("tool_use_id", "toolu_zz"),
        ("json_pointer", "/message/content/9/input"),
        ("proposal_sha256", "f" * 64),
    ):
        with pytest.raises(JankiError):
            capture_recovery.stage_capture_proposal(
                config, operation_id, replace(good, **{field: value})
            )
    assert _state(config, operation_id) == "result_captured"
    assert not (config.staging_dir / "one.pdf.yaml").exists()


def test_two_identical_proposals_need_the_owners_exact_location_tuple(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One content hash is not a location, and a group keeps all of its own.

    Mutant: collapse a content group to its first location.
    """
    answer = _answer()
    config, _plan, operation_id = _captured(
        tmp_path, monkeypatch, reply=_stream(tool_arguments=[answer, answer])
    )

    plan = capture_recovery.inspect_capture_proposals(config, operation_id)

    assert len(plan.groups) == 1
    assert plan.valid_group_count == 1
    assert len(plan.groups[0].locations) == 2
    assert [location.tool_use_id for location in plan.groups[0].locations] == [
        "toolu_01",
        "toolu_02",
    ]

    with pytest.raises(JankiError) as caught:
        capture_recovery.stage_capture_proposal(config, operation_id)
    assert "--proposal" in str(caught.value)

    outcome = capture_recovery.stage_capture_proposal(
        config, operation_id, plan.select(plan.groups[0].proposal_sha256)
    )

    staging_module = __import__(
        "japanese_anki.staging", fromlist=["read_staging"]
    )
    _records, meta = staging_module.read_staging(outcome.target)
    recorded = meta["capture_recovery"]
    assert [entry["tool_use_id"] for entry in recorded["locations"]] == [
        "toolu_01",
        "toolu_02",
    ]
    assert recorded["selected_by"] == "repository-owner"


def test_two_different_proposals_refuse_until_the_owner_names_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Competing content is an owner decision, never latest-or-largest.

    Mutant: stage the first valid group when the capture holds several.
    """
    first = _answer(document_title="First reading")
    second = _answer(document_title="Second reading")
    config, _plan, operation_id = _captured(
        tmp_path, monkeypatch, reply=_stream(tool_arguments=[first, second])
    )

    plan = capture_recovery.inspect_capture_proposals(config, operation_id)
    assert plan.valid_group_count == 2

    with pytest.raises(JankiError):
        capture_recovery.stage_capture_proposal(config, operation_id)

    chosen = plan.groups[1]
    outcome = capture_recovery.stage_capture_proposal(
        config, operation_id, plan.select(chosen.proposal_sha256)
    )

    staging_module = __import__(
        "japanese_anki.staging", fromlist=["read_staging"]
    )
    _records, meta = staging_module.read_staging(outcome.target)
    assert meta["capture_recovery"]["proposal_sha256"] == chosen.proposal_sha256


def test_a_selection_tuple_that_does_not_match_the_capture_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every component of the owner's tuple is compared, not just the hash.

    Mutant: compare only ``proposal_sha256`` and ignore the rest of the tuple.
    """
    from dataclasses import replace

    config, _plan, operation_id = _captured(
        tmp_path, monkeypatch, reply=_stream(tool_arguments=[_answer()])
    )
    plan = capture_recovery.inspect_capture_proposals(config, operation_id)
    good = plan.select(plan.groups[0].proposal_sha256)

    for field, value in (
        ("capture_sha256", "f" * 64),
        ("response_schema_fingerprint", "f" * 64),
        ("frame_index", good.frame_index + 1),
        ("block_index", good.block_index + 1),
        ("tool_use_id", "toolu_zz"),
        ("json_pointer", "/message/content/9/input"),
        ("proposal_sha256", "f" * 64),
    ):
        with pytest.raises(JankiError):
            capture_recovery.stage_capture_proposal(
                config, operation_id, replace(good, **{field: value})
            )
    assert _state(config, operation_id) == "result_captured"


# --- what recovery still has to be true about now -------------------------------


def _bind_case(config: ProjectConfig, case: str) -> None:
    if case == "source":
        (config.scan_inbox / "one.pdf").write_bytes(b"%PDF-1.7 different")
    elif case == "staging":
        config.staging_dir.mkdir(parents=True, exist_ok=True)
        (config.staging_dir / "one.pdf.yaml").write_text(
            "source_file: one.pdf\nrecords: []\n", encoding="utf-8"
        )
    elif case == "patterns":
        text = (config.root / "janki.toml").read_text(encoding="utf-8")
        (config.root / "janki.toml").write_text(
            text.replace('patterns_file = "patterns.json"', 'patterns_file = "p2.json"'),
            encoding="utf-8",
        )
    else:  # pragma: no cover - the parametrization is closed
        raise AssertionError(case)


@pytest.mark.parametrize("case", ("source", "staging", "patterns"))
def test_every_binding_the_batch_recovery_checks_is_checked_here(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    """Salvage is a decoder, not a shortcut past the destination bindings.

    Mutant: skip the ``source_sha256`` comparison in the shared binding.
    """
    config, _plan, operation_id = _captured(
        tmp_path, monkeypatch, reply=_stream(tool_arguments=[_answer()])
    )
    _bind_case(config, case)
    current = ProjectConfig.load(config.root)

    with pytest.raises(JankiError):
        capture_recovery.stage_capture_proposal(current, operation_id)

    assert _state(current, operation_id) == "result_captured"
    payload = operations.OperationJournal.load(
        current.operations_file
    ).read_reply(operation_id)
    assert extraction.extraction_capture_parts(payload)[1] is not None


def test_a_changed_destination_deck_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The deck the owner chose is part of what this answer was bought for.

    Mutant: skip ``require_current_destination`` in the shared binding.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    deck = _deck(config)
    config, _plan, operation_id = _captured(
        tmp_path,
        monkeypatch,
        reply=_stream(tool_arguments=[_answer()]),
        destination_deck=deck,
    )
    deck.write_text(
        deck.read_text(encoding="utf-8") + "  exclude_tags: [drafts]\n",
        encoding="utf-8",
    )

    with pytest.raises(JankiError) as caught:
        capture_recovery.stage_capture_proposal(config, operation_id)

    assert "destination deck" in str(caught.value)
    assert _state(config, operation_id) == "result_captured"


def test_a_changed_response_contract_preserves_both_halves_and_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A capture is never reinterpreted under a contract it was not asked for.

    Mutant: drop the stored/current response-fingerprint comparison.
    """
    config, _plan, operation_id = _captured(
        tmp_path, monkeypatch, reply=_stream(tool_arguments=[_answer()])
    )
    before = config.operations_file.read_bytes()
    monkeypatch.setattr(
        extraction.claude_client, "wire_schema", lambda schema: {"changed": True}
    )

    with pytest.raises(JankiError) as caught:
        capture_recovery.inspect_capture_proposals(config, operation_id)

    assert "contract" in str(caught.value)
    assert config.operations_file.read_bytes() == before


def test_an_operation_with_no_saved_request_refuses_rather_than_being_salvaged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The four shapes exist on the transport that saves its request.

    An ``anthropic-api`` capture has no saved provider manifest, so it is
    already unrecoverable by the ordinary path; salvage does not invent a
    fifth shape for it.

    Mutant: accept a capture with no saved request and read a fifth envelope
    shape out of it.
    """
    # The transport that saves no request beside its reply, written through the
    # real capture path so the journal's receipt describes exactly these bytes.
    monkeypatch.setattr(
        extraction, "extraction_capture_envelope", lambda provenance, raw: raw
    )
    config, _plan, operation_id = _captured(
        tmp_path, monkeypatch, reply=_stream(tool_arguments=[_answer()])
    )
    payload = operations.OperationJournal.load(
        config.operations_file
    ).read_reply(operation_id)
    assert extraction.extraction_capture_parts(payload)[1] is None

    with pytest.raises(JankiError) as caught:
        capture_recovery.inspect_capture_proposals(config, operation_id)

    assert "no saved request" in str(caught.value)
    assert _state(config, operation_id) == "result_captured"


def test_a_live_operation_refuses_and_names_the_owners_end_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a person can say a vanished process is gone. Inspection cannot.

    Mutant: accept every state instead of only a captured or committed one.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    source = _source(config)
    runner = FakeClaudeRunner(reply=_stream(tool_arguments=[_answer()]))
    plan = extraction_batch.plan_extraction_batch(
        config,
        [source],
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
    )

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise JankiError("the subscription CLI is not available")

    monkeypatch.setattr(extraction_batch, "prepare_extraction_transport", refuse)
    extraction_batch.dispatch_extraction_batch(
        config,
        plan,
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
        provider_spawn=runner.spawn,
    )
    operation_id = plan.children[0].operation_id
    assert _state(config, operation_id) == "authorized"

    with pytest.raises(JankiError) as caught:
        capture_recovery.inspect_capture_proposals(config, operation_id)

    assert "janki operations --end" in str(caught.value)


def test_an_unknown_outcome_refuses_rather_than_being_read_or_resent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A call that may have gone and may have cost money is not this reader's.

    Mutant: treat ``outcome_unknown`` as inspectable.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    source = _source(config)
    runner = FakeClaudeRunner(reply=_stream(tool_arguments=[_answer()]))
    plan = extraction_batch.plan_extraction_batch(
        config,
        [source],
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
    )

    def die(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("the CLI died with the request already written")

    extraction_batch.dispatch_extraction_batch(
        config,
        plan,
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
        provider_spawn=die,
    )
    operation_id = plan.children[0].operation_id
    assert _state(config, operation_id) == "outcome_unknown"

    with pytest.raises(JankiError) as caught:
        capture_recovery.inspect_capture_proposals(config, operation_id)

    assert "never sent again" in str(caught.value)


def test_inspection_discloses_pointers_and_hashes_but_no_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plan is safe to show a model: no reply, candidate or source bytes.

    Mutant: carry the parsed proposal object on the returned plan.
    """
    from dataclasses import asdict

    config, _plan, operation_id = _captured(
        tmp_path, monkeypatch, reply=_stream(tool_arguments=[_answer()])
    )

    plan = capture_recovery.inspect_capture_proposals(config, operation_id)

    disclosed = json.dumps(asdict(plan), ensure_ascii=False, default=str)
    assert "走る" not in disclosed
    assert "はしる" not in disclosed
    assert "%PDF" not in disclosed
    assert plan.request_fingerprint
    assert plan.capture_sha256
    assert plan.response_schema_fingerprint
    assert plan.source_sha256


#: Stand-ins for the two kinds of bytes §5.1 promises a plan never carries: a
#: fact the model returned, and the verbatim source line a candidate was read
#: from. ASCII on purpose — the property under test is that *no* returned
#: content reaches the plan, and nothing here reads what the content says.
CANDIDATE_MARKER = "ZQ7cand"
SOURCE_MARKER = "WV7src"


def _marked_answer() -> dict[str, Any]:
    """The fixture answer with its candidate and source text made traceable."""
    value = _answer()
    value["candidates"][0]["expression"] = CANDIDATE_MARKER
    for candidate in value["candidates"]:
        candidate["context"] = SOURCE_MARKER
    for unit in value["source_units"]:
        unit["context"] = SOURCE_MARKER
    return value


def _without(value: dict[str, Any], field: str) -> dict[str, Any]:
    return {
        **value,
        "candidates": [
            {key: item for key, item in candidate.items() if key != field}
            for candidate in value["candidates"]
        ],
    }


#: Each case: how to break the marked answer, and the field name the verdict
#: still has to name.
LEAKY = {
    "extra property": (
        lambda value: {**value, "invented": f"commentary quoting {SOURCE_MARKER}"},
        "invented",
    ),
    "missing source_kind": (
        lambda value: _without(value, "source_kind"),
        "source_kind",
    ),
    "missing meanings": (
        lambda value: _without(value, "meanings"),
        "meanings",
    ),
}


@pytest.mark.parametrize("kind", sorted(LEAKY))
def test_an_invalid_verdict_carries_structure_and_none_of_the_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """The population this module exists to inspect is the invalid one.

    A verdict travels into the plan §5.1 designs as the model-safe projection
    and into the refusal an owner reads, so it says where the error is and what
    kind it is — never the offending value, and never the verbatim source line
    a candidate was read from.

    Mutant: build the verdict from ``str(ValidationError)``.
    """
    from dataclasses import asdict

    mutate, named = LEAKY[kind]
    config, _plan, operation_id = _captured(
        tmp_path,
        monkeypatch,
        reply=_stream(tool_arguments=[mutate(_marked_answer())]),
    )

    plan = capture_recovery.inspect_capture_proposals(config, operation_id)
    disclosed = json.dumps(asdict(plan), ensure_ascii=False, default=str)
    with pytest.raises(JankiError) as caught:
        capture_recovery.stage_capture_proposal(config, operation_id)
    refusal = str(caught.value)

    assert plan.valid_group_count == 0, kind
    for leak in (CANDIDATE_MARKER, SOURCE_MARKER, "走る", "はしる", "%PDF"):
        assert leak not in disclosed, (kind, leak)
        assert leak not in refusal, (kind, leak)
    # What a diagnostic is actually for, unchanged: the exact pointer, the
    # exact field, and a fresh retry rather than a hand-typed value.
    verdict = plan.groups[0].schema_verdict
    assert verdict.startswith("/message/content/0/input: "), kind
    assert named in verdict, kind
    assert named in refusal and "/message/content/0/input" in refusal
    assert "extract-batch retry" in refusal


def test_a_missing_field_is_a_diagnostic_and_a_fresh_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nobody hand-supplies a field the paid answer did not contain.

    Mutant: accept an owner-supplied field value.
    """
    broken = _answer()
    broken["candidates"][0].pop("meanings", None)
    config, _plan, operation_id = _captured(
        tmp_path, monkeypatch, reply=_stream(tool_arguments=[broken])
    )

    with pytest.raises(JankiError) as caught:
        capture_recovery.stage_capture_proposal(config, operation_id)

    message = str(caught.value)
    assert "meanings" in message
    assert "extract-batch retry" in message
    assert not hasattr(capture_recovery, "supply_missing_field")


# --- the local rendering the owner chooses against ------------------------------


@needs_preview
def test_render_capture_proposals_draws_real_cards_and_writes_no_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A local page of the actual cards, and a diagnostic beside it.

    Mutant: let the renderer stage the sole valid proposal as a side effect.
    """
    first = _answer(document_title="First reading")
    second = _answer(document_title="Second reading")
    config, _plan, operation_id = _captured(
        tmp_path, monkeypatch, reply=_stream(tool_arguments=[first, second])
    )
    before = config.operations_file.read_bytes()
    output = tmp_path / "proposals.html"

    written = capture_recovery.render_capture_proposals(
        config, operation_id, output
    )

    assert written == output
    index = output.read_text(encoding="utf-8")
    plan = capture_recovery.inspect_capture_proposals(config, operation_id)
    for group in plan.groups:
        card_page = output.parent / f"{output.stem}-{group.proposal_sha256[:12]}.html"
        diagnostic = output.parent / f"{output.stem}-{group.proposal_sha256[:12]}.json"
        assert card_page.exists()
        assert "走る" in card_page.read_text(encoding="utf-8")
        assert json.loads(diagnostic.read_text(encoding="utf-8"))["document_title"]
        assert card_page.name in index
        assert group.proposal_sha256 in index
    assert config.operations_file.read_bytes() == before
    assert not (config.staging_dir / "one.pdf.yaml").exists()
    assert _state(config, operation_id) == "result_captured"


# --- the template is the first remedy -------------------------------------------


DECODER_PARAGRAPH = (
    "Return the answer by calling the structured-output tool exactly once. "
    "The tool argument is the answer object this schema describes, and "
    "nothing else: every REQUIRED field present on every object it contains, "
    "no property the schema does not define, no wrapper object around it such "
    'as {"input": ...}, and no JSON string or other encoding of it in place '
    "of the object itself."
)


@pytest.mark.parametrize(
    "name", ("extract-auto", "extract-table", "extract-prose")
)
def test_every_extraction_template_asks_for_the_bare_schema_object(
    name: str,
) -> None:
    """The paragraph that prevents this whole module's failure, in all three.

    Mutant: delete the paragraph from one template.
    """
    text = " ".join(prompts.load(PROJECT_ROOT, name).split())

    assert DECODER_PARAGRAPH in text


def test_the_prompt_index_records_the_shared_decoder_paragraph() -> None:
    """`prompts/README.md` is the index a reader trusts about these files."""
    text = (PROJECT_ROOT / "prompts" / "README.md").read_text(encoding="utf-8")

    assert "structured-output tool" in text
    assert "capture" in text.lower()


def test_no_extraction_prompt_is_selected_by_a_python_instruction_branch() -> None:
    """One paragraph, duplicated verbatim: no per-template variant in code."""
    source = (
        PROJECT_ROOT / "src" / "japanese_anki" / "application" / "capture_recovery.py"
    ).read_text(encoding="utf-8")

    assert "structured-output tool exactly once" not in source
    # Every mode is one complete template file, resolved by name. A fourth
    # entry here means `prompts/extract-table-layout.md`, not a rule block
    # somewhere in Python choosing between two askings.
    assert extract.MODES == ("table", "prose", "table-layout")
