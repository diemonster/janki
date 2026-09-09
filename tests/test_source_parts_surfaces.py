"""The two surfaces over source preparation: the Assistant's and the CLI's.

The service is tested in ``tests/test_source_parts.py``. What is tested here is
who may reach it. A model may ask for the editor and may not author, widen or
alter a coordinate; the owner's controls inside the editor plan and publish;
the CLI does the same over the same service; and a published receipt discovers
and snapshots as ids, hashes, states and counts, never as source bytes.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

import pytest
from test_workbench_assistant_integration import _agent_intent, _agent_result

from conftest import REPO_ROOT, seed_prompts
from japanese_anki import cli, cli_source_parts
from japanese_anki.application import assistant_context, source_parts
from japanese_anki.config import ProjectConfig
from japanese_anki.workbench import assistant_adapter, source_part_editor
from japanese_anki.workbench.assistant import RevisionRefusal

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "synthetic_table.pdf"
HEADER = [0.05, 0.03, 0.95, 0.13]
BODY = [0.05, 0.15, 0.95, 0.45]


def _project(tmp_path: Path) -> ProjectConfig:
    seed_prompts(tmp_path)
    (tmp_path / "janki.toml").write_text(
        "[assistant]\nenabled = true\n[ai]\nextract_provider = \"anthropic-api\"\n",
        encoding="utf-8",
    )
    config = ProjectConfig.load(tmp_path)
    config.scan_inbox.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(FIXTURE, config.scan_inbox / "synthetic_table.pdf")
    return config


def _digest(config: ProjectConfig) -> str:
    return hashlib.sha256(
        (config.scan_inbox / "synthetic_table.pdf").read_bytes()
    ).hexdigest()


def _recipe(config: ProjectConfig, parts: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {
        "version": 1,
        "recipe_id": extra.pop("recipe_id", str(uuid.uuid4())),
        "parent_name": "synthetic_table.pdf",
        "parent_sha256": _digest(config),
        "render_dpi": 150,
        "parts": parts,
        **extra,
    }


def _publish_parts(
    config: ProjectConfig, parts: list[dict[str, Any]]
) -> tuple[str, Any]:
    """One really rendered, really published recipe, as a healthy sibling."""

    recipe_id = str(uuid.uuid4())
    recipe, token = source_parts.recipe_from_bytes(
        json.dumps(_recipe(config, parts, recipe_id=recipe_id)).encode("utf-8")
    )
    plan = source_parts.plan_source_parts(
        config, "synthetic_table.pdf", recipe, recipe_token=token
    )
    source_parts.execute_source_parts(config, plan, publish_token=plan.plan_fingerprint)
    return recipe_id, plan


def _receipt_entry(**overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "ordinal": 1,
        "target_name": "synthetic_table--p001-rabcd1234-01.png",
        "sha256": "0" * 64,
        "byte_length": 4,
        "page_index": 0,
        "page_rotate": 0,
        "published": False,
    }
    entry.update(overrides)
    return entry


def _record_reads(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Every path read through ``Path.read_bytes`` from here on.

    ``io.read_bytes_bound`` — how a receipt itself is read — goes through the
    bound-directory reader instead, so what this records is the corpus reads
    ``_part_is_published`` makes.
    """

    opened: list[Path] = []
    read_bytes = Path.read_bytes

    def _record(self: Path) -> bytes:
        opened.append(Path(self))
        return read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", _record)
    return opened


def _adapter(config: ProjectConfig) -> assistant_adapter.RevisionAssistantAdapter:
    return assistant_adapter.RevisionAssistantAdapter(
        config=config, deck_choices=(), _targets=()
    )


class _SourceBroker:
    """The one broker seam the intent path uses, with no repository read."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config

    def source_path(self, resource_id: str) -> Path:
        if resource_id != "resource_source":
            raise assistant_context.AssistantContextError("Unknown Assistant resource.")
        return self.config.scan_inbox / "synthetic_table.pdf"


def _bound_adapter(
    config: ProjectConfig, monkeypatch: pytest.MonkeyPatch
) -> assistant_adapter.RevisionAssistantAdapter:
    adapter = _adapter(config)
    adapter.bind_source_part_editors("http://127.0.0.1:9/token/source-parts/")
    monkeypatch.setattr(
        assistant_adapter.assistant_context,
        "AssistantContextBroker",
        lambda _config: _SourceBroker(config),
    )
    return adapter


def _open_editor(
    config: ProjectConfig, monkeypatch: pytest.MonkeyPatch, **intent: Any
) -> tuple[Any, Any]:
    adapter = _bound_adapter(config, monkeypatch)
    result = _agent_result(
        answer="Open the editor and choose the rows you want.",
        intents=(
            _agent_intent(
                kind="open_source_part_editor",
                resource_ids=("resource_source",),
                instruction="Let me choose parts of this worksheet.",
                **intent,
            ),
        ),
    )
    return adapter, adapter._prepare_agent_intent(config, result=result, deck_scope="")


# --- the model may open the editor, and may do nothing else -------------------


def test_open_source_part_editor_is_a_closed_model_intent() -> None:
    """It is in the schema, and the geometry routes deliberately are not.

    Mutant: add `plan_source_parts` or a recipe/coordinate field to the
    model-facing schema.
    """

    from japanese_anki import ai_schema

    schema = ai_schema.assistant_agent_schema().model_json_schema()
    definitions = schema["$defs"]
    kinds = set(definitions["AssistantActionIntent"]["properties"]["kind"]["enum"])
    assert "open_source_part_editor" in kinds
    assert not kinds & {"plan_source_parts", "publish_source_parts", "source_parts"}
    options = set(definitions["AssistantActionOptions"]["properties"])
    assert not options & {
        "recipe",
        "recipe_token",
        "recipe_path",
        "regions",
        "render_dpi",
        "plan_fingerprint",
        "page_index",
    }
    assert definitions["AssistantActionOptions"].get("additionalProperties") is False


def test_the_model_opens_the_editor_and_nothing_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Opening an editor renders real pages and changes no repository byte.

    Mutant: let the editor intent pre-render and publish parts speculatively.
    """

    config = _project(tmp_path)
    before = {
        path: path.read_bytes()
        for path in sorted(config.root.rglob("*"))
        if path.is_file()
    }
    adapter, reply = _open_editor(config, monkeypatch)

    assert reply.action is None
    assert "/source-parts/" in reply.text
    after = {
        path: path.read_bytes()
        for path in sorted(config.root.rglob("*"))
        if path.is_file()
    }
    assert after == before
    assert not (config.operations_file.parent / "source_parts").exists()

    token = reply.text.split("/source-parts/")[1].split(")")[0]
    document = adapter._source_part_editors.read(token)
    assert document.source_name == "synthetic_table.pdf"
    assert document.page_count == 2
    assert b"data:image/png;base64," in document.html
    assert "script-src 'sha256-" in document.content_security_policy


def test_the_editor_intent_refuses_anything_but_one_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No option on this intent can become geometry, because none is accepted.

    Mutant: accept options on `open_source_part_editor`.
    """

    config = _project(tmp_path)
    with pytest.raises(RevisionRefusal):
        _open_editor(config, monkeypatch, options={"render_dpi": 300})
    with pytest.raises(RevisionRefusal):
        _open_editor(config, monkeypatch, record_ids=("word:one",))


# --- the owner's controls inside the editor -----------------------------------


def test_the_owner_controls_plan_and_then_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Render, review, publish — one service, bound by the reviewed fingerprint.

    Mutant: publish a freshly planned recipe instead of the remembered plan.
    """

    config = _project(tmp_path)
    adapter, reply = _open_editor(config, monkeypatch)
    store = adapter._source_part_editors
    token = reply.text.split("/source-parts/")[1].split(")")[0]

    planned = store.act(
        token,
        {
            "action": "plan",
            "recipe": _recipe(
                config,
                [
                    {"page_index": 0, "page_rotate": 0, "regions": [HEADER, BODY]},
                    {"page_index": 1, "page_rotate": 90},
                ],
            ),
        },
    )
    assert planned["ok"] is True
    assert len(planned["parts"]) == 2
    assert all(part["thumbnail_png_base64"] for part in planned["parts"])
    assert not list(config.scan_inbox.glob("*.png")), "planning publishes nothing"

    published = store.act(
        token, {"action": "publish", "plan_fingerprint": planned["plan_fingerprint"]}
    )
    assert published["ok"] is True
    assert sorted(published["published"]) == sorted(
        part["target_name"] for part in planned["parts"]
    )
    for part in planned["parts"]:
        landed = config.scan_inbox / part["target_name"]
        assert hashlib.sha256(landed.read_bytes()).hexdigest() == part["sha256"]
    assert source_parts.receipt_path(config, planned["recipe_id"]).exists()


def test_publishing_an_unreviewed_fingerprint_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The publish control can only publish the plan this workbench rendered.

    Mutant: fall back to any remembered plan when the fingerprint is unknown.
    """

    config = _project(tmp_path)
    adapter, reply = _open_editor(config, monkeypatch)
    store = adapter._source_part_editors
    token = reply.text.split("/source-parts/")[1].split(")")[0]
    # A plan really was rendered, so a fallback would have something to publish.
    store.act(
        token,
        {
            "action": "plan",
            "recipe": _recipe(
                config, [{"page_index": 0, "page_rotate": 0, "regions": [HEADER]}]
            ),
        },
    )
    with pytest.raises(source_part_editor.SourcePartEditorError):
        store.act(token, {"action": "publish", "plan_fingerprint": "0" * 64})
    with pytest.raises(source_part_editor.SourcePartEditorError):
        store.act(token, {"action": "sneak", "recipe": {}})
    assert not list(config.scan_inbox.glob("*.png"))
    assert not (config.operations_file.parent / "source_parts").exists()


def test_an_editor_recipe_cannot_name_another_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The session decides which source this editor may prepare parts from.

    Mutant: take the parent name from the posted recipe instead of the session.
    """

    config = _project(tmp_path)
    shutil.copyfile(FIXTURE, config.scan_inbox / "other.pdf")
    adapter, reply = _open_editor(config, monkeypatch)
    store = adapter._source_part_editors
    token = reply.text.split("/source-parts/")[1].split(")")[0]
    recipe = _recipe(config, [{"page_index": 0, "page_rotate": 0}])
    recipe["parent_name"] = "other.pdf"
    with pytest.raises(source_parts.SourcePartsError):
        store.act(token, {"action": "plan", "recipe": recipe})
    assert not list(config.scan_inbox.glob("*.png"))


def test_an_unserved_editor_refuses_and_names_commands_that_exist(
    tmp_path: Path,
) -> None:
    """A workbench with no editor store sends the owner to the terminal.

    And to commands `janki` actually has: every backticked command in the
    refusal is parsed here, so a refusal cannot point at a command nobody can
    run. (It once named `janki source-parts --recipe FILE`, which never
    existed — preparation is `source-parts prepare`, and the chooser is
    `source-parts choose`.)

    Mutant: drop the subcommand from either command the refusal names.
    """

    config = _project(tmp_path)
    adapter = _adapter(config)  # deliberately never bound to an editor store
    with pytest.raises(RevisionRefusal) as refusal:
        adapter.open_source_part_editor("synthetic_table.pdf")

    named = re.findall(r"`janki ([^`]+)`", str(refusal.value))
    assert len(named) == 2
    parser = cli.build_parser()
    for command in named:
        argv = [
            str(tmp_path / "recipe.json") if token == "FILE" else token
            for token in command.split()
        ]
        argv += ["--source", "synthetic_table.pdf"]
        if argv[1] == "choose":
            argv += ["--output", str(tmp_path / "sheet.html")]
        parsed = parser.parse_args(argv)
        assert parsed.source_parts_command == argv[1]
    assert {argv.split()[1] for argv in named} == {"choose", "prepare"}


def test_an_unknown_editor_token_refuses(tmp_path: Path) -> None:
    """Editors live in memory; a restart forgets them and says so."""

    store = source_part_editor.LocalSourcePartEditorStore(
        editor_prefix="http://127.0.0.1:9/token/source-parts/",
        plan_recipe=lambda name, data: {"ok": True},
        publish_plan=lambda name, fingerprint: {"ok": True},
    )
    with pytest.raises(source_part_editor.SourcePartEditorError):
        store.read("nothing")
    with pytest.raises(source_part_editor.SourcePartEditorError):
        store.act("nothing", {"action": "plan", "recipe": {}})


# --- the source_part resource -------------------------------------------------


def test_a_published_receipt_discovers_and_snapshots(tmp_path: Path) -> None:
    """Ids, hashes, versions, states and counts — and no source bytes.

    Mutant: disclose the parts' bytes or thumbnails in the snapshot.
    """

    config = _project(tmp_path)
    recipe_id = str(uuid.uuid4())
    recipe, token = source_parts.recipe_from_bytes(
        json.dumps(
            _recipe(
                config,
                [{"page_index": 0, "page_rotate": 0, "regions": [HEADER]}],
                recipe_id=recipe_id,
            )
        ).encode("utf-8")
    )
    plan = source_parts.plan_source_parts(
        config, "synthetic_table.pdf", recipe, recipe_token=token
    )
    source_parts.execute_source_parts(
        config, plan, publish_token=plan.plan_fingerprint
    )

    broker = assistant_context.AssistantContextBroker(config)
    catalog = json.loads(broker.catalog().wire)["data"]
    entries = [
        entry for entry in catalog["resources"] if entry["kind"] == "source_part"
    ]
    assert len(entries) == 1
    # Titled from the receipt; publication state is the selected snapshot's.
    assert entries[0]["title"] == "synthetic_table.pdf parts (1 part)"

    snapshot = broker.snapshot(entries[0]["resource_id"])
    disclosed = json.loads(snapshot.wire)["data"]["source_parts"]
    assert disclosed["recipe_id"] == recipe_id
    assert disclosed["parent_sha256"] == _digest(config)
    assert "pdfium." in disclosed["renderer_version"]
    assert disclosed["parts"][0]["sha256"] == plan.parts[0].sha256
    assert disclosed["parts"][0]["published"] is True
    assert "png_base64" not in snapshot.wire and "thumbnail" not in snapshot.wire

    # The published part is also an ordinary source resource, which is what a
    # later batch plan names.
    titles = {entry["title"] for entry in catalog["resources"]}
    assert plan.parts[0].target_name in titles


def test_a_malformed_receipt_refuses_alone(tmp_path: Path) -> None:
    """One bad receipt never makes the catalog or its siblings disappear.

    Unreadable JSON is only the easy half. A receipt that parses but whose
    parts are missing keys, carry the wrong types, or are not mappings at all
    used to raise a bare ``KeyError``/``TypeError`` — neither is a
    ``JankiError``, so it escaped discovery, escaped the broker constructor,
    and ended every ordinary Assistant turn until the file was removed
    (DESIGN.md: a malformed individual object "does not make unrelated library
    objects unavailable"). The envelope beside those parts is hand-edited by
    the same hand: an array or object DPI raised ``TypeError`` out of the same
    three guards, and a stringified or boolean one was silently coerced into a
    receipt the catalog then titled as healthy.

    Mutant: read a receipt's part entries by raw subscript, or its DPI as
    ``int(parsed.get("render_dpi") or 0)``.
    """

    config = _project(tmp_path)
    healthy_id, healthy_plan = _publish_parts(
        config, [{"page_index": 0, "page_rotate": 0, "regions": [HEADER]}]
    )
    foreign = tmp_path / "elsewhere" / "not-a-part.png"
    foreign.parent.mkdir(parents=True, exist_ok=True)
    foreign.write_bytes(b"a file that is not part of this corpus")

    directory = config.operations_file.parent / source_parts.SOURCE_PARTS_DIR_NAME
    directory.mkdir(parents=True, exist_ok=True)
    cases: list[tuple[str, Any]] = [
        ("unreadable JSON", None),
        ("a part missing every key but one", {"parts": [{"target_name": "x.png"}]}),
        ("a null ordinal", {"parts": [_receipt_entry(ordinal=None)]}),
        ("an entry that is not a mapping", {"parts": ["x.png"]}),
        ("parts that are not a list", {"parts": "seven"}),
        (
            "a target name that is a path",
            {"parts": [_receipt_entry(target_name=str(foreign))]},
        ),
        ("an array DPI", {"render_dpi": [200], "parts": [_receipt_entry()]}),
        ("an object DPI", {"render_dpi": {"dpi": 200}, "parts": [_receipt_entry()]}),
        ("a stringified DPI", {"render_dpi": "200", "parts": [_receipt_entry()]}),
        ("a boolean DPI", {"render_dpi": True, "parts": [_receipt_entry()]}),
    ]
    broken: dict[str, str] = {}
    for label, body in cases:
        recipe_id = str(uuid.uuid4())
        text = (
            "{not json"
            if body is None
            else json.dumps({"version": 1, "recipe_id": recipe_id, **body})
        )
        (directory / f"{recipe_id}.json").write_text(text, encoding="utf-8")
        broken[recipe_id[:8]] = label

    broker = assistant_context.AssistantContextBroker(config)
    catalog = json.loads(broker.catalog().wire)["data"]
    entries = [
        item for item in catalog["resources"] if item["kind"] == "source_part"
    ]
    assert len(entries) == len(broken) + 1

    seen: set[str] = set()
    for entry in entries:
        if entry["title"].startswith("Source parts "):
            label = broken[entry["title"].removeprefix("Source parts ")]
            seen.add(label)
            assert entry["available"] is False, label
            with pytest.raises(assistant_context.AssistantContextError):
                broker.snapshot(entry["resource_id"])
            continue
        # The healthy sibling is still listed, still available, and still
        # discloses its own exact parts.
        assert entry["available"] is True
        disclosed = json.loads(broker.snapshot(entry["resource_id"]).wire)["data"]
        assert disclosed["source_parts"]["recipe_id"] == healthy_id
        assert disclosed["source_parts"]["parts"][0]["sha256"] == (
            healthy_plan.parts[0].sha256
        )
    assert seen == set(broken.values())

    # And the context an ordinary Assistant turn is built from still builds.
    context = assistant_adapter._agent_context(config, deck_scope="")
    assert healthy_plan.parts[0].target_name in context.wire
    assert foreign.read_bytes() == b"a file that is not part of this corpus"


def test_catalog_discovery_titles_a_receipt_without_reading_its_published_parts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Discovery reads receipts; only a selected receipt reads the corpus.

    A broker is constructed for every ordinary Assistant turn, and receipts are
    immutable and never retired (AGENTS.md 1e), so hashing every published part
    to title a catalog entry makes every turn cost the whole prepared corpus.
    The receipt the owner actually selects still reports exact publication
    state, which is the only place that state is disclosed.

    Mutant: have discovery verify publication while titling the catalog.
    """

    config = _project(tmp_path)
    recipe_id, plan = _publish_parts(
        config,
        [
            {"page_index": 0, "page_rotate": 0, "regions": [HEADER]},
            {"page_index": 1, "page_rotate": 90},
        ],
    )
    published = [config.scan_inbox / part.target_name for part in plan.parts]
    assert all(path.is_file() for path in published)

    opened = _record_reads(monkeypatch)
    broker = assistant_context.AssistantContextBroker(config)
    discovery = list(opened)
    catalog = json.loads(broker.catalog().wire)["data"]
    entry = next(
        item for item in catalog["resources"] if item["kind"] == "source_part"
    )
    assert entry["available"] is True
    assert entry["title"] == "synthetic_table.pdf parts (2 parts)"
    assert not [path for path in discovery if path in published]

    opened.clear()
    disclosed = json.loads(broker.snapshot(entry["resource_id"]).wire)["data"]
    record = disclosed["source_parts"]
    assert record["recipe_id"] == recipe_id
    assert [part["published"] for part in record["parts"]] == [True, True]
    assert record["published_count"] == 2
    # The selected receipt is the one that verifies, so it had to read them.
    assert sorted(path for path in opened if path in published) == sorted(published)

    # And that state is measured rather than remembered.
    published[1].unlink()
    fresh = assistant_context.AssistantContextBroker(config)
    catalog = json.loads(fresh.catalog().wire)["data"]
    entry = next(
        item for item in catalog["resources"] if item["kind"] == "source_part"
    )
    assert entry["available"] is True
    record = json.loads(fresh.snapshot(entry["resource_id"]).wire)["data"]["source_parts"]
    assert [part["published"] for part in record["parts"]] == [True, False]
    assert record["published_count"] == 1


# --- the CLI, over the same service -------------------------------------------


def _run_cli(config: ProjectConfig, argv: list[str]) -> int:
    parser = cli.build_parser()
    args = parser.parse_args(["--root", str(config.root), *argv])
    return cli_source_parts.run_source_parts_command(config, args)


def test_the_cli_chooses_plans_publishes_and_reports(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The same preparation, explicitly, from the terminal.

    Mutant: publish without --publish, or publish a plan other than the one
    whose fingerprint was printed.
    """

    config = _project(tmp_path)
    sheet = tmp_path / "sheet.html"
    assert _run_cli(
        config,
        ["source-parts", "choose", "--source", "synthetic_table.pdf", "--output", str(sheet)],
    ) == 0
    assert b"data:image/png;base64," in sheet.read_bytes()

    recipe_id = str(uuid.uuid4())
    recipe_file = tmp_path / "recipe.json"
    recipe_file.write_text(
        json.dumps(
            _recipe(
                config,
                [{"page_index": 0, "page_rotate": 0, "regions": [HEADER, BODY]}],
                recipe_id=recipe_id,
            )
        ),
        encoding="utf-8",
    )
    assert _run_cli(
        config,
        [
            "source-parts",
            "prepare",
            "--source",
            "synthetic_table.pdf",
            "--recipe",
            str(recipe_file),
        ],
    ) == 0
    planned = capsys.readouterr().out
    assert "Nothing was published" in planned
    assert not list(config.scan_inbox.glob("*.png"))
    fingerprint = planned.split("Plan fingerprint: ")[1].split("\n")[0].strip()

    # A fingerprint from a different plan refuses before publishing anything.
    assert _run_cli(
        config,
        [
            "source-parts",
            "prepare",
            "--source",
            "synthetic_table.pdf",
            "--recipe",
            str(recipe_file),
            "--publish",
            "--expect-plan",
            "0" * 64,
        ],
    ) == 1
    assert "Refusing" in capsys.readouterr().out
    assert not list(config.scan_inbox.glob("*.png"))

    assert _run_cli(
        config,
        [
            "source-parts",
            "prepare",
            "--source",
            "synthetic_table.pdf",
            "--recipe",
            str(recipe_file),
            "--publish",
            "--expect-plan",
            fingerprint,
        ],
    ) == 0
    published = capsys.readouterr().out
    assert "Published " in published
    assert len(list(config.scan_inbox.glob("*.png"))) == 1

    assert _run_cli(config, ["source-parts", "status"]) == 0
    listed = capsys.readouterr().out
    assert recipe_id in listed and "1/1 published" in listed

    assert _run_cli(config, ["source-parts", "status", recipe_id]) == 0
    shown = capsys.readouterr().out
    assert "pdfium." in shown and "published" in shown


def test_a_recipe_file_that_cannot_be_read_is_an_error_not_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--recipe FILE` is an owner-typed path, so it gets the OSError boundary.

    `cli.main` formats `JankiError` and nothing else, which is why every other
    owner-supplied file reader here names its OS failure first
    (`inputs._read`, `importers.csv_base.inspect_file`). A missing file is the
    most common typo for this command's primary input.

    Mutant: read the recipe with a bare `Path(args.recipe).read_bytes()`.
    """

    config = _project(tmp_path)
    a_directory = tmp_path / "recipes"
    a_directory.mkdir()
    for recipe in (tmp_path / "no-such-recipe.json", a_directory):
        assert (
            cli.main(
                [
                    "--root",
                    str(config.root),
                    "source-parts",
                    "prepare",
                    "--source",
                    "synthetic_table.pdf",
                    "--recipe",
                    str(recipe),
                ]
            )
            == 1
        )
        captured = capsys.readouterr()
        assert captured.err.startswith("error: "), captured.err
        assert str(recipe) in captured.err
        assert "Traceback" not in captured.err
    assert not list(config.scan_inbox.glob("*.png"))
    assert not (config.operations_file.parent / "source_parts").exists()


def test_status_reports_a_malformed_receipt_as_an_error_not_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Both `status` branches read receipts, and `cli.main` formats JankiError only.

    A receipt whose DPI is a JSON array or text reached `cli.main` as a
    `TypeError`/`ValueError` and printed a traceback at the terminal — Finding
    4's shape arriving through the receipt reader instead of `--recipe FILE`.
    The really published receipt beside it still reports its own recipe.

    Mutant: read the DPI back as `int(parsed.get("render_dpi") or 0)`.
    """

    config = _project(tmp_path)
    healthy_id, _ = _publish_parts(
        config, [{"page_index": 0, "page_rotate": 0, "regions": [HEADER]}]
    )

    def _status(*argv: str) -> tuple[int, str, str]:
        code = cli.main(["--root", str(config.root), "source-parts", "status", *argv])
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    code, shown, _ = _status(healthy_id)
    assert code == 0 and "at 150 DPI" in shown

    directory = config.operations_file.parent / source_parts.SOURCE_PARTS_DIR_NAME
    for render_dpi in ([200], "abc"):
        broken_id = str(uuid.uuid4())
        (directory / f"{broken_id}.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "recipe_id": broken_id,
                    "render_dpi": render_dpi,
                    "parts": [_receipt_entry()],
                }
            ),
            encoding="utf-8",
        )
        for argv in ((broken_id,), ()):
            code, _, err = _status(*argv)
            assert code == 1, argv
            assert err.startswith("error: "), err
            assert broken_id in err and "Traceback" not in err
        (directory / f"{broken_id}.json").unlink()

    # The healthy receipt is unchanged by any of it.
    code, shown, _ = _status(healthy_id)
    assert code == 0 and "at 150 DPI" in shown


# --- the isolated origin serves the editor, and only the owner's controls ------


def _sidecar(config: ProjectConfig, monkeypatch: pytest.MonkeyPatch) -> Any:
    """One real sidecar whose editor store is the real adapter's."""

    from test_workbench_assistant import _FakeRevisions

    from japanese_anki.workbench.assistant_http import create_assistant_sidecar

    adapter = _bound_adapter(config, monkeypatch)
    revisions = _FakeRevisions()
    revisions.bind_source_part_editors = adapter.bind_source_part_editors
    sidecar = create_assistant_sidecar(revisions, deck_choices=())
    sidecar.server.test_adapter = adapter
    return sidecar


def _http(
    sidecar: Any, method: str, path: str, body: bytes | None = None
) -> tuple[int, bytes, dict[str, str]]:
    import http.client

    connection = http.client.HTTPConnection(
        "127.0.0.1", sidecar.server.server_address[1], timeout=5
    )
    headers = {"Content-Type": "application/json", "Origin": sidecar.origin} if body else {}
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    payload = response.read()
    received = {key.casefold(): value for key, value in response.getheaders()}
    connection.close()
    return response.status, payload, received


def test_the_isolated_origin_serves_the_editor_and_its_owner_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The editor is a GET; plan and publish are the owner's POSTs to it.

    Mutant: serve the editor under the ChatKit shell policy, or accept a POST
    to an unknown editor token.
    """

    config = _project(tmp_path)
    sidecar = _sidecar(config, monkeypatch)
    sidecar.start()
    try:
        adapter = sidecar.server.test_adapter
        offer = adapter.open_source_part_editor("synthetic_table.pdf")
        path = "/" + offer.url.split("/", 3)[3]

        status, payload, headers = _http(sidecar, "GET", path)
        assert status == 200
        assert payload == adapter._source_part_editors.read(offer.token).html
        assert headers["content-security-policy"].startswith("default-src 'none'")
        assert "script-src 'sha256-" in headers["content-security-policy"]

        status, payload, _headers = _http(
            sidecar,
            "POST",
            path,
            json.dumps(
                {
                    "action": "plan",
                    "recipe": _recipe(
                        config, [{"page_index": 0, "page_rotate": 0, "regions": [HEADER]}]
                    ),
                }
            ).encode("utf-8"),
        )
        assert status == 200
        planned = json.loads(payload)
        assert planned["ok"] is True and len(planned["parts"]) == 1
        assert not list(config.scan_inbox.glob("*.png"))

        status, payload, _headers = _http(
            sidecar,
            "POST",
            path,
            json.dumps(
                {"action": "publish", "plan_fingerprint": planned["plan_fingerprint"]}
            ).encode("utf-8"),
        )
        assert status == 200
        published = json.loads(payload)
        assert published["ok"] is True
        assert (config.scan_inbox / planned["parts"][0]["target_name"]).is_file()

        status, payload, _headers = _http(
            sidecar, "GET", "/" + offer.url.split("/", 3)[3].rsplit("/", 1)[0] + "/nope"
        )
        assert status == 404
        status, payload, _headers = _http(
            sidecar,
            "POST",
            "/" + offer.url.split("/", 3)[3].rsplit("/", 1)[0] + "/nope",
            b"{}",
        )
        assert status == 404
    finally:
        sidecar.close()
