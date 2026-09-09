"""Rendering and publishing the owner's own page and region choices.

Every render here is real: the ``sources`` extra is a declared ``[dev]``
dependency precisely so these tests exercise PDFium and Pillow rather than
skipping and reporting green for the code this milestone exists to add.
``tests/fixtures/synthetic_table.pdf`` is a hand-written, uncompressed,
deterministic fixture — no private source, no Japanese, and nothing here reads
what a rectangle contains.

The subject is provenance, not pictures: which pixels of which page were
rasterized at which DPI, that the bytes hash to what the receipt bound, and
that a derivative never overwrites a namesake.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest
from test_revision_provider import FakeClaudeRunner, _which

from conftest import REPO_ROOT, seed_prompts
from japanese_anki import inputs
from japanese_anki.application import source_parts
from japanese_anki.application.source_parts import (
    SourcePartsError,
    execute_source_parts,
    load_source_part_receipt,
    plan_source_parts,
    receipt_path,
    recipe_from_bytes,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.identifiers import short_fingerprint

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "synthetic_table.pdf"

#: The first page's table body, and the header band above it. Chosen by reading
#: the fixture's own drawing coordinates, which is what an owner does in the
#: editor — janki detects nothing.
HEADER = [0.05, 0.03, 0.95, 0.13]
BODY = [0.05, 0.15, 0.95, 0.45]
#: The rotated page's black square, in the rendered (post-``/Rotate``) frame.
SQUARE = [0.06, 0.07, 0.18, 0.24]


def _project(tmp_path: Path) -> ProjectConfig:
    seed_prompts(tmp_path)
    (tmp_path / "janki.toml").write_text(
        "[ai]\n"
        'extract_provider = "claude-code"\n'
        'extract_model = "claude-opus-5"\n',
        encoding="utf-8",
    )
    return ProjectConfig.load(tmp_path)


def _seed_parent(config: ProjectConfig, name: str = "synthetic_table.pdf") -> str:
    config.scan_inbox.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(FIXTURE, config.scan_inbox / name)
    return hashlib.sha256(FIXTURE.read_bytes()).hexdigest()


def _recipe_bytes(
    parent_sha256: str,
    parts: list[dict[str, Any]],
    *,
    parent_name: str = "synthetic_table.pdf",
    render_dpi: int = 150,
    recipe_id: str | None = None,
    **extra: Any,
) -> bytes:
    payload: dict[str, Any] = {
        "version": 1,
        "recipe_id": recipe_id or str(uuid.uuid4()),
        "parent_name": parent_name,
        "parent_sha256": parent_sha256,
        "render_dpi": render_dpi,
        "parts": parts,
        **extra,
    }
    return json.dumps(payload).encode("utf-8")


def _plan(config: ProjectConfig, data: bytes, *, source: str = "synthetic_table.pdf") -> Any:
    recipe, token = recipe_from_bytes(data)
    return plan_source_parts(config, source, recipe, recipe_token=token)


def _publish(config: ProjectConfig, plan: Any) -> Any:
    return execute_source_parts(config, plan, publish_token=plan.plan_fingerprint)


def _image(data: bytes) -> Any:
    from PIL import Image

    return Image.open(io.BytesIO(data)).convert("RGB")


def _mean_luminance(data: bytes) -> float:
    image = _image(data).convert("L")
    try:
        pixels = image.tobytes()
    finally:
        image.close()
    return sum(pixels) / len(pixels)


# --- P1: names and fingerprints come from the recipe --------------------------


def test_part_names_and_hashes_are_derived_from_the_recipe_fingerprint(
    tmp_path: Path,
) -> None:
    """Two runs of one recipe plan identical names, hashes and fingerprints.

    Mutant: derive the name suffix from a timestamp or the parent's path
    instead of the recipe fingerprint.
    """

    config = _project(tmp_path)
    digest = _seed_parent(config)
    recipe_id = str(uuid.uuid4())
    data = _recipe_bytes(
        digest,
        [
            {"page_index": 0, "page_rotate": 0},
            {"page_index": 0, "page_rotate": 0, "regions": [HEADER, BODY]},
        ],
        recipe_id=recipe_id,
    )

    first = _plan(config, data)
    second = _plan(config, data)

    assert [part.target_name for part in first.parts] == [
        part.target_name for part in second.parts
    ]
    assert [part.sha256 for part in first.parts] == [
        part.sha256 for part in second.parts
    ]
    assert first.plan_fingerprint == second.plan_fingerprint
    suffix = short_fingerprint(first.recipe_sha256, length=8)
    assert first.parts[0].target_name == f"synthetic_table--p001-r{suffix}-01.png"
    assert first.parts[1].target_name == f"synthetic_table--p001-r{suffix}-02.png"
    # The bytes really were rendered, and each part's hash describes them.
    for part, payload in zip(first.parts, first.payloads, strict=True):
        assert hashlib.sha256(payload).hexdigest() == part.sha256
        assert payload.startswith(b"\x89PNG\r\n\x1a\n")


def test_a_changed_rectangle_changes_the_planned_names(tmp_path: Path) -> None:
    """Different geometry under a new recipe id is a different set of parts."""

    config = _project(tmp_path)
    digest = _seed_parent(config)
    first = _plan(
        config,
        _recipe_bytes(digest, [{"page_index": 0, "page_rotate": 0, "regions": [HEADER]}]),
    )
    second = _plan(
        config,
        _recipe_bytes(
            digest, [{"page_index": 0, "page_rotate": 0, "regions": [[0.05, 0.03, 0.95, 0.2]]}]
        ),
    )
    assert first.parts[0].target_name != second.parts[0].target_name
    assert first.parts[0].sha256 != second.parts[0].sha256


# --- P2: rotation ------------------------------------------------------------


def test_a_region_maps_to_pixels_after_the_pages_own_rotate(tmp_path: Path) -> None:
    """A /Rotate 90 page's normalized region crops the rotated rendering.

    The fixture's second page draws one solid black square in its PDF-space
    bottom-left corner, which a 90° rotation puts at the top-left of what a
    person sees. A region over that corner must therefore come back black.

    Mutant: render the page without applying its ``/Rotate``
    (``page.render(scale=scale, rotation=-rotation)``).
    """

    config = _project(tmp_path)
    digest = _seed_parent(config)
    plan = _plan(
        config,
        _recipe_bytes(
            digest,
            [
                {"page_index": 1, "page_rotate": 90},
                {"page_index": 1, "page_rotate": 90, "regions": [SQUARE]},
            ],
        ),
    )

    whole, cropped = plan.parts
    assert whole.page_size_pt == (792.0, 612.0)  # the rotated page, as displayed
    assert whole.pixel_rect[2] > whole.pixel_rect[3]  # landscape once rotated
    assert _mean_luminance(plan.payloads[1]) < 40  # the square, not the margin
    assert _mean_luminance(plan.payloads[0]) > 200  # the page is mostly white


def test_a_rotation_the_recipe_did_not_record_refuses(tmp_path: Path) -> None:
    """The page's own /Rotate is part of what the owner reviewed.

    Mutant: drop the recorded-rotation comparison in the worker.
    """

    config = _project(tmp_path)
    digest = _seed_parent(config)
    with pytest.raises(SourcePartsError) as error:
        _plan(
            config,
            _recipe_bytes(digest, [{"page_index": 1, "page_rotate": 0, "regions": [SQUARE]}]),
        )
    assert "90" in str(error.value)
    assert not list(config.scan_inbox.glob("*.png"))


# --- P3: the renderer never runs in this process ------------------------------


def _subprocess(script: str, *, cwd: Path) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def test_rendering_is_confined_to_the_worker_process(tmp_path: Path) -> None:
    """A completed render leaves no renderer module in the parent.

    Run in a subprocess because a sibling test may legitimately import Pillow
    to look at a PNG, and ``sys.modules`` is per process.

    Mutant: import ``pypdfium2`` at ``source_parts`` module scope, or render in
    the calling process.
    """

    config = _project(tmp_path)
    digest = _seed_parent(config)
    data = _recipe_bytes(digest, [{"page_index": 0, "page_rotate": 0, "regions": [HEADER]}])
    (tmp_path / "recipe.json").write_bytes(data)
    script = """
import json, sys
from pathlib import Path
from japanese_anki.application import source_parts
from japanese_anki.config import ProjectConfig

config = ProjectConfig.load(Path("."))
recipe, token = source_parts.recipe_from_bytes(Path("recipe.json").read_bytes())
plan = source_parts.plan_source_parts(
    config, "synthetic_table.pdf", recipe, recipe_token=token
)
assert plan.parts and plan.payloads[0].startswith(b"\\x89PNG")
leaked = sorted(name for name in ("pypdfium2", "PIL") if name in sys.modules)
print(json.dumps({"leaked": leaked, "renderer": plan.renderer_version}))
"""
    result = _subprocess(script, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    reported = json.loads(result.stdout.strip().splitlines()[-1])
    assert reported["leaked"] == []
    assert "pdfium" in reported["renderer"]


def test_the_ordinary_build_path_never_loads_the_renderer(tmp_path: Path) -> None:
    """A real deck build must not import pypdfium2 or Pillow.

    The base install has neither package, so anything the ordinary build or
    import path pulls in at module scope would break a checkout without the
    optional extra — the discipline ``preview_unavailable`` already keeps.

    Mutant: import either package at module scope anywhere the CLI reaches.
    """

    output = tmp_path / "gate-check.apkg"
    script = f"""
import json, sys
import japanese_anki.cli as cli
import japanese_anki.application.source_parts  # noqa: F401

code = cli.main(["build", "data/decks/verbs.yaml", "--output", {str(output)!r}])
assert code == 0, code
print(json.dumps(sorted(n for n in ("pypdfium2", "PIL") if n in sys.modules)))
"""
    result = _subprocess(script, cwd=REPO_ROOT)
    assert result.returncode == 0, result.stderr
    assert output.is_file()
    assert json.loads(result.stdout.strip().splitlines()[-1]) == []


# --- P4: namesakes and reuse ---------------------------------------------------


def test_a_namesake_with_different_bytes_refuses_the_whole_recipe(
    tmp_path: Path,
) -> None:
    """One refusal covers every part, because half a recipe was not reviewed.

    Mutant: refuse only the colliding part and publish the rest.
    """

    config = _project(tmp_path)
    digest = _seed_parent(config)
    plan = _plan(
        config,
        _recipe_bytes(
            digest,
            [
                {"page_index": 0, "page_rotate": 0, "regions": [HEADER]},
                {"page_index": 0, "page_rotate": 0, "regions": [BODY]},
            ],
        ),
    )
    # A different file elsewhere under the durable inbox root, under the name
    # the *second* part is planned to take.
    elsewhere = config.root / "data" / "inbox" / "shirabe"
    elsewhere.mkdir(parents=True, exist_ok=True)
    (elsewhere / plan.parts[1].target_name).write_bytes(b"\x89PNG\r\n\x1a\nnot the part")

    with pytest.raises(JankiError) as error:
        _publish(config, plan)

    assert plan.parts[1].target_name in str(error.value)
    assert not (config.scan_inbox / plan.parts[0].target_name).exists()
    assert not (config.scan_inbox / plan.parts[1].target_name).exists()


def test_publishing_the_same_recipe_again_reuses_every_part(tmp_path: Path) -> None:
    """An exact republication writes nothing and keeps the same receipt."""

    config = _project(tmp_path)
    digest = _seed_parent(config)
    data = _recipe_bytes(
        digest,
        [
            {"page_index": 0, "page_rotate": 0},
            {"page_index": 0, "page_rotate": 0, "regions": [HEADER, BODY]},
        ],
    )
    first = _publish(config, _plan(config, data))
    assert len(first.published) == 2 and first.reused == ()

    before = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(config.scan_inbox.glob("*.png"))
    }
    second = _publish(config, _plan(config, data))

    assert second.published == () and len(second.reused) == 2
    assert second.receipt_sha256 == first.receipt_sha256
    after = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(config.scan_inbox.glob("*.png"))
    }
    assert after == before


# --- P5/P6: the receipt is the expectation ------------------------------------


def test_an_interrupted_publication_resumes_from_the_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The receipt is written first, so a resumed run republishes only what is absent.

    Mutant: write the receipt after publication.
    """

    config = _project(tmp_path)
    digest = _seed_parent(config)
    recipe_id = str(uuid.uuid4())
    data = _recipe_bytes(
        digest,
        [
            {"page_index": 0, "page_rotate": 0, "regions": [HEADER]},
            {"page_index": 0, "page_rotate": 0, "regions": [BODY]},
        ],
        recipe_id=recipe_id,
    )
    plan = _plan(config, data)

    real_write = inputs.atomic_write_bytes
    written: list[Path] = []

    def fail_on_the_second(path: Path, payload: bytes) -> Any:
        if written:
            raise OSError("the disk went away")
        written.append(Path(path))
        return real_write(path, payload)

    monkeypatch.setattr(inputs, "atomic_write_bytes", fail_on_the_second)
    with pytest.raises(OSError):
        _publish(config, plan)
    monkeypatch.undo()

    receipt = receipt_path(config, recipe_id)
    assert receipt.exists(), "the expectation must exist before the first part"
    assert (config.scan_inbox / plan.parts[0].target_name).exists()
    assert not (config.scan_inbox / plan.parts[1].target_name).exists()

    resumed = _publish(config, _plan(config, data))
    assert [path.name for path in resumed.reused] == [plan.parts[0].target_name]
    assert [path.name for path in resumed.published] == [plan.parts[1].target_name]
    record = load_source_part_receipt(config, recipe_id)
    assert [part.published for part in record.parts] == [True, True]


def test_a_receipt_naming_different_parts_refuses_and_overwrites_nothing(
    tmp_path: Path,
) -> None:
    """One recipe id names one set of parts, forever.

    Mutant: overwrite the receipt when the planned parts differ.
    """

    config = _project(tmp_path)
    digest = _seed_parent(config)
    recipe_id = str(uuid.uuid4())
    first = _plan(
        config,
        _recipe_bytes(
            digest, [{"page_index": 0, "page_rotate": 0, "regions": [HEADER]}], recipe_id=recipe_id
        ),
    )
    receipt = _publish(config, first)
    original = receipt.path.read_bytes()

    second = _plan(
        config,
        _recipe_bytes(
            digest, [{"page_index": 0, "page_rotate": 0, "regions": [BODY]}], recipe_id=recipe_id
        ),
    )
    with pytest.raises(SourcePartsError) as error:
        _publish(config, second)

    assert "immutable" in str(error.value)
    assert receipt.path.read_bytes() == original
    assert not (config.scan_inbox / second.parts[0].target_name).exists()


def test_publishing_needs_the_exact_reviewed_plan_fingerprint(tmp_path: Path) -> None:
    """A plan alone publishes nothing.

    Mutant: accept any nonempty publish token.
    """

    config = _project(tmp_path)
    digest = _seed_parent(config)
    plan = _plan(
        config, _recipe_bytes(digest, [{"page_index": 0, "page_rotate": 0, "regions": [HEADER]}])
    )
    with pytest.raises(SourcePartsError):
        execute_source_parts(config, plan, publish_token="not-the-fingerprint")
    assert not list(config.scan_inbox.glob("*.png"))
    assert not receipt_path(config, plan.recipe_id).exists()


# --- P7: two part forms, and only two -----------------------------------------


def test_a_region_list_composes_top_to_bottom(tmp_path: Path) -> None:
    """Header above body, in the owner's order, with nothing between them.

    Mutant: compose the regions in reverse order.
    """

    config = _project(tmp_path)
    digest = _seed_parent(config)
    plan = _plan(
        config,
        _recipe_bytes(
            digest,
            [
                {"page_index": 0, "page_rotate": 0, "regions": [HEADER]},
                {"page_index": 0, "page_rotate": 0, "regions": [BODY]},
                {"page_index": 0, "page_rotate": 0, "regions": [HEADER, BODY]},
            ],
        ),
    )
    header = _image(plan.payloads[0])
    body = _image(plan.payloads[1])
    composed = _image(plan.payloads[2])
    try:
        assert composed.size == (
            max(header.width, body.width),
            header.height + body.height,
        )
        assert composed.crop((0, 0, header.width, header.height)).tobytes() == (
            header.tobytes()
        )
        assert composed.crop(
            (0, header.height, body.width, header.height + body.height)
        ).tobytes() == body.tobytes()
    finally:
        header.close()
        body.close()
        composed.close()


def test_a_present_but_empty_region_list_is_not_a_whole_page(tmp_path: Path) -> None:
    """An empty selection is a mistake, never a silent whole-page publication.

    Mutant: treat a present-but-empty ``regions`` list as a whole page.
    """

    config = _project(tmp_path)
    digest = _seed_parent(config)
    with pytest.raises(SourcePartsError) as error:
        _plan(config, _recipe_bytes(digest, [{"page_index": 0, "page_rotate": 0, "regions": []}]))
    assert "whole page" in str(error.value)


def test_regions_must_run_top_to_bottom_without_overlap(tmp_path: Path) -> None:
    """A part composes bands; it does not reorder or blend them."""

    config = _project(tmp_path)
    digest = _seed_parent(config)
    with pytest.raises(SourcePartsError):
        _plan(
            config,
            _recipe_bytes(digest, [{"page_index": 0, "page_rotate": 0, "regions": [BODY, HEADER]}]),
        )


def test_an_unknown_part_key_refuses(tmp_path: Path) -> None:
    """There is no third part form: a part is a page, its /Rotate and regions."""

    config = _project(tmp_path)
    digest = _seed_parent(config)
    with pytest.raises(SourcePartsError) as error:
        _plan(
            config,
            _recipe_bytes(
                digest, [{"page_index": 0, "page_rotate": 0, "columns": ["dictionary form"]}]
            ),
        )
    assert "columns" in str(error.value)


# --- P9: geometry provenance ---------------------------------------------------


def test_a_token_minted_over_different_recipe_bytes_refuses(tmp_path: Path) -> None:
    """The token binds the exact geometry the owner reviewed.

    Mutant: accept the token without comparing it to the recipe.
    """

    config = _project(tmp_path)
    digest = _seed_parent(config)
    reviewed, token = recipe_from_bytes(
        _recipe_bytes(digest, [{"page_index": 0, "page_rotate": 0, "regions": [HEADER]}])
    )
    widened, _other = recipe_from_bytes(
        _recipe_bytes(
            digest,
            [{"page_index": 0, "page_rotate": 0, "regions": [[0.0, 0.0, 1.0, 0.9]]}],
            recipe_id=reviewed.recipe_id,
        )
    )
    with pytest.raises(SourcePartsError) as error:
        plan_source_parts(config, "synthetic_table.pdf", widened, recipe_token=token)
    assert "different recipe bytes" in str(error.value)

    with pytest.raises(SourcePartsError):
        plan_source_parts(config, "synthetic_table.pdf", reviewed, recipe_token="")
    assert not list(config.scan_inbox.glob("*.png"))


def test_a_parent_whose_bytes_moved_refuses(tmp_path: Path) -> None:
    """A recipe is written over one exact document."""

    config = _project(tmp_path)
    _seed_parent(config)
    with pytest.raises(SourcePartsError) as error:
        _plan(
            config,
            _recipe_bytes("0" * 64, [{"page_index": 0, "page_rotate": 0}]),
        )
    assert "not the file this recipe was written over" in str(error.value)


def test_a_declared_renderer_version_must_match_this_checkout(tmp_path: Path) -> None:
    """The renderer and encoder pair is the pixels, so it is bound."""

    config = _project(tmp_path)
    digest = _seed_parent(config)
    with pytest.raises(SourcePartsError) as error:
        _plan(
            config,
            _recipe_bytes(
                digest,
                [{"page_index": 0, "page_rotate": 0}],
                renderer="pypdfium2",
                renderer_version="0.0.1+pdfium.1",
            ),
        )
    assert "renderer version" in str(error.value)


# --- P8/P11: the parts are ordinary corpus sources ----------------------------


def test_published_parts_are_what_the_batch_planner_prepares(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Publish, then plan: every child binds a part that already exists.

    Mutant: publish into ``data/inbox/`` instead of ``config.scan_inbox``.
    """

    from japanese_anki.application.extraction_batch import plan_extraction_batch

    config = _project(tmp_path)
    digest = _seed_parent(config)
    receipt = _publish(
        config,
        _plan(
            config,
            _recipe_bytes(
                digest,
                [
                    {"page_index": 0, "page_rotate": 0, "regions": [HEADER, BODY]},
                    # Byte-for-byte the same crop, published under its own name.
                    {"page_index": 0, "page_rotate": 0, "regions": [HEADER, BODY]},
                ],
            ),
        ),
    )
    published = [path for path in receipt.published]
    assert len(published) == 2
    assert receipt.plan.parts[0].sha256 == receipt.plan.parts[1].sha256

    runner = FakeClaudeRunner(reply=b"")
    plan = plan_extraction_batch(
        config,
        published,
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
    )
    assert [child.expectation.source_sha256 for child in plan.children] == [
        part.sha256 for part in receipt.plan.parts
    ]
    # Identical bytes under two names are two requests, not one duplicate.
    fingerprints = {child.expectation.request_fingerprint for child in plan.children}
    assert len(fingerprints) == 2
    for child in plan.children:
        assert child.expectation.source.parent == config.scan_inbox


def test_an_unpublished_part_path_refuses_before_any_call(tmp_path: Path) -> None:
    """Planning binds bytes that exist; publication is what makes them exist."""

    from japanese_anki.application.extraction_batch import plan_extraction_batch

    config = _project(tmp_path)
    digest = _seed_parent(config)
    plan = _plan(
        config, _recipe_bytes(digest, [{"page_index": 0, "page_rotate": 0, "regions": [HEADER]}])
    )
    runner = FakeClaudeRunner(reply=b"")
    with pytest.raises(JankiError):
        plan_extraction_batch(
            config,
            [config.scan_inbox / plan.parts[0].target_name],
            provider_env={},
            provider_runner=runner,
            provider_which=_which,
        )


# --- the receipt, read back ----------------------------------------------------


def test_the_receipt_records_every_planned_name_and_hash(tmp_path: Path) -> None:
    """Ids, hashes, versions and counts — the whole publication expectation."""

    config = _project(tmp_path)
    digest = _seed_parent(config)
    recipe_id = str(uuid.uuid4())
    plan = _plan(
        config,
        _recipe_bytes(
            digest,
            [{"page_index": 0, "page_rotate": 0}, {"page_index": 1, "page_rotate": 90}],
            recipe_id=recipe_id,
        ),
    )
    receipt = _publish(config, plan)

    assert receipt.path == receipt_path(config, recipe_id)
    assert receipt.path.parent.name == "source_parts"
    assert receipt.path.parent.parent == config.operations_file.parent
    saved = json.loads(receipt.path.read_text(encoding="utf-8"))
    assert saved["parent_sha256"] == digest
    assert saved["plan_fingerprint"] == plan.plan_fingerprint
    assert [entry["target_name"] for entry in saved["parts"]] == [
        part.target_name for part in plan.parts
    ]
    assert [entry["sha256"] for entry in saved["parts"]] == [
        part.sha256 for part in plan.parts
    ]
    assert saved["renderer"] == "pypdfium2" and "pdfium." in saved["renderer_version"]
    assert saved["encoder"] == "Pillow"
    assert "thumbnail_png_base64" not in json.dumps(saved)

    record = load_source_part_receipt(config, recipe_id)
    assert record.receipt_sha256 == receipt.receipt_sha256
    assert record.render_dpi == 150
    assert record.to_dict()["published_count"] == 2
    assert source_parts.list_source_part_receipts(config) == (recipe_id,)


def test_the_parent_source_is_never_touched(tmp_path: Path) -> None:
    """Intake is immutable: preparing parts reads the parent and nothing else."""

    config = _project(tmp_path)
    digest = _seed_parent(config)
    parent = config.scan_inbox / "synthetic_table.pdf"
    before = parent.read_bytes()
    _publish(
        config,
        _plan(
            config,
            _recipe_bytes(digest, [{"page_index": 0, "page_rotate": 0, "regions": [BODY]}]),
        ),
    )
    assert parent.read_bytes() == before
    assert hashlib.sha256(parent.read_bytes()).hexdigest() == digest


def test_a_source_outside_the_corpus_refuses(tmp_path: Path) -> None:
    """Geometry never names a path; a source name is a corpus entry or nothing."""

    config = _project(tmp_path)
    digest = _seed_parent(config)
    recipe, token = recipe_from_bytes(
        _recipe_bytes(digest, [{"page_index": 0, "page_rotate": 0}])
    )
    for name in ("../janki.toml", "/etc/hosts", "missing.pdf"):
        with pytest.raises(SourcePartsError):
            plan_source_parts(config, name, recipe, recipe_token=token)


# --- reading a receipt back is validation, not indexing ------------------------


def _hand_written_receipt(config: ProjectConfig, recipe_id: str, payload: Any) -> Path:
    """Put one receipt where the loader looks, without going through a plan.

    A receipt is machine-written and immutable, so the only way it holds one of
    these shapes is a hand edit or corruption — which is exactly the case the
    reader's guard exists for.
    """

    path = receipt_path(config, recipe_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        payload if isinstance(payload, str) else json.dumps(payload),
        encoding="utf-8",
    )
    return path


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


def test_a_structurally_malformed_receipt_entry_refuses_as_a_janki_error(
    tmp_path: Path,
) -> None:
    """Valid JSON is not a valid receipt, and the difference has to be an error.

    Every guard that reads a receipt — the Assistant catalog, ``snapshot``, the
    broker constructor — catches ``JankiError``. A part entry read by raw
    subscript raises ``KeyError`` for a missing key and ``TypeError`` for a
    ``None`` or list value, and neither is one, so both escape the guard and
    take unrelated work down with them.

    Mutant: read the entry's fields by raw subscript again —
    ``int(entry["ordinal"])``, ``str(entry["target_name"])``.
    """

    config = _project(tmp_path)
    recipe_id = str(uuid.uuid4())
    malformed: list[tuple[str, Any]] = [
        ("parts is not a list", "seven"),
        ("an entry that is not a mapping", ["synthetic_table--p001.png"]),
        ("a null entry", [None]),
        ("an entry missing every key but one", [{"target_name": "x.png"}]),
        ("a null ordinal", [_receipt_entry(ordinal=None)]),
        ("a list where a name belongs", [_receipt_entry(target_name=["x.png"])]),
        ("a stringified byte length", [_receipt_entry(byte_length="4")]),
        ("a null page index", [_receipt_entry(page_index=None)]),
        ("a float ordinal", [_receipt_entry(ordinal=1.5)]),
    ]
    for description, parts in malformed:
        _hand_written_receipt(
            config,
            recipe_id,
            {"version": 1, "recipe_id": recipe_id, "parts": parts},
        )
        with pytest.raises(SourcePartsError):
            load_source_part_receipt(config, recipe_id)
        # JankiError is what every caller's guard is written against; a bare
        # KeyError or TypeError would pass the line above only by accident.
        try:
            load_source_part_receipt(config, recipe_id)
        except JankiError:
            pass
        except Exception as exc:  # pragma: no cover - the defect this pins
            raise AssertionError(
                f"{description} raised {type(exc).__name__}, not a JankiError"
            ) from exc


def test_a_structurally_malformed_receipt_envelope_refuses_as_a_janki_error(
    tmp_path: Path,
) -> None:
    """The envelope beside the parts is hand-edited by the same hand.

    ``int(parsed.get("render_dpi") or 0)`` raises ``TypeError`` for a JSON
    array or object and ``ValueError`` for text, and neither is a
    ``JankiError``, so both escape the three guards this reader's contract
    names — exactly as a raw part subscript did. It also silently coerced
    ``"200"`` and ``true`` into whole numbers, which the same reader refuses
    in every part field. One typed rule for both halves of the receipt.

    Mutant: read the DPI back as ``int(parsed.get("render_dpi") or 0)``.
    """

    config = _project(tmp_path)
    recipe_id = str(uuid.uuid4())
    malformed: list[tuple[str, Any]] = [
        # The two the old expression accepted, before the two it died on.
        ("a stringified DPI", "200"),
        ("a boolean DPI", True),
        ("an array DPI", [200]),
        ("an object DPI", {"dpi": 200}),
        ("text where a DPI belongs", "abc"),
        ("a float DPI", 199.5),
    ]
    for description, render_dpi in malformed:
        _hand_written_receipt(
            config,
            recipe_id,
            {
                "version": 1,
                "recipe_id": recipe_id,
                "render_dpi": render_dpi,
                "parts": [_receipt_entry()],
            },
        )
        with pytest.raises(SourcePartsError):
            load_source_part_receipt(config, recipe_id)
        # JankiError is what every caller's guard is written against; a bare
        # TypeError or ValueError would pass the line above only by accident.
        try:
            load_source_part_receipt(config, recipe_id)
        except JankiError:
            pass
        except Exception as exc:  # pragma: no cover - the defect this pins
            raise AssertionError(
                f"{description} raised {type(exc).__name__}, not a JankiError"
            ) from exc

    # A whole number still reads back as itself, which is all a receipt this
    # code wrote ever holds.
    _hand_written_receipt(
        config,
        recipe_id,
        {
            "version": 1,
            "recipe_id": recipe_id,
            "render_dpi": 200,
            "parts": [_receipt_entry()],
        },
    )
    assert load_source_part_receipt(config, recipe_id).render_dpi == 200

    # An absent envelope number keeps the envelope's own default, the way its
    # absent text fields keep ``""`` — so a receipt perturbed anywhere else
    # still refuses for its own reason rather than for this one.
    _hand_written_receipt(
        config,
        recipe_id,
        {"version": 1, "recipe_id": recipe_id, "parts": [_receipt_entry()]},
    )
    assert load_source_part_receipt(config, recipe_id).render_dpi == 0


def test_a_receipt_target_name_that_is_not_a_basename_refuses_before_any_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The read side applies publication's basename rule, and applies it first.

    ``config.scan_inbox / name`` *is* ``name`` when the name is absolute, and
    leaves the corpus the ordinary way when it holds ``..``. Publication
    already re-checks that every planned name is exactly
    ``inputs.safe_upload_name(name)``; a receipt read back has to clear the
    same bar before it becomes a path to open and hash. Artifact provenance,
    not content.

    Mutant: drop the basename check from the receipt reader.
    """

    config = _project(tmp_path)
    foreign = tmp_path / "elsewhere" / "not-a-part.png"
    foreign.parent.mkdir(parents=True, exist_ok=True)
    foreign.write_bytes(b"a file that is not part of this corpus")
    config.scan_inbox.mkdir(parents=True, exist_ok=True)

    opened: list[Path] = []
    read_bytes = Path.read_bytes

    def _record(self: Path) -> bytes:
        opened.append(Path(self).resolve())
        return read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", _record)

    names = [
        str(foreign),
        os.path.relpath(foreign, config.scan_inbox),
        "nested/not-a-part.png",
        "..",
    ]
    for name in names:
        recipe_id = str(uuid.uuid4())
        _hand_written_receipt(
            config,
            recipe_id,
            {
                "version": 1,
                "recipe_id": recipe_id,
                "parts": [_receipt_entry(target_name=name)],
            },
        )
        with pytest.raises(SourcePartsError):
            load_source_part_receipt(config, recipe_id)

    assert foreign.resolve() not in opened
    # Nothing at all was opened: the name is refused before it becomes a path.
    assert opened == []
    assert foreign.read_bytes() == b"a file that is not part of this corpus"
