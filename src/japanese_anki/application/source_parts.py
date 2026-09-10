"""Rendering the owner's own page and region choices into immutable parts.

A *part* is a preserved source file (DESIGN.md, "The exception is one confirmed
extraction batch"): a whole document, or a derivative janki rendered from one
page or from explicitly reviewed regions of one page. This module is that
derivative's whole life — plan, receipt, publication — and nothing else. It
renders pixels. It never decides that a rectangle is a table, a row, or a word,
never reads a printed label, and never proposes geometry: coordinates arrive
from the owner's editor or from an owner-written ``--recipe`` file, both of
which mint a token bound to the exact recipe bytes this planner re-derives.

Three rules shape the code below.

**The receipt is written first.** ``execute_source_parts`` publishes nothing
until ``data/source_parts/<recipe_id>.json`` records every planned name and its
expected hash. A crash after that leaves an expectation an interrupted
publication can resume against; a crash before it leaves nothing at all. The
receipt is immutable — a second run either matches it exactly or refuses the
whole recipe.

**Publication is intake, not overwriting.** Parts land in
``config.scan_inbox`` because that is the root
``extraction_batch.plan_extraction_batch`` prepares its children from, while
the namesake check spans the wider ``extraction.durable_inbox_root`` so a part
cannot collide with a name elsewhere under ``data/inbox/``. Same name and same
bytes is a reuse; same name and *different* bytes refuses the whole recipe
rather than the one part, because half a recipe is not what the owner reviewed.
The parent document is never touched.

**PDFium runs in one bounded worker process.** The pypdfium2 API documentation
states that PDFium is not thread-safe, even across different documents, so
nothing here calls it from this process or from a thread: a spawned worker
reads one document, renders one page at a time, closes every handle, and
returns bytes. The optional ``sources`` extra is therefore imported nowhere on
the ordinary build or import path — :func:`sources_unavailable` reports its
absence the way ``card_preview.preview_unavailable`` does.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import multiprocessing
import os
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from japanese_anki import inputs
from japanese_anki.application.extraction import durable_inbox_root
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.identifiers import short_fingerprint
from japanese_anki.io import (
    atomic_write_text_bound,
    prepare_bound_directory,
    read_bytes_bound,
)

__all__ = [
    "MAX_PARTS",
    "MAX_RENDER_DPI",
    "MIN_RENDER_DPI",
    "SOURCE_PARTS_DIR_NAME",
    "ContactSheet",
    "ContactSheetPage",
    "PartSpec",
    "PlannedPart",
    "SourcePartRecipe",
    "SourcePartsError",
    "SourcePartsPlan",
    "SourcePartsReceipt",
    "execute_source_parts",
    "list_source_part_receipts",
    "load_source_part_receipt",
    "new_recipe_id",
    "plan_source_parts",
    "planned_receipt_sha256",
    "published_receipt_sha256",
    "recipe_from_bytes",
    "receipt_bytes",
    "receipt_path",
    "render_contact_sheet",
    "sources_unavailable",
]


class SourcePartsError(JankiError):
    """A source part could not be planned, rendered, or published."""


#: Beside the journal, exactly as ``extraction_batch.BATCH_DIR_NAME`` is, so no
#: ``[paths]`` key is added for it.
SOURCE_PARTS_DIR_NAME = "source_parts"

#: Wire version of both the recipe file and the receipt.
RECIPE_VERSION = 1

#: Domain separation for the token that binds geometry to the exact bytes one
#: of the two producers serialized. It is not a secret: no model-emittable
#: field can carry a recipe or a coordinate at all (contracts §9.1), so this
#: guards the one thing left — a recipe edited after it was reviewed.
_TOKEN_DOMAIN = "janki-source-part-recipe-v1"

#: Bounds, stated rather than discovered at render time. A recipe past one of
#: these refuses before a worker starts.
MAX_PARTS = 64
MAX_REGIONS_PER_PART = 8
MIN_RENDER_DPI = 72
MAX_RENDER_DPI = 600
#: One rendered page, in pixels. 80 megapixels is far past any study page and
#: far short of the memory a mistyped DPI on a plotter-sized page would ask for.
MAX_PAGE_PIXELS = 80_000_000
#: The worker renders one document; a bounded wait means a wedged PDFium call
#: fails the command instead of hanging the workbench.
RENDER_TIMEOUT_SECONDS = 180
#: Contact-sheet cell, long edge in pixels. A review aid, never published bytes.
THUMBNAIL_EDGE = 320
#: The editor's own sheet: every page, big enough to choose a band on and small
#: enough that a whole document fits in one self-contained document.
CONTACT_SHEET_DPI = 110
CONTACT_SHEET_EDGE = 1400
MAX_CONTACT_SHEET_PAGES = 40

_SAFE_STEM_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)


def sources_unavailable() -> str | None:
    """Why source parts cannot be rendered here, or ``None`` when they can.

    Spec-only, and naming top-level packages, for the reason
    ``card_preview.preview_unavailable`` is: importing ``pypdfium2`` loads a
    bundled PDFium binary, and the ordinary build and import path must not do
    that. The worker imports both packages for real, lazily, inside itself.
    """

    missing = [
        name
        for name in ("pypdfium2", "PIL")
        if importlib.util.find_spec(name) is None
    ]
    if not missing:
        return None
    shown = ", ".join("Pillow" if name == "PIL" else name for name in missing)
    return (
        "Rendering source parts needs the optional source dependencies "
        f"({shown}). Install them with: python -m pip install -e '.[sources]'"
    )


@dataclass(frozen=True, slots=True)
class PartSpec:
    """One part the owner asked for: a whole page, or regions of one page.

    ``regions`` is empty for a whole page and holds at least one rectangle
    otherwise. The distinction is made when a recipe is parsed, not here: a
    *present but empty* region list is a third form and refuses, because an
    editor that serialized an empty selection must not silently publish the
    whole page instead.

    Coordinates are normalized fractions in ``[0, 1]`` of the page **as
    rendered**, which is to say after ``/Rotate`` is applied, with the origin
    at its top-left corner. That makes a recipe stable under a DPI change and
    unambiguous under rotation.
    """

    page_index: int
    page_rotate: int
    regions: tuple[tuple[float, float, float, float], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "page_index": self.page_index,
            "page_rotate": self.page_rotate,
        }
        if self.regions:
            value["regions"] = [list(region) for region in self.regions]
        return value


@dataclass(frozen=True, slots=True)
class SourcePartRecipe:
    """The owner's complete instruction, and the only source of geometry."""

    recipe_id: str
    parent_name: str
    parent_sha256: str
    render_dpi: int
    parts: tuple[PartSpec, ...]
    renderer: str = ""
    renderer_version: str = ""
    encoder: str = ""
    encoder_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "version": RECIPE_VERSION,
            "recipe_id": self.recipe_id,
            "parent_name": self.parent_name,
            "parent_sha256": self.parent_sha256,
            "render_dpi": self.render_dpi,
            "parts": [part.to_dict() for part in self.parts],
        }
        for key, declared in (
            ("renderer", self.renderer),
            ("renderer_version", self.renderer_version),
            ("encoder", self.encoder),
            ("encoder_version", self.encoder_version),
        ):
            if declared:
                value[key] = declared
        return value


@dataclass(frozen=True, slots=True)
class PlannedPart:
    """One rendered part: what it is, where it came from, what it hashes to."""

    ordinal: int
    target_name: str
    sha256: str
    byte_length: int
    page_index: int
    page_rotate: int
    page_size_pt: tuple[float, float]
    pixel_rect: tuple[int, int, int, int]
    regions: tuple[tuple[float, float, float, float], ...]
    thumbnail_png_base64: str = ""

    def to_dict(self) -> dict[str, Any]:
        """The bound half of a part. The thumbnail is deliberately absent.

        A contact-sheet cell is a review aid rendered for a person to look at;
        the receipt binds the bytes that get published, and a thumbnail encoder
        change must not read as a different recipe.
        """

        return {
            "ordinal": self.ordinal,
            "target_name": self.target_name,
            "sha256": self.sha256,
            "byte_length": self.byte_length,
            "page_index": self.page_index,
            "page_rotate": self.page_rotate,
            "page_size_pt": [self.page_size_pt[0], self.page_size_pt[1]],
            "pixel_rect": list(self.pixel_rect),
            "regions": [list(region) for region in self.regions],
        }


@dataclass(frozen=True, slots=True)
class SourcePartsPlan:
    """Exactly what publishing this recipe would put in the corpus.

    ``payloads`` holds the exact rendered bytes of each planned part, in
    ``parts`` order. It never reaches the receipt or the plan fingerprint —
    the hash of each payload is what is bound — and it exists because the
    owner publishes the bytes they previewed rather than a second render.
    """

    recipe_id: str
    parent_name: str
    parent_sha256: str
    renderer: str
    renderer_version: str
    encoder: str
    encoder_version: str
    render_dpi: int
    plan_fingerprint: str
    parts: tuple[PlannedPart, ...]
    recipe_sha256: str = ""
    payloads: tuple[bytes, ...] = field(default=(), repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": RECIPE_VERSION,
            "recipe_id": self.recipe_id,
            "parent_name": self.parent_name,
            "parent_sha256": self.parent_sha256,
            "renderer": self.renderer,
            "renderer_version": self.renderer_version,
            "encoder": self.encoder,
            "encoder_version": self.encoder_version,
            "render_dpi": self.render_dpi,
            "recipe_sha256": self.recipe_sha256,
            "parts": [part.to_dict() for part in self.parts],
        }


@dataclass(frozen=True, slots=True)
class SourcePartsReceipt:
    """The immutable publication authority, and what this run actually did."""

    path: Path
    receipt_sha256: str
    plan: SourcePartsPlan
    published: tuple[Path, ...]
    reused: tuple[Path, ...]


# --- recipes ------------------------------------------------------------------


def canonical_recipe_bytes(recipe: SourcePartRecipe) -> bytes:
    """The one serialization a recipe token is minted over."""

    return (
        json.dumps(
            recipe.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _mint_recipe_token(recipe: SourcePartRecipe) -> str:
    payload = _TOKEN_DOMAIN.encode("utf-8") + b"\0" + canonical_recipe_bytes(recipe)
    return hashlib.sha256(payload).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourcePartsError(message)


def _fraction(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SourcePartsError(f"{label} must be a number between 0 and 1.")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise SourcePartsError(f"{label} must be between 0 and 1, not {number}.")
    # Rounded to the precision an editor can honestly claim, so a recipe's
    # bytes, its token and its fingerprint cannot differ by float noise.
    return round(number, 6)


def _parse_regions(raw: Any, *, where: str) -> tuple[tuple[float, float, float, float], ...]:
    if not isinstance(raw, list):
        raise SourcePartsError(
            f"{where}: 'regions' must be a list of [x0, y0, x1, y1] rectangles."
        )
    if not raw:
        raise SourcePartsError(
            f"{where}: 'regions' is present but empty. Leave 'regions' out to "
            "render the whole page, or name at least one region — an empty "
            "selection is not a whole-page request."
        )
    if len(raw) > MAX_REGIONS_PER_PART:
        raise SourcePartsError(
            f"{where}: {len(raw)} regions is more than the {MAX_REGIONS_PER_PART} "
            "one part composes. Split it into more parts."
        )
    regions: list[tuple[float, float, float, float]] = []
    previous_bottom = 0.0
    for index, entry in enumerate(raw):
        label = f"{where} region {index + 1}"
        if not isinstance(entry, list | tuple) or len(entry) != 4:
            raise SourcePartsError(f"{label} must be exactly [x0, y0, x1, y1].")
        x0, y0, x1, y1 = (
            _fraction(entry[0], f"{label} x0"),
            _fraction(entry[1], f"{label} y0"),
            _fraction(entry[2], f"{label} x1"),
            _fraction(entry[3], f"{label} y1"),
        )
        if x1 <= x0 or y1 <= y0:
            raise SourcePartsError(
                f"{label} has no area: x0 {x0} x1 {x1}, y0 {y0} y1 {y1}."
            )
        if y0 < previous_bottom:
            raise SourcePartsError(
                f"{label} starts at {y0}, above the previous region's bottom "
                f"{previous_bottom}. A part composes its regions top to bottom, "
                "so they may not overlap or run backwards."
            )
        previous_bottom = y1
        regions.append((x0, y0, x1, y1))
    return tuple(regions)


def _parse_part(raw: Any, *, index: int) -> PartSpec:
    where = f"Part {index + 1}"
    if not isinstance(raw, Mapping):
        raise SourcePartsError(f"{where} must be an object.")
    unknown = sorted(set(raw) - {"page_index", "page_rotate", "regions"})
    if unknown:
        raise SourcePartsError(
            f"{where} carries unknown key(s): {', '.join(unknown)}. A part is a "
            "page index, that page's recorded /Rotate, and an optional region "
            "list."
        )
    page_index = raw.get("page_index")
    if isinstance(page_index, bool) or not isinstance(page_index, int) or page_index < 0:
        raise SourcePartsError(f"{where}: 'page_index' must be a 0-based page number.")
    page_rotate = raw.get("page_rotate")
    if (
        isinstance(page_rotate, bool)
        or not isinstance(page_rotate, int)
        or page_rotate not in {0, 90, 180, 270}
    ):
        raise SourcePartsError(
            f"{where}: 'page_rotate' must record the page's own /Rotate as 0, "
            "90, 180 or 270."
        )
    regions = (
        _parse_regions(raw["regions"], where=where) if "regions" in raw else ()
    )
    return PartSpec(page_index=page_index, page_rotate=page_rotate, regions=regions)


def recipe_from_bytes(data: bytes) -> tuple[SourcePartRecipe, str]:
    """Parse one owner-produced recipe and mint the token bound to its bytes.

    The two producers contracts §3.2 allows — the inline region editor's
    serialized output and ``--recipe FILE`` — both come through here, and
    nothing else may hand :func:`plan_source_parts` a coordinate. The token is
    a digest of the parsed recipe's canonical form, so a rectangle widened
    between review and planning no longer matches the token it was reviewed
    under.
    """

    try:
        parsed = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise SourcePartsError(f"That source-part recipe is not readable JSON: {exc}") from exc
    if not isinstance(parsed, Mapping):
        raise SourcePartsError("A source-part recipe must be a JSON object.")
    known = {
        "version",
        "recipe_id",
        "parent_name",
        "parent_sha256",
        "render_dpi",
        "parts",
        "renderer",
        "renderer_version",
        "encoder",
        "encoder_version",
    }
    unknown = sorted(set(parsed) - known)
    if unknown:
        raise SourcePartsError(
            f"That recipe carries unknown key(s): {', '.join(unknown)}."
        )
    version = parsed.get("version", RECIPE_VERSION)
    if version != RECIPE_VERSION:
        raise SourcePartsError(
            f"That recipe is version {version!r}; janki writes and reads "
            f"version {RECIPE_VERSION}."
        )
    recipe_id = str(parsed.get("recipe_id") or "")
    if not _valid_recipe_id(recipe_id):
        raise SourcePartsError(
            "A recipe needs a 'recipe_id' that is a UUID, so it can never name "
            "a path outside its own store."
        )
    parent_name = str(parsed.get("parent_name") or "")
    _require(
        bool(parent_name) and inputs.safe_upload_name(parent_name) == parent_name,
        "A recipe's 'parent_name' must be the source's own filename.",
    )
    parent_sha256 = str(parsed.get("parent_sha256") or "").lower()
    _require(
        len(parent_sha256) == 64 and all(c in "0123456789abcdef" for c in parent_sha256),
        "A recipe's 'parent_sha256' must be the parent source's sha256 digest.",
    )
    render_dpi = parsed.get("render_dpi")
    if (
        isinstance(render_dpi, bool)
        or not isinstance(render_dpi, int)
        or not MIN_RENDER_DPI <= render_dpi <= MAX_RENDER_DPI
    ):
        raise SourcePartsError(
            f"A recipe's 'render_dpi' must be a whole number from {MIN_RENDER_DPI} "
            f"to {MAX_RENDER_DPI}."
        )
    raw_parts = parsed.get("parts")
    if not isinstance(raw_parts, list) or not raw_parts:
        raise SourcePartsError("A recipe must name at least one part.")
    if len(raw_parts) > MAX_PARTS:
        raise SourcePartsError(
            f"That recipe names {len(raw_parts)} parts; janki publishes at most "
            f"{MAX_PARTS} in one recipe."
        )
    parts = tuple(
        _parse_part(entry, index=index) for index, entry in enumerate(raw_parts)
    )
    recipe = SourcePartRecipe(
        recipe_id=recipe_id,
        parent_name=parent_name,
        parent_sha256=parent_sha256,
        render_dpi=render_dpi,
        parts=parts,
        renderer=str(parsed.get("renderer") or ""),
        renderer_version=str(parsed.get("renderer_version") or ""),
        encoder=str(parsed.get("encoder") or ""),
        encoder_version=str(parsed.get("encoder_version") or ""),
    )
    return recipe, _mint_recipe_token(recipe)


def _valid_recipe_id(recipe_id: str) -> bool:
    """A recipe id is a uuid, so it can never name a path outside its store."""

    try:
        return str(uuid.UUID(str(recipe_id))) == str(recipe_id)
    except (AttributeError, TypeError, ValueError):
        return False


def new_recipe_id() -> str:
    """One fresh recipe identity, for a producer serializing a new recipe."""

    return str(uuid.uuid4())


# --- planning -----------------------------------------------------------------


def _safe_stem(parent_name: str) -> str:
    stem = Path(parent_name).stem
    cleaned = "".join(
        character if character in _SAFE_STEM_CHARACTERS else "-" for character in stem
    ).strip("-.")
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    if not cleaned:
        raise SourcePartsError(
            f"{parent_name} has no name janki can derive a part filename from. "
            "Rename the source before preparing parts from it."
        )
    return cleaned


def _parent_source(config: ProjectConfig, source_name: str) -> tuple[Path, bytes]:
    """The exact preserved parent, read from the configured corpus root."""

    name = str(source_name)
    if not name or inputs.safe_upload_name(name) != name:
        raise SourcePartsError(
            f"{source_name!r} is not one source filename in your corpus."
        )
    if Path(name).suffix.lower() != ".pdf":
        raise SourcePartsError(
            f"{name} is not a PDF. Janki renders parts from PDF pages; an image "
            "is already one part and can be sent as it is."
        )
    root = config.scan_inbox.absolute()
    path = (root / name).absolute()
    if path.parent != root:
        raise SourcePartsError(f"{name} does not remain a direct corpus entry.")
    if path.is_symlink() or not path.is_file():
        raise SourcePartsError(
            f"{name} is not in your corpus yet, or is not a readable file. Add "
            "it first — adding a source and preparing parts from it are "
            "separate steps."
        )
    if not inputs.inside(path, root):
        raise SourcePartsError(f"{name} does not resolve inside your corpus.")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise SourcePartsError(f"Could not read {name}: {exc}") from exc
    if not data:
        raise SourcePartsError(f"{name} is empty; there is nothing to render.")
    return path, data


def plan_source_parts(
    config: ProjectConfig,
    source_name: str,
    recipe: SourcePartRecipe,
    *,
    recipe_token: str,
) -> SourcePartsPlan:
    """Render exactly what the recipe asks for, and bind it. Publish nothing.

    Every refusal that does not need the corpus written to happens here: a
    token that does not match these exact recipe bytes, a parent whose hash
    moved, a page whose ``/Rotate`` the recipe did not record, a DPI or part
    count past the stated bounds.
    """

    if not isinstance(recipe_token, str) or not recipe_token:
        raise SourcePartsError(
            "Preparing source parts needs the token its producer minted over "
            "the exact recipe bytes. Geometry reaches this planner from the "
            "region editor or from --recipe, and from nowhere else."
        )
    if recipe_token != _mint_recipe_token(recipe):
        raise SourcePartsError(
            "That recipe token was minted over different recipe bytes. Nothing "
            "was rendered: review the geometry again and publish from what you "
            "reviewed."
        )
    unavailable = sources_unavailable()
    if unavailable is not None:
        raise SourcePartsError(unavailable)

    path, parent_bytes = _parent_source(config, source_name)
    parent_sha256 = hashlib.sha256(parent_bytes).hexdigest()
    if recipe.parent_name != path.name:
        raise SourcePartsError(
            f"That recipe was written for {recipe.parent_name}, not {path.name}."
        )
    if recipe.parent_sha256 != parent_sha256:
        raise SourcePartsError(
            f"{path.name} is not the file this recipe was written over: it now "
            f"hashes to {parent_sha256}, and the recipe binds "
            f"{recipe.parent_sha256}. Prepare the regions again over the "
            "current source."
        )

    rendered = _render_parts(
        parent_bytes,
        parent_sha256=parent_sha256,
        render_dpi=recipe.render_dpi,
        parts=recipe.parts,
    )
    for declared, resolved, label in (
        (recipe.renderer, rendered["renderer"], "renderer"),
        (recipe.renderer_version, rendered["renderer_version"], "renderer version"),
        (recipe.encoder, rendered["encoder"], "encoder"),
        (recipe.encoder_version, rendered["encoder_version"], "encoder version"),
    ):
        if declared and declared != resolved:
            raise SourcePartsError(
                f"That recipe records {label} {declared!r}, and this checkout "
                f"renders with {resolved!r}. The pixels would not be the ones "
                "the recipe binds, so nothing was rendered."
            )

    recipe_sha256 = hashlib.sha256(canonical_recipe_bytes(recipe)).hexdigest()
    suffix = short_fingerprint(recipe_sha256, length=8)
    stem = _safe_stem(recipe.parent_name)
    planned: list[PlannedPart] = []
    payloads: list[bytes] = []
    for ordinal, (spec, result) in enumerate(
        zip(recipe.parts, rendered["parts"], strict=True), start=1
    ):
        data = base64.standard_b64decode(result["png_base64"])
        digest = hashlib.sha256(data).hexdigest()
        if digest != result["sha256"] or len(data) != int(result["byte_length"]):
            raise SourcePartsError(
                "The renderer returned bytes that do not match the hash it "
                "reported for them; nothing was published."
            )
        # The worker compared the page's own /Rotate against what the recipe
        # recorded before it rendered anything, which is the one place that
        # comparison can be made — this process never opens the document.
        planned.append(
            PlannedPart(
                ordinal=ordinal,
                target_name=(
                    f"{stem}--p{spec.page_index + 1:03d}-r{suffix}-{ordinal:02d}.png"
                ),
                sha256=digest,
                byte_length=len(data),
                page_index=spec.page_index,
                page_rotate=spec.page_rotate,
                page_size_pt=(
                    float(result["page_size_pt"][0]),
                    float(result["page_size_pt"][1]),
                ),
                pixel_rect=tuple(int(value) for value in result["pixel_rect"]),
                regions=spec.regions,
                thumbnail_png_base64=str(result.get("thumbnail_png_base64") or ""),
            )
        )
        payloads.append(data)

    names = [part.target_name for part in planned]
    if len(set(names)) != len(names):
        raise SourcePartsError(
            "Two parts of this recipe would be published under one name; "
            "nothing was rendered into your corpus."
        )
    plan = SourcePartsPlan(
        recipe_id=recipe.recipe_id,
        parent_name=recipe.parent_name,
        parent_sha256=parent_sha256,
        renderer=str(rendered["renderer"]),
        renderer_version=str(rendered["renderer_version"]),
        encoder=str(rendered["encoder"]),
        encoder_version=str(rendered["encoder_version"]),
        render_dpi=recipe.render_dpi,
        plan_fingerprint="",
        parts=tuple(planned),
        recipe_sha256=recipe_sha256,
        payloads=tuple(payloads),
    )
    return _with_fingerprint(plan)


def _with_fingerprint(plan: SourcePartsPlan) -> SourcePartsPlan:
    payload = json.dumps(
        plan.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    fingerprint = hashlib.sha256(
        b"janki-source-parts-plan-v1\0" + payload
    ).hexdigest()
    return SourcePartsPlan(
        recipe_id=plan.recipe_id,
        parent_name=plan.parent_name,
        parent_sha256=plan.parent_sha256,
        renderer=plan.renderer,
        renderer_version=plan.renderer_version,
        encoder=plan.encoder,
        encoder_version=plan.encoder_version,
        render_dpi=plan.render_dpi,
        plan_fingerprint=fingerprint,
        parts=plan.parts,
        recipe_sha256=plan.recipe_sha256,
        payloads=plan.payloads,
    )


# --- rendering, in one bounded worker process ---------------------------------


def _render_parts(
    parent_bytes: bytes,
    *,
    parent_sha256: str,
    render_dpi: int,
    parts: Sequence[PartSpec],
) -> dict[str, Any]:
    """Render one recipe's parts in the worker and return its exact answer."""

    return _run_worker(
        parent_bytes,
        {
            "version": RECIPE_VERSION,
            "kind": "parts",
            "parent_sha256": parent_sha256,
            "render_dpi": render_dpi,
            "max_page_pixels": MAX_PAGE_PIXELS,
            "thumbnail_edge": THUMBNAIL_EDGE,
            "parts": [part.to_dict() for part in parts],
        },
    )


def _run_worker(parent_bytes: bytes, request: Mapping[str, Any]) -> dict[str, Any]:
    """Run the renderer in one spawned worker and return its exact answer.

    PDFium is not thread-safe even across documents, so this process never
    imports it: the child does, renders one document one page at a time, and
    dies. The request and the answer travel through files in a private
    temporary directory rather than a pipe, so a large page cannot deadlock a
    parent that is waiting on ``join``.
    """

    context = multiprocessing.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix="janki-source-parts-") as work:
        workdir = Path(work)
        (workdir / "parent.bin").write_bytes(parent_bytes)
        (workdir / "request.json").write_text(
            json.dumps(dict(request), ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        process = context.Process(
            target=render_source_parts_worker,
            args=(str(workdir),),
            name="janki-source-parts",
        )
        process.start()
        process.join(RENDER_TIMEOUT_SECONDS)
        timed_out = process.is_alive()
        if timed_out:  # pragma: no cover - a wedged PDFium call
            process.terminate()
            process.join(10)
            if process.is_alive():
                process.kill()
                process.join(10)
        exitcode = process.exitcode
        process.close()
        if timed_out:  # pragma: no cover - a wedged PDFium call
            raise SourcePartsError(
                f"The source-part renderer did not finish within "
                f"{RENDER_TIMEOUT_SECONDS}s; nothing was published."
            )
        error_path = workdir / "error.json"
        result_path = workdir / "result.json"
        if error_path.exists():
            detail = json.loads(error_path.read_text(encoding="utf-8"))
            raise SourcePartsError(str(detail.get("message") or "The render failed."))
        if not result_path.exists():
            raise SourcePartsError(
                "The source-part renderer stopped without an answer "
                f"(exit {exitcode}); nothing was published."
            )
        return json.loads(result_path.read_text(encoding="utf-8"))


def render_source_parts_worker(workdir: str) -> None:
    """The spawned child's whole job: render one document, then exit.

    Module level on purpose. A spawn child re-imports its target's module, and
    under pytest ``__main__`` is the test runner — a nested function or a
    ``__main__`` entry point would not survive the trip.
    """

    directory = Path(workdir)
    try:
        payload = _render_in_worker(directory)
    except BaseException as exc:  # noqa: BLE001 - the child's only report channel
        (directory / "error.json").write_text(
            json.dumps({"message": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False),
            encoding="utf-8",
        )
        return
    (directory / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )


def _render_in_worker(directory: Path) -> dict[str, Any]:
    request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    parent_bytes = (directory / "parent.bin").read_bytes()
    if hashlib.sha256(parent_bytes).hexdigest() != request["parent_sha256"]:
        raise SourcePartsError("The parent source changed while it was being read.")
    if str(request.get("kind")) == "contact_sheet":
        return _contact_sheet_in_worker(request, parent_bytes)
    return _parts_in_worker(request, parent_bytes)


def _parts_in_worker(request: Mapping[str, Any], parent_bytes: bytes) -> dict[str, Any]:
    import pypdfium2  # noqa: PLC0415 - the isolation this module exists for
    from PIL import Image  # noqa: PLC0415

    render_dpi = int(request["render_dpi"])
    max_page_pixels = int(request["max_page_pixels"])
    thumbnail_edge = int(request["thumbnail_edge"])
    specs = [
        (
            int(entry["page_index"]),
            int(entry["page_rotate"]),
            [tuple(float(value) for value in region) for region in entry.get("regions", ())],
        )
        for entry in request["parts"]
    ]

    renderer_version = (
        f"{pypdfium2.PYPDFIUM_INFO}+pdfium.{pypdfium2.PDFIUM_INFO}"
    )
    results: list[dict[str, Any] | None] = [None] * len(specs)
    scale = render_dpi / 72.0
    document = pypdfium2.PdfDocument(parent_bytes)
    try:
        page_count = len(document)
        for page_index in sorted({spec[0] for spec in specs}):
            if page_index >= page_count:
                raise SourcePartsError(
                    f"This source has {page_count} page(s); the recipe asks for "
                    f"page {page_index + 1}."
                )
            page = document.get_page(page_index)
            bitmap = None
            image = None
            try:
                rotation = int(page.get_rotation())
                width_pt, height_pt = (float(value) for value in page.get_size())
                declared = {
                    spec[1] for spec in specs if spec[0] == page_index
                }
                if declared != {rotation}:
                    raise SourcePartsError(
                        f"Page {page_index + 1} rotates {rotation}°, and the "
                        f"recipe recorded {sorted(declared)}°. Prepare the "
                        "regions again over the page as it is."
                    )
                if (width_pt * scale) * (height_pt * scale) > max_page_pixels:
                    raise SourcePartsError(
                        f"Page {page_index + 1} at {render_dpi} DPI is larger "
                        f"than the {max_page_pixels} pixels janki renders. Use "
                        "a lower DPI."
                    )
                bitmap = page.render(scale=scale)
                image = bitmap.to_pil().convert("RGB")
                for ordinal, (spec_page, _rotate, regions) in enumerate(specs):
                    if spec_page != page_index:
                        continue
                    results[ordinal] = _compose_part(
                        Image,
                        image,
                        regions=regions,
                        page_size_pt=(width_pt, height_pt),
                        rotation=rotation,
                        thumbnail_edge=thumbnail_edge,
                    )
            finally:
                if image is not None:
                    image.close()
                if bitmap is not None:
                    bitmap.close()
                page.close()
    finally:
        document.close()

    missing = [index for index, value in enumerate(results) if value is None]
    if missing:  # pragma: no cover - every part names a page that was rendered
        raise SourcePartsError(f"The renderer skipped part(s) {missing}.")
    return {
        "renderer": "pypdfium2",
        "renderer_version": renderer_version,
        "encoder": "Pillow",
        "encoder_version": str(_pillow_version()),
        "parts": results,
    }


def _contact_sheet_in_worker(
    request: Mapping[str, Any], parent_bytes: bytes
) -> dict[str, Any]:
    """Every page, whole, bounded — the sheet the owner draws their regions on.

    Deliberately the same document read as a part render: what the editor
    displays and what a recipe crops are the same pixels at two scales, so a
    rectangle drawn here means the same thing there.
    """

    import pypdfium2  # noqa: PLC0415 - the isolation this module exists for

    render_dpi = int(request["render_dpi"])
    max_edge = int(request["max_edge"])
    max_pages = int(request["max_pages"])
    max_page_pixels = int(request["max_page_pixels"])
    scale = render_dpi / 72.0
    pages: list[dict[str, Any]] = []
    document = pypdfium2.PdfDocument(parent_bytes)
    try:
        page_count = len(document)
        if page_count > max_pages:
            raise SourcePartsError(
                f"This source has {page_count} pages; janki draws a contact "
                f"sheet for up to {max_pages}. Split the document first."
            )
        for page_index in range(page_count):
            page = document.get_page(page_index)
            bitmap = None
            image = None
            try:
                rotation = int(page.get_rotation())
                width_pt, height_pt = (float(value) for value in page.get_size())
                if (width_pt * scale) * (height_pt * scale) > max_page_pixels:
                    raise SourcePartsError(
                        f"Page {page_index + 1} at {render_dpi} DPI is larger "
                        f"than the {max_page_pixels} pixels janki renders."
                    )
                bitmap = page.render(scale=scale)
                image = bitmap.to_pil().convert("RGB")
                image.thumbnail((max_edge, max_edge))
                pages.append(
                    {
                        "page_index": page_index,
                        "page_rotate": rotation,
                        "page_size_pt": [round(width_pt, 4), round(height_pt, 4)],
                        "width": image.width,
                        "height": image.height,
                        "png_base64": base64.standard_b64encode(
                            _encode_png(image)
                        ).decode("ascii"),
                    }
                )
            finally:
                if image is not None:
                    image.close()
                if bitmap is not None:
                    bitmap.close()
                page.close()
    finally:
        document.close()
    return {
        "renderer": "pypdfium2",
        "renderer_version": (
            f"{pypdfium2.PYPDFIUM_INFO}+pdfium.{pypdfium2.PDFIUM_INFO}"
        ),
        "encoder": "Pillow",
        "encoder_version": str(_pillow_version()),
        "pages": pages,
    }


def _pillow_version() -> str:
    import PIL  # noqa: PLC0415 - worker-side only

    return str(PIL.__version__)


def _compose_part(
    image_module: Any,
    page_image: Any,
    *,
    regions: Sequence[Sequence[float]],
    page_size_pt: tuple[float, float],
    rotation: int,
    thumbnail_edge: int,
) -> dict[str, Any]:
    """One part's exact PNG bytes, composed from the rendered page.

    The rendered page is the page **after** ``/Rotate``: pypdfium2 applies the
    document's own rotation, so a normalized rectangle means the same pixels a
    person saw in the editor. Regions are cropped from that image and stacked
    top to bottom, left-aligned on white — no scaling, no reflow, and no
    judgement about what any band contains.
    """

    width, height = page_image.size
    if not regions:
        crops = [(0, 0, width, height)]
    else:
        crops = []
        for x0, y0, x1, y1 in regions:
            left = min(max(int(round(x0 * width)), 0), width)
            top = min(max(int(round(y0 * height)), 0), height)
            right = min(max(int(round(x1 * width)), 0), width)
            bottom = min(max(int(round(y1 * height)), 0), height)
            if right <= left or bottom <= top:
                raise SourcePartsError(
                    "A region rounds to no pixels at this DPI: "
                    f"[{x0}, {y0}, {x1}, {y1}] on a {width}x{height} page."
                )
            crops.append((left, top, right, bottom))

    tiles = [page_image.crop(box) for box in crops]
    try:
        if len(tiles) == 1:
            composed = tiles[0].copy()
        else:
            canvas_width = max(tile.width for tile in tiles)
            canvas_height = sum(tile.height for tile in tiles)
            composed = image_module.new(
                "RGB", (canvas_width, canvas_height), (255, 255, 255)
            )
            offset = 0
            for tile in tiles:
                composed.paste(tile, (0, offset))
                offset += tile.height
    finally:
        for tile in tiles:
            tile.close()

    try:
        data = _encode_png(composed)
        thumbnail = composed.copy()
        try:
            thumbnail.thumbnail((thumbnail_edge, thumbnail_edge))
            thumbnail_bytes = _encode_png(thumbnail)
        finally:
            thumbnail.close()
    finally:
        composed.close()

    pixel_rect = (
        min(box[0] for box in crops),
        min(box[1] for box in crops),
        max(box[2] for box in crops),
        max(box[3] for box in crops),
    )
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "byte_length": len(data),
        "png_base64": base64.standard_b64encode(data).decode("ascii"),
        "thumbnail_png_base64": base64.standard_b64encode(thumbnail_bytes).decode("ascii"),
        "page_rotate": rotation,
        "page_size_pt": [round(page_size_pt[0], 4), round(page_size_pt[1], 4)],
        "pixel_rect": list(pixel_rect),
    }


def _encode_png(image: Any) -> bytes:
    """Encode with the settings written down, so two runs agree byte for byte."""

    import io  # noqa: PLC0415 - worker-side only

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=6)
    return buffer.getvalue()


# --- the contact sheet the owner chooses regions on ---------------------------


@dataclass(frozen=True, slots=True)
class ContactSheetPage:
    """One rendered page, as displayed: after ``/Rotate``, bounded in pixels."""

    page_index: int
    page_rotate: int
    page_size_pt: tuple[float, float]
    width: int
    height: int
    png_base64: str


@dataclass(frozen=True, slots=True)
class ContactSheet:
    """Every page of one parent, rendered for review. Nothing is published."""

    parent_name: str
    parent_sha256: str
    render_dpi: int
    renderer: str
    renderer_version: str
    encoder: str
    encoder_version: str
    pages: tuple[ContactSheetPage, ...]


def render_contact_sheet(
    config: ProjectConfig,
    source_name: str,
    *,
    render_dpi: int = CONTACT_SHEET_DPI,
    max_edge: int = CONTACT_SHEET_EDGE,
) -> ContactSheet:
    """Render every page of one corpus source for the owner to look at.

    A read: it writes nothing, publishes nothing, mints no receipt and reaches
    no provider. The editor needs the real pixels — a region chosen over a
    schematic would not be the region that gets cropped.
    """

    unavailable = sources_unavailable()
    if unavailable is not None:
        raise SourcePartsError(unavailable)
    if (
        isinstance(render_dpi, bool)
        or not isinstance(render_dpi, int)
        or not MIN_RENDER_DPI <= render_dpi <= MAX_RENDER_DPI
    ):
        raise SourcePartsError(
            f"A contact sheet renders between {MIN_RENDER_DPI} and "
            f"{MAX_RENDER_DPI} DPI."
        )
    path, parent_bytes = _parent_source(config, source_name)
    parent_sha256 = hashlib.sha256(parent_bytes).hexdigest()
    rendered = _run_worker(
        parent_bytes,
        {
            "version": RECIPE_VERSION,
            "kind": "contact_sheet",
            "parent_sha256": parent_sha256,
            "render_dpi": render_dpi,
            "max_edge": max_edge,
            "max_pages": MAX_CONTACT_SHEET_PAGES,
            "max_page_pixels": MAX_PAGE_PIXELS,
        },
    )
    return ContactSheet(
        parent_name=path.name,
        parent_sha256=parent_sha256,
        render_dpi=render_dpi,
        renderer=str(rendered["renderer"]),
        renderer_version=str(rendered["renderer_version"]),
        encoder=str(rendered["encoder"]),
        encoder_version=str(rendered["encoder_version"]),
        pages=tuple(
            ContactSheetPage(
                page_index=int(page["page_index"]),
                page_rotate=int(page["page_rotate"]),
                page_size_pt=(
                    float(page["page_size_pt"][0]),
                    float(page["page_size_pt"][1]),
                ),
                width=int(page["width"]),
                height=int(page["height"]),
                png_base64=str(page["png_base64"]),
            )
            for page in rendered["pages"]
        ),
    )


# --- the receipt, then publication --------------------------------------------


def receipt_path(config: ProjectConfig, recipe_id: str) -> Path:
    """The one durable receipt path for a recipe id."""

    if not _valid_recipe_id(recipe_id):
        raise SourcePartsError(f"Not a source-part recipe identity: {recipe_id!r}")
    return config.operations_file.parent / SOURCE_PARTS_DIR_NAME / f"{recipe_id}.json"


def receipt_bytes(plan: SourcePartsPlan, *, created_at: str) -> bytes:
    """The exact durable receipt bytes this plan writes at this stamp.

    Public because a caller that must bind the receipt's hash *before* the
    receipt exists has no other way to know it: ``created_at`` is inside the
    hashed JSON, so the hash is a function of the stamp as well as the plan.
    """

    value = plan.to_dict()
    value["plan_fingerprint"] = plan.plan_fingerprint
    value["created_at"] = created_at
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def planned_receipt_sha256(plan: SourcePartsPlan, *, created_at: str) -> str:
    """The hash a first publication of this plan at this stamp would land."""

    return hashlib.sha256(receipt_bytes(plan, created_at=created_at)).hexdigest()


def _proved_receipt(path: Path, payload: bytes) -> bytes:
    """The receipt already at ``path``, proved to be this payload's own.

    The immutability rule itself: a receipt whose plan fingerprint or part
    list differs from this one names different parts, and different parts need
    a new recipe id. ``created_at`` is the only field a resumed run may differ
    in, so it is compared out rather than byte-matched — every binding a
    publication is checked against is inside ``plan_fingerprint``.

    Readable JSON is not yet a receipt: a hand edit or a truncation can leave
    a list, a scalar or ``null`` at the top level, and comparing that one's
    fields raises ``AttributeError``, which is not a ``JankiError``. Every
    caller of this function is a refusal route written against one —
    ``cli.main`` prints ``error:`` for a ``JankiError`` and lets anything else
    out as a traceback — so the shape is proved here, exactly as
    :func:`load_source_part_receipt` proves it for the reader.
    """

    try:
        held = read_bytes_bound(path)
    except (JankiError, OSError) as exc:
        raise SourcePartsError(
            f"Could not read the existing source-part receipt {path}: {exc}"
        ) from exc
    try:
        existing = json.loads(held.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise SourcePartsError(
            f"The source-part receipt {path} is not readable JSON: {exc}. "
            "Nothing was published."
        ) from exc
    if not isinstance(existing, Mapping):
        raise SourcePartsError(
            f"The source-part receipt {path} is not a JSON object, so it "
            "records no recipe this one can be proved against. Nothing was "
            "published."
        )
    current = json.loads(payload.decode("utf-8"))
    if {key: value for key, value in existing.items() if key != "created_at"} != {
        key: value for key, value in current.items() if key != "created_at"
    }:
        raise SourcePartsError(
            f"The source-part receipt {path} records a different recipe than "
            "this one. A receipt is immutable: different parts need a new "
            "recipe id and a new receipt, and nothing was published."
        )
    return held


def published_receipt_sha256(
    config: ProjectConfig, plan: SourcePartsPlan
) -> str | None:
    """The hash of the receipt already published for this exact plan.

    ``None`` when this recipe id has no receipt yet. When one exists it is
    proved to be *this* plan's by the same comparison the writer performs, so
    a caller that must bind an authority before publishing binds it to a
    receipt :func:`execute_source_parts` will accept rather than to whatever
    happens to carry that recipe id. A plan the writer would refuse refuses
    here too, with the same words and before anything is recorded.
    """

    path = receipt_path(config, plan.recipe_id)
    if not path.exists():
        return None
    return hashlib.sha256(
        _proved_receipt(path, receipt_bytes(plan, created_at=""))
    ).hexdigest()


def _write_receipt(
    config: ProjectConfig, plan: SourcePartsPlan, *, created_at: str = ""
) -> tuple[Path, str, bool]:
    """Publish the expectation, or prove the one on disk is already it.

    Written **before** the first part, so an interrupted publication has an
    exact expectation to resume against, and immutable — see
    :func:`_proved_receipt`.

    An empty ``created_at`` stamps the clock here; a caller that froze the
    stamp at prepare time passes it so the hash it bound is the hash that
    lands.
    """

    path = receipt_path(config, plan.recipe_id)
    payload = receipt_bytes(
        plan, created_at=created_at or datetime.now(UTC).isoformat()
    )
    if path.exists():
        held = _proved_receipt(path, payload)
        return path, hashlib.sha256(held).hexdigest(), False
    prepare_bound_directory(config.operations_file.parent / SOURCE_PARTS_DIR_NAME)
    try:
        atomic_write_text_bound(path, payload.decode("utf-8"), expected_absent=True)
    except JankiError as exc:
        raise SourcePartsError(
            f"Could not write the source-part receipt {path}: {exc}"
        ) from exc
    return path, hashlib.sha256(payload).hexdigest(), True


def execute_source_parts(
    config: ProjectConfig,
    plan: SourcePartsPlan,
    *,
    publish_token: str,
    created_at: str = "",
) -> SourcePartsReceipt:
    """Write the receipt, then put every absent part in the corpus.

    ``publish_token`` is the owner's explicit publish action, bound to the
    exact ``plan_fingerprint`` they previewed. A plan alone publishes nothing.
    Ordinary local work: no paid call, no provider disclosure, no edit of an
    original, and no new approval gate.

    ``created_at`` is the stamp a caller froze before it bound this receipt's
    hash — the same freeze ``coverage._approval_payload(approved_at=…)`` and
    ``ledger.record_export(at=)`` already make. Empty is this module's own
    clock, which is what every caller with nothing to bind passes.
    """

    if not isinstance(publish_token, str) or publish_token != plan.plan_fingerprint:
        raise SourcePartsError(
            "Publishing needs the exact plan fingerprint the owner reviewed. "
            "Nothing was published."
        )
    if len(plan.payloads) != len(plan.parts):
        raise SourcePartsError(
            "This plan no longer carries the bytes it was rendered from; render "
            "the recipe again before publishing."
        )
    for part, data in zip(plan.parts, plan.payloads, strict=True):
        if hashlib.sha256(data).hexdigest() != part.sha256:
            raise SourcePartsError(
                f"The rendered bytes for {part.target_name} do not match the "
                "hash this plan binds; nothing was published."
            )

    path, receipt_sha256, _fresh = _write_receipt(config, plan, created_at=created_at)
    intakes = inputs.publish_derived_part(
        [(part.target_name, data) for part, data in zip(plan.parts, plan.payloads, strict=True)],
        scan_inbox=config.scan_inbox,
        inbox_root=durable_inbox_root(config),
    )
    published = tuple(intake.path for intake in intakes if intake.stored)
    reused = tuple(intake.path for intake in intakes if not intake.stored)
    return SourcePartsReceipt(
        path=path,
        receipt_sha256=receipt_sha256,
        plan=plan,
        published=published,
        reused=reused,
    )


# --- reading receipts back ----------------------------------------------------


@dataclass(frozen=True, slots=True)
class SourcePartRecord:
    """One part as a receipt records it, plus whether it is in the corpus now.

    ``published`` is ``None`` when the corpus was not consulted — see
    :func:`load_source_part_receipt`'s ``verify_published``. Unknown, not
    absent: the only reader that discloses this state is the snapshot, and it
    always measures.
    """

    ordinal: int
    target_name: str
    sha256: str
    byte_length: int
    page_index: int
    page_rotate: int
    published: bool | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "target_name": self.target_name,
            "sha256": self.sha256,
            "byte_length": self.byte_length,
            "page_index": self.page_index,
            "page_rotate": self.page_rotate,
            "published": self.published,
        }


@dataclass(frozen=True, slots=True)
class SourcePartsRecord:
    """A receipt read back: ids, hashes, versions and counts. No bytes."""

    recipe_id: str
    receipt_sha256: str
    parent_name: str
    parent_sha256: str
    renderer: str
    renderer_version: str
    encoder: str
    encoder_version: str
    render_dpi: int
    plan_fingerprint: str
    recipe_sha256: str
    created_at: str
    parts: tuple[SourcePartRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe_id": self.recipe_id,
            "receipt_sha256": self.receipt_sha256,
            "parent_name": self.parent_name,
            "parent_sha256": self.parent_sha256,
            "renderer": self.renderer,
            "renderer_version": self.renderer_version,
            "encoder": self.encoder,
            "encoder_version": self.encoder_version,
            "render_dpi": self.render_dpi,
            "plan_fingerprint": self.plan_fingerprint,
            "recipe_sha256": self.recipe_sha256,
            "created_at": self.created_at,
            "part_count": len(self.parts),
            "published_count": sum(1 for part in self.parts if part.published),
            "parts": [part.to_dict() for part in self.parts],
        }


def _part_is_published(config: ProjectConfig, name: str, sha256: str) -> bool:
    """Whether the corpus holds exactly these bytes under this name.

    ``name`` reaches here already checked as a corpus basename by
    :func:`_receipt_part_name`, which is what makes the join below a file in
    ``scan_inbox`` rather than wherever the receipt asked for.
    """

    path = config.scan_inbox / name
    try:
        if path.is_symlink() or not path.is_file():
            return False
        return hashlib.sha256(path.read_bytes()).hexdigest() == sha256
    except OSError:
        return False


def _receipt_text(recipe_id: str, entry: Mapping[str, Any], key: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str):
        raise SourcePartsError(
            f"The source-part receipt for {recipe_id} records a part whose "
            f"{key} is not text."
        )
    return value


def _receipt_int(
    recipe_id: str, entry: Mapping[str, Any], key: str, *, subject: str = "a part"
) -> int:
    value = entry.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SourcePartsError(
            f"The source-part receipt for {recipe_id} records {subject} whose "
            f"{key} is not a whole number."
        )
    return value


def _receipt_envelope_int(recipe_id: str, parsed: Mapping[str, Any], key: str) -> int:
    """An envelope whole number, typed exactly the way a part's fields are.

    The envelope's text fields read an absent or null value as ``""``, and this
    keeps that shape at 0; everything else clears :func:`_receipt_int`'s bar.
    ``int(parsed.get(key) or 0)`` did neither: an array or object raised
    ``TypeError`` and text raised ``ValueError``, both outside
    ``SourcePartsError`` and so outside every guard this reader's contract
    names, while ``"200"`` and ``true`` were silently coerced into numbers the
    same reader refuses in a part.
    """

    if parsed.get(key) is None:
        return 0
    return _receipt_int(recipe_id, parsed, key, subject="a recipe")


def _receipt_part_name(recipe_id: str, entry: Mapping[str, Any]) -> str:
    """The part's own name, re-checked as a basename before it becomes a path.

    ``config.scan_inbox / name`` *is* ``name`` when the name is absolute, and
    leaves the corpus the ordinary way when it holds ``..``. Publication
    already re-checks every planned name against ``inputs.safe_upload_name``
    before it writes one, and the read side has to clear the same bar before
    anything opens or hashes it — a hand-edited or corrupted receipt is exactly
    the case this reader exists to survive, and it must not become a reader of
    arbitrary files.
    """

    name = _receipt_text(recipe_id, entry, "target_name")
    try:
        checked = inputs.safe_upload_name(name)
    except inputs.InputError as exc:
        raise SourcePartsError(
            f"The source-part receipt for {recipe_id} names {name!r}, which is "
            f"not a corpus filename: {exc}"
        ) from exc
    if checked != name:
        raise SourcePartsError(
            f"The source-part receipt for {recipe_id} names {name!r}, which is "
            "a path rather than one filename in the scan inbox."
        )
    return name


def _receipt_part(
    config: ProjectConfig,
    recipe_id: str,
    entry: Any,
    *,
    verify_published: bool,
) -> SourcePartRecord:
    """One validated part entry. Every field is checked, none is indexed."""

    if not isinstance(entry, Mapping):
        raise SourcePartsError(
            f"The source-part receipt for {recipe_id} records a part that is "
            "not an object."
        )
    name = _receipt_part_name(recipe_id, entry)
    sha256 = _receipt_text(recipe_id, entry, "sha256")
    return SourcePartRecord(
        ordinal=_receipt_int(recipe_id, entry, "ordinal"),
        target_name=name,
        sha256=sha256,
        byte_length=_receipt_int(recipe_id, entry, "byte_length"),
        page_index=_receipt_int(recipe_id, entry, "page_index"),
        page_rotate=_receipt_int(recipe_id, entry, "page_rotate"),
        published=(
            _part_is_published(config, name, sha256) if verify_published else None
        ),
    )


def load_source_part_receipt(
    config: ProjectConfig, recipe_id: str, *, verify_published: bool = True
) -> SourcePartsRecord:
    """Read one receipt back, validating every entry it records.

    A receipt is machine-written and immutable, so an entry that is missing a
    key, carries the wrong type, or is not an object at all means a hand edit
    or corruption. That has to refuse as a ``SourcePartsError``, the way
    unreadable JSON already does: every caller's guard — catalog discovery,
    ``snapshot``, the broker constructor — is written against ``JankiError``,
    and a raw ``KeyError`` or ``TypeError`` escapes all three and takes
    unrelated library objects down with it. The envelope beside the entries is
    hand-editable in exactly the same way, so its numbers are read through the
    same typed helper rather than coerced.

    ``verify_published`` decides whether the corpus is consulted for each part.
    Publication state is real state and is measured rather than remembered, but
    measuring it reads and hashes every published part, so only a caller that
    discloses it asks for it: the snapshot of the receipt the owner selected,
    and ``janki source-parts status``. Catalog discovery titles an entry from
    the receipt alone, because a broker is built for every ordinary Assistant
    turn and receipts are never retired.
    """

    path = receipt_path(config, recipe_id)
    try:
        held = read_bytes_bound(path)
    except (JankiError, OSError) as exc:
        raise SourcePartsError(
            f"Could not read the source-part receipt for {recipe_id}: {exc}"
        ) from exc
    try:
        parsed = json.loads(held.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise SourcePartsError(
            f"The source-part receipt for {recipe_id} is not readable JSON: {exc}"
        ) from exc
    if not isinstance(parsed, Mapping) or parsed.get("recipe_id") != recipe_id:
        raise SourcePartsError(
            f"The source-part receipt at {path} does not record {recipe_id}."
        )
    raw_parts = parsed.get("parts")
    if not isinstance(raw_parts, list):
        raise SourcePartsError(
            f"The source-part receipt for {recipe_id} records no parts."
        )
    parts = tuple(
        _receipt_part(config, recipe_id, entry, verify_published=verify_published)
        for entry in raw_parts
    )
    return SourcePartsRecord(
        recipe_id=recipe_id,
        receipt_sha256=hashlib.sha256(held).hexdigest(),
        parent_name=str(parsed.get("parent_name") or ""),
        parent_sha256=str(parsed.get("parent_sha256") or ""),
        renderer=str(parsed.get("renderer") or ""),
        renderer_version=str(parsed.get("renderer_version") or ""),
        encoder=str(parsed.get("encoder") or ""),
        encoder_version=str(parsed.get("encoder_version") or ""),
        render_dpi=_receipt_envelope_int(recipe_id, parsed, "render_dpi"),
        plan_fingerprint=str(parsed.get("plan_fingerprint") or ""),
        recipe_sha256=str(parsed.get("recipe_sha256") or ""),
        created_at=str(parsed.get("created_at") or ""),
        parts=parts,
    )


def list_source_part_receipts(config: ProjectConfig) -> tuple[str, ...]:
    """Every recipe id with a receipt, in sorted order. Bad siblings are skipped."""

    directory = config.operations_file.parent / SOURCE_PARTS_DIR_NAME
    try:
        with os.scandir(directory) as scan:
            entries = sorted(entry.name for entry in scan if entry.is_file())
    except (FileNotFoundError, NotADirectoryError, OSError):
        return ()
    found: list[str] = []
    for name in entries:
        if not name.endswith(".json"):
            continue
        recipe_id = name[: -len(".json")]
        if _valid_recipe_id(recipe_id):
            found.append(recipe_id)
    return tuple(found)


