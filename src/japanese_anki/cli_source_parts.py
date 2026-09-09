"""``janki source-parts`` — the same preparation the Assistant's editor does.

The CLI is a fully supported surface over the same application service, not a
reduced one: `choose` renders the real pages so a person can pick bands,
`prepare` plans exactly what a recipe asks for and prints every name and hash,
and `--publish` binds the fingerprint that plan printed. Geometry reaches the
service from this command's `--recipe FILE` or from the editor, and from
nowhere else.

Preparing parts is ordinary local work — no paid call, no provider disclosure,
no edit of an original — so there is deliberately no consent prompt and no
`--yes` here. The one thing this surface will not do is publish something other
than the plan it showed.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any


def _core() -> Any:
    return importlib.import_module("japanese_anki.application.source_parts")


def _editor() -> Any:
    return importlib.import_module("japanese_anki.workbench.source_part_editor")


def _describe_part(part: Any) -> list[str]:
    regions = (
        "the whole page"
        if not part.regions
        else "; ".join(
            "[" + ", ".join(f"{value:.6g}" for value in region) + "]"
            for region in part.regions
        )
    )
    return [
        f"  {part.ordinal}. {part.target_name}",
        f"     page {part.page_index + 1} (/Rotate {part.page_rotate}), {regions}",
        f"     {part.byte_length} bytes, sha256 {part.sha256}",
    ]


def _command_choose(config: Any, args: argparse.Namespace) -> int:
    """Render every page of one source into a self-contained chooser."""

    core = _core()
    unavailable = core.sources_unavailable()
    if unavailable is not None:
        print(unavailable)
        return 1
    sheet = core.render_contact_sheet(config, args.source, render_dpi=args.dpi)
    document = _editor().render_source_part_editor(
        sheet,
        recipe_id=core.new_recipe_id(),
        submit_url="",
        default_render_dpi=args.publication_dpi,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(document.html)
    print(
        f"Wrote {len(sheet.pages)} rendered page(s) of {sheet.parent_name} to "
        f"{output}."
    )
    print(
        "Choose pages and regions there, copy the recipe it writes, and run "
        "`janki source-parts prepare --source "
        f"{sheet.parent_name} --recipe FILE`. Nothing was published, and the "
        "source is unchanged."
    )
    return 0


def _command_prepare(config: Any, args: argparse.Namespace) -> int:
    """Plan one recipe, print every bound name and hash, publish on request."""

    core = _core()
    unavailable = core.sources_unavailable()
    if unavailable is not None:
        print(unavailable)
        return 1
    # `cli.main` formats `JankiError` and nothing else, so an owner-supplied
    # path gets its OS failure named here rather than reaching them as a
    # traceback — the rule `inputs._read` states and `csv_base.inspect_file`
    # follows. A missing file is the common typo for this command's only input.
    try:
        written_recipe = Path(args.recipe).read_bytes()
    except OSError as exc:
        raise core.SourcePartsError(
            f"Could not read the recipe file {args.recipe}: {exc}"
        ) from exc
    recipe, token = core.recipe_from_bytes(written_recipe)
    plan = core.plan_source_parts(config, args.source, recipe, recipe_token=token)
    print(f"Recipe {plan.recipe_id} over {plan.parent_name} ({plan.parent_sha256}).")
    print(
        f"Rendered by {plan.renderer} {plan.renderer_version}, encoded by "
        f"{plan.encoder} {plan.encoder_version}, at {plan.render_dpi} DPI."
    )
    print(f"Plan fingerprint: {plan.plan_fingerprint}")
    for part in plan.parts:
        for line in _describe_part(part):
            print(line)
    if args.expect_plan and args.expect_plan != plan.plan_fingerprint:
        print(
            "Refusing: this recipe now renders a different plan than the "
            f"fingerprint you supplied ({args.expect_plan}). Nothing was "
            "published."
        )
        return 1
    if not args.publish:
        print(
            "Nothing was published. Re-run with --publish to put exactly these "
            "parts in your corpus."
        )
        return 0
    receipt = core.execute_source_parts(
        config, plan, publish_token=plan.plan_fingerprint
    )
    for path in receipt.published:
        print(f"Published {path}")
    for path in receipt.reused:
        print(f"Already in your corpus, byte for byte: {path}")
    print(f"Receipt {receipt.path} (sha256 {receipt.receipt_sha256}).")
    print(
        "The original source is unchanged, and nothing was sent to a model. "
        "These parts are ordinary corpus sources now."
    )
    return 0


def _command_status(config: Any, args: argparse.Namespace) -> int:
    """List publication receipts, or show one recipe's parts."""

    core = _core()
    if not args.recipe_id:
        recipe_ids = core.list_source_part_receipts(config)
        if not recipe_ids:
            print("No source parts have been published in this project yet.")
            return 0
        for recipe_id in recipe_ids:
            record = core.load_source_part_receipt(config, recipe_id)
            published = sum(1 for part in record.parts if part.published)
            print(
                f"{recipe_id}  {record.parent_name}  "
                f"{published}/{len(record.parts)} published  {record.created_at}"
            )
        return 0
    record = core.load_source_part_receipt(config, args.recipe_id)
    print(f"Recipe {record.recipe_id} over {record.parent_name} ({record.parent_sha256}).")
    print(
        f"Rendered by {record.renderer} {record.renderer_version}, encoded by "
        f"{record.encoder} {record.encoder_version}, at {record.render_dpi} DPI."
    )
    print(f"Plan fingerprint: {record.plan_fingerprint}")
    for part in record.parts:
        state = "published" if part.published else "not in the corpus"
        print(
            f"  {part.ordinal}. {part.target_name} — page {part.page_index + 1} "
            f"(/Rotate {part.page_rotate}), {part.byte_length} bytes, {state}"
        )
        print(f"     sha256 {part.sha256}")
    return 0


_COMMANDS = {
    "choose": _command_choose,
    "prepare": _command_prepare,
    "status": _command_status,
}


def run_source_parts_command(config: Any, args: argparse.Namespace) -> int:
    return _COMMANDS[args.source_parts_command](config, args)


def add_source_parts_parser(subparsers: Any, path: Any, handler: Any) -> None:
    """Its own command, registered exactly as ``add_batch_parser`` is."""

    parser = subparsers.add_parser(
        "source-parts",
        help="Render the pages or regions you choose into new immutable sources",
    )
    parser.set_defaults(handler=handler)
    commands = parser.add_subparsers(dest="source_parts_command", required=True)

    choose = commands.add_parser(
        "choose",
        help="Render one source's pages so you can choose parts of them.",
    )
    choose.add_argument("--source", required=True, metavar="NAME")
    choose.add_argument("--output", type=path, required=True, metavar="FILE")
    choose.add_argument(
        "--dpi",
        type=int,
        default=110,
        metavar="N",
        help="How finely to draw the pages for review. Publication DPI is separate.",
    )
    choose.add_argument(
        "--publication-dpi",
        type=int,
        default=200,
        metavar="N",
        help="The DPI the recipe it writes will ask for.",
    )

    prepare = commands.add_parser(
        "prepare",
        help="Plan the parts a recipe names, and publish them with --publish.",
    )
    prepare.add_argument("--source", required=True, metavar="NAME")
    prepare.add_argument(
        "--recipe",
        type=path,
        required=True,
        metavar="FILE",
        help="The recipe you wrote or the editor produced. The only geometry.",
    )
    prepare.add_argument(
        "--publish",
        action="store_true",
        help=(
            "Put exactly these parts in your corpus. Local intake: no model "
            "call, no cost, and no original is edited."
        ),
    )
    prepare.add_argument(
        "--expect-plan",
        default="",
        metavar="FINGERPRINT",
        help=(
            "The plan fingerprint a previous run printed. Publishing refuses if "
            "this recipe now renders anything else."
        ),
    )

    status = commands.add_parser(
        "status", help="List publication receipts, or show one recipe's parts."
    )
    status.add_argument(
        "recipe_id",
        nargs="?",
        default="",
        metavar="RECIPE",
        help="One recipe id. Omit to list every receipt this project holds.",
    )
