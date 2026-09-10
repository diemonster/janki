"""``janki study`` — the same study-job services the Assistant calls.

The Assistant conversation is the primary end-to-end study workflow; this is
the fully supported equivalent over the same application services and the same
authority, not a reduced one. Every subcommand below calls exactly what the
Assistant's own route calls, and none of them is a shim over a retired path.

Two rules shape it.

**A local job write is ordinary local work.** Opening a job, recording a
choice and reading a status spend nothing, disclose nothing to a provider and
edit no original. There is deliberately no ``--yes`` on them, because there is
no consent prompt to answer in advance.

**``--yes`` answers a prompt that already exists.** It never manufactures an
owner decision. The one command here that spends money — ``extract`` — asks
the same question the batch command asks, and a non-TTY without ``--yes``
refuses rather than sending.

The owner-decision commands ``review``, ``coverage`` and ``disposition`` are
deliberately absent: the job schema reserves their choice keys, the editors
that record them ship with the finish, and a CLI that saved one now would be
recording a decision over a rendering nobody had seen.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import sys
from pathlib import Path
from typing import Any


def _core() -> Any:
    return importlib.import_module("japanese_anki.application.study_job")


def _batches() -> Any:
    return importlib.import_module("japanese_anki.application.extraction_batch")


def _capture() -> Any:
    return importlib.import_module("japanese_anki.application.capture_recovery")


def _assignment() -> Any:
    return importlib.import_module("japanese_anki.application.assistant_assignment")


def _deck_creation() -> Any:
    return importlib.import_module("japanese_anki.application.deck_creation")


def _source_parts() -> Any:
    return importlib.import_module("japanese_anki.application.source_parts")


def _confirm_paid(prompt: str, *, yes: bool) -> bool:
    """The same consent shape every paid command here already uses.

    A non-TTY without ``--yes`` refuses rather than proceeding: an agent must
    not answer an approval prompt as the owner, which is why this cannot fall
    back to assuming a default.
    """

    if yes:
        return True
    if not sys.stdin.isatty():
        print(
            "Refusing: this run is unattended and nobody has agreed to these "
            "paid model calls. Re-run with --yes to consent in advance."
        )
        return False
    return input(f"{prompt} [y/N] ").strip().lower() in {"y", "yes"}


def _print_status(config: Any, status: Any) -> None:
    print(f"Study job {status.job_id} ({status.kind}), opened {status.created_at}.")
    print(
        f"  Source {status.parent_source_name} ({status.parent_sha256})"
    )
    deck_note = "" if status.deck_current else "  [the deck definition changed]"
    print(f"  Deck   {status.deck_path}{deck_note}")
    if status.choices:
        for key in sorted(status.choices):
            print(f"  Choice {key}: {status.choices[key]}")
    for part in status.parts:
        if part.missing:
            print(f"  Parts {part.recipe_id}: unresolved — {part.refusal}")
            continue
        print(
            f"  Parts {part.recipe_id}: {part.published_count} of "
            f"{part.part_count} published from {part.parent_name}"
        )
    for batch in status.batches:
        if batch.missing:
            print(f"  Batch {batch.batch_id}: unresolved — {batch.refusal}")
            continue
        print(
            f"  Batch {batch.batch_id} ({batch.kind}): "
            f"{batch.committed_count} settled, {batch.pending_count} in flight "
            f"or unsent, {batch.unknown_count} unknown, {batch.failed_count} failed"
        )
        for child in batch.children:
            print(
                f"    {child.index}. {child.source_name} — {child.state}"
                + ("" if child.bookkeeping_complete else " (bookkeeping unfinished)")
                + (f", {child.records} proposal(s)" if child.records is not None else "")
                + (" [archived]" if child.archive_present else "")
            )
        # Read from the batch's own durable eligibility, never inferred from
        # the states above: `janki study resume` and this line answer the same
        # question, so they may not be two different answers.
        if batch.resume_available:
            print(
                "    `janki study resume` can continue this batch under the "
                "authority it already recorded."
            )
        elif batch.resume_refusal:
            print(f"    {batch.resume_refusal}")
    if status.blocking_operation_ids:
        print(
            f"  {len(status.blocking_operation_ids)} model call(s) still block "
            "spending: " + ", ".join(status.blocking_operation_ids)
        )
    if status.open_intents:
        print(
            f"  {len(status.open_intents)} recorded action(s) have no outcome; "
            "`janki study resume` finds their exact artifacts."
        )
    if status.pending_audio_count:
        print(
            f"  {status.pending_audio_count} audio clip(s) are still in the "
            "ledger's write-ahead record."
        )


def _command_new(config: Any, args: argparse.Namespace) -> int:
    """Open one job over a preserved source and a destination deck."""

    core = _core()
    source = config.scan_inbox / args.source
    if bool(args.deck) == bool(args.create_deck):
        print("Name exactly one of --deck STEM or --create-deck NAME.")
        return 1
    if args.deck:
        if args.directions or args.standalone:
            print(
                "Directions and standalone are settled when a deck is created. "
                "An existing deck's card set is its own, and a job never "
                "changes it."
            )
            return 1
        job = core.open_study_job(
            config,
            kind="source_extraction",
            parent_source=source,
            deck_path=config.deck_dir / f"{args.deck}.yaml",
        )
        print(f"Opened study job {job.header.job_id} over {job.header.deck_path}.")
        return 0

    deck_creation = _deck_creation()
    directions = tuple(args.directions or ("recognition",))
    plan = deck_creation.plan_study_deck(
        config,
        name=args.create_deck,
        recognition="recognition" in directions,
        production="production" in directions,
        reading="reading" in directions,
        standalone=bool(args.standalone),
    )
    print(f"Create {plan.path} with directions: {', '.join(directions)}.")
    # One confirmation, for the deck. The job's own write rides inside this
    # action after the deck exists: it is a local record, so it asks nothing
    # again and a refused deck creation leaves no job behind.
    created = deck_creation.create_study_deck(config, plan)
    job = core.open_study_job(
        config,
        kind="source_extraction",
        parent_source=source,
        deck_path=created.path,
        deck_sha256=hashlib.sha256(created.yaml_bytes).hexdigest(),
    )
    print(f"Created {created.path}.")
    print(f"Opened study job {job.header.job_id} over {job.header.deck_path}.")
    return 0


def _command_parts(config: Any, args: argparse.Namespace) -> int:
    """Plan this job's parts from an owner-written recipe, and publish them.

    ``--select`` is the same owner decision the Assistant's per-part control
    saves: which already-published parts this job's next batch covers. It is a
    reversible local preference through the job's own compare-and-swap writer —
    no render, no publication, no model call — so it needs no recipe.
    """

    core = _core()
    if args.select:
        return _select_parts(core, config, args)
    if not args.recipe:
        print(
            "Name what to do with this job's parts: --recipe FILE plans them "
            "from your own recipe, and --select NAME … records which already "
            "published parts its next batch covers."
        )
        return 1
    source_parts = _source_parts()
    unavailable = source_parts.sources_unavailable()
    if unavailable is not None:
        print(unavailable)
        return 1
    job = core.load_study_job(config, args.job)
    try:
        written = Path(args.recipe).read_bytes()
    except OSError as exc:
        raise source_parts.SourcePartsError(
            f"Could not read the recipe file {args.recipe}: {exc}"
        ) from exc
    recipe, token = source_parts.recipe_from_bytes(written)
    plan = source_parts.plan_source_parts(
        config, job.header.parent_source_name, recipe, recipe_token=token
    )
    print(f"Recipe {plan.recipe_id} over {plan.parent_name} ({plan.parent_sha256}).")
    print(f"Plan fingerprint: {plan.plan_fingerprint}")
    for part in plan.parts:
        print(
            f"  {part.ordinal}. {part.target_name} — {part.byte_length} bytes, "
            f"sha256 {part.sha256}"
        )
    if not args.publish:
        print(
            "Nothing was published. Re-run with --publish to put exactly these "
            "parts in your corpus and record this job's binding to them."
        )
        return 0
    receipt = core.publish_job_source_parts(
        config, job.header.job_id, plan, publish_token=plan.plan_fingerprint
    )
    for path in receipt.published:
        print(f"Published {path}")
    for path in receipt.reused:
        print(f"Already in your corpus, byte for byte: {path}")
    print(f"Receipt {receipt.path} (sha256 {receipt.receipt_sha256}).")
    print(
        "The receipt binds the parts, not this job: any later job or CLI run "
        "may reuse it. Nothing was sent to a model."
    )
    return 0


def _select_parts(core: Any, config: Any, args: argparse.Namespace) -> int:
    """Record which published parts this job's next batch covers."""

    if args.recipe:
        print(
            "Choose one: --recipe FILE plans and publishes parts, --select "
            "NAME … records which published ones this job's batch covers."
        )
        return 1
    job = core.load_study_job(config, args.job)
    published = core.job_published_parts(config, args.job)
    wanted = list(dict.fromkeys(args.select))
    missing = [name for name in wanted if name not in published]
    if missing:
        print(
            "This job has not published: "
            + ", ".join(missing)
            + ". It published: "
            + (", ".join(published) if published else "nothing yet")
            + "."
        )
        return 1
    selection = [name for name in published if name in set(wanted)]
    core.record_choice(
        config,
        args.job,
        {"part_selections": selection},
        expected_revision=job.revision,
    )
    print(
        f"This job's next batch covers {len(selection)} of {len(published)} "
        "published part(s): " + ", ".join(selection) + "."
    )
    print(
        "A local preference over parts you already published: nothing was "
        "sent, nothing was published and no deck definition changed."
    )
    return 0


def _command_extract(config: Any, args: argparse.Namespace) -> int:
    """Plan this job's batch over its published parts, then send it once."""

    core = _core()
    plan = core.plan_job_extraction_batch(
        config,
        args.job,
        mode=args.mode,
        concurrency_limit=args.concurrency,
    )
    print(
        f"Batch {plan.batch_id} would send {len(plan.children)} source(s) to "
        f"{plan.model} through {plan.provider}, {plan.concurrency_limit} at a time:"
    )
    for child in plan.children:
        print(
            f"  {child.index}. {child.source.name} — source sha256 "
            f"{child.source_sha256}, request {child.request_fingerprint}, "
            f"operation {child.operation_id}"
        )
    print(f"Consent fingerprint: {plan.fingerprint}")
    if not _confirm_paid(
        f"Send these {len(plan.children)} paid model calls?", yes=args.yes
    ):
        print("Nothing was sent.")
        return 1
    outcome = core.dispatch_job_batch(config, args.job, plan)
    print(
        f"{outcome.committed_count} settled, {outcome.pending_count} in flight "
        f"or unsent, {outcome.unknown_count} unknown, {outcome.failed_count} failed."
    )
    return 0


def _command_recover(config: Any, args: argparse.Namespace) -> int:
    """Stage one already-paid captured reply through the same S2 service."""

    core = _core()
    batches = _batches()
    capture = _capture()
    core.load_study_job(config, args.job)
    plan, _child = batches.find_capture_child(config, args.operation)
    if plan.job_id != args.job:
        print(
            f"Model call {args.operation} belongs to "
            + (f"study job {plan.job_id}" if plan.job_id else "no study job")
            + f", not {args.job}. Nothing was staged."
        )
        return 1
    inspected = capture.inspect_capture_proposals(config, args.operation)
    print(
        f"{inspected.valid_group_count} valid proposal(s) across "
        f"{inspected.valid_location_count} location(s); capture sha256 "
        f"{inspected.capture_sha256}."
    )
    for group in inspected.groups:
        for location in group.locations:
            print(
                f"  {group.proposal_sha256} {group.schema_verdict} — frame "
                f"{location.frame_index} block {location.block_index} "
                f"{location.json_pointer} ({location.envelope_shape})"
            )
    selection = None
    if args.proposal:
        selection = inspected.select(args.proposal, json_pointer=args.at or None)
    outcome = capture.stage_capture_proposal(config, args.operation, selection)
    print(
        f"Staged {outcome.records} proposal(s) into {outcome.target}. No call "
        "was made and nothing was billed."
    )
    return 0


def _command_assign(config: Any, args: argparse.Namespace) -> int:
    """Assign this job's staged cards to a deck through the sole writer."""

    core = _core()
    assignment = _assignment()
    job = core.load_study_job(config, args.job)
    proposal = config.staging_dir / f"{args.part}.yaml"
    plan = assignment.plan_assignment_for_paths(
        config,
        proposal_path=proposal,
        destination_path=config.deck_dir / f"{args.deck}.yaml",
        record_ids=args.records,
        instruction=f"Assign study job {job.header.job_id} part {args.part}.",
    )
    execution = assignment.execute_assignment(config, plan)
    print(
        f"Assigned {len(execution.assigned_record_ids)} card(s) in "
        f"{plan.proposal_path.name} to {plan.destination_name}: "
        + ", ".join(execution.assigned_record_ids)
    )
    return 0


def _command_status(config: Any, args: argparse.Namespace) -> int:
    """Where one job stands, or which jobs exist. Reads only."""

    core = _core()
    if not args.job:
        job_ids = core.list_study_jobs(config)
        if not job_ids:
            print("No study job has been opened in this project yet.")
            return 0
        for job_id in job_ids:
            status = core.study_job_status(config, job_id)
            print(
                f"{job_id}  {status.parent_source_name} → {status.deck_path}  "
                f"{len(status.batches)} batch(es)"
            )
        return 0
    _print_status(config, core.study_job_status(config, args.job))
    return 0


def _command_preview(config: Any, args: argparse.Namespace) -> int:
    """Draw this job's saved proposals as the deck's real cards."""

    core = _core()
    rendered = core.render_job_preview(config, args.job)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(rendered.preview.html)
    print(f"Wrote {rendered.preview.card_count} card(s) to {output}.")
    print(f"Rendered content fingerprint: {rendered.rendering_fingerprint}")
    for conflict in rendered.conflicts:
        print(f"  {conflict}")
    print("Looking at these accepts nothing and adds nothing to a deck.")
    return 0


def _command_resume(config: Any, args: argparse.Namespace) -> int:
    """Finish what this job's open intents already have authority for."""

    core = _core()
    actions = core.resume_job_actions(config, args.job)
    if not actions:
        print(
            "Nothing in this job is waiting to be finished: every action it "
            "recorded already has an outcome. Nothing was sent."
        )
        return 0
    for action in actions:
        batch = action.batch
        if batch is not None:
            print(
                f"{action.subject}: {batch.committed_count} settled, "
                f"{batch.pending_count} in flight or unsent, "
                f"{batch.unknown_count} unknown, {batch.failed_count} failed."
            )
        else:
            print(f"{action.subject}:")
        print(f"  {action.detail}")
    return 0


_COMMANDS = {
    "new": _command_new,
    "parts": _command_parts,
    "extract": _command_extract,
    "recover": _command_recover,
    "assign": _command_assign,
    "status": _command_status,
    "preview": _command_preview,
    "resume": _command_resume,
}


def run_study_command(config: Any, args: argparse.Namespace) -> int:
    return _COMMANDS[args.study_command](config, args)


def add_study_parser(subparsers: Any, path: Any, handler: Any) -> None:
    """Its own command, registered exactly as ``add_batch_parser`` is."""

    parser = subparsers.add_parser(
        "study",
        help="Take one preserved source through to a deck as one study job",
    )
    parser.set_defaults(handler=handler)
    commands = parser.add_subparsers(dest="study_command", required=True)

    new = commands.add_parser(
        "new", help="Open a study job over one source and one destination deck."
    )
    new.add_argument("--source", required=True, metavar="NAME")
    new.add_argument("--deck", default="", metavar="STEM")
    new.add_argument("--create-deck", default="", metavar="NAME")
    new.add_argument(
        "--standalone",
        action="store_true",
        help="Create the new deck with its own copies and review progress.",
    )
    new.add_argument(
        "--directions",
        nargs="+",
        default=[],
        choices=["recognition", "production", "reading"],
        metavar="DIRECTION",
        help="Card directions for a new deck. Never inferred for an existing one.",
    )

    parts = commands.add_parser(
        "parts", help="Plan this job's parts from a recipe, and publish them."
    )
    parts.add_argument("job", metavar="JOB")
    parts.add_argument("--recipe", type=path, default=None, metavar="FILE")
    parts.add_argument(
        "--publish",
        action="store_true",
        help=(
            "Put exactly these parts in your corpus and record this job's "
            "binding to their receipt. Local intake: no model call, no cost."
        ),
    )
    parts.add_argument(
        "--select",
        nargs="+",
        default=[],
        metavar="NAME",
        help=(
            "Record which already published parts this job's next batch "
            "covers. A reversible local preference: it publishes nothing, "
            "sends nothing and changes no deck definition."
        ),
    )

    extract = commands.add_parser(
        "extract", help="Send this job's published parts as one confirmed batch."
    )
    extract.add_argument("job", metavar="JOB")
    extract.add_argument("--mode", default=None, metavar="MODE")
    extract.add_argument("--concurrency", type=int, default=2, metavar="N")
    extract.add_argument(
        "--yes",
        action="store_true",
        help=(
            "Consent in advance to sending these exact files to the paid "
            "model. Without it, an unattended run refuses rather than sending."
        ),
    )

    recover = commands.add_parser(
        "recover", help="Stage a reply this job already paid for. Sends nothing."
    )
    recover.add_argument("job", metavar="JOB")
    recover.add_argument("--operation", required=True, metavar="ID")
    recover.add_argument(
        "--proposal",
        default="",
        metavar="SHA256",
        help="The exact proposal you chose, when the reply holds more than one.",
    )
    recover.add_argument(
        "--at",
        default="",
        metavar="POINTER",
        help="The exact location of that proposal, when one hash names several.",
    )

    assign = commands.add_parser(
        "assign", help="Assign this job's staged cards to one study deck."
    )
    assign.add_argument("job", metavar="JOB")
    assign.add_argument("--part", required=True, metavar="PART")
    assign.add_argument("--deck", required=True, metavar="STEM")
    assign.add_argument("--records", nargs="+", required=True, metavar="ID")

    status = commands.add_parser(
        "status", help="Where one job stands, or which jobs this project holds."
    )
    status.add_argument("job", nargs="?", default="", metavar="JOB")

    preview = commands.add_parser(
        "preview", help="Draw this job's saved proposals as the deck's real cards."
    )
    preview.add_argument("job", metavar="JOB")
    preview.add_argument("--output", type=path, required=True, metavar="FILE")

    resume = commands.add_parser(
        "resume", help="Continue what this job already has authority for."
    )
    resume.add_argument("job", metavar="JOB")
