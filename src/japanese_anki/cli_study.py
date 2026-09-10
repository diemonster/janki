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

``layout`` and ``curate`` are here, and both are owner-authored throughout.
The layout file is written by the owner: janki runs no header detector, mints
no column identity of its own and matches no printed label, and no
model-emittable field carries a geometry, an identity, a witness, a display
label or a revision. ``curate`` settles one identity's printed cells across
every part that staged it, over the current edited staged values.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any

from japanese_anki.errors import JankiError


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


def _extract() -> Any:
    return importlib.import_module("japanese_anki.extract")


def _curation() -> Any:
    return importlib.import_module("japanese_anki.application.study_curation")


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
    for part in sorted(status.layout_bindings):
        print(f"  Layout {part}: {status.layout_bindings[part]}")
    if status.pending_curation_intents:
        print(
            f"  {len(status.pending_curation_intents)} curation decision(s) "
            "recorded and not finished; promotion of the files they name is "
            "blocked until they are: "
            + ", ".join(status.pending_curation_intents)
        )
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


def _command_layout(config: Any, args: argparse.Namespace) -> int:
    """Bind one owner-authored layout revision to one published part.

    The layout file is the owner's. It carries the opaque column identities
    they minted at review, the exact printed strings they recorded as each
    column's witnesses, their ordered display labels, and the revision they
    are saving. janki reads no header, matches no label and mints no identity;
    it appends the revision and repoints the binding in one compare-and-swap
    write, and refuses to rewrite a revision it already recorded.
    """

    core = _core()
    extract = _extract()
    if not args.layout:
        job = core.load_study_job(config, args.job)
        bound = core.job_layout_bindings(job)
        if not bound:
            print(
                f"Study job {args.job} binds no source-form layout. Write one "
                "and save it with --part PART --layout FILE."
            )
            return 0
        for part in sorted(bound):
            layout = bound[part]
            print(f"{part}: layout {layout.identity}")
            for column in layout.columns:
                print(
                    f"  {column.ordinal}. {column.column_id} — "
                    f"{column.display_label} (printed: "
                    + " | ".join(column.label_witnesses)
                    + ")"
                )
        return 0
    if not args.part:
        print("Name the published part this layout covers with --part NAME.")
        return 1
    published = core.job_published_parts(config, args.job)
    if args.part not in published:
        print(
            f"This job has not published {args.part}. It published: "
            + (", ".join(published) if published else "nothing yet")
            + "."
        )
        return 1
    try:
        written = Path(args.layout).read_bytes()
    except OSError as exc:
        print(f"error: could not read the layout file {args.layout}: {exc}")
        return 1
    try:
        parsed = json.loads(written.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise extract.ExtractError(
            f"The layout file {args.layout} is not readable JSON: {exc}",
            code="extract-layout-invalid",
        ) from exc
    layout = extract.TableLayout.from_wire(parsed, where=str(args.layout))
    job = core.load_study_job(config, args.job)
    core.append_layout(
        config,
        args.job,
        layout,
        bind=(args.part,),
        allow_identical=bool(args.rebind),
        expected_revision=job.revision,
    )
    print(
        f"Bound {args.part} to layout {layout.identity} "
        f"({len(layout.columns)} column(s))."
    )
    print(
        "A local owner decision: the revision is immutable, nothing was sent, "
        "and no card or deck definition changed."
    )
    return 0


def _curation_choices(curation: Any, config: Any, args: argparse.Namespace) -> Any:
    """The exact choices the owner typed, and the words they typed them in."""

    if args.choose:
        record_id, _, part = str(args.choose).partition("=")
        if not record_id or not part:
            print("--choose names one identity and one part: RECORD_ID=PART.")
            return None, ""
        source = None
        for group in curation.read_curation_groups(config, args.job):
            if group.record_id != record_id:
                continue
            for occurrence in group.occurrences:
                if occurrence.part_name == part:
                    source = occurrence
        if source is None:
            print(
                f"{part} stages no proposal for {record_id}, so there is no "
                "table there to adopt."
            )
            return None, ""
        return (
            (
                curation.CurationChoice(
                    record_id=record_id,
                    action="replace",
                    key="",
                    value=source.forms_wire,
                ),
            ),
            f"adopt {part}'s source_forms for {record_id}",
        )
    if args.drop_table:
        return (
            (
                curation.CurationChoice(
                    record_id=args.record, action="remove", key=""
                ),
            ),
            f"remove source_forms from {args.record}",
        )
    if args.remove:
        return (
            (
                curation.CurationChoice(
                    record_id=args.record, action="remove", key=args.column
                ),
            ),
            f"remove source_forms[{args.column}] from {args.record}",
        )
    value = "" if args.blank else args.set
    return (
        (
            curation.CurationChoice(
                record_id=args.record,
                action="replace",
                key=args.column,
                value=value,
            ),
        ),
        f"set source_forms[{args.column}] of {args.record} to {value!r}",
    )


def _print_open_curation_intents(curation: Any, config: Any, job_id: str) -> None:
    """Every recorded decision this job has not closed, and how to close it.

    Printed where an owner already is when they go looking: the digest an
    abandonment has to carry is here rather than in a second command, and a
    decision no replay can finish says so instead of being listed beside one
    that can.
    """

    try:
        barriers = [
            barrier
            for barrier in curation.open_curation_barriers(config)
            if barrier.job_id == job_id
        ]
    except JankiError as exc:
        print(f"This job's recorded decisions could not be read: {exc}")
        return
    for barrier in barriers:
        print(
            f"Open decision {barrier.intent_id} ({barrier.decision}) over "
            + ", ".join(barrier.paths)
        )
        print(f"  intent digest {barrier.intent_sha256}")
        if barrier.unsatisfiable:
            print(
                "  at neither recorded digest, so no replay can finish it: "
                + ", ".join(barrier.unsatisfiable)
            )
            print(
                f"  close it with --abandon {barrier.intent_id} --expect "
                f"{barrier.intent_sha256}"
            )
        elif barrier.remaining:
            print(
                "  still to be written: " + ", ".join(barrier.remaining) + "; "
                f"finish it with --resume {barrier.intent_id}"
            )


def _command_curate(config: Any, args: argparse.Namespace) -> int:
    """Settle one identity's printed cells across every part that staged it."""

    curation = _curation()
    if args.resume and args.abandon:
        print(
            "Finishing a decision and closing it unwritten are two different "
            "decisions; name one. Nothing was written."
        )
        return 1
    if args.resume:
        outcome = curation.resume_curation(config, args.job, args.resume)
        print(f"Curation {outcome.intent_id}: {outcome.detail}")
        for path in outcome.written:
            print(f"  wrote {path}")
        for path in outcome.already_current:
            print(f"  already at the recorded result: {path}")
        return 0
    if args.abandon:
        # The owner's own decision, and never inferred: janki neither picks the
        # intent nor supplies the digest that names it.
        if not args.expect:
            print(
                "--abandon closes the exact decision you read, so name its "
                "digest with --expect SHA256. `janki study curate "
                f"{args.job}` prints it. Nothing was closed."
            )
            return 1
        outcome = curation.abandon_intent(
            config,
            args.job,
            args.abandon,
            expected_intent_sha256=args.expect,
        )
        print(f"Curation {outcome.intent_id} abandoned: {outcome.detail}")
        for path, digest in outcome.observed:
            print(f"  observed {path} at {digest}")
        print(
            "Its record and evidence stay in this job's log. Decide again over "
            "these files with an ordinary `janki study curate` command."
        )
        return 0

    acting = bool(args.choose or args.drop_table or args.column)
    if not acting:
        _print_open_curation_intents(curation, config, args.job)
        groups = curation.read_curation_groups(config, args.job)
        if not groups:
            print(
                f"Study job {args.job} has no staged proposals to curate yet. "
                "Nothing was read from a model and nothing was written."
            )
            return 0
        for group in groups:
            marker = " [parts disagree]" if group.conflicting else ""
            print(f"{group.record_id} ({group.expression}){marker}")
            for occurrence in group.occurrences:
                table = occurrence.source_forms
                cells = (
                    ", ".join(
                        f"{column.id}={table.cells[column.id]!r}"
                        for column in table.columns
                        if column.id in table.cells
                    )
                    if table is not None
                    else "no printed table"
                )
                print(f"  {occurrence.part_name}: {cells or 'no printed cells'}")
        print(
            "Looking at these changes nothing. Settle one with --record ID "
            "--column CELL, --record ID --drop-table, or --choose ID=PART."
        )
        return 0

    if args.column and not args.record:
        print("--column names the cell of one identity; name it with --record ID.")
        return 1
    if args.drop_table and not args.record:
        print("--drop-table removes one identity's table; name it with --record ID.")
        return 1
    if args.column and not (args.remove or args.blank or args.set is not None):
        print(
            "Say what to do with that cell: --set TEXT, --blank (the printed "
            "blank), or --remove."
        )
        return 1

    choices, decision = _curation_choices(curation, config, args)
    if choices is None:
        return 1
    if args.note:
        decision = f"{decision} — {args.note}"
    plan = curation.plan_curation(config, args.job, choices, decision=decision)
    print(f"{decision}.")
    for update in plan.prepared:
        print(f"  {update.staging_path}: {update.sha256_before} → {update.sha256_after}")
    for absent in plan.absent:
        print(f"  no staged occurrence of {absent}; nothing to settle for it")
    outcome = curation.apply_curation(config, plan)
    print(f"Curation {outcome.intent_id}: {outcome.detail}")
    if outcome.supersedes:
        # The replan §6.2 describes: the decision it settles again keeps its
        # own record and evidence, and this one says which it was.
        print(
            f"  recorded over closed decision {outcome.supersedes}, which "
            "keeps its own record and evidence in this job's log"
        )
    print(
        "A local owner decision over already-staged proposals: nothing was "
        "sent, nothing was promoted, and no paid retry was needed."
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
    # One confirmed batch carries one mode, and where the owner bound printed
    # columns the job pinned it rather than the flag: say which, and say which
    # revision each child is sent under, before anything is agreed to.
    modes = sorted({child.expectation.mode or "auto" for child in plan.children})
    print(f"Mode: {', '.join(modes)}")
    for child in plan.children:
        bound = child.expectation.table_layout
        print(
            f"  {child.index}. {child.source.name} — source sha256 "
            f"{child.source_sha256}, request {child.request_fingerprint}, "
            f"operation {child.operation_id}"
            + (f", layout {bound.identity}" if bound is not None else "")
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
    "layout": _command_layout,
    "curate": _command_curate,
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

    layout = commands.add_parser(
        "layout",
        help=(
            "Bind one owner-written source-form layout to one published part. "
            "With no --layout, print what this job already binds."
        ),
    )
    layout.add_argument("job", metavar="JOB")
    layout.add_argument("--part", default="", metavar="PART")
    layout.add_argument(
        "--layout",
        type=path,
        default=None,
        metavar="FILE",
        help=(
            "Your own layout file: the opaque column ids you minted, each "
            "column's printed position, the exact printed headings you "
            "recorded and your display labels. janki reads no header and "
            "matches no label."
        ),
    )
    layout.add_argument(
        "--rebind",
        action="store_true",
        help=(
            "Accept a file that saves a revision this job already records "
            "identically, and point this part at it. A revision whose columns "
            "differ is always refused: it is immutable."
        ),
    )

    curate = commands.add_parser(
        "curate",
        help=(
            "Settle one identity's printed cells across every part that staged "
            "it. With no action, list what the parts propose."
        ),
    )
    curate.add_argument("job", metavar="JOB")
    curate.add_argument("--record", default="", metavar="ID")
    curate.add_argument("--column", default="", metavar="CELL_ID")
    curate.add_argument("--set", default=None, metavar="TEXT")
    curate.add_argument(
        "--blank",
        action="store_true",
        help="The printed blank: a declared row whose value the source left empty.",
    )
    curate.add_argument(
        "--remove",
        action="store_true",
        help="Delete that one cell. Absent is a different fact from blank.",
    )
    curate.add_argument(
        "--drop-table",
        action="store_true",
        help="Delete this identity's whole printed table from every staged copy.",
    )
    curate.add_argument(
        "--choose",
        default="",
        metavar="RECORD_ID=PART",
        help=(
            "Adopt the table that part staged for this identity into every "
            "other part's staged copy of it."
        ),
    )
    curate.add_argument(
        "--note",
        default="",
        metavar="TEXT",
        help="Your own words about the decision, recorded with it.",
    )
    curate.add_argument(
        "--resume",
        default="",
        metavar="INTENT",
        help=(
            "Finish a decision that was recorded but not completely written. "
            "It writes exactly the text that decision recorded."
        ),
    )
    curate.add_argument(
        "--abandon",
        default="",
        metavar="INTENT",
        help=(
            "Close a recorded decision without writing it, when a bound file "
            "has moved to neither digest it recorded and no replay can finish "
            "it. Records the mixed snapshot it measures; reverts nothing and "
            "deletes no evidence. Needs --expect."
        ),
    )
    curate.add_argument(
        "--expect",
        default="",
        metavar="SHA256",
        help=(
            "The digest of the exact intent --abandon closes, as this job "
            "records it. `janki study curate JOB` and the promotion refusal "
            "both print it."
        ),
    )

    extract = commands.add_parser(
        "extract", help="Send this job's published parts as one confirmed batch."
    )
    extract.add_argument("job", metavar="JOB")
    extract.add_argument(
        "--mode",
        default=None,
        metavar="MODE",
        help=(
            "Send these parts under one named extraction mode. Omit it: a job "
            "whose parts carry bound source-form layouts pins the layout mode "
            "from those bindings, and every other batch lets the model judge "
            "each page."
        ),
    )
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
