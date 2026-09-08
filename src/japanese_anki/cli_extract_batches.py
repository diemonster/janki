"""``janki extract`` across several sources, and the local ``extract-batch`` desk.

Every durable decision belongs to ``application.extraction_batch``: which calls
a plan contains, whether a child may be retried, what a resume is allowed to
resend, and how the saved children combine into one rendered preview. What
lives here is the sentence an owner reads before saying yes, and the exact
question asked — nothing in this module writes to the journal, discards a
recorded call, or decides eligibility.

The core is resolved at call time rather than imported at module scope so that
this surface and its tests share exactly one seam to it.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

CONCURRENCY_DEFAULT = 2
CONCURRENCY_CAP = 4


def concurrency(value: str) -> int:
    """Parse ``--concurrency`` inside the range the batch core accepts."""

    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--concurrency needs a whole number, not {value!r}."
        ) from exc
    if not 1 <= parsed <= CONCURRENCY_CAP:
        raise argparse.ArgumentTypeError(
            f"--concurrency must be between 1 and {CONCURRENCY_CAP}; "
            f"{parsed} was requested."
        )
    return parsed


def _core() -> Any:
    return importlib.import_module("japanese_anki.application.extraction_batch")


def _ask(assume_yes: bool) -> bool:
    """The one owner question. ``--yes`` is consent given in advance, in words."""

    if assume_yes:
        return True
    if not sys.stdin.isatty():
        print(
            "Refusing: this needs a person. Re-run with --yes to consent in "
            "advance.",
            file=sys.stderr,
        )
        return False
    try:
        answer = input("Send them? [y/N] ")
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer.strip().lower() in {"y", "yes"}


def _source_name(value: Any) -> str:
    return Path(str(value)).name


def _describe_children(plan: Any) -> list[str]:
    return [
        f"  {child.index}. {_source_name(child.source)}" for child in plan.children
    ]


def _progress_printer(total: int) -> Any:
    """Number every live event, because several sources are in flight at once."""

    def report(event: Any) -> None:
        line = f"Source {event.index} of {total}: {event.state}"
        if event.message:
            line += f" — {event.message}"
        print(line)

    return report


def _report_outcome(outcome: Any, *, retry_hint: bool) -> int:
    """Print what each child ended as, and say what is left to do."""

    print(
        f"{outcome.committed_count} saved, {outcome.failed_count} failed, "
        f"{outcome.unknown_count} unknown, {outcome.pending_count} still waiting."
    )
    for child in outcome.children:
        line = f"  {child.index}. {_source_name(child.source)} — {child.state}"
        if child.records:
            line += f" ({child.records} proposals)"
        if child.error:
            line += f": {child.error}"
        if not child.bookkeeping_complete:
            line += " (bookkeeping unfinished)"
        print(line)
    unfinished = tuple(
        child.index
        for child in outcome.children
        if child.state not in {"committed"}
    )
    if not unfinished:
        return 0
    if retry_hint:
        listed = " ".join(str(index) for index in unfinished)
        print(
            f"Sources still unfinished: {listed}. "
            f"`janki extract-batch status {outcome.batch_id}` says where each "
            "one stopped, and only the core decides which may be sent again."
        )
    return 1


def run_batch_extraction(
    config: Any,
    args: argparse.Namespace,
    prepared: Sequence[Any],
) -> int:
    """Plan one batch over the already-preserved sources, ask once, dispatch."""

    core = _core()
    plan = core.plan_extraction_batch(
        config,
        [item.origin_path for item in prepared],
        mode=args.mode,
        model=args.model or config.extract_model,
        concurrency_limit=args.concurrency,
        force=args.force,
    )
    print(
        f"About to send {len(plan.children)} sources to {plan.model} through "
        f"{plan.provider}, {plan.concurrency_limit} at a time, leaving this "
        "machine:"
    )
    print("\n".join(_describe_children(plan)))
    print(f"Batch: {plan.batch_id}")
    if not _ask(args.yes):
        # Says what was kept, not only what was not sent: `prepare_inputs` has
        # already copied any outside file into the durable inbox.
        told = ["Nothing was sent."]
        told += [
            f"  kept in the inbox: {item.origin_path}"
            for item in prepared
            if getattr(item, "copied", False)
        ]
        if len(told) > 1:
            told.append("  Delete those if you did not want them stored.")
        print("\n".join(told))
        return 1
    outcome = core.dispatch_extraction_batch(
        config, plan, progress=_progress_printer(len(plan.children))
    )
    return _report_outcome(outcome, retry_hint=True)


def _print_batch(outcome: Any) -> None:
    print(f"Batch: {outcome.batch_id}")
    print(f"  {outcome.concurrency_limit} at a time")
    for child in outcome.children:
        line = f"  {child.index}. {_source_name(child.source)} — {child.state}"
        if child.records:
            line += f" ({child.records} proposals)"
        if child.error:
            line += f": {child.error}"
        print(line)
    print(
        f"  {outcome.committed_count} saved, {outcome.failed_count} failed, "
        f"{outcome.unknown_count} unknown, {outcome.pending_count} still waiting."
    )


def _command_status(config: Any, args: argparse.Namespace) -> int:
    core = _core()
    if args.batch_id:
        _print_batch(core.extraction_batch_status(config, args.batch_id))
        return 0
    outcomes = core.list_extraction_batches(config)
    if not outcomes:
        print("No extraction batch has been started.")
        return 0
    for outcome in outcomes:
        _print_batch(outcome)
    return 0


def _command_resume(config: Any, args: argparse.Namespace) -> int:
    print(
        f"Continuing batch {args.batch_id} under model calls it already "
        "authorized. Nothing is asked again, and nothing already sent is sent "
        "a second time."
    )
    core = _core()
    total = len(core.extraction_batch_status(config, args.batch_id).children)
    outcome = core.resume_extraction_batch(
        config, args.batch_id, progress=_progress_printer(total)
    )
    return _report_outcome(outcome, retry_hint=True)


def _command_retry(config: Any, args: argparse.Namespace) -> int:
    core = _core()
    requested = tuple(dict.fromkeys(args.children))
    plan = core.plan_extraction_batch_retry(config, args.batch_id, requested)
    print(f"Retrying batch {args.batch_id}. This does two things:")
    print("Retires this recorded evidence, which is then gone:")
    for discard in plan.discards:
        line = (
            f"  {discard.source_file} — model call {discard.operation_id}, "
            f"state {discard.state}, model {discard.model}"
        )
        if discard.detail:
            line += f": {discard.detail}"
        print(line)
        print(f"    request fingerprint {discard.request_fp}")
    states = {discard.state for discard in plan.discards}
    if "outcome_unknown" in states:
        print(
            "  One of those calls may or may not have been made, so its "
            "outcome and its cost are unknown. Sending again discards what "
            "janki kept about it and makes a distinct new call with a new id; "
            "the old one is never sent again."
        )
    if "result_captured" in states:
        print(
            "  A reply janki already received and saved is discarded. It was "
            "already paid for and cannot be recovered afterwards."
        )
    print(
        f"Sends {len(plan.children)} new paid call(s) to {plan.model} through "
        f"{plan.provider}, {plan.concurrency_limit} at a time:"
    )
    print("\n".join(_describe_children(plan)))
    if not _ask(args.yes):
        print("Nothing was sent. The old evidence above is untouched.")
        return 1
    outcome = core.dispatch_extraction_batch(
        config, plan, progress=_progress_printer(len(plan.children))
    )
    return _report_outcome(outcome, retry_hint=False)


def _command_preview(config: Any, args: argparse.Namespace) -> int:
    rendered = _core().render_extraction_batch_preview(config, args.batch_id)
    args.output.write_bytes(bytes(rendered.preview.html))
    covered = ", ".join(str(index) for index in rendered.child_indices) or "none"
    print(
        f"Wrote {rendered.preview.card_count} cards from sources {covered} to "
        f"{args.output}."
    )
    print(
        "This is a rendered look at proposals that are already saved. It "
        "reviews nothing and changes nothing."
    )
    if rendered.conflicts:
        print("The sources disagree about these, which the preview cannot settle:")
        for conflict in rendered.conflicts:
            print(f"  {conflict}")
    return 0


_COMMANDS = {
    "status": _command_status,
    "resume": _command_resume,
    "retry": _command_retry,
    "preview": _command_preview,
}


def run_batch_command(config: Any, args: argparse.Namespace) -> int:
    return _COMMANDS[args.batch_command](config, args)


def add_batch_parser(subparsers: Any, path: Any, handler: Any) -> None:
    """Its own command, so ``resume`` can never be read as a filename."""

    parser = subparsers.add_parser(
        "extract-batch",
        help="Check, continue, retry, or preview a multi-source extraction",
    )
    parser.set_defaults(handler=handler)
    commands = parser.add_subparsers(dest="batch_command", required=True)

    status = commands.add_parser(
        "status", help="Show every source in a batch and where it stopped."
    )
    status.add_argument(
        "batch_id",
        nargs="?",
        default="",
        metavar="BATCH",
        help="One batch. Omit to list every batch this project has started.",
    )

    resume = commands.add_parser(
        "resume",
        help="Continue a batch under the model calls it already authorized.",
    )
    resume.add_argument("batch_id", metavar="BATCH")

    retry = commands.add_parser(
        "retry",
        help="Send named failed sources again, retiring their old evidence.",
    )
    retry.add_argument("batch_id", metavar="BATCH")
    retry.add_argument(
        "--children",
        nargs="+",
        type=int,
        required=True,
        metavar="N",
        help=(
            "The exact source numbers to send again, as `status` numbers them. "
            "There is no retry-everything flag: each one costs a model call."
        ),
    )
    retry.add_argument(
        "--yes",
        action="store_true",
        help=(
            "Consent in advance to retiring the named evidence and making the "
            "new calls. Without it, an unattended run refuses."
        ),
    )

    preview = commands.add_parser(
        "preview",
        help="Write one rendered HTML card preview over a batch's saved sources.",
    )
    preview.add_argument("batch_id", metavar="BATCH")
    preview.add_argument(
        "--output",
        type=path,
        required=True,
        metavar="FILE",
        help="Where to write the rendered HTML.",
    )
