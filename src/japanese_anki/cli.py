from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from japanese_anki import ledger, status
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.exporters.anki import AnkiBuildError, build_deck, resolve_deck_records
from japanese_anki.importers.shirabe import import_file, inspect_file
from japanese_anki.io import (
    MERGE_LABELS,
    DataError,
    MergeOutcome,
    load_records,
    load_structured,
    merge_records,
    parse_prefer_incoming,
    save_records_json,
)
from japanese_anki.models import VocabularyRecord
from japanese_anki.preview import build_preview
from japanese_anki.staging import StagingError, write_staging
from japanese_anki.validation import has_errors, validate_records


def _path(value: str) -> Path:
    return Path(value).expanduser()


def _load_config(args: argparse.Namespace) -> ProjectConfig:
    root = args.root.resolve() if args.root else None
    return ProjectConfig.load(root)


def _print_issues(issues: list) -> None:
    for issue in issues:
        print(issue.format())


def command_inspect(args: argparse.Namespace) -> int:
    result = inspect_file(args.file.resolve(), sample_size=args.rows)
    payload = {
        "delimiter": result.delimiter,
        "headers": result.headers,
        "mapping": result.mapping,
        "unknown_headers": result.unknown_headers,
        "sample_rows": result.sample_rows,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _format_merge_value(value: object) -> str:
    if isinstance(value, str):
        text = value
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        text = "; ".join(value)
    else:
        text = str(value)
    text = " ".join(text.split())
    return text if len(text) <= 60 else f"{text[:57]}..."


def _print_merge_summary(outcomes: dict[str, MergeOutcome]) -> None:
    counts = Counter(outcome.label for outcome in outcomes.values())
    print(
        "Merge result: "
        + ", ".join(f"{counts.get(label, 0)} {label}" for label in MERGE_LABELS)
    )
    conflicts = [
        (record_id, conflict)
        for record_id, outcome in sorted(outcomes.items())
        for conflict in outcome.conflicts
    ]
    if not conflicts:
        return
    print("Conflicts (existing values kept; --prefer-incoming FIELD takes the import's):")
    for record_id, (name, existing_value, incoming_value) in conflicts:
        print(
            f"  {record_id} {name}: existing {_format_merge_value(existing_value)} "
            f"| incoming {_format_merge_value(incoming_value)}"
        )


def _confirm_replace(count: int | None, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        # Fat-finger protection, not CI protection: unattended runs proceed.
        return True
    what = "the existing records" if count is None else f"{count} existing records"
    try:
        answer = input(f"Replace {what}? [y/N] ")
    except (EOFError, KeyboardInterrupt):
        # Closed stdin or Ctrl-C at the prompt reads as "no", not a traceback.
        print()
        return False
    return answer.strip().lower() in {"y", "yes"}


_NEEDS_READING_NOTES = (
    "Every record here has a reading janki cannot use: it is missing, or it is written "
    "in kanji. The reading is part of the record ID, so importing one would mint "
    "word:<expression>: or word:<kanji>:<kanji> — an ID that cannot be corrected later "
    "without orphaning its Anki review history. For each row worth keeping, fill in "
    "'reading' with kana AND delete that row's 'id:' line: the ID here is the malformed "
    "one, and an empty ID is re-minted from expression + reading when the file is read. "
    "Delete the rows not worth keeping. Then run 'janki validate' on this file — it "
    "lists every row still malformed — and promote it once a human has confirmed the "
    "readings."
)

_HELD_SUMMARY = "row(s) whose reading janki cannot use (missing, or written in kanji)"


def _stage_needs_reading(
    config: ProjectConfig, source_path: Path, records: list[VocabularyRecord]
) -> str:
    """Divert rows with an unusable reading to staging; return what to print.

    Called *before* the records are written. These rows exist nowhere but the
    source CSV — they are deliberately excluded from the records — so a staging
    failure has to abort the import, not report failure after vocabulary.json
    has already been replaced.
    """
    if not records:
        return ""
    target = config.staging_dir / f"shirabe-{source_path.stem}-needs-reading.yaml"
    if target.is_dir():
        # Not the "may hold review edits" case: a directory holds no edits, and
        # calling it one would have the import claim to protect something that
        # does not exist while the held rows go nowhere.
        raise StagingError(
            f"{target} is a directory, so the {len(records)} {_HELD_SUMMARY} cannot be "
            "staged there. Move it aside and re-run this import."
        )
    if target.is_file():
        return (
            f"Held {len(records)} {_HELD_SUMMARY} out of the import. {target} already "
            "exists and was left untouched — it may hold review edits that are not in "
            "git. Resolve that file, then re-run this import."
        )
    write_staging(
        target,
        records,
        {
            "source_file": source_path.name,
            "extracted_at": date.today().isoformat(),
            "review_notes": _NEEDS_READING_NOTES,
        },
    )
    return (
        f"Held {len(records)} {_HELD_SUMMARY} out of the import and wrote them to "
        f"{target} for reading review — a record without a usable reading gets a "
        f"malformed, uncorrectable ID. Run 'janki validate {target}' to see what is "
        "still outstanding."
    )


def command_import_shirabe(args: argparse.Namespace) -> int:
    config = _load_config(args)
    prefer_incoming = parse_prefer_incoming(args.prefer_incoming)
    result = import_file(args.file.resolve())
    output_path = (args.output or config.normalized_file).resolve()

    if args.replace:
        # --replace only needs the count, so an unreadable file must not block
        # the one command that can recover from it.
        count: int | None = None
        unreadable = False
        if output_path.exists():
            try:
                count = len(load_records(output_path))
            except DataError as exc:
                print(
                    f"warning: could not read the existing records to count them: {exc}",
                    file=sys.stderr,
                )
                unreadable = True
        # Nothing to lose when the file is absent or empty; still confirm when
        # it exists but could not be read, since it may hold curated records.
        if (unreadable or bool(count)) and not _confirm_replace(count, args.yes):
            print("Aborted: nothing was written.", file=sys.stderr)
            return 1
        existing: list[VocabularyRecord] = []
    else:
        existing = load_records(output_path) if output_path.exists() else []

    # Stage before committing the records: the held rows have no other home, and
    # a failure here must leave the import having done nothing rather than exit
    # non-zero over an output file it has already replaced.
    held_message = _stage_needs_reading(config, args.file, result.needs_reading)

    records, outcomes = merge_records(existing, result.records, prefer_incoming)
    save_records_json(output_path, records)
    print(f"Imported {len(result.records)} source rows into {output_path}")
    _print_merge_summary(outcomes)
    if held_message:
        print(held_message)
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
    return 0


def _validate_path(path: Path) -> tuple[list, int]:
    raw = load_structured(path)
    if isinstance(raw, dict) and "deck" in raw:
        _, records = resolve_deck_records(path)
    else:
        records = load_records(path)
    issues = validate_records(records, path)
    return issues, len(records)


def command_validate(args: argparse.Namespace) -> int:
    config = _load_config(args)
    paths: list[Path]
    if args.path:
        paths = [args.path.resolve()]
    else:
        paths = sorted(
            [*config.deck_dir.glob("*.yaml"), *config.deck_dir.glob("*.yml")]
        )
        if not paths:
            print(f"No deck files found under {config.deck_dir}", file=sys.stderr)
            return 1

    all_issues = []
    total_records = 0
    for path in paths:
        issues, count = _validate_path(path)
        all_issues.extend(issues)
        total_records += count

    _print_issues(all_issues)
    error_count = sum(issue.level == "error" for issue in all_issues)
    warning_count = sum(issue.level == "warning" for issue in all_issues)
    print(
        f"Validated {total_records} records across {len(paths)} file(s): "
        f"{error_count} error(s), {warning_count} warning(s)"
    )
    return 1 if has_errors(all_issues) else 0


def _build_one(deck_path: Path, config: ProjectConfig, output: Path | None = None) -> None:
    result = build_deck(deck_path, config, output)
    cards = ", ".join(result.card_types)
    print(
        f"Built {result.output_path} — {result.note_count} notes, "
        f"cards: {cards}, media: {result.media_count}"
    )


def command_build(args: argparse.Namespace) -> int:
    config = _load_config(args)
    if args.all:
        if args.deck:
            raise AnkiBuildError("Do not provide a deck path together with --all")
        deck_paths = sorted(
            [*config.deck_dir.glob("*.yaml"), *config.deck_dir.glob("*.yml")]
        )
        if not deck_paths:
            raise AnkiBuildError(f"No deck files found under {config.deck_dir}")
        for deck_path in deck_paths:
            _build_one(deck_path, config)
        return 0

    if not args.deck:
        raise AnkiBuildError("Provide a deck YAML path or use --all")
    output = args.output.resolve() if args.output else None
    _build_one(args.deck.resolve(), config, output)
    return 0


def command_status(args: argparse.Namespace) -> int:
    config = _load_config(args)
    book = ledger.load(config.ledger_file)
    universe = status.collect_records(config)

    # With --format ids, stdout carries nothing but ids so the output can be
    # piped straight into another command; everything a human reads goes to
    # stderr.
    ids_only = args.format == "ids"
    prose = sys.stderr if ids_only else sys.stdout

    for warning in universe.warnings:
        print(f"warning: {warning}", file=sys.stderr)

    if args.rebuild:
        summary = status.rebuild(book, universe.records, config.media_dir)
        book.save()
        for line in status.format_rebuild(summary, config.root):
            print(line, file=prose)

    report = status.build_report(config, universe, book)
    groups = status.find_duplicates(universe.records) if args.duplicates else []

    if ids_only:
        for record_id in status.selected_ids(
            report,
            groups,
            unexported=args.unexported,
            missing_audio=args.missing_audio,
            duplicates=args.duplicates,
        ):
            print(record_id)
        return 0

    for line in status.format_report(report):
        print(line)
    if args.unexported:
        for line in status.format_unexported(report):
            print(line)
    if args.missing_audio:
        for line in status.format_missing_audio(report):
            print(line)
    if args.duplicates:
        for line in status.format_duplicates(groups):
            print(line)
    if not (args.unexported or args.missing_audio or args.duplicates):
        print(
            "Details: --unexported, --missing-audio, --duplicates "
            "(add --format ids to pipe them)."
        )
    return 0


def command_preview(args: argparse.Namespace) -> int:
    config = _load_config(args)
    output = args.output or (config.dist_dir / f"{args.deck.stem}-preview.html")
    built = build_preview(args.deck.resolve(), output.resolve())
    print(f"Wrote preview to {built}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="janki",
        description="Import Japanese vocabulary and build deterministic Anki decks.",
    )
    parser.add_argument(
        "--root",
        type=_path,
        help="Project root containing janki.toml (normally discovered automatically).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect a CSV export")
    inspect_parser.add_argument("file", type=_path)
    inspect_parser.add_argument("--rows", type=int, default=5)
    inspect_parser.set_defaults(handler=command_inspect)

    import_parser = subparsers.add_parser(
        "import-shirabe", help="Import a Shirabe-style CSV export"
    )
    import_parser.add_argument("file", type=_path)
    import_parser.add_argument("--output", type=_path)
    import_parser.add_argument(
        "--replace",
        action="store_true",
        help=(
            "Replace output instead of merging with existing normalized records; "
            "prompts for confirmation on a terminal unless --yes."
        ),
    )
    import_parser.add_argument(
        "--prefer-incoming",
        metavar="FIELD[,FIELD]",
        help=(
            "Content fields the import may overwrite. By default an import only "
            "fills fields that are empty."
        ),
    )
    import_parser.add_argument(
        "--yes",
        action="store_true",
        help="Answer the --replace confirmation prompt with yes.",
    )
    import_parser.set_defaults(handler=command_import_shirabe)

    validate_parser = subparsers.add_parser("validate", help="Validate records or decks")
    validate_parser.add_argument("path", type=_path, nargs="?")
    validate_parser.set_defaults(handler=command_validate)

    build_command = subparsers.add_parser("build", help="Build one or all Anki decks")
    build_command.add_argument("deck", type=_path, nargs="?")
    build_command.add_argument("--all", action="store_true")
    build_command.add_argument("--output", type=_path)
    build_command.set_defaults(handler=command_build)

    status_parser = subparsers.add_parser(
        "status", help="Summarize records, ledger state, and duplicate candidates"
    )
    status_parser.add_argument(
        "--unexported",
        action="store_true",
        help="List the record ids each deck has never been built with.",
    )
    status_parser.add_argument(
        "--missing-audio",
        action="store_true",
        help="List the record ids that have no word audio.",
    )
    status_parser.add_argument(
        "--duplicates",
        action="store_true",
        help=(
            "List records that look like the same word twice: one expression under two "
            "ids, or one reading under a kanji and a kana spelling."
        ),
    )
    status_parser.add_argument(
        "--rebuild",
        action="store_true",
        help=(
            "Reconstruct the ledger entries records and media files still prove "
            "(sources and audio) and write the ledger. Export state cannot be rebuilt."
        ),
    )
    status_parser.add_argument(
        "--format",
        choices=("text", "ids"),
        default="text",
        help="'ids' prints bare record ids, one per line, for piping into other commands.",
    )
    status_parser.set_defaults(handler=command_status)

    preview_parser = subparsers.add_parser("preview", help="Build a static HTML preview")
    preview_parser.add_argument("deck", type=_path)
    preview_parser.add_argument("--output", type=_path)
    preview_parser.set_defaults(handler=command_preview)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except JankiError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
