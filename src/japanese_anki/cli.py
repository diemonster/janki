from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import date
from pathlib import Path

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
from japanese_anki.staging import write_staging
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
    "Every record here has kanji but no reading. The reading is part of the record ID, "
    "so importing one would mint word:<expression>: — an ID that cannot be corrected "
    "later without orphaning its Anki review history. Fill in 'reading' for the rows "
    "worth keeping, delete the rest, and promote the file once a human has confirmed "
    "the readings."
)


def _stage_needs_reading(
    config: ProjectConfig, source_path: Path, records: list[VocabularyRecord]
) -> None:
    """Divert reading-less kanji rows to a staging file instead of importing them."""
    if not records:
        return
    target = config.staging_dir / f"shirabe-{source_path.stem}-needs-reading.yaml"
    if target.exists():
        print(
            f"Held {len(records)} row(s) with kanji but no reading out of the import. "
            f"{target} already exists and was left untouched — it may hold review edits "
            "that are not in git. Resolve that file, then re-run this import."
        )
        return
    write_staging(
        target,
        records,
        {
            "source_file": source_path.name,
            "extracted_at": date.today().isoformat(),
            "review_notes": _NEEDS_READING_NOTES,
        },
    )
    print(
        f"Held {len(records)} row(s) with kanji but no reading out of the import and "
        f"wrote them to {target} for reading review — a record without a reading gets a "
        "malformed, uncorrectable ID."
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

    records, outcomes = merge_records(existing, result.records, prefer_incoming)
    save_records_json(output_path, records)
    print(f"Imported {len(result.records)} source rows into {output_path}")
    _print_merge_summary(outcomes)
    _stage_needs_reading(config, args.file, result.needs_reading)
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
