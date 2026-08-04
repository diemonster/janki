from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from japanese_anki.config import ConfigError, ProjectConfig
from japanese_anki.exporters.anki import AnkiBuildError, build_deck, resolve_deck_records
from japanese_anki.importers.shirabe import ShirabeImportError, import_file, inspect_file
from japanese_anki.io import (
    DataError,
    load_records,
    load_structured,
    merge_records,
    save_records_json,
)
from japanese_anki.preview import build_preview
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


def command_import_shirabe(args: argparse.Namespace) -> int:
    config = _load_config(args)
    result = import_file(args.file.resolve())
    output_path = (args.output or config.normalized_file).resolve()
    records = result.records
    counts = {"added": len(records), "updated": 0, "unchanged": 0}

    if output_path.exists() and not args.replace:
        existing = load_records(output_path)
        records, counts = merge_records(existing, records)

    save_records_json(output_path, records)
    print(f"Imported {len(result.records)} source rows into {output_path}")
    print(
        f"Merge result: {counts['added']} added, {counts['updated']} updated, "
        f"{counts['unchanged']} unchanged"
    )
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
        help="Replace output instead of merging with existing normalized records.",
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
    except (AnkiBuildError, ConfigError, DataError, ShirabeImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
