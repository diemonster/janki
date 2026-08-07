from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from japanese_anki import enrich, jpdb, ledger, migrate, status
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.exporters.anki import AnkiBuildError, build_deck, resolve_deck_records
from japanese_anki.importers import jpdb_import
from japanese_anki.importers.shirabe import import_file, inspect_file
from japanese_anki.io import (
    MERGE_LABELS,
    PREFER_INCOMING_PROTECTED,
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
from japanese_anki.staging import StagingError, read_staging, write_staging
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
        # The header's remedy does not apply to identity fields: the flag
        # refuses them, so pointing the user at it would send them into an
        # error. Say on the line itself that this one is a hand fix.
        note = (
            " (identity — resolve by hand; --prefer-incoming refuses it)"
            if name in PREFER_INCOMING_PROTECTED
            else ""
        )
        print(
            f"  {record_id} {name}{note}: existing {_format_merge_value(existing_value)} "
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
        answer = input(f"Replace {what} (their ledger entries go too)? [y/N] ")
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
    "lists every row still malformed. Once it is clean, this file is finished: move its "
    "records into vocabulary.json, run 'janki status --rebuild' so the ledger learns "
    "about them, and delete this file. ('janki promote' ships in Milestone 3 and will "
    "do those three steps for you; there is no need to wait for it.)"
)

_HELD_SUMMARY = "{unit} whose reading janki cannot use (missing, or written in kanji)"


def _held_summary(held_unit: str) -> str:
    return _HELD_SUMMARY.format(unit=held_unit)


def _staging_state(path: Path) -> str:
    """What a staging file already under review still needs.

    ``"clean"`` — every row validates, so the review has been resolved and
    telling its owner to "resolve that file, then re-run" would be a nag with no
    satisfiable end, since re-running never consumes it. ``"empty"`` — it holds
    no rows at all, which ``validate`` calls a warning and not an error, so
    calling it clean would tell its owner to move records that do not exist.
    ``"unresolved"`` — anything else, *including any failure to read it*: this
    decides which advice to print, and the cautious advice is the one that keeps
    a file of hand-typed readings nothing can regenerate.
    """
    try:
        records, _ = read_staging(path)
    except JankiError:
        return "unresolved"
    if not records:
        return "empty"
    return "unresolved" if has_errors(validate_records(records, path)) else "clean"


def _stage_needs_reading(
    config: ProjectConfig,
    staging_stem: str,
    source_ref: str,
    records: Sequence[VocabularyRecord],
    held_unit: str = "row(s)",
) -> str:
    """Divert rows with an unusable reading to staging; return what to print.

    Called *before* the records are written. These rows exist nowhere but the
    source they came from — they are deliberately excluded from the records —
    so a staging failure has to abort the import, not report failure after
    vocabulary.json has already been replaced.

    ``staging_stem`` names the file (``<stem>-needs-reading.yaml``),
    ``source_ref`` is recorded in it as the origin, and ``held_unit`` is what
    one held item is called. All three are the importer's to choose: a jpdb deck
    sync has no CSV file to take a stem from and no rows to count.
    """
    if not records:
        return ""
    summary = _held_summary(held_unit)
    if not staging_stem.strip():
        raise StagingError(
            f"Cannot stage {len(records)} {summary}: the import did not say what "
            "to name the staging file."
        )
    target = config.staging_dir / f"{staging_stem}-needs-reading.yaml"
    if target.is_dir():
        # Not the "may hold review edits" case: a directory holds no edits, and
        # calling it one would have the import claim to protect something that
        # does not exist while the held rows go nowhere.
        raise StagingError(
            f"{target} is a directory, so the {len(records)} {summary} cannot be "
            "staged there. Move it aside and re-run this import."
        )
    if target.is_file():
        held = f"Held {len(records)} {summary} out of the import. {target} already "
        state = _staging_state(target)
        if state == "empty":
            # Nothing to move and nothing to resolve: the reviewer kept none of
            # the rows. Reusing the "move its records" text would name records
            # that do not exist.
            return (
                held + "exists and was left untouched — it holds no rows, so nothing "
                "is waiting in it. Delete it, then re-run this import to stage these "
                "rows there."
            )
        if state == "clean":
            # The review is done; the import cannot consume the file, so the
            # only honest instruction is the one that ends the loop.
            return (
                held + "exists and was left untouched. Its review is finished — "
                "'janki validate' reports no errors for it — so move its records "
                "into your records file, run 'janki status --rebuild' so the ledger "
                "learns about them, and delete it."
            )
        return (
            held + "exists and was left untouched — it may hold review edits you have "
            "not committed. Resolve that file, then re-run this import."
        )
    write_staging(
        target,
        list(records),
        {
            # A historical key name: staging files predate importers that have
            # no file to name, and it holds whatever `source_ref` the importer
            # chose — a CSV filename, a jpdb deck. It is part of staging.py's
            # META_KEYS contract, so it is not this call site's to rename.
            "source_file": source_ref,
            "extracted_at": date.today().isoformat(),
            "review_notes": _NEEDS_READING_NOTES,
        },
    )
    return (
        f"Held {len(records)} {summary} out of the import and wrote them to "
        f"{target} for reading review — a record without a usable reading gets a "
        f"malformed, uncorrectable ID. Run 'janki validate {target}' to see what is "
        "still outstanding."
    )


def _ledger_line(added: int, sources: int, *, written: bool) -> str:
    """The one sentence every writing command says about the ledger."""
    if not written:
        return "Ledger: NOT written — the records landed, the ledger did not (see above)."
    return f"Ledger: registered {added} new record(s) and {sources} new source sighting(s)."


def _save_ledger(book: ledger.Ledger) -> ledger.LedgerError | None:
    """Save the ledger, returning the failure instead of raising it.

    A failing save must not take the summary down with it: the records are
    already written at this point, and an `error:` with no transcript tells the
    user the command did nothing when in fact it did almost everything.

    ``LedgerError`` is the whole contract, and it is ``ledger.Ledger.save``'s
    job to keep it that way: the filesystem failures a real save hits
    (permission denied, read-only mount, ENOSPC) reach it as ``DataError`` from
    the atomic writer, which is a sibling under ``JankiError`` and would sail
    straight past this handler into the generic one in ``main``.
    """
    try:
        book.save()
    except ledger.LedgerError as exc:
        return exc
    return None


def _report_ledger_failure(exc: ledger.LedgerError) -> None:
    print(f"warning: {exc}", file=sys.stderr)
    print(
        "The records are written; the ledger is not. Run 'janki status --rebuild' "
        "once that file is writable to recover the entries it did not get.",
        file=sys.stderr,
    )


def _prune_discarded(
    config: ProjectConfig,
    book: ledger.Ledger,
    discarded: Sequence[str],
    kept_ids: Sequence[str],
    written: Sequence[VocabularyRecord],
    output_path: Path,
) -> tuple[int, str]:
    """Drop the ledger entries of records ``--replace`` removed from the collection.

    Returns ``(entries dropped, why none were)``.

    "Discarded from this file" is not "gone from the collection". The ledger is
    collection-wide while ``--replace`` rewrites exactly one file, and the
    collection is the normalized file *plus every deck's inline notes* (see
    ``status.collect_records``) — so a record ``--replace`` dropped from
    vocabulary.json may still be sitting in a deck, and ``--output`` can point
    ``--replace`` at a file that is not part of the collection at all.

    The asymmetry decides every judgement call here. An entry left behind
    over-reports the collection, which ``status`` shows and a later import
    fixes. An entry removed by mistake destroys ``added_at`` and ``exports``,
    which ``status --rebuild`` documents as *not* reconstructible by anything.
    So an id is pruned only when the surviving set can be built and does not
    contain it; when it cannot be built — an ``--output`` outside the
    collection, a deck that will not parse — every entry stays and the caller
    says why.
    """
    keep = set(kept_ids)
    candidates = [
        record_id
        for record_id in dict.fromkeys(discarded)
        if record_id not in keep and record_id in book.records
    ]
    if not candidates:
        return 0, ""
    if output_path.resolve() != config.normalized_file.resolve():
        return 0, (
            f"--output wrote {output_path}, which is not the records file, so the "
            f"{len(candidates)} discarded record(s) may still be in the collection: "
            "their ledger entries were kept."
        )
    surviving, unreadable = status.surviving_ids(config, written)
    if unreadable:
        return 0, (
            f"the collection could not be read in full ({'; '.join(unreadable)}), so "
            f"the {len(candidates)} discarded record(s) kept their ledger entries."
        )
    dropped = sum(
        book.remove(record_id) for record_id in candidates if record_id not in surviving
    )
    still_here = len(candidates) - dropped
    if not still_here:
        return dropped, ""
    return dropped, (
        f"{still_here} of the discarded record(s) are still in the collection (a deck "
        "carries them inline), so their ledger entries were kept."
    )


def run_import(
    config: ProjectConfig,
    records: Sequence[VocabularyRecord],
    *,
    source_type: str,
    source_ref: str,
    output_path: Path,
    unit: str,
    staging_stem: str,
    needs_reading: Sequence[VocabularyRecord] = (),
    held_unit: str = "row(s)",
    warnings: Sequence[str] = (),
    prefer_incoming: Sequence[str] = (),
    replace: bool = False,
    assume_yes: bool = False,
) -> int:
    """Land an import: stage, merge, write records, write the ledger, report.

    The one pipeline every importer shares. ``import-shirabe`` calls it today
    and ``import-jpdb`` calls it from Milestone 2, so nothing here may assume a
    CSV: what the source is called (``source_type``/``source_ref``), what one
    incoming item is called in the summary (``unit``), what one *held* item is
    called (``held_unit``), and what a staging file for it is named
    (``staging_stem``) are all the caller's to say.

    **``source_ref`` must be the same value the importer stored as each
    record's own ``source.imported_from``.** ``status --rebuild`` reconstructs
    a source reference from that field, and a reference's identity is every key
    but ``seen_at`` — so a ``source_ref`` of any other shape (or any extra
    detail passed alongside it) has the documented recovery command append a
    second, near-duplicate reference to every record it just imported. This is a
    cross-module contract: ``migrate.migrate_inline`` satisfies it too, and
    M2.5's importer must.

    Ordering is load-bearing, in both directions:

    * Everything that can refuse the import — an unreadable ledger, a staging
      file that cannot be written — happens *before* ``vocabulary.json`` is
      replaced. Held rows exist nowhere else, so a late failure would report
      that the import did not happen over records it has already rewritten.
    * The records are written before the ledger. The ledger is metadata and
      ``janki status --rebuild`` can reconstruct it; the records cannot be
      reconstructed from anything. A ledger that will not save is therefore a
      warning over a full summary, never an error instead of one.
    """
    discarded: list[str] = []
    if replace:
        # --replace only needs the count, so an unreadable file must not block
        # the one command that can recover from it.
        count: int | None = None
        unreadable = False
        if output_path.exists():
            try:
                replaced = load_records(output_path)
                count = len(replaced)
                discarded = [record.id for record in replaced]
            except DataError as exc:
                print(
                    f"warning: could not read the existing records to count them: {exc}. "
                    "Their ledger entries cannot be identified either, so they stay — "
                    "the ids are unknown, and guessing would drop live records.",
                    file=sys.stderr,
                )
                unreadable = True
        # Nothing to lose when the file is absent or empty; still confirm when
        # it exists but could not be read, since it may hold curated records.
        if (unreadable or bool(count)) and not _confirm_replace(count, assume_yes):
            print("Aborted: nothing was written.", file=sys.stderr)
            return 1
        existing: list[VocabularyRecord] = []
    else:
        existing = load_records(output_path) if output_path.exists() else []

    # Read the ledger before anything is written, and only once: it is a
    # whole-file rewrite, so a command loads it once and saves it once.
    book = ledger.load(config.ledger_file)

    held_message = _stage_needs_reading(
        config, staging_stem, source_ref, needs_reading, held_unit
    )

    merged, outcomes = merge_records(existing, list(records), prefer_incoming)
    save_records_json(output_path, merged)

    # Held rows are deliberately absent here: they never reached the records, so
    # the ledger must not claim janki has them.
    added = sum(
        book.record_added(record_id)
        for record_id, outcome in outcomes.items()
        if outcome.label == "added"
    )
    # Every incoming record, added or not. This is what makes "later sightings
    # go to the ledger" true: the merge leaves the record's own `source` alone,
    # so a re-import's provenance would otherwise be lost. Keep the reference to
    # {type, ref} — see the docstring's cross-module contract.
    incoming_ids = dict.fromkeys(record.id for record in records)
    seen = sum(
        book.record_source_seen(record_id, source_type, source_ref)
        for record_id in incoming_ids
    )
    # --replace discards records; their ledger entries would otherwise outlive
    # them forever, over-reporting the collection and — once M5.6 makes
    # `build --only-new` read `exports` — silently omitting a re-imported record
    # from a deck because a dead entry says it was already exported. Only what
    # actually left the collection is pruned; see _prune_discarded.
    dropped, kept_reason = _prune_discarded(
        config, book, discarded, list(incoming_ids), merged, output_path
    )
    ledger_error = _save_ledger(book)

    print(f"Imported {len(records)} {unit} into {output_path}")
    _print_merge_summary(outcomes)
    if held_message:
        print(held_message)
    print(_ledger_line(added, seen, written=ledger_error is None))
    if dropped and ledger_error is None:
        print(
            f"  Dropped {dropped} ledger entr{'y' if dropped == 1 else 'ies'} "
            "for records --replace discarded."
        )
    if kept_reason and ledger_error is None:
        print(f"  Kept ledger entries: {kept_reason}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    if ledger_error is not None:
        _report_ledger_failure(ledger_error)
        return 1
    return 0


def command_import_shirabe(args: argparse.Namespace) -> int:
    config = _load_config(args)
    prefer_incoming = parse_prefer_incoming(args.prefer_incoming)
    source_path = args.file.resolve()
    result = import_file(source_path)
    return run_import(
        config,
        result.records,
        source_type="shirabe",
        # The file's name, matching what the importer stored as the record's own
        # `source.imported_from` — `status --rebuild` reconstructs the reference
        # from that field, and a mismatch would double every entry.
        source_ref=source_path.name,
        output_path=(args.output or config.normalized_file).resolve(),
        unit="source rows",
        staging_stem=f"shirabe-{source_path.stem}",
        needs_reading=result.needs_reading,
        warnings=result.warnings,
        prefer_incoming=prefer_incoming,
        replace=args.replace,
        assume_yes=args.yes,
    )


def command_import_jpdb(args: argparse.Namespace) -> int:
    """Import from jpdb — a userscript CSV, or decks straight from the API.

    A deck import runs the whole pipeline **once per deck** rather than pooling
    every deck's words into one pass. That is what keeps ``run_import``'s
    ``source_ref`` contract satisfiable: a record's ``source.imported_from`` is
    the deck it came from, so one label per call is the only shape that matches
    it. A word in two decks earns a ledger reference for each, which is the
    truth about where it came from, and the second pass merges over the first
    with the usual existing-wins semantics.
    """
    chosen = [
        label
        for label, given in (
            ("FILE", args.file is not None),
            ("--deck", bool(args.deck)),
            ("--all-decks", args.all_decks),
        )
        if given
    ]
    if len(chosen) != 1:
        raise JankiError(
            "import-jpdb needs exactly one source: a CSV file, --deck NAME "
            f"(repeatable), or --all-decks. Got {', '.join(chosen) or 'none'}."
        )

    config = _load_config(args)
    prefer_incoming = parse_prefer_incoming(args.prefer_incoming)
    output_path = (args.output or config.normalized_file).resolve()

    if args.file is not None:
        source_path = args.file.resolve()
        result = jpdb_import.import_csv(source_path)
        return run_import(
            config,
            result.records,
            source_type="jpdb",
            source_ref=source_path.name,
            output_path=output_path,
            unit="source rows",
            staging_stem=f"jpdb-{source_path.stem}",
            needs_reading=result.needs_reading,
            warnings=result.warnings,
            prefer_incoming=prefer_incoming,
            replace=args.replace,
            assume_yes=args.yes,
        )

    client = jpdb.JpdbClient(jpdb.api_key_from_env())
    available = client.list_user_decks()
    decks = available if args.all_decks else jpdb_import.select_decks(available, args.deck)
    if not decks:
        print("No jpdb decks on this account; nothing to import.", file=sys.stderr)
        return 0

    exit_code = 0
    for index, deck in enumerate(decks):
        name = str(deck.get("name", "")).strip()
        if len(decks) > 1:
            # A blank line between decks: each pass prints a full summary, and
            # run together they read as one confusing report.
            print(f"{'' if index == 0 else chr(10)}jpdb deck: {name}")
        result = jpdb_import.import_deck(client, deck)
        # --replace against several decks would have deck two discard deck one.
        # It is honoured on the first pass only; the rest merge, which is what
        # "replace the collection with these decks" has to mean.
        code = run_import(
            config,
            result.records,
            source_type="jpdb",
            # The deck name, matching what the importer stored as each record's
            # own `source.imported_from` — see run_import's contract.
            source_ref=name,
            output_path=output_path,
            unit="jpdb entries",
            staging_stem=f"jpdb-{_slug_for_file(name)}",
            needs_reading=result.needs_reading,
            held_unit="entr(y/ies)",
            warnings=result.warnings,
            prefer_incoming=prefer_incoming,
            replace=args.replace and index == 0,
            assume_yes=args.yes,
        )
        exit_code = exit_code or code
    return exit_code


def _slug_for_file(name: str) -> str:
    """A deck name reduced to something safe as part of a filename.

    Deck names carry colons, slashes and spaces; a staging file named after one
    verbatim would land in the wrong directory or refuse to be created.
    """
    return jpdb_import.deck_tag(name).removeprefix("jpdb:") or "deck"


def _confirm_enrich(count: int, assume_yes: bool) -> bool:
    if assume_yes or not sys.stdin.isatty():
        # Same rule as --replace: fat-finger protection, not CI protection.
        return True
    try:
        answer = input(f"Write these changes to {count} record(s)? [y/N] ")
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer.strip().lower() in {"y", "yes"}


def _enrich_staging(
    client: jpdb.JpdbClient, path: Path, assume_yes: bool
) -> int:
    """Annotate a needs-reading staging file with the readings jpdb proposes.

    The whole API pass happens before the file is touched. A staging file holds
    hand-typed readings that exist nowhere else, so a run that dies halfway
    must leave the reviewer exactly what they had — which is also why this is
    the one caller that legitimately passes ``force=True`` to ``write_staging``
    (the no-overwrite guard is there to stop an *import* landing on a review in
    progress; the file being annotated here is by definition already there).
    """
    records, meta = read_staging(path)
    result = enrich.suggest_readings(client, records)
    if not result.held:
        print(f"No held rows in {path}; nothing to suggest readings for.")
        return 0

    for record_id, reading in result.suggested.items():
        print(f"  {record_id}: {reading}")
    if result.suggested:
        if not _confirm_enrich(len(result.suggested), assume_yes):
            print("Aborted: the staging file was not touched.", file=sys.stderr)
            return 1
        write_staging(path, result.records, meta, force=True)

    print(
        f"Suggested a reading for {len(result.suggested)} of {result.held} "
        f"held row(s) in {path}."
    )
    print(
        "  Each is a proposal in 'suggested_reading'; the row stays held until a "
        "human types the reading into 'reading' and deletes the row's 'id:' line."
    )
    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


def command_enrich(args: argparse.Namespace) -> int:
    """Fill empty fields on existing records from a dictionary.

    ``--jpdb`` is the only source implemented; ``--ai`` arrives in M4.2. One is
    required, because "enrich" without saying from what is a command whose
    behavior would change under the user when the second source lands.
    """
    if args.ai:
        raise JankiError(
            "enrich --ai is not implemented until M4.2. Use --jpdb for dictionary "
            "enrichment (furigana, pitch accent, frequency rank, part of speech)."
        )
    if not args.jpdb:
        raise JankiError(
            "enrich needs a source: --jpdb fills fields from the jpdb dictionary. "
            "(--ai arrives in M4.2.)"
        )
    force_fields = enrich.parse_force_fields(args.force_fields)
    if args.staging is not None and (force_fields or args.ids):
        raise JankiError(
            "enrich --staging proposes readings for held rows and writes nothing "
            "else, so it takes neither --force-fields nor record ids."
        )

    config = _load_config(args)
    client = jpdb.JpdbClient(jpdb.api_key_from_env())

    if args.staging is not None:
        return _enrich_staging(client, args.staging.resolve(), args.yes)

    output_path = config.normalized_file.resolve()
    records = load_records(output_path) if output_path.exists() else []
    if not records:
        print(f"No records to enrich in {output_path}.")
        return 0

    # Read the ledger before anything is written, and only once.
    book = ledger.load(config.ledger_file)
    result = enrich.enrich_records(
        client, records, force_fields=force_fields, ids=args.ids or None
    )

    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if not result.changes:
        print(
            f"Nothing to fill: looked up {result.looked_up} record(s), skipped "
            f"{result.skipped} with no empty fields."
        )
        return 0

    for line in enrich.format_field_diff(result.changes):
        print(line)
    if not _confirm_enrich(len(result.changes), args.yes):
        print("Aborted: nothing was written.", file=sys.stderr)
        return 1

    save_records_json(output_path, result.records)
    for record_id, changed in result.changes.items():
        book.record_enriched(record_id, kind="jpdb", model="jpdb", fields=changed)
    ledger_error = _save_ledger(book)

    print(
        f"Enriched {len(result.changes)} record(s) in {output_path} "
        f"(looked up {result.looked_up}, skipped {result.skipped} with no empty fields)."
    )
    if ledger_error is None:
        print(f"Ledger: recorded a jpdb pass over {len(result.changes)} record(s).")
    else:
        _report_ledger_failure(ledger_error)
        return 1
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
    staged, staged_warnings = status.collect_staged(config)

    # With --format ids, stdout carries nothing but ids so the output can be
    # piped straight into another command; everything a human reads goes to
    # stderr.
    ids_only = args.format == "ids"
    prose = sys.stderr if ids_only else sys.stdout

    for warning in [*universe.warnings, *staged_warnings]:
        print(f"warning: {warning}", file=sys.stderr)

    if args.rebuild:
        summary = status.rebuild(
            book, universe.records, config.media_dir, sources_by_id=universe.normalized_sources
        )
        book.save()
        for line in status.format_rebuild(summary, config.root):
            print(line, file=prose)

    report = status.build_report(config, universe, book, staged)
    groups = status.find_duplicates(universe.records) if args.duplicates else []

    if ids_only:
        for record_id in status.selected_ids(
            report,
            groups,
            unexported=args.unexported,
            missing_audio=args.missing_audio,
            duplicates=args.duplicates,
            staged=args.staged,
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
    if args.staged:
        for line in status.format_staged(report):
            print(line)
    if not (args.unexported or args.missing_audio or args.duplicates or args.staged):
        print(
            "Details: --unexported, --missing-audio, --duplicates, --staged "
            "(add --format ids to pipe them)."
        )
    return 0


def command_migrate_inline(args: argparse.Namespace) -> int:
    config = _load_config(args)
    # Load once, save once: the ledger is a whole-file rewrite.
    book = ledger.load(config.ledger_file)
    result = migrate.migrate_inline(args.deck.resolve(), config, book)

    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if not result.changed:
        print(
            f"Nothing to migrate: {args.deck} has no inline notes. "
            "Its records already live in the normalized file."
        )
        return 0

    # The ledger is saved before the transcript that describes it: everything
    # else is already on disk, and a summary claiming registrations that did not
    # persist is worse than a summary that says so.
    ledger_error = _save_ledger(book)
    print(
        f"Migrated {len(result.migrated)} inline note(s) from {args.deck} into "
        f"{result.normalized_file}"
    )
    _print_merge_summary(result.outcomes)
    for line in migrate.format_details(result, config.root):
        print(line)
    print(
        _ledger_line(
            result.ledger_added, result.ledger_sources, written=ledger_error is None
        )
    )
    if ledger_error is not None:
        _report_ledger_failure(ledger_error)
        return 1
    return 0


def command_jpdb_ping(args: argparse.Namespace) -> int:
    """Check that jpdb answers and that JPDB_API_KEY is accepted.

    No config is loaded on purpose: this command needs a network and an
    environment variable, not a project, so it works from anywhere — which is
    exactly where someone debugging a key will run it.
    """
    client = jpdb.JpdbClient(jpdb.api_key_from_env())
    client.ping()
    print(f"jpdb: ok — the API answered and {jpdb.API_KEY_ENV} was accepted.")
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

    jpdb_import_parser = subparsers.add_parser(
        "import-jpdb",
        help="Import jpdb decks over the API, or a JPDB-Export userscript CSV",
    )
    jpdb_import_parser.add_argument(
        "file",
        type=_path,
        nargs="?",
        help="A JPDB-Export userscript CSV. Omit to import decks over the API.",
    )
    jpdb_import_parser.add_argument(
        "--deck",
        action="append",
        default=[],
        metavar="NAME",
        help="A jpdb deck to import, by name. Repeat for several.",
    )
    jpdb_import_parser.add_argument(
        "--all-decks",
        action="store_true",
        help="Import every deck on the account.",
    )
    jpdb_import_parser.add_argument("--output", type=_path)
    jpdb_import_parser.add_argument(
        "--replace",
        action="store_true",
        help=(
            "Replace output instead of merging with existing normalized records; "
            "prompts for confirmation on a terminal unless --yes. With several "
            "decks it applies to the first only, so the rest are not discarded."
        ),
    )
    jpdb_import_parser.add_argument(
        "--prefer-incoming",
        metavar="FIELD[,FIELD]",
        help=(
            "Content fields the import may overwrite. By default an import only "
            "fills fields that are empty."
        ),
    )
    jpdb_import_parser.add_argument(
        "--yes",
        action="store_true",
        help="Answer the --replace confirmation prompt with yes.",
    )
    jpdb_import_parser.set_defaults(handler=command_import_jpdb)

    enrich_parser = subparsers.add_parser(
        "enrich",
        help="Fill empty fields on existing records from a dictionary",
    )
    enrich_parser.add_argument(
        "ids",
        nargs="*",
        metavar="ID",
        help="Record ids to enrich. Omit to consider every record.",
    )
    enrich_parser.add_argument(
        "--jpdb",
        action="store_true",
        help="Fill fields from the jpdb dictionary.",
    )
    enrich_parser.add_argument(
        "--ai",
        action="store_true",
        help="Fill fields with the Claude API (not implemented until M4.2).",
    )
    enrich_parser.add_argument(
        "--force-fields",
        metavar="FIELD[,FIELD]",
        help=(
            "Fields the pass may overwrite. By default it only fills empty ones. "
            f"Valid: {', '.join(enrich.ENRICHABLE_FIELDS)}."
        ),
    )
    enrich_parser.add_argument(
        "--staging",
        type=_path,
        metavar="FILE",
        help=(
            "Annotate a needs-reading staging file with the reading jpdb proposes "
            "for each held row, for a human to confirm."
        ),
    )
    enrich_parser.add_argument(
        "--yes",
        action="store_true",
        help="Write the shown changes without confirming.",
    )
    enrich_parser.set_defaults(handler=command_enrich)

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
        "--staged",
        action="store_true",
        help=(
            "List the record ids waiting in data/staging for a human, with the "
            "reason each was held back."
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

    migrate_parser = subparsers.add_parser(
        "migrate-inline",
        help="Move a deck's inline notes into the normalized records file",
    )
    migrate_parser.add_argument("deck", type=_path)
    migrate_parser.set_defaults(handler=command_migrate_inline)

    jpdb_parser = subparsers.add_parser("jpdb", help="Talk to the jpdb.io API")
    # `required=True` so a bare `janki jpdb` prints usage instead of failing on
    # a missing handler; the group exists to hold `import-jpdb`'s neighbours as
    # later milestones add them.
    jpdb_commands = jpdb_parser.add_subparsers(dest="jpdb_command", required=True)
    jpdb_ping_parser = jpdb_commands.add_parser(
        "ping", help=f"Check the jpdb API and the {jpdb.API_KEY_ENV} key"
    )
    jpdb_ping_parser.set_defaults(handler=command_jpdb_ping)

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
