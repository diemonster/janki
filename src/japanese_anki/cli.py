from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

from japanese_anki import (
    audio_cmd,
    claude_client,
    enrich,
    extract,
    jpdb,
    ledger,
    migrate,
    promote,
    status,
)
from japanese_anki.audio_cmd import AudioError
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.exporters.anki import AnkiBuildError, build_deck, resolve_deck_records
from japanese_anki.identifiers import short_fingerprint
from japanese_anki.importers import jpdb_import, jpdb_reviews
from japanese_anki.importers.shirabe import import_file, inspect_file
from japanese_anki.inputs import prepare_inputs
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
from japanese_anki.promote import PromoteError
from japanese_anki.staging import (
    STAGING_SUFFIXES,
    StagingError,
    check_rewritable,
    prune_staging,
    read_staging,
    rewrite_staging,
    write_staging,
)
from japanese_anki.tts import openai_tts, voicevox
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


def _print_merge_summary(
    outcomes: dict[str, MergeOutcome], *, prefer_incoming_available: bool = True
) -> None:
    """Report what a merge did, and how to resolve what it would not.

    ``prefer_incoming_available`` because the remedy is not universal: only the
    import commands take ``--prefer-incoming``. Printing that advice elsewhere
    hands the reader a command that exits with "unrecognized arguments" —
    output worse than silence, because it reads like the tool telling them what
    to do next. ``promote`` merges existing-wins; ``migrate-inline`` prefers the
    inline note field by field (it is the authoritative copy there) and refuses
    the identity disagreements it cannot merge. Neither has the flag, which is
    the whole of the argument.

    The *identity* marker is not part of that: a disagreement about
    ``expression`` or ``reading`` is not one more field to settle by hand, it is
    the two copies disagreeing about which word this is, and the record id is
    derived from them. So it keeps a marker wherever it prints — one that names
    the flag only where the flag exists.
    """
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
    if prefer_incoming_available:
        header = (
            "Conflicts (existing values kept; --prefer-incoming FIELD takes the "
            "import's):"
        )
    else:
        header = "Conflicts (existing values kept; resolve these by hand):"
    print(header)
    for record_id, (name, existing_value, incoming_value) in conflicts:
        # The header's remedy does not apply to identity fields: the flag
        # refuses them, so pointing the user at it would send them into an
        # error. Say on the line itself that this one is a hand fix.
        if name not in PREFER_INCOMING_PROTECTED:
            note = ""
        elif prefer_incoming_available:
            note = " (identity — resolve by hand; --prefer-incoming refuses it)"
        else:
            note = " (identity — the two copies disagree about which word this is)"
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
    "lists every row still malformed. Once it is clean, run 'janki promote' on it: that "
    "merges the surviving records into vocabulary.json, registers them in the ledger, "
    "archives them under data/staging/done/, and deletes this file once nothing is left "
    "held back. Rows it still cannot accept stay here with the reason written in."
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
                "'janki validate' reports no errors for it — so run 'janki promote' "
                "on it: that lands its records, archives them, and removes the file."
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


def _report_enrichment_ledger_failure(
    exc: ledger.LedgerError, *, rerun: str, aftermath: str
) -> None:
    """The honest report for an ``enriched`` entry that did not get written.

    Deliberately not :func:`_report_ledger_failure`: its advice is
    ``status --rebuild``, which reconstructs source references and audio from
    what the records and media files *prove*. An ``enriched`` entry is provable
    by nothing — a filled field does not say who filled it — so that advice
    would promise a recovery that silently never happens, and report success
    while doing it.

    What a *re-run* does differs by pass, which is why the caller supplies that
    sentence rather than this function guessing at one, and why there is no
    two-way split worth generalising. ``--jpdb`` never reaches the ledger, since
    what it could fill is filled — but whether it pays for a look-up first
    depends on the word: a record the dictionary described completely is
    skipped, and one it described partly is looked up again and proposes
    nothing new. (A noun is the second kind: its ``conjugations`` never fill, so
    it stays fillable forever.) ``--ai`` does reach the ledger — a record with
    examples and no usage note is a target again. ``--polish-meanings`` looks at
    every record every time. Whichever it is, saying it accurately is the whole
    point of this message: nobody should chase a repair on a wrong description
    of it.
    """
    print(f"warning: {exc}", file=sys.stderr)
    print(
        "The records are written; the ledger entry recording it is not. "
        "'status --rebuild' cannot bring it back — an enrichment pass is not "
        f"provable from the records. {rerun} {aftermath}",
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
        if code != 0:
            # A pass that did not land stops the run rather than being carried
            # as an exit code past the decks after it. Both ways it fails say
            # so: a declined --replace has already printed "nothing was
            # written", and importing the *rest* of the account on top of that
            # refusal would leave a collection missing exactly one deck, with
            # the summary of every other deck reading like a complete import.
            remaining = [
                str(other.get("name", "")).strip() for other in decks[index + 1 :]
            ]
            if remaining:
                print(
                    f"Stopped at '{name}': {len(remaining)} deck(s) not imported "
                    f"({', '.join(remaining)}).",
                    file=sys.stderr,
                )
            return code
    return 0


def _slug_for_file(name: str) -> str:
    """A deck name reduced to something safe, readable and unique in a filename.

    Deck names carry colons, slashes and spaces; a staging file named after one
    verbatim would land in the wrong directory or refuse to be created. The
    flattening that fixes that is lossy — ``Lesson 1``, ``lesson-1`` and
    ``Lesson: 1`` all reduce to ``lesson-1`` — so a fingerprint of the full name
    is appended. Without it, two such decks share one needs-reading file: the
    first deck writes it, the second is told the file already exists and to
    resolve it and re-run, and re-running has the first deck claim it again.
    The advice can never converge and those held rows are never stageable.

    The fingerprint is of the deck name alone, so it is stable across runs — the
    same deck finds the same file next month — and it is deliberately *not* the
    tag: ``jpdb:<slug>`` is what a re-import matches on and what a human types
    into a deck filter, so it stays readable and its collisions stay harmless.
    """
    slug = jpdb_import.deck_tag(name).removeprefix("jpdb:")
    return f"{slug}-{short_fingerprint(name, length=8)}" if slug else "deck"


def command_import_jpdb_reviews(args: argparse.Namespace) -> int:
    """Mark the records jpdb is already drilling, so a deck can leave them out.

    Creates nothing: an entry matching no record is reported, because a word in
    jpdb that janki does not have is a fact worth seeing and not an error. What
    it writes is the ``jpdb-known`` tag and a review count in ``raw_fields`` —
    the count deliberately not in the ledger, whose references are identified by
    every key but ``seen_at``, so a weekly run would append a near-duplicate
    line per record forever.
    """
    config = _load_config(args)
    source_path = args.file.resolve()
    output_path = config.normalized_file.resolve()
    records = load_records(output_path) if output_path.exists() else []
    if not records:
        print(f"No records to match against in {output_path}.")
        return 0

    entries, skipped = jpdb_reviews.read_reviews(source_path)
    # Read the ledger before anything is written, and only once.
    book = ledger.load(config.ledger_file)
    result = jpdb_reviews.apply_reviews(records, entries)

    for section in skipped:
        print(
            f"warning: skipped '{section}' in {source_path.name}: not a vocabulary "
            "card list, so its entries are not words janki could match",
            file=sys.stderr,
        )

    if result.changed:
        save_records_json(output_path, result.records)
    # Every matched record, changed or not: the sighting says this export saw
    # the word, which is true whether or not its count moved since last time.
    seen = sum(
        book.record_source_seen(record_id, "jpdb-reviews", source_path.name)
        for record_id in result.matched
    )
    ledger_error = _save_ledger(book) if result.matched else None

    print(
        f"Matched {len(result.matched)} of {len(entries)} jpdb entr"
        f"{'y' if len(entries) == 1 else 'ies'} against {output_path}."
    )
    print(
        f"  Tagged '{jpdb_reviews.KNOWN_TAG}' and updated review counts on "
        f"{len(result.changed)} record(s); the rest already said the same thing."
    )
    if result.unmatched:
        shown = ", ".join(entry.label for entry in result.unmatched[:5])
        more = "" if len(result.unmatched) <= 5 else f", and {len(result.unmatched) - 5} more"
        print(
            f"  {len(result.unmatched)} entr"
            f"{'y' if len(result.unmatched) == 1 else 'ies'} matched no record: "
            f"{shown}{more}. These are words jpdb knows and janki does not; "
            "'janki import-jpdb' is what adds them."
        )
    print(_ledger_line(0, seen, written=ledger_error is None))
    if ledger_error is not None:
        _report_ledger_failure(ledger_error)
        return 1
    return 0


#: Where a large AI pass puts its proposals — live or batched, one file, so a
#: run of either kind refuses rather than overwrite the other's review.
STAGING_FILE_NAME = "ai-enrichment.yaml"


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
    hand-typed readings that exist nowhere else, so a run that dies halfway must
    leave the reviewer exactly what they had — and the write itself goes through
    ``rewrite_staging``, which edits the document rather than re-rendering it, so
    a comment or a key the reviewer added survives being annotated.
    """
    records, _meta = read_staging(path)
    # Before the API pass, not after it: a file that cannot be rewritten
    # faithfully should say so while the only thing spent is a file read.
    check_rewritable(path)
    result = enrich.suggest_readings(client, records)
    if not result.held:
        print(f"No held rows in {path}; nothing to suggest readings for.")
        return 0

    # Before the gate, not after it: a row jpdb could not help with is part of
    # what the user is being asked to approve, and answering "no" must not be
    # the reason they never saw it.
    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    for record_id, reading in result.suggested.items():
        print(f"  {record_id}: {reading}")
    if result.suggested:
        if not _confirm_enrich(len(result.suggested), assume_yes):
            print("Aborted: the staging file was not touched.", file=sys.stderr)
            return 1
        rewrite_staging(path, result.records)

    print(
        f"Suggested a reading for {len(result.suggested)} of {result.held} "
        f"held row(s) in {path}."
    )
    print(
        "  Each is a proposal in 'suggested_reading'; the row stays held until a "
        "human types the reading into 'reading' and deletes the row's 'id:' line."
    )
    return 0


def command_enrich(args: argparse.Namespace) -> int:
    """Fill empty fields on existing records from a dictionary.

    Three passes, one per run: ``--jpdb`` fills what a dictionary knows,
    ``--ai`` writes what it does not, and ``--polish-meanings`` rewrites English
    that is already there. One is required, because "enrich" without saying how
    is a command whose meaning depends on which pass is newest.
    """
    passes = [
        name
        for name, chosen in (
            ("--jpdb", args.jpdb),
            ("--ai", args.ai),
            ("--polish-meanings", args.polish_meanings),
        )
        if chosen
    ]
    if len(passes) > 1:
        raise JankiError(
            "enrich takes one pass at a time: --jpdb fills what a dictionary "
            "knows, --ai writes what it does not, and --polish-meanings rewrites "
            f"English that is already there. Got {', '.join(passes)}. Run them "
            "separately so each shows you its own diff."
        )
    if not passes:
        raise JankiError(
            "enrich needs a pass: --jpdb fills fields from the jpdb dictionary, "
            "--ai writes examples and usage notes, --polish-meanings proposes "
            "better English glosses."
        )
    batch_flags = [
        name
        for name, chosen in (
            ("--batch-submit", args.batch_submit),
            ("--batch-fetch", args.batch_fetch),
            ("--batch-forget", args.batch_forget),
        )
        if chosen
    ]
    if len(batch_flags) > 1:
        raise JankiError(
            f"enrich takes one batch action at a time; got {', '.join(batch_flags)}. "
            "--batch-submit sends a batch, --batch-fetch collects it hours later, "
            "and --batch-forget drops one that can no longer be collected."
        )
    if (args.batch_fetch or args.batch_forget) and (args.ids or args.model):
        raise JankiError(
            "enrich --batch-fetch and --batch-forget act on a batch that was "
            "already submitted, so neither takes record ids or a model: which "
            "records it covers and which model answered them were both decided "
            "at submit time and are recorded in the ledger. --batch-forget in "
            "particular drops the whole entry, so naming one id and dropping "
            "the rest is not something it could mean. (--batch-fetch does take "
            "--force-fields: that decides how an answer already in hand is "
            "applied.)"
        )
    if args.batch_forget and (args.force_fields or args.force):
        raise JankiError(
            "enrich --batch-forget drops the pending entry and writes no "
            "records, so it has no field list to widen and no staging file to "
            "overwrite."
        )
    if batch_flags and not args.ai:
        raise JankiError(
            f"enrich {batch_flags[0]} batches the --ai pass, so it needs --ai. "
            "The dictionary pass is not billed per token and the polish pass is "
            "confirmed one record at a time, so neither has anything to batch."
        )
    if args.polish_meanings and args.force_fields:
        raise JankiError(
            "enrich --polish-meanings writes 'meanings' and nothing else, so "
            "there is no field list to widen. It always overwrites — that is "
            "what it is for, and why it confirms one record at a time."
        )
    force_fields = enrich.parse_force_fields(args.force_fields, ai=args.ai)
    if args.staging is not None and (force_fields or args.ids):
        raise JankiError(
            "enrich --staging proposes readings for held rows and writes nothing "
            "else, so it takes neither --force-fields nor record ids."
        )
    if args.staging is not None and (args.ai or args.polish_meanings):
        raise JankiError(
            "enrich --staging annotates held rows with the reading jpdb proposes; "
            f"{passes[0]} writes into records that already exist. Run them "
            "separately."
        )
    # A flag the running pass never reads is a typo, not a no-op: silently
    # running a pass that ignores it answers a question the user did not ask.
    if args.model and not (args.ai or args.polish_meanings):
        raise JankiError(
            "enrich --model applies to the passes that call a model. The --jpdb "
            "pass reads a dictionary, so it has no model to choose."
        )
    if args.force and not args.ai:
        raise JankiError(
            "enrich --force overwrites the staging file a large --ai run writes. "
            f"{passes[0]} writes no staging file."
        )

    config = _load_config(args)

    # --polish-meanings rewrites English and asks jpdb nothing, so it must not
    # require a key to run. The other paths do: --jpdb enriches *from* jpdb and
    # --ai verifies example furigana *with* it.
    if args.polish_meanings:
        return _polish_meanings(config, args)

    # Neither of these asks jpdb anything — one writes a ledger entry, the
    # other builds requests — so neither may demand a key to run.
    if args.batch_forget:
        return _batch_forget(config)

    if args.batch_submit:
        return _batch_submit(config, args, force_fields)

    client = jpdb.JpdbClient(jpdb.api_key_from_env())

    if args.staging is not None:
        return _enrich_staging(client, args.staging.resolve(), args.yes)

    if args.batch_fetch:
        return _batch_fetch(config, args, force_fields)

    if args.ai:
        return _enrich_ai(config, args, force_fields)

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
        _report_enrichment_ledger_failure(
            ledger_error,
            rerun=(
                "Nor does a re-run: what jpdb can fill is filled, so the next "
                "pass either skips these records or looks them up and proposes "
                "nothing — it never reaches the ledger either way."
            ),
            aftermath=(
                "The records are correct; 'status' will simply not know jpdb is "
                "what filled them."
            ),
        )
        return 1
    return 0


def _enrich_ai(
    config: ProjectConfig, args: argparse.Namespace, force_fields: Sequence[str]
) -> int:
    """Write examples and usage notes, by diff for a few records or by staging file.

    Past a certain size a y/n diff stops being review — nobody reads five
    hundred proposed sentences in a terminal and means it — so a large run
    writes a staging file and goes through ``janki promote`` instead, which is
    the same route a photographed handout takes and for the same reason.
    """
    output_path = config.normalized_file.resolve()
    records = load_records(output_path) if output_path.exists() else []
    if not records:
        print(f"No records to enrich in {output_path}.")
        return 0

    model = args.model or config.enrich_model
    style_guide = claude_client.read_style_guide(config.root)
    targets = enrich.ai_targets(records, args.ids or None)
    if not targets:
        print("Nothing to write: every record already has examples and usage notes.")
        return 0

    staging = len(targets) >= enrich.STAGING_THRESHOLD
    staging_target = config.staging_dir / STAGING_FILE_NAME
    if staging and staging_target.exists() and not args.force:
        # Before the pass, not after it: a run this size is one API call per
        # record, and finding the file at write time throws all of them away.
        # The same reasoning as `check_rewritable` on the --staging path.
        raise StagingError(
            f"{staging_target} already exists and would be overwritten by these "
            f"{len(targets)} record(s). Promote or move it first, or pass --force."
        )
    book = ledger.load(config.ledger_file)
    result = enrich.enrich_ai(
        records,
        model=model,
        style_guide=style_guide,
        force_fields=force_fields,
        ids=args.ids or None,
        jpdb_client=jpdb.JpdbClient(jpdb.api_key_from_env()),
    )

    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if not result.changes:
        print(
            f"Nothing written: looked at {result.looked_up} record(s), and none "
            "of the answers had anything to fill."
        )
        return 0

    return _write_ai_result(
        result,
        records,
        book,
        output_path=output_path,
        staging_target=staging_target if staging else None,
        model=model,
        force=args.force,
        assume_yes=args.yes,
    )


def _write_ai_result(
    result: enrich.AiResult,
    records: Sequence[VocabularyRecord],
    book: ledger.Ledger,
    *,
    output_path: Path,
    staging_target: Path | None,
    model: str,
    force: bool,
    assume_yes: bool,
) -> int:
    """Land an AI pass's proposals: staging file for a large one, diff for a small.

    Shared with ``--batch-fetch`` (M4.4), which produces the same kind of result
    hours later and has the same reason to route a large one through
    ``janki promote``: a batch is what janki reaches for at a thousand records,
    which is not a number of sentences anyone reviews in a terminal.
    """
    if staging_target is not None:
        staging_target.parent.mkdir(parents=True, exist_ok=True)
        written = [
            result.records[index]
            for index, record in enumerate(records)
            if record.id in result.changes
        ]
        write_staging(
            staging_target,
            written,
            {
                "source_file": str(output_path.name),
                "extracted_at": date.today().isoformat(),
                "model": model,
                "review_notes": (
                    f"{len(result.changes)} record(s) enriched by {model}. These "
                    "records already exist; promoting merges the new fields into "
                    "them. Sentences whose furigana jpdb did not confirm are "
                    f"flagged with '{enrich.UNVERIFIED_KEY}' — check those before "
                    "audio is generated for them."
                ),
            },
            force=force,
        )
        print(
            f"{len(result.changes)} record(s) is too many to review as one diff, "
            f"so they went to {staging_target}."
        )
        print("  Review it, then: janki promote " + str(staging_target))
        return 0

    for line in enrich.format_field_diff(result.changes):
        print(line)
    if not _confirm_enrich(len(result.changes), assume_yes):
        print("Aborted: nothing was written.", file=sys.stderr)
        return 1

    save_records_json(output_path, result.records)
    for record_id, changed in result.changes.items():
        book.record_enriched(record_id, kind="ai", model=model, fields=changed)
    ledger_error = _save_ledger(book)

    print(f"Enriched {len(result.changes)} record(s) in {output_path}.")
    if result.unverified:
        print(
            f"  {len(result.unverified)} record(s) carry an example whose furigana "
            "jpdb did not confirm; they are flagged in the record."
        )
    if ledger_error is None:
        print(f"Ledger: recorded an AI pass over {len(result.changes)} record(s).")
    else:
        _report_enrichment_ledger_failure(
            ledger_error,
            rerun=(
                "A re-run is not a free repair either: a record whose examples "
                "landed without a usage note is still a target, so the next pass "
                "would call the model for it again and record a pass only where "
                "it writes something."
            ),
            aftermath=(
                "The examples and notes are correct; 'status' will simply not "
                f"know {model} wrote them."
            ),
        )
        return 1
    return 0


def _batch_submit(
    config: ProjectConfig, args: argparse.Namespace, force_fields: Sequence[str]
) -> int:
    """Send the AI pass as one batch and remember it in the ledger.

    Half price, answered within a day rather than within seconds. The id is
    written down before anything else can go wrong with the run, because a
    submitted batch nobody kept the id of is work that was paid for and cannot
    be collected.
    """
    output_path = config.normalized_file.resolve()
    records = load_records(output_path) if output_path.exists() else []
    if not records:
        print(f"No records to enrich in {output_path}.")
        return 0

    book = ledger.load(config.ledger_file)
    if (pending := book.pending_batch()) is not None:
        batch_id, entry = pending
        raise JankiError(
            f"Batch {batch_id} is still out, covering {len(entry.get('pending_ids', []))} "
            f"record(s) since {entry.get('submitted_at', 'an unknown date')}. Fetch it "
            "first: janki enrich --ai --batch-fetch. Two batches at once would leave "
            "two answers for the same word and no way to say which is current."
        )

    model = args.model or config.enrich_model
    style_guide = claude_client.read_style_guide(config.root)
    requests, pending_ids = enrich.batch_requests(
        records, model=model, style_guide=style_guide, ids=args.ids or None
    )
    if not requests:
        print("Nothing to submit: every record already has examples and usage notes.")
        return 0

    batch_id = claude_client.submit_batch(requests)
    book.record_batch(
        batch_id,
        kind="ai",
        model=model,
        pending_ids=pending_ids,
        force_fields=force_fields,
    )
    ledger_error = _save_ledger(book)
    if ledger_error is not None:
        print(f"warning: {ledger_error}", file=sys.stderr)
        print(
            f"Batch {batch_id} was submitted and is being processed, but the "
            "ledger entry that would let janki fetch it was not written. Write "
            "that id down now — with it, 'janki enrich --ai --batch-fetch' works "
            "once the ledger is writable and you have re-recorded it; without it, "
            "the results are only reachable from the Anthropic console.",
            file=sys.stderr,
        )
        return 1

    print(f"Submitted {len(requests)} record(s) as batch {batch_id}.")
    print("  Most batches finish within an hour; the limit is 24.")
    print("  Collect it with: janki enrich --ai --batch-fetch")
    return 0


def _batch_forget(config: ProjectConfig) -> int:
    """Drop the pending batch entry without collecting it.

    The only supported way out of a batch that can never be applied — the
    project it was submitted against is gone, or its records were re-minted
    while it was out. Without this the remedy would be editing
    ``data/ledger.json`` by hand, which AGENTS.md forbids and which is a bad
    habit to teach for a file janki writes.

    Nothing is destroyed: the results stay on Anthropic's side for weeks, and
    the id is printed on the way out so a console lookup is still possible.
    """
    book = ledger.load(config.ledger_file)
    pending = book.pending_batch()
    if pending is None:
        print("No batch is pending; nothing to forget.")
        return 0
    batch_id, entry = pending
    book.clear_batch(batch_id)
    if (ledger_error := _save_ledger(book)) is not None:
        print(f"warning: {ledger_error}", file=sys.stderr)
        print(
            f"Batch {batch_id} is still recorded as pending on disk — only the "
            "in-memory copy was cleared — so --batch-submit will keep refusing. "
            "Run --batch-forget again once that file is writable; that is the "
            "whole remedy, and it is why this command exists rather than an "
            "edit to data/ledger.json.",
            file=sys.stderr,
        )
        return 1
    print(
        f"Forgot batch {batch_id} ({len(entry.get('pending_ids', []))} record(s), "
        f"submitted {entry.get('submitted_at', 'on an unknown date')}). Its "
        "results are still on Anthropic's side and reachable from the console "
        "under that id; janki will not look for them again."
    )
    return 0


def _batch_fetch(
    config: ProjectConfig, args: argparse.Namespace, force_fields: Sequence[str]
) -> int:
    """Collect a finished batch, or say how far along it is.

    One poll, never a wait loop: the point of batching is that nobody is sitting
    here. A batch still running prints its status and exits 0, which is a
    successful answer to "is it ready" — the question this command asks.
    """
    output_path = config.normalized_file.resolve()
    records = load_records(output_path) if output_path.exists() else []
    book = ledger.load(config.ledger_file)
    pending = book.pending_batch()
    if pending is not None and not records:
        # An empty collection is a --root pointed at the wrong project or a
        # normalized file that went missing, not a batch that failed — so the
        # batch stays pending rather than being cleared against nothing.
        #
        # One of a pair. Its sibling below catches the same problem in the
        # other shape: a file that has records, none of them this batch's. That
        # one waits for the batch's *status*, so that a run which has nothing to
        # collect yet still says so rather than erroring; this one is checked
        # first, before even that call. `--batch-forget` is the way out of both.
        raise JankiError(
            f"Batch {pending[0]} is pending, but there are no records in "
            f"{output_path} to apply it to. Point --root at the right project, "
            "or restore the file, and fetch again — fetching costs nothing. If "
            "the collection is gone for good, 'janki enrich --ai --batch-forget' "
            "drops the entry."
        )
    if pending is None:
        print("No batch is pending. Submit one with: janki enrich --ai --batch-submit")
        return 0
    batch_id, entry = pending
    pending_ids = [str(item) for item in entry.get("pending_ids", [])]
    # Both from the entry, not from this invocation: the batch was submitted
    # under one model and one set of overwritable fields, and it was answered
    # under those. Reading them off a config that has moved since, or off flags
    # this command refuses to take, would apply the answer to a different
    # question than the one that was asked.
    model = str(entry.get("model") or config.enrich_model)
    submitted_fields = [str(item) for item in entry.get("force_fields", [])]
    # A held batch's second fetch has one job: the rows whose answers did not
    # parse. Everything else is settled, whatever route it took — records,
    # staging file, or nothing at all — which is why this is the list of
    # failures rather than of successes.
    retry = [str(item) for item in entry.get("retry_ids", [])]
    candidates = retry or pending_ids

    status_now = claude_client.batch_status(batch_id)
    if status_now != claude_client.BATCH_ENDED:
        print(
            f"Batch {batch_id} is {status_now or 'in an unreported state'} "
            f"({len(pending_ids)} record(s), submitted "
            f"{entry.get('submitted_at', 'on an unknown date')}). Nothing to "
            "collect yet."
        )
        return 0

    # Whether this batch can land at all, asked of the collection and asked
    # before a single result is read. Two things follow from the placement. It
    # cannot be defeated by what the API said about individual rows — the
    # classification of a row depends on its outcome as well as on presence, so
    # counting the missing would let the mix of outcomes decide the guard. And
    # streaming the results first would schema-validate every succeeded row on
    # the way to a verdict that needed none of them, so a malformed row in a
    # batch whose records are all gone would report a schema problem instead of
    # the moved collection that is actually wrong.
    present = {record.id for record in records}
    if pending_ids and not any(record_id in present for record_id in pending_ids):
        # A --replace import, a promote that re-minted ids after a reading fix,
        # a restore from another revision. Clearing here would drop the id of a
        # batch whose answers are alive on Anthropic's side for weeks, and exit
        # 0 doing it. Nothing is lost by refusing: the entry survives, and
        # fetching again after the file is restored collects whatever the batch
        # returned and reports each row by name.
        raise JankiError(
            f"Batch {batch_id} came back, but none of the {len(pending_ids)} "
            f"record(s) it covers are in {output_path} any more, so nothing can "
            "be applied. The batch is left pending: point --root at the right "
            "project, or restore the file, and fetch again — fetching costs "
            "nothing. If those records are gone for good, "
            "'janki enrich --ai --batch-forget' drops the entry."
        )

    # Unlike the model and the ids, this one is not settled at submit time: it
    # decides how an answer already in hand is applied, and the records may have
    # gained fields while the batch was out. Overriding is allowed for that
    # reason, announced because it is not what was submitted, and announced
    # *down here* — past the status poll and the guard — because a run that
    # collects nothing applies nothing and should claim nothing. The names
    # themselves were validated by `command_enrich` before any of this, so a
    # misspelt field is caught whatever state the batch turns out to be in.
    if force_fields:
        print(
            f"Applying with --force-fields {', '.join(force_fields)} instead of "
            f"the {', '.join(submitted_fields) or 'none'} this batch was "
            "submitted with."
        )
    else:
        force_fields = submitted_fields

    # The batch's own model, not the config's: a run submitted under one model
    # and fetched after the config changed was still answered by the first.
    outcome = enrich.apply_batch_results(
        records,
        claude_client.batch_results(batch_id, enrich.ai_schema(), model),
        pending_ids,
        model=model,
        force_fields=force_fields,
        only=retry,
        jpdb_client=jpdb.JpdbClient(jpdb.api_key_from_env()),
    )
    result = outcome.result

    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    for record_id, reason in outcome.failed.items():
        # The reason carries its own ending: whether the record is still there
        # to leave untouched is something only the apply pass knows.
        print(f"warning: {record_id}: {reason}.", file=sys.stderr)
    if outcome.settled:
        print(
            f"{len(outcome.settled)} record(s) were settled by an earlier fetch "
            "of this batch and were left as they are."
        )
    for record_id, reason in outcome.invalid.items():
        print(
            f"warning: {record_id}: the answer did not parse ({reason}); left "
            "untouched.",
            file=sys.stderr,
        )
    if outcome.missing:
        print(
            f"warning: {len(outcome.missing)} record(s) the batch covered are no "
            "longer in the collection, so their answers were dropped: "
            f"{', '.join(outcome.missing)}",
            file=sys.stderr,
        )

    staging = len(candidates) >= enrich.STAGING_THRESHOLD
    staging_target = config.staging_dir / STAGING_FILE_NAME
    if staging and staging_target.exists() and not args.force:
        raise StagingError(
            f"{staging_target} already exists and would be overwritten by batch "
            f"{batch_id}. Promote or move it first, then fetch again — the batch "
            "is still recorded as pending, so nothing was lost."
        )

    if not result.changes:
        if outcome.invalid:
            return _keep_for_invalid(book, batch_id, outcome.invalid)
        print(
            f"Batch {batch_id} is collected: looked at {result.looked_up} "
            "record(s), and none of the answers had anything to fill."
        )
        book.clear_batch(batch_id)
        if (ledger_error := _save_ledger(book)) is not None:
            _report_batch_ledger_failure(ledger_error, batch_id, landed="nothing")
            return 1
        return 0

    code = _write_ai_result(
        result,
        records,
        book,
        output_path=output_path,
        staging_target=staging_target if staging else None,
        model=model,
        force=args.force,
        assume_yes=args.yes,
    )
    if code != 0:
        # Declined, or the ledger write failed. Either way the batch stays
        # pending: its results live on Anthropic's side for weeks, and fetching
        # again is free, so forgetting the id here would be the only
        # irreversible part of the run.
        print(
            f"Batch {batch_id} is still recorded as pending; fetching it again "
            "costs nothing.",
            file=sys.stderr,
        )
        return code

    if outcome.invalid:
        # The good rows are written; the unparseable ones are not, and their
        # answers are still there to be had. Clearing now would be the one
        # irreversible thing this command can do, for the one failure class
        # that was janki's and not the API's.
        return _keep_for_invalid(
            book, batch_id, outcome.invalid, wrote=len(result.changes), staged=staging
        )

    book.clear_batch(batch_id)
    if (ledger_error := _save_ledger(book)) is not None:
        _report_batch_ledger_failure(
            ledger_error, batch_id, landed="staging" if staging else "records"
        )
        return 1
    return 0


def _keep_for_invalid(
    book: ledger.Ledger,
    batch_id: str,
    invalid: Mapping[str, str],
    *,
    wrote: int = 0,
    staged: bool = False,
) -> int:
    """Hold the batch id because some answers came back unreadable.

    The one row class a later fetch can still do something about: the answers
    exist, complete and paid for, and it is janki's schema that turned them
    away. Dropping the id over that would make an 800-record batch unreachable
    because one field was added to a Pydantic model.

    Holding it is what makes a second fetch possible, so the entry is narrowed
    to exactly these ids — a later fetch retries them and leaves every other row
    alone, however that row was settled. Recording the failures rather than the
    successes is what keeps that true when the successes went to a staging file.

    Which is the thing the message has to be straight about. A staging file *is*
    the answers, the same way `janki extract`'s is: this batch has delivered
    them and will not offer them again, so deleting that file loses them. Saying
    "promote or discard" would offer the second as a free alternative to the
    first, when it is the one irreversible choice on the table.
    """
    book.record_batch_retry(batch_id, invalid)
    if staged:
        written = (
            f"{wrote} record(s) went to {STAGING_FILE_NAME}. Promote it to land "
            "them — that file is where those answers live now, and this batch "
            "will not offer them again, so deleting it loses them. "
        )
    elif wrote:
        written = f"{wrote} record(s) were written. "
    else:
        written = ""
    print(
        f"{written}{len(invalid)} answer(s) in batch {batch_id} did not parse, "
        f"so it is still recorded as pending, narrowed to those {len(invalid)}. "
        "They are intact on Anthropic's side and fetching again costs nothing, "
        "which is worth doing if the schema they failed was the thing at fault. "
        "If they are not worth chasing, 'janki enrich --ai --batch-forget' "
        "drops the entry.",
        file=sys.stderr,
    )
    if (ledger_error := _save_ledger(book)) is not None:
        print(f"warning: {ledger_error}", file=sys.stderr)
        print(
            "The ledger could not record which rows are left to retry, so a "
            "later fetch of this batch would consider all of them again. Fix "
            "that file before fetching again.",
            file=sys.stderr,
        )
    return 1


def _report_batch_ledger_failure(
    exc: ledger.LedgerError, batch_id: str, *, landed: str
) -> None:
    """Say the batch is still pending, and what fetching it again will do.

    ``landed`` is where the answers went, because that decides what a re-fetch
    actually costs the reader. All three cases end with the batch still
    pending — which is the safe end, since results wait on Anthropic's side for
    weeks — but telling someone a re-fetch is a no-op when it will raise on an
    existing staging file sends them after a repair that fails.
    """
    advice = {
        "records": (
            "The records are written. Fetching again once the ledger is "
            "writable applies nothing new — those fields are filled — and "
            "clears the entry."
        ),
        "staging": (
            f"The answers are in {STAGING_FILE_NAME}, not in the records yet. "
            "Promote that file first: fetching again while it exists refuses "
            "rather than overwrite a review in progress."
        ),
        "nothing": (
            "Nothing was written — no answer had anything to fill — so "
            "fetching again once the ledger is writable simply clears the entry."
        ),
    }[landed]
    print(f"warning: {exc}", file=sys.stderr)
    print(
        f"Batch {batch_id} is still recorded as pending, so --batch-submit will "
        f"refuse until it is cleared. {advice}",
        file=sys.stderr,
    )


def _confirm_polish(assume_yes: bool) -> str:
    """``"yes"``, ``"no"`` or ``"quit"`` for one proposed gloss list.

    ``--yes`` and a non-tty both accept, the same rule every other confirm in
    janki uses: fat-finger protection, not CI protection. Quitting is offered
    because this pass calls the model per record as the loop runs, so walking
    away stops the spending there — and what was accepted before that is still
    written, since it was accepted. Note that the cost is one call per record
    *reached*, which is not the same as per record shown: one whose glosses are
    already right is paid for and passed over without a prompt.
    """
    if assume_yes or not sys.stdin.isatty():
        return "yes"
    try:
        answer = input("Replace these meanings? [y/N/q] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return "quit"
    if answer in {"q", "quit"}:
        return "quit"
    return "yes" if answer in {"y", "yes"} else "no"


def _polish_meanings(config: ProjectConfig, args: argparse.Namespace) -> int:
    """Propose better English glosses, one record at a time.

    The only pass that rewrites a field instead of filling one, which is why it
    is its own flag and why the confirmation is per record rather than one y/n
    over a whole diff: the thing being replaced may have been typed by hand out
    of a textbook, and "these thirty are all fine except the fourth" is not an
    answer a single prompt can take.
    """
    output_path = config.normalized_file.resolve()
    records = load_records(output_path) if output_path.exists() else []
    if not records:
        print(f"No records to enrich in {output_path}.")
        return 0

    model = args.model or config.enrich_model
    style_guide = claude_client.read_style_guide(config.root)
    targets = enrich.polish_targets(records, args.ids or None)
    print(
        f"Polishing {len(targets)} record(s) with {model}, one call each. "
        "Nothing is written until you say so."
    )

    book = ledger.load(config.ledger_file)
    positions = {record.id: index for index, record in enumerate(records)}
    updated = list(records)
    accepted: dict[str, dict[str, tuple[Any, Any]]] = {}
    declined = 0
    unchanged = 0
    stopped = False

    # The prompt is not the only place a run gets interrupted, and it is not the
    # likely one: the call is the slow step, so a Ctrl-C most often lands there.
    # Both have to end the same way, or "what you accepted is written" is true
    # only when the timing is lucky.
    try:
        for outcome in enrich.polish_meanings(
            records, model=model, style_guide=style_guide, ids=args.ids or None
        ):
            if outcome.warning:
                print(f"warning: {outcome.warning}", file=sys.stderr)
                continue
            if outcome.proposed is None:
                unchanged += 1
                continue
            for line in enrich.format_field_diff(
                {outcome.record.id: dict(outcome.changes)}
            ):
                print(line)
            answer = _confirm_polish(args.yes)
            if answer == "quit":
                stopped = True
                break
            if answer == "no":
                declined += 1
                continue
            updated[positions[outcome.record.id]] = outcome.proposed
            accepted[outcome.record.id] = dict(outcome.changes)
    except KeyboardInterrupt:
        print()
        stopped = True

    if stopped:
        print("Stopped; the records after this one were not looked at.")
    if unchanged:
        print(f"{unchanged} record(s) already had the glosses {model} would write.")
    if declined:
        print(f"{declined} proposal(s) declined.")
    if not accepted:
        print("Nothing written.")
        return 0

    save_records_json(output_path, updated)
    for record_id in accepted:
        book.record_enriched(
            record_id, kind="polish", model=model, fields=enrich.POLISH_FIELDS
        )
    ledger_error = _save_ledger(book)

    print(f"Rewrote the meanings of {len(accepted)} record(s) in {output_path}.")
    if ledger_error is not None:
        _report_enrichment_ledger_failure(
            ledger_error,
            rerun=(
                "A re-run is not a free repair either: this pass looks at every "
                "record every time, so it would call the model once per record "
                "again and record a pass only where you accept a further change."
            ),
            aftermath=(
                "The new glosses are on file; 'status' will simply not know "
                f"{model} wrote them in place of what was there."
            ),
        )
        return 1
    print(f"Ledger: recorded a polish pass over {len(accepted)} record(s).")
    return 0


def command_extract(args: argparse.Namespace) -> int:
    """Read vocabulary off PDFs and photos into staging files for review.

    One staging file per input, and nothing anywhere near ``vocabulary.json``:
    this is the one command that guesses, so everything it produces is a
    proposal a human still has to accept. Inputs are copied into the inbox
    first (M3.2), so a candidate can always be checked against the page it came
    from.

    Files are processed one at a time and written as they succeed. A later file
    failing therefore leaves the earlier ones' staging files in place — which is
    the useful direction: the work already paid for is kept, and the error names
    what is left to do.
    """
    config = _load_config(args)
    style_guide = claude_client.read_style_guide(config.root)
    model = args.model or config.extract_model
    prepared = prepare_inputs([path for path in args.files], config.scan_inbox)

    existing = (
        load_records(config.normalized_file)
        if config.normalized_file.exists()
        else []
    )
    known = extract.known_ids(existing)
    # Only prose mode is told what janki already has: a table is transcribed
    # row by row, and telling the model to skip rows would put holes in a
    # faithful transcription.
    skip_list = (
        sorted({record.expression for record in existing})
        if args.mode == "prose"
        else ()
    )

    # Every target resolved before the first API call: a batch that would write
    # two inputs to one file is refused now rather than after paying for both.
    targets = extract.staging_targets(config.staging_dir, prepared)

    written = 0
    for item, target in zip(prepared, targets, strict=True):
        candidates = extract.extract_candidates(
            item,
            model=model,
            style_guide=style_guide,
            mode=args.mode,
            known=skip_list,
        )
        records = extract.build_records(candidates, item, known)
        target.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            # The basename, like every other writer of this key. An absolute
            # path is stale on any other clone, and this file is committed.
            "source_file": item.origin_path.name,
            "extracted_at": date.today().isoformat(),
            "model": model,
        }
        # Held back into the file, not just onto the terminal: the staging file
        # is what a reviewer reads later, and a count that lives only in
        # scrollback is the same silent discard with an extra step.
        held = extract.unusable(candidates)
        if held:
            meta["review_notes"] = extract.unusable_note(held)
        write_staging(target, records, meta, force=args.force)
        written += 1
        already = sum(
            1 for record in records if "already_known" in record.source.raw_fields
        )
        note = f", {len(held)} unusable" if held else ""
        print(
            f"{item.origin_path.name}: {len(records)} candidate(s) "
            f"({already} already known{note}) -> {target}"
        )

    if not written:
        print("Nothing to extract.")
        return 0
    print(
        f"Wrote {written} staging file(s). Review them, then 'janki validate "
        "<file>' — nothing reaches vocabulary.json until you promote it."
    )
    return 0


def _inside_archive(path: Path, archive_dir: Path) -> bool:
    """Is ``path`` the promoted archive, or inside it?

    Identity where the filesystem can answer it, because a lexical comparison
    is wrong on a case-insensitive filesystem: ``staging/Done/lesson.yaml``
    opens the real archive while comparing unequal to ``staging/done/...``, and
    promoting it doubles a committed file that is the only copy of a finished
    review. The lexical test stays as the fallback for a path that does not
    exist yet.
    """
    try:
        if archive_dir.exists() and path.parent.samefile(archive_dir):
            return True
    except OSError:
        pass
    return path.is_relative_to(archive_dir)


def _provider_name(config: ProjectConfig, chosen: str | None) -> str:
    """The engine this run speaks words through, normalised once.

    One definition, because two normalisations disagree: ``_speech_provider``
    lower-cased and defaulted while ``_sentence_provider`` compared the raw
    config string, so ``provider = "Voicevox"`` — or an empty one, or
    ``--provider voicevox`` overriding an ``azure`` file — built a VOICEVOX word
    provider and then silently discarded the configured sentence voice.
    """
    return (chosen or config.tts_provider or "voicevox").strip().lower()


def _sentence_provider(config: ProjectConfig, chosen: str | None, words: Any) -> Any:
    """The provider that reads example sentences.

    ``words`` when nothing else is configured, so the default is one voice
    throughout and the ledger keeps recording what it always did. A separate
    sentence voice is worth having because the two recordings do different
    jobs: a word is a thing to identify, a sentence is a thing to follow, and
    hearing them in one voice makes the sentence sound like a longer word.
    """
    name = (config.sentence_provider or "").strip().lower()
    if name == "openai":
        return openai_tts.OpenAiSpeechProvider(
            voice=config.openai_voice,
            model=config.openai_model,
            instructions=config.openai_instructions,
            speed=config.voicevox_speed,
        )
    if name not in {"", "voicevox"}:
        raise AudioError(
            f"Unknown [tts] sentence_provider {name!r}. Known: voicevox, openai, "
            "or leave it empty to read sentences in the same voice as the words."
        )
    if _provider_name(config, chosen) == "voicevox":
        speaker = config.voicevox_sentence_speaker
        # `is not None`, not truthiness: 0 is a real style id.
        if speaker is not None and speaker != config.voicevox_speaker:
            return voicevox.VoicevoxProvider(
                base_url=config.voicevox_url,
                speaker=speaker,
                speed=config.voicevox_speed,
            )
    return words


def _speech_provider(config: ProjectConfig, chosen: str | None) -> Any:
    """The provider this run speaks *words* through.

    Only VOICEVOX can force a pitch accent, which is what a word clip is for,
    so this stays VOICEVOX. ``azure`` is still refused by name rather than
    falling through to "unknown": it was a real plan, it was evaluated and
    dropped, and a user who wrote it in their config deserves to hear which of
    those happened. Sentences are a separate choice — see
    :func:`_sentence_provider`.
    """
    name = _provider_name(config, chosen)
    if name == "voicevox":
        return voicevox.VoicevoxProvider(
            base_url=config.voicevox_url,
            speaker=config.voicevox_speaker,
            speed=config.voicevox_speed,
        )
    if name == "azure":
        raise AudioError(
            "The Azure provider was evaluated and dropped — VOICEVOX reads the "
            "ambiguous kanji correctly and needs no account (see M5.7 in "
            "docs/IMPLEMENTATION_PLAN.md). Words use VOICEVOX, which is the "
            "only engine here that can force a pitch accent; for sentences set "
            "[tts] sentence_provider = \"openai\"."
        )
    raise AudioError(
        f"Unknown TTS provider {name!r}. Words are voiced by voicevox. For "
        "sentences, set [tts] sentence_provider to voicevox or openai."
    )


def command_audio(args: argparse.Namespace) -> int:
    """Generate the audio a card plays.

    Refuses before spending anything when the engine is not answering: a run
    that synthesizes forty clips and then fails on the forty-first has written
    forty files and half a ledger, and the common reason is simply that the
    engine is not running.
    """
    config = _load_config(args)
    output_path = config.normalized_file.resolve()
    records = load_records(output_path) if output_path.exists() else []
    if not records:
        print(f"No records to voice in {output_path}.")
        return 0

    provider = _speech_provider(config, args.provider)
    # Only what this run will actually use. A `--words` run never reaches the
    # sentence provider, and refusing to start because *that* engine lacks a
    # key would abort a job it plays no part in.
    sentences = _sentence_provider(config, args.provider, provider) if args.examples else provider
    # Keyed by identity so one engine doing both jobs is checked once.
    engines: dict[int, Any] = {}
    if args.words:
        engines[id(provider)] = provider
    if args.examples:
        engines[id(sentences)] = sentences
    for engine in engines.values():
        if not engine.available():
            raise AudioError(f"{engine.name}: {engine.launch_hint}")

    book = ledger.load(config.ledger_file)
    media_dir = config.media_dir.resolve()
    result = audio_cmd.generate_audio(
        records,
        provider=provider,
        sentence_provider=sentences,
        book=book,
        media_dir=media_dir,
        # Passed straight through: `generate_audio` refuses when neither is
        # asked for, and a default here would make that refusal unreachable and
        # quietly voice every record in the collection.
        words=args.words,
        examples=args.examples,
        ids=args.ids or None,
        force=args.force,
        allow_default_accent=args.allow_default_accent,
    )

    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if result.no_pattern:
        print(
            f"warning: {len(result.no_pattern)} record(s) have no accent pattern "
            "and were skipped rather than voiced with a guessed one — "
            "'janki enrich --jpdb' fills it, or --allow-default-accent opts into "
            f"the guess: {', '.join(result.no_pattern[:5])}"
            + (" ..." if len(result.no_pattern) > 5 else ""),
            file=sys.stderr,
        )
    if result.unverified:
        print(
            f"warning: {len(result.unverified)} example(s) carry furigana jpdb "
            "never confirmed and were left unvoiced; check them first.",
            file=sys.stderr,
        )
    if result.no_reading:
        print(
            f"warning: {len(result.no_reading)} record(s) have no reading to "
            f"speak: {', '.join(result.no_reading[:5])}"
            + (" ..." if len(result.no_reading) > 5 else ""),
            file=sys.stderr,
        )

    if result.file_count:
        save_records_json(output_path, result.records)
    ledger_error = _save_ledger(book)

    print(
        f"Wrote {result.file_count} clip(s) for {len(result.written)} record(s) "
        f"into {config.media_dir.resolve() / audio_cmd.AUDIO_SUBDIR}."
        + (f" {result.up_to_date} already current." if result.up_to_date else "")
    )
    if args.prune:
        removed = audio_cmd.prune_unreferenced(result.records, media_dir, book)
        print(f"Pruned {len(removed)} unreferenced clip(s).")
        if removed:
            ledger_error = _save_ledger(book) or ledger_error
    # Both are reported, and the run's own failure first: a ledger warning on
    # its own reads as a successful partial run, and the records this never
    # reached would be invisible.
    if result.stopped_by:
        print(f"error: {result.stopped_by}", file=sys.stderr)
        print(
            "The clips written before this are saved; re-running picks up where "
            "it stopped.",
            file=sys.stderr,
        )
    if ledger_error is not None:
        _report_ledger_failure(ledger_error)
    return 1 if (result.stopped_by or ledger_error is not None) else 0


def command_promote(args: argparse.Namespace) -> int:
    """Move a reviewed staging file's records into the normalized collection.

    Ordering is load-bearing in the same way ``run_import``'s is, for the same
    reason: everything that can refuse — an unreadable ledger, a staging file
    that cannot be pruned — happens before ``vocabulary.json`` is rewritten, and
    the records are written before the ledger, because the ledger is metadata
    ``status --rebuild`` can reconstruct and the records are not.

    The staging file is only ever pruned of rows that actually landed, so a
    promote that fails part-way leaves the review intact and re-running it is
    safe.
    """
    config = _load_config(args)
    path = args.file.resolve()
    done = (config.staging_dir / "done" / path.name).resolve()
    if _inside_archive(path, done.parent):
        raise PromoteError(
            f"{path} is inside the promoted archive. Those records are already in "
            "the collection; promoting the archive would only duplicate it."
        )

    records, meta = read_staging(path)
    if not records:
        print(f"{path} holds no records; nothing to promote.")
        return 0

    # Everything that can refuse, before anything is written. read_staging goes
    # through PyYAML, which accepts a duplicate key silently; the rewrite goes
    # through ruamel, which does not. Finding that out *after* the records and
    # the archive were written leaves the promoted rows still in the staging
    # file, and the re-run then appends them to the archive a second time.
    check_rewritable(path)
    # The archive is written under the same name, and write_staging refuses a
    # suffix read_staging could not parse back. read_staging accepts .json and
    # JSON is valid YAML, so a hand-made .json staging file gets all the way to
    # the archive write before failing — after the records and the ledger have
    # landed, leaving a review that can never be finished however often it is
    # retried.
    if path.suffix.lower() not in STAGING_SUFFIXES:
        raise PromoteError(
            f"{path} is not a staging file janki can rewrite: the promoted archive "
            f"is written under the same name, and that needs "
            f"{' or '.join(STAGING_SUFFIXES)}. Rename it and re-run."
        )
    archived: list[VocabularyRecord] = []
    if done.exists():
        previous, _previous_meta = read_staging(done)
        archived = list(previous)

    client = None
    if not args.skip_reading_check:
        client = jpdb.JpdbClient(jpdb.api_key_from_env())
    # Read before the ids are decided, not after: a staged id the collection
    # already holds must keep it, or the re-mint adds a second record beside
    # the curated one and leaves the original untouched.
    #
    # "The collection" here means what it means everywhere else in janki — the
    # normalized file *plus every deck's inline notes* (`status.surviving_ids`).
    # A record living only in a deck YAML has the same stale-id problem and the
    # same exported GUID, and a narrower set would re-mint it just as happily.
    # A deck that will not resolve leaves ids unknown, so nothing can be proved
    # absent. Rows whose id would change are then held back rather than
    # promoted: writing the id they arrived with would put it in the store
    # permanently, since a stored id is exempt from the re-mint that repairs it.
    # Held rows stay in data/staging/, which is committed.
    output_path = config.normalized_file.resolve()
    existing = load_records(output_path) if output_path.exists() else []
    stored_ids, unreadable = status.surviving_ids(config, existing)
    for problem in unreadable:
        print(
            f"warning: {problem}; ids in that deck cannot be checked, so any "
            "row needing a new id stays in the staging file until it parses.",
            file=sys.stderr,
        )
    result = promote.check_readings(
        records,
        client=client,
        skip_reading_check=args.skip_reading_check,
        already_stored=stored_ids,
        remint_blocked=bool(unreadable),
    )

    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)

    if not result.promoted:
        # The reasons still go in: a row held for a reading no dictionary lists
        # is something only promote can determine, and `status --staged` reads
        # it off the file rather than from this run's scrollback.
        rewrite_staging(path, result.held)
        print(
            f"Nothing promoted: all {len(result.held)} row(s) are still held back, "
            f"and {path} now records why."
        )
        return 0

    # `existing` and `output_path` were read above, before any id was decided.
    # Read the ledger before anything is written, and only once.
    book = ledger.load(config.ledger_file)

    merged, outcomes = merge_records(existing, result.promoted, ())
    save_records_json(output_path, merged)

    added = sum(
        book.record_added(record_id)
        for record_id, outcome in outcomes.items()
        if outcome.label == "added"
    )
    seen = sum(
        book.record_source_seen(record_id, source_type, source_ref)
        for record_id, source_type, source_ref in promote.source_references(
            result.promoted
        )
    )
    ledger_error = _save_ledger(book)

    # The archive is appended to, not replaced: promoting a file in two passes
    # must not lose the first pass's rows.
    done.parent.mkdir(parents=True, exist_ok=True)
    archived = archived + list(result.promoted)
    write_staging(done, archived, promote.archive_meta(meta, len(archived)), force=True)

    removed = prune_staging(path, result.keep)
    if result.held:
        # Rewritten in place with the reason each surviving row is still held,
        # so a partly-promoted file always says what still needs attention.
        rewrite_staging(path, result.held)
    if not result.held:
        # An emptied review is finished work; leaving it would have the next
        # import report a file that can never be resolved.
        path.unlink()

    print(f"Promoted {len(result.promoted)} record(s) from {path} into {output_path}")
    _print_merge_summary(outcomes, prefer_incoming_available=False)
    if result.reminted:
        print("Re-minted malformed IDs (these records were never in Anki):")
        for old, new in sorted(result.reminted.items()):
            print(f"  {old} -> {new}")
    if result.held:
        print(f"  {len(result.held)} row(s) still held back; {path} keeps them.")
    else:
        print(f"  {path} is finished and was deleted ({removed} row(s) promoted).")
    print(f"  Archived to {done}")
    print(_ledger_line(added, seen, written=ledger_error is None))
    if ledger_error is not None:
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


def resolve_deck_path(deck: Path, config: ProjectConfig) -> Path:
    """A deck argument as a path, or as a bare name under ``deck_dir``.

    `janki build verbs` is what a person types; requiring
    `data/decks/verbs.yaml` every time is friction with no safety in it, since
    the name has to match a file either way. An argument that exists as written
    wins, so a deck file in the working directory is never shadowed by a
    same-named one under ``deck_dir``.
    """
    if deck.exists():
        return deck.resolve()
    if deck.parent == Path("") or deck.parent == Path("."):
        for suffix in ("", ".yaml", ".yml"):
            candidate = config.deck_dir / f"{deck.name}{suffix}"
            if candidate.exists():
                return candidate.resolve()
    raise AnkiBuildError(
        f"No such deck: {deck}. Give a path to a deck file, or the bare name "
        f"of one under {config.deck_dir}."
    )


#: What `--only-new` looks for in the records it is about to ship, and what to
#: call each gap. Every one of these is a card that behaves differently from its
#: neighbours for a reason invisible on the card itself, which is why they are
#: reported by count before the build rather than discovered during study.
_BUILD_GAPS: tuple[tuple[str, str], ...] = (
    ("no word audio", "audio"),
    ("no example sentence", "examples"),
    ("no pitch accent", "accent"),
)


def _record_gaps(record: VocabularyRecord) -> tuple[str, ...]:
    """What this record is missing, by the names the ledger stores.

    One definition, used both to warn before a build and to record what the
    build shipped without — so "it went out silent" and "it has a clip now"
    are answered against the same question rather than two similar ones.
    """
    gaps = []
    if not record.audio:
        gaps.append("audio")
    if not any(example.japanese.strip() for example in record.examples):
        gaps.append("examples")
    if not record.pitch_accent and not record.audio_accent.strip():
        gaps.append("accent")
    return tuple(gaps)


def _gap_counts(records: Sequence[VocabularyRecord]) -> dict[str, int]:
    counts = {"audio": 0, "examples": 0, "accent": 0}
    for record in records:
        for gap in _record_gaps(record):
            counts[gap] += 1
    return counts


def _confirm_gaps(
    records: Sequence[VocabularyRecord], assume_yes: bool, *, stem: str
) -> bool:
    """Report what the new records are missing, and ask before shipping them."""
    counts = _gap_counts(records)
    described = [
        f"{counts[key]} of {len(records)} new records have {label}"
        for label, key in _BUILD_GAPS
        if counts[key]
    ]
    if not described:
        return True
    for line in described:
        print(f"warning: {stem}: {line}", file=sys.stderr)
    if assume_yes or not sys.stdin.isatty():
        # Unattended runs proceed: `janki refresh` drives this, and a pipeline
        # that stops for an unanswerable question is worse than one that ships
        # a card missing its audio and says so.
        return True
    try:
        answer = input("Build them anyway? [y/N] ")
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer.strip().lower() in {"y", "yes"}


def _build_one(
    deck_path: Path,
    config: ProjectConfig,
    output: Path | None = None,
    *,
    book: ledger.Ledger | None = None,
    only_new: bool = False,
    assume_yes: bool = False,
) -> bool:
    """Build one deck. Returns whether any export entry is now pending.

    Not "was a package written": the two diverge for a `--output` build, which
    writes a real package and deliberately records nothing, and the caller uses
    this to decide whether a failed ledger save has anything to apologise for.

    ``book`` is the caller's ledger — loaded once per command and saved once,
    the rule this module follows everywhere. Exports are recorded for a plain
    build too, not only `--only-new`: `unexported` is what makes the next
    `--only-new` correct, and a full build that shipped a record without
    saying so would make that record look new forever.
    """
    stem = deck_path.stem
    include_ids: set[str] | None = None
    recorded = False
    _, records = resolve_deck_records(deck_path)
    if only_new:
        # Validated before the "nothing new" shortcut, not after it. A deck
        # whose already-shipped records are broken is a broken deck, and an
        # incremental build that exits 0 on one a full build refuses would hide
        # that until the next full build.
        issues = validate_records(records, deck_path)
        if has_errors(issues):
            formatted = "\n".join(issue.format() for issue in issues)
            raise AnkiBuildError(f"Deck validation failed:\n{formatted}")

        ids = [record.id for record in records]
        new_ids = book.unexported(stem, ids) if book else []
        if book is not None:
            behind = set(book.exported_before_their_work(stem, ids))
            behind |= set(book.shipped_incomplete(stem, records))
            if behind:
                print(
                    f"warning: {stem}: {len(behind)} record(s) shipped without "
                    "audio, examples or accent and have it now, or changed "
                    "after this deck last built them. --only-new cannot see "
                    f"them; run 'janki build {stem}' to rebuild the whole deck.",
                    file=sys.stderr,
                )
        if not new_ids:
            print(f"{stem}: nothing new to build ({len(records)} record(s) already exported)")
            return False
        include_ids = set(new_ids)
        included = [record for record in records if record.id in include_ids]
        if not _confirm_gaps(included, assume_yes, stem=stem):
            print(f"{stem}: not built.")
            return False

    result = build_deck(deck_path, config, output, include_ids=include_ids)
    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    cards = ", ".join(result.card_types)
    scope = " (new only)" if only_new else ""
    plural = "" if result.note_count == 1 else "s"
    print(
        f"Built {result.output_path}{scope} — {result.note_count} note{plural}, "
        f"cards: {cards}, media: {result.media_count}"
    )
    if book is None:
        return True
    if output is not None:
        # Only a build to the deck's *own* declared package records exports. A
        # `--output` build is a throwaway — a package to eyeball, or `make
        # gates` proving the exporter still runs — and claiming those records
        # shipped consumes their new-ness: the next `--only-new` skips them and
        # they never reach a card, with nothing to say so. Said out loud,
        # because a `--only-new --output` run that quietly records nothing
        # rebuilds the identical package every day and never says why.
        print(
            f"not recorded as exported: --output builds are throwaways. "
            f"Run 'janki build {stem}' to build the deck's own package and "
            "record it.",
            file=sys.stderr,
        )
        return recorded
    by_id = {record.id: record for record in records}
    for record_id in result.record_ids:
        shipped = by_id.get(record_id)
        gaps = _record_gaps(shipped) if shipped is not None else ()
        recorded = book.record_export(record_id, stem, gaps=gaps) or recorded
    return recorded


def _finish_build(book: ledger.Ledger, built: bool) -> int:
    """Save the export entries, reporting a failure without losing the build.

    ``built`` says whether any package reached disk. A run that built nothing —
    "nothing new to build", or a declined prompt — has no export entries
    pending, so a failing save there must not announce packages that do not
    exist and records that will be revisited.
    """
    error = _save_ledger(book)
    if error is None:
        return 0
    _report_ledger_failure(error)
    if built:
        print(
            "The package(s) above were written; only the export history was "
            "not. The next '--only-new' will include those records again.",
            file=sys.stderr,
        )
    return 1


def command_build(args: argparse.Namespace) -> int:
    config = _load_config(args)
    # Loaded before anything is built, so a ledger janki cannot read refuses
    # the command rather than surfacing after a package is already on disk.
    book = ledger.load(config.ledger_file)
    if args.all:
        if args.deck:
            raise AnkiBuildError("Do not provide a deck path together with --all")
        if args.output:
            raise AnkiBuildError(
                "--output names one file; it cannot be combined with --all. "
                "Each deck's own 'output:' key names its file."
            )
        deck_paths = sorted(
            [*config.deck_dir.glob("*.yaml"), *config.deck_dir.glob("*.yml")]
        )
        if not deck_paths:
            raise AnkiBuildError(f"No deck files found under {config.deck_dir}")
        built = False
        try:
            for deck_path in deck_paths:
                built = _build_one(
                    deck_path, config, book=book,
                    only_new=args.only_new, assume_yes=args.yes,
                ) or built
        finally:
            # In a `finally` because a later deck refusing must not discard
            # what earlier decks already recorded in memory. Export state is
            # reconstructible by nothing, so a lost entry means those records
            # ship again on every future --only-new while `janki status` keeps
            # calling them unexported.
            _finish_build(book, built)
        return 0

    if not args.deck:
        raise AnkiBuildError("Provide a deck YAML path or use --all")
    output = args.output.resolve() if args.output else None
    built = _build_one(
        resolve_deck_path(args.deck, config), config, output,
        book=book, only_new=args.only_new, assume_yes=args.yes,
    )
    return _finish_build(book, built)


#: The refresh pipeline, in order. Each entry is the stage's own command line,
#: parsed by the real parser rather than assembled as a Namespace — a stage
#: that grows a flag then keeps its default here instead of raising
#: AttributeError halfway through a run.
_REFRESH_STAGES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("jpdb", "--no-jpdb", ("enrich", "--jpdb")),
    ("ai", "--no-ai", ("enrich", "--ai")),
    ("audio", "--no-audio", ("audio", "--words", "--examples")),
    ("build", "--no-build", ("build",)),
)


def command_refresh(args: argparse.Namespace) -> int:
    """Run the whole pipeline in order: enrich, voice, build what is new.

    The stages are the four commands a person runs by hand after adding words,
    in the only order that works — jpdb fills the readings and accents that
    `audio` needs to force a pitch, `--ai` writes the examples that `audio`
    then voices, and `--only-new` ships what the earlier stages just finished.
    Running them out of order silently produces less: a build before `audio`
    ships cards with no sound and marks them exported, so the next `--only-new`
    will not revisit them.

    Each stage is the real command, called in-process — not a subprocess, so a
    failure is a `JankiError` with a stack rather than an exit code, and not a
    reimplementation, so there is one definition of what "enrich --jpdb" means.
    A stage that fails stops the run: every later stage depends on what the
    failed one was supposed to produce, and the alternative is a package built
    from half-enriched records.
    """
    parser = build_parser()
    root = ["--root", str(args.root)] if args.root else []
    deck = [str(args.deck)] if args.deck else ["--all"]

    ran: list[str] = []
    for name, flag, command in _REFRESH_STAGES:
        if getattr(args, f"no_{name}"):
            print(f"— {name}: skipped ({flag})")
            continue
        argv = [*root, *command]
        if name == "build":
            # No `--yes`: refresh is an interactive command — the enrich stages
            # ahead of it prompt too — and the gap prompt exists precisely
            # because shipping bare cards *and marking them exported* hides
            # them from every later `--only-new`. A non-TTY still proceeds, so
            # scripted runs are unaffected.
            argv += [*deck, "--only-new"]
        stage = parser.parse_args(argv)
        print(f"— {name}: janki {' '.join(argv[len(root):])}")
        code = stage.handler(stage)
        if code != 0:
            print(
                f"refresh stopped at '{name}' (exit {code}). The stages after it "
                "depend on what it was supposed to produce.",
                file=sys.stderr,
            )
            return code
        ran.append(name)

    print(f"refresh: {len(ran)} stage(s) completed: {', '.join(ran) or 'none'}")
    return 0


def command_status(args: argparse.Namespace) -> int:
    config = _load_config(args)
    # --rebuild is the one command asking to *fix* the ledger, so it is the one
    # that loads a misshapen entry instead of refusing it. Every other command
    # refuses, and refuses here — before anything is written.
    book = ledger.load(config.ledger_file, repair=args.rebuild)
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
        for record_id, parked in book.repaired.items():
            for where in parked:
                print(
                    f"repaired: {record_id} had an unreadable structured key; "
                    f"it was reset to empty and its old value kept as {where!r}. "
                    "The rebuild below refills what it can, which is never all "
                    "of it — check that key before deleting it.",
                    file=prose,
                )
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
    # Defensive rather than reachable: migrate-inline prefers the inline note
    # for every non-empty mergeable field, so the only conflict left is an
    # identity one — and `_check_merged_faithfully` refuses those before this
    # line, because two copies that disagree about what a record *is* cannot be
    # merged by a rule. Should that ever change, this command still has no
    # --prefer-incoming to offer.
    _print_merge_summary(result.outcomes, prefer_incoming_available=False)
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

    reviews_parser = subparsers.add_parser(
        "import-jpdb-reviews",
        help="Tag records jpdb already drills, from its review export",
    )
    reviews_parser.add_argument(
        "file",
        type=_path,
        help="jpdb's 'Export vocabulary reviews' JSON file.",
    )
    reviews_parser.set_defaults(handler=command_import_jpdb_reviews)

    enrich_parser = subparsers.add_parser(
        "enrich",
        help="Fill or improve fields on records janki already has",
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
        help="Write examples and usage notes with the Claude API.",
    )
    enrich_parser.add_argument(
        "--batch-submit",
        action="store_true",
        help=(
            "Send the --ai pass as one Message Batch: half price, answered "
            "within a day. Collect it later with --batch-fetch."
        ),
    )
    enrich_parser.add_argument(
        "--batch-fetch",
        action="store_true",
        help=(
            "Collect the pending batch if it has finished, or report how far "
            "along it is."
        ),
    )
    enrich_parser.add_argument(
        "--batch-forget",
        action="store_true",
        help=(
            "Drop the pending batch without collecting it, for one that can no "
            "longer be applied. Its results stay reachable from the console."
        ),
    )
    enrich_parser.add_argument(
        "--polish-meanings",
        action="store_true",
        help=(
            "Propose better English glosses for records that already have some, "
            "confirmed one record at a time."
        ),
    )
    enrich_parser.add_argument(
        "--force-fields",
        metavar="FIELD[,FIELD]",
        help=(
            "Fields the pass may overwrite. By default it only fills empty ones. "
            f"With --jpdb: {', '.join(enrich.ENRICHABLE_FIELDS)}. "
            f"With --ai: {', '.join(enrich.AI_FIELDS)}."
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
        "--model",
        metavar="ID",
        help=(
            "Override the configured model for this run "
            "(--ai and --polish-meanings)."
        ),
    )
    enrich_parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite the staging file a large --ai run writes. Without it, a "
            "file already under review is left alone."
        ),
    )
    enrich_parser.add_argument(
        "--yes",
        action="store_true",
        help="Write the shown changes without confirming.",
    )
    enrich_parser.set_defaults(handler=command_enrich)

    extract_parser = subparsers.add_parser(
        "extract",
        help="Read vocabulary off PDFs and photos into staging files",
    )
    extract_parser.add_argument(
        "files", nargs="+", type=_path, metavar="FILE", help="PDFs or photos to read."
    )
    extract_parser.add_argument(
        "--mode",
        choices=extract.MODES,
        help=(
            "Force how the source is read. Omit to let the model judge each "
            "page, which is right when one document holds both."
        ),
    )
    extract_parser.add_argument(
        "--model",
        metavar="ID",
        help="Override the configured extract model for this run.",
    )
    extract_parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite an existing staging file. Without it, a file already "
            "under review is left alone."
        ),
    )
    extract_parser.set_defaults(handler=command_extract)

    audio_parser = subparsers.add_parser(
        "audio", help="Generate word and example audio for records"
    )
    audio_parser.add_argument(
        "ids", nargs="*", metavar="ID", help="Record ids. Omit for every record."
    )
    audio_parser.add_argument(
        "--words", action="store_true", help="Word audio, with the accent forced."
    )
    audio_parser.add_argument(
        "--examples", action="store_true", help="Example-sentence audio, read naturally."
    )
    audio_parser.add_argument(
        "--provider",
        choices=("voicevox", "azure"),
        help="Override [tts] provider for this run.",
    )
    audio_parser.add_argument(
        "--force", action="store_true", help="Regenerate clips that already exist."
    )
    audio_parser.add_argument(
        "--prune", action="store_true", help="Delete janki-* clips nothing references."
    )
    audio_parser.add_argument(
        "--allow-default-accent",
        action="store_true",
        help=(
            "Voice records with no accent pattern, letting the engine choose — "
            "tagged 'accent_unverified' in the ledger."
        ),
    )
    audio_parser.set_defaults(handler=command_audio)

    promote_parser = subparsers.add_parser(
        "promote",
        help="Move a reviewed staging file's records into the collection",
    )
    promote_parser.add_argument(
        "file", type=_path, metavar="FILE", help="The staging file to promote."
    )
    promote_parser.add_argument(
        "--skip-reading-check",
        action="store_true",
        help=(
            "Do not ask jpdb whether each reading exists. Readings still have "
            "to be kana — that rule is about whether an ID can exist at all."
        ),
    )
    promote_parser.set_defaults(handler=command_promote)

    validate_parser = subparsers.add_parser("validate", help="Validate records or decks")
    validate_parser.add_argument("path", type=_path, nargs="?")
    validate_parser.set_defaults(handler=command_validate)

    build_command = subparsers.add_parser("build", help="Build one or all Anki decks")
    build_command.add_argument("deck", type=_path, nargs="?")
    build_command.add_argument("--all", action="store_true")
    build_command.add_argument("--output", type=_path)
    build_command.add_argument(
        "--only-new",
        action="store_true",
        help=(
            "Include only records this deck has never been built with, per the "
            "ledger's export history."
        ),
    )
    build_command.add_argument(
        "--yes",
        action="store_true",
        help="Do not ask about records missing audio, examples, or pitch accent.",
    )
    build_command.set_defaults(handler=command_build)

    refresh_parser = subparsers.add_parser(
        "refresh",
        help="Enrich, voice, and build what is new — the whole pipeline in order.",
    )
    refresh_parser.add_argument(
        "--deck",
        type=_path,
        help="Build only this deck (path or bare name). Default: every deck.",
    )
    for _name, _flag, _command in _REFRESH_STAGES:
        refresh_parser.add_argument(
            _flag, action="store_true", help=f"Skip the {_name} stage."
        )
    refresh_parser.set_defaults(handler=command_refresh)

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
