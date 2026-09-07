"""Whether a deck or a staging file would survive being built, asked anywhere.

`WORKBENCH_PLAN.md` W1.1b. `japanese_anki.validation` holds the rules; this
holds the *sweep* — resolving what to look at, reading each file, and
collecting what it found — which lived inside `command_validate` and was
therefore reachable only by typing the command.

It is the last shape a browser needs from validation: a page that says a
source is ready to add should not have to guess whether the deck it lands in
will then refuse to build.

Nothing here writes, and nothing here prints. A refusal comes back as an
issue in the list rather than as an exception, including the refusal to read
the file at all — a hand-edited deck is the likeliest thing in `data/decks/`
to be malformed, and its own unreadability must not cancel the sweep before
any other deck is reported.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from japanese_anki import patterns
from japanese_anki import status as status_module
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.exporters import kanji_cards, pattern_cards
from japanese_anki.exporters.anki import deck_kind, resolve_deck_records
from japanese_anki.io import DataError, load_records, load_structured
from japanese_anki.validation import ValidationIssue, has_errors, validate_records

__all__ = ["ValidationReport", "validate_path", "validate_project"]


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """What one sweep found, and over how much."""

    issues: tuple[ValidationIssue, ...] = ()
    #: Records examined, which is the denominator the count line needs: "0
    #: errors" over nothing read is a different answer from "0 errors" over
    #: nine hundred records.
    records: int = 0
    paths: tuple[Path, ...] = field(default_factory=tuple)

    @property
    def errors(self) -> int:
        return sum(issue.level == "error" for issue in self.issues)

    @property
    def warnings(self) -> int:
        return sum(issue.level == "warning" for issue in self.issues)

    @property
    def failed(self) -> bool:
        return has_errors(list(self.issues))


def validate_path(
    path: Path,
    store: Callable[[], Mapping[str, patterns.PatternSet]] | None = None,
    config: ProjectConfig | None = None,
) -> tuple[list, int]:
    try:
        raw = load_structured(path)
    except JankiError as exc:
        # A hand-edited deck file is the likeliest thing in `data/decks/` to be
        # malformed — an unterminated quote, a tab, a merge marker — and its own
        # unreadability used to cancel the sweep before a single deck was
        # reported, with no "Validated N records" line at all.
        return [
            ValidationIssue(
                "error",
                str(exc),
                source=str(path),
                code="validation-input-unreadable",
            )
        ], 0
    # `exporters.anki.deck_kind`, so this, the build and `status` cannot drift
    # about what a kind is. On truthiness alone a typo — or a deliberate
    # `kind: vocabulary` — sent an ordinary deck down the pattern path, which
    # invented two errors that are false for it and skipped every record the
    # file actually holds.
    #
    # Guarded on the shape first: `validate` also takes a records file, which is
    # a list, and `deck_kind` refuses a non-mapping because for a *deck* that is
    # a real error.
    if isinstance(raw, dict):
        try:
            kind = deck_kind(path)
        except DataError as exc:
            # Reported as this file's error rather than raised, so a sweep still
            # validates every other deck — the rule this command follows for a
            # deck it cannot read.
            return [
                ValidationIssue(
                    "error",
                    str(exc),
                    source=str(path),
                    code="validation-deck-kind-invalid",
                )
            ], 0
    else:
        kind = ""
    if kind == "kanji":
        # A character deck holds no records, so the ordinary path found none
        # and called the file clean — leaving every defect the build refuses
        # invisible to the command whose job is catching one first.
        return [
            ValidationIssue(
                "error",
                problem,
                source=str(path),
                code="validation-kanji-deck-invalid",
            )
            for problem in kanji_cards.deck_problems(path, config)
        ], 0
    if kind in ("pattern", "conjugation"):
        # A pattern or conjugation deck holds no records, so the ordinary path
        # found none and called the file clean — leaving every defect the build
        # refuses invisible to the command whose job is catching one first.
        try:
            problems = pattern_cards.deck_problems(
                # `store()` is resolved *inside* the guard: `patterns.json` is
                # machine-written and committed, so it can carry a merge marker,
                # and `deck_problems` — careful about every other failure it can
                # meet — never entered its own frame to catch that one.
                path, store() if store else None, config
            )
        except JankiError as exc:
            return [
                ValidationIssue(
                    "error",
                    str(exc),
                    source=str(path),
                    code="validation-pattern-input-unreadable",
                )
            ], 0
        return [
            ValidationIssue(
                "error",
                problem,
                source=str(path),
                code="validation-pattern-deck-invalid",
            )
            for problem in problems
        ], 0

    try:
        if isinstance(raw, dict) and "deck" in raw:
            _, records = resolve_deck_records(path)
        else:
            records = load_records(path)
    except JankiError as exc:
        # This file's error, like the two branches above. A missing or
        # unparseable collection used to escape to `main`, so the deck naming it
        # — first by name in `data/decks/` — cancelled the sweep, and the one
        # line printed named the collection but not the deck that pointed at it.
        return [
            ValidationIssue(
                "error",
                str(exc),
                source=str(path),
                code="validation-records-unreadable",
            )
        ], 0
    issues = validate_records(records, path)
    return issues, len(records)


def validate_project(
    config: ProjectConfig, path: Path | None = None
) -> ValidationReport:
    """Validate one file, or every deck under the configured deck tree.

    The pattern store is read lazily and at most once. It is machine-written
    and committed, so it can carry a merge marker — and reading it up front
    made that failure cancel `janki validate data/staging/…yaml`, a command
    with nothing to do with it.
    """
    loaded: dict[str, patterns.PatternSet] | None = None

    def store() -> Mapping[str, patterns.PatternSet]:
        nonlocal loaded
        if loaded is None:
            loaded = patterns.load_store(config.patterns_file)
        return loaded

    paths: Sequence[Path] = (
        [path.resolve()] if path is not None else status_module.deck_files(config)
    )
    issues: list[ValidationIssue] = []
    records = 0
    for one in paths:
        found, count = validate_path(one, store, config)
        issues.extend(found)
        records += count
    return ValidationReport(
        issues=tuple(issues), records=records, paths=tuple(paths)
    )
