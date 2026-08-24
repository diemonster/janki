"""What an extraction run would send, decided before anybody consents to it.

`WORKBENCH_PLAN.md` W1.1b, for W3. `command_extract` did all of this inline
between reading the config and asking the question, which meant the only way to
find out what a run would send was to start one.

W3 needs it as a value: *"Adding a file to the corpus and sending it to a model
are two separate actions, always"*, and the consent button has to name exactly
what leaves the computer — this file, this model, a paid API call. A dialog
cannot render any of that from a function that asks and dispatches in one
breath.

So the run splits where the consent does. `plan_extraction` resolves every
target, every fingerprint and every refusal that can be known without paying —
a staging collision, an unreadable pattern store — and returns them. Whether to
spend the money is then the caller's decision, made on a value it can show
someone, and the same value is what the paid half consumes.

**This is not a pure preview and must not be mistaken for one.** By the time it
is called the inputs are already copied into `data/inbox/`, because reading a
file is how its collision and its fingerprint are known at all. Nothing here
contacts a provider, journals an operation, or writes a staging file; the
inbox copy has already happened, and `kept_in_inbox` is what a caller tells
somebody who then declines.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from japanese_anki import extract, patterns
from japanese_anki.config import ProjectConfig
from japanese_anki.inputs import PreparedInput
from japanese_anki.io import load_records
from japanese_anki.models import VocabularyRecord

__all__ = ["ExtractionPlan", "ExtractionTarget", "plan_extraction"]


@dataclass(frozen=True, slots=True)
class ExtractionTarget:
    """One input, and everything decided about it before the call."""

    item: PreparedInput
    #: The staging file this input's proposals would be written to.
    staging_path: Path
    source_sha256: str
    #: The exact request identity, including `request_fingerprint`. Journalled
    #: before dispatch so a crash between sending and parsing can still say
    #: which call was made — which is only true while it describes the request
    #: that is actually sent, so the dispatch reads it from here rather than
    #: rebuilding one of its own.
    provenance: Mapping[str, Any]
    #: A staging file this input would overwrite. Empty unless `force` was
    #: given: without it, planning refuses instead. W3 confirms a replacement
    #: separately and has to name the review it invalidates.
    replaces: Path | None = None

    @property
    def name(self) -> str:
        """The permanent filename, which is what a person recognizes."""
        return self.item.origin_path.name


@dataclass(frozen=True, slots=True)
class ExtractionPlan:
    """What a run would send, and to whom, before anyone agrees to it."""

    model: str
    mode: str | None
    targets: tuple[ExtractionTarget, ...]
    #: The instructions every input in this run is extracted under. Read once
    #: for the run: re-reading per file would let a mid-run edit split one
    #: command across two prompts.
    style_guide: str
    system: str
    #: Expressions the collection already has, sent only in prose mode — a
    #: table is transcribed row by row, and telling the model to skip rows
    #: would put holes in a faithful transcription.
    #:
    #: Sorted, and that is not cosmetic: this text goes into the prompt the
    #: request fingerprint is computed over, so an unordered set would give the
    #: same corpus a different request identity on every run — and that
    #: identity is what the journal records and a retry compares.
    skip_list: tuple[str, ...]
    known: frozenset[str]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(target.name for target in self.targets)

    @property
    def kept_in_inbox(self) -> tuple[Path, ...]:
        """Files *this* run copied into the inbox, whether or not it sends them.

        The half a refusal message forgets. The copy happens before consent, so
        "nothing was sent" alone reads as "nothing happened" while a private
        document sits staged for the next `git add`.

        `item.copied`, not "is this path under the inbox": every branch of the
        copy returns a path under the inbox, including the ones that copy
        nothing, so a containment test announces files the run never touched.
        """
        return tuple(
            target.item.origin_path for target in self.targets if target.item.copied
        )


def plan_extraction(
    config: ProjectConfig,
    prepared: Sequence[PreparedInput],
    *,
    mode: str | None,
    model: str,
    style_guide: str,
    system: str,
    force: bool = False,
) -> ExtractionPlan:
    """Resolve everything a run can know before it spends anything.

    Every refusal that does not need a provider happens here, which is the
    point: a batch that would write two inputs to one staging file is refused
    now rather than after paying for both.
    """
    existing: list[VocabularyRecord] = (
        load_records(config.normalized_file)
        if config.normalized_file.exists()
        else []
    )
    known = frozenset(extract.known_ids(existing))
    skip_list = (
        tuple(sorted({record.expression for record in existing}))
        if mode == "prose"
        else ()
    )

    targets = extract.staging_targets(config.staging_dir, prepared, force=force)
    fingerprints = [
        item.source_sha256 or extract.source_fingerprint(item.origin_path)
        for item in prepared
    ]
    # Validate the complete store before a paid call. The parsed mapping is
    # deliberately not carried: a human can finish reviewing a source while the
    # call is in flight, and the post-call decision must see that newer state
    # under the writer lock rather than replacing it from this stale snapshot.
    patterns.load_store(config.patterns_file)

    return ExtractionPlan(
        model=model,
        mode=mode,
        targets=tuple(
            ExtractionTarget(
                item=item,
                staging_path=staging_path,
                source_sha256=fingerprint,
                replaces=staging_path if staging_path.exists() else None,
                provenance=extract.prompt_provenance(
                    item,
                    model=model,
                    style_guide=style_guide,
                    system=system,
                    mode=mode,
                    known=skip_list,
                    source_sha256=fingerprint,
                ),
            )
            for item, staging_path, fingerprint in zip(
                prepared, targets, fingerprints, strict=True
            )
        ),
        style_guide=style_guide,
        system=system,
        skip_list=skip_list,
        known=known,
    )
