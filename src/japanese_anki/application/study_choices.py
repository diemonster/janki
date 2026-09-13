"""The owner's own decisions about one study job: one service, two front doors.

`docs/ASSISTANT_STUDY_JOBS_CONTRACTS.md` §9.1 and §9.3. The Assistant's
owner-only review/disposition editor and the explicit `janki study
review|coverage|disposition` commands are two ways of reaching **this**, so
neither can hold a second opinion about what the owner decided, what it was
decided over, or whether it still applies.

Three rules shape everything below.

**Nothing here derives what somebody else owns.** The parts of a job come from
:func:`study_job.job_effective_frontier`; the page they were decided over comes
from :func:`study_job.render_job_preview`; a part's current bytes, rows and
pattern control come from :class:`~japanese_anki.workbench.review.ReviewPanel`;
and every write is one :func:`study_job.record_choice` compare-and-swap against
its own closed validators. There is no second frontier traversal, no second
staging or coverage validator, and no finish receipt is read. When an owning
reader refuses a state, that refusal is what the caller sees.

**A decision is bound to what the owner was looking at.** A per-part save takes
the job revision and the rendering fingerprint the surface *displayed*,
resolves the current part and its current staging sha256, and compares that
displayed fingerprint with a fresh rendering. A difference refuses. The fresh
value never quietly takes the displayed one's place, because storing it would
record the owner as having decided over a page they never saw.

**No decision is inferred.** An empty review selection is a decision the owner
stated and is stored; an omitted pattern choice for a part that has one is a
refusal; a reason is the owner's literal text, saved exactly as typed; an
unreadable row selection refuses rather than widening to the whole part; and a
missing audio preference means sentence audio is included, never an implicit
opt-out.

This writes no staging review mark, no coverage verdict, no canonical byte and
no provider request: §7's one **Apply and finish** confirmation authorizes
those exact writes from these saved decisions. The rows a part holds are read
structurally, by identity — the Japanese on them belongs to the owner's local
preview and never to this projection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from japanese_anki.application import study_job
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.workbench.review import ReviewPanel

__all__ = [
    "PART_CHOICE_KEYS",
    "JobDecisions",
    "PartDecisions",
    "SavedChoice",
    "StudyChoicesError",
    "read_job_decisions",
    "save_audio_preference",
    "save_coverage_reason",
    "save_disposition",
    "save_review",
    "withdraw_choice",
]


class StudyChoicesError(JankiError):
    """An owner decision could not be read, or could not be safely saved."""


#: The four per-part decisions this service reads, saves and withdraws. The
#: job-wide sentence-audio preference is deliberately not one of them: it binds
#: no part, so it is neither displayed per part nor withdrawn per part.
PART_CHOICE_KEYS = (
    "review_flags",
    "review_patterns",
    "coverage_reasons",
    "dispositions",
)

#: §9.1's four bindings, in the order a refusal reads them back. The write path
#: never uses this list — `record_choice` validates and stores the bindings
#: through its own closed schema — so this is only how a *read* says whether a
#: stored decision still describes what is on the page now.
_BINDINGS = ("job_id", "part", "staging_sha256", "rendering_fingerprint")


# --- what a surface displays --------------------------------------------------


@dataclass(frozen=True, slots=True)
class SavedChoice:
    """One decision this job holds for one part, and whether it still applies.

    ``entry`` is the stored mapping exactly as it was saved, bindings and saved
    date included, so an editor shows what the owner decided rather than a
    re-derivation of it. ``stale`` compares its four bindings with the four
    current values; a read **discloses** a stale decision instead of refusing
    over it, because the owner has to see what they decided before and that it
    no longer applies. The finish is where a stale choice refuses.
    """

    key: str
    entry: Mapping[str, Any]
    stale: bool


@dataclass(frozen=True, slots=True)
class PartDecisions:
    """One current, settled part of a job as an owner decides over it."""

    #: This part's name, which is its effective attempt's own source name.
    part: str
    #: That attempt's typed identity: a job is not one batch, and two batches'
    #: source 1 are two different things.
    ref: study_job.JobChildRef
    staging_path: Path
    #: Those exact bytes now, which is one of the four bindings a save records.
    staging_sha256: str
    #: The rows this part currently holds, by structural identity and in
    #: document order. Identities only: nothing here says what a row means.
    record_ids: tuple[str, ...]
    #: Which of those rows a review may flag — `ReviewPanel.reviewable_record_ids`,
    #: the owning set, kept in this part's own document order. A row is left out
    #: when its sentences are already approved or when there is nothing on it to
    #: approve; *why* is the panel's answer, not one this service re-derives.
    reviewable_record_ids: tuple[str, ...]
    #: Whether this part's pattern-store entry is one a review may name at all
    #: — `ReviewPanel.pattern_selectable`, which is presence plus the lineage
    #: check. A review save states a pattern decision exactly when this is true.
    pattern_selectable: bool
    #: `ReviewPanel.pattern_warning`, carried word for word, so a surface can
    #: say *why* there is no pattern control instead of only that there is none.
    #: `None` exactly when the entry is selectable.
    pattern_warning: str | None
    #: Keyed by :data:`PART_CHOICE_KEYS`, holding only what this part really has.
    choices: Mapping[str, SavedChoice]


@dataclass(frozen=True, slots=True)
class JobDecisions:
    """Everything one owner-decision surface needs, read once and consistently."""

    job_id: str
    #: The exact job revision these choices were read at, and the one a save
    #: made from this display compares and swaps against.
    revision: str
    #: The owning preview: the real interactive card page and its own sha256.
    #: The owner's local view of their own proposals — never model context.
    preview: study_job.StudyJobPreview
    #: The job's current parts whose one effective attempt is complete.
    parts: tuple[PartDecisions, ...]
    #: Every other effective attempt, carrying its own state. A part still in
    #: flight is named rather than silently absent; whether the *job* may finish
    #: stays the finish's question.
    pending: tuple[study_job.JobChildAttempt, ...]
    #: The effective job-wide choice: what is saved, or the default.
    include_example_audio: bool
    #: Whether the owner really stated it, which the default is not.
    audio_choice_saved: bool

    @property
    def rendering_fingerprint(self) -> str:
        """The fingerprint a save made from this display must supply back."""

        return self.preview.rendering_fingerprint


# --- the owning readers, used once each ---------------------------------------


def _effective(config: ProjectConfig, job_id: str) -> study_job.JobFrontier:
    """This job's frontier, with one current attempt per part or a refusal.

    §9.5's derivation, unchanged. What is added here is the question an owner
    decision asks of it: a decision binds *one* attempt at one part, so a job
    holding two current attempts at the same part has no answer to give. Taking
    the newer one would bind the decision to an attempt nobody chose, so both
    are named and nothing is read or written.
    """

    frontier = study_job.job_effective_frontier(config, job_id)
    found: dict[str, list[study_job.JobChildAttempt]] = {}
    for attempt in frontier.effective:
        found.setdefault(attempt.ref.source_name, []).append(attempt)
    ambiguous = sorted(name for name, attempts in found.items() if len(attempts) > 1)
    if ambiguous:
        raise StudyChoicesError(
            f"Study job {job_id} holds more than one current attempt at "
            + "; ".join(
                f"{name}: "
                + " and ".join(attempt.ref.named for attempt in found[name])
                for name in ambiguous
            )
            + ". An owner decision binds one attempt at one part, and which of "
            "these is current is not settled by which ran last, so janki "
            "refuses to choose between them."
        )
    return frontier


@dataclass(frozen=True, slots=True)
class _Target:
    """One part exactly as it stands now.

    The same view a read displays and a save resolves again at save time, so
    the two cannot answer differently about which rows a part holds, which of
    them a review may flag, or what its pattern control is.
    """

    part: str
    staging_sha256: str
    record_ids: tuple[str, ...]
    reviewable_record_ids: tuple[str, ...]
    pattern_selectable: bool
    pattern_warning: str | None


def _part_view(
    config: ProjectConfig, job_id: str, attempt: study_job.JobChildAttempt
) -> _Target:
    """This part's current staging bytes, row identities and pattern control.

    All of it through :class:`ReviewPanel`, which already owns reading a
    staging document beside the pattern store: nothing here parses either file
    a second way, and nothing here reads what a row says.
    """

    try:
        panel = ReviewPanel.open(
            attempt.staging_path,
            staging_dir=config.staging_dir,
            patterns_path=config.patterns_file,
            collection_name=config.normalized_file.name,
        )
    except JankiError as exc:
        raise StudyChoicesError(
            f"{attempt.ref.source_name} of study job {job_id} is staged at "
            f"{attempt.staging_path}, which cannot be read: {exc}"
        ) from exc
    record_ids = tuple(record.id for record in panel.records)
    reviewable = panel.reviewable_record_ids
    return _Target(
        part=attempt.ref.source_name,
        staging_sha256=panel.staging_fingerprint,
        record_ids=record_ids,
        # The panel's own set, put back in the document's order: *which* rows
        # are in it is entirely its answer, and a frozenset has no order for a
        # surface to display rows in.
        reviewable_record_ids=tuple(
            record_id for record_id in record_ids if record_id in reviewable
        ),
        pattern_selectable=panel.pattern_selectable,
        pattern_warning=panel.pattern_warning,
    )


def _saved_choices(
    job: study_job.StudyJob,
    part: str,
    *,
    staging_sha256: str,
    rendering_fingerprint: str,
) -> dict[str, SavedChoice]:
    """Every decision this job holds for this part, each with its standing."""

    fresh = (job.header.job_id, part, staging_sha256, rendering_fingerprint)
    held: dict[str, SavedChoice] = {}
    for key in PART_CHOICE_KEYS:
        stored = job.choices.get(key)
        entry = stored.get(part) if isinstance(stored, Mapping) else None
        if not isinstance(entry, Mapping):
            continue
        bound = tuple(str(entry.get(name) or "") for name in _BINDINGS)
        held[key] = SavedChoice(key=key, entry=entry, stale=bound != fresh)
    return held


def read_job_decisions(config: ProjectConfig, job_id: str) -> JobDecisions:
    """One read of everything an owner-decision surface displays. Writes nothing.

    The job document is loaded first, so the revision reported is never newer
    than the choices reported with it: a save made from a display that has since
    been overtaken refuses at the compare-and-swap rather than landing against a
    revision whose choices nobody saw.

    A part is offered when its one effective attempt is ``complete`` — committed
    *and* with its grammar half written — because that is what a decision has to
    bind to. Looking at a partly finished job is legitimate, so the rest are
    named in ``pending`` with their own states, and whether the job may finish
    remains the finish's question rather than this read's.
    """

    job = study_job.load_study_job(config, job_id)
    frontier = _effective(config, job_id)
    pending: list[study_job.JobChildAttempt] = []
    settled: list[tuple[study_job.JobChildAttempt, _Target]] = []
    for attempt in frontier.effective:
        if not attempt.complete:
            pending.append(attempt)
            continue
        settled.append((attempt, _part_view(config, job_id, attempt)))
    # Last, and by the owning renderer: a job whose page cannot be drawn has no
    # rendering for a decision to bind to, and that refusal is the honest answer
    # rather than a page this service invented a fingerprint for.
    preview = study_job.render_job_preview(config, job_id)
    audio = job.choices.get("include_example_audio")
    stated = isinstance(audio, Mapping) and isinstance(audio.get("value"), bool)
    return JobDecisions(
        job_id=job_id,
        revision=job.revision,
        preview=preview,
        parts=tuple(
            PartDecisions(
                part=view.part,
                ref=attempt.ref,
                staging_path=attempt.staging_path,
                staging_sha256=view.staging_sha256,
                record_ids=view.record_ids,
                reviewable_record_ids=view.reviewable_record_ids,
                pattern_selectable=view.pattern_selectable,
                pattern_warning=view.pattern_warning,
                choices=_saved_choices(
                    job,
                    view.part,
                    staging_sha256=view.staging_sha256,
                    rendering_fingerprint=preview.rendering_fingerprint,
                ),
            )
            for attempt, view in settled
        ),
        pending=tuple(pending),
        include_example_audio=bool(audio["value"]) if stated else True,
        audio_choice_saved=stated,
    )


# --- one resolution behind all three per-part carriers ------------------------


def _resolve(
    config: ProjectConfig, job_id: str, part: str, rendering_fingerprint: Any
) -> _Target:
    """The current part, its current bytes, and proof of the page it was decided on.

    In that order, because each step is cheaper and more specific than the next:
    a part this job does not currently hold is named before anything is drawn.
    The comparison at the end is the whole point of §9.3's ``--rendering``: the
    supplied fingerprint is what a decision is stored against, and a fresh one
    that happens to be computed here is never substituted for it.
    """

    if not isinstance(rendering_fingerprint, str) or not rendering_fingerprint.strip():
        raise StudyChoicesError(
            "An owner decision records the fingerprint of the rendering it was "
            "taken over, and this one names nothing readable. Nothing was "
            "written."
        )
    frontier = _effective(config, job_id)
    attempt = next(
        (
            candidate
            for candidate in frontier.effective
            if candidate.ref.source_name == part and candidate.complete
        ),
        None,
    )
    if attempt is None:
        offered = [
            candidate.ref.source_name
            for candidate in frontier.effective
            if candidate.complete
        ]
        raise StudyChoicesError(
            f"Study job {job_id} holds no settled current part named {part!r} to "
            "decide over; it holds "
            + (", ".join(offered) if offered else "no settled part yet")
            + ". Nothing was written."
        )
    target = _part_view(config, job_id, attempt)
    fresh = study_job.render_job_preview(config, job_id).rendering_fingerprint
    if rendering_fingerprint != fresh:
        raise StudyChoicesError(
            f"The decision for {part} was taken over rendering "
            f"{rendering_fingerprint}, and study job {job_id} renders as {fresh} "
            "now. janki never records a decision against a page you did not "
            "look at, so open the current preview and decide again. Nothing was "
            "written."
        )
    return target


def _bindings(
    job_id: str, target: _Target, rendering_fingerprint: str
) -> dict[str, str]:
    """§9.1's four values, with the owner's own fingerprint among them."""

    return {
        "job_id": job_id,
        "part": target.part,
        "staging_sha256": target.staging_sha256,
        "rendering_fingerprint": rendering_fingerprint,
    }


def _outside(stated: Sequence[Any], offered: Sequence[str]) -> list[str]:
    """The readable ids in a stated selection that ``offered`` does not contain."""

    return [
        item
        for item in stated
        if isinstance(item, str) and item.strip() and item not in offered
    ]


def _stated_ids(
    value: Any, target: _Target, *, label: str, offered: Sequence[str] | None = None
) -> Any:
    """The selection exactly as the caller stated it, proved to name this part.

    Row identity is structural and is checked against the part's own current
    rows — never against what a row says. Anything that is not a plain sequence
    of ids is handed to ``record_choice``'s closed validator untouched: a
    repair here is the widening §2.4 refuses, where one held row's disposition
    quietly becomes the whole page's.

    ``offered`` narrows that further for a decision whose consumer is narrower
    than the part: a review flags only the rows :class:`ReviewPanel` offers for
    review, while a disposition holds back any row the part holds at all.
    """

    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        return value
    stated = list(value)
    unknown = _outside(stated, target.record_ids)
    if unknown:
        raise StudyChoicesError(
            f"The {label} for {target.part} names "
            + ", ".join(unknown)
            + ", which this part does not currently hold. Nothing was written."
        )
    refused = _outside(stated, offered) if offered is not None else []
    if refused:
        raise StudyChoicesError(
            f"The {label} for {target.part} names "
            + ", ".join(refused)
            + ", which this part holds but does not offer for review. Nothing "
            "was written."
        )
    return stated


# --- the saves ----------------------------------------------------------------


def save_review(
    config: ProjectConfig,
    job_id: str,
    part: str,
    *,
    expected_revision: str,
    rendering_fingerprint: str,
    record_ids: Sequence[str],
    patterns: bool | None = None,
) -> study_job.StudyJob:
    """Save which rows this part flags, and its standalone pattern choice.

    ``record_ids`` has no default: "no rows flagged" is something the owner
    states, and ``[]`` says it. It may name only rows this part *offers for
    review*, which is :attr:`ReviewPanel.reviewable_record_ids` and exactly what
    this decision's one consumer accepts — the finish forwards it into
    ``ReviewPanel._validate_actions``. A row the part holds but does not offer
    refuses here, at the control the owner is looking at, rather than later
    against the whole job. ``patterns`` is required exactly when the part
    carries a pattern set a review may name, and must be omitted when it does
    not — §9.3's ``--patterns``/``--no-patterns``, never inferred from a missing
    argument and never accepted where there is nothing to mark.

    Both land in **one** compare-and-swap. They stay two independent choices in
    storage — the pattern mark's one home is ``review_patterns[part].value``, and
    the review selection holds no second copy for the two to disagree over.
    """

    target = _resolve(config, job_id, part, rendering_fingerprint)
    if target.pattern_selectable:
        if not isinstance(patterns, bool):
            raise StudyChoicesError(
                f"{part} carries a pattern set this review may mark, so this "
                "save states exactly whether it marks it. janki never reads an "
                "omitted pattern decision as a no. Nothing was written."
            )
    elif patterns is not None:
        raise StudyChoicesError(
            f"{part} has no pattern set a review of this staging run may name, "
            "so it takes no pattern decision. Nothing was written."
        )
    bindings = _bindings(job_id, target, rendering_fingerprint)
    choice: dict[str, Any] = {
        "review_flags": {
            part: {
                **bindings,
                "record_ids": _stated_ids(
                    record_ids,
                    target,
                    label="review selection",
                    offered=target.reviewable_record_ids,
                ),
            }
        }
    }
    if patterns is not None:
        choice["review_patterns"] = {part: {**bindings, "value": patterns}}
    return study_job.record_choice(
        config, job_id, choice, expected_revision=expected_revision
    )


def save_coverage_reason(
    config: ProjectConfig,
    job_id: str,
    part: str,
    *,
    expected_revision: str,
    rendering_fingerprint: str,
    reason: str,
) -> study_job.StudyJob:
    """Save this part's coverage reason, in the owner's own literal words.

    Passed through exactly as it was typed. Nothing here strips, collapses or
    normalizes it, and nothing here composes one: a blank reason refuses in
    ``record_choice``'s own validator, which is the same refusal
    ``coverage._approval_payload`` already makes.
    """

    target = _resolve(config, job_id, part, rendering_fingerprint)
    return study_job.record_choice(
        config,
        job_id,
        {
            "coverage_reasons": {
                part: {
                    **_bindings(job_id, target, rendering_fingerprint),
                    "reason": reason,
                }
            }
        },
        expected_revision=expected_revision,
    )


def save_disposition(
    config: ProjectConfig,
    job_id: str,
    part: str,
    *,
    expected_revision: str,
    rendering_fingerprint: str,
    action: str,
    reason: str,
    record_ids: Sequence[str] | None = None,
) -> study_job.StudyJob:
    """Exclude or defer this whole part, or exactly the rows it holds back.

    ``record_ids=None`` omits the field, which is §7.7's documented whole-part
    spelling; an explicit ``[]`` says the same thing. Anything else is recorded
    as supplied and validated as supplied, because a selection janki cannot read
    is a mistake about which rows are meant, not permission to mean all of them.
    ``action`` and the literal ``reason`` are checked by ``record_choice``'s own
    closed validator.

    Every row the part currently holds is a target, including one a review can
    no longer flag: §7.7 disposes held and excluded rows, and holding back an
    already-approved row is an ordinary decision rather than a review of it.
    """

    target = _resolve(config, job_id, part, rendering_fingerprint)
    entry: dict[str, Any] = {
        **_bindings(job_id, target, rendering_fingerprint),
        "action": action,
        "reason": reason,
    }
    if record_ids is not None:
        entry["record_ids"] = _stated_ids(record_ids, target, label="disposition")
    return study_job.record_choice(
        config,
        job_id,
        {"dispositions": {part: entry}},
        expected_revision=expected_revision,
    )


def withdraw_choice(
    config: ProjectConfig,
    job_id: str,
    part: str,
    key: str,
    *,
    expected_revision: str,
) -> study_job.StudyJob:
    """Take back exactly one saved per-part decision, and nothing else.

    §2.4's existing per-part ``None`` convention through the same
    compare-and-swap that recorded it: no second delete pathway, and no
    confirmation dialog, because a reversible local decision being reversed buys
    nothing. Only the named key and part are dropped; every other decision keeps
    exactly what it held. A withdrawal of something this job does not hold at
    this revision refuses inside ``record_choice`` rather than spending a
    revision on a no-op, and this needs no rendering: retracting a stored choice
    is not a decision taken over a page.
    """

    if key not in PART_CHOICE_KEYS:
        raise StudyChoicesError(
            f"{key!r} is not a per-part owner decision, so there is no per-part "
            "entry to withdraw. This service withdraws "
            + ", ".join(PART_CHOICE_KEYS)
            + "."
        )
    return study_job.record_choice(
        config, job_id, {key: {part: None}}, expected_revision=expected_revision
    )


def save_audio_preference(
    config: ProjectConfig,
    job_id: str,
    *,
    expected_revision: str,
    include_example_audio: bool,
) -> study_job.StudyJob:
    """Save the job-wide sentence-audio choice. It binds the job and nothing else.

    §9.1 makes this preference carry ``job_id`` and the job document's CAS
    revision **only** — no part, no staging hash, no rendering fingerprint and
    no written reason. Nothing is rendered or read from staging here, which is
    what makes a re-rendered part unable to stale it, and what stops this save
    from staling a saved content decision merely by advancing the job's
    revision. ``True`` re-enables a previous opt-out; a missing choice means
    sentence audio is included and is never an implicit ``False``.

    It buys nothing: no clip, no provider request, and no change to an
    already-recorded finish authority.
    """

    return study_job.record_choice(
        config,
        job_id,
        {"include_example_audio": {"value": include_example_audio}},
        expected_revision=expected_revision,
    )
