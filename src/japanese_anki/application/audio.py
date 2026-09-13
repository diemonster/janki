"""Plan and execute one exact durable audio transaction.

The browser and CLI share this module all the way from an exact display plan to
the write-ahead provider transaction.  Targeted calls never interpret an empty
id set as the corpus, and a browser can bind its click to the fresh plan that is
recomputed under every repository-owner lock before any provider is contacted.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from japanese_anki import audio_cmd, ledger, operations, status
from japanese_anki.application import deck_capabilities
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.exporters import pattern_cards
from japanese_anki.io import (
    DataError,
    RecordsRevision,
    exclusive_path_lock,
    load_records,
    load_records_snapshot,
    load_structured,
    read_bytes_bound,
    records_revision,
    save_records_json_locked,
)
from japanese_anki.models import VocabularyRecord
from japanese_anki.tts import (
    SentenceProfileSelector,
    SpeechProvider,
    openai_realtime,
    sentence_profile_for,
    voicevox,
)

SentenceProvider = SpeechProvider | SentenceProfileSelector

AudioAccess = Literal["local-network", "paid-network"]
AudioPhase = Literal[
    "Preparing audio",
    "Creating audio",
    "Saving audio",
    "Cleaning up audio",
]
AudioExecutionState = Literal[
    "complete",
    "no-records",
    "records-stale",
    "generation-stopped",
    "finalization-stopped",
    "prune-incomplete",
    "ledger-incomplete",
]
AudioProgress = Callable[[AudioPhase], None]


class AudioPlanError(JankiError):
    """An audio request does not name one exact canonical scope."""


@dataclass(frozen=True, slots=True)
class AudioExecutionOutcome:
    """Truthful, surface-neutral result of one durable audio transaction."""

    state: AudioExecutionState
    plan: AudioPlan | None
    output_dir: Path
    file_count: int = 0
    written_record_ids: tuple[str, ...] = ()
    up_to_date: int = 0
    pruned_paths: tuple[Path, ...] = ()
    warnings: tuple[str, ...] = ()
    guessed_accent: tuple[str, ...] = ()
    no_reading: tuple[str, ...] = ()
    stopped_by: str | None = None
    prune_error: str | None = None
    ledger_error: str | None = None
    pending_recovery: bool = False
    record_references_written: bool = False
    media_published: bool = False
    ledger_committed: bool = False
    no_records: bool = False
    prune_requested: bool = False

    @property
    def succeeded(self) -> bool:
        return self.state in {"complete", "no-records"}


@dataclass(frozen=True, slots=True)
class AudioProviderPlan:
    """The provider/profile a planned clip kind will use."""

    name: str
    access: AudioAccess
    voice: int | str
    speed: float
    suffix: str
    destination: str
    transport: str
    settings: dict[str, str]


@dataclass(frozen=True, slots=True)
class AudioClipCounts:
    """How one clip kind will be satisfied by the existing transaction."""

    total: int
    current: int
    recoverable: int
    provider_required: int


@dataclass(frozen=True, slots=True)
class AudioClipPlan:
    """The exact request identity and disposition behind one displayed count."""

    record_id: str
    kind: Literal["word", "example"]
    target: str
    request_input: str
    forced_accent: bool
    content_fingerprint: str
    provider: AudioProviderPlan
    state: audio_cmd.AudioClipState
    recovery_key: str | None
    recovery_sha256: str | None
    recovery_source: audio_cmd.AudioRecoverySource | None


@dataclass(frozen=True, slots=True)
class AudioPlan:
    """A display-only, fingerprinted audio scope."""

    repository_root: Path
    canonical_path: Path
    canonical_revision: RecordsRevision
    ledger_path: Path
    media_dir: Path
    record_ids: tuple[str, ...]
    targeted: bool
    words: bool
    examples: bool
    force: bool
    word_counts: AudioClipCounts
    example_counts: AudioClipCounts
    word_provider: AudioProviderPlan | None
    example_provider: AudioProviderPlan | None
    protected_records: tuple[VocabularyRecord, ...]
    clips: tuple[AudioClipPlan, ...]
    fingerprint: str

    @property
    def clip_count(self) -> int:
        return self.word_counts.total + self.example_counts.total

    @property
    def provider_required_count(self) -> int:
        return self.word_counts.provider_required + self.example_counts.provider_required


@dataclass(frozen=True, slots=True)
class PaidAudioDispatch:
    """One fresh paid call at the last durable boundary before transport."""

    operation_id: str
    clip: AudioClipPlan


def _provider_name(config: ProjectConfig, chosen: str | None) -> str:
    return (chosen or config.tts_provider or "voicevox").strip().lower()


def _all_durable_audio_records(
    config: ProjectConfig,
    deck_paths: Sequence[Path],
    normalized_records: Sequence[VocabularyRecord] | None = None,
    *,
    deck_revisions: Mapping[Path, RecordsRevision] | None = None,
    drill_overrides: Mapping[Path, Sequence[VocabularyRecord]] | None = None,
) -> list[VocabularyRecord]:
    """Every record-shaped owner that can keep a media reference alive.

    The per-deck question belongs to
    :mod:`japanese_anki.application.deck_capabilities`; what stays here is the
    corpus-level assembly and the two ``drill-audio:`` refusals that only a
    whole-repository view can make.  Those refusals are why this consumes the
    declared and drill halves in separate passes: a synthetic owner colliding
    with a record version declared by a *later* deck has to be caught too.
    """
    revisions = {path.resolve(): revision for path, revision in (deck_revisions or {}).items()}
    projected_drills = {
        path.resolve(): list(records) for path, records in (drill_overrides or {}).items()
    }
    # Every kind first, before anything reads a source. A known kind with no
    # capability row refuses the whole census rather than contributing an
    # empty owner set that would make its clips look unreferenced.
    for deck_path in deck_paths:
        target = deck_path.resolve()
        deck_capabilities.deck_capability(target, revision=revisions.get(target))
    durable = list(
        normalized_records
        if normalized_records is not None
        else (load_records(config.normalized_file) if config.normalized_file.exists() else [])
    )
    for deck_path in deck_paths:
        target = deck_path.resolve()
        durable.extend(
            deck_capabilities.declared_media_owners(
                target, revision=revisions.get(target)
            )
        )
    ordinary_ids = {record.id for record in durable}
    drill_origins: dict[str, Path] = {}
    for deck_path in deck_paths:
        target = deck_path.resolve()
        drill_records = projected_drills.get(target)
        if drill_records is None:
            drill_records = deck_capabilities.drill_media_owners(
                config, target, revision=revisions.get(target)
            )
        for record in drill_records:
            if record.id in ordinary_ids:
                raise AudioPlanError(
                    f"Drill audio owner {record.id!r} collides with a durable "
                    "vocabulary record id. The 'drill-audio:' namespace is "
                    "reserved for conjugation-deck audio."
                )
            prior = drill_origins.setdefault(record.id, deck_path.resolve())
            if prior != deck_path.resolve():
                raise AudioPlanError(
                    f"Drill audio owner {record.id!r} is shared by {prior} and "
                    f"{deck_path.resolve()}. Give the decks distinct deck_id values "
                    "before generating, recovering, or pruning audio."
                )
        durable.extend(drill_records)
    return durable


def resolve_word_provider(config: ProjectConfig, chosen: str | None) -> SpeechProvider:
    """Construct the word provider without contacting it."""
    name = _provider_name(config, chosen)
    if name == "voicevox":
        return voicevox.VoicevoxProvider(
            base_url=config.voicevox_url,
            speaker=config.voicevox_speaker,
            speed=config.voicevox_speed,
        )
    if name == "azure":
        raise AudioPlanError(
            "The Azure provider was evaluated and dropped — VOICEVOX reads the "
            "ambiguous kanji correctly and needs no account (see M5.7 in "
            "docs/IMPLEMENTATION_PLAN.md). Words use VOICEVOX, which is the "
            "only engine here that can force a pitch accent; for sentences set "
            '[tts] sentence_provider = "openai-realtime".'
        )
    raise AudioPlanError(
        f"Unknown TTS provider {name!r}. Words are voiced by voicevox. For "
        "sentences, set [tts] sentence_provider to voicevox or openai-realtime."
    )


def resolve_sentence_provider(
    config: ProjectConfig,
    chosen: str | None,
    words: SpeechProvider,
) -> SentenceProvider:
    """Construct the example provider without contacting it."""
    name = (config.sentence_provider or "").strip().lower()
    if name == "openai-realtime":
        return openai_realtime.OpenAiRealtimePool(
            operations_path=config.operations_file,
        )
    if name not in {"", "voicevox"}:
        raise AudioPlanError(
            f"Unknown [tts] sentence_provider {name!r}. Known: voicevox, "
            "openai-realtime, or leave it empty to read sentences in the same "
            "voice as the words."
        )
    if _provider_name(config, chosen) == "voicevox":
        speaker = config.voicevox_sentence_speaker
        if speaker is not None and speaker != config.voicevox_speaker:
            return voicevox.VoicevoxProvider(
                base_url=config.voicevox_url,
                speaker=speaker,
                speed=config.voicevox_speed,
            )
    return words


def _provider_plan(provider: SentenceProvider) -> AudioProviderPlan:
    access: AudioAccess = "paid-network" if provider.name == "openai-realtime" else "local-network"
    if provider.name == "openai-realtime":
        destination = openai_realtime.ENDPOINT
    else:
        destination = str(getattr(provider, "_base_url", provider.name))
    transport_value = getattr(provider, "_transport", None)
    transport_type = transport_value if transport_value is not None else type(provider)
    transport = (
        f"{getattr(transport_type, '__module__', type(transport_type).__module__)}."
        f"{getattr(transport_type, '__qualname__', type(transport_type).__qualname__)}"
    )
    return AudioProviderPlan(
        name=provider.name,
        access=access,
        voice=provider.voice,
        speed=provider.speed,
        suffix=provider.suffix,
        destination=destination,
        transport=transport,
        settings=dict(provider.settings),
    )


def provider_plan_wire(provider: AudioProviderPlan | None) -> dict[str, object] | None:
    """The exact durable form of one planned provider profile.

    One definition, because the plan fingerprint, the confirmed enumeration and
    the completion proof all have to describe a voice the same way: a profile
    written two ways is a profile two readers can disagree about, and the voice
    is deliberately absent from a clip's content fingerprint.
    """

    if provider is None:
        return None
    return {
        "name": provider.name,
        "access": provider.access,
        "voice": provider.voice,
        "speed": provider.speed,
        "suffix": provider.suffix,
        "destination": provider.destination,
        "transport": provider.transport,
        "settings": dict(provider.settings),
    }


def _exact_ids(record_ids: Sequence[str]) -> tuple[str, ...]:
    if isinstance(record_ids, (str, bytes)):
        raise AudioPlanError("Audio record ids must be a nonempty sequence, not text.")
    selected = tuple(record_ids)
    if not selected:
        raise AudioPlanError(
            "Targeted audio needs at least one exact record id; an empty scope "
            "never means the whole collection."
        )
    if any(not isinstance(record_id, str) or not record_id for record_id in selected):
        raise AudioPlanError("Every targeted audio record id must be nonempty text.")
    repeated = [item for item, count in Counter(selected).items() if count > 1]
    if repeated:
        raise AudioPlanError(f"Audio record id {repeated[0]!r} was supplied twice.")
    return selected


def _fingerprint(plan: AudioPlan, records: Sequence[VocabularyRecord]) -> str:
    provider_wire = provider_plan_wire

    def counts_wire(counts: AudioClipCounts) -> dict[str, int]:
        return {
            "total": counts.total,
            "current": counts.current,
            "recoverable": counts.recoverable,
            "provider_required": counts.provider_required,
        }

    payload = {
        "version": 1,
        "repository_root": str(plan.repository_root),
        "canonical_path": str(plan.canonical_path),
        "canonical_revision": {
            "present": plan.canonical_revision.text is not None,
            "sha256": hashlib.sha256(
                (plan.canonical_revision.text or "").encode("utf-8")
            ).hexdigest(),
        },
        "ledger_path": str(plan.ledger_path),
        "media_dir": str(plan.media_dir),
        "record_ids": plan.record_ids,
        "targeted": plan.targeted,
        "words": plan.words,
        "examples": plan.examples,
        "force": plan.force,
        "word_counts": counts_wire(plan.word_counts),
        "example_counts": counts_wire(plan.example_counts),
        "word_provider": provider_wire(plan.word_provider),
        "example_provider": provider_wire(plan.example_provider),
        "protected_records": [record.to_dict() for record in plan.protected_records],
        "clips": [
            {
                "record_id": clip.record_id,
                "kind": clip.kind,
                "target": clip.target,
                "request_input": clip.request_input,
                "forced_accent": clip.forced_accent,
                "content_fingerprint": clip.content_fingerprint,
                "provider": provider_wire(clip.provider),
                "state": clip.state,
                "recovery_key": clip.recovery_key,
                "recovery_sha256": clip.recovery_sha256,
                "recovery_source": clip.recovery_source,
            }
            for clip in plan.clips
        ],
        "records": [record.to_dict() for record in records],
    }
    wire = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(wire).hexdigest()


def _plan_records(
    config: ProjectConfig,
    records: Sequence[VocabularyRecord],
    revision: RecordsRevision,
    record_ids: tuple[str, ...] | None,
    *,
    words: bool,
    examples: bool,
    force: bool,
    chosen_provider: str | None,
    word_provider: SpeechProvider | None,
    sentence_provider: SentenceProvider | None,
    protected_records: Sequence[VocabularyRecord],
    ledger_book: ledger.Ledger | None = None,
    owner_path: Path | None = None,
) -> AudioPlan:
    if not words and not examples:
        raise AudioPlanError(
            "janki audio needs --words, --examples, or both: they are different "
            "recordings made different ways, and neither is the obvious default."
        )
    canonical_path = (owner_path or config.normalized_file).resolve()
    if revision.path.resolve() != canonical_path:
        raise AudioPlanError(
            f"Audio records revision for {revision.path} cannot bind {canonical_path}."
        )
    ids = tuple(record.id for record in records) if record_ids is None else record_ids
    counts = Counter(record.id for record in records)
    missing = [record_id for record_id in ids if counts[record_id] == 0]
    if missing:
        raise AudioPlanError(f"No audio record has id {missing[0]!r}.")
    ambiguous = [record_id for record_id in ids if counts[record_id] > 1]
    if ambiguous:
        raise AudioPlanError(f"Audio record id {ambiguous[0]!r} occurs more than once.")
    by_id = {record.id: record for record in records}
    selected = tuple(by_id[record_id] for record_id in ids)
    word_provider = word_provider or resolve_word_provider(config, chosen_provider)
    sentence_provider = sentence_provider or (
        resolve_sentence_provider(config, chosen_provider, word_provider)
        if examples
        else word_provider
    )
    prepared: dict[str, list[SpeechProvider]] = {}
    if examples:
        try:
            prepared = audio_cmd.prepare_example_audio_profiles(
                selected,
                sentence_provider=sentence_provider,
            )
        except audio_cmd.AudioError as exc:
            raise AudioPlanError(str(exc)) from exc
    try:
        requirements = audio_cmd.audio_clip_requirements(
            records,
            wanted=set(ids),
            book=(
                ledger_book
                if ledger_book is not None
                else ledger.load(config.ledger_file.resolve())
            ),
            audio_dir=config.media_dir.resolve() / audio_cmd.AUDIO_SUBDIR,
            word_provider=word_provider,
            prepared_examples=prepared,
            words=words,
            examples=examples,
            force=force,
            # Every durable CLI/workbench run uses the write-ahead transaction.
            # Planning therefore counts exact staged recovery even though it
            # never adopts an orphan row itself.
            stage_only=True,
            adopt_pending=False,
            protected_records=protected_records,
        )
    except JankiError as exc:
        raise AudioPlanError(str(exc)) from exc

    clips = tuple(
        AudioClipPlan(
            record_id=requirement.record_id,
            kind=requirement.kind,
            target=requirement.target,
            request_input=requirement.request_input,
            forced_accent=requirement.forced_accent,
            content_fingerprint=requirement.content_fingerprint,
            provider=_provider_plan(requirement.provider),
            state=requirement.state,
            recovery_key=requirement.recovery_key,
            recovery_sha256=requirement.recovery_sha256,
            recovery_source=requirement.recovery_source,
        )
        for requirement in requirements
    )

    def clip_counts(kind: Literal["word", "example"]) -> AudioClipCounts:
        states = Counter(clip.state for clip in clips if clip.kind == kind)
        total = sum(states.values())
        current = states["current"]
        recoverable = states["recoverable"]
        provider_required = states["provider-required"]
        assert total == current + recoverable + provider_required
        return AudioClipCounts(
            total=total,
            current=current,
            recoverable=recoverable,
            provider_required=provider_required,
        )

    draft = AudioPlan(
        repository_root=config.root.resolve(),
        canonical_path=canonical_path,
        canonical_revision=revision,
        ledger_path=config.ledger_file.resolve(),
        media_dir=config.media_dir.resolve(),
        record_ids=ids,
        targeted=record_ids is not None,
        words=words,
        examples=examples,
        force=force,
        word_counts=clip_counts("word"),
        example_counts=clip_counts("example"),
        word_provider=_provider_plan(word_provider) if words else None,
        example_provider=_provider_plan(sentence_provider) if examples else None,
        protected_records=tuple(protected_records),
        clips=clips,
        fingerprint="",
    )
    return replace(draft, fingerprint=_fingerprint(draft, selected))


def plan_targeted_audio(
    config: ProjectConfig,
    record_ids: Sequence[str],
    *,
    words: bool = False,
    examples: bool = False,
    force: bool = False,
    chosen_provider: str | None = None,
    word_provider: SpeechProvider | None = None,
    sentence_provider: SentenceProvider | None = None,
) -> AudioPlan:
    """Plan one exact nonempty promoted-record scope."""
    records, revision = load_records_snapshot(config.normalized_file.resolve())
    protected_records = tuple(
        _all_durable_audio_records(config, status.deck_files(config), records)
    )
    return _plan_records(
        config,
        records,
        revision,
        _exact_ids(record_ids),
        words=words,
        examples=examples,
        force=force,
        chosen_provider=chosen_provider,
        word_provider=word_provider,
        sentence_provider=sentence_provider,
        protected_records=protected_records,
    )


def plan_targeted_audio_revision(
    config: ProjectConfig,
    records: Sequence[VocabularyRecord],
    revision: RecordsRevision,
    record_ids: Sequence[str],
    *,
    words: bool = False,
    examples: bool = False,
    force: bool = False,
    chosen_provider: str | None = None,
    word_provider: SpeechProvider | None = None,
    sentence_provider: SentenceProvider | None = None,
) -> AudioPlan:
    """Plan targeted canonical audio against exact prospective record bytes.

    A reviewed card revision needs to disclose its later audio spend before
    the canonical merge lands.  This uses the ordinary planner with the same
    durable-media protection set, changing only the exact canonical snapshot
    supplied by the caller.
    """

    canonical = config.normalized_file.resolve()
    if revision.path.resolve() != canonical or revision.text is None:
        raise AudioPlanError(
            "Prospective targeted audio needs a present revision of the canonical "
            "vocabulary collection."
        )
    protected_records = tuple(
        _all_durable_audio_records(config, status.deck_files(config), records)
    )
    return _plan_records(
        config,
        records,
        revision,
        _exact_ids(record_ids),
        words=words,
        examples=examples,
        force=force,
        chosen_provider=chosen_provider,
        word_provider=word_provider,
        sentence_provider=sentence_provider,
        protected_records=protected_records,
    )


def plan_corpus_audio(
    config: ProjectConfig,
    *,
    words: bool = False,
    examples: bool = False,
    force: bool = False,
    chosen_provider: str | None = None,
    word_provider: SpeechProvider | None = None,
    sentence_provider: SentenceProvider | None = None,
) -> AudioPlan:
    """Plan the CLI's explicit whole-corpus maintenance operation."""
    records, revision = load_records_snapshot(config.normalized_file.resolve())
    protected_records = tuple(
        _all_durable_audio_records(config, status.deck_files(config), records)
    )
    return _plan_records(
        config,
        records,
        revision,
        None,
        words=words,
        examples=examples,
        force=force,
        chosen_provider=chosen_provider,
        word_provider=word_provider,
        sentence_provider=sentence_provider,
        protected_records=protected_records,
    )


def plan_audio_records(
    config: ProjectConfig,
    records: Sequence[VocabularyRecord],
    revision: RecordsRevision,
    record_ids: Sequence[str] | None,
    *,
    words: bool = False,
    examples: bool = False,
    force: bool = False,
    chosen_provider: str | None = None,
    word_provider: SpeechProvider | None = None,
    sentence_provider: SentenceProvider | None = None,
    protected_records: Sequence[VocabularyRecord] = (),
    ledger_book: ledger.Ledger | None = None,
    owner_path: Path | None = None,
) -> AudioPlan:
    """Plan the CLI's already owner-locked record universe without reloading it."""
    exact_ids = None if record_ids is None else _exact_ids(record_ids)
    return _plan_records(
        config,
        records,
        revision,
        exact_ids,
        words=words,
        examples=examples,
        force=force,
        chosen_provider=chosen_provider,
        word_provider=word_provider,
        sentence_provider=sentence_provider,
        protected_records=protected_records,
        ledger_book=ledger_book,
        owner_path=owner_path,
    )


def plan_deck_audio(
    config: ProjectConfig,
    deck_path: Path,
    *,
    record_ids: Sequence[str] | None = None,
    words: bool = False,
    examples: bool = True,
    force: bool = False,
    chosen_provider: str | None = None,
    word_provider: SpeechProvider | None = None,
    sentence_provider: SentenceProvider | None = None,
) -> AudioPlan:
    """Plan the existing sentence-audio transaction for one rich drill deck."""
    target, known = _deck_audio_plan_scope(config, deck_path, words, examples)
    records = pattern_cards.drill_audio_records(target, config)
    revision = records_revision(target)
    protected = tuple(_all_durable_audio_records(config, known))
    return _plan_records(
        config,
        records,
        revision,
        None if record_ids is None else _exact_ids(record_ids),
        words=False,
        examples=True,
        force=force,
        chosen_provider=chosen_provider,
        word_provider=word_provider,
        sentence_provider=sentence_provider,
        protected_records=protected,
        owner_path=target,
    )


def plan_deck_audio_revision(
    config: ProjectConfig,
    deck_path: Path,
    revision: RecordsRevision,
    *,
    record_ids: Sequence[str] | None = None,
    words: bool = False,
    examples: bool = True,
    force: bool = False,
    chosen_provider: str | None = None,
    word_provider: SpeechProvider | None = None,
    sentence_provider: SentenceProvider | None = None,
) -> AudioPlan:
    """Plan deck audio as though one exact deck revision had already landed."""
    target, known = _deck_audio_plan_scope(config, deck_path, words, examples)
    records = pattern_cards.drill_audio_records_from_revision(
        target,
        config,
        revision,
    )
    protected = tuple(
        _all_durable_audio_records(
            config,
            known,
            deck_revisions={target: revision},
            drill_overrides={target: records},
        )
    )
    return _plan_records(
        config,
        records,
        revision,
        None if record_ids is None else _exact_ids(record_ids),
        words=False,
        examples=True,
        force=force,
        chosen_provider=chosen_provider,
        word_provider=word_provider,
        sentence_provider=sentence_provider,
        protected_records=protected,
        owner_path=target,
    )


def _deck_audio_plan_scope(
    config: ProjectConfig,
    deck_path: Path,
    words: bool,
    examples: bool,
) -> tuple[Path, list[Path]]:
    """Validate the shared deck-scoped clip scope for disk and revision plans.

    The deck is resolved first because the answer is the deck's, not the
    flags': which clip kinds a deck-scoped call may ask for is exactly what
    its capability row states.
    """
    target = deck_path.resolve()
    known = [path.resolve() for path in status.deck_files(config)]
    if target not in known:
        raise AudioPlanError(
            f"Deck audio target {target} is not a configured deck under "
            f"{config.deck_dir.resolve()}."
        )
    capability = deck_capabilities.deck_capability(target)
    named_kind = capability.kind or "vocabulary"
    if not capability.owns_drill_examples:
        raise AudioPlanError(
            f"Deck-authored audio needs a deck that owns drill examples; "
            f"{target} is a {named_kind} deck."
        )
    if words and not capability.synthesizes_word_audio:
        raise AudioPlanError(
            f"A {named_kind} deck synthesizes no word audio; word audio belongs "
            "to the canonical vocabulary records."
        )
    if not examples or not capability.synthesizes_example_audio:
        raise AudioPlanError(
            f"A {named_kind} deck's own audio is example audio; there is no "
            "other clip kind for a deck-scoped call to voice."
        )
    return target, known


def preflight_paid_deck_audio_plan(
    plan: AudioPlan,
    *,
    word_provider: SpeechProvider | None = None,
    sentence_provider: SentenceProvider | None = None,
) -> None:
    """Refuse unavailable paid clips without contacting a provider or writing.

    Only ``provider-required`` clips whose exact plan declares
    ``paid-network`` are checked. Current and recoverable clips need no fresh
    provider authority, while probing a local-network service would itself be
    contact and carries no billing risk. The only paid provider currently
    supported here is OpenAI Realtime, whose ``available_for`` method performs
    an exact journal/credential lookup without opening a connection.
    """
    for clip in plan.clips:
        if clip.state != "provider-required" or clip.provider.access != "paid-network":
            continue
        selected: SentenceProvider | None = (
            word_provider if clip.kind == "word" else sentence_provider
        )
        if selected is None:
            raise AudioPlanError(
                f"The exact {clip.provider.name} provider for {clip.kind} audio "
                "is required for paid-provider preflight."
            )
        try:
            provider = sentence_profile_for(selected, clip.record_id)
        except JankiError as exc:
            raise AudioPlanError(str(exc)) from exc
        if _provider_plan(provider) != clip.provider:
            raise AudioPlanError(
                f"The {clip.kind} audio provider changed after this plan was made; "
                "make and confirm a fresh audio plan."
            )
        if not isinstance(provider, openai_realtime.OpenAiRealtimeProvider):
            raise AudioPlanError(
                f"{clip.provider.name} has no supported no-contact paid-provider preflight."
            )
        try:
            available = provider.available_for(
                clip.request_input,
                forced_accent=clip.forced_accent,
                source_file=audio_cmd.audio_journal_source(
                    clip.record_id,
                    of=clip.kind,
                    target=clip.target,
                ),
                source_sha256=clip.content_fingerprint,
            )
        except JankiError as exc:
            raise AudioPlanError(str(exc)) from exc
        if not available:
            raise AudioPlanError(f"{provider.name}: {provider.launch_hint}")


def _paid_dispatch_adapter(
    plan: AudioPlan,
    callback: Callable[[PaidAudioDispatch], None] | None,
) -> Callable[[audio_cmd.AudioPaidDispatch], None] | None:
    if callback is None:
        return None

    def reserve(dispatched: audio_cmd.AudioPaidDispatch) -> None:
        provider = _provider_plan(dispatched.provider)
        matches = [
            clip
            for clip in plan.clips
            if clip.record_id == dispatched.record_id
            and clip.kind == dispatched.kind
            and clip.target == dispatched.target
            and clip.request_input == dispatched.request_input
            and clip.forced_accent == dispatched.forced_accent
            and clip.content_fingerprint == dispatched.content_fingerprint
            and clip.provider == provider
        ]
        if (
            len(matches) != 1
            or matches[0].state != "provider-required"
            or matches[0].provider.access != "paid-network"
        ):
            raise AudioPlanError(
                "Paid audio dispatch no longer matches one provider-required "
                "clip in the fresh plan."
            )
        callback(PaidAudioDispatch(dispatched.operation_id, matches[0]))

    return reserve


def _emit_progress(progress: AudioProgress | None, phase: AudioPhase) -> None:
    """Progress is advisory and must never interrupt the durable transaction."""
    if progress is None:
        return
    try:
        progress(phase)
    except Exception:
        # A disconnected browser or failed status sink cannot be allowed to
        # strand paid bytes between the WAL and their canonical commit.
        return


def _save_audio_ledger(book: ledger.Ledger) -> ledger.LedgerError | None:
    """Return a final ledger failure after records/media already landed."""
    try:
        book.save()
    except ledger.LedgerError as exc:
        return exc
    return None


def _persist_audio_wal(
    book: ledger.Ledger,
    keys: Sequence[str],
    *,
    replace: bool = False,
    expected: Mapping[str, Mapping[str, Any]] | None = None,
) -> ledger.Ledger:
    """Durably merge only this call's additive paid-audio WAL rows."""
    book.merge_pending_audio(keys, replace=replace, expected=expected)
    return book


def _audio_wal_persister(force: bool) -> audio_cmd.PersistPendingAudio:
    """The durable write-ahead seam one audio run hands its writer.

    A predecessor arrives only from the writer's one non-additive update —
    recovery making a still-live reply's attribution durable on the row it just
    read — and is forwarded as that row's compare-and-swap, so this run's own
    second write to a row is not mistaken for another command's. ``--force``
    stays what it already was: the owner's explicit decision to replace
    whatever is on disk.
    """

    def persist(
        current: ledger.Ledger, key: str, expected: Mapping[str, Any] | None
    ) -> None:
        _persist_audio_wal(
            current,
            [key],
            replace=force,
            expected=None if expected is None else {key: expected},
        )

    return persist


def _audio_deck_source_paths(deck_paths: Sequence[Path]) -> set[Path]:
    """Record files that current deck definitions make durable audio owners."""
    sources: set[Path] = set()
    for deck_path in deck_paths:
        raw = load_structured(deck_path)
        if not isinstance(raw, Mapping):
            raise DataError(f"Deck file must contain a mapping: {deck_path}")
        deck_config = raw.get("deck") or {}
        if not isinstance(deck_config, Mapping):
            raise DataError(f"The deck section must be a mapping: {deck_path}")
        source = deck_config.get("source")
        if source:
            sources.add((deck_path.parent / str(source)).resolve())
    return sources


def _assert_audio_owners_current(
    revisions: Mapping[Path, RecordsRevision],
    *,
    config: ProjectConfig,
    deck_paths: Sequence[Path],
    source_paths: set[Path],
) -> None:
    current_decks = [path.resolve() for path in status.deck_files(config)]
    expected_decks = [path.resolve() for path in deck_paths]
    if current_decks != expected_decks:
        raise DataError(
            "The deck file set changed after the audio owner census; refusing "
            "to publish or prune media until the command is re-run."
        )
    if _audio_deck_source_paths(current_decks) != source_paths:
        raise DataError(
            "A deck source dependency changed after the audio owner census; "
            "refusing to publish or prune media until the command is re-run."
        )
    for path, expected in revisions.items():
        if records_revision(path).text != expected.text:
            raise DataError(
                f"Audio owner file {path} changed after its locked snapshot; "
                "refusing to publish or prune media from stale references. Re-run."
            )


def _cleanup_unclaimed_audio_stages(
    config: ProjectConfig,
    book: ledger.Ledger,
    records: Sequence[VocabularyRecord],
    *,
    chosen: str | None,
    word_provider: SpeechProvider | None = None,
) -> list[str]:
    """Clean no-row stages only after protecting every current exact request."""
    if not records:
        current_keys: set[str] = set()
    else:
        try:
            words = word_provider or resolve_word_provider(config, chosen)
            sentences = resolve_sentence_provider(config, chosen, words)
            prepared = audio_cmd.prepare_example_audio_profiles(
                records,
                sentence_provider=sentences,
            )
            current_keys = audio_cmd.current_pending_audio_keys(
                records,
                book=book,
                word_provider=words,
                prepared_examples=prepared,
            )
        except JankiError as exc:
            return [
                "could not prove which unregistered pending stages remain "
                f"current, so none were removed: {exc}"
            ]
    return audio_cmd.cleanup_unclaimed_pending_stages(
        book,
        config.media_dir.resolve() / audio_cmd.AUDIO_SUBDIR,
        current_keys=current_keys,
    )


def _selected_pending(book: ledger.Ledger, selected_slots: set[tuple[str, str]]) -> bool:
    return any(
        isinstance(entry, Mapping)
        and (str(entry.get("record_id") or ""), str(entry.get("of") or "")) in selected_slots
        for entry in book.pending_audio.values()
    )


def execute_targeted_audio(
    config: ProjectConfig,
    record_ids: Sequence[str],
    *,
    words: bool = False,
    examples: bool = False,
    expected_fingerprint: str | None = None,
    force: bool = False,
    prune: bool = False,
    chosen_provider: str | None = None,
    progress: AudioProgress | None = None,
    word_provider: SpeechProvider | None = None,
    sentence_provider: SentenceProvider | None = None,
) -> AudioExecutionOutcome:
    """Execute audio for one exact nonempty id set; it can never widen.

    A rendered surface supplies ``expected_fingerprint``. ``None`` is reserved
    for the immediate CLI path, whose first plan is already made under this
    transaction's operation and repository-owner locks.
    """
    return _execute_audio(
        config,
        _exact_ids(record_ids),
        deck_path=None,
        words=words,
        examples=examples,
        expected_fingerprint=expected_fingerprint,
        force=force,
        prune=prune,
        chosen_provider=chosen_provider,
        progress=progress,
        word_provider=word_provider,
        sentence_provider=sentence_provider,
    )


def execute_targeted_audio_locked(
    config: ProjectConfig,
    record_ids: Sequence[str],
    *,
    words: bool = False,
    examples: bool = False,
    expected_fingerprint: str | None = None,
    force: bool = False,
    prune: bool = False,
    chosen_provider: str | None = None,
    progress: AudioProgress | None = None,
    word_provider: SpeechProvider | None = None,
    sentence_provider: SentenceProvider | None = None,
    before_paid_dispatch: Callable[[PaidAudioDispatch], None] | None = None,
) -> AudioExecutionOutcome:
    """Execute targeted audio while the caller owns the audio-operation lock."""

    return _execute_audio_locked(
        config,
        _exact_ids(record_ids),
        deck_path=None,
        words=words,
        examples=examples,
        expected_fingerprint=expected_fingerprint,
        force=force,
        prune=prune,
        chosen_provider=chosen_provider,
        progress=progress,
        word_provider=word_provider,
        sentence_provider=sentence_provider,
        before_paid_dispatch=before_paid_dispatch,
    )


def execute_corpus_audio(
    config: ProjectConfig,
    *,
    words: bool = False,
    examples: bool = False,
    expected_fingerprint: str | None = None,
    force: bool = False,
    prune: bool = False,
    chosen_provider: str | None = None,
    progress: AudioProgress | None = None,
    word_provider: SpeechProvider | None = None,
    sentence_provider: SentenceProvider | None = None,
) -> AudioExecutionOutcome:
    """Execute the CLI's explicit whole-corpus audio maintenance operation."""
    return _execute_audio(
        config,
        None,
        deck_path=None,
        words=words,
        examples=examples,
        expected_fingerprint=expected_fingerprint,
        force=force,
        prune=prune,
        chosen_provider=chosen_provider,
        progress=progress,
        word_provider=word_provider,
        sentence_provider=sentence_provider,
    )


def execute_deck_audio(
    config: ProjectConfig,
    deck_path: Path,
    *,
    record_ids: Sequence[str] | None = None,
    words: bool = False,
    examples: bool = True,
    expected_fingerprint: str | None = None,
    force: bool = False,
    prune: bool = False,
    chosen_provider: str | None = None,
    progress: AudioProgress | None = None,
    word_provider: SpeechProvider | None = None,
    sentence_provider: SentenceProvider | None = None,
) -> AudioExecutionOutcome:
    """Execute the shared transaction against one rich drill deck owner."""
    _validate_deck_audio_execution(words=words, examples=examples, prune=prune)
    return _execute_audio(
        config,
        None if record_ids is None else _exact_ids(record_ids),
        deck_path=deck_path.resolve(),
        words=False,
        examples=True,
        expected_fingerprint=expected_fingerprint,
        force=force,
        prune=False,
        chosen_provider=chosen_provider,
        progress=progress,
        word_provider=word_provider,
        sentence_provider=sentence_provider,
    )


def execute_deck_audio_locked(
    config: ProjectConfig,
    deck_path: Path,
    *,
    record_ids: Sequence[str] | None = None,
    words: bool = False,
    examples: bool = True,
    expected_fingerprint: str | None = None,
    force: bool = False,
    prune: bool = False,
    chosen_provider: str | None = None,
    progress: AudioProgress | None = None,
    word_provider: SpeechProvider | None = None,
    sentence_provider: SentenceProvider | None = None,
    before_paid_dispatch: Callable[[PaidAudioDispatch], None] | None = None,
) -> AudioExecutionOutcome:
    """Execute deck audio while the caller owns ``.janki-audio-operation``."""
    _validate_deck_audio_execution(words=words, examples=examples, prune=prune)
    return _execute_audio_locked(
        config,
        None if record_ids is None else _exact_ids(record_ids),
        deck_path=deck_path.resolve(),
        words=False,
        examples=True,
        expected_fingerprint=expected_fingerprint,
        force=force,
        prune=False,
        chosen_provider=chosen_provider,
        progress=progress,
        word_provider=word_provider,
        sentence_provider=sentence_provider,
        before_paid_dispatch=before_paid_dispatch,
    )


def _validate_deck_audio_execution(
    *,
    words: bool,
    examples: bool,
    prune: bool,
) -> None:
    """Keep locked and ordinary deck execution on one narrow contract."""
    if words or not examples:
        raise AudioPlanError(
            "Deck-authored drill audio supports --examples only; word audio "
            "belongs to the canonical vocabulary records."
        )
    if prune:
        raise AudioPlanError(
            "Run corpus 'janki audio --examples --prune' for global media cleanup; "
            "a deck-scoped call only voices that deck."
        )


def _execute_audio(
    config: ProjectConfig,
    record_ids: tuple[str, ...] | None,
    *,
    deck_path: Path | None,
    words: bool,
    examples: bool,
    expected_fingerprint: str | None,
    force: bool,
    prune: bool,
    chosen_provider: str | None,
    progress: AudioProgress | None,
    word_provider: SpeechProvider | None,
    sentence_provider: SentenceProvider | None,
) -> AudioExecutionOutcome:
    """Take the operation lock before any repository or ledger snapshot."""
    with exclusive_path_lock(config.root / ".janki-audio-operation"):
        return _execute_audio_locked(
            config,
            record_ids,
            deck_path=deck_path,
            words=words,
            examples=examples,
            expected_fingerprint=expected_fingerprint,
            force=force,
            prune=prune,
            chosen_provider=chosen_provider,
            progress=progress,
            word_provider=word_provider,
            sentence_provider=sentence_provider,
            before_paid_dispatch=None,
        )


def _execute_audio_locked(
    config: ProjectConfig,
    record_ids: tuple[str, ...] | None,
    *,
    deck_path: Path | None,
    words: bool,
    examples: bool,
    expected_fingerprint: str | None,
    force: bool,
    prune: bool,
    chosen_provider: str | None,
    progress: AudioProgress | None,
    word_provider: SpeechProvider | None,
    sentence_provider: SentenceProvider | None,
    before_paid_dispatch: Callable[[PaidAudioDispatch], None] | None,
) -> AudioExecutionOutcome:
    """Lock every current record owner before taking its exact snapshot."""
    _emit_progress(progress, "Preparing audio")
    deck_paths = status.deck_files(config)
    if deck_path is not None and deck_path.resolve() not in {path.resolve() for path in deck_paths}:
        raise AudioPlanError(
            f"Deck audio target {deck_path.resolve()} is not a configured deck "
            f"under {config.deck_dir.resolve()}."
        )
    source_paths = _audio_deck_source_paths(deck_paths)
    owner_paths = sorted(
        {
            config.normalized_file.resolve(),
            *(path.resolve() for path in deck_paths),
            *source_paths,
        },
        key=lambda path: str(path),
    )
    with ExitStack() as locks:
        for path in owner_paths:
            locks.enter_context(exclusive_path_lock(path))
        owner_revisions = {path: records_revision(path) for path in owner_paths}
        _assert_audio_owners_current(
            owner_revisions,
            config=config,
            deck_paths=deck_paths,
            source_paths=source_paths,
        )
        return _execute_audio_owner_locked(
            config,
            record_ids,
            deck_paths,
            source_paths,
            owner_revisions,
            deck_path=deck_path,
            words=words,
            examples=examples,
            expected_fingerprint=expected_fingerprint,
            force=force,
            prune=prune,
            chosen_provider=chosen_provider,
            progress=progress,
            word_provider=word_provider,
            sentence_provider=sentence_provider,
            before_paid_dispatch=before_paid_dispatch,
        )


def _execute_audio_owner_locked(
    config: ProjectConfig,
    record_ids: tuple[str, ...] | None,
    deck_paths: Sequence[Path],
    source_paths: set[Path],
    owner_revisions: dict[Path, RecordsRevision],
    *,
    deck_path: Path | None,
    words: bool,
    examples: bool,
    expected_fingerprint: str | None,
    force: bool,
    prune: bool,
    chosen_provider: str | None,
    progress: AudioProgress | None,
    word_provider: SpeechProvider | None,
    sentence_provider: SentenceProvider | None,
    before_paid_dispatch: Callable[[PaidAudioDispatch], None] | None,
) -> AudioExecutionOutcome:
    """Run the provider/WAL/record/media/ledger transaction under owner locks."""
    normalized_path = config.normalized_file.resolve()
    normalized_records = load_records(normalized_path) if normalized_path.exists() else []
    output_path = deck_path.resolve() if deck_path is not None else normalized_path
    output_revision = owner_revisions[output_path]
    records = (
        pattern_cards.drill_audio_records(output_path, config)
        if deck_path is not None
        else normalized_records
    )
    output_dir = config.media_dir.resolve() / audio_cmd.AUDIO_SUBDIR
    if not records and record_ids is not None:
        raise AudioPlanError(f"No audio record has id {record_ids[0]!r}.")
    if not records and expected_fingerprint is not None:
        raise AudioPlanError(
            "The audio scope changed after it was displayed; nothing was sent. "
            "Reload and review the fresh clip plan."
        )
    if not records and not prune:
        return AudioExecutionOutcome(
            state="no-records",
            plan=None,
            output_dir=output_dir,
            no_records=True,
            prune_requested=False,
        )

    protected_records = _all_durable_audio_records(config, deck_paths, normalized_records)
    media_dir = config.media_dir.resolve()
    warnings: list[str] = []
    book: ledger.Ledger | None = None
    words_engine: SpeechProvider | None = None
    sentences_engine: SentenceProvider | None = None
    plan: AudioPlan | None = None
    if expected_fingerprint is not None:
        words_engine = word_provider or resolve_word_provider(config, chosen_provider)
        sentences_engine = sentence_provider or (
            resolve_sentence_provider(config, chosen_provider, words_engine)
            if examples
            else words_engine
        )
        book = ledger.load(config.ledger_file)
        plan = plan_audio_records(
            config,
            records,
            output_revision,
            record_ids,
            words=words,
            examples=examples,
            force=force,
            chosen_provider=chosen_provider,
            word_provider=words_engine,
            sentence_provider=sentences_engine,
            protected_records=protected_records,
            ledger_book=book,
            owner_path=output_path,
        )
        if plan.fingerprint != expected_fingerprint:
            raise AudioPlanError(
                "The audio scope changed after it was displayed; nothing was sent. "
                "Reload and review the fresh clip plan."
            )
    if prune:
        # Missing-record WAL rows can never match a future request. Retire them
        # before provider resolution so recovery does not depend on an engine.
        book = book or ledger.load(config.ledger_file)
        book.save()
        deleted_pending = book.discard_pending_audio_for_missing_records(
            record.id for record in protected_records
        )
        if deleted_pending:
            book.save()
            warnings.extend(audio_cmd.cleanup_pending_stages(deleted_pending, output_dir))
        if not records:
            try:
                removed = audio_cmd.prune_unreferenced(
                    protected_records,
                    media_dir,
                    book,
                    persist=lambda current: current.save(),
                    assert_current=lambda: _assert_audio_owners_current(
                        owner_revisions,
                        config=config,
                        deck_paths=deck_paths,
                        source_paths=source_paths,
                    ),
                )
            except audio_cmd.PruneError as exc:
                return AudioExecutionOutcome(
                    state="prune-incomplete",
                    plan=None,
                    output_dir=output_dir,
                    pruned_paths=tuple(exc.removed),
                    warnings=tuple(warnings),
                    prune_error=str(exc),
                    no_records=True,
                    prune_requested=True,
                )
            _assert_audio_owners_current(
                owner_revisions,
                config=config,
                deck_paths=deck_paths,
                source_paths=source_paths,
            )
            warnings.extend(
                _cleanup_unclaimed_audio_stages(
                    config,
                    book,
                    protected_records,
                    chosen=chosen_provider,
                )
            )
            return AudioExecutionOutcome(
                state="no-records",
                plan=None,
                output_dir=output_dir,
                pruned_paths=tuple(removed),
                warnings=tuple(warnings),
                no_records=True,
                prune_requested=True,
            )

    words_engine = words_engine or word_provider or resolve_word_provider(config, chosen_provider)
    # A words-only run must not fail because an unused sentence provider is
    # unavailable or misconfigured.
    sentences_engine = (
        sentences_engine
        or sentence_provider
        or (
            resolve_sentence_provider(config, chosen_provider, words_engine)
            if examples
            else words_engine
        )
    )
    if book is None:
        book = ledger.load(config.ledger_file)
    if not prune:
        # A static read-only failure is knowable before a paid provider call.
        book.save()

    if plan is None:
        plan = plan_audio_records(
            config,
            records,
            output_revision,
            record_ids,
            words=words,
            examples=examples,
            force=force,
            chosen_provider=chosen_provider,
            word_provider=words_engine,
            sentence_provider=sentences_engine,
            protected_records=protected_records,
            ledger_book=book,
            owner_path=output_path,
        )

    _emit_progress(progress, "Creating audio")
    result = audio_cmd.generate_audio(
        records,
        provider=words_engine,
        sentence_provider=sentences_engine,
        book=book,
        media_dir=media_dir,
        words=words,
        examples=examples,
        ids=plan.record_ids,
        force=force,
        protected_records=protected_records,
        stage_only=True,
        persist_pending=_audio_wal_persister(force),
        before_paid_dispatch=_paid_dispatch_adapter(plan, before_paid_dispatch),
    )
    generation_stopped = bool(result.stopped_by)
    record_references_written = False
    media_published = False
    ledger_committed = False
    selected_slots = {
        (record_id, kind)
        for record_id in plan.record_ids
        for kind, enabled in (("word", words), ("example", examples))
        if enabled
    }
    stale_selected_pending = any(
        isinstance(entry, Mapping)
        and (str(entry.get("record_id") or ""), str(entry.get("of") or "")) in selected_slots
        and key not in result.pending_keys
        for key, entry in book.pending_audio.items()
    )
    warnings.extend(result.warnings)
    _emit_progress(progress, "Saving audio")

    if (
        result.pending_keys
        or stale_selected_pending
        or result.file_count
        or result.records != records
    ):
        try:
            if deck_path is None:
                save_records_json_locked(output_path, result.records, expected=output_revision)
            else:
                pattern_cards.save_drill_audio_records(
                    output_path, result.records, expected=output_revision
                )
            record_references_written = result.records != records
            owner_revisions[output_path] = records_revision(output_path)
        except DataError as exc:
            return AudioExecutionOutcome(
                state="records-stale",
                plan=plan,
                output_dir=output_dir,
                file_count=result.file_count,
                written_record_ids=tuple(result.written),
                up_to_date=result.up_to_date,
                warnings=tuple(warnings),
                guessed_accent=tuple(result.guessed_accent),
                no_reading=tuple(result.no_reading),
                stopped_by=(
                    f"{exc} Paid audio from this run remains staged in "
                    "pending_audio; re-run the same command to adopt an exact "
                    "request/profile without another provider call. Canonical "
                    "media and its audio ledger entries were not changed."
                ),
                pending_recovery=bool(result.pending_keys) or stale_selected_pending,
                prune_requested=prune,
            )

    ledger_error: ledger.LedgerError | None = None
    committed_pending: list[dict[str, object]] = []
    ledger_changed = False
    if result.pending_keys:
        try:
            audio_cmd.promote_pending_audio(
                book,
                result.pending_keys,
                output_dir,
                assert_current=lambda: _assert_audio_owners_current(
                    owner_revisions,
                    config=config,
                    deck_paths=deck_paths,
                    source_paths=source_paths,
                ),
            )
            media_published = True
            committed_pending = audio_cmd.commit_promoted_audio(
                book, result.pending_keys, result.records
            )
            ledger_changed = True
        except JankiError as exc:
            result.stopped_by = result.stopped_by or str(exc)
    if not result.stopped_by:
        retired = book.discard_pending_audio_for_slots(selected_slots)
        if retired:
            committed_pending.extend(retired)
            ledger_changed = True
    if ledger_changed:
        ledger_error = _save_audio_ledger(book)
        if ledger_error is None:
            ledger_committed = True
            warnings.extend(audio_cmd.cleanup_pending_stages(committed_pending, output_dir))

    _emit_progress(progress, "Cleaning up audio")
    prune_error = ""
    removed: list[Path] = []
    if prune and ledger_error is None:
        prune_protected_records = _all_durable_audio_records(
            config,
            deck_paths,
            result.records if deck_path is None else normalized_records,
        )
        deleted_pending = book.discard_pending_audio_for_missing_records(
            record.id for record in prune_protected_records
        )
        if deleted_pending:
            ledger_error = _save_audio_ledger(book)
            if ledger_error is None:
                warnings.extend(audio_cmd.cleanup_pending_stages(deleted_pending, output_dir))
        if ledger_error is None:
            try:
                removed = audio_cmd.prune_unreferenced(
                    prune_protected_records,
                    media_dir,
                    book,
                    persist=lambda current: current.save(),
                    assert_current=lambda: _assert_audio_owners_current(
                        owner_revisions,
                        config=config,
                        deck_paths=deck_paths,
                        source_paths=source_paths,
                    ),
                )
            except audio_cmd.PruneError as exc:
                removed = exc.removed
                prune_error = str(exc)

    if not result.stopped_by and not prune_error and ledger_error is None:
        current_durable_records = _all_durable_audio_records(
            config,
            deck_paths,
            result.records if deck_path is None else normalized_records,
        )
        _assert_audio_owners_current(
            owner_revisions,
            config=config,
            deck_paths=deck_paths,
            source_paths=source_paths,
        )
        warnings.extend(
            _cleanup_unclaimed_audio_stages(
                config,
                book,
                current_durable_records,
                chosen=chosen_provider,
                word_provider=words_engine,
            )
        )

    if generation_stopped:
        state: AudioExecutionState = "generation-stopped"
    elif result.stopped_by:
        state = "finalization-stopped"
    elif prune_error:
        state = "prune-incomplete"
    elif ledger_error is not None:
        state = "ledger-incomplete"
    else:
        state = "complete"
    selected_pending = _selected_pending(book, selected_slots)
    # A failed save leaves the prior on-disk WAL intact even though the
    # in-memory book has already converted or retired those rows.
    pending_recovery = ledger_changed if ledger_error is not None else selected_pending
    return AudioExecutionOutcome(
        state=state,
        plan=plan,
        output_dir=output_dir,
        file_count=result.file_count,
        written_record_ids=tuple(result.written),
        up_to_date=result.up_to_date,
        pruned_paths=tuple(removed),
        warnings=tuple(warnings),
        guessed_accent=tuple(result.guessed_accent),
        no_reading=tuple(result.no_reading),
        stopped_by=result.stopped_by or None,
        prune_error=prune_error or None,
        ledger_error=str(ledger_error) if ledger_error is not None else None,
        pending_recovery=pending_recovery,
        record_references_written=record_references_written,
        media_published=media_published,
        ledger_committed=ledger_committed,
        prune_requested=prune,
    )


# --- the durable audio-completion proof -------------------------------------
#
# `AudioExecutionOutcome` is a return value. A process that dies between the
# last ledger write and whatever was going to read that value loses it, and a
# package built on "the run said it worked" is built on nothing. What survives
# is the repository: the canonical references, the media bytes, the ledger's
# own currency answer for the exact request *and profile*, and — for a clip
# this finish paid for — the attempt the paid writer recorded into that same
# ledger entry before its successful call was forgotten. This
# section turns those durable facts into one immutable, revalidatable proof so
# the packager can replace the prospective `None` media hashes its authority
# bound before the WAV bytes could exist (contracts §7.10, §7.11).
#
# The proof is evidence about *artifacts*. It is not job authority and not
# spending authority: a coordinator still binds it to its own exact phase
# receipt and owner authority, and a shared canonical/root/media scope does not
# by itself prove two jobs are one.


AUDIO_COMPLETION_SCHEMA = "janki-audio-completion-proof-v1"

#: Forward-only, request-bound clip evolution. A confirmed clip may improve
#: towards `current`; nothing may regress, and nothing may widen.
CLIP_EVOLUTION: Mapping[str, frozenset[str]] = {
    "current": frozenset({"current"}),
    "recoverable": frozenset({"recoverable", "current"}),
    "provider-required": frozenset({"provider-required", "recoverable", "current"}),
}

ProvenSlotOrigin = Literal["reused", "recovered", "synthesized"]

_ORIGIN_FOR_INITIAL_STATE: Mapping[str, ProvenSlotOrigin] = {
    "current": "reused",
    "recoverable": "recovered",
    "provider-required": "synthesized",
}

_PROVEN_PAID_OPERATION_KIND = "audio-realtime"


class AudioProofError(JankiError):
    """A claimed audio completion is not proven by durable repository state."""


def _proof_sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_wire(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _one_media_component(target: str) -> str:
    pure = PurePosixPath(target)
    if (
        not target
        or pure.is_absolute()
        or len(pure.parts) != 1
        or pure.name != target
        or target in {".", ".."}
    ):
        raise AudioProofError(f"An audio target must be one media filename: {target!r}")
    return target


def media_target_path(config: ProjectConfig, target: str) -> Path:
    """The absolute canonical media path one clip target names."""

    return _media_target(config.media_dir.resolve(), target)


def _media_target(media_dir: Path, target: str) -> Path:
    return media_dir / audio_cmd.AUDIO_SUBDIR / _one_media_component(target)


def _proof_relative(root: Path, path: Path, *, label: str) -> str:
    target = Path(path).absolute()
    if target == root:
        return "."
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise AudioProofError(
            f"Audio completion proof {label} escapes the repository: {target}"
        ) from exc
    return relative.as_posix()


def _proof_path(root: Path, value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise AudioProofError(f"Audio completion proof {label} path is malformed.")
    if value == ".":
        return root
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != value:
        raise AudioProofError(
            f"Audio completion proof {label} path is not repository-relative."
        )
    return (root / Path(*pure.parts)).absolute()


@dataclass(frozen=True, slots=True)
class ExpectedAudioSlot:
    """One record or example slot the finish promised to voice.

    Counted from record fields, never from ``AudioPlan.clips``: an empty clip
    list must not be able to satisfy a promise, and several slots may share one
    identity-addressed clip.
    """

    record_id: str
    kind: Literal["word", "example"]
    #: Index into ``record.examples``; ``None`` for a word slot.
    position: int | None


def expected_audio_slots(
    records: Sequence[VocabularyRecord],
    record_ids: Sequence[str],
    *,
    words: bool,
    examples: bool,
) -> tuple[ExpectedAudioSlot, ...]:
    """The independent slot census for one exact selection.

    Exactly the conditions ``audio_cmd.audio_clip_requirements`` itself uses —
    a word slot where ``words`` and the record has a reading, an example slot
    for every nonblank ``example.japanese`` where ``examples``. It reads no
    Japanese and makes no judgement about a sentence; it counts fields.
    """

    by_id: dict[str, VocabularyRecord] = {}
    for record in records:
        by_id.setdefault(record.id, record)
    slots: list[ExpectedAudioSlot] = []
    for record_id in record_ids:
        record = by_id.get(record_id)
        if record is None:
            raise AudioProofError(f"No audio record has id {record_id!r}.")
        if words and record.reading:
            slots.append(ExpectedAudioSlot(record_id, "word", None))
        if not examples:
            continue
        for position, example in enumerate(record.examples):
            if example.japanese:
                slots.append(ExpectedAudioSlot(record_id, "example", position))
    return tuple(slots)


@dataclass(frozen=True, slots=True)
class ConfirmedAudioClip:
    """One clip exactly as the owner's confirmation enumerated it."""

    record_id: str
    kind: Literal["word", "example"]
    target: str
    request_input: str
    forced_accent: bool
    content_fingerprint: str
    provider: AudioProviderPlan
    initial_state: audio_cmd.AudioClipState
    #: The exact bytes an already-``current`` clip had at confirmation.
    initial_media_sha256: str | None
    initial_recovery_key: str | None
    initial_recovery_sha256: str | None
    initial_recovery_source: audio_cmd.AudioRecoverySource | None


@dataclass(frozen=True, slots=True)
class ConfirmedAudioPlan:
    """The immutable confirmed audio enumeration and its configuration.

    Everything ``_validate_audio_evolution`` compares, frozen before the
    confirmation: the scope, the inclusion flags, both provider profiles, and
    per clip its request identity, provider and initial disposition with the
    fixed hashes that disposition implies. The proof is checked against this
    whole value, not against a state map: a matching current clip in a
    different voice is a different clip.
    """

    repository_root: Path
    canonical_path: Path
    ledger_path: Path
    media_dir: Path
    record_ids: tuple[str, ...]
    targeted: bool
    words: bool
    examples: bool
    force: bool
    word_provider: AudioProviderPlan | None
    example_provider: AudioProviderPlan | None
    clips: tuple[ConfirmedAudioClip, ...]

    def __post_init__(self) -> None:
        targets = [clip.target for clip in self.clips]
        if len(set(targets)) != len(targets):
            raise ValueError("A confirmed audio enumeration repeats a clip target")

    def clip_for(self, target: str) -> ConfirmedAudioClip | None:
        for clip in self.clips:
            if clip.target == target:
                return clip
        return None

    def to_wire(self) -> dict[str, Any]:
        root = self.repository_root
        return {
            "repository_root": _proof_relative(root, root, label="root"),
            "canonical_path": _proof_relative(
                root, self.canonical_path, label="audio owner"
            ),
            "ledger_path": _proof_relative(root, self.ledger_path, label="audio ledger"),
            "media_dir": _proof_relative(root, self.media_dir, label="media directory"),
            "record_ids": list(self.record_ids),
            "targeted": self.targeted,
            "words": self.words,
            "examples": self.examples,
            "force": self.force,
            "word_provider": provider_plan_wire(self.word_provider),
            "example_provider": provider_plan_wire(self.example_provider),
            "clips": [
                {
                    "record_id": clip.record_id,
                    "kind": clip.kind,
                    "target": clip.target,
                    "request_input": clip.request_input,
                    "forced_accent": clip.forced_accent,
                    "content_fingerprint": clip.content_fingerprint,
                    "provider": provider_plan_wire(clip.provider),
                    "initial_state": clip.initial_state,
                    "initial_media_sha256": clip.initial_media_sha256,
                    "initial_recovery_key": clip.initial_recovery_key,
                    "initial_recovery_sha256": clip.initial_recovery_sha256,
                    "initial_recovery_source": clip.initial_recovery_source,
                }
                for clip in self.clips
            ],
        }

    @classmethod
    def from_wire(
        cls, config: ProjectConfig, raw: Mapping[str, Any]
    ) -> ConfirmedAudioPlan:
        root = config.root.resolve()
        clips = raw.get("clips")
        if not isinstance(clips, list):
            raise AudioProofError("A confirmed audio enumeration is malformed.")
        return cls(
            repository_root=_proof_path(root, raw.get("repository_root"), label="root"),
            canonical_path=_proof_path(
                root, raw.get("canonical_path"), label="audio owner"
            ),
            ledger_path=_proof_path(root, raw.get("ledger_path"), label="audio ledger"),
            media_dir=_proof_path(root, raw.get("media_dir"), label="media directory"),
            record_ids=tuple(_wire_text_list(raw.get("record_ids"), "record ids")),
            targeted=_wire_bool(raw.get("targeted"), "targeted"),
            words=_wire_bool(raw.get("words"), "words"),
            examples=_wire_bool(raw.get("examples"), "examples"),
            force=_wire_bool(raw.get("force"), "force"),
            word_provider=_provider_from_wire(raw.get("word_provider")),
            example_provider=_provider_from_wire(raw.get("example_provider")),
            clips=tuple(_confirmed_clip_from_wire(item) for item in clips),
        )


def _wire_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise AudioProofError(f"Audio completion {label} must be a boolean.")
    return value


def _wire_text(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise AudioProofError(f"Audio completion {label} must be text.")
    return value


def _wire_text_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise AudioProofError(f"Audio completion {label} must be a list of text.")
    return [str(item) for item in value]


def _wire_sha(value: object, label: str) -> str:
    if not _is_sha256(value):
        raise AudioProofError(f"Audio completion {label} must be a lowercase SHA-256.")
    return str(value)


def _wire_kind(value: object) -> Literal["word", "example"]:
    if value not in {"word", "example"}:
        raise AudioProofError("An audio slot kind must be 'word' or 'example'.")
    return "word" if value == "word" else "example"


def _provider_from_wire(raw: object) -> AudioProviderPlan | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise AudioProofError("An audio provider profile is malformed.")
    expected = {
        "name",
        "access",
        "voice",
        "speed",
        "suffix",
        "destination",
        "transport",
        "settings",
    }
    if set(raw) != expected:
        raise AudioProofError("An audio provider profile is malformed.")
    access = raw.get("access")
    if access not in {"local-network", "paid-network"}:
        raise AudioProofError("An audio provider access class is unknown.")
    voice = raw.get("voice")
    if isinstance(voice, bool) or not isinstance(voice, int | str):
        raise AudioProofError("An audio provider voice is malformed.")
    speed = raw.get("speed")
    if isinstance(speed, bool) or not isinstance(speed, int | float):
        raise AudioProofError("An audio provider speed is malformed.")
    settings = raw.get("settings")
    if not isinstance(settings, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in settings.items()
    ):
        raise AudioProofError("An audio provider settings map is malformed.")
    return AudioProviderPlan(
        name=_wire_text(raw.get("name"), "provider name"),
        access="paid-network" if access == "paid-network" else "local-network",
        voice=voice,
        speed=float(speed),
        suffix=_wire_text(raw.get("suffix"), "provider suffix"),
        destination=_wire_text(raw.get("destination"), "provider destination"),
        transport=_wire_text(raw.get("transport"), "provider transport"),
        settings=dict(settings),
    )


def _confirmed_clip_from_wire(raw: object) -> ConfirmedAudioClip:
    if not isinstance(raw, Mapping):
        raise AudioProofError("A confirmed audio clip is malformed.")
    initial = raw.get("initial_state")
    if initial not in CLIP_EVOLUTION:
        raise AudioProofError("A confirmed audio clip carries an unknown disposition.")
    provider = _provider_from_wire(raw.get("provider"))
    if provider is None:
        raise AudioProofError("A confirmed audio clip needs its provider profile.")
    media_sha = raw.get("initial_media_sha256")
    recovery_sha = raw.get("initial_recovery_sha256")
    return ConfirmedAudioClip(
        record_id=_wire_text(raw.get("record_id"), "record id"),
        kind=_wire_kind(raw.get("kind")),
        target=_one_media_component(_wire_text(raw.get("target"), "clip target")),
        request_input=_wire_text(raw.get("request_input"), "request input"),
        forced_accent=_wire_bool(raw.get("forced_accent"), "forced accent"),
        content_fingerprint=_wire_sha(
            raw.get("content_fingerprint"), "content fingerprint"
        ),
        provider=provider,
        initial_state=initial,  # type: ignore[arg-type]
        initial_media_sha256=(
            None if media_sha is None else _wire_sha(media_sha, "initial media hash")
        ),
        initial_recovery_key=(
            None
            if raw.get("initial_recovery_key") is None
            else _wire_text(raw.get("initial_recovery_key"), "recovery key")
        ),
        initial_recovery_sha256=(
            None
            if recovery_sha is None
            else _wire_sha(recovery_sha, "initial recovery hash")
        ),
        initial_recovery_source=raw.get("initial_recovery_source"),  # type: ignore[arg-type]
    )


def confirm_audio_plan(config: ProjectConfig, plan: AudioPlan) -> ConfirmedAudioPlan:
    """Freeze one fresh plan as the immutable confirmed enumeration.

    Called once, before the owner's confirmation, so the fixed bytes of an
    already-``current`` clip are bound at the moment the counts were disclosed
    rather than re-read afterwards.
    """

    clips: list[ConfirmedAudioClip] = []
    for clip in plan.clips:
        media_sha256: str | None = None
        if clip.state == "current":
            try:
                media_sha256 = _proof_sha(
                    read_bytes_bound(_media_target(plan.media_dir, clip.target))
                )
            except (DataError, OSError) as exc:
                raise AudioProofError(
                    f"Could not bind current audio {clip.target!r}: {exc}"
                ) from exc
        clips.append(
            ConfirmedAudioClip(
                record_id=clip.record_id,
                kind=clip.kind,
                target=_one_media_component(clip.target),
                request_input=clip.request_input,
                forced_accent=clip.forced_accent,
                content_fingerprint=clip.content_fingerprint,
                provider=clip.provider,
                initial_state=clip.state,
                initial_media_sha256=media_sha256,
                initial_recovery_key=clip.recovery_key,
                initial_recovery_sha256=clip.recovery_sha256,
                initial_recovery_source=clip.recovery_source,
            )
        )
    try:
        return ConfirmedAudioPlan(
            repository_root=config.root.resolve(),
            canonical_path=plan.canonical_path,
            ledger_path=plan.ledger_path,
            media_dir=plan.media_dir,
            record_ids=plan.record_ids,
            targeted=plan.targeted,
            words=plan.words,
            examples=plan.examples,
            force=plan.force,
            word_provider=plan.word_provider,
            example_provider=plan.example_provider,
            clips=tuple(clips),
        )
    except ValueError as exc:
        raise AudioProofError(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class ReservedPaidAttempt:
    """One paid clip's durable reservation, as the dispatcher recorded it.

    ``expected_request_fp`` is the value the writer derived from the *approved*
    request and provider profile at ``before_paid_dispatch`` and wrote down
    before the call left. Proving reads the journal and compares it to this
    expectation; it never copies the journal's own fingerprint, which would
    prove only that the journal agrees with itself.
    """

    target: str
    operation_id: str
    expected_request_fp: str


@dataclass(frozen=True, slots=True)
class ProvenPaidOperation:
    """How one newly synthesized paid clip's attempt is accounted for.

    ``state`` is ``"committed"`` while the journal still carries the entry, and
    ``"accounted"`` once it does not — the ordinary successful outcome, since
    ``synthesize_journaled`` commits the captured reply through the audio stage
    and then ``forget``\\ s the same entry. A ``committed`` entry that is still
    present is a forget that was interrupted or has not run yet.

    ``state`` is a *classification*, never the binding. Absence from the
    journal is equally what a proven-unsent refusal, a reconciler's forget and
    an owner's discarded unknown outcome leave behind, so neither value is
    accepted on its own: both require the paid writer's own recorded attempt —
    operation id, request fingerprint, model and rendered byte hash, written
    into the clip's durable audio entry before the forget — to match this
    attempt and the bytes actually on disk.
    """

    operation_id: str
    kind: str
    model: str
    source_file: str
    source_sha256: str
    expected_request_fp: str
    state: Literal["committed", "accounted"]

    def to_wire(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "kind": self.kind,
            "model": self.model,
            "source_file": self.source_file,
            "source_sha256": self.source_sha256,
            "expected_request_fp": self.expected_request_fp,
            "state": self.state,
        }

    @classmethod
    def from_wire(cls, raw: Mapping[str, Any]) -> ProvenPaidOperation:
        expected = {
            "operation_id",
            "kind",
            "model",
            "source_file",
            "source_sha256",
            "expected_request_fp",
            "state",
        }
        if set(raw) != expected:
            raise AudioProofError("A proven paid audio operation is malformed.")
        state = raw.get("state")
        if state not in {"committed", "accounted"}:
            raise AudioProofError(
                f"A proven paid audio operation cannot be {state!r}."
            )
        return cls(
            operation_id=_wire_text(raw.get("operation_id"), "operation id"),
            kind=_wire_text(raw.get("kind"), "operation kind"),
            model=_wire_text(raw.get("model"), "operation model"),
            source_file=_wire_text(raw.get("source_file"), "operation source"),
            source_sha256=_wire_sha(raw.get("source_sha256"), "operation source hash"),
            expected_request_fp=_wire_text(
                raw.get("expected_request_fp"), "expected request fingerprint"
            ),
            state="committed" if state == "committed" else "accounted",
        )


@dataclass(frozen=True, slots=True)
class ProvenAudioSlot:
    """One census slot proven against durable repository state."""

    record_id: str
    kind: Literal["word", "example"]
    position: int | None
    target: str
    #: The exact stored ``record.audio`` / ``examples[i].audio`` value.
    reference: str
    media_sha256: str
    request_input: str
    forced_accent: bool
    content_fingerprint: str
    provider: AudioProviderPlan
    origin: ProvenSlotOrigin
    paid_operation: ProvenPaidOperation | None

    def to_wire(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "kind": self.kind,
            "position": self.position,
            "target": self.target,
            "reference": self.reference,
            "media_sha256": self.media_sha256,
            "request_input": self.request_input,
            "forced_accent": self.forced_accent,
            "content_fingerprint": self.content_fingerprint,
            "provider": provider_plan_wire(self.provider),
            "origin": self.origin,
            "paid_operation": (
                None if self.paid_operation is None else self.paid_operation.to_wire()
            ),
        }

    @classmethod
    def from_wire(cls, raw: Mapping[str, Any]) -> ProvenAudioSlot:
        expected = {
            "record_id",
            "kind",
            "position",
            "target",
            "reference",
            "media_sha256",
            "request_input",
            "forced_accent",
            "content_fingerprint",
            "provider",
            "origin",
            "paid_operation",
        }
        if set(raw) != expected:
            raise AudioProofError("A proven audio slot is malformed.")
        kind = _wire_kind(raw.get("kind"))
        position = raw.get("position")
        if kind == "word":
            if position is not None:
                raise AudioProofError("A proven word slot carries no example position.")
        elif isinstance(position, bool) or not isinstance(position, int) or position < 0:
            raise AudioProofError("A proven example slot needs its example position.")
        origin = raw.get("origin")
        if origin not in {"reused", "recovered", "synthesized"}:
            raise AudioProofError("A proven audio slot carries an unknown origin.")
        provider = _provider_from_wire(raw.get("provider"))
        if provider is None:
            raise AudioProofError("A proven audio slot needs its provider profile.")
        paid = raw.get("paid_operation")
        if paid is not None and not isinstance(paid, Mapping):
            raise AudioProofError("A proven paid audio operation is malformed.")
        if paid is not None and origin == "reused":
            raise AudioProofError(
                "A reused audio clip cannot carry this finish's paid operation."
            )
        return cls(
            record_id=_wire_text(raw.get("record_id"), "record id"),
            kind=kind,
            position=None if kind == "word" else int(position),  # type: ignore[arg-type]
            target=_one_media_component(_wire_text(raw.get("target"), "clip target")),
            reference=_wire_text(raw.get("reference"), "stored audio reference"),
            media_sha256=_wire_sha(raw.get("media_sha256"), "media hash"),
            request_input=_wire_text(raw.get("request_input"), "request input"),
            forced_accent=_wire_bool(raw.get("forced_accent"), "forced accent"),
            content_fingerprint=_wire_sha(
                raw.get("content_fingerprint"), "content fingerprint"
            ),
            provider=provider,
            origin=origin,  # type: ignore[arg-type]
            paid_operation=None if paid is None else ProvenPaidOperation.from_wire(paid),
        )


@dataclass(frozen=True, slots=True)
class AudioCompletionProof:
    """Durable proof that one exact audio scope is complete on disk.

    Every field is recomputable from committed repository state, so a lost
    return value costs nothing: re-proving from the same durable inputs yields
    the same fingerprint. The fingerprint proves the payload's integrity and
    nothing else — a consumer still revalidates the artifacts, the requests and
    the accounted operations before it relies on any of it.
    """

    schema: str
    repository_root: Path
    canonical_path: Path
    canonical_sha256: str
    ledger_path: Path
    media_dir: Path
    record_ids: tuple[str, ...]
    words: bool
    examples: bool
    #: One entry per census slot; several slots may share one clip target.
    slots: tuple[ProvenAudioSlot, ...]
    fingerprint: str

    def __post_init__(self) -> None:
        seen = {(slot.record_id, slot.kind, slot.position) for slot in self.slots}
        if len(seen) != len(self.slots):
            raise ValueError("An audio completion proof repeats a slot")

    @property
    def clip_count(self) -> int:
        """Unique clips — deliberately a different number from the slot count."""

        return len({slot.target for slot in self.slots})

    @property
    def stored_slot_count(self) -> int:
        return len(self.slots)

    @property
    def realized_media_sha256(self) -> Mapping[Path, str]:
        """Absolute media path to proven byte hash, one entry per clip."""

        realized: dict[Path, str] = {}
        for slot in self.slots:
            realized[_media_target(self.media_dir, slot.target)] = slot.media_sha256
        return realized

    @property
    def origin_by_media_path(self) -> Mapping[Path, ProvenSlotOrigin]:
        origins: dict[Path, ProvenSlotOrigin] = {}
        for slot in self.slots:
            origins[_media_target(self.media_dir, slot.target)] = slot.origin
        return origins

    def slots_for(
        self, record_id: str, kind: str, position: int | None
    ) -> ProvenAudioSlot | None:
        for slot in self.slots:
            if (slot.record_id, slot.kind, slot.position) == (record_id, kind, position):
                return slot
        return None

    def to_wire(self) -> dict[str, Any]:
        payload = self._payload()
        payload["fingerprint"] = self.fingerprint
        return payload

    def _payload(self) -> dict[str, Any]:
        root = self.repository_root
        return {
            "schema": self.schema,
            "repository_root": _proof_relative(root, root, label="root"),
            "canonical_path": _proof_relative(
                root, self.canonical_path, label="audio owner"
            ),
            "canonical_sha256": self.canonical_sha256,
            "ledger_path": _proof_relative(root, self.ledger_path, label="audio ledger"),
            "media_dir": _proof_relative(root, self.media_dir, label="media directory"),
            "record_ids": list(self.record_ids),
            "words": self.words,
            "examples": self.examples,
            "clip_count": self.clip_count,
            "slots": [slot.to_wire() for slot in self.slots],
        }

    @classmethod
    def from_wire(
        cls, config: ProjectConfig, raw: Mapping[str, Any]
    ) -> AudioCompletionProof:
        if raw.get("schema") != AUDIO_COMPLETION_SCHEMA:
            raise AudioProofError(
                f"Unknown audio completion proof schema {raw.get('schema')!r}."
            )
        slots = raw.get("slots")
        if not isinstance(slots, list) or any(
            not isinstance(item, Mapping) for item in slots
        ):
            raise AudioProofError("An audio completion proof is malformed.")
        root = config.root.resolve()
        try:
            proof = cls(
                schema=AUDIO_COMPLETION_SCHEMA,
                repository_root=_proof_path(root, raw.get("repository_root"), label="root"),
                canonical_path=_proof_path(
                    root, raw.get("canonical_path"), label="audio owner"
                ),
                canonical_sha256=_wire_sha(
                    raw.get("canonical_sha256"), "canonical hash"
                ),
                ledger_path=_proof_path(
                    root, raw.get("ledger_path"), label="audio ledger"
                ),
                media_dir=_proof_path(root, raw.get("media_dir"), label="media directory"),
                record_ids=tuple(_wire_text_list(raw.get("record_ids"), "record ids")),
                words=_wire_bool(raw.get("words"), "words"),
                examples=_wire_bool(raw.get("examples"), "examples"),
                slots=tuple(ProvenAudioSlot.from_wire(item) for item in slots),
                fingerprint=_wire_sha(raw.get("fingerprint"), "proof fingerprint"),
            )
        except ValueError as exc:
            raise AudioProofError(str(exc)) from exc
        if raw.get("clip_count") != proof.clip_count:
            raise AudioProofError(
                "An audio completion proof miscounts its unique clips."
            )
        if _proof_fingerprint(proof) != proof.fingerprint:
            raise AudioProofError(
                "An audio completion proof does not match its own fingerprint."
            )
        return proof


def _proof_fingerprint(proof: AudioCompletionProof) -> str:
    return _proof_sha(_canonical_wire(proof._payload()))


def _assert_confirmed_configuration(
    authority: ConfirmedAudioPlan, fresh: AudioPlan
) -> None:
    fixed: tuple[tuple[str, object, object], ...] = (
        ("repository root", authority.repository_root, fresh.repository_root),
        ("owner", authority.canonical_path, fresh.canonical_path),
        ("ledger", authority.ledger_path, fresh.ledger_path),
        ("media directory", authority.media_dir, fresh.media_dir),
        ("record ids", authority.record_ids, fresh.record_ids),
        ("targeted scope", authority.targeted, fresh.targeted),
        ("word inclusion", authority.words, fresh.words),
        ("example inclusion", authority.examples, fresh.examples),
        ("force", authority.force, fresh.force),
        (
            "word provider",
            provider_plan_wire(authority.word_provider),
            provider_plan_wire(fresh.word_provider),
        ),
        (
            "example provider",
            provider_plan_wire(authority.example_provider),
            provider_plan_wire(fresh.example_provider),
        ),
    )
    for label, expected, observed in fixed:
        if expected != observed:
            raise AudioProofError(
                f"The confirmed audio {label} changed before completion was proven."
            )


def _assert_confirmed_clips(
    authority: ConfirmedAudioPlan, fresh: AudioPlan, media_dir: Path
) -> None:
    if len(authority.clips) != len(fresh.clips):
        raise AudioProofError(
            "The confirmed audio clip set changed before completion was proven."
        )
    for confirmed, clip in zip(authority.clips, fresh.clips, strict=True):
        for label, expected, observed in (
            ("record", confirmed.record_id, clip.record_id),
            ("kind", confirmed.kind, clip.kind),
            ("target", confirmed.target, clip.target),
            ("request input", confirmed.request_input, clip.request_input),
            ("forced accent", confirmed.forced_accent, clip.forced_accent),
            (
                "content fingerprint",
                confirmed.content_fingerprint,
                clip.content_fingerprint,
            ),
            (
                "provider",
                provider_plan_wire(confirmed.provider),
                provider_plan_wire(clip.provider),
            ),
        ):
            if expected != observed:
                raise AudioProofError(
                    f"The confirmed audio {label} for {confirmed.target!r} changed "
                    "before completion was proven."
                )
        allowed = CLIP_EVOLUTION.get(confirmed.initial_state)
        if allowed is None or clip.state not in allowed:
            raise AudioProofError(
                f"Audio clip {confirmed.target!r} would require authority wider "
                "than the confirmed plan."
            )
        if confirmed.initial_state == "current":
            if confirmed.initial_media_sha256 != _current_media_sha(
                media_dir, confirmed.target
            ):
                raise AudioProofError(
                    f"An already-current audio file changed after confirmation: "
                    f"{confirmed.target!r}."
                )
        elif (
            confirmed.initial_state == "recoverable"
            and clip.state == "current"
            and confirmed.initial_recovery_sha256
            != _current_media_sha(media_dir, confirmed.target)
        ):
            raise AudioProofError(
                f"A recovered audio clip no longer has its authorized bytes: "
                f"{confirmed.target!r}."
            )


def _current_media_sha(media_dir: Path, target: str) -> str:
    try:
        return _proof_sha(read_bytes_bound(_media_target(media_dir, target)))
    except (DataError, OSError) as exc:
        raise AudioProofError(
            f"Could not read proven audio {target!r}: {exc}"
        ) from exc


def _slot_request(
    record: VocabularyRecord, slot: ExpectedAudioSlot
) -> tuple[str, str, bool, str, str]:
    """``(filename prefix, request input, forced accent, content fp, reference)``."""

    if slot.kind == "word":
        utterance, forced, _warning = ledger.word_audio_request(record)
        return (
            f"janki-{ledger.word_audio_filename_fingerprint(record)}",
            utterance,
            forced,
            ledger.word_audio_content_fingerprint(record),
            record.audio,
        )
    position = slot.position
    if position is None or position >= len(record.examples):
        raise AudioProofError(
            f"Audio slot for {slot.record_id!r} names example {slot.position}, "
            "which the record does not have."
        )
    example = record.examples[position]
    return (
        f"janki-{ledger.example_audio_filename_fingerprint(record, example)}",
        ledger.example_audio_request(example),
        False,
        ledger.example_audio_content_fingerprint(example),
        example.audio,
    )


def _records_by_id(
    records: Sequence[VocabularyRecord],
) -> dict[str, VocabularyRecord]:
    by_id: dict[str, VocabularyRecord] = {}
    for record in records:
        if record.id in by_id:
            raise AudioProofError(f"Audio record id {record.id!r} occurs more than once.")
        by_id[record.id] = record
    return by_id


def _proven_slot_facts(
    *,
    slot: ExpectedAudioSlot,
    record: VocabularyRecord,
    target: str,
    provider: AudioProviderPlan,
    content_fingerprint: str,
    reference: str,
    media_dir: Path,
    book: ledger.Ledger,
) -> tuple[str, Mapping[str, Any]]:
    """Prove one slot's reference, bytes and exact ledger currency.

    Returns the media hash and the ledger entry that answered — the entry
    rather than only the file name, because a paid clip's attribution lives in
    the same durable row the currency answer came from, and re-querying for it
    would risk proving currency against one entry and money against another.
    """

    path = _media_target(media_dir, target)
    expected_reference = audio_cmd.media_relative(path, media_dir)
    if not reference or reference != expected_reference:
        raise AudioProofError(
            f"{record.id} slot {slot.kind}/{slot.position} does not reference its "
            f"proven clip {target!r}; a file on disk is not a canonical link."
        )
    media_sha256 = _current_media_sha(media_dir, target)
    entry = book.audio_entry_for(
        record.id,
        of=slot.kind,
        content_fp=content_fingerprint,
        provider=provider.name,
        voice=provider.voice,
        speed=provider.speed,
        settings=provider.settings,
    )
    if entry is None or (str(entry.get("file") or "") or None) != target:
        raise AudioProofError(
            f"The ledger does not record {target!r} as current for "
            f"{record.id} slot {slot.kind}/{slot.position} under this exact "
            "request and provider profile."
        )
    return media_sha256, entry


def _recorded_paid_attempt(
    entry: Mapping[str, Any],
    *,
    attempt: ReservedPaidAttempt,
    target: str,
    media_sha256: str,
) -> None:
    """Require the writer's own witness for these exact bytes, or refuse.

    This is the whole binding for the ordinary successful paid clip. A
    journaled call that succeeds is *forgotten*, so absence from the journal
    says nothing: it is equally what a refusal proven before send, a
    reconciler's forget of a `failed_before_send` row, an owner's
    ``--forget --force`` discard of an unknown outcome, and an id that never
    existed all leave behind. Meanwhile any other authorized run may
    legitimately voice the same identity-addressed target, because the same
    sentence and profile always address the same clip.

    So the question is never "is the journal quiet?" but "does the durable
    record of *these* bytes name *this* attempt?" — and the answer must come
    from something the paid writer wrote while its call still existed, bound
    to the operation id, the request fingerprint the dispatcher independently
    expected, the model, and the bytes now on disk. Never from the reservation,
    which is the claim being checked, and never from absence.
    """

    witness = entry.get(audio_cmd.PAID_ATTEMPT)
    if not isinstance(witness, Mapping):
        raise AudioProofError(
            f"Paid audio clip {target!r} carries no recorded attempt in its "
            "ledger entry, so nothing says this finish's call produced its "
            "bytes; a forgotten attempt is not an accounted one."
        )
    recorded_id = str(witness.get("operation_id") or "")
    if recorded_id != attempt.operation_id:
        raise AudioProofError(
            f"Paid audio attempt {attempt.operation_id} did not produce "
            f"{target!r}: its bytes are recorded as the result of attempt "
            f"{recorded_id or 'an unnamed call'}."
        )
    if str(witness.get("request_fp") or "") != attempt.expected_request_fp:
        raise AudioProofError(
            f"Paid audio attempt {attempt.operation_id} recorded another "
            f"request for {target!r} than the one this finish expected."
        )
    if str(witness.get("model") or "") != openai_realtime.MODEL:
        raise AudioProofError(
            f"Paid audio attempt {attempt.operation_id} recorded model "
            f"{witness.get('model')!r} for {target!r}, not "
            f"{openai_realtime.MODEL!r}."
        )
    if str(witness.get("audio_sha256") or "") != media_sha256:
        raise AudioProofError(
            f"Paid audio attempt {attempt.operation_id} rendered other bytes "
            f"than the {target!r} on disk; a later render is not this "
            "attempt's result."
        )


def _accounted_paid_operation(
    config: ProjectConfig,
    *,
    attempt: ReservedPaidAttempt,
    record_id: str,
    kind: str,
    target: str,
    content_fingerprint: str,
    entry: Mapping[str, Any],
    media_sha256: str,
) -> ProvenPaidOperation:
    expected_source = audio_cmd.audio_journal_source(record_id, of=kind, target=target)
    # Required on *both* paths. A still-present entry proves the call was
    # billed and committed; only the recorded witness proves it is the call
    # whose reply became these bytes.
    _recorded_paid_attempt(
        entry, attempt=attempt, target=target, media_sha256=media_sha256
    )
    journal = operations.OperationJournal.load(config.operations_file)
    operation = journal.operations.get(attempt.operation_id)
    if operation is None:
        # The ordinary successful outcome: `synthesize_journaled` persisted the
        # attempt through the audio stage, wrote `committed`, and then forgot
        # the entry, and that durable forget is the repository's statement that
        # the money was dealt with. The binding that survives it is the witness
        # checked above, together with the committed bytes and the ledger's
        # currency for the exact request and profile — both proven beside this.
        return ProvenPaidOperation(
            operation_id=attempt.operation_id,
            kind=_PROVEN_PAID_OPERATION_KIND,
            model=openai_realtime.MODEL,
            source_file=expected_source,
            source_sha256=content_fingerprint,
            expected_request_fp=attempt.expected_request_fp,
            state="accounted",
        )
    if (
        operation.kind != _PROVEN_PAID_OPERATION_KIND
        or operation.source_file != expected_source
        or operation.source_sha256 != content_fingerprint
        or operation.model != openai_realtime.MODEL
        or operation.request_fp != attempt.expected_request_fp
    ):
        raise AudioProofError(
            f"Paid audio attempt {attempt.operation_id} does not match the exact "
            f"confirmed request for {target!r}."
        )
    if operation.state != "committed":
        # Everything else is either money nobody has accounted for
        # (`result_captured`, `outcome_unknown`, or a live state) or an attempt
        # that never produced these bytes.
        raise AudioProofError(
            f"Paid audio attempt {attempt.operation_id} for {target!r} is "
            f"{operation.state}, which is not an accounted committed call."
        )
    return ProvenPaidOperation(
        operation_id=operation.operation_id,
        kind=operation.kind,
        model=operation.model,
        source_file=operation.source_file,
        source_sha256=operation.source_sha256,
        expected_request_fp=attempt.expected_request_fp,
        state="committed",
    )


def _reservations_by_target(
    authority: ConfirmedAudioPlan,
    reservations: Sequence[ReservedPaidAttempt],
) -> dict[str, ReservedPaidAttempt]:
    attempts: dict[str, ReservedPaidAttempt] = {}
    for attempt in reservations:
        confirmed = authority.clip_for(attempt.target)
        if confirmed is None:
            raise AudioProofError(
                f"A paid audio reservation names {attempt.target!r}, which the "
                "confirmed enumeration does not contain."
            )
        if (
            confirmed.initial_state != "provider-required"
            or confirmed.provider.access != "paid-network"
        ):
            raise AudioProofError(
                f"A paid audio reservation names {attempt.target!r}, which this "
                "finish did not dispatch; reuse and recovery invent no attempt."
            )
        if attempt.target in attempts:
            raise AudioProofError(
                f"Paid audio clip {attempt.target!r} carries two reservations."
            )
        attempts[attempt.target] = attempt
    return attempts


def prove_audio_completion(
    config: ProjectConfig,
    plan: AudioPlan,
    *,
    authority: ConfirmedAudioPlan,
    expected_slots: Sequence[ExpectedAudioSlot],
    reservations: Sequence[ReservedPaidAttempt] = (),
) -> AudioCompletionProof:
    """Prove one finished audio scope from durable repository state.

    The caller **must** already hold
    ``exclusive_path_lock(config.root / ".janki-audio-operation")`` and must
    pass a plan re-made under it after the writer finished. Nothing here
    resolves a provider, contacts one, or writes anything.

    ``authority`` is the immutable confirmed enumeration; ``expected_slots`` is
    the caller's disclosed census, which is checked against an independent one
    derived from the post-audio canonical records rather than trusted.
    """

    root = config.root.resolve()
    canonical = config.normalized_file.resolve()
    if plan.repository_root != root:
        raise AudioProofError("The audio plan belongs to another repository.")
    if plan.canonical_path != canonical:
        raise AudioProofError(
            "An audio completion proof covers the canonical vocabulary "
            f"collection; {plan.canonical_path} is another owner."
        )
    if not plan.targeted or plan.force:
        raise AudioProofError(
            "An audio completion proof needs one exact targeted, unforced scope."
        )
    _assert_confirmed_configuration(authority, plan)
    _assert_confirmed_clips(authority, plan, plan.media_dir)
    unfinished = [clip.target for clip in plan.clips if clip.state != "current"]
    if unfinished:
        raise AudioProofError(
            f"Audio clip {unfinished[0]!r} is not current, so this scope is "
            "not complete."
        )

    records, revision = load_records_snapshot(canonical)
    if revision.text is None or revision.text != plan.canonical_revision.text:
        raise AudioProofError(
            "The canonical collection changed between the audio plan and its proof."
        )
    canonical_sha256 = _proof_sha(revision.text.encode("utf-8"))
    census = expected_audio_slots(
        records, plan.record_ids, words=plan.words, examples=plan.examples
    )
    declared = tuple(expected_slots)
    if set(declared) != set(census) or len(set(declared)) != len(declared):
        raise AudioProofError(
            f"The disclosed audio slot census does not match the {len(census)} "
            "slots the accepted records carry."
        )
    by_id = _records_by_id(records)
    try:
        book = ledger.load(config.ledger_file.resolve())
    except (JankiError, OSError) as exc:
        raise AudioProofError(f"Could not read the audio ledger: {exc}") from exc
    attempts = _reservations_by_target(authority, reservations)

    slots: list[ProvenAudioSlot] = []
    proven_targets: set[str] = set()
    for slot in census:
        record = by_id[slot.record_id]
        prefix, request_input, forced, content_fp, reference = _slot_request(record, slot)
        matches = [
            clip
            for clip in plan.clips
            if clip.record_id == slot.record_id
            and clip.kind == slot.kind
            and clip.target == f"{prefix}{clip.provider.suffix}"
        ]
        if len(matches) != 1:
            raise AudioProofError(
                f"{record.id} slot {slot.kind}/{slot.position} is covered by "
                f"{len(matches)} planned clips; every expected slot needs exactly one."
            )
        clip = matches[0]
        if (
            clip.request_input != request_input
            or clip.content_fingerprint != content_fp
            or clip.forced_accent != forced
        ):
            raise AudioProofError(
                f"{record.id} slot {slot.kind}/{slot.position} no longer matches "
                "the exact request its clip was planned for."
            )
        media_sha256, entry = _proven_slot_facts(
            slot=slot,
            record=record,
            target=clip.target,
            provider=clip.provider,
            content_fingerprint=clip.content_fingerprint,
            reference=reference,
            media_dir=plan.media_dir,
            book=book,
        )
        confirmed = authority.clip_for(clip.target)
        if confirmed is None:  # pragma: no cover - clip sets already compared
            raise AudioProofError(
                f"Audio clip {clip.target!r} is outside the confirmed enumeration."
            )
        origin = _ORIGIN_FOR_INITIAL_STATE[confirmed.initial_state]
        paid: ProvenPaidOperation | None = None
        if (
            confirmed.initial_state == "provider-required"
            and confirmed.provider.access == "paid-network"
        ):
            attempt = attempts.get(clip.target)
            if attempt is None:
                raise AudioProofError(
                    f"Paid audio clip {clip.target!r} was confirmed as new work but "
                    "no reserved attempt accounts for it."
                )
            paid = _accounted_paid_operation(
                config,
                attempt=attempt,
                record_id=record.id,
                kind=slot.kind,
                target=clip.target,
                content_fingerprint=clip.content_fingerprint,
                entry=entry,
                media_sha256=media_sha256,
            )
        proven_targets.add(clip.target)
        slots.append(
            ProvenAudioSlot(
                record_id=record.id,
                kind=slot.kind,
                position=slot.position,
                target=clip.target,
                reference=reference,
                media_sha256=media_sha256,
                request_input=request_input,
                forced_accent=forced,
                content_fingerprint=clip.content_fingerprint,
                provider=clip.provider,
                origin=origin,
                paid_operation=paid,
            )
        )
    missing = [clip.target for clip in plan.clips if clip.target not in proven_targets]
    if missing:
        raise AudioProofError(
            f"Audio clip {missing[0]!r} reaches no expected slot; the plan and the "
            "accepted records disagree about what was voiced."
        )
    draft = AudioCompletionProof(
        schema=AUDIO_COMPLETION_SCHEMA,
        repository_root=root,
        canonical_path=canonical,
        canonical_sha256=canonical_sha256,
        ledger_path=config.ledger_file.resolve(),
        media_dir=plan.media_dir,
        record_ids=plan.record_ids,
        words=plan.words,
        examples=plan.examples,
        slots=tuple(slots),
        fingerprint="0" * 64,
    )
    return replace(draft, fingerprint=_proof_fingerprint(draft))


def revalidate_audio_completion(
    config: ProjectConfig,
    proof: AudioCompletionProof,
) -> None:
    """Re-prove a saved completion against current repository state.

    Resolves **no** provider and sends no call: the paid request fingerprint
    was bound against the live provider at dispatch and is compared here to the
    journal, never recomputed. The caller must already hold
    ``.janki-audio-operation``.

    A cleared write-ahead row and a retired capture are what success leaves
    behind, so their absence is expected and never read as missing proof.
    """

    root = config.root.resolve()
    for label, expected, observed in (
        ("repository root", root, proof.repository_root),
        ("canonical collection", config.normalized_file.resolve(), proof.canonical_path),
        ("ledger", config.ledger_file.resolve(), proof.ledger_path),
        ("media directory", config.media_dir.resolve(), proof.media_dir),
    ):
        if expected != observed:
            raise AudioProofError(
                f"The audio completion proof names another {label}: {observed}."
            )
    if proof.schema != AUDIO_COMPLETION_SCHEMA:
        raise AudioProofError(
            f"Unknown audio completion proof schema {proof.schema!r}."
        )
    if _proof_fingerprint(proof) != proof.fingerprint:
        raise AudioProofError(
            "An audio completion proof does not match its own fingerprint."
        )
    records, revision = load_records_snapshot(proof.canonical_path)
    if revision.text is None or _proof_sha(revision.text.encode("utf-8")) != (
        proof.canonical_sha256
    ):
        raise AudioProofError(
            "The canonical collection changed since audio completion was proven."
        )
    census = expected_audio_slots(
        records, proof.record_ids, words=proof.words, examples=proof.examples
    )
    carried = {(slot.record_id, slot.kind, slot.position) for slot in proof.slots}
    expected_set = {(slot.record_id, slot.kind, slot.position) for slot in census}
    if carried != expected_set:
        raise AudioProofError(
            f"The audio completion proof carries {len(carried)} slots but the "
            f"proven records hold {len(expected_set)}."
        )
    by_id = _records_by_id(records)
    try:
        book = ledger.load(proof.ledger_path)
    except (JankiError, OSError) as exc:
        raise AudioProofError(f"Could not read the audio ledger: {exc}") from exc
    for slot in proof.slots:
        record = by_id[slot.record_id]
        expected_slot = ExpectedAudioSlot(slot.record_id, slot.kind, slot.position)
        prefix, request_input, forced, content_fp, reference = _slot_request(
            record, expected_slot
        )
        if (
            slot.request_input != request_input
            or slot.content_fingerprint != content_fp
            or slot.forced_accent != forced
            or slot.target != f"{prefix}{slot.provider.suffix}"
        ):
            raise AudioProofError(
                f"{record.id} slot {slot.kind}/{slot.position} no longer matches "
                "the exact request the proof bound."
            )
        if reference != slot.reference:
            raise AudioProofError(
                f"{record.id} slot {slot.kind}/{slot.position} no longer links the "
                f"proven clip {slot.target!r}."
            )
        media_sha256, entry = _proven_slot_facts(
            slot=expected_slot,
            record=record,
            target=slot.target,
            provider=slot.provider,
            content_fingerprint=slot.content_fingerprint,
            reference=reference,
            media_dir=proof.media_dir,
            book=book,
        )
        if media_sha256 != slot.media_sha256:
            raise AudioProofError(
                f"The proven audio {slot.target!r} no longer holds its proven bytes."
            )
        if slot.paid_operation is None:
            continue
        # The classification may legitimately move from `committed` to
        # `accounted` between proving and now, as an interrupted forget
        # finishes. It is deliberately not pinned: what is checked is that the
        # ledger still records this attempt as the one that rendered these
        # exact bytes, and that whatever the journal holds under its id still
        # binds this exact request and is not money nobody has accounted for.
        _accounted_paid_operation(
            config,
            attempt=ReservedPaidAttempt(
                target=slot.target,
                operation_id=slot.paid_operation.operation_id,
                expected_request_fp=slot.paid_operation.expected_request_fp,
            ),
            record_id=slot.record_id,
            kind=slot.kind,
            target=slot.target,
            content_fingerprint=slot.content_fingerprint,
            entry=entry,
            media_sha256=media_sha256,
        )
