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
from pathlib import Path
from typing import Literal

from japanese_anki import audio_cmd, ledger, status
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
    def provider_wire(provider: AudioProviderPlan | None) -> dict[str, object] | None:
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
            "settings": provider.settings,
        }

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
    book: ledger.Ledger, keys: Sequence[str], *, replace: bool = False
) -> ledger.Ledger:
    """Durably merge only this call's additive paid-audio WAL rows."""
    book.merge_pending_audio(keys, replace=replace)
    return book


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
        persist_pending=lambda current, key: _persist_audio_wal(current, [key], replace=force),
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
