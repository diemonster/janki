"""Plan one exact audio scope without contacting a speech provider.

The durable audio transaction still lives in the CLI while it is extracted in
small, reviewable pieces.  This module owns the first shared seam: the browser
can describe an exact nonempty set of promoted records, provider cost class,
and clip scope without allowing an empty id list to mean the whole corpus.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from japanese_anki import audio_cmd, ledger, status
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.exporters.anki import deck_declared_record_versions
from japanese_anki.io import RecordsRevision, load_records_snapshot
from japanese_anki.models import VocabularyRecord
from japanese_anki.tts import SpeechProvider, openai_tts, voicevox

AudioAccess = Literal["local-network", "paid-network"]


class AudioPlanError(JankiError):
    """An audio request does not name one exact canonical scope."""


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


def _provider_name(config: ProjectConfig, chosen: str | None) -> str:
    return (chosen or config.tts_provider or "voicevox").strip().lower()


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
            '[tts] sentence_provider = "openai".'
        )
    raise AudioPlanError(
        f"Unknown TTS provider {name!r}. Words are voiced by voicevox. For "
        "sentences, set [tts] sentence_provider to voicevox or openai."
    )


def resolve_sentence_provider(
    config: ProjectConfig,
    chosen: str | None,
    words: SpeechProvider,
) -> SpeechProvider:
    """Construct the example provider without contacting it."""
    name = (config.sentence_provider or "").strip().lower()
    if name == "openai":
        return openai_tts.OpenAiSpeechProvider(
            voice=config.openai_voice,
            model=config.openai_model,
            instructions=config.openai_instructions,
        )
    if name not in {"", "voicevox"}:
        raise AudioPlanError(
            f"Unknown [tts] sentence_provider {name!r}. Known: voicevox, openai, "
            "or leave it empty to read sentences in the same voice as the words."
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


def _provider_plan(provider: SpeechProvider) -> AudioProviderPlan:
    access: AudioAccess = "paid-network" if provider.name == "openai" else "local-network"
    if provider.name == "openai":
        destination = openai_tts.API_URL
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
    sentence_provider: SpeechProvider | None,
    protected_records: Sequence[VocabularyRecord],
) -> AudioPlan:
    if not words and not examples:
        raise AudioPlanError(
            "janki audio needs --words, --examples, or both: they are different "
            "recordings made different ways, and neither is the obvious default."
        )
    canonical_path = config.normalized_file.resolve()
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
            book=ledger.load(config.ledger_file.resolve()),
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
    sentence_provider: SpeechProvider | None = None,
) -> AudioPlan:
    """Plan one exact nonempty promoted-record scope."""
    records, revision = load_records_snapshot(config.normalized_file.resolve())
    protected_records = tuple(
        record
        for deck_path in status.deck_files(config)
        for record in deck_declared_record_versions(deck_path)
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
    sentence_provider: SpeechProvider | None = None,
) -> AudioPlan:
    """Plan the CLI's explicit whole-corpus maintenance operation."""
    records, revision = load_records_snapshot(config.normalized_file.resolve())
    protected_records = tuple(
        record
        for deck_path in status.deck_files(config)
        for record in deck_declared_record_versions(deck_path)
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
    sentence_provider: SpeechProvider | None = None,
    protected_records: Sequence[VocabularyRecord] = (),
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
    )
