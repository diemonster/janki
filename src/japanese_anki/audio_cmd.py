"""Generating the audio a card plays, and knowing when not to.

Two kinds, and they are not variations of each other. A **word** is spoken with
its pitch accent forced, because 橋 and 箸 are the pair a card exists to
distinguish and an engine left to guess renders them alike. A **sentence** is
read naturally, because forcing an accent across one needs accent data janki
does not have and a natural adult voice matters more there than a guaranteed
contour.

**Every word gets a clip.** When janki has the pitch pattern it forces the
accent; when it does not, the engine reads the kana and picks one, the ledger
entry is tagged ``accent_unverified`` so the guess is visible afterwards rather
than indistinguishable from a verified clip, and every run says which records
are carrying one — not only the run that wrote them. Earlier this skipped the
record instead — DESIGN_V2 argued that a guess is wrong on exactly the
homographs a pitch card is for, which is true — but a silent card teaches
nothing at all, and the owner's decision (2026-08-15) is that a clip with an
unverified accent beats no clip. The trade is safe because it is temporary:
word audio fingerprints the exact bare-reading or forced-AquesTalk utterance
path, so the day ``enrich --jpdb`` fills the accent the guessed clip reads as
stale and the next ``janki audio`` replaces it with a forced one.

M8.3 deleted the furigana flag and the teaching-suitability hold: janki's logic
enriches the card, it never audits the model, and what the sentence says is the
model's answer to a template that asked precisely.

Files are identity-addressed (``janki-<fingerprint><provider suffix>``): a word
by record id, an example by record id plus Japanese text. Change a sentence and
its audio moves to a new name; change a voice or per-clip instruction and the
ledger makes that stable file stale so it is rewritten in place. ``--prune``
sweeps what neither a durable card nor a pending paid-audio transaction
references.
"""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from japanese_anki import ledger as ledger_mod
from japanese_anki.errors import JankiError
from japanese_anki.io import DataError, atomic_write_bytes, atomic_write_bytes_bound
from japanese_anki.models import (
    ExampleSentence,
    VocabularyRecord,
)
from japanese_anki.tts import (
    SpeechProvider,
    TtsError,
    clip_provider,
    validate_utterance,
)

__all__ = [
    "ACCENT_UNVERIFIED",
    "AUDIO_SUBDIR",
    "AudioError",
    "AudioResult",
    "SynthesisError",
    "cleanup_pending_stages",
    "cleanup_unclaimed_pending_stages",
    "commit_promoted_audio",
    "current_pending_audio_keys",
    "generate_audio",
    "media_relative",
    "promote_pending_audio",
    "prune_unreferenced",
]


class AudioError(JankiError):
    """Audio was asked for in a way janki refuses — a caller error.

    Deliberately narrow. These are raised before anything is spent and cannot
    happen mid-run, which is what lets the run loop re-raise them untouched
    while salvaging everything else.
    """


class SynthesisError(JankiError):
    """The provider answered, and the answer was not usable audio.

    Separate from :class:`AudioError` because it happens *during* a run, on one
    record out of many. Folding it in would send an empty response from record
    40 of 300 unwinding past the code that saves the 39 clips already written —
    leaving files on disk with no record reference and no ledger entry, which
    the next ``--prune`` deletes.
    """


class PruneError(JankiError):
    """A prune stopped after removing and accounting for earlier files."""

    def __init__(self, message: str, *, removed: Sequence[Path]) -> None:
        super().__init__(message)
        self.removed = list(removed)


#: Where clips live, under the project's ``media_dir``.
AUDIO_SUBDIR = "audio"

#: The ledger detail marking a clip whose accent the engine chose rather than
#: janki. Recorded so "we let it guess here" stays answerable later, instead of
#: looking identical to a clip whose accent came from a dictionary.
ACCENT_UNVERIFIED = "accent_unverified"


def media_relative(path: Path, media_dir: Path) -> str:
    """The stored form of a media path: relative to ``media_dir``, forward slashes.

    Records travel between machines and into Anki, so an absolute path is wrong
    twice over. Anki flattens media into one folder by basename, but janki keeps
    the subdirectory in the record because that is what its own ``--prune`` and
    the exporter read.
    """
    return path.relative_to(media_dir).as_posix()


def _pending_stage_name(key: str, staged_sha256: str) -> str:
    return f".pending/{key}-{staged_sha256}.stage"


def _pending_directory(audio_dir: Path, *, create: bool) -> Path | None:
    """Return the real private staging directory, refusing a symlink."""
    root = audio_dir.resolve()
    pending = root / ".pending"
    if create:
        try:
            pending.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise SynthesisError(
                f"Could not create pending audio directory {pending}: "
                f"{exc.strerror or exc}"
            ) from exc
    try:
        details = os.lstat(pending)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise SynthesisError(
            f"Could not inspect pending audio directory {pending}: "
            f"{exc.strerror or exc}"
        ) from exc
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise SynthesisError(
            f"Pending audio directory {pending} must be a real directory, not a link"
        )
    return pending


def _read_pending_stage(audio_dir: Path, staged_name: str) -> bytes | None:
    """Read one direct regular stage without following either path symlink."""
    path = Path(staged_name)
    if path.parent.as_posix() != ".pending" or path.name in {"", ".", ".."}:
        raise SynthesisError(f"Refusing unsafe pending stage path {staged_name!r}")
    pending = _pending_directory(audio_dir, create=False)
    if pending is None:
        return None
    directory_fd = -1
    descriptor = -1
    try:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        directory_fd = os.open(pending, flags)
        descriptor = os.open(
            path.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode):
            raise SynthesisError(f"Pending audio stage {staged_name!r} is not regular")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            return handle.read()
    except FileNotFoundError:
        return None
    except SynthesisError:
        raise
    except OSError as exc:
        raise SynthesisError(
            f"Could not safely read pending audio stage {staged_name!r}: "
            f"{exc.strerror or exc}"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if directory_fd >= 0:
            os.close(directory_fd)


def _read_canonical_audio(audio_dir: Path, target_name: str) -> bytes | None:
    """Read one canonical media entry without following a target symlink."""
    if Path(target_name).name != target_name:
        raise SynthesisError(f"Refusing unsafe canonical audio path {target_name!r}")
    directory = audio_dir.resolve()
    if not directory.is_dir():
        return None
    directory_fd = -1
    descriptor = -1
    try:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        directory_fd = os.open(directory, flags)
        descriptor = os.open(
            target_name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode):
            raise SynthesisError(
                f"Canonical audio target {target_name!r} is not a regular file"
            )
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            return handle.read()
    except FileNotFoundError:
        return None
    except SynthesisError:
        raise
    except OSError as exc:
        raise SynthesisError(
            f"Could not safely read canonical audio target {target_name!r}: "
            f"{exc.strerror or exc}"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if directory_fd >= 0:
            os.close(directory_fd)


def _canonical_target_exists(audio_dir: Path, target_name: str) -> bool:
    """Whether a selected destination is a real regular entry, never a link."""
    if Path(target_name).name != target_name:
        raise SynthesisError(f"Refusing unsafe canonical audio path {target_name!r}")
    target = audio_dir.resolve() / target_name
    try:
        details = os.lstat(target)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise SynthesisError(
            f"Could not inspect canonical audio target {target}: "
            f"{exc.strerror or exc}"
        ) from exc
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
        raise SynthesisError(
            f"Canonical audio target {target} must be a regular file, not a link"
        )
    return True


def _orphan_stages_for(
    audio_dir: Path, key: str, *, tolerate_corrupt: bool = False
) -> list[tuple[str, str, bytes]]:
    """Verified self-describing stages for one exact request key."""
    pending = _pending_directory(audio_dir, create=False)
    if pending is None:
        return []
    prefix = f"{key}-"
    candidates: list[tuple[str, str, bytes]] = []
    try:
        names = sorted(os.listdir(pending))
    except OSError as exc:
        raise SynthesisError(
            f"Could not inspect pending audio directory {pending}: "
            f"{exc.strerror or exc}"
        ) from exc
    for name in names:
        if not name.startswith(prefix) or not name.endswith(".stage"):
            continue
        digest = name[len(prefix) : -len(".stage")]
        staged_name = f".pending/{name}"
        data = _read_pending_stage(audio_dir, staged_name)
        if (
            data is None
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or hashlib.sha256(data).hexdigest() != digest
        ):
            if tolerate_corrupt:
                continue
            raise SynthesisError(
                f"Pending audio stage {staged_name!r} is corrupt; use --force "
                "to replace this exact request explicitly."
            )
        candidates.append((staged_name, digest, data))
    return candidates


@dataclass(slots=True)
class AudioResult:
    """What a run generated, skipped, and refused."""

    records: list[VocabularyRecord] = field(default_factory=list)
    #: ``record id -> [file names written]``
    written: dict[str, list[str]] = field(default_factory=dict)
    #: Exact write-ahead entries that must be promoted only after the records
    #: file compare-and-swap succeeds. Empty for the direct, immediate helper
    #: mode used by focused synthesis tests.
    pending_keys: list[str] = field(default_factory=list)
    #: Records whose accent janki could not force, so the engine chose one.
    #: Their clips exist and are tagged ``accent_unverified``. Reported on
    #: *every* run, not only the one that wrote the clip: a guessed accent is a
    #: standing property of the record until the dictionary supplies a pattern,
    #: and a report that fires once is one nobody sees again.
    guessed_accent: list[str] = field(default_factory=list)
    #: Records with no reading at all, which cannot be voiced and are not the
    #: same problem as a missing accent.
    no_reading: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    #: Clips that already existed and were left alone.
    up_to_date: int = 0
    #: Why the run stopped early, if it did. The clips written before it are
    #: still in ``records`` and their entries still in the ledger: throwing that
    #: away would make the next run pay for all of them again, and names are
    #: deterministic, so what landed is exactly what a re-run would skip.
    stopped_by: str = ""

    @property
    def file_count(self) -> int:
        return sum(len(names) for names in self.written.values())


def _is_current(
    book: ledger_mod.Ledger,
    record_id: str,
    *,
    of: str,
    content_fp: str,
    named: str,
    audio_dir: Path,
    provider: SpeechProvider,
) -> bool:
    """Is there a usable clip for this content already?

    Four facts, and the ledger holds only two: an entry saying this was
    recorded, in this run's voice and at this run's rate, the record still
    pointing at that file, and the file existing.

    Asking the ledger alone calls a record current whose reference was dropped —
    a reverted ``vocabulary.json``, a ``migrate-inline`` that rewrote
    ``examples`` — and then ``--prune``, which reads the records, deletes the
    clip nothing appears to want. The ledger goes on answering "already
    recorded" and nothing ever synthesizes it again.

    Asking about content alone calls a clip current that says the right words
    in the wrong voice. That is what made a changed ``voicevox_speaker`` need
    ``--force`` to take effect, and what made an interrupted re-voice
    unresumable: the clips already redone were current, and so — wrongly — were
    the ones still in the old voice.
    """
    if not named:
        return False
    recorded = book.audio_file_for(
        record_id,
        of=of,
        content_fp=content_fp,
        provider=provider.name,
        voice=provider.voice,
        speed=provider.speed,
        settings=provider.settings,
    )
    if recorded is None or recorded != Path(named).name:
        return False
    return _canonical_target_exists(audio_dir, recorded)


def _pending_current_file(
    book: ledger_mod.Ledger,
    record_id: str,
    *,
    of: str,
    content_fp: str,
    expected: str,
    request_input: str,
    forced_accent: bool,
    audio_dir: Path,
    provider: SpeechProvider,
    details: dict[str, object] | None = None,
    persist_pending: Callable[[ledger_mod.Ledger, str], None] | None = None,
    replace_corrupt: bool = False,
) -> tuple[str, str] | None:
    """Recover an exact staged render whose record CAS did not commit.

    A self-verifying stage can precede its WAL row if the process is interrupted
    inside the persistence callback. Its request key and byte SHA in the name
    are enough to reconstruct the exact row before deciding to call a provider.
    """
    arguments = {
        "of": of,
        "target": expected,
        "request_input": request_input,
        "forced_accent": forced_accent,
        "content_fp": content_fp,
        "provider": provider.name,
        "voice": provider.voice,
        "speed": provider.speed,
        "settings": provider.settings,
    }
    key = book.pending_audio_key_for(record_id, **arguments)
    found = book.pending_audio_for(
        record_id,
        **arguments,
    )
    if found is not None:
        key, entry = found
        staged_sha = str(entry["staged_sha256"])
        staged_name = str(entry["staged_file"])
        staged_data = _read_pending_stage(audio_dir, staged_name)
        if (
            staged_data is not None
            and hashlib.sha256(staged_data).hexdigest() == staged_sha
        ):
            return key, expected
        canonical_data = _read_canonical_audio(audio_dir, expected)
        if (
            canonical_data is not None
            and hashlib.sha256(canonical_data).hexdigest() == staged_sha
        ):
            return key, expected
        if replace_corrupt:
            replacements = _orphan_stages_for(
                audio_dir, key, tolerate_corrupt=True
            )
            if len(replacements) > 1:
                raise SynthesisError(
                    f"More than one valid replacement stage exists for exact "
                    f"request {key}; refusing to choose between paid renders."
                )
            if replacements:
                staged_name, staged_sha, _ = replacements[0]
                recorded = book.record_pending_audio(
                    record_id,
                    **arguments,
                    staged_file=staged_name,
                    staged_sha256=staged_sha,
                    **(details or {}),
                )
                if recorded != key:  # pragma: no cover - ledger owns this
                    raise SynthesisError(
                        "Pending audio identity changed during stage recovery"
                    )
                if persist_pending is not None:
                    persist_pending(book, key)
                return key, expected
            return None
        raise SynthesisError(
            f"Pending audio for {expected} no longer has bytes matching its "
            "recorded SHA-256; refusing to bill the exact request again. Use "
            "--force to replace the corrupt recovery entry explicitly."
        )

    try:
        orphaned = _orphan_stages_for(
            audio_dir, key, tolerate_corrupt=replace_corrupt
        )
    except SynthesisError:
        if replace_corrupt:
            return None
        raise
    if len(orphaned) > 1:
        raise SynthesisError(
            f"More than one completed pending stage exists for exact request {key}; "
            "refusing to choose between paid renders."
        )
    if not orphaned:
        return None
    staged_name, staged_sha, _ = orphaned[0]
    recorded = book.record_pending_audio(
        record_id,
        **arguments,
        staged_file=staged_name,
        staged_sha256=staged_sha,
        **(details or {}),
    )
    if recorded != key:  # pragma: no cover - ledger owns the identity formula
        raise SynthesisError("Pending audio identity changed during stage recovery")
    if persist_pending is not None:
        persist_pending(book, key)
    return key, expected


def _stage_audio(
    data: bytes,
    *,
    book: ledger_mod.Ledger,
    record_id: str,
    of: str,
    target: str,
    request_input: str,
    forced_accent: bool,
    content_fp: str,
    provider: SpeechProvider,
    audio_dir: Path,
    details: dict[str, object] | None = None,
    persist_pending: Callable[[ledger_mod.Ledger, str], None] | None = None,
) -> str:
    """Write paid bytes away from canonical media and add their exact WAL row."""
    arguments = {
        "of": of,
        "target": target,
        "request_input": request_input,
        "forced_accent": forced_accent,
        "content_fp": content_fp,
        "provider": provider.name,
        "voice": provider.voice,
        "speed": provider.speed,
        "settings": provider.settings,
    }
    staged_sha256 = hashlib.sha256(data).hexdigest()
    key = book.pending_audio_key_for(record_id, **arguments)
    staged_file = _pending_stage_name(key, staged_sha256)
    try:
        atomic_write_bytes_bound(audio_dir / staged_file, data)
    except DataError as exc:
        raise SynthesisError(
            f"Could not save synthesized audio to {audio_dir / staged_file}: {exc}"
        ) from exc
    recorded = book.record_pending_audio(
        record_id,
        **arguments,
        staged_file=staged_file,
        staged_sha256=staged_sha256,
        **(details or {}),
    )
    if recorded != key:  # pragma: no cover - the ledger owns this formula
        raise SynthesisError("Pending audio identity changed while it was staged")
    # Inside the per-clip operation, before control can reach another provider
    # call. A process death or OOM after this point therefore leaves the exact
    # paid bytes reachable from the durable WAL, not merely from this object.
    if persist_pending is not None:
        persist_pending(book, key)
    return key


def _write(path: Path, data: bytes) -> None:
    if not data:
        raise SynthesisError(f"The provider returned no audio for {path.name}.")
    try:
        atomic_write_bytes(path, data)
    except (DataError, OSError) as exc:
        raise SynthesisError(
            f"Could not save synthesized audio to {path}: {exc}"
        ) from exc


def _word_request(record: VocabularyRecord) -> tuple[str, bool, str | None]:
    """Return the exact provider input/flag for a word, plus a render warning."""
    return ledger_mod.word_audio_request(record)


def _word_audio(
    record: VocabularyRecord,
    *,
    provider: SpeechProvider,
    book: ledger_mod.Ledger,
    audio_dir: Path,
    media_dir: Path,
    result: AudioResult,
    force: bool,
    stage_only: bool,
    persist_pending: Callable[[ledger_mod.Ledger, str], None] | None,
) -> VocabularyRecord:
    # Decided before the currency check, deliberately. Whether this record's
    # accent can be forced is a property of the *record*, true on every run,
    # not of the one run that happened to write the clip — appended after the
    # early return below, the report fired once and then went quiet while the
    # clips went on carrying a guess. The mismatch warning is the same: bad
    # data stays bad, so it keeps saying so.
    utterance, forced, render_error = _word_request(record)
    if render_error is not None:
        # A pattern that does not fit its reading cannot be forced —
        # `_word_request` refuses rather than inventing an alignment — so this
        # is the no-pattern case wearing different clothes. Falling back voices
        # the card; the warning keeps the bad source data visible.
        result.warnings.append(f"{record.id}: {render_error}")
    if not forced:
        # The explicit flag, never utterance truthiness: a bare empty reading
        # and an impossible forced form are different structural states even
        # if both happen to carry an empty provider input.
        result.guessed_accent.append(record.id)

    content_fp = ledger_mod.word_audio_content_fingerprint(record)
    name = (
        f"janki-{ledger_mod.word_audio_filename_fingerprint(record)}{provider.suffix}"
    )
    details = {} if forced else {ACCENT_UNVERIFIED: True}
    # Exact paid recovery outranks canonical currency, including on the plain
    # rerun prescribed by status after a failed --force transaction.
    if stage_only and (
        recovered := _pending_current_file(
            book,
            record.id,
            of="word",
            content_fp=content_fp,
            expected=name,
            request_input=utterance,
            forced_accent=forced,
            audio_dir=audio_dir,
            provider=provider,
            details=details,
            persist_pending=persist_pending,
            replace_corrupt=force,
        )
    ):
        key, recovered_name = recovered
        if key not in result.pending_keys:
            result.pending_keys.append(key)
        result.up_to_date += 1
        return replace(
            record,
            audio=media_relative(audio_dir / recovered_name, media_dir),
        )
    if not force and _is_current(
        book,
        record.id,
        of="word",
        content_fp=content_fp,
        named=record.audio,
        audio_dir=audio_dir,
        provider=provider,
    ):
        result.up_to_date += 1
        return record

    if forced:
        data = provider.synthesize(utterance, forced_accent=True)
    else:
        # The engine reads the kana and picks the accent. Marked, because a
        # listener cannot tell this clip from a verified one and later work
        # needs to.
        data = provider.synthesize(utterance, forced_accent=False)

    if stage_only:
        key = _stage_audio(
            data,
            book=book,
            record_id=record.id,
            of="word",
            target=name,
            request_input=utterance,
            forced_accent=forced,
            content_fp=content_fp,
            provider=provider,
            audio_dir=audio_dir,
            details=details,
            persist_pending=persist_pending,
        )
        if key not in result.pending_keys:
            result.pending_keys.append(key)
    else:
        _write(audio_dir / name, data)
        book.record_audio(
            record.id,
            file=name,
            of="word",
            provider=provider.name,
            voice=provider.voice,
            speed=provider.speed,
            settings=provider.settings,
            content_fp=content_fp,
            **details,
        )
    result.written.setdefault(record.id, []).append(name)
    return replace(record, audio=media_relative(audio_dir / name, media_dir))


def _example_audio(
    record: VocabularyRecord,
    *,
    providers: Sequence[SpeechProvider],
    book: ledger_mod.Ledger,
    audio_dir: Path,
    media_dir: Path,
    result: AudioResult,
    force: bool,
    stage_only: bool,
    persist_pending: Callable[[ledger_mod.Ledger, str], None] | None,
) -> VocabularyRecord:
    """Voice this record's examples, keeping whatever gets written.

    The order matters twice, and both were wrong before.

    Superseded ledger entries are dropped **after** the loop, not before it, and
    only when nothing still points at their file. Dropping first deletes the one
    durable record of a mismatch in exactly the cases where no replacement
    follows — a provider that fails on the first example, say — leaving the
    record naming clip A while its sentence is B, with nothing left to report
    it.

    And a provider failure is caught **here**, so the partially-updated example
    list is returned rather than discarded. Unwinding past this point loses the
    reference to every clip already written for the record: the files stay on
    disk, unreferenced, and the next ``--prune`` deletes them — while the run
    reports "the clips written before this are saved".
    """
    examples: list[ExampleSentence] = []
    changed = False
    # Identical sentence text within one record is one identity-addressed clip.
    # Find a current referenced member *before* walking stored order: otherwise
    # ``[blank, current]`` pays to overwrite a clip that ``[current, blank]``
    # reuses for free. Preflight has already proved duplicate instructions agree.
    resolved: dict[str, str] = {}
    if not force or stage_only:
        for position, example in enumerate(record.examples):
            if not example.japanese or example.japanese in resolved:
                continue
            provider = providers[position]
            expected = (
                f"janki-"
                f"{ledger_mod.example_audio_filename_fingerprint(record, example)}"
                f"{provider.suffix}"
            )
            if stage_only and (recovered := _pending_current_file(
                book,
                record.id,
                of="example",
                content_fp=ledger_mod.example_audio_content_fingerprint(example),
                expected=expected,
                request_input=example.japanese,
                forced_accent=False,
                audio_dir=audio_dir,
                provider=provider,
                persist_pending=persist_pending,
                replace_corrupt=force,
            )):
                key, recovered_name = recovered
                if key not in result.pending_keys:
                    result.pending_keys.append(key)
                resolved[example.japanese] = media_relative(
                    audio_dir / recovered_name, media_dir
                )
                result.up_to_date += 1
                continue
            if not force and _is_current(
                book,
                record.id,
                of="example",
                content_fp=ledger_mod.example_audio_content_fingerprint(example),
                named=example.audio,
                audio_dir=audio_dir,
                provider=provider,
            ):
                resolved[example.japanese] = example.audio
                result.up_to_date += 1

    for position, example in enumerate(record.examples):
        if result.stopped_by:
            # Keep the rest of the list intact so the record still names every
            # clip it had before this run.
            examples.append(example)
            continue
        if not example.japanese:
            examples.append(example)
            continue
        provider = providers[position]
        if example.japanese in resolved:
            audio = resolved[example.japanese]
            examples.append(replace(example, audio=audio))
            changed = changed or example.audio != audio
            continue
        content_fp = ledger_mod.example_audio_content_fingerprint(example)

        try:
            data = provider.synthesize(example.japanese, forced_accent=False)
            name = (
                f"janki-"
                f"{ledger_mod.example_audio_filename_fingerprint(record, example)}"
                f"{provider.suffix}"
            )
            if stage_only:
                key = _stage_audio(
                    data,
                    book=book,
                    record_id=record.id,
                    of="example",
                    target=name,
                    request_input=example.japanese,
                    forced_accent=False,
                    content_fp=content_fp,
                    provider=provider,
                    audio_dir=audio_dir,
                    persist_pending=persist_pending,
                )
                if key not in result.pending_keys:
                    result.pending_keys.append(key)
            else:
                _write(audio_dir / name, data)
        except AudioError:
            raise
        except JankiError as exc:
            result.stopped_by = f"{record.id} (example {position + 1}): {exc}"
            examples.append(example)
            continue

        if not stage_only:
            book.record_audio(
                record.id,
                file=name,
                of="example",
                provider=provider.name,
                voice=provider.voice,
                speed=provider.speed,
                settings=provider.settings,
                content_fp=content_fp,
            )
        result.written.setdefault(record.id, []).append(name)
        relative = media_relative(audio_dir / name, media_dir)
        examples.append(replace(example, audio=relative))
        resolved[example.japanese] = relative
        changed = True

    updated = replace(record, examples=examples) if changed else record

    # Now that whatever could be written has been: forget entries for sentences
    # this record no longer has *and* whose file nothing points at. An entry
    # whose clip is still referenced is the only evidence that the card names
    # one sentence and plays another, so it stays until that is actually fixed.
    if not stage_only:
        book.drop_superseded_audio(
            updated.id,
            of="example",
            keep={
                ledger_mod.example_audio_content_fingerprint(item)
                for item in updated.examples
                if item.japanese
            },
            keep_files={
                Path(item.audio).name for item in updated.examples if item.audio
            },
        )
    return updated


def generate_audio(
    records: Sequence[VocabularyRecord],
    *,
    provider: SpeechProvider,
    book: ledger_mod.Ledger,
    media_dir: Path,
    sentence_provider: SpeechProvider | None = None,
    words: bool = True,
    examples: bool = False,
    ids: Sequence[str] | None = None,
    force: bool = False,
    protected_records: Sequence[VocabularyRecord] = (),
    stage_only: bool = False,
    persist_pending: Callable[[ledger_mod.Ledger, str], None] | None = None,
) -> AudioResult:
    """Synthesize what is missing, and report what was deliberately not.

    Both kinds are off unless asked for, and at least one must be: "generate
    audio" without saying which is a request whose meaning would change under
    the user the day the other kind gains a default.

    ``sentence_provider`` voices examples; ``provider`` voices words. They
    default to the same engine, and the seam exists because the two jobs are
    not the same job: a word is spoken with its accent forced, which only an
    engine that can be told an accent can do, while a sentence is read
    naturally, where nothing is forced and a different voice is a free choice.
    Two providers rather than one provider with two voices, because everything
    downstream — which voice the ledger records, which voice ``_is_current``
    compares against — is already asked of *a provider*, and a provider that
    answered differently depending on the utterance would make both wrong.
    """
    sentence_provider = sentence_provider or provider
    if not (words or examples):
        raise AudioError(
            "janki audio needs --words, --examples, or both: they are different "
            "recordings made different ways, and neither is the obvious default."
        )

    result = AudioResult(records=list(records))
    audio_dir = media_dir / AUDIO_SUBDIR
    wanted = _targets(result.records, ids)
    selected_ids: set[str] = set()
    for record in result.records:
        if record.id not in wanted:
            continue
        if record.id in selected_ids:
            raise AudioError(
                f"Cannot generate audio for duplicate record id {record.id!r}. "
                "Each selected record id must identify one record."
            )
        selected_ids.add(record.id)
    prepared_examples: dict[str, list[SpeechProvider]] = {}
    if examples:
        # Resolve every selected clip before either provider is called. A
        # non-OpenAI provider cannot honor prose steering, and discovering that
        # after the word half ran would leave a partial mutation for a request
        # that was invalid from the start.
        for record in result.records:
            if record.id not in wanted:
                continue
            seen: dict[str, str] = {}
            profiles: list[SpeechProvider] = []
            for example in record.examples:
                instruction = example.instructions.strip()
                if example.japanese:
                    previous = seen.get(example.japanese)
                    if previous is not None and previous != instruction:
                        raise AudioError(
                            f"{record.id} has the same audio file for duplicate "
                            f"sentence {example.japanese!r} but different "
                            "instructions. Make the instructions agree or keep "
                            "only one copy of the sentence."
                        )
                    seen[example.japanese] = instruction
                try:
                    profile = (
                        clip_provider(sentence_provider, instruction)
                        if example.japanese
                        else sentence_provider
                    )
                    if example.japanese:
                        validate_utterance(profile, example.japanese)
                    profiles.append(profile)
                except TtsError as exc:
                    raise AudioError(f"{record.id}: {exc}") from exc
            prepared_examples[record.id] = profiles
    _refuse_address_collisions(
        result.records,
        book=book,
        wanted=wanted,
        word_provider=provider,
        prepared_examples=prepared_examples,
        protected_records=protected_records,
        words=words,
        examples=examples,
    )
    _preflight_provider_availability(
        result.records,
        wanted=wanted,
        book=book,
        audio_dir=audio_dir,
        word_provider=provider,
        prepared_examples=prepared_examples,
        words=words,
        examples=examples,
        force=force,
        stage_only=stage_only,
        persist_pending=persist_pending,
    )

    for index, record in enumerate(result.records):
        if record.id not in wanted or result.stopped_by:
            continue
        updated = record
        if words and not updated.reading:
            # The one skip that used to be silent, which made a run's summary
            # identical to one where the record did not exist.
            result.no_reading.append(record.id)
        try:
            if words and updated.reading:
                updated = _word_audio(
                    updated,
                    provider=provider,
                    book=book,
                    audio_dir=audio_dir,
                    media_dir=media_dir,
                    result=result,
                    force=force,
                    stage_only=stage_only,
                    persist_pending=persist_pending,
                )
            if examples:
                updated = _example_audio(
                    updated,
                    providers=prepared_examples[updated.id],
                    book=book,
                    audio_dir=audio_dir,
                    media_dir=media_dir,
                    result=result,
                    force=force,
                    stage_only=stage_only,
                    persist_pending=persist_pending,
                )
        except AudioError:
            # This module's own refusals — a caller error, not a run to salvage.
            raise
        except JankiError as exc:
            # The provider failed. Stop, but keep what was already written: the
            # clips are on disk with their ledger entries in hand, and throwing
            # that away would make the next run pay for all of them again.
            # Names are deterministic, so what landed is exactly what a re-run
            # would skip.
            result.stopped_by = f"{record.id}: {exc}"
        result.records[index] = updated
    return result


def _preflight_provider_availability(
    records: Sequence[VocabularyRecord],
    *,
    wanted: set[str],
    book: ledger_mod.Ledger,
    audio_dir: Path,
    word_provider: SpeechProvider,
    prepared_examples: dict[str, list[SpeechProvider]],
    words: bool,
    examples: bool,
    force: bool,
    stage_only: bool,
    persist_pending: Callable[[ledger_mod.Ledger, str], None] | None,
) -> None:
    """Check every engine still needed before the first provider call.

    Exact WAL/stage adoption is a local hash-and-rename operation and must stay
    possible while an engine is offline or an API key is unavailable. Engines
    are therefore required only for clips that neither recovery nor canonical
    currency can satisfy, while the complete census still happens before any
    synthesis call.
    """
    required: dict[int, SpeechProvider] = {}
    for record in records:
        if record.id not in wanted:
            continue
        if words and record.reading:
            utterance, forced, _ = _word_request(record)
            content_fp = ledger_mod.word_audio_content_fingerprint(record)
            target = (
                f"janki-{ledger_mod.word_audio_filename_fingerprint(record)}"
                f"{word_provider.suffix}"
            )
            _canonical_target_exists(audio_dir, target)
            recovered = None
            if stage_only:
                recovered = _pending_current_file(
                    book,
                    record.id,
                    of="word",
                    content_fp=content_fp,
                    expected=target,
                    request_input=utterance,
                    forced_accent=forced,
                    audio_dir=audio_dir,
                    provider=word_provider,
                    details={} if forced else {ACCENT_UNVERIFIED: True},
                    persist_pending=persist_pending,
                    replace_corrupt=force,
                )
            current = not force and _is_current(
                book,
                record.id,
                of="word",
                content_fp=content_fp,
                named=record.audio,
                audio_dir=audio_dir,
                provider=word_provider,
            )
            if recovered is None and not current:
                required[id(word_provider)] = word_provider
        if not examples:
            continue
        by_sentence: dict[str, list[int]] = {}
        for position, example in enumerate(record.examples):
            if example.japanese:
                by_sentence.setdefault(example.japanese, []).append(position)
        for positions in by_sentence.values():
            recovered = False
            current = False
            for position in positions:
                example = record.examples[position]
                sentence_provider = prepared_examples[record.id][position]
                content_fp = ledger_mod.example_audio_content_fingerprint(example)
                target = (
                    f"janki-"
                    f"{ledger_mod.example_audio_filename_fingerprint(record, example)}"
                    f"{sentence_provider.suffix}"
                )
                _canonical_target_exists(audio_dir, target)
                if stage_only and _pending_current_file(
                    book,
                    record.id,
                    of="example",
                    content_fp=content_fp,
                    expected=target,
                    request_input=example.japanese,
                    forced_accent=False,
                    audio_dir=audio_dir,
                    provider=sentence_provider,
                    persist_pending=persist_pending,
                    replace_corrupt=force,
                ):
                    recovered = True
                    break
                if not force and _is_current(
                    book,
                    record.id,
                    of="example",
                    content_fp=content_fp,
                    named=example.audio,
                    audio_dir=audio_dir,
                    provider=sentence_provider,
                ):
                    current = True
            if not recovered and not current:
                sentence_provider = prepared_examples[record.id][positions[0]]
                required[id(sentence_provider)] = sentence_provider
    for engine in required.values():
        if not engine.available():
            raise AudioError(f"{engine.name}: {engine.launch_hint}")


def _refuse_address_collisions(
    records: Sequence[VocabularyRecord],
    *,
    book: ledger_mod.Ledger,
    wanted: set[str],
    word_provider: SpeechProvider,
    prepared_examples: dict[str, list[SpeechProvider]],
    protected_records: Sequence[VocabularyRecord],
    words: bool,
    examples: bool,
) -> None:
    """Refuse two semantic clip identities that resolve to one output path.

    M8.1 promises no media-address migration. The existing concatenation is not
    framed and its 48-bit digest is finite, so every selected destination must
    be unique among the other selected writes and every durable reference.
    Otherwise the later clip overwrites the earlier while both records and the
    ledger can continue to name the shared file as current.
    """
    owners: dict[
        str,
        dict[tuple[str, ...], tuple[str, bool]],
    ] = {}

    def add(
        filename: str,
        identity: tuple[str, ...],
        description: str,
        selected: bool,
    ) -> None:
        # The generated namespace is ASCII. Case-fold anyway because the
        # repository commonly lives on case-insensitive APFS: a hand-edited
        # ``JANKI-….WAV`` reference and our lowercase destination are the same
        # file there even though POSIX ``normcase`` leaves them different.
        identities = owners.setdefault(filename.casefold(), {})
        prior = identities.get(identity)
        identities[identity] = (
            prior[0] if prior is not None else description,
            selected or (prior[1] if prior is not None else False),
        )

    # Protect every file a card already plays, regardless of which kind this
    # invocation will synthesize or which provider originally made it.
    for record in [*records, *protected_records]:
        if record.audio:
            utterance, forced, _ = _word_request(record)
            add(
                Path(record.audio).name,
                (
                    "word",
                    record.id,
                    utterance,
                    str(forced),
                ),
                f"word reference {record.id}",
                False,
            )
        for example in record.examples:
            if example.audio:
                add(
                    Path(example.audio).name,
                (
                    "example",
                    record.id,
                    example.japanese,
                    ledger_mod.example_audio_content_fingerprint(example),
                    example.instructions.strip(),
                ),
                    f"example reference {record.id} / {example.japanese}",
                    False,
                )

    # Selected destinations use the exact utterance-scoped provider that the
    # synthesis loop will use. A future provider may vary its suffix in
    # ``for_clip``; looking only at the collection-level provider would then
    # preflight a different path from the one actually written.
    for record in records:
        if record.id not in wanted:
            continue
        if words and record.reading:
            utterance, forced, _ = _word_request(record)
            add(
                f"janki-{ledger_mod.word_audio_filename_fingerprint(record)}"
                f"{word_provider.suffix}",
                (
                    "word",
                    record.id,
                    utterance,
                    str(forced),
                ),
                f"selected word {record.id}",
                True,
            )
        if examples:
            for example, sentence_provider in zip(
                record.examples, prepared_examples[record.id], strict=True
            ):
                if not example.japanese:
                    continue
                add(
                    f"janki-"
                    f"{ledger_mod.example_audio_filename_fingerprint(record, example)}"
                    f"{sentence_provider.suffix}",
                    (
                        "example",
                        record.id,
                        example.japanese,
                        ledger_mod.example_audio_content_fingerprint(example),
                        example.instructions.strip(),
                    ),
                    f"selected example {record.id} / {example.japanese}",
                    True,
                )

    # A paid staged render is already a durable owner even though no card may
    # reference its target yet. Ignoring it lets a colliding selected identity
    # overwrite the target and then erase the first owner's WAL and stage.
    for key in book.pending_audio:
        entry = book.pending_audio_entry(key)
        if entry is None:  # pragma: no cover - load validation owns the shape
            continue
        request = entry["request"]
        if not isinstance(request, dict):  # pragma: no cover - validated above
            continue
        kind = str(entry["of"])
        if kind == "word":
            identity = (
                "word",
                str(entry["record_id"]),
                str(request["input"]),
                str(request["forced_accent"]),
            )
            target_owners = owners.get(str(entry["target"]).casefold(), {})
            same_selected_slot = any(
                candidate[:2] == ("word", str(entry["record_id"])) and selected
                for candidate, (_, selected) in target_owners.items()
            )
            if same_selected_slot and identity not in target_owners:
                # Word filenames are deliberately stable across accent/reading
                # request changes. A stale WAL for this selected slot is
                # superseded after the new exact request commits; it is not a
                # second owner unless a durable current/protected reference
                # above still represents that old request identity.
                continue
        else:
            instruction = ""
            for record in [*records, *protected_records]:
                if record.id != str(entry["record_id"]):
                    continue
                for example in record.examples:
                    if (
                        example.japanese == str(request["input"])
                        and ledger_mod.example_audio_content_fingerprint(example)
                        == str(entry["content_fp"])
                    ):
                        instruction = example.instructions.strip()
                        break
            identity = (
                "example",
                str(entry["record_id"]),
                str(request["input"]),
                str(entry["content_fp"]),
                instruction,
            )
        add(
            str(entry["target"]),
            identity,
            f"pending {kind} {entry['record_id']} ({key})",
            False,
        )

    for filename, identities in owners.items():
        if len(identities) < 2 or not any(
            selected for _, selected in identities.values()
        ):
            continue
        descriptions = [description for description, _ in identities.values()]
        raise AudioError(
            "Different audio identities resolve to the same filename "
            f"{filename}: {'; '.join(descriptions)}. Nothing was synthesized; "
            "resolve the conflicting sentence or audio reference first. An "
            "identity change requires an explicit, reviewed migration."
        )


def _targets(records: Sequence[VocabularyRecord], ids: Sequence[str] | None) -> set[str]:
    if ids is None:
        return {record.id for record in records}
    known = {record.id for record in records}
    missing = [item for item in ids if item not in known]
    if missing:
        raise AudioError(
            f"No record with id {missing[0]!r}. Ids come from vocabulary.json; "
            "'janki status' lists them."
        )
    return set(ids)


def _pending_stage(
    key: str, entry: dict[str, object], audio_dir: Path
) -> tuple[Path, Path, bytes | None]:
    """Validate one WAL row and return stage, target, and bytes to publish.

    ``None`` bytes means an earlier attempt already published exactly the
    staged payload. The WAL remains the authority until its canonical ledger
    commit succeeds, so that interrupted state is safe to finish.
    """
    target_name = str(entry.get("target") or "")
    if not target_name or Path(target_name).name != target_name:
        raise SynthesisError("A pending audio entry has an unsafe target path")
    staged_name = str(entry.get("staged_file") or "")
    expected_sha = str(entry.get("staged_sha256") or "")
    expected_stage = _pending_stage_name(key, expected_sha)
    if staged_name != expected_stage:
        raise SynthesisError(
            f"Pending audio {key} staged_file must be exactly {expected_stage}; "
            "refusing to treat canonical media as disposable staging."
        )
    stage = audio_dir.resolve() / staged_name
    expected = expected_sha
    if (
        len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise SynthesisError("A pending audio entry has an invalid staged SHA-256")
    target = audio_dir / target_name
    staged_data = _read_pending_stage(audio_dir, staged_name)
    if staged_data is not None and hashlib.sha256(staged_data).hexdigest() == expected:
        canonical_data = _read_canonical_audio(audio_dir, target_name)
        if (
            canonical_data is not None
            and hashlib.sha256(canonical_data).hexdigest() == expected
        ):
            return stage, target, None
        return stage, target, staged_data
    canonical_data = _read_canonical_audio(audio_dir, target_name)
    if (
        canonical_data is not None
        and hashlib.sha256(canonical_data).hexdigest() == expected
    ):
        return stage, target, None
    raise SynthesisError(
        f"Pending audio for {target_name} has neither its exact staged bytes nor "
        "an already-promoted exact target; it must be synthesized again."
    )


def promote_pending_audio(
    book: ledger_mod.Ledger,
    keys: Sequence[str],
    audio_dir: Path,
    *,
    assert_current: Callable[[], None] | None = None,
) -> list[dict[str, object]]:
    """Atomically publish staged clips after the record-reference CAS won.

    Every staged hash is checked before the first canonical path moves. Each
    target is then replaced atomically, preserving its old bytes if a write
    fails. The WAL and canonical ledger are intentionally left untouched here;
    the caller commits both together only after every promotion succeeds.
    """
    entries: list[dict[str, object]] = []
    prepared: list[tuple[Path, Path, bytes | None]] = []
    if assert_current is not None:
        assert_current()
    for key in dict.fromkeys(keys):
        raw = book.pending_audio_entry(key)
        if raw is None:
            raise SynthesisError(f"Pending audio entry {key!r} disappeared")
        entry = dict(raw)
        entries.append(entry)
        prepared.append(_pending_stage(key, entry, audio_dir))
    for _, target, data in prepared:
        if data is not None:
            try:
                atomic_write_bytes_bound(target, data)
            except DataError as exc:
                raise SynthesisError(
                    f"Could not publish synthesized audio to {target}: {exc}"
                ) from exc
    return entries


def commit_promoted_audio(
    book: ledger_mod.Ledger,
    keys: Sequence[str],
    records: Sequence[VocabularyRecord],
) -> list[dict[str, object]]:
    """Apply promoted rows, retire old example rows, and clear their WAL rows."""
    committed: list[dict[str, object]] = []
    for key in dict.fromkeys(keys):
        committed.append(dict(book.commit_pending_audio(key)))
    example_ids = {
        str(entry.get("record_id") or "")
        for entry in committed
        if entry.get("of") == "example"
    }
    for record in records:
        if record.id not in example_ids:
            continue
        book.drop_superseded_audio(
            record.id,
            of="example",
            keep={
                ledger_mod.example_audio_content_fingerprint(example)
                for example in record.examples
                if example.japanese
            },
            keep_files={
                Path(example.audio).name
                for example in record.examples
                if example.audio
            },
        )
    return committed


def cleanup_pending_stages(
    entries: Iterable[dict[str, object]], audio_dir: Path
) -> list[str]:
    """Remove committed staging copies; failures are non-destructive warnings."""
    warnings: list[str] = []
    staged_names: set[str] = set()
    keys: set[str] = set()
    for entry in entries:
        staged_name = str(entry.get("staged_file") or "")
        staged_names.add(staged_name)
        staged_path = Path(staged_name)
        if (
            staged_path.parent.as_posix() != ".pending"
            or staged_path.suffix != ".stage"
        ):
            warnings.append(f"refused unsafe pending stage path {staged_name!r}")
            continue
        key = staged_path.name.split("-", 1)[0]
        if len(key) == 64 and all(
            character in "0123456789abcdef" for character in key
        ):
            keys.add(key)
    pending = _pending_directory(audio_dir, create=False)
    if pending is not None and keys:
        try:
            for name in os.listdir(pending):
                if any(name.startswith(f"{key}-") for key in keys) and name.endswith(
                    ".stage"
                ):
                    staged_names.add(f".pending/{name}")
        except OSError as exc:
            warnings.append(f"could not inspect committed pending stages: {exc}")
    for staged_name in sorted(staged_names):
        staged_path = Path(staged_name)
        if (
            staged_path.parent.as_posix() != ".pending"
            or staged_path.suffix != ".stage"
        ):
            continue
        try:
            data = _read_pending_stage(audio_dir, staged_name)
            if data is None:
                continue
            current_pending = _pending_directory(audio_dir, create=False)
            if current_pending is None:
                continue
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            directory_fd = os.open(current_pending, flags)
            try:
                os.unlink(staged_path.name, dir_fd=directory_fd)
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except (OSError, SynthesisError) as exc:
            warnings.append(
                f"could not remove committed pending stage {staged_name!r}: {exc}"
            )
    return warnings


def current_pending_audio_keys(
    records: Iterable[VocabularyRecord],
    *,
    book: ledger_mod.Ledger,
    word_provider: SpeechProvider,
    sentence_provider: SpeechProvider,
) -> set[str]:
    """Exact request keys whose no-row stages still belong to current owners."""
    keys: set[str] = set()
    for record in records:
        if record.reading:
            utterance, forced, _ = _word_request(record)
            keys.add(
                book.pending_audio_key_for(
                    record.id,
                    of="word",
                    target=(
                        f"janki-{ledger_mod.word_audio_filename_fingerprint(record)}"
                        f"{word_provider.suffix}"
                    ),
                    request_input=utterance,
                    forced_accent=forced,
                    content_fp=ledger_mod.word_audio_content_fingerprint(record),
                    provider=word_provider.name,
                    voice=word_provider.voice,
                    speed=word_provider.speed,
                    settings=word_provider.settings,
                )
            )
        for example in record.examples:
            if not example.japanese:
                continue
            profile = clip_provider(sentence_provider, example.instructions.strip())
            keys.add(
                book.pending_audio_key_for(
                    record.id,
                    of="example",
                    target=(
                        f"janki-"
                        f"{ledger_mod.example_audio_filename_fingerprint(record, example)}"
                        f"{profile.suffix}"
                    ),
                    request_input=example.japanese,
                    forced_accent=False,
                    content_fp=ledger_mod.example_audio_content_fingerprint(example),
                    provider=profile.name,
                    voice=profile.voice,
                    speed=profile.speed,
                    settings=profile.settings,
                )
            )
    return keys


def cleanup_unclaimed_pending_stages(
    book: ledger_mod.Ledger,
    audio_dir: Path,
    *,
    current_keys: set[str],
) -> list[str]:
    """Retire self-verifying stages claimed by neither WAL nor current owners."""
    pending = _pending_directory(audio_dir, create=False)
    if pending is None:
        return []
    live_files: set[str] = set()
    live_keys: set[str] = set()
    for key in book.pending_audio:
        entry = book.pending_audio_entry(key)
        if entry is None:  # pragma: no cover - load validation owns shape
            continue
        live_keys.add(key)
        live_files.add(str(entry["staged_file"]))
    warnings: list[str] = []
    try:
        names = sorted(os.listdir(pending))
    except OSError as exc:
        return [f"could not inspect pending audio stages: {exc}"]
    retired_keys: set[str] = set()
    for name in names:
        if not name.endswith(".stage"):
            continue
        staged_name = f".pending/{name}"
        if staged_name in live_files:
            continue
        pieces = name[: -len(".stage")].split("-", 1)
        if len(pieces) != 2:
            warnings.append(f"refused malformed pending stage {staged_name!r}")
            continue
        key, digest = pieces
        if (
            len(key) != 64
            or len(digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in f"{key}{digest}"
            )
        ):
            warnings.append(f"refused malformed pending stage {staged_name!r}")
            continue
        if key in live_keys or key in current_keys or key in retired_keys:
            continue
        retired_keys.add(key)
        warnings.extend(
            cleanup_pending_stages([{"staged_file": staged_name}], audio_dir)
        )
    return warnings


def prune_unreferenced(
    records: Iterable[VocabularyRecord],
    media_dir: Path,
    book: ledger_mod.Ledger | None = None,
    *,
    persist: Callable[[ledger_mod.Ledger], None] | None = None,
    assert_current: Callable[[], None] | None = None,
) -> list[Path]:
    """Delete ``janki-*`` clips nothing points at, and return what went.

    Only janki's own files: a clip somebody dropped into ``data/media`` by hand
    is theirs, and a sweep that removed it would be janki deleting a file it
    never created. Identity-addressed naming makes the candidate set bounded;
    a file is removable only when neither a durable record nor a pending-audio
    WAL row names it. Editing a sentence changes the identity address rather
    than silently transferring ownership of the old file.
    """
    audio_dir = media_dir / AUDIO_SUBDIR
    if not audio_dir.is_dir() and book is None:
        return []
    if book is not None:
        # Deletion cannot be rolled back. A stale audio command must discover
        # that another command committed new references before it removes the
        # corresponding new clip and only later loses its ledger CAS.
        book.assert_current()
    if assert_current is not None:
        assert_current()
    referenced = set()
    for record in records:
        if record.audio:
            referenced.add(Path(record.audio).name.casefold())
        for example in record.examples:
            if example.audio:
                referenced.add(Path(example.audio).name.casefold())
    if book is not None:
        # A paid clip whose record CAS lost is deliberately not referenced yet.
        # Its pending ledger marker is the durable reference until the next run
        # adopts it; an unrelated targeted prune must not erase that recovery.
        referenced.update(name.casefold() for name in book.pending_audio_files())
    candidates = [
        path
        for path in (sorted(audio_dir.glob("janki-*")) if audio_dir.is_dir() else [])
        if path.name.casefold() not in referenced
    ]
    if book is not None:
        # Include ledger rows whose file disappeared in an earlier interrupted
        # prune. Saving the complete forget set *before* unlinking makes both
        # halves self-healing: a save failure deletes nothing, while an unlink
        # failure leaves only harmless unledgered files for the next sweep.
        ledger_files = {
            str(item.get("file") or "")
            for record_entry in book.records.values()
            if isinstance(record_entry, dict)
            for item in (
                record_entry.get("audio")
                if isinstance(record_entry.get("audio"), list)
                else []
            )
            if isinstance(item, dict) and str(item.get("file") or "")
        }
        forget = {
            name
            for name in ledger_files | {path.name for path in candidates}
            if name.casefold() not in referenced
        }
        forgotten = book.forget_audio_files(forget)
        if forgotten and persist is not None:
            persist(book)
    removed: list[Path] = []
    for path in candidates:
        try:
            path.unlink()
        except OSError as exc:
            raise PruneError(
                f"Could not prune {path}: {exc.strerror or exc}",
                removed=removed,
            ) from exc
        removed.append(path)
    return removed
