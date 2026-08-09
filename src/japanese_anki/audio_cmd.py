"""Generating the audio a card plays, and knowing when not to.

Two kinds, and they are not variations of each other. A **word** is spoken with
its pitch accent forced, because 橋 and 箸 are the pair a card exists to
distinguish and an engine left to guess renders them alike. A **sentence** is
read naturally, because forcing an accent across one needs accent data janki
does not have and a natural adult voice matters more there than a guaranteed
contour.

Everything here is arranged around one rule: **never speak what janki is not
sure of.** A record with no pitch pattern is skipped and reported, not
synthesized with whatever the engine guesses — DESIGN_V2 is explicit that the
homographs a guess gets wrong are exactly the ones a pitch card is for.
``--allow-default-accent`` opts into the guess deliberately and tags the ledger
entry ``accent_unverified`` so the choice is visible afterwards rather than
indistinguishable from a verified one. An example whose furigana jpdb never
confirmed is skipped for the same reason: M4.2 flagged it precisely because
nobody has checked its segmentation, and speaking it would launder that doubt
into a recording.

Files are content-addressed (``janki-<fingerprint>.wav``), which is what makes
staleness fall out rather than needing tracking: change a sentence and its audio
simply no longer exists under the new name. ``--prune`` sweeps what nothing
references; ``--force`` rewrites in place, under the same name, because Anki's
media sync notices content rather than names.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from japanese_anki import ledger as ledger_mod
from japanese_anki import pitch
from japanese_anki.errors import JankiError
from japanese_anki.identifiers import short_fingerprint
from japanese_anki.models import ExampleSentence, VocabularyRecord
from japanese_anki.tts import SpeechProvider

__all__ = [
    "ACCENT_UNVERIFIED",
    "AUDIO_SUBDIR",
    "AudioError",
    "AudioResult",
    "SynthesisError",
    "generate_audio",
    "media_relative",
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


@dataclass(slots=True)
class AudioResult:
    """What a run generated, skipped, and refused."""

    records: list[VocabularyRecord] = field(default_factory=list)
    #: ``record id -> [file names written]``
    written: dict[str, list[str]] = field(default_factory=dict)
    #: Records skipped for having no accent pattern, which is the interesting
    #: skip: it means a card will have no word audio until somebody supplies one.
    no_pattern: list[str] = field(default_factory=list)
    #: Records with no reading at all, which cannot be voiced and are not the
    #: same problem as a missing accent.
    no_reading: list[str] = field(default_factory=list)
    #: Examples skipped because their furigana was never confirmed (M4.2's flag).
    unverified: list[str] = field(default_factory=list)
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
        voice=provider.voice,
        speed=provider.speed,
    )
    if recorded is None or recorded != Path(named).name:
        return False
    return (audio_dir / recorded).is_file()


def _write(path: Path, data: bytes) -> None:
    if not data:
        raise SynthesisError(f"The provider returned no audio for {path.name}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _word_audio(
    record: VocabularyRecord,
    *,
    provider: SpeechProvider,
    book: ledger_mod.Ledger,
    audio_dir: Path,
    media_dir: Path,
    result: AudioResult,
    force: bool,
    allow_default_accent: bool,
) -> VocabularyRecord:
    pattern = pitch.select_pattern(record)
    if not pattern and not allow_default_accent:
        # The skip DESIGN_V2 asks for. Reported rather than silent, because a
        # word with no audio is a card that behaves differently from its
        # neighbours and the reason is not visible on the card.
        result.no_pattern.append(record.id)
        return record

    content_fp = ledger_mod.word_audio_content_fingerprint(record)
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

    if pattern:
        try:
            spoken = pitch.to_aquestalk(record.reading, pattern)
        except pitch.PitchError as exc:
            # A pattern that does not fit its reading is exactly the case
            # `to_aquestalk` refuses rather than guesses at, and this command
            # must not convert that refusal into a guess of its own.
            result.warnings.append(f"{record.id}: {exc}")
            result.no_pattern.append(record.id)
            return record
        data = provider.synthesize(spoken, forced_accent=True)
        details = {}
    else:
        # `--allow-default-accent`: the engine reads the kana and picks the
        # accent. Marked, because a listener cannot tell this clip from a
        # verified one and later work needs to.
        data = provider.synthesize(record.reading, forced_accent=False)
        details = {ACCENT_UNVERIFIED: True}

    name = (
        f"janki-{ledger_mod.word_audio_filename_fingerprint(record)}{provider.suffix}"
    )
    _write(audio_dir / name, data)
    book.record_audio(
        record.id,
        file=name,
        of="word",
        provider=provider.name,
        voice=provider.voice,
        speed=provider.speed,
        content_fp=content_fp,
        **details,
    )
    result.written.setdefault(record.id, []).append(name)
    return replace(record, audio=media_relative(audio_dir / name, media_dir))


def _unverified_fingerprints(record: VocabularyRecord) -> set[str]:
    """The example fingerprints M4.2 flagged as having unconfirmed furigana."""
    raw = record.source.raw_fields.get("furigana_unverified", "")
    return {item.strip() for item in raw.split(",") if item.strip()}


def _example_audio(
    record: VocabularyRecord,
    *,
    provider: SpeechProvider,
    book: ledger_mod.Ledger,
    audio_dir: Path,
    media_dir: Path,
    result: AudioResult,
    force: bool,
) -> VocabularyRecord:
    """Voice this record's examples, keeping whatever gets written.

    The order matters twice, and both were wrong before.

    Superseded ledger entries are dropped **after** the loop, not before it, and
    only when nothing still points at their file. Dropping first deletes the one
    durable record of a mismatch in exactly the cases where no replacement
    follows — an edited sentence whose new text is flagged unverified, or a
    provider that fails on the first example — leaving the record naming clip A
    while its sentence is B, with nothing left to report it.

    And a provider failure is caught **here**, so the partially-updated example
    list is returned rather than discarded. Unwinding past this point loses the
    reference to every clip already written for the record: the files stay on
    disk, unreferenced, and the next ``--prune`` deletes them — while the run
    reports "the clips written before this are saved".
    """
    flagged = _unverified_fingerprints(record)
    examples: list[ExampleSentence] = []
    changed = False

    for position, example in enumerate(record.examples):
        if result.stopped_by:
            # Keep the rest of the list intact so the record still names every
            # clip it had before this run.
            examples.append(example)
            continue
        if not example.japanese:
            examples.append(example)
            continue
        if short_fingerprint(example.japanese) in flagged:
            # M4.2 flagged this because nobody confirmed its segmentation.
            # Speaking it would turn an open question into a recording.
            result.unverified.append(f"{record.id}: {example.japanese}")
            examples.append(example)
            continue

        content_fp = ledger_mod.example_audio_content_fingerprint(example)
        if not force and _is_current(
            book,
            record.id,
            of="example",
            content_fp=content_fp,
            named=example.audio,
            audio_dir=audio_dir,
            provider=provider,
        ):
            result.up_to_date += 1
            examples.append(example)
            continue

        try:
            data = provider.synthesize(example.japanese, forced_accent=False)
            name = (
                f"janki-"
                f"{ledger_mod.example_audio_filename_fingerprint(record, example)}"
                f"{provider.suffix}"
            )
            _write(audio_dir / name, data)
        except AudioError:
            raise
        except JankiError as exc:
            result.stopped_by = f"{record.id} (example {position + 1}): {exc}"
            examples.append(example)
            continue

        book.record_audio(
            record.id,
            file=name,
            of="example",
            provider=provider.name,
            voice=provider.voice,
            speed=provider.speed,
            content_fp=content_fp,
        )
        result.written.setdefault(record.id, []).append(name)
        examples.append(
            replace(example, audio=media_relative(audio_dir / name, media_dir))
        )
        changed = True

    updated = replace(record, examples=examples) if changed else record

    # Now that whatever could be written has been: forget entries for sentences
    # this record no longer has *and* whose file nothing points at. An entry
    # whose clip is still referenced is the only evidence that the card names
    # one sentence and plays another, so it stays until that is actually fixed.
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
    allow_default_accent: bool = False,
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
                    allow_default_accent=allow_default_accent,
                )
            if examples:
                updated = _example_audio(
                    updated,
                    provider=sentence_provider,
                    book=book,
                    audio_dir=audio_dir,
                    media_dir=media_dir,
                    result=result,
                    force=force,
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


def prune_unreferenced(
    records: Iterable[VocabularyRecord],
    media_dir: Path,
    book: ledger_mod.Ledger | None = None,
) -> list[Path]:
    """Delete ``janki-*`` clips nothing points at, and return what went.

    Only janki's own files: a clip somebody dropped into ``data/media`` by hand
    is theirs, and a sweep that removed it would be janki deleting a file it
    never created. Content-addressed naming is what makes this safe — a file is
    unreferenced exactly when no record names it, and editing a sentence changes
    the name rather than the contents.
    """
    audio_dir = media_dir / AUDIO_SUBDIR
    if not audio_dir.is_dir():
        return []
    referenced = set()
    for record in records:
        if record.audio:
            referenced.add(Path(record.audio).name)
        for example in record.examples:
            if example.audio:
                referenced.add(Path(example.audio).name)
    removed = []
    for path in sorted(audio_dir.glob("janki-*")):
        if path.name not in referenced:
            path.unlink()
            removed.append(path)
    if book is not None and removed:
        # Or the ledger outlives the media: an entry for a deleted file goes on
        # answering "already recorded" and nothing ever synthesizes it again.
        book.forget_audio_files(path.name for path in removed)
    return removed
