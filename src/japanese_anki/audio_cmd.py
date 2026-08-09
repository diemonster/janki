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
    "generate_audio",
    "media_relative",
    "prune_unreferenced",
]


class AudioError(JankiError):
    """Audio could not be generated, or was asked for in a way janki refuses."""


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
    #: Examples skipped because their furigana was never confirmed (M4.2's flag).
    unverified: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    #: Clips that already existed and were left alone.
    up_to_date: int = 0

    @property
    def file_count(self) -> int:
        return sum(len(names) for names in self.written.values())


def _write(path: Path, data: bytes) -> None:
    if not data:
        raise AudioError(f"The provider returned no audio for {path.name}.")
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
    if not force and book.has_audio(record.id, of="word", content_fp=content_fp):
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

    name = f"janki-{ledger_mod.word_audio_filename_fingerprint(record)}.wav"
    _write(audio_dir / name, data)
    book.record_audio(
        record.id,
        file=name,
        of="word",
        provider=provider.name,
        voice=provider.voice,
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
    flagged = _unverified_fingerprints(record)
    examples: list[ExampleSentence] = []
    changed = False

    for example in record.examples:
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
        if not force and book.has_audio(
            record.id, of="example", content_fp=content_fp
        ):
            result.up_to_date += 1
            examples.append(example)
            continue

        data = provider.synthesize(example.japanese, forced_accent=False)
        name = (
            f"janki-{ledger_mod.example_audio_filename_fingerprint(record, example)}.wav"
        )
        _write(audio_dir / name, data)
        book.record_audio(
            record.id,
            file=name,
            of="example",
            provider=provider.name,
            voice=provider.voice,
            content_fp=content_fp,
        )
        result.written.setdefault(record.id, []).append(name)
        examples.append(
            replace(example, audio=media_relative(audio_dir / name, media_dir))
        )
        changed = True

    if not changed:
        return record
    return replace(record, examples=examples)


def generate_audio(
    records: Sequence[VocabularyRecord],
    *,
    provider: SpeechProvider,
    book: ledger_mod.Ledger,
    media_dir: Path,
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
    """
    if not (words or examples):
        raise AudioError(
            "janki audio needs --words, --examples, or both: they are different "
            "recordings made different ways, and neither is the obvious default."
        )

    result = AudioResult(records=list(records))
    audio_dir = media_dir / AUDIO_SUBDIR
    wanted = _targets(result.records, ids)

    for index, record in enumerate(result.records):
        if record.id not in wanted:
            continue
        updated = record
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
                provider=provider,
                book=book,
                audio_dir=audio_dir,
                media_dir=media_dir,
                result=result,
                force=force,
            )
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
    records: Iterable[VocabularyRecord], media_dir: Path
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
    return removed
