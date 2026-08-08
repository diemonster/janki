"""Operational metadata about records, deliberately kept out of the records.

`vocabulary.json` holds card content (human-owned); the ledger holds what the
tools did to it — when a record arrived, which sources have mentioned it, which
enrichment passes ran, which audio files exist for which content, and which
decks it has been exported to. Nothing here is card content and nothing in a
record is provenance. See docs/DESIGN_V2.md "The ledger".

File shape (`data/ledger.json`)::

    {
      "version": 1,
      "records": {
        "word:話す:はなす": {
          "added_at": "2026-08-06",
          "sources": [{"type": "shirabe", "ref": "export.csv", "seen_at": "2026-08-06"}],
          "enriched": [{"at": "...", "kind": "jpdb", "model": "jpdb", "fields": [...]}],
          "audio": [{"file": "janki-<fp>.wav", "of": "word", "provider": "voicevox",
                     "voice": 46, "content_fp": "<fp>", "at": "..."}],
          "exports": {"personal-vocabulary": "2026-08-12"}
        }
      },
      "pending_batches": {}
    }

`enriched` is a **list** (it is a single object in the design document): the
jpdb pass and the AI pass each append, so neither overwrites the other's record
of what it wrote.

Mutators only touch memory and report whether they changed anything; call
:meth:`Ledger.save` once when a command is done. Every mutator is idempotent,
and idempotent by *identity* rather than by date: re-running the same
enrichment pass or re-recording byte-identical audio next month is still a
no-op, and the stored date stays the first run's. Re-running an import or a
build must not grow the file, ever, not just today.

:meth:`Ledger.save` is a whole-file rewrite, so **load once per command and
save once**. Two live :class:`Ledger` objects on one path would otherwise lose
whichever saved first — silently, since the atomic writer guarantees the loser
still finds a well-formed file. ``save`` refuses to overwrite a file that
changed since it was read rather than clobber it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from japanese_anki.errors import JankiError
from japanese_anki.identifiers import short_fingerprint
from japanese_anki.io import DataError, atomic_write_text
from japanese_anki.models import ExampleSentence, VocabularyRecord


class LedgerError(JankiError):
    pass


LEDGER_VERSION = 1

# Who did the enriching. The kind is what keeps a jpdb pass and an AI pass from
# being mistaken for each other; the model is which one of them ran. ``polish``
# is separate from ``ai`` although the same model does it: it is the one pass
# that rewrites a field rather than filling it, and "this record's glosses were
# replaced by a model" is a different fact about a record than "its examples
# were written by one".
ENRICHMENT_KINDS: tuple[str, ...] = ("jpdb", "ai", "polish")

# What a piece of audio says: the word itself, or one example sentence.
AUDIO_KINDS: tuple[str, ...] = ("word", "example")


def _iso_date(value: str | None) -> str:
    """Validate an ISO ``YYYY-MM-DD`` date, defaulting to today."""
    if value is None:
        return date.today().isoformat()
    try:
        parsed = date.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise LedgerError(f"Ledger dates must look like YYYY-MM-DD, got {value!r}") from exc
    # fromisoformat accepts far more than the one format the ledger stores: the
    # compact form '20260806' and the week date '2026-W32-4' both parse, and
    # both normalise to 2026-08-06 without complaint. (A timestamp is rejected
    # outright on 3.11+.) Anything that does not round-trip is not a plain day.
    if parsed.isoformat() != str(value):
        raise LedgerError(f"Ledger dates must look like YYYY-MM-DD, got {value!r}")
    return parsed.isoformat()


def _record_key(record_id: str) -> str:
    key = str(record_id).strip()
    if not key:
        raise LedgerError("A ledger entry needs a record id")
    return key


# ---------------------------------------------------------------------------
# Fingerprints — the only place these formulas exist
#
# Two families, deliberately different (docs/IMPLEMENTATION_PLAN.md
# "Conventions"). Callers import these; nobody restates the formulas:
#
# * *Filename* fingerprints are stable addresses. A file's name changes only
#   when the thing it belongs to changes identity, so editing an example
#   orphans its old audio file (visible, flagged) instead of silently
#   re-binding an existing card to the wrong sentence.
# * *Content* fingerprints detect staleness. They cover what was actually
#   spoken, so a changed reading, accent, or sentence makes the recorded audio
#   detectably out of date even though its filename may be unchanged.
# ---------------------------------------------------------------------------


def _selected_pitch_pattern(record: Any) -> str:
    """The accent pattern word audio would be generated with, or ``""``.

    ``pitch_accent`` and ``audio_accent`` arrive with the schema change in
    M2.2; until then every record answers ``""`` and word audio is fingerprinted
    on its reading alone. Once patterns exist, audio recorded without one is
    correctly reported as stale.
    """
    override = str(getattr(record, "audio_accent", "") or "").strip()
    if override:
        return override
    patterns = getattr(record, "pitch_accent", None) or []
    return str(patterns[0]).strip() if patterns else ""


def word_audio_filename_fingerprint(record: VocabularyRecord) -> str:
    """Address of a record's word audio: ``fp(record.id)``."""
    return short_fingerprint(record.id)


def example_audio_filename_fingerprint(
    record: VocabularyRecord, example: ExampleSentence
) -> str:
    """Address of one example's audio: ``fp(record.id + example.japanese)``."""
    return short_fingerprint(record.id + example.japanese)


def word_audio_content_fingerprint(record: VocabularyRecord) -> str:
    """What word audio says: ``fp(reading + selected pitch pattern)``."""
    return short_fingerprint(record.reading + _selected_pitch_pattern(record))


def example_audio_content_fingerprint(example: ExampleSentence) -> str:
    """What example audio says: ``fp(example.japanese)``."""
    return short_fingerprint(example.japanese)


def _without(reference: dict[str, Any], key: str) -> dict[str, Any]:
    """A reference minus its date field, which is its identity.

    ``key`` is the date key of that reference (``at`` or ``seen_at``). Dates say
    when a mutator ran, never what it recorded, so comparing them would make
    every mutator idempotent within one calendar day and no longer.
    """
    return {name: value for name, value in reference.items() if name != key}


def _checked_details(caller: str, details: dict[str, Any]) -> dict[str, Any]:
    """Refuse detail the ledger could not write, naming the caller's own key.

    Without this the failure surfaces much later, inside :meth:`Ledger.save`, as
    a bare ``TypeError`` from ``json.dumps`` — which is not a ``JankiError``, so
    ``cli.main`` does not catch it and the user gets a traceback instead of
    ``error: ...``. Every other failure in this module is a clean LedgerError.
    """
    for key, value in details.items():
        try:
            json.dumps(value)
        except (TypeError, ValueError) as exc:
            raise LedgerError(
                f"{caller} detail '{key}' is not JSON-serialisable "
                f"({type(value).__name__}); the ledger is a JSON file, so convert it "
                "to a string or a number at the call site"
            ) from exc
    return details


@dataclass
class Ledger:
    """An in-memory ledger bound to the file it was loaded from."""

    path: Path
    records: dict[str, dict[str, Any]] = field(default_factory=dict)
    pending_batches: dict[str, Any] = field(default_factory=dict)
    # Top-level keys this version does not know about. A later janki (or a
    # human) may add a whole section beside ``records``; dropping it because
    # ``save`` rebuilds the payload from three literals would delete data this
    # version never even read.
    extra: dict[str, Any] = field(default_factory=dict)
    # Exactly what :func:`load` read, so ``save`` can tell whether the file
    # changed underneath it. ``None`` means the file was absent.
    baseline: str | None = field(default=None, compare=False, repr=False)
    # Only a ledger that read the file has something to compare against; one
    # built directly is in-memory state that never claimed to mirror a file.
    guarded: bool = field(default=False, compare=False, repr=False)

    # -- persistence -------------------------------------------------------

    def _text_on_disk(self) -> str | None:
        try:
            return self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise LedgerError(
                f"Could not read ledger {self.path}: {exc.strerror or exc}"
            ) from exc

    def save(self) -> None:
        """Write the whole file atomically, in a stable order.

        ``sort_keys`` orders record ids — and every nested key, including keys
        a caller passed as extra detail — so a ledger diff only ever shows what
        actually changed. Unrecognised top-level keys are written back
        unchanged: a section this version does not understand is not this
        version's to delete.

        This is a whole-file read-modify-write, so it first checks the file is
        still the one that was read. A second in-flight ledger on the same path
        — two commands, or one command that called ``load`` twice — would
        otherwise silently drop everything the first one wrote.

        **Every failure to save is a LedgerError**, including the filesystem
        ones the atomic writer reports as ``DataError``. This module owns the
        invariant "the ledger did not get written", and its callers say so in
        their own words — ``cli._save_ledger`` turns it into a warning over a
        full transcript rather than an ``error:`` that reads as "nothing
        happened". A ``DataError`` escaping from here is a *sibling* of
        LedgerError under JankiError, so it would sail straight past every one
        of those handlers; permission denied, a read-only mount and ENOSPC are
        the shapes a real save failure actually has.
        """
        if self.guarded and self._text_on_disk() != self.baseline:
            raise LedgerError(
                f"Ledger {self.path} changed on disk since it was read; saving now "
                "would discard those changes. Load the ledger once per command and "
                "save once, then re-run."
            )
        payload = {
            **self.extra,
            "version": LEDGER_VERSION,
            "records": self.records,
            "pending_batches": self.pending_batches,
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        try:
            atomic_write_text(self.path, text)
        except DataError as exc:
            raise LedgerError(str(exc)) from exc
        # What we just wrote is now what we have read: saving twice in one
        # command is fine, it is saving over *someone else* that is not.
        self.baseline = text

    # -- mutators ----------------------------------------------------------

    def _entry(self, record_id: str, at: str | None = None) -> dict[str, Any]:
        key = _record_key(record_id)
        entry = self.records.get(key)
        if entry is None:
            entry = {
                "added_at": _iso_date(at),
                "sources": [],
                "enriched": [],
                "audio": [],
                "exports": {},
            }
            self.records[key] = entry
        return entry

    def record_added(self, record_id: str, *, at: str | None = None) -> bool:
        """Register a record's first sighting. Returns whether it was new.

        A record already in the ledger keeps its original ``added_at``: this is
        the date the record entered the collection, not the date of the most
        recent import that mentioned it.
        """
        if _record_key(record_id) in self.records:
            return False
        self._entry(record_id, at)
        return True

    def record_source_seen(
        self,
        record_id: str,
        source_type: str,
        ref: str = "",
        *,
        seen_at: str | None = None,
        **details: Any,
    ) -> bool:
        """Append a source reference; identical references are a no-op.

        This is where later sightings go — a re-import never overwrites the
        record's own ``source``, so provenance accumulates here instead of
        being lost. Two references are the same reference when everything but
        ``seen_at`` matches, so re-importing the same file tomorrow adds
        nothing and the stored date stays the first sighting of that source.
        Callers with genuinely changing detail (review counts, say) get a new
        entry each time, which is why such detail belongs in the record's
        ``raw_fields``, not in a source reference.
        """
        kind = str(source_type).strip()
        if not kind:
            raise LedgerError("A source reference needs a type (shirabe, jpdb, pdf, ...)")
        _checked_details("record_source_seen", details)
        entry = self._entry(record_id, seen_at)
        reference: dict[str, Any] = {
            "type": kind,
            "ref": str(ref),
            **details,
            "seen_at": _iso_date(seen_at),
        }
        sources = entry.setdefault("sources", [])
        target = _without(reference, "seen_at")
        for existing in sources:
            if isinstance(existing, dict) and _without(existing, "seen_at") == target:
                return False
        sources.append(reference)
        return True

    def record_enriched(
        self,
        record_id: str,
        *,
        kind: str,
        fields: Iterable[str],
        model: str | None = None,
        at: str | None = None,
    ) -> bool:
        """Note that an enrichment pass wrote ``fields``.

        Entries accumulate rather than replace: a jpdb pass and an AI pass
        describe different work, and neither may erase the other's record of
        what it wrote. A pass is identified by ``(kind, model, fields)`` and
        nothing else — re-running it changes nothing however long ago it last
        ran, and the stored ``at`` stays the first run's, the same way
        ``record_source_seen`` ignores ``seen_at``. Otherwise a monthly
        ``janki enrich`` would leave ``status`` reporting one record enriched
        five times by the same pass with the same fields.
        """
        if kind not in ENRICHMENT_KINDS:
            raise LedgerError(
                f"Unknown enrichment kind '{kind}'; expected one of "
                f"{', '.join(ENRICHMENT_KINDS)}"
            )
        names = sorted({str(name).strip() for name in fields if str(name).strip()})
        if not names:
            raise LedgerError(
                "record_enriched needs the field names the pass wrote; "
                "a pass that wrote nothing has nothing to record"
            )
        if kind == "jpdb":
            # jpdb is a dictionary lookup, not a model that could have been any
            # other model. Accepting one here is how an AI pass copy-pasted from
            # this call ends up recorded as jpdb work.
            if model is not None and str(model) != "jpdb":
                raise LedgerError(
                    f"record_enriched(kind='jpdb') writes model 'jpdb'; got {model!r}. "
                    "An enrichment pass driven by a model is kind='ai'."
                )
            model = "jpdb"
        elif model is None:
            raise LedgerError(f"record_enriched(kind='{kind}') needs the model that ran")
        reference = {
            "at": _iso_date(at),
            "kind": kind,
            "model": str(model),
            "fields": names,
        }
        identity = _without(reference, "at")
        entries = self._entry(record_id, at).setdefault("enriched", [])
        for existing in entries:
            if isinstance(existing, dict) and _without(existing, "at") == identity:
                return False
        entries.append(reference)
        return True

    def record_audio(
        self,
        record_id: str,
        *,
        file: str,
        of: str,
        provider: str,
        voice: int,
        content_fp: str,
        at: str | None = None,
        **details: Any,
    ) -> bool:
        """Record the audio file that currently holds ``content_fp``.

        Entries are keyed by file name: regenerating audio for the same
        content-addressed file (a new voice, say) replaces its entry rather
        than adding a second, so ``stale_audio`` never has to guess which of
        two entries describes the file on disk. An entry that differs only in
        ``at`` describes the same file saying the same thing, so re-recording
        it reports no change and keeps the original date — a re-run that
        regenerated nothing must not mark the ledger dirty.
        """
        if of not in AUDIO_KINDS:
            raise LedgerError(
                f"Unknown audio kind '{of}'; expected one of {', '.join(AUDIO_KINDS)}"
            )
        name = str(file).strip()
        if not name:
            raise LedgerError("An audio entry needs the media file name")
        try:
            voice_id = int(voice)
        except (TypeError, ValueError) as exc:
            raise LedgerError(f"Audio voice must be an integer, got {voice!r}") from exc
        _checked_details("record_audio", details)
        reference: dict[str, Any] = {
            "file": name,
            "of": of,
            "provider": str(provider),
            "voice": voice_id,
            "content_fp": str(content_fp),
            **details,
            "at": _iso_date(at),
        }
        entries = self._entry(record_id, at).setdefault("audio", [])
        for index, existing in enumerate(entries):
            if isinstance(existing, dict) and existing.get("file") == name:
                if _without(existing, "at") == _without(reference, "at"):
                    return False
                entries[index] = reference
                return True
        entries.append(reference)
        return True

    def record_export(self, record_id: str, deck_stem: str, *, at: str | None = None) -> bool:
        """Note that a deck build included this record.

        Keyed by deck file stem, not Anki deck name: stems are unique in the
        repo, deck names collide when two deck files fall back to the default.
        """
        stem = str(deck_stem).strip()
        if not stem:
            raise LedgerError("An export entry needs a deck file stem")
        when = _iso_date(at)
        exports = self._entry(record_id, at).setdefault("exports", {})
        if exports.get(stem) == when:
            return False
        exports[stem] = when
        return True

    def remove(self, record_id: str) -> bool:
        """Drop a record's entry entirely. Returns whether there was one."""
        return self.records.pop(_record_key(record_id), None) is not None

    # -- pending batches ---------------------------------------------------
    #
    # A batch runs for up to a day, in someone else's datacentre, with the
    # results waiting there until asked for. The ledger is what remembers a
    # submission across that gap: without the id, work that was paid for is
    # unreachable, and without the ids of the records it covers, a fetch cannot
    # tell which words a result belongs to.

    def record_batch(
        self,
        batch_id: str,
        *,
        kind: str,
        model: str,
        pending_ids: Iterable[str],
        force_fields: Iterable[str] = (),
        at: str | None = None,
    ) -> None:
        """Remember a submitted batch and the records it covers.

        Unlike ``record_enriched`` this deliberately *replaces* rather than
        accumulates for a given id: a batch is one thing that is either in
        flight or collected, and two entries for one id would mean the ledger
        disagreed with itself about which records are waiting on it.
        """
        if kind not in ENRICHMENT_KINDS:
            raise LedgerError(
                f"Unknown enrichment kind {kind!r}; expected one of "
                f"{', '.join(ENRICHMENT_KINDS)}"
            )
        ids = list(dict.fromkeys(str(item) for item in pending_ids))
        if not ids:
            raise LedgerError(
                "A pending batch with no record ids could never be applied to "
                "anything; nothing was recorded."
            )
        self.pending_batches[str(batch_id)] = {
            "kind": kind,
            "model": model,
            "submitted_at": _iso_date(at),
            "pending_ids": ids,
            # Stored because the fetch has to apply what the submit asked for.
            # Without it, submitting with --force-fields and fetching plainly
            # would report "nothing to fill" and throw away a paid answer.
            "force_fields": list(dict.fromkeys(str(item) for item in force_fields)),
        }

    def pending_batch(self) -> tuple[str, dict[str, Any]] | None:
        """The batch in flight, as ``(batch_id, entry)``, or ``None``.

        One at a time on purpose (DESIGN_V2, "Batch mode"): submitting a second
        while the first is out would leave two sets of pending ids overlapping,
        and a fetch unable to say which answer is the current one for a word.
        The first entry is returned rather than an arbitrary one so the answer
        is stable across runs.
        """
        for batch_id in self.pending_batches:
            entry = self.pending_batches[batch_id]
            return str(batch_id), dict(entry) if isinstance(entry, dict) else {}
        return None

    def clear_batch(self, batch_id: str) -> bool:
        """Forget a collected batch. Returns whether there was one to forget."""
        return self.pending_batches.pop(str(batch_id), None) is not None

    # -- queries -----------------------------------------------------------

    def _audio_entries(self, record_id: str) -> list[dict[str, Any]]:
        entry = self.records.get(str(record_id)) or {}
        return [item for item in (entry.get("audio") or []) if isinstance(item, dict)]

    def unexported(self, deck_stem: str, ids: Iterable[str]) -> list[str]:
        """Which of ``ids`` this deck has never been built with, in order."""
        stem = str(deck_stem).strip()
        if not stem:
            raise LedgerError("Exports are tracked per deck file stem; none was given")
        result: list[str] = []
        seen: set[str] = set()
        for record_id in ids:
            key = str(record_id)
            if key in seen:
                continue
            seen.add(key)
            entry = self.records.get(key) or {}
            if not (entry.get("exports") or {}).get(stem):
                result.append(key)
        return result

    def missing_audio(self, records: Iterable[VocabularyRecord]) -> list[str]:
        """Records with no word audio recorded at all.

        Word audio only — the design's "words missing audio". Example audio is
        optional and per-example, so counting a record as deficient for every
        unvoiced example would bury the word-level signal.
        """
        return [
            record.id
            for record in records
            if not any(entry.get("of") == "word" for entry in self._audio_entries(record.id))
        ]

    def stale_audio(self, records: Iterable[VocabularyRecord]) -> list[str]:
        """Records whose recorded audio no longer matches their content.

        Word audio goes stale when the reading or the selected accent changes;
        example audio goes stale when the sentence it was generated from is
        edited away (its content fingerprint matches none of the record's
        current examples). Entries of an unrecognized kind are left alone —
        this reports what it can prove.
        """
        result: list[str] = []
        for record in records:
            entries = self._audio_entries(record.id)
            if not entries:
                continue
            word_fp = word_audio_content_fingerprint(record)
            example_fps = {
                example_audio_content_fingerprint(example)
                for example in record.examples
                if example.japanese
            }
            for entry in entries:
                fingerprint = str(entry.get("content_fp") or "")
                kind = entry.get("of")
                if kind == "word":
                    stale = fingerprint != word_fp
                elif kind == "example":
                    stale = fingerprint not in example_fps
                else:
                    continue
                if stale:
                    result.append(record.id)
                    break
        return result

    @staticmethod
    def missing_enrichment(records: Iterable[VocabularyRecord]) -> list[str]:
        """Records that still need enrichment, defined purely by their content.

        Deliberately a static method: it never consults the ledger's own
        ``enriched`` entries. Those say who ran and what they wrote; they do
        not say the record is finished. A jpdb pass fills readings and accents
        and would otherwise hide a record from ``enrich --ai``, which is what
        actually writes examples and usage notes.
        """
        return [
            record.id
            for record in records
            if not any(example.japanese.strip() for example in record.examples)
            or not record.usage_notes.strip()
        ]


def load(path: Path) -> Ledger:
    """Read a ledger. A missing (or empty) file is an empty ledger, not an error."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return Ledger(path=path, baseline=None, guarded=True)
    except OSError as exc:
        raise LedgerError(f"Could not read ledger {path}: {exc.strerror or exc}") from exc

    if not text.strip():
        return Ledger(path=path, baseline=text, guarded=True)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LedgerError(f"Could not parse ledger {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise LedgerError(f"Ledger {path} must hold a JSON object at the top level")

    version = data.get("version", LEDGER_VERSION)
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise LedgerError(f"Ledger {path} has an unreadable version: {version!r}")
    if version > LEDGER_VERSION:
        raise LedgerError(
            f"Ledger {path} was written by a newer janki (version {version}, "
            f"this one understands {LEDGER_VERSION}); upgrade before writing to it"
        )

    records = data.get("records")
    if records is None:
        records = {}
    if not isinstance(records, dict):
        raise LedgerError(f"Ledger {path}: 'records' must be an object keyed by record id")
    for record_id, entry in records.items():
        if not isinstance(entry, dict):
            raise LedgerError(f"Ledger {path}: entry for '{record_id}' must be an object")

    pending_batches = data.get("pending_batches")
    if pending_batches is None:
        pending_batches = {}
    if not isinstance(pending_batches, dict):
        raise LedgerError(f"Ledger {path}: 'pending_batches' must be an object")

    return Ledger(
        path=path,
        records=dict(records),
        pending_batches=dict(pending_batches),
        extra={
            key: value
            for key, value in data.items()
            if key not in {"version", "records", "pending_batches"}
        },
        baseline=text,
        guarded=True,
    )
