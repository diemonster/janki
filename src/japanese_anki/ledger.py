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
                     "voice": 46, "speed": 1.0, "content_fp": "<fp>", "at": "..."},
                    {"file": "janki-<fp>.mp3", "of": "example", "provider": "openai",
                     "voice": "onyx", "speed": 1.0, "content_fp": "<fp>", "at": "...",
                     "settings": {"model": "...", "instructions": "..."}}],
          "exports": {"personal-vocabulary": "2026-08-12",
                      "verbs": {"at": "2026-08-12", "missing": ["audio"]}}
        }
      },
      "pending_batches": {}
    }

`enriched` is a **list** (it is a single object in the design document): the
jpdb pass and the AI pass each append, so neither overwrites the other's record
of what it wrote.

Two values in that shape are conditional, and both stay absent when there is
nothing to say so no existing ledger moves:

* **`exports[stem]`** is a plain date when the build shipped a complete record,
  and `{"at": ..., "missing": [...]}` when it did not — `missing` naming what
  the record lacked (`audio`, `examples`, `accent`). Dates are days, so
  comparing them cannot tell that a record shipped at 09:00 was voiced at
  10:00; what it went out without is a fact rather than an inference. Readers
  must handle both forms.
* **`audio[].settings`** carries anything beyond voice and rate that decides how
  a clip sounds — for OpenAI, the model and the prose `instructions` that are
  its only pace control. Absent for engines with nothing to add.

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
from collections.abc import Container, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from japanese_anki.errors import JankiError
from japanese_anki.identifiers import short_fingerprint
from japanese_anki.io import DataError, atomic_write_text, exclusive_path_lock
from japanese_anki.models import ExampleSentence, VocabularyRecord, example_accepted


class LedgerError(JankiError):
    pass


LEDGER_VERSION = 1
BATCH_RECOVERY_VERSION = 1
_BATCH_RECOVERY_MARKER = "janki-polish-batch-recovery"

# Who did the enriching. The kind is what keeps a jpdb pass and an AI pass from
# being mistaken for each other; the model is which one of them ran. ``polish``
# is separate from ``ai`` although the same model does it: it is the one pass
# that rewrites a field rather than filling it, and "this record's glosses were
# replaced by a model" is a different fact about a record than "its examples
# were written by one".
ENRICHMENT_KINDS: tuple[str, ...] = ("jpdb", "ai", "polish", "human")

# ``human`` is the fourth because a dictionary can be wrong and a person has to
# be able to say so. jpdb's parse reads 日本語 as にっぽんご; the language is
# にほんご, and without a way to record "a human overruled this" the only
# options were to accept a reading nobody uses or to leave a correct sentence
# unvoiced forever. The pitch side has had this since M5.1, as ``audio_accent``.

# What a piece of audio says: the word itself, or one example sentence.
AUDIO_KINDS: tuple[str, ...] = ("word", "example")


@dataclass(frozen=True)
class BatchRecovery:
    """Durable state for the gap between a polish batch and its ledger write."""

    path: Path
    batch_id: str
    entry: dict[str, Any]
    phase: str
    accepted_meanings: dict[str, list[str]]
    retry_ids: list[str]
    text: str = field(repr=False, compare=False)


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


#: The structured keys of a record entry, and the type each must hold. A ledger
#: is committed and hand-edited often enough that one of these arriving as
#: ``null`` — or as anything else — is a real input, and a reader that trusts
#: the shape turns it into a ``TypeError`` or an ``AttributeError`` rather than
#: the clean ``LedgerError`` this module promises.
_ENTRY_SHAPE: dict[str, type] = {
    "sources": list,
    "enriched": list,
    "audio": list,
    "exports": dict,
}

#: A repair parks every unreadable value; none of these keys is reconstructible.
#:
#: An earlier version exempted ``sources`` and ``audio`` on the grounds that
#: ``status --rebuild`` refills them. It does not, in three separate ways:
#:
#: * ``rebuild`` iterates the *record universe*, so an entry whose record is no
#:   longer in ``vocabulary.json`` — hand-deleted, renamed, or in a deck that
#:   was removed — is never visited at all, and ``Ledger.remove`` is called from
#:   one narrow path, so those entries are an ordinary state rather than a bug.
#: * A rebuilt ``sources`` is a *single* reference dated today. ``sources`` is
#:   append-only by design (DESIGN_V2): a word imported from Shirabe and later
#:   mined from jpdb has two sightings with two real dates, and a rebuild can
#:   only ever restore the one the record itself carries.
#: * A rebuilt ``audio`` entry is strictly weaker than the one it replaces —
#:   ``provider`` unknown, ``voice`` and ``speed`` the ``-1`` sentinels (so the
#:   next ``janki audio`` re-synthesizes everything), no ``accent_unverified``
#:   marker, an empty content fingerprint for any accented record, and nothing
#:   at all for a superseded clip.
#:
#: So the copy is unconditional. It is cheap, ``_shape_problems`` ignores the
#: extra key, and it is the only thing that makes ``_shape_error``'s promise
#: — nothing is thrown away — true.


def quarantine_key(key: str, taken: Container[str] = ()) -> str:
    """Where :func:`load` parks a value it could not read but must not lose.

    Suffixed when the obvious name is already in use, because a second repair
    of the same key must not overwrite what the first one saved — which would
    destroy the history this whole mechanism exists to keep, silently.
    """
    base = f"{key}_unreadable"
    if base not in taken:
        return base
    index = 2
    while f"{base}_{index}" in taken:
        index += 1
    return f"{base}_{index}"


def _shape_problems(entry: dict[str, Any]) -> list[tuple[str, str]]:
    """``(key, type name)`` for every structured key holding the wrong type."""
    return [
        (key, type(entry[key]).__name__)
        for key, want in _ENTRY_SHAPE.items()
        if key in entry and not isinstance(entry[key], want)
    ]


def _shape_error(path: Path, record_id: str, problems: list[tuple[str, str]]) -> LedgerError:
    detail = ", ".join(
        f"{key!r} as {got}, not a {_ENTRY_SHAPE[key].__name__}" for key, got in problems
    )
    # Every slot, not the first: an entry can carry two bad keys, and a message
    # naming one leaves the reader with no reason to look for the other.
    slots = ", ".join(f"'{key}_unreadable'" for key, _ in problems) or "'<key>_unreadable'"
    return LedgerError(
        f"Ledger {path}: record {record_id!r} has {detail}. That file is "
        "machine-written; 'janki status --rebuild' repairs entries like this "
        "in place. Nothing is thrown away — each unreadable value is parked "
        f"beside the key it came from, under {slots} (numbered if an earlier "
        "repair already used that name), and the repair prints the exact names "
        "it used."
    )


def _voice_key(voice: Any) -> int | str:
    """A voice as the ledger stores it: an id if numeric, else its name.

    Engines disagree about what a voice is — VOICEVOX numbers them, OpenAI
    names them — and the ledger records what was used rather than normalising
    it into whichever shape came first.
    """
    # Narrow on purpose. `str(voice)` for anything at all would write
    # `"voice": "None"` — or `"{'id': 13}"` — into a committed file and report
    # success, where the `int()` this replaced raised.
    if isinstance(voice, bool) or not isinstance(voice, int | str):
        raise LedgerError(f"Audio voice must be an id or a name, got {voice!r}")
    if isinstance(voice, int):
        return voice
    name = voice.strip()
    if not name:
        raise LedgerError("An audio entry needs the voice it was spoken in")
    return name


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

    Delegates to :func:`japanese_anki.pitch.select_pattern`, which is the one
    definition — this fingerprint is computed *over* that choice, so a second
    definition would mean audio generated under one rule is fingerprinted under
    another and never reads as stale when the accent changes. Imported here
    rather than at module scope because ``pitch`` imports ``models`` and this
    module is imported by half the package; the local import keeps the graph
    flat and costs one dict lookup per call.

    ``None`` becomes ``""``: a record with no pattern is fingerprinted on its
    reading alone, and once a pattern arrives the recorded audio is correctly
    reported as stale.
    """
    from japanese_anki.pitch import select_pattern

    return select_pattern(record) or ""


def word_audio_filename_fingerprint(record: VocabularyRecord) -> str:
    """Address of a record's word audio: ``fp(record.id)``."""
    return short_fingerprint(record.id)


def example_audio_filename_fingerprint(
    record: VocabularyRecord, example: ExampleSentence
) -> str:
    """Address of one example's audio: ``fp(record.id + example.japanese)``."""
    return short_fingerprint(record.id + example.japanese)


def word_audio_content_fingerprint(record: VocabularyRecord) -> str:
    """What word audio actually *says* — the utterance, not the stored pattern.

    The AquesTalk notation when the pattern can be forced, and the bare reading
    when it cannot, because those are the two different things the engine is
    sent and the clip has to go stale between them.

    Fingerprinting the raw pattern instead was a trap. ``short_fingerprint``
    normalizes NFKC and ``pitch.select_pattern`` does not, so a fullwidth
    ``ＬＨＬ`` — what a Japanese IME produces while you are already typing kana
    — and an ASCII ``LHL`` hash identically. `to_aquestalk` refuses the first
    and accepts the second, so a curator who typed the pattern in full width,
    got a guessed clip, then corrected it to ASCII would find the clip
    unchanged: same fingerprint, so `_is_current` calls it up to date, no
    warning fires because the pattern now parses, and the ledger goes on saying
    `accent_unverified` forever. Over the utterance the two differ, which is
    the whole point.
    """
    pattern = _selected_pitch_pattern(record)
    return short_fingerprint(record.reading + _spoken_form(record, pattern))


def _spoken_form(record: VocabularyRecord, pattern: str) -> str:
    """The AquesTalk string a forced clip would use, or ``""`` for a guess."""
    if not pattern:
        return ""
    from japanese_anki import pitch

    try:
        return pitch.to_aquestalk(record.reading, pattern)
    except pitch.PitchError:
        # Unusable: the clip is voiced with the engine's own accent, which is
        # the same utterance a record with no pattern at all produces.
        return ""


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


def batch_recovery_path(ledger_path: Path) -> Path:
    """Return the one recovery journal associated with ``ledger_path``."""
    path = Path(ledger_path)
    suffix = path.suffix or ".json"
    stem = path.stem if path.suffix else path.name
    return path.with_name(f"{stem}.polish-batch-recovery{suffix}")


def load_batch_recovery(ledger_path: Path) -> BatchRecovery | None:
    """Load a janki-owned polish recovery journal, refusing unknown content."""
    path = batch_recovery_path(ledger_path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise LedgerError(
            f"Could not read batch recovery journal {path}: {exc.strerror or exc}"
        ) from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LedgerError(f"Could not parse batch recovery journal {path}: {exc}") from exc
    if not isinstance(data, dict) or data.get("marker") != _BATCH_RECOVERY_MARKER:
        raise LedgerError(
            f"Refusing to treat {path} as janki batch recovery state because its "
            "ownership marker is missing. Move the unrelated file aside and retry."
        )
    if data.get("version") != BATCH_RECOVERY_VERSION:
        raise LedgerError(
            f"Batch recovery journal {path} has unsupported version "
            f"{data.get('version')!r}. Upgrade janki before changing it."
        )
    batch_id = data.get("batch_id")
    entry = data.get("batch")
    phase = data.get("phase")
    accepted = data.get("accepted_meanings", {})
    retry_ids = data.get("retry_ids", [])
    if not isinstance(batch_id, str) or not batch_id.strip():
        raise LedgerError(f"Batch recovery journal {path} has no batch id")
    if not isinstance(entry, dict) or entry.get("kind") != "polish":
        raise LedgerError(f"Batch recovery journal {path} has no polish batch descriptor")
    if phase not in {"submitted", "applied"}:
        raise LedgerError(f"Batch recovery journal {path} has unknown phase {phase!r}")
    if not isinstance(accepted, dict) or any(
        not isinstance(record_id, str)
        or not isinstance(meanings, list)
        or any(not isinstance(item, str) for item in meanings)
        for record_id, meanings in accepted.items()
    ):
        raise LedgerError(
            f"Batch recovery journal {path}: 'accepted_meanings' must map ids to strings"
        )
    if not isinstance(retry_ids, list) or any(
        not isinstance(item, str) for item in retry_ids
    ):
        raise LedgerError(
            f"Batch recovery journal {path}: 'retry_ids' must be a list of strings"
        )
    return BatchRecovery(
        path=path,
        batch_id=batch_id,
        entry=dict(entry),
        phase=phase,
        accepted_meanings={
            str(record_id): list(meanings) for record_id, meanings in accepted.items()
        },
        retry_ids=list(retry_ids),
        text=text,
    )


def write_batch_recovery(
    ledger_path: Path,
    batch_id: str,
    entry: Mapping[str, Any],
    *,
    accepted_meanings: Mapping[str, Iterable[str]] | None = None,
    retry_ids: Iterable[str] | None = None,
) -> BatchRecovery:
    """Atomically preserve enough polish state for ``--batch-fetch`` to resume.

    ``retry_ids is None`` is the just-submitted phase.  Supplying it, including
    an empty iterable, marks the records/ledger handoff after local review.
    Existing non-janki content is never overwritten.
    """
    key = str(batch_id).strip()
    if not key:
        raise LedgerError("A batch recovery journal needs a batch id")
    descriptor = dict(entry)
    if descriptor.get("kind") != "polish":
        raise LedgerError("Only a meaning-polish batch can use polish recovery state")
    phase = "submitted" if retry_ids is None else "applied"
    payload = {
        "marker": _BATCH_RECOVERY_MARKER,
        "version": BATCH_RECOVERY_VERSION,
        "batch_id": key,
        "batch": descriptor,
        "phase": phase,
        "accepted_meanings": {
            str(record_id): [str(item) for item in meanings]
            for record_id, meanings in (accepted_meanings or {}).items()
        },
        "retry_ids": []
        if retry_ids is None
        else list(dict.fromkeys(str(item) for item in retry_ids)),
    }
    try:
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    except (TypeError, ValueError) as exc:
        raise LedgerError(
            f"Could not serialise batch recovery state for {key}: {exc}"
        ) from exc
    path = batch_recovery_path(ledger_path)
    try:
        with exclusive_path_lock(path):
            existing = load_batch_recovery(ledger_path)
            if existing is not None:
                if existing.text == text:
                    return existing
                if existing.batch_id != key:
                    raise LedgerError(
                        f"Batch {existing.batch_id} still has recovery state in "
                        f"{existing.path}; fetch or forget it before recording "
                        "another batch."
                    )
                raise LedgerError(
                    f"Recovery state for batch {key} changed while this command "
                    "was reviewing it. Another fetch may still be writing; its "
                    "journal was kept intact. Let that command finish, then fetch "
                    "again."
                )
            atomic_write_text(path, text)
            written = load_batch_recovery(ledger_path)
            if written is None:  # pragma: no cover - atomic write just landed
                raise LedgerError(
                    f"Batch recovery journal {path} disappeared after writing"
                )
            return written
    except DataError as exc:
        raise LedgerError(str(exc)) from exc


def clear_batch_recovery(
    ledger_path: Path,
    batch_id: str,
    *,
    expected: BatchRecovery | None = None,
) -> bool:
    """Remove only the validated journal for ``batch_id``.

    Loading first is deliberate: this command must never delete a same-named
    artifact that it did not create or a recovery belonging to another batch.
    """
    try:
        with exclusive_path_lock(batch_recovery_path(ledger_path)):
            recovery = load_batch_recovery(ledger_path)
            if recovery is None:
                return False
            if recovery.batch_id != str(batch_id):
                raise LedgerError(
                    f"Refusing to remove recovery for batch {recovery.batch_id} while "
                    f"finishing batch {batch_id}."
                )
            if expected is not None and recovery.text != expected.text:
                raise LedgerError(
                    f"Recovery state for batch {batch_id} changed after this command "
                    "read it. The newer journal was kept intact; let the other "
                    "fetch finish, then retry."
                )
            try:
                recovery.path.unlink()
            except FileNotFoundError:
                return True
            except OSError as exc:
                raise LedgerError(
                    f"Could not remove completed batch recovery journal "
                    f"{recovery.path}: {exc.strerror or exc}"
                ) from exc
    except DataError as exc:
        raise LedgerError(str(exc)) from exc
    return True


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
    # Record ids whose structured keys ``load(repair=True)`` coerced back into
    # shape, so ``status --rebuild`` can report what it fixed rather than fixing
    # it silently. Empty on every ordinary load, which refuses instead.
    repaired: dict[str, list[str]] = field(
        default_factory=dict, compare=False, repr=False
    )

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
        payload = {
            **self.extra,
            "version": LEDGER_VERSION,
            "records": self.records,
            "pending_batches": self.pending_batches,
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        try:
            with exclusive_path_lock(self.path):
                if self.guarded and self._text_on_disk() != self.baseline:
                    raise LedgerError(
                        f"Ledger {self.path} changed on disk since it was read; "
                        "saving now would discard those changes. Load the ledger "
                        "once per command and save once, then re-run."
                    )
                atomic_write_text(self.path, text)
        except DataError as exc:
            raise LedgerError(str(exc)) from exc
        # What we just wrote is now what we have read: saving twice in one
        # command is fine, it is saving over *someone else* that is not.
        self.baseline = text

    # -- mutators ----------------------------------------------------------

    def _entry(self, record_id: str, at: str | None = None) -> dict[str, Any]:
        """The entry for a record, created if absent.

        The shape check here is a backstop, not the gate: :func:`load` refuses a
        misshapen ledger before any caller has written anything, because a
        refusal raised from a mutator fires *after* ``vocabulary.json`` has been
        replaced — leaving an import that did almost everything and reported
        nothing. This still runs for a ``Ledger`` built directly rather than
        loaded, which has no file to have been checked.
        """
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
        problems = _shape_problems(entry)
        if problems:
            raise _shape_error(self.path, record_id, problems)
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
        voice: int | str,
        content_fp: str,
        speed: float,
        settings: Mapping[str, str] | None = None,
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

        ``voice`` and ``speed`` are stored because both are audible and neither
        is in ``content_fp``, which covers what was said and deliberately not
        who said it or how fast. Without them on disk, a re-voice interrupted
        part way — the OOM a small VM hits routinely — leaves half the
        collection in one voice and half in another, every clip's fingerprint
        still correct, and no way for ``janki status`` or a resumed run to tell
        which is which.
        """
        if of not in AUDIO_KINDS:
            raise LedgerError(
                f"Unknown audio kind '{of}'; expected one of {', '.join(AUDIO_KINDS)}"
            )
        name = str(file).strip()
        if not name:
            raise LedgerError("An audio entry needs the media file name")
        voice_id = _voice_key(voice)
        try:
            rate = float(speed)
        except (TypeError, ValueError) as exc:
            raise LedgerError(f"Audio speed must be a number, got {speed!r}") from exc
        # Type-checked but not range-checked, exactly like ``voice`` beside it:
        # ``status --rebuild`` stores a negative sentinel for both, because a
        # file on disk carries no record of how fast it was spoken and a
        # plausible guess in the ledger is worse than an admitted unknown.
        _checked_details("record_audio", details)
        reference: dict[str, Any] = {
            "file": name,
            "of": of,
            "provider": str(provider),
            "voice": voice_id,
            "speed": rate,
            "content_fp": str(content_fp),
            # Only when there is something to say, so a VOICEVOX entry keeps
            # exactly the shape it has always had and no committed ledger moves.
            **({"settings": dict(settings)} if settings else {}),
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

    def record_export(
        self,
        record_id: str,
        deck_stem: str,
        *,
        gaps: Iterable[str] = (),
        at: str | None = None,
    ) -> bool:
        """Note that a deck build included this record.

        Keyed by deck file stem, not Anki deck name: stems are unique in the
        repo, deck names collide when two deck files fall back to the default.

        ``gaps`` names what the record was *missing* when it shipped — no
        audio, no example, no accent. Recording it is what makes
        :meth:`shipped_incomplete` exact: dates are days, so a record shipped
        at 09:00 and voiced at 10:00 the same day can never be caught by
        comparing them, and that record is silently wrong forever. What it
        lacked at build time is a fact, not an inference.
        """
        stem = str(deck_stem).strip()
        if not stem:
            raise LedgerError("An export entry needs a deck file stem")
        when = _iso_date(at)
        missing = sorted({str(gap).strip() for gap in gaps if str(gap).strip()})
        exports = self._entry(record_id, at).setdefault("exports", {})
        # The value is a date when there is nothing more to say, so every
        # existing ledger keeps its shape and only a gap makes it grow.
        value: Any = {"at": when, "missing": missing} if missing else when
        if exports.get(stem) == value:
            return False
        exports[stem] = value
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
        prompt_fingerprints: Mapping[str, str] | None = None,
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
        entry: dict[str, Any] = {
            "kind": kind,
            "model": model,
            "submitted_at": _iso_date(at),
            "pending_ids": ids,
            # Stored because the fetch has to apply what the submit asked for.
            # Without it, submitting with --force-fields and fetching plainly
            # would report "nothing to fill" and throw away a paid answer.
            "force_fields": list(dict.fromkeys(str(item) for item in force_fields)),
        }
        if prompt_fingerprints is not None:
            fingerprints = {
                str(record_id): str(fingerprint)
                for record_id, fingerprint in prompt_fingerprints.items()
            }
            missing = [record_id for record_id in ids if not fingerprints.get(record_id)]
            extra = [record_id for record_id in fingerprints if record_id not in ids]
            if missing or extra:
                details = []
                if missing:
                    details.append(f"missing {', '.join(missing)}")
                if extra:
                    details.append(f"unknown {', '.join(extra)}")
                raise LedgerError(
                    "Batch prompt fingerprints must exactly cover its pending ids: "
                    + "; ".join(details)
                )
            entry["prompt_fingerprints"] = fingerprints
        self.pending_batches[str(batch_id)] = entry

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

    def record_batch_retry(self, batch_id: str, retry_ids: Iterable[str]) -> None:
        """Narrow a held batch to the records still worth another look.

        A batch is held for exactly one reason — some answers did not parse —
        so a second fetch has exactly one job, and this is the list of it.

        Deliberately the *failures* rather than the successes. Recording what
        landed sounds equivalent and is not: a large fetch lands in a staging
        file rather than in records, a ledger write can fail after the records
        are already on disk, and a fetch that wrote nothing records nothing —
        in each case "what landed" is unknown or not yet settled, while "what
        did not parse" is exact at the moment the batch is held. Replaced
        rather than unioned, because the set only ever shrinks.
        """
        entry = self.pending_batches.get(str(batch_id))
        if not isinstance(entry, dict):
            return
        entry["retry_ids"] = list(dict.fromkeys(str(item) for item in retry_ids))

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

    def audio_file_for(
        self,
        record_id: str,
        *,
        of: str,
        content_fp: str,
        voice: int | str | None = None,
        speed: float | None = None,
        settings: Mapping[str, str] | None = None,
    ) -> str | None:
        """The clip recorded for this exact content, or ``None``.

        The *name*, not a yes/no, because "is this current?" needs three facts
        and the ledger only holds one of them: an entry proves something was
        recorded saying this, but not that the record still points at it or that
        the file survives on disk. A caller that asks only the ledger will call
        a record current whose reference was dropped — and then a prune,
        reading the records, deletes the clip nothing appears to want.

        ``voice`` and ``speed`` narrow it to a clip that also *sounds* the way
        this run would make it. A caller that passes them is asking "would
        regenerating change anything?", which is the question ``janki audio``
        actually has; content alone answers a different one and answers it yes
        for a clip in last week's voice. An entry missing either key — anything
        written before they were recorded — never matches, so it is
        regenerated: a clip janki cannot describe is not one it should keep,
        and re-synthesizing costs seconds.
        """
        for entry in self._audio_entries(record_id):
            if entry.get("of") != of or str(entry.get("content_fp") or "") != content_fp:
                continue
            if voice is not None and entry.get("voice") != _voice_key(voice):
                continue
            if speed is not None:
                recorded = entry.get("speed")
                if not isinstance(recorded, int | float) or isinstance(recorded, bool):
                    continue
                if float(recorded) != float(speed):
                    continue
            # `or {}` on both sides: an engine with nothing to say and an entry
            # written before there was anywhere to say it are the same fact,
            # and must not read as a difference.
            if settings is not None and dict(entry.get("settings") or {}) != dict(settings):
                continue
            return str(entry.get("file") or "") or None
        return None

    def forget_audio_files(self, files: Iterable[str]) -> int:
        """Drop every audio entry naming one of ``files``. Returns how many.

        So the ledger cannot outlive the media it describes: a pruned file whose
        entry survived would keep answering "already recorded" forever, and
        nothing would ever synthesize it again.
        """
        names = {str(item) for item in files}
        dropped = 0
        for entry in self.records.values():
            # Same guard every other reader in this module uses: a hand-edited
            # or older ledger can carry `"audio": null` or a bare string, and a
            # TypeError out of `janki audio --prune` is not the clean
            # `LedgerError` this module promises.
            audio = entry.get("audio")
            if not isinstance(audio, list):
                continue
            # Partitioned, not filtered: writing the recognised subset back
            # would erase entries this method cannot read, from a committed file
            # that `status --rebuild` cannot reconstruct, silently.
            unreadable = [item for item in audio if not isinstance(item, dict)]
            keep = [
                item
                for item in audio
                if isinstance(item, dict) and str(item.get("file") or "") not in names
            ]
            dropped += len(audio) - len(unreadable) - len(keep)
            entry["audio"] = unreadable + keep
        return dropped

    def drop_superseded_audio(
        self,
        record_id: str,
        *,
        of: str,
        keep: set[str],
        keep_files: set[str] | None = None,
    ) -> int:
        """Drop this record's ``of`` entries whose content is no longer current.

        Example clips are addressed by ``fp(record.id + sentence)``, so editing
        a sentence produces a *new* file and a *new* entry while the old one
        stays — reported as stale forever, since nothing removes entries, and
        worse on a revert: the old entry answers "already recorded" while the
        record still points at the other sentence's clip, so the card shows one
        sentence and plays another. Word audio is immune, being addressed by
        record id alone and so replaced in place.
        """
        entry = self.records.get(_record_key(record_id))
        if not entry or not entry.get("audio"):
            return 0
        audio = entry.get("audio")
        if not isinstance(audio, list):
            return 0
        unreadable = [item for item in audio if not isinstance(item, dict)]
        spared = keep_files or set()
        kept = [
            item
            for item in audio
            if isinstance(item, dict)
            if item.get("of") != of
            or str(item.get("content_fp") or "") in keep
            # An entry whose clip a record still names is the only evidence that
            # the card shows one sentence and plays another. Dropping it hides
            # the mismatch instead of fixing it.
            or str(item.get("file") or "") in spared
        ]
        dropped = len(audio) - len(unreadable) - len(kept)
        # Same partition as `forget_audio_files`, and for the same reason.
        entry["audio"] = unreadable + kept
        return dropped

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

    def unvoiced_examples(
        self, records: Iterable[VocabularyRecord]
    ) -> list[tuple[str, int]]:
        """``(record id, example index)`` for every example with no audio.

        Separate from :meth:`missing_audio` rather than folded into it, for the
        reason that method gives: a record with nine voiced examples and one
        silent one is not "missing audio" the way a record with no word clip
        is, and merging them buries the word-level signal.

        Reported as its own count instead. 188 of 343 sentences shipped silent
        because nothing asked this question — `status` counted words only and
        the build skipped an empty field without a word, so every indicator
        read green while half the cards said nothing.
        """
        return [
            (record.id, index)
            for record in records
            for index, example in enumerate(record.examples)
            if example.japanese and not example.audio
        ]

    def shipped_incomplete(self, deck_stem: str, records: Iterable[VocabularyRecord]) -> list[str]:
        """Records this deck shipped with a hole that has since been filled.

        Exact, where :meth:`exported_before_their_work` can only compare days:
        the build recorded what each record was missing, so "it went out silent
        and now it has a clip" is a fact about two states rather than an
        inference from two dates. This is the case that mattered — ship at
        09:00 with the gap warning, voice at 10:00, and no date comparison can
        ever tell.
        """
        stem = str(deck_stem).strip()
        if not stem:
            raise LedgerError("Exports are tracked per deck file stem; none was given")
        result: list[str] = []
        for record in records:
            entry = self.records.get(str(record.id))
            exports = entry.get("exports") if isinstance(entry, dict) else None
            shipped = exports.get(stem) if isinstance(exports, dict) else None
            if not isinstance(shipped, dict):
                continue
            missing = {str(gap) for gap in shipped.get("missing") or []}
            filled = {
                gap
                for gap in missing
                if (gap == "audio" and record.audio)
                or (
                    gap == "examples"
                    and any(example.japanese.strip() for example in record.examples)
                )
                or (
                    gap == "accent"
                    and (record.pitch_accent or record.audio_accent.strip())
                )
            }
            if filled:
                result.append(record.id)
        return result

    def exported_before_their_work(self, deck_stem: str, ids: Iterable[str]) -> list[str]:
        """Records this deck shipped *before* their newest audio or enrichment.

        ``unexported`` asks only whether a stem key exists, so once a record has
        shipped it is invisible to ``--only-new`` forever — including when a
        later run gives it the audio or examples it was shipped without. That is
        the quiet half of an incremental build: `janki refresh` voices a word on
        Tuesday and the deck that already carries it silently never learns.

        Dates are days, not timestamps, so "later" means a strictly later day —
        and this **permanently** cannot see work finished on the shipping day,
        not merely during that day's run. Both sides are day-granular, so
        timestamping exports alone would not fix it; both would have to move,
        which is a change across every ``at`` in the file.

        :meth:`shipped_incomplete` covers the case that made this matter — a
        record that went out missing audio and has it now — exactly and without
        dates. What is left here is the looser question of content edited after
        a build, where a same-day edit is a known blind spot.
        """
        stem = str(deck_stem).strip()
        if not stem:
            raise LedgerError("Exports are tracked per deck file stem; none was given")
        result: list[str] = []
        for record_id in ids:
            entry = self.records.get(str(record_id))
            if not isinstance(entry, dict):
                continue
            exports = entry.get("exports")
            value = exports.get(stem) if isinstance(exports, dict) else None
            # Two shapes: a bare date, or `{at, missing}` once a build had a
            # gap to record. Both carry the date; only one carries more.
            shipped = value.get("at") if isinstance(value, dict) else value
            if not isinstance(shipped, str) or not shipped:
                continue
            dates = [
                str(item.get("at") or "")
                for key in ("audio", "enriched")
                for item in (entry.get(key) or [])
                if isinstance(item, dict)
            ]
            if any(date > shipped for date in dates if date):
                result.append(str(record_id))
        return result

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
            or (
                record.source.type == "extract"
                and any(
                    # Only examples the pass may act on: an *unaccepted*
                    # incomplete example is one the prompt refuses to pin and
                    # the absorb refuses to overwrite, so selecting on it made
                    # every run pay a model call that could write nothing —
                    # forever. Its remedy is a user decision (acceptance or
                    # --force-fields), not another paid attempt.
                    example.needs_ai_annotations()
                    and example_accepted(record, example)
                    for example in record.examples
                )
            )
        ]


def load(path: Path, *, repair: bool = False) -> Ledger:
    """Read a ledger. A missing (or empty) file is an empty ledger, not an error.

    This is where a misshapen entry is refused, and the timing is the point: a
    ledger is loaded before an import writes anything, so a refusal here costs
    the user nothing. The same refusal raised from a mutator fires *after*
    ``vocabulary.json`` has been replaced and rows have been staged, so the
    command has done almost all of its work and reports none of it.

    ``repair=True`` coerces a wrong-typed structured key back to an empty one
    and names the record in :attr:`Ledger.repaired` instead of refusing. Only
    ``status --rebuild`` passes it, because only ``--rebuild`` is asking to fix
    the file: deleting the ledger is the other way out of a bad shape, and it
    destroys the export and enrichment history that no rebuild can reconstruct.
    """
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
    repaired: dict[str, list[str]] = {}
    for record_id, entry in records.items():
        if not isinstance(entry, dict):
            raise LedgerError(f"Ledger {path}: entry for '{record_id}' must be an object")
        problems = _shape_problems(entry)
        if not problems:
            continue
        if not repair:
            raise _shape_error(path, record_id, problems)
        parked: list[str] = []
        for key, _ in problems:
            where = quarantine_key(key, entry)
            entry[where] = entry[key]
            entry[key] = _ENTRY_SHAPE[key]()
            parked.append(where)
        repaired[record_id] = parked

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
        repaired=repaired,
    )
