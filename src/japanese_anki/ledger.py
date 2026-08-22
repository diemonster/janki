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
      "pending_batches": {},
      "pending_audio": {
        "<exact-request-sha256>": {
          "record_id": "word:話す:はなす", "of": "word",
          "target": "janki-<fp>.wav",
          "request": {"input": "ハナス'", "forced_accent": true},
          "profile": {"provider": "voicevox", "voice": 53, "speed": 1.0,
                      "settings": {}},
          "content_fp": "<raw-sha256>",
          "staged_file": ".pending/<exact-request-sha256>-<bytes-sha256>.stage",
          "staged_sha256": "<raw-sha256>", "details": {}, "at": "..."
        }
      }
    }

`enriched` is a **list** (it is a single object in the design document): the
jpdb pass and the AI pass each append, so neither overwrites the other's record
of what it wrote.

Three values in that shape are conditional, and all stay absent when there is
nothing to say so no existing ledger moves:

* **`exports[stem]`** is a plain date when the build shipped a complete record,
  and `{"at": ..., "missing": [...]}` when it did not — `missing` naming what
  the record lacked (`audio`, `examples`, `accent`). Dates are days, so
  comparing them cannot tell that a record shipped at 09:00 was voiced at
  10:00; what it went out without is a fact rather than an inference. Readers
  must handle both forms.
* **`audio[].settings`** carries anything beyond voice and rate that decides how
  a clip sounds — for OpenAI, the model and prose `instructions`, the pace
  control janki exposes. Absent for engines with nothing to add.
* **`pending_audio`** is the per-clip write-ahead record for paid bytes whose
  guarded record/media/ledger transaction has not finished. Its self-verifying
  stage name binds the exact request key and byte SHA, so an interruption
  between the stage write and WAL merge is still recoverable without rebilling.

Ordinary mutators only touch memory and report whether they changed anything;
call :meth:`Ledger.save` once when an ordinary command is done. Every mutator
is idempotent, and idempotent by *identity* rather than by date: re-running the
same enrichment pass or re-recording byte-identical audio next month is still
a no-op, and the stored date stays the first run's. Re-running an import or a
build must not grow the file, ever, not just today.

:meth:`Ledger.save` is a whole-file rewrite, so ordinary commands **load once
and save once**. Two live :class:`Ledger` objects on one path would otherwise
lose whichever saved first — silently, since the atomic writer guarantees the
loser still finds a well-formed file. ``save`` refuses to overwrite a file that
changed since it was read rather than clobber it. Paid audio is the deliberate
exception: :meth:`Ledger.merge_pending_audio` locks and additively persists
each completed clip before another provider call, then the command finalizes
the canonical audio entries in a later guarded save.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Container, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any

from japanese_anki.errors import JankiError
from japanese_anki.identifiers import short_fingerprint
from japanese_anki.io import DataError, atomic_write_text, exclusive_path_lock
from japanese_anki.models import ExampleSentence, VocabularyRecord, example_accepted
from japanese_anki.tts import RenderProfile, TtsError, clip_provider


class LedgerError(JankiError):
    pass


LEDGER_VERSION = 1

# Who did the enriching. The kind is what keeps a jpdb pass and an AI pass from
# being mistaken for each other; the model is which one of them ran. Rich AI
# enrichment owns meanings, examples, and usage notes in one answer.
ENRICHMENT_KINDS: tuple[str, ...] = ("jpdb", "ai", "human")
AI_PROVIDERS: tuple[str, ...] = ("anthropic", "codex")

# ``human`` is the fourth because a dictionary can be wrong and a person has to
# be able to say so. jpdb's parse reads 日本語 as にっぽんご; the language is
# にほんご, and without a way to record "a human overruled this" the only
# options were to accept a reading nobody uses or to leave a correct sentence
# unvoiced forever. The pitch side has had this since M5.1, as ``audio_accent``.

# What a piece of audio says: the word itself, or one example sentence.
AUDIO_KINDS: tuple[str, ...] = ("word", "example")


def exact_ai_batch_model(value: Any) -> str:
    """Return the exact submitted model, refusing absent attribution."""
    if not isinstance(value, str) or not value.strip():
        raise LedgerError("An AI message batch needs a non-empty model")
    return value


def exact_fingerprint_map(
    values: Mapping[str, str] | None,
    record_ids: Iterable[str],
    *,
    label: str,
) -> dict[str, str]:
    """Validate one exact per-record SHA-256 map used by AI provenance."""
    ids = list(dict.fromkeys(str(item) for item in record_ids))
    if values is None:
        fingerprints: dict[str, str] = {}
    elif not isinstance(values, Mapping):
        raise LedgerError(f"Batch {label} fingerprints must be a mapping")
    else:
        if any(not isinstance(record_id, str) or not record_id for record_id in values):
            raise LedgerError(
                f"Batch {label} fingerprint record ids must be non-empty text"
            )
        fingerprints = dict(values)
    missing = [record_id for record_id in ids if record_id not in fingerprints]
    extra = [record_id for record_id in fingerprints if record_id not in ids]
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if extra:
            details.append(f"unknown {', '.join(extra)}")
        raise LedgerError(
            f"Batch {label} fingerprints must exactly cover its pending ids: "
            + "; ".join(details)
        )
    malformed = [
        record_id
        for record_id, fingerprint in fingerprints.items()
        if not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ]
    if malformed:
        raise LedgerError(
            f"Batch {label} fingerprints must be lowercase SHA-256 values: "
            + ", ".join(malformed)
        )
    return fingerprints


def exact_retry_ids(values: Any, pending_ids: Iterable[str]) -> list[str]:
    """Validate a held batch's exact, non-empty retry subset.

    Presence means a previous fetch deliberately narrowed the batch.  An empty
    or malformed value must therefore never fall back to the full pending set:
    doing so can reapply rows already settled, while an unknown id can skip the
    one recoverable row and let the caller clear the batch.
    """
    if not isinstance(values, list):
        raise LedgerError("Batch retry_ids must be a list")
    if not values:
        raise LedgerError("Batch retry_ids must name at least one pending record")
    if any(not isinstance(record_id, str) or not record_id.strip() for record_id in values):
        raise LedgerError("Batch retry_ids must contain non-empty text record ids")
    if len(set(values)) != len(values):
        raise LedgerError("Batch retry_ids must not contain duplicates")
    pending = set(str(record_id) for record_id in pending_ids)
    unknown = [record_id for record_id in values if record_id not in pending]
    if unknown:
        raise LedgerError(
            "Batch retry_ids must be a subset of pending_ids; unknown "
            + ", ".join(unknown)
        )
    return list(values)


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


def _audio_profile_matches(
    entry: Mapping[str, Any],
    *,
    provider: str | None = None,
    voice: int | str | None = None,
    speed: float | None = None,
    settings: Mapping[str, str] | None = None,
) -> bool:
    """Whether an audio entry was rendered with the requested profile.

    Content fingerprints deliberately say what was spoken, not which engine or
    voice spoke it. This is the render-profile half of currency, shared by
    ``janki audio`` and ``janki status`` so they cannot disagree about whether
    a configured synthesis choice changed.
    """
    if provider is not None and entry.get("provider") != str(provider):
        return False
    if voice is not None and entry.get("voice") != _voice_key(voice):
        return False
    if speed is not None:
        recorded = entry.get("speed")
        if not isinstance(recorded, int | float) or isinstance(recorded, bool):
            return False
        if float(recorded) != float(speed):
            return False
    if settings is not None:
        recorded_settings = entry.get("settings")
        if recorded_settings is None:
            recorded_settings = {}
        elif not isinstance(recorded_settings, Mapping):
            return False
        if dict(recorded_settings) != dict(settings):
            return False
    return True


def _audio_entry_is_current(
    entry: Mapping[str, Any],
    *,
    content_fp: str,
    profile: RenderProfile | None,
) -> bool:
    """Whether one referenced entry matches both content and render profile."""
    if str(entry.get("content_fp") or "") != content_fp:
        return False
    if profile is None:
        return True
    return _audio_profile_matches(
        entry,
        provider=profile.name,
        voice=profile.voice,
        speed=profile.speed,
        settings=profile.settings,
    )


def _record_key(record_id: str) -> str:
    key = str(record_id).strip()
    if not key:
        raise LedgerError("A ledger entry needs a record id")
    return key


def _validate_pending_audio_entry(
    key: str, entry: Mapping[str, Any], *, where: str = "Pending audio entry"
) -> None:
    """Validate a complete, exactly keyed WAL row before any consumer uses it."""
    if (
        len(key) != 64
        or any(character not in "0123456789abcdef" for character in key)
    ):
        raise LedgerError(f"{where} key {key!r} must be a lowercase SHA-256")
    target = str(entry.get("target") or "")
    if not target or PurePosixPath(target).name != target:
        raise LedgerError(f"{where} {key!r} needs one canonical target filename")
    staged_name = str(entry.get("staged_file") or "")
    staged_path = PurePosixPath(staged_name)
    if staged_path.parent.as_posix() != ".pending" or staged_path.suffix != ".stage":
        raise LedgerError(
            f"{where} {key!r} staged_file must be a direct .pending stage; "
            "canonical media can never double as removable WAL staging"
        )
    request = entry.get("request")
    profile = entry.get("profile")
    details = entry.get("details")
    if (
        not isinstance(request, Mapping)
        or set(request) != {"input", "forced_accent"}
        or not isinstance(request.get("input"), str)
        or not isinstance(request.get("forced_accent"), bool)
    ):
        raise LedgerError(f"{where} {key!r} has a malformed exact request")
    if not isinstance(profile, Mapping) or set(profile) != {
        "provider",
        "voice",
        "speed",
        "settings",
    }:
        raise LedgerError(f"{where} {key!r} has a malformed render profile")
    settings = profile.get("settings")
    if (
        not isinstance(profile.get("provider"), str)
        or not str(profile.get("provider") or "").strip()
        or isinstance(profile.get("speed"), bool)
        or not isinstance(profile.get("speed"), int | float)
        or not isinstance(settings, Mapping)
        or any(
            not isinstance(name, str) or not isinstance(value, str)
            for name, value in settings.items()
        )
    ):
        raise LedgerError(f"{where} {key!r} has a malformed render profile")
    # Raises the field-specific clean error for blank/invalid voice values.
    _voice_key(profile.get("voice"))
    content_fp = entry.get("content_fp")
    staged_sha = entry.get("staged_sha256")
    for label, value in (("content_fp", content_fp), ("staged_sha256", staged_sha)):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise LedgerError(f"{where} {key!r} has malformed {label}")
    expected = f".pending/{key}-{staged_sha}.stage"
    if staged_name != expected:
        raise LedgerError(
            f"{where} {key!r} staged_file must be exactly {expected!r}; "
            "canonical media can never double as a removable WAL stage"
        )
    if not isinstance(details, Mapping) or any(
        not isinstance(name, str) for name in details
    ):
        raise LedgerError(f"{where} {key!r} has malformed details")
    reserved_details = {
        "record_id",
        "file",
        "of",
        "provider",
        "voice",
        "speed",
        "settings",
        "content_fp",
        "at",
    }
    if reserved_details.intersection(details):
        raise LedgerError(f"{where} {key!r} uses a reserved detail key")
    _checked_details("pending_audio", dict(details))
    recorded_at = entry.get("at")
    if not isinstance(recorded_at, str):
        raise LedgerError(f"{where} {key!r} has a malformed at date")
    _iso_date(recorded_at)
    identity = Ledger._pending_audio_identity(
        str(entry.get("record_id") or ""),
        of=str(entry.get("of") or ""),
        target=target,
        request_input=request["input"],
        forced_accent=request["forced_accent"],
        content_fp=content_fp,
        provider=profile["provider"],
        voice=profile["voice"],
        speed=profile["speed"],
        settings=settings,
    )
    if Ledger._pending_audio_key(identity) != key:
        raise LedgerError(
            f"{where} {key!r} does not match its exact request/profile identity"
        )


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

    Unlike the frozen 48-bit filename address, this is a raw, framed SHA-256.
    NFKC-equivalent provider inputs and forced/natural requests must not compare
    equal: they can produce different bytes at the same stable filename.
    """
    utterance, forced, _ = word_audio_request(record)
    return _audio_content_fingerprint(
        "word",
        "forced" if forced else "natural",
        utterance,
    )


def word_audio_request(record: VocabularyRecord) -> tuple[str, bool, str | None]:
    """The exact provider input/flag for a word, plus a render warning."""
    pattern = _selected_pitch_pattern(record)
    if not pattern:
        return record.reading, False, None
    from japanese_anki import pitch

    try:
        return pitch.to_aquestalk(record.reading, pattern), True, None
    except pitch.PitchError as exc:
        # Unusable: audio generation falls back to the raw reading and reports
        # the error. Currency still follows the request actually sent.
        return record.reading, False, str(exc)


def example_audio_content_fingerprint(example: ExampleSentence) -> str:
    """What example audio says, without filename-style normalization."""
    return _audio_content_fingerprint("example", example.japanese)


def _audio_content_fingerprint(*parts: str) -> str:
    """A framed, raw SHA-256 over the exact synthesis request channels."""
    digest = hashlib.sha256()
    for part in parts:
        encoded = str(part).encode("utf-8", errors="surrogatepass")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


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
    # Paid audio bytes that are intentionally not canonical yet. Unlike an
    # ``audio`` entry this is a write-ahead record: it is additive, names a
    # staged file, and survives a failed records-file compare-and-swap so the
    # exact request can be adopted on the next run without another paid call.
    pending_audio: dict[str, Any] = field(default_factory=dict)
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

    def _serialized(self) -> str:
        payload = {
            **self.extra,
            "version": LEDGER_VERSION,
            "records": self.records,
            "pending_batches": self.pending_batches,
            # Sparse so every healthy pre-M8.1 ledger remains byte-shaped the
            # same after an unrelated command. The key exists only while there
            # is actually a recovery transaction to describe.
            **({"pending_audio": self.pending_audio} if self.pending_audio else {}),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    def _save_locked(self) -> None:
        """Save while the caller owns this ledger path's exclusive lock."""
        text = self._serialized()
        if self.guarded and self._text_on_disk() != self.baseline:
            raise LedgerError(
                f"Ledger {self.path} changed on disk since it was read; "
                "saving now would discard those changes. Re-load the current "
                "ledger and re-run."
            )
        try:
            atomic_write_text(self.path, text)
        except DataError as exc:
            raise LedgerError(str(exc)) from exc
        self.baseline = text

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
        try:
            with exclusive_path_lock(self.path):
                self._save_locked()
        except DataError as exc:
            raise LedgerError(str(exc)) from exc

    def merge_pending_audio(
        self, keys: Iterable[str], *, replace: bool = False
    ) -> None:
        """Durably merge additive WAL rows under one ledger lock.

        Retrying a stale whole-ledger snapshot can starve under repeated
        unrelated writes and, worse, leave a just-paid stage unregistered when
        an arbitrary retry bound is exhausted. This operation takes the lock
        once, reads the latest state inside it, adds only the named exact rows,
        and writes that merged state without clobbering concurrent provenance.
        """
        rows: dict[str, dict[str, Any]] = {}
        for raw_key in dict.fromkeys(str(item) for item in keys):
            entry = self.pending_audio_entry(raw_key)
            if entry is None:
                raise LedgerError(f"Pending audio entry {raw_key!r} disappeared")
            rows[raw_key] = entry
        try:
            with exclusive_path_lock(self.path):
                latest = load(self.path)
                for key, entry in rows.items():
                    existing = latest.pending_audio.get(key)
                    if existing is not None and existing != entry and not replace:
                        raise LedgerError(
                            f"Pending audio entry {key!r} changed concurrently; "
                            "refusing to overwrite either paid render. Re-run "
                            "after the other audio command finishes."
                        )
                    latest.pending_audio[key] = dict(entry)
                latest._save_locked()
        except DataError as exc:
            raise LedgerError(str(exc)) from exc
        self.records = latest.records
        self.pending_batches = latest.pending_batches
        self.pending_audio = latest.pending_audio
        self.extra = latest.extra
        self.baseline = latest.baseline
        self.guarded = latest.guarded
        self.repaired = latest.repaired

    def assert_current(self) -> None:
        """Refuse a destructive side effect based on a stale ledger snapshot."""
        if not self.guarded:
            return
        try:
            with exclusive_path_lock(self.path):
                if self._text_on_disk() != self.baseline:
                    raise LedgerError(
                        f"Ledger {self.path} changed on disk since it was read; "
                        "a destructive operation based on that stale snapshot "
                        "was refused before touching media. Re-run it."
                    )
        except DataError as exc:
            raise LedgerError(str(exc)) from exc

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
        provider: str | None = None,
        request_fingerprint: str | None = None,
        at: str | None = None,
    ) -> bool:
        """Note that an enrichment pass wrote ``fields``.

        Entries accumulate rather than replace: a jpdb pass and an AI pass
        describe different work, and neither may erase the other's record of
        what it wrote. A dictionary pass is identified by ``(kind, model,
        fields)``; an AI pass also includes the complete request fingerprint,
        so two templates that wrote the same fields remain distinct provenance.
        Re-running an identical pass changes nothing however long ago it last
        ran, and the stored ``at`` stays the first run's, the same way
        ``record_source_seen`` ignores ``seen_at``.
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
        ai_provider = str(provider or "").strip()
        if kind == "ai" and ai_provider not in AI_PROVIDERS:
            raise LedgerError(
                "record_enriched(kind='ai') needs provider "
                f"{' or '.join(AI_PROVIDERS)}"
            )
        if kind != "ai" and provider is not None:
            raise LedgerError("Only AI enrichment records a model provider")
        prompt_fp = str(request_fingerprint or "").strip()
        if kind == "ai" and not prompt_fp:
            raise LedgerError(
                "record_enriched(kind='ai') needs the full request fingerprint "
                "that produced the answer"
            )
        if kind == "ai" and (
            len(prompt_fp) != 64
            or any(character not in "0123456789abcdef" for character in prompt_fp)
        ):
            raise LedgerError(
                "record_enriched(kind='ai') request fingerprint must be a "
                "lowercase SHA-256"
            )
        reference = {
            "at": _iso_date(at),
            "kind": kind,
            "model": str(model),
            "fields": names,
            **({"provider": ai_provider} if kind == "ai" else {}),
            **({"request_fingerprint": prompt_fp} if prompt_fp else {}),
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
        identity-addressed file (a new voice, say) replaces its entry rather
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
        provider: str | None = None,
        pending_ids: Iterable[str],
        force_fields: Iterable[str] = (),
        request_fingerprints: Mapping[str, str] | None = None,
        input_fingerprints: Mapping[str, str] | None = None,
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
        batch_provider = str(provider or "").strip()
        if kind == "ai" and batch_provider != "anthropic":
            raise LedgerError(
                "An AI message batch needs provider 'anthropic'"
            )
        batch_model = exact_ai_batch_model(model) if kind == "ai" else str(model)
        entry: dict[str, Any] = {
            "kind": kind,
            "model": batch_model,
            **({"provider": batch_provider} if kind == "ai" else {}),
            "submitted_at": _iso_date(at),
            "pending_ids": ids,
            # Stored because the fetch has to apply what the submit asked for.
            # Without it, submitting with --force-fields and fetching plainly
            # would report "nothing to fill" and throw away a paid answer.
            "force_fields": list(dict.fromkeys(str(item) for item in force_fields)),
        }
        if kind == "ai":
            entry["request_fingerprints"] = exact_fingerprint_map(
                request_fingerprints, ids, label="request"
            )
            entry["input_fingerprints"] = exact_fingerprint_map(
                input_fingerprints, ids, label="input"
            )
        elif request_fingerprints is not None or input_fingerprints is not None:
            raise LedgerError("Only an AI batch records model-request fingerprints")
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
        if isinstance(retry_ids, (str, bytes)):
            candidates: Any = retry_ids
        else:
            candidates = list(retry_ids)
        entry["retry_ids"] = exact_retry_ids(
            candidates, entry.get("pending_ids", [])
        )

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

    def ever_exported(self, ids: Iterable[str]) -> frozenset[str]:
        """Which of ``ids`` have shipped in *any* deck build.

        `unexported` asks about one deck because that is what `build
        --only-new` needs. Re-identification asks a different question: an
        Anki note is matched by a GUID derived from the record ID, so changing
        that ID strands whatever has already shipped under the old one — in
        any deck at all.
        """
        return frozenset(
            key
            for key in {str(record_id) for record_id in ids}
            if (self.records.get(key) or {}).get("exports")
        )

    def audio_file_for(
        self,
        record_id: str,
        *,
        of: str,
        content_fp: str,
        provider: str | None = None,
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

        ``provider``, ``voice``, ``speed`` and ``settings`` narrow it to a clip
        that also *sounds* the way this run would make it. A caller that passes
        them is asking "would regenerating change anything?", which is the
        question ``janki audio`` actually has; content alone answers a different
        one and answers it yes for a clip in last week's voice. An entry missing
        requested metadata never matches, so it is regenerated: a clip janki
        cannot describe is not one it should keep, and re-synthesizing costs
        seconds.
        """
        for entry in self._audio_entries(record_id):
            if entry.get("of") != of or str(entry.get("content_fp") or "") != content_fp:
                continue
            if not _audio_profile_matches(
                entry,
                provider=provider,
                voice=voice,
                speed=speed,
                settings=settings,
            ):
                continue
            return str(entry.get("file") or "") or None
        return None

    @staticmethod
    def _pending_audio_identity(
        record_id: str,
        *,
        of: str,
        target: str,
        request_input: str,
        forced_accent: bool,
        content_fp: str,
        provider: str,
        voice: int | str,
        speed: float,
        settings: Mapping[str, str] | None,
    ) -> dict[str, Any]:
        """Build the exact, date-free identity of one pending render."""
        if of not in AUDIO_KINDS:
            raise LedgerError(
                f"Unknown audio kind '{of}'; expected one of {', '.join(AUDIO_KINDS)}"
            )
        name = str(target).strip()
        if not name or PurePosixPath(name).name != name:
            raise LedgerError("A pending audio target must be one media filename")
        if not isinstance(forced_accent, bool):
            raise LedgerError("A pending audio request needs a boolean forced_accent")
        fingerprint = str(content_fp)
        if (
            len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
        ):
            raise LedgerError(
                "A pending audio content fingerprint must be a lowercase SHA-256"
            )
        try:
            rate = float(speed)
        except (TypeError, ValueError) as exc:
            raise LedgerError(f"Audio speed must be a number, got {speed!r}") from exc
        return {
            "record_id": _record_key(record_id),
            "of": of,
            "target": name,
            "request": {
                "input": str(request_input),
                "forced_accent": forced_accent,
            },
            "profile": {
                "provider": str(provider),
                "voice": _voice_key(voice),
                "speed": rate,
                "settings": dict(settings or {}),
            },
            "content_fp": fingerprint,
        }

    @staticmethod
    def _pending_audio_key(identity: Mapping[str, Any]) -> str:
        encoded = json.dumps(
            dict(identity),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def record_pending_audio(
        self,
        record_id: str,
        *,
        of: str,
        target: str,
        request_input: str,
        forced_accent: bool,
        content_fp: str,
        provider: str,
        voice: int | str,
        speed: float,
        settings: Mapping[str, str] | None,
        staged_file: str,
        staged_sha256: str,
        at: str | None = None,
        **details: Any,
    ) -> str:
        """Add one staged paid render to the write-ahead log.

        No canonical ``records[*].audio`` entry is changed here. The caller
        persists this additive block, compare-and-swap saves the record
        reference, and only then publishes the bytes and commits the canonical
        entry. Repeating the exact render replaces its own WAL value; a
        different request/profile gets a different key and cannot be adopted.
        """
        identity = self._pending_audio_identity(
            record_id,
            of=of,
            target=target,
            request_input=request_input,
            forced_accent=forced_accent,
            content_fp=content_fp,
            provider=provider,
            voice=voice,
            speed=speed,
            settings=settings,
        )
        staged = str(staged_file).strip()
        staged_path = PurePosixPath(staged)
        if (
            not staged
            or staged_path.is_absolute()
            or ".." in staged_path.parts
            or staged_path.name in {"", ".", ".."}
        ):
            raise LedgerError("A pending audio staged_file must be a safe relative path")
        staged_sha = str(staged_sha256)
        if (
            len(staged_sha) != 64
            or any(character not in "0123456789abcdef" for character in staged_sha)
        ):
            raise LedgerError("A pending audio staged_sha256 must be a lowercase SHA-256")
        _checked_details("record_pending_audio", details)
        key = self._pending_audio_key(identity)
        expected_stage = f".pending/{key}-{staged_sha}.stage"
        if staged_path.as_posix() != expected_stage:
            raise LedgerError(
                f"A pending audio staged_file must be exactly {expected_stage!r}"
            )
        existing = self.pending_audio.get(key)
        if isinstance(existing, Mapping) and any(
            existing.get(name) != value for name, value in identity.items()
        ):
            raise LedgerError(
                f"Pending audio identity collision at {key}; refusing to replace it"
            )
        candidate = {
            **identity,
            "staged_file": staged_path.as_posix(),
            "staged_sha256": staged_sha,
            "details": dict(details),
            "at": _iso_date(at),
        }
        _validate_pending_audio_entry(key, candidate)
        self.pending_audio[key] = candidate
        return key

    def pending_audio_key_for(
        self,
        record_id: str,
        *,
        of: str,
        target: str,
        request_input: str,
        forced_accent: bool,
        content_fp: str,
        provider: str,
        voice: int | str,
        speed: float,
        settings: Mapping[str, str] | None,
    ) -> str:
        """Stable key/path component for an exact pending render identity."""
        return self._pending_audio_key(
            self._pending_audio_identity(
                record_id,
                of=of,
                target=target,
                request_input=request_input,
                forced_accent=forced_accent,
                content_fp=content_fp,
                provider=provider,
                voice=voice,
                speed=speed,
                settings=settings,
            )
        )

    def pending_audio_for(
        self,
        record_id: str,
        *,
        of: str,
        target: str,
        request_input: str,
        forced_accent: bool,
        content_fp: str,
        provider: str,
        voice: int | str,
        speed: float,
        settings: Mapping[str, str] | None,
    ) -> tuple[str, dict[str, Any]] | None:
        """Return only a WAL entry for this exact request and render profile."""
        identity = self._pending_audio_identity(
            record_id,
            of=of,
            target=target,
            request_input=request_input,
            forced_accent=forced_accent,
            content_fp=content_fp,
            provider=provider,
            voice=voice,
            speed=speed,
            settings=settings,
        )
        key = self._pending_audio_key(identity)
        entry = self.pending_audio.get(key)
        if not isinstance(entry, dict):
            return None
        _validate_pending_audio_entry(key, entry)
        if any(entry.get(name) != value for name, value in identity.items()):
            return None
        return key, dict(entry)

    def pending_audio_entry(self, key: str) -> dict[str, Any] | None:
        entry = self.pending_audio.get(str(key))
        if not isinstance(entry, dict):
            return None
        _validate_pending_audio_entry(str(key), entry)
        return dict(entry)

    def commit_pending_audio(self, key: str) -> dict[str, Any]:
        """Apply one published WAL entry to canonical audio state in memory."""
        entry = self.pending_audio_entry(key)
        if entry is None:
            raise LedgerError(f"No pending audio entry {key!r}")
        profile = entry.get("profile")
        details = entry.get("details")
        if not isinstance(profile, Mapping) or not isinstance(details, Mapping):
            raise LedgerError(f"Pending audio entry {key!r} is malformed")
        reserved = {
            "record_id",
            "file",
            "of",
            "provider",
            "voice",
            "speed",
            "settings",
            "content_fp",
            "at",
        }
        collisions = sorted(str(name) for name in details if name in reserved)
        if collisions:
            raise LedgerError(
                f"Pending audio entry {key!r} has reserved detail key(s): "
                + ", ".join(collisions)
            )
        self.record_audio(
            str(entry.get("record_id") or ""),
            file=str(entry.get("target") or ""),
            of=str(entry.get("of") or ""),
            provider=str(profile.get("provider") or ""),
            voice=profile.get("voice"),
            speed=profile.get("speed"),
            settings=(
                profile.get("settings")
                if isinstance(profile.get("settings"), Mapping)
                else None
            ),
            content_fp=str(entry.get("content_fp") or ""),
            at=str(entry.get("at") or "") or None,
            **dict(details),
        )
        del self.pending_audio[str(key)]
        return entry

    def discard_pending_audio_for_targets(
        self, targets: Iterable[str]
    ) -> list[dict[str, Any]]:
        """Drop superseded WAL entries after those targets have committed."""
        names = {str(target).casefold() for target in targets}
        removed: list[dict[str, Any]] = []
        for key, raw in list(self.pending_audio.items()):
            if not isinstance(raw, dict):
                continue
            if str(raw.get("target") or "").casefold() not in names:
                continue
            removed.append(dict(raw))
            del self.pending_audio[key]
        return removed

    def discard_pending_audio_for_slots(
        self,
        slots: Iterable[tuple[str, str]],
        *,
        keep: Iterable[str] = (),
    ) -> list[dict[str, Any]]:
        """Retire WAL rows for selected record/kind slots no longer current."""
        wanted = {(str(record_id), str(kind)) for record_id, kind in slots}
        spared = {str(key) for key in keep}
        removed: list[dict[str, Any]] = []
        for key, raw in list(self.pending_audio.items()):
            if key in spared or not isinstance(raw, dict):
                continue
            _validate_pending_audio_entry(str(key), raw)
            slot = (str(raw.get("record_id") or ""), str(raw.get("of") or ""))
            if slot not in wanted:
                continue
            removed.append(dict(raw))
            del self.pending_audio[key]
        return removed

    def discard_pending_audio_for_missing_records(
        self, record_ids: Iterable[str]
    ) -> list[dict[str, Any]]:
        """Retire WAL rows whose owning normalized record was deleted."""
        present = {str(record_id) for record_id in record_ids}
        removed: list[dict[str, Any]] = []
        for key, raw in list(self.pending_audio.items()):
            if not isinstance(raw, dict):
                continue
            _validate_pending_audio_entry(str(key), raw)
            if str(raw.get("record_id") or "") in present:
                continue
            removed.append(dict(raw))
            del self.pending_audio[key]
        return removed

    def pending_audio_files(self) -> set[str]:
        """Canonical targets protected while a paid render is pending."""
        files: set[str] = set()
        for key, entry in self.pending_audio.items():
            if not isinstance(entry, Mapping):
                raise LedgerError(f"Pending audio entry {key!r} must be an object")
            _validate_pending_audio_entry(str(key), entry)
            files.add(str(entry["target"]))
        return files

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

        Asks the **ledger**, not just the field, which is what makes it the
        same kind of answer as :meth:`missing_audio` one method up. A record
        naming a clip the ledger never wrote is a record whose sentence is
        silent for a different reason, and a version of this that read
        `example.audio`'s truthiness alone reported it as voiced — the same
        green-while-mute failure one level along.
        """
        voiced = {
            (record_id, entry.get("file"))
            for record_id in {record.id for record in records}
            for entry in self._audio_entries(record_id)
            if entry.get("of") == "example"
        }
        return [
            (record.id, index)
            for record in records
            for index, example in enumerate(record.examples)
            # Basename: the record stores `audio/janki-<fp>.mp3` and the
            # ledger stores the file, which is how every other lookup here
            # compares them.
            if example.japanese
            and (
                not example.audio
                or (record.id, PurePosixPath(example.audio).name) not in voiced
            )
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

    def stale_audio(
        self,
        records: Iterable[VocabularyRecord],
        *,
        word_provider: RenderProfile | None,
        example_provider: RenderProfile | None,
    ) -> list[str]:
        """Records whose audio no longer matches content or render profile.

        Currency belongs to the file the record currently names. This matters
        when switching sentence engines changes the extension: the ledger can
        retain the unreferenced old file beside its current replacement, and a
        historical entry must not leave the record stale forever.

        Word audio goes stale when its reference is dropped, its reading or
        selected accent changes, or its configured render profile changes.
        Referenced example audio goes stale when its sentence or an available
        render profile changes. When a configured provider cannot be resolved,
        status warns and this method still compares content and references. A
        missing example reference is reported by
        :meth:`unvoiced_examples`; entries of an unrecognized kind are left
        alone.
        """
        result: list[str] = []
        for record in records:
            entries = self._audio_entries(record.id)
            if not entries:
                continue
            word_fp = word_audio_content_fingerprint(record)
            word_entries = [entry for entry in entries if entry.get("of") == "word"]
            if word_entries:
                word_file = PurePosixPath(record.audio).name if record.audio else ""
                referenced = [entry for entry in word_entries if entry.get("file") == word_file]
                if not referenced or not any(
                    _audio_entry_is_current(
                        entry,
                        content_fp=word_fp,
                        profile=word_provider,
                    )
                    for entry in referenced
                ):
                    result.append(record.id)
                    continue

            example_entries = [entry for entry in entries if entry.get("of") == "example"]
            for example in record.examples:
                if not example.japanese or not example.audio:
                    continue
                example_file = PurePosixPath(example.audio).name
                referenced = [
                    entry for entry in example_entries if entry.get("file") == example_file
                ]
                profile = example_provider
                if profile is not None:
                    try:
                        profile = clip_provider(profile, example.instructions)
                    except TtsError:
                        # A configured engine that cannot honor an explicit
                        # clip setting cannot describe this file as current.
                        result.append(record.id)
                        break
                # No ledger entry is the unvoiced-example signal, not a second
                # stale signal for the same hole.
                if referenced and not any(
                    _audio_entry_is_current(
                        entry,
                        content_fp=example_audio_content_fingerprint(example),
                        profile=profile,
                    )
                    for entry in referenced
                ):
                    result.append(record.id)
                    break
        # Multiple deck files may persist different versions of one record id.
        # Any stale version matters, but the status surface counts record ids.
        return list(dict.fromkeys(result))

    @staticmethod
    def missing_enrichment(records: Iterable[VocabularyRecord]) -> list[str]:
        """Records that still need enrichment, defined purely by their content.

        Deliberately a static method: it never consults the ledger's own
        ``enriched`` entries. Those say who ran and what they wrote; they do
        not say the record is finished. A jpdb pass fills readings and accents
        and would otherwise hide a record from ``enrich --ai``, which is what
        actually writes meanings and examples. Usage notes are optional: every
        rich prompt explicitly permits an empty note when there is no useful,
        certain nuance to add, so emptiness cannot mean a paid answer is
        unfinished.
        """
        return [
            record.id
            for record in records
            if not any(str(meaning).strip() for meaning in record.meanings)
            or not any(example.japanese.strip() for example in record.examples)
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

    pending_audio = data.get("pending_audio")
    if pending_audio is None:
        pending_audio = {}
    if not isinstance(pending_audio, dict):
        raise LedgerError(f"Ledger {path}: 'pending_audio' must be an object")
    for pending_key, pending_entry in pending_audio.items():
        if not isinstance(pending_key, str) or not isinstance(pending_entry, Mapping):
            raise LedgerError(
                f"Ledger {path}: each pending_audio entry must be keyed by text "
                "and hold an object"
            )
        _validate_pending_audio_entry(
            pending_key,
            pending_entry,
            where=f"Ledger {path}: pending audio entry",
        )

    return Ledger(
        path=path,
        records=dict(records),
        pending_batches=dict(pending_batches),
        pending_audio=dict(pending_audio),
        extra={
            key: value
            for key, value in data.items()
            if key not in {"version", "records", "pending_batches", "pending_audio"}
        },
        baseline=text,
        guarded=True,
        repaired=repaired,
    )
