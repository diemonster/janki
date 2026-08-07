from __future__ import annotations

import contextlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

import yaml

from japanese_anki.errors import JankiError
from japanese_anki.models import VocabularyRecord


class DataError(JankiError):
    pass


def _target_mode(path: Path) -> int:
    try:
        return stat.S_IMODE(os.stat(path).st_mode)
    except FileNotFoundError:
        current_umask = os.umask(0)
        os.umask(current_umask)
        return 0o666 & ~current_umask


def _fsync_directory(directory: Path) -> None:
    # Best effort: not every filesystem lets you open or fsync a directory.
    try:
        dir_fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` so a reader never observes a partial file.

    The text goes to a uniquely named temporary file in the target's
    directory (concurrent writers cannot collide), is fsynced, then renamed
    over the target. A failure mid-write leaves the previous contents
    untouched. Symlinked targets are written through to their real file, and
    an existing target keeps its permission bits. Filesystem failures are
    reported as ``DataError``.
    """
    path = Path(os.path.realpath(path))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            dir=path.parent, prefix=f"{path.name}.", suffix=".tmp"
        )
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temp_path, _target_mode(path))
            os.replace(temp_path, path)
        except BaseException:
            # Cleanup must never replace the real error with its own.
            with contextlib.suppress(OSError):
                temp_path.unlink(missing_ok=True)
            raise
        _fsync_directory(path.parent)
    except OSError as exc:
        # exc may name the random temp file; report the target the caller asked for.
        raise DataError(f"Could not write {path}: {exc.strerror or exc}") from exc


def load_structured(path: Path) -> Any:
    suffix = path.suffix.lower()
    try:
        with path.open("r", encoding="utf-8") as handle:
            if suffix == ".json":
                return json.load(handle)
            if suffix in {".yaml", ".yml"}:
                return yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise DataError(f"File not found: {path}") from exc
    except OSError as exc:
        raise DataError(f"Could not read {path}: {exc.strerror or exc}") from exc
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise DataError(f"Could not parse {path}: {exc}") from exc
    raise DataError(f"Unsupported file type for {path}; expected JSON or YAML")


def load_records(path: Path) -> list[VocabularyRecord]:
    data = load_structured(path)
    if isinstance(data, dict) and "records" in data:
        data = data["records"]
    if not isinstance(data, list):
        raise DataError(f"Expected a list of vocabulary records in {path}")
    for item in data:
        if not isinstance(item, dict):
            raise DataError(
                f"Each record in {path} must be a mapping, got {type(item).__name__}"
            )
    return [VocabularyRecord.from_dict(item) for item in data]


def save_records_json(path: Path, records: list[VocabularyRecord]) -> None:
    payload = [record.to_dict() for record in sorted(records, key=lambda item: item.id)]
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    atomic_write_text(path, text)


def merge_records(
    existing: list[VocabularyRecord], incoming: list[VocabularyRecord]
) -> tuple[list[VocabularyRecord], dict[str, int]]:
    """Merge mechanical imports while preserving existing human enrichment."""
    by_id = {record.id: record for record in existing}
    counts = {"added": 0, "updated": 0, "unchanged": 0}

    for new in incoming:
        old = by_id.get(new.id)
        if old is None:
            by_id[new.id] = new
            counts["added"] += 1
            continue

        merged = VocabularyRecord(
            id=old.id,
            expression=new.expression or old.expression,
            reading=new.reading or old.reading,
            furigana=new.furigana or old.furigana,
            romaji=new.romaji or old.romaji,
            meanings=new.meanings or old.meanings,
            part_of_speech=new.part_of_speech or old.part_of_speech,
            verb_group=new.verb_group or old.verb_group,
            transitivity=new.transitivity or old.transitivity,
            examples=new.examples or old.examples,
            conjugations=new.conjugations or old.conjugations,
            tags=sorted(set(old.tags) | set(new.tags)),
            usage_notes=new.usage_notes or old.usage_notes,
            audio=new.audio or old.audio,
            image=new.image or old.image,
            source=new.source,
        )
        if merged.to_dict() == old.to_dict():
            counts["unchanged"] += 1
        else:
            counts["updated"] += 1
        by_id[new.id] = merged

    return sorted(by_id.values(), key=lambda item: item.id), counts
