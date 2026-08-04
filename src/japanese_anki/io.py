from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from japanese_anki.models import VocabularyRecord


class DataError(RuntimeError):
    pass


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
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise DataError(f"Could not parse {path}: {exc}") from exc
    raise DataError(f"Unsupported file type for {path}; expected JSON or YAML")


def load_records(path: Path) -> list[VocabularyRecord]:
    data = load_structured(path)
    if isinstance(data, dict) and "records" in data:
        data = data["records"]
    if not isinstance(data, list):
        raise DataError(f"Expected a list of vocabulary records in {path}")
    return [VocabularyRecord.from_dict(item) for item in data]


def save_records_json(path: Path, records: list[VocabularyRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [record.to_dict() for record in sorted(records, key=lambda item: item.id)]
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


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
