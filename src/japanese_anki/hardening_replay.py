"""Deterministic, offline replay for hardening case bundles."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import yaml

from japanese_anki import (
    claude_client,
    enrich,
    extract,
    hardening,
    jpdb,
    kanji,
    promote,
    qc,
    staging,
    validation,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.exporters.anki import build_deck
from japanese_anki.inputs import PreparedInput
from japanese_anki.io import save_records_json
from japanese_anki.models import VocabularyRecord


@dataclass(frozen=True, slots=True)
class ReplayResult:
    case_id: str
    passed: bool
    gating: bool
    expected: Any
    actual: Any


class _Pairs(dict[str, Any]):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: _Pairs = _Pairs()
    for key, value in pairs:
        if key in result:
            raise hardening.HardeningError(f"Replay JSON has duplicate key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite number {value}")


def _json_bytes(raw: bytes, where: str) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except UnicodeDecodeError as exc:
        raise hardening.HardeningError(
            f"Could not decode {where} as UTF-8: {exc.reason}"
        ) from exc
    except (json.JSONDecodeError, ValueError) as exc:
        raise hardening.HardeningError(f"Could not parse {where}: {exc}") from exc


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise hardening.HardeningError(f"{where} must be a JSON object")
    return value


def _only(data: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise hardening.HardeningError(
            f"{where} has unknown field(s): {', '.join(unknown)}"
        )


def _string_list(value: Any, where: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise hardening.HardeningError(f"{where} must be a JSON string list")
    if len(set(value)) != len(value):
        raise hardening.HardeningError(f"{where} contains duplicate values")
    return value


def _optional_bool(data: dict[str, Any], key: str, default: bool = False) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise hardening.HardeningError(f"{key} must be true or false")
    return value


def _model_validate(schema: Any, value: Any, where: str) -> Any:
    try:
        return schema.model_validate(value)
    except (TypeError, ValueError) as exc:
        raise hardening.HardeningError(f"Invalid structured data in {where}: {exc}") from exc


def _records(value: Any, where: str = "records") -> list[VocabularyRecord]:
    if not isinstance(value, list):
        raise hardening.HardeningError(f"{where} must be a JSON list")
    records: list[VocabularyRecord] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise hardening.HardeningError(f"{where}[{index}] must be a JSON object")
        allowed = set(VocabularyRecord(id="", expression="").to_dict())
        _only(item, allowed, f"{where}[{index}]")
        records.append(VocabularyRecord.from_dict(item))
    return records


def _observed_records(
    records: list[VocabularyRecord], observations: Any
) -> list[dict[str, Any]]:
    if not isinstance(observations, list):
        raise hardening.HardeningError("observe must be a JSON list")
    by_id = {record.id: record for record in records}
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(observations):
        item = _mapping(raw, f"observe[{index}]")
        _only(item, {"id", "fields"}, f"observe[{index}]")
        record_id = str(item.get("id", ""))
        fields = _string_list(item.get("fields"), f"observe[{index}].fields")
        if record_id not in by_id:
            raise hardening.HardeningError(f"observe names unknown record {record_id!r}")
        payload = by_id[record_id].to_dict()
        unknown = [name for name in fields if name not in payload]
        if unknown:
            raise hardening.HardeningError(
                f"observe[{index}].fields names unknown field(s): {', '.join(unknown)}"
            )
        result.append(
            {
                "id": record_id,
                "fields": {name: payload[name] for name in fields},
            }
        )
    return result


class _CannedTransport:
    def __init__(self, responses: Any) -> None:
        if not isinstance(responses, list):
            raise hardening.HardeningError("responses must be a JSON list")
        self.responses = list(responses)
        self.calls: list[str] = []

    def __call__(
        self, url: str, body: dict[str, Any], headers: dict[str, str]
    ) -> tuple[int, Any]:
        del headers
        endpoint = url.rsplit("/api/v1/", 1)[-1]
        if not self.responses:
            raise hardening.HardeningError(
                f"Canned jpdb responses ended before request {endpoint!r}"
            )
        raw = _mapping(self.responses.pop(0), "responses entry")
        _only(raw, {"endpoint", "body", "status", "payload"}, "responses entry")
        expected_endpoint = str(raw.get("endpoint", ""))
        if expected_endpoint != endpoint:
            raise hardening.HardeningError(
                f"Canned jpdb response expected {expected_endpoint!r}, got {endpoint!r}"
            )
        if "body" in raw and raw["body"] != body:
            raise hardening.HardeningError(
                f"Canned jpdb request body does not match for {endpoint!r}"
            )
        status = raw.get("status", 200)
        if isinstance(status, bool) or not isinstance(status, int):
            raise hardening.HardeningError("Canned jpdb status must be an integer")
        self.calls.append(endpoint)
        return status, raw.get("payload")

    def finish(self) -> None:
        if self.responses:
            raise hardening.HardeningError(
                f"Replay left {len(self.responses)} unused canned jpdb response(s)"
            )


def _client(responses: Any) -> tuple[jpdb.JpdbClient, _CannedTransport]:
    transport = _CannedTransport(responses)
    return (
        jpdb.JpdbClient(
            "offline-replay",
            transport,
            max_tries=1,
            sleep=lambda _seconds: None,
            jitter=lambda: 0.0,
        ),
        transport,
    )


def _candidate_response(data: dict[str, Any], root: Path) -> Any:
    del root
    _only(data, {"candidates", "source_name", "known_ids", "observe"}, "candidate-response")
    raw_candidates = data.get("candidates")
    if not isinstance(raw_candidates, list):
        raise hardening.HardeningError("candidates must be a JSON list")
    candidate_type = extract.candidate_schema().model_fields[
        "candidates"
    ].annotation.__args__[0]
    candidate_fields = set(candidate_type.model_fields)
    candidates = []
    for index, item in enumerate(raw_candidates):
        candidate = _mapping(item, f"candidates[{index}]")
        _only(candidate, candidate_fields, f"candidates[{index}]")
        candidates.append(_model_validate(candidate_type, candidate, f"candidates[{index}]"))
    known_ids = _string_list(data.get("known_ids", []), "known_ids")
    prepared = PreparedInput(
        kind="image",
        media_type="image/png",
        data_b64="",
        origin_path=Path(str(data.get("source_name", "offline-fixture.png"))),
    )
    records = extract.build_records(candidates, prepared, known_ids)
    return {
        "records": _observed_records(records, data.get("observe", [])),
        "record_ids": [record.id for record in records],
        "unusable": len(extract.unusable(candidates)),
    }


def _staging_promote(data: dict[str, Any], root: Path) -> Any:
    del root
    _only(
        data,
        {
            "records",
            "metadata",
            "skip_reading_check",
            "responses",
            "already_stored",
            "remint_blocked",
        },
        "staging-promote",
    )
    records = _records(data.get("records"))
    with tempfile.TemporaryDirectory(prefix="janki-replay-") as temporary:
        path = Path(temporary) / "staging.yaml"
        staging.write_staging(
            path,
            records,
            _mapping(data.get("metadata", {}), "metadata"),
        )
        loaded, metadata = staging.read_staging(path)
        client: jpdb.JpdbClient | None = None
        transport: _CannedTransport | None = None
        skip = _optional_bool(data, "skip_reading_check")
        if not skip:
            client, transport = _client(data.get("responses", []))
        outcome = promote.check_readings(
            loaded,
            client=client,
            skip_reading_check=skip,
            already_stored=set(
                _string_list(data.get("already_stored", []), "already_stored")
            ),
            remint_blocked=_optional_bool(data, "remint_blocked"),
        )
        held = iter(outcome.held)
        promoted = iter(outcome.promoted)
        rewritten = [
            next(held) if keep else next(promoted) for keep in outcome.keep
        ]
        staging.rewrite_staging(path, rewritten)
        round_trip, round_trip_meta = staging.read_staging(path)
        if transport is not None:
            transport.finish()
    return {
        "promoted_ids": [record.id for record in outcome.promoted],
        "held": [
            {
                "id": record.id,
                "reason": record.source.raw_fields.get("hold_reason", ""),
            }
            for record in outcome.held
        ],
        "reminted": outcome.reminted,
        "metadata_keys": sorted(metadata),
        "round_trip_metadata_equal": round_trip_meta == metadata,
        "round_trip_record_ids": [record.id for record in round_trip],
        "warning_count": len(outcome.warnings),
    }


def _validation_qc(data: dict[str, Any], root: Path) -> Any:
    del root
    _only(data, {"records"}, "validation-qc")
    records = _records(data.get("records"))
    issues = validation.validate_records(records, "offline-case")
    return {
        "errors": sum(issue.level == "error" for issue in issues),
        "warnings": sum(issue.level == "warning" for issue in issues),
        "spilled_furigana": [
            {
                "id": record.id,
                "groups": [list(group) for group in qc.spilled_furigana_groups(record.furigana)],
            }
            for record in records
        ],
    }


def _dictionary_enrichment(data: dict[str, Any], root: Path) -> Any:
    del root
    _only(
        data,
        {"records", "responses", "kanji", "force_fields", "ids", "observe"},
        "dictionary-enrichment",
    )
    records = _records(data.get("records"))
    client, transport = _client(data.get("responses", []))
    raw_kanji = data.get("kanji", {})
    if not isinstance(raw_kanji, dict):
        raise hardening.HardeningError("kanji must be a JSON object")
    with tempfile.TemporaryDirectory(prefix="janki-replay-") as temporary:
        store_path = Path(temporary) / "kanji.json"
        store_path.write_text(
            json.dumps(raw_kanji, ensure_ascii=False), encoding="utf-8"
        )
        store = kanji.load_store(store_path)
    outcome = enrich.enrich_records(
        client,
        records,
        force_fields=tuple(_string_list(data.get("force_fields", []), "force_fields")),
        ids=(
            _string_list(data["ids"], "ids") if "ids" in data else None
        ),
        kanji_store=store,
    )
    transport.finish()
    return {
        "records": _observed_records(outcome.records, data.get("observe", [])),
        "changes": {
            record_id: sorted(fields) for record_id, fields in sorted(outcome.changes.items())
        },
        "looked_up": outcome.looked_up,
        "skipped": outcome.skipped,
        "warning_count": len(outcome.warnings),
        "calls": transport.calls,
    }


def _ai_enrichment(data: dict[str, Any], root: Path) -> Any:
    del root
    _only(
        data,
        {
            "records",
            "model",
            "response",
            "sentence_responses",
            "force_fields",
            "observe",
        },
        "ai-enrichment",
    )
    records = _records(data.get("records"))
    if len(records) != 1:
        raise hardening.HardeningError("ai-enrichment requires exactly one record")
    response = _mapping(data.get("response"), "response")
    _only(response, {"parsed", "stop_reason", "refusal"}, "response")
    parsed_raw = response.get("parsed")
    parsed = (
        _model_validate(enrich.ai_schema(), parsed_raw, "response.parsed")
        if parsed_raw is not None
        else None
    )
    refusal_raw = response.get("refusal")
    refusal = None
    if refusal_raw is not None:
        refusal_data = _mapping(refusal_raw, "response.refusal")
        _only(refusal_data, {"category", "explanation"}, "response.refusal")
        try:
            refusal = claude_client.Refusal(**refusal_data)
        except TypeError as exc:
            raise hardening.HardeningError(
                f"Invalid structured data in response.refusal: {exc}"
            ) from exc
    client: jpdb.JpdbClient | None = None
    transport: _CannedTransport | None = None
    if "sentence_responses" in data:
        client, transport = _client(data["sentence_responses"])
    result = enrich.AiResult(records=list(records), looked_up=1)
    enrich.absorb_ai_call(
        result,
        records[0],
        claude_client.CallResult(parsed, response.get("stop_reason"), refusal),
        model=str(data.get("model", "offline-model")),
        positions={records[0].id: 0},
        recent=[],
        force_fields=tuple(_string_list(data.get("force_fields", []), "force_fields")),
        jpdb_client=client,
    )
    if transport is not None:
        transport.finish()
    return {
        "records": _observed_records(result.records, data.get("observe", [])),
        "changes": {
            record_id: sorted(fields) for record_id, fields in sorted(result.changes.items())
        },
        "no_changes": result.no_changes,
        "report": enrich.format_ai_no_changes(result),
        "rejected_ids": sorted(result.rejected),
        "unverified_ids": sorted(result.unverified),
        "warning_count": len(result.warnings),
    }


def _render_build(data: dict[str, Any], root: Path) -> Any:
    _only(data, {"records", "cards"}, "render-build")
    records = _records(data.get("records"))
    cards = _mapping(data.get("cards", {"recognition": True}), "cards")
    _only(cards, {"recognition", "production", "reading"}, "cards")
    if not all(isinstance(value, bool) for value in cards.values()):
        raise hardening.HardeningError("cards values must be true or false")
    with tempfile.TemporaryDirectory(prefix="janki-replay-") as temporary:
        temporary_root = Path(temporary)
        (temporary_root / "janki.toml").write_text(
            "[project]\nname = 'Offline hardening replay'\n[review]\nrequire = false\n",
            encoding="utf-8",
        )
        records_path = temporary_root / "records.json"
        save_records_json(records_path, records)
        deck_path = temporary_root / "deck.yaml"
        deck_path.write_text(
            yaml.safe_dump(
                {
                    "deck": {
                        "name": "Offline hardening replay",
                        "deck_id": 2059400991,
                        "model_id": 1607392991,
                        "source": "records.json",
                        "cards": cards,
                    }
                },
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        config = ProjectConfig.load(temporary_root)
        config = dataclass_replace(
            config,
            template_dir=root / "templates" / "japanese-study",
            kanji_file=temporary_root / "kanji.json",
            media_dir=temporary_root / "media",
        )
        output = temporary_root / "case.apkg"
        result = build_deck(deck_path, config, output)
        with ZipFile(output) as package:
            names = sorted(package.namelist())
            collection_name = (
                "collection.anki21" if "collection.anki21" in names else "collection.anki2"
            )
            database = temporary_root / "collection.db"
            database.write_bytes(package.read(collection_name))
        connection = sqlite3.connect(database)
        try:
            note_count = connection.execute("select count(*) from notes").fetchone()[0]
            card_count = connection.execute("select count(*) from cards").fetchone()[0]
            field_rows = connection.execute("select flds from notes order by id").fetchall()
        finally:
            connection.close()
    return {
        "notes": note_count,
        "cards": card_count,
        "record_count": result.note_count,
        "field_rows": [row[0].split("\x1f") for row in field_rows],
        "package_entries": names,
    }


RUNNERS: dict[str, Callable[[dict[str, Any], Path], Any]] = {
    "candidate-response": _candidate_response,
    "staging-promote": _staging_promote,
    "validation-qc": _validation_qc,
    "render-build": _render_build,
    "dictionary-enrichment": _dictionary_enrichment,
    "ai-enrichment": _ai_enrichment,
}


def discover_cases(
    root: Path, names: list[str] | tuple[str, ...] = ()
) -> tuple[hardening.HardeningCase, ...]:
    repository = hardening.load_repository(root)
    by_id = {case.id: case for case in repository.cases}
    if names:
        missing = sorted(set(names) - set(by_id))
        if missing:
            raise hardening.HardeningError(
                "Unknown hardening case(s): " + ", ".join(missing)
            )
        return tuple(by_id[name] for name in dict.fromkeys(names))
    return tuple(case for case in repository.cases if case.gating)


def replay_case(root: Path, case: hardening.HardeningCase) -> ReplayResult:
    fixtures = {fixture.path: fixture for fixture in case.fixtures}
    input_fixture = fixtures[case.runner_input]
    oracle_fixture = fixtures[case.oracle]
    case_dir = case.path.parent
    input_data = _mapping(
        _json_bytes(
            hardening.read_verified_bytes(
                case_dir / input_fixture.path,
                case_dir,
                input_fixture.fingerprint,
            ),
            f"{case.relative_path}:{input_fixture.path}",
        ),
        case.runner_input,
    )
    expected = _json_bytes(
        hardening.read_verified_bytes(
            case_dir / oracle_fixture.path,
            case_dir,
            oracle_fixture.fingerprint,
        ),
        f"{case.relative_path}:{oracle_fixture.path}",
    )
    actual = RUNNERS[case.runner](input_data, root.resolve())
    return ReplayResult(
        case_id=case.id,
        passed=actual == expected,
        gating=case.gating,
        expected=expected,
        actual=actual,
    )


def replay(
    root: Path, names: list[str] | tuple[str, ...] = ()
) -> tuple[ReplayResult, ...]:
    return tuple(replay_case(root, case) for case in discover_cases(root, names))


def replay_payload(results: tuple[ReplayResult, ...]) -> dict[str, Any]:
    return {
        "passed": all(result.passed for result in results),
        "cases": [
            {
                "id": result.case_id,
                "gating": result.gating,
                "passed": result.passed,
                **(
                    {}
                    if result.passed
                    else {"expected": result.expected, "actual": result.actual}
                ),
            }
            for result in results
        ],
    }


def format_replay(results: tuple[ReplayResult, ...]) -> list[str]:
    lines = [
        f"Hardening replay: {sum(result.passed for result in results)}/"
        f"{len(results)} passed"
    ]
    lines.extend(
        f"{'PASS' if result.passed else 'FAIL'} {result.case_id}"
        for result in results
    )
    return lines
