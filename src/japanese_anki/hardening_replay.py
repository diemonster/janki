"""Deterministic, offline replay for hardening case bundles."""

from __future__ import annotations

import contextlib
import io
import json
import re
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
    identifiers,
    jpdb,
    kanji,
    promote,
    qc,
    repairs,
    review,
    staging,
    status,
    validation,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.exporters.anki import (
    FIELD_NAMES,
    build_deck,
    deck_kind,
    resolve_deck_records,
)
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


def _structured_schema(factory: Callable[[], Any], where: str) -> Any:
    try:
        return factory()
    except ImportError as exc:
        raise hardening.HardeningError(
            f"{where} needs the AI schema dependency. Install the project with '.[ai]'."
        ) from exc


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
    _only(
        data,
        {
            "candidates",
            "source_units",
            "model_reported_unit_count",
            "mode",
            "source_name",
            "source_fingerprint",
            "oracle_units",
            "observe_coverage",
            "known_ids",
            "observe",
        },
        "candidate-response",
    )
    observe_coverage = _optional_bool(data, "observe_coverage")
    if not observe_coverage and (
        "oracle_units" in data or "source_fingerprint" in data
    ):
        raise hardening.HardeningError(
            "candidate-response oracle fields require observe_coverage"
        )
    raw_candidates = data.get("candidates")
    if not isinstance(raw_candidates, list):
        raise hardening.HardeningError("candidates must be a JSON list")
    response_type = _structured_schema(
        extract.candidate_schema, "candidate-response replay"
    )
    candidate_type = response_type.model_fields["candidates"].annotation.__args__[0]
    candidate_fields = set(candidate_type.model_fields)
    candidates = []
    for index, item in enumerate(raw_candidates):
        candidate = _mapping(item, f"candidates[{index}]")
        _only(candidate, candidate_fields, f"candidates[{index}]")
        candidates.append(_model_validate(candidate_type, candidate, f"candidates[{index}]"))
    raw_units = data.get("source_units", [])
    if not isinstance(raw_units, list):
        raise hardening.HardeningError("source_units must be a JSON list")
    unit_type = response_type.model_fields["source_units"].annotation.__args__[0]
    unit_fields = set(unit_type.model_fields)
    units = []
    for index, item in enumerate(raw_units):
        unit = _mapping(item, f"source_units[{index}]")
        _only(unit, unit_fields, f"source_units[{index}]")
        units.append(_model_validate(unit_type, unit, f"source_units[{index}]"))
    mode = data.get("mode")
    if mode not in {None, *extract.MODES}:
        raise hardening.HardeningError("mode must be table, prose, or null")
    response = _model_validate(
        response_type,
        {
            "candidates": candidates,
            "source_units": units,
            "model_reported_unit_count": data.get("model_reported_unit_count", 0),
        },
        "candidate-response",
    )
    result = extract.normalize_response(
        response, mode, str(data.get("source_name", "offline-fixture.png"))
    )
    known_ids = _string_list(data.get("known_ids", []), "known_ids")
    prepared = PreparedInput(
        kind="image",
        media_type="image/png",
        data_b64="",
        origin_path=Path(str(data.get("source_name", "offline-fixture.png"))),
    )
    records = extract.build_records(result.candidates, prepared, known_ids)
    output = {
        "records": _observed_records(records, data.get("observe", [])),
        "record_ids": [record.id for record in records],
        "unusable": len(extract.unusable(result.candidates)),
    }
    if observe_coverage:
        raw_oracle_units = data.get("oracle_units")
        if not isinstance(raw_oracle_units, list) or not raw_oracle_units:
            raise hardening.HardeningError(
                "observe_coverage requires a non-empty oracle_units list"
            )
        oracle_units: list[hardening.OracleUnit] = []
        allowed = {
            "page",
            "section",
            "ordinal",
            "context_fingerprint",
            "disposition",
        }
        for index, item in enumerate(raw_oracle_units):
            raw = _mapping(item, f"oracle_units[{index}]")
            _only(raw, allowed, f"oracle_units[{index}]")
            page = raw.get("page")
            ordinal = raw.get("ordinal")
            if (
                isinstance(page, bool)
                or not isinstance(page, int)
                or page < 1
                or isinstance(ordinal, bool)
                or not isinstance(ordinal, int)
                or ordinal < 1
            ):
                raise hardening.HardeningError(
                    f"oracle_units[{index}] page and ordinal must be positive integers"
                )
            section = raw.get("section")
            fingerprint = raw.get("context_fingerprint")
            disposition = raw.get("disposition")
            if not isinstance(section, str) or not section:
                raise hardening.HardeningError(
                    f"oracle_units[{index}].section must be text"
                )
            if not re.fullmatch(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", section):
                raise hardening.HardeningError(
                    f"oracle_units[{index}].section must be a lowercase slug"
                )
            if (
                not isinstance(fingerprint, str)
                or len(fingerprint) != 64
                or any(character not in "0123456789abcdef" for character in fingerprint)
            ):
                raise hardening.HardeningError(
                    f"oracle_units[{index}].context_fingerprint must be SHA-256"
                )
            if disposition not in hardening.UNIT_DISPOSITIONS:
                raise hardening.HardeningError(
                    f"oracle_units[{index}].disposition is invalid"
                )
            oracle_units.append(
                hardening.OracleUnit(
                    page, section, ordinal, fingerprint, disposition
                )
            )
        oracle_keys = [unit.key for unit in oracle_units]
        if len(set(oracle_keys)) != len(oracle_keys):
            raise hardening.HardeningError("oracle_units has duplicate unit keys")
        if oracle_keys != sorted(oracle_keys):
            raise hardening.HardeningError(
                "oracle_units must use page, section, ordinal order"
            )
        source_fingerprint = data.get("source_fingerprint")
        if (
            not isinstance(source_fingerprint, str)
            or not re.fullmatch(r"[0-9a-f]{64}", source_fingerprint)
        ):
            raise hardening.HardeningError(
                "observe_coverage requires a SHA-256 source_fingerprint"
            )
        oracle = hardening.UnitOracle(
            path=Path("offline-oracle.yaml"),
            relative_path="offline-oracle.yaml",
            id="offline-oracle",
            source_fingerprint=source_fingerprint,
            type="exhaustive",
            case_ids=(),
            units=tuple(oracle_units),
            targets=(),
            selection_rubric=None,
            approval=None,
        )
        block = extract.coverage_block(
            result,
            source_sha256=oracle.source_fingerprint,
            mode=mode,
            oracle=oracle,
            oracle_fingerprint=hardening.oracle_content_fingerprint(oracle),
        )
        output["coverage"] = {
            key: block[key]
            for key in (
                "status",
                "model_reported_unit_count",
                "observed_unit_count",
                "missing_units",
                "unexpected_units",
                "duplicate_keys",
                "context_mismatched_units",
                "disposition_mismatched_units",
                "candidate_units",
                "duplicate_units",
                "omission_units",
                "unreadable_units",
            )
        }
    return output


def _input_provenance(data: dict[str, Any], root: Path) -> Any:
    """Run the real CLI over a source in the parent inbox."""
    del root
    _only(
        data,
        {
            "source_name",
            "content",
            "scan_content",
            "scan_source_name",
            "external_content",
        },
        "input-provenance",
    )
    source_name = data.get("source_name")
    content = data.get("content")
    scan_content = data.get("scan_content")
    scan_source_name = data.get("scan_source_name", source_name)
    external_content = data.get("external_content")
    if (
        not isinstance(source_name, str)
        or not source_name
        or Path(source_name).name != source_name
    ):
        raise hardening.HardeningError(
            "input-provenance.source_name must be one filename"
        )
    if not isinstance(content, str) or not content:
        raise hardening.HardeningError(
            "input-provenance.content must be non-empty text"
        )
    if (
        not isinstance(scan_source_name, str)
        or not scan_source_name
        or Path(scan_source_name).name != scan_source_name
    ):
        raise hardening.HardeningError(
            "input-provenance.scan_source_name must be one filename"
        )
    if scan_content is not None and (
        not isinstance(scan_content, str) or not scan_content
    ):
        raise hardening.HardeningError(
            "input-provenance.scan_content must be non-empty text when present"
        )
    if external_content is not None and (
        not isinstance(external_content, str) or not external_content
    ):
        raise hardening.HardeningError(
            "input-provenance.external_content must be non-empty text when present"
        )
    with tempfile.TemporaryDirectory() as directory:
        # Import here because cli imports this replay module. At execution time
        # both modules are complete, so the local import does not form a cycle.
        from japanese_anki import cli

        project = Path(directory).resolve()
        inbox = project / "data" / "inbox"
        scan_inbox = inbox / "scans"
        inbox.mkdir(parents=True)
        source = inbox / source_name
        source.write_text(content, encoding="utf-8")
        if scan_content is not None:
            scan_inbox.mkdir(parents=True)
            (scan_inbox / scan_source_name).write_text(
                scan_content, encoding="utf-8"
            )
        command_source = source
        if external_content is not None:
            desk = project / "desk"
            desk.mkdir()
            command_source = desk / source_name
            command_source.write_text(external_content, encoding="utf-8")
        (project / "janki.toml").write_text(
            '[project]\nname = "Input provenance replay"\n', encoding="utf-8"
        )
        docs = project / "docs"
        docs.mkdir()
        (docs / "JAPANESE_STYLE_GUIDE.md").write_text(
            "Use the stored source.", encoding="utf-8"
        )

        observed_origins: list[Path] = []
        real_extract = extract.extract_candidates

        def offline_extract(prepared: PreparedInput, **_kwargs: Any) -> Any:
            observed_origins.append(prepared.origin_path)
            return extract.ExtractionResult((), (), 0)

        stdout = io.StringIO()
        stderr = io.StringIO()
        extract.extract_candidates = offline_extract
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(
                stderr
            ):
                exit_code = cli.main(
                    ["--root", str(project), "extract", str(command_source)]
                )
        finally:
            extract.extract_candidates = real_extract

        stored_files = sorted(
            path.relative_to(project).as_posix()
            for path in inbox.rglob("*")
            if path.is_file()
        )
        error = stderr.getvalue()
        error_named_files = sorted(
            path.relative_to(project).as_posix()
            for path in inbox.rglob("*")
            if path.is_file() and str(path) in error
        )
        return {
            "exit_code": exit_code,
            "origin_relative_path": (
                observed_origins[0].relative_to(project).as_posix()
                if observed_origins
                else None
            ),
            "scan_copy_exists": (scan_inbox / scan_source_name).exists(),
            "stored_files": stored_files,
            "error_has_name_collision": "same basename" in error,
            "error_named_files": error_named_files,
        }


def _extraction_prompt(data: dict[str, Any], root: Path) -> Any:
    """Observe whether extraction binds the applicable human oracle facts."""
    del root
    _only(
        data,
        {
            "source_name",
            "known",
            "unit_keys",
            "selection_targets",
            "selection_rubric",
        },
        "extraction-prompt",
    )
    source_name = data.get("source_name")
    if not isinstance(source_name, str) or not source_name:
        raise hardening.HardeningError("source_name must be non-empty text")
    known = _string_list(data.get("known", []), "known")
    raw_keys = data.get("unit_keys", [])
    raw_targets = data.get("selection_targets", [])
    if bool(raw_keys) == bool(raw_targets):
        raise hardening.HardeningError(
            "extraction-prompt requires exactly one of unit_keys or selection_targets"
        )
    if not isinstance(raw_keys, list):
        raise hardening.HardeningError("unit_keys must be a JSON list")
    unit_keys: list[tuple[int, str, int]] = []
    for index, value in enumerate(raw_keys):
        item = _mapping(value, f"unit_keys[{index}]")
        _only(item, {"page", "section", "ordinal"}, f"unit_keys[{index}]")
        page = item.get("page")
        section = item.get("section")
        ordinal = item.get("ordinal")
        if (
            isinstance(page, bool)
            or not isinstance(page, int)
            or page < 1
            or not isinstance(section, str)
            or not re.fullmatch(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", section)
            or isinstance(ordinal, bool)
            or not isinstance(ordinal, int)
            or ordinal < 1
        ):
            raise hardening.HardeningError(
                f"unit_keys[{index}] must contain a positive page and ordinal "
                "and a lowercase section slug"
            )
        unit_keys.append((page, section, ordinal))
    if len(set(unit_keys)) != len(unit_keys):
        raise hardening.HardeningError("unit_keys contains duplicate keys")
    if unit_keys != sorted(unit_keys):
        raise hardening.HardeningError("unit_keys must be in source order")

    if unit_keys:
        prompt = extract.prompt_for(source_name, known, unit_keys)
        expected = [
            f"page={page} section={section} ordinal={ordinal}"
            for page, section, ordinal in unit_keys
        ]
        return {
            "prompt_has_unit_keys": all(key in prompt for key in expected),
        }

    if not isinstance(raw_targets, list) or not raw_targets:
        raise hardening.HardeningError(
            "selection_targets must be a non-empty JSON list"
        )
    targets: list[tuple[str, str]] = []
    for index, value in enumerate(raw_targets):
        item = _mapping(value, f"selection_targets[{index}]")
        _only(item, {"identity", "locator"}, f"selection_targets[{index}]")
        identity = item.get("identity")
        locator = item.get("locator")
        if not isinstance(identity, str) or not identity.startswith("word:"):
            raise hardening.HardeningError(
                f"selection_targets[{index}].identity must be a word identity"
            )
        if not isinstance(locator, str) or not locator:
            raise hardening.HardeningError(
                f"selection_targets[{index}].locator must be non-empty text"
            )
        targets.append((identity, locator))
    if len(set(targets)) != len(targets):
        raise hardening.HardeningError("selection_targets contains duplicates")
    rubric = data.get("selection_rubric")
    if not isinstance(rubric, str) or not rubric.strip():
        raise hardening.HardeningError(
            "selection_rubric must be non-empty text"
        )
    try:
        prompt = extract.prompt_for(
            source_name,
            known,
            selection_targets=targets,
            selection_rubric=rubric,
        )
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        prompt = extract.prompt_for(source_name, known)
    return {
        "prompt_has_selection_targets": all(
            f"identity={identity} locator={locator}" in prompt
            for identity, locator in targets
        ),
        "prompt_has_selection_rubric": rubric in prompt,
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
        if skip and data.get("responses"):
            raise hardening.HardeningError(
                "staging-promote cannot declare responses when skip_reading_check is true"
            )
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
    _only(data, {"records", "repair_spilled_punctuation"}, "validation-qc")
    records = _records(data.get("records"))
    if _optional_bool(data, "repair_spilled_punctuation"):
        records = [
            dataclass_replace(
                record,
                examples=[
                    dataclass_replace(
                        example,
                        furigana=qc.repair_spilled_punctuation(example.furigana),
                    )
                    for example in record.examples
                ],
            )
            for record in records
        ]
    issues = validation.validate_records(records, "offline-case")
    return {
        "errors": sum(issue.level == "error" for issue in issues),
        "warnings": sum(issue.level == "warning" for issue in issues),
        "diagnostics": [
            {"code": issue.code, "level": issue.level, "record_id": issue.record_id}
            for issue in issues
        ],
        "furigana": [
            {
                "id": record.id,
                "record": record.furigana,
                "examples": [example.furigana for example in record.examples],
            }
            for record in records
        ],
        "spilled_furigana": [
            {
                "id": record.id,
                "groups": [list(group) for group in qc.spilled_furigana_groups(record.furigana)],
            }
            for record in records
        ],
    }


def _repair_plan(data: dict[str, Any], root: Path) -> Any:
    del root
    _only(data, {"records", "codes", "observe"}, "repair-plan")
    records = _records(data.get("records"))
    codes = _string_list(data.get("codes", []), "codes")
    declarations = repairs.REGISTRY.select(codes)
    updated, changes = repairs.apply_declarations(
        records,
        declarations,
        modes=frozenset({"ingest-safe"}),
    )
    issues = validation.validate_records(updated, "offline-case")
    return {
        "records": _observed_records(updated, data.get("observe", [])),
        "changes": [change.to_dict() for change in changes],
        "errors": sum(issue.level == "error" for issue in issues),
    }


def _fixture_kanji_store(raw_kanji: Any, where: str) -> Any:
    if not isinstance(raw_kanji, dict):
        raise hardening.HardeningError(f"{where} must be a JSON object")
    with tempfile.TemporaryDirectory(prefix="janki-replay-") as temporary:
        store_path = Path(temporary) / "kanji.json"
        store_path.write_text(
            json.dumps(raw_kanji, ensure_ascii=False), encoding="utf-8"
        )
        return kanji.load_store(store_path)


def _dictionary_enrichment(data: dict[str, Any], root: Path) -> Any:
    del root
    _only(
        data,
        {"records", "responses", "kanji", "force_fields", "ids", "observe"},
        "dictionary-enrichment",
    )
    records = _records(data.get("records"))
    client, transport = _client(data.get("responses", []))
    store = _fixture_kanji_store(data.get("kanji", {}), "kanji")
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
            "kanji",
            "force_fields",
            "known_expressions",
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
        _model_validate(
            _structured_schema(enrich.ai_schema, "ai-enrichment replay"),
            parsed_raw,
            "response.parsed",
        )
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
    store = (
        _fixture_kanji_store(data["kanji"], "kanji")
        if "kanji" in data
        else None
    )
    # Normalized like the production set (`enrich._known_expressions`): a
    # fixture written with decomposed dakuten must exercise the same bound the
    # live paths apply, or the gating oracle measures a third behavior.
    known = (
        frozenset(
            normalized
            for item in _string_list(data["known_expressions"], "known_expressions")
            if (normalized := identifiers.normalize_identity_part(item))
        )
        if "known_expressions" in data
        else None
    )
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
        kanji_store=store,
        known_expressions=known,
    )
    if transport is not None:
        transport.finish()
    output = {
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
    if known is not None:
        # Only when the input opted into the learner-load bound, so the oracles
        # of cases written before the key existed keep their exact shape — and
        # the same condition that enabled the bound gates its oracle key.
        output["load_held_ids"] = sorted(result.load_held)
    return output


def _ai_enrichment_prompt(data: dict[str, Any], root: Path) -> Any:
    """Observe the contracts sent to the AI enrichment model."""
    del root
    _only(data, {"records", "observe_example_pinning"}, "ai-enrichment-prompt")
    records = _records(data.get("records"))
    if len(records) != 1:
        raise hardening.HardeningError(
            "ai-enrichment-prompt requires exactly one record"
        )
    combined = enrich.AI_INSTRUCTIONS + "\n" + enrich.ai_prompt(records[0])
    output = {
        "exact_spelling_required": (
            "Use the exact spelling shown in Expression" in combined
        ),
        "expression_in_prompt": records[0].expression in combined,
    }
    if _optional_bool(data, "observe_example_pinning"):
        # The selection comes from the same first-class function the prompt
        # renders (`enrich.pinned_examples`) — probing the prompt's quoting
        # would let a wording change silently disarm this gate. The block
        # header is still checked in the text, so a selection the prompt
        # failed to render (or a render with an empty selection) shows up as
        # the two fields disagreeing. Opt-in, so cases written before this
        # key keep their shape.
        output["pins_existing_examples"] = (
            "Existing curated examples need annotations" in combined
        )
        output["pinned_sentences"] = [
            example.japanese for example in enrich.pinned_examples(records[0])
        ]
    return output


def _semantic_review_recheck(data: dict[str, Any], root: Path) -> Any:
    """Replay an initial review and any production-managed recheck."""
    del root
    _only(data, {"records", "model", "responses"}, "semantic-review-recheck")
    records = _records(data.get("records"))
    if len(records) != 1:
        raise hardening.HardeningError(
            "semantic-review-recheck requires exactly one record"
        )
    raw_responses = data.get("responses")
    if not isinstance(raw_responses, list):
        raise hardening.HardeningError("responses must be a JSON list")
    responses = list(raw_responses)
    prompts: list[str] = []

    def canned_call(
        model: str,
        blocks: Any,
        content: Any,
        schema: Any,
        client: Any = None,
        **options: Any,
    ) -> claude_client.CallResult:
        del model, blocks, schema, client, options
        if not responses:
            raise hardening.HardeningError(
                "Canned review responses ended before the review finished"
            )
        prompts.append(json.dumps(content, ensure_ascii=False, sort_keys=True))
        raw = _mapping(responses.pop(0), "responses entry")
        _only(raw, {"parsed", "stop_reason"}, "responses entry")
        parsed_raw = raw.get("parsed")
        parsed = (
            _model_validate(
                _structured_schema(review.review_schema, "semantic review replay"),
                parsed_raw,
                "responses entry.parsed",
            )
            if parsed_raw is not None
            else None
        )
        return claude_client.CallResult(parsed, raw.get("stop_reason"), None)

    reviewed, failures = review.review_records(
        records,
        model=str(data.get("model", "offline-model")),
        style_guide="Offline hardening replay.",
        parse_call=canned_call,
    )
    entry = next(iter(reviewed.values()), None)
    return {
        "call_count": len(prompts),
        "responses_remaining": len(responses),
        "recheck_prompt_has_pitch_facts": (
            len(prompts) > 1 and "Mechanical pitch facts:" in prompts[1]
        ),
        "recheck_prompt_binds_source_authority": (
            len(prompts) > 1
            and "content-bound jpdb source marker" in prompts[1]
            and "do not replace this valid lexical pattern from model memory"
            in prompts[1]
        ),
        "findings": [finding.to_dict() for finding in entry.findings] if entry else [],
        "failures": failures,
    }


def _ai_enrichment_retry(data: dict[str, Any], root: Path) -> Any:
    """Replay a sequence of structured AI answers through the full pass."""
    del root
    _only(data, {"records", "model", "responses"}, "ai-enrichment-retry")
    records = _records(data.get("records"))
    if len(records) != 1:
        raise hardening.HardeningError(
            "ai-enrichment-retry requires exactly one record"
        )
    raw_responses = data.get("responses")
    if not isinstance(raw_responses, list):
        raise hardening.HardeningError("responses must be a JSON list")
    responses = list(raw_responses)
    prompts: list[str] = []

    def canned_call(
        model: str,
        blocks: Any,
        content: str,
        schema: Any,
        client: Any = None,
        **options: Any,
    ) -> claude_client.CallResult:
        del model, blocks, schema, client, options
        if not responses:
            raise hardening.HardeningError(
                "Canned AI responses ended before the enrichment pass finished"
            )
        prompts.append(content)
        raw = _mapping(responses.pop(0), "responses entry")
        _only(raw, {"parsed", "stop_reason", "refusal"}, "responses entry")
        parsed_raw = raw.get("parsed")
        parsed = (
            _model_validate(
                _structured_schema(enrich.ai_schema, "ai-enrichment retry replay"),
                parsed_raw,
                "responses entry.parsed",
            )
            if parsed_raw is not None
            else None
        )
        refusal_raw = raw.get("refusal")
        refusal = None
        if refusal_raw is not None:
            refusal_data = _mapping(refusal_raw, "responses entry.refusal")
            _only(
                refusal_data,
                {"category", "explanation"},
                "responses entry.refusal",
            )
            try:
                refusal = claude_client.Refusal(**refusal_data)
            except TypeError as exc:
                raise hardening.HardeningError(
                    f"Invalid structured data in responses entry.refusal: {exc}"
                ) from exc
        return claude_client.CallResult(parsed, raw.get("stop_reason"), refusal)

    result = enrich.enrich_ai(
        records,
        model=str(data.get("model", "offline-model")),
        style_guide="Offline hardening replay.",
        parse_call=canned_call,
    )
    record = result.records[0]
    return {
        "call_count": len(prompts),
        "responses_remaining": len(responses),
        "example_texts": [example.japanese for example in record.examples],
        "usage_notes": record.usage_notes,
        "rejected_ids": sorted(result.rejected),
        "no_changes": result.no_changes,
        "retry_prompt_has_allowed_forms": (
            len(prompts) > 1 and "Permitted written target forms:" in prompts[1]
        ),
    }


def _render_build(data: dict[str, Any], root: Path) -> Any:
    _only(data, {"records", "cards", "observe_fields"}, "render-build")
    records = _records(data.get("records"))
    cards = _mapping(data.get("cards", {"recognition": True}), "cards")
    _only(cards, {"recognition", "production", "reading"}, "cards")
    if not all(isinstance(value, bool) for value in cards.values()):
        raise hardening.HardeningError("cards values must be true or false")
    observe_fields = _string_list(
        data.get("observe_fields", ["RecordID"]), "observe_fields"
    )
    unknown_fields = sorted(set(observe_fields) - set(FIELD_NAMES))
    if unknown_fields:
        raise hardening.HardeningError(
            "observe_fields names unknown rendered field(s): "
            + ", ".join(unknown_fields)
        )
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
            field_rows = connection.execute("select flds from notes").fetchall()
        finally:
            connection.close()
    rendered = []
    for row in field_rows:
        values = dict(zip(FIELD_NAMES, row[0].split("\x1f"), strict=True))
        rendered.append(
            {
                "record_id": values["RecordID"],
                "fields": {name: values[name] for name in observe_fields},
            }
        )
    return {
        "notes": note_count,
        "cards": card_count,
        "record_count": result.note_count,
        "record_ids": list(result.record_ids),
        "rendered": sorted(rendered, key=lambda item: item["record_id"]),
        "package_entries": names,
    }


def _deck_membership(data: dict[str, Any], root: Path) -> Any:
    """Report empty word decks and stable IDs claimed by more than one."""
    _only(data, set(), "deck-membership")
    config = ProjectConfig.load(root)
    project_root = config.root.resolve()
    deck_root = config.deck_dir.resolve()
    if not deck_root.is_relative_to(project_root):
        raise hardening.HardeningError(
            "deck-membership configured deck directory must be inside the "
            f"repository: {config.deck_dir}"
        )
    discovered_paths = status.deck_files(config)
    resolved_paths: dict[Path, Path] = {}
    for path in discovered_paths:
        resolved_path = path.resolve()
        if not resolved_path.is_relative_to(project_root):
            raise hardening.HardeningError(
                "deck-membership discovered a deck outside the repository "
                f"through {config.deck_dir}: {path}"
            )
        previous = resolved_paths.get(resolved_path)
        if previous is not None:
            raise hardening.HardeningError(
                "deck-membership discovered two paths for the same deck file: "
                f"{previous} and {path}"
            )
        resolved_paths[resolved_path] = path
    deck_paths = [
        path
        for path in discovered_paths
        if deck_kind(path) in {"", "vocabulary"}
    ]
    if not deck_paths:
        raise hardening.HardeningError(
            "deck-membership found no vocabulary decks under the configured "
            f"deck directory: {config.deck_dir}"
        )

    memberships: dict[str, set[str]] = {}
    for path in deck_paths:
        value = path.relative_to(project_root).as_posix()
        _, records = resolve_deck_records(path)
        memberships[value] = {record.id for record in records}

    overlaps: dict[str, list[str]] = {}
    decks = list(memberships)
    for left_index, left in enumerate(decks):
        for right in decks[left_index + 1 :]:
            shared = sorted(memberships[left] & memberships[right])
            if shared:
                overlaps[f"{left} <> {right}"] = shared
    return {
        "decks": sorted(memberships),
        "empty_decks": sorted(
            name for name, record_ids in memberships.items() if not record_ids
        ),
        "overlaps": overlaps,
    }


RUNNERS: dict[str, Callable[[dict[str, Any], Path], Any]] = {
    "deck-membership": _deck_membership,
    "input-provenance": _input_provenance,
    "candidate-response": _candidate_response,
    "extraction-prompt": _extraction_prompt,
    "staging-promote": _staging_promote,
    "validation-qc": _validation_qc,
    "render-build": _render_build,
    "dictionary-enrichment": _dictionary_enrichment,
    "ai-enrichment": _ai_enrichment,
    "ai-enrichment-retry": _ai_enrichment_retry,
    "ai-enrichment-prompt": _ai_enrichment_prompt,
    "semantic-review-recheck": _semantic_review_recheck,
    "repair-plan": _repair_plan,
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
