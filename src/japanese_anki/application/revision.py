"""Plan, dispatch, and stage one owner-requested conjugation-deck revision.

``revise`` is the third card-writing model pass. It receives only explicitly
selected existing drill cards plus the owner's exact instruction, and its paid
answer becomes a JSON proposal under ``data/staging``. Nothing in this module
writes a canonical deck, generates audio, or records a review decision.

Planning is read-only. Execution plans again at the click boundary, compares
both the complete provider request and the wider local plan identity, prepares
the paid client before authority exists, and then uses the shared operation
journal and exact-response capture seam. A proposal is committed only after its
JSON is durable.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import shutil
import subprocess
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from japanese_anki import ai_schema, claude_client, operations, prompts
from japanese_anki.application import revision_provider
from japanese_anki.application.extraction import classify_dispatch_failure
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.exporters import pattern_cards
from japanese_anki.io import (
    DataError,
    atomic_write_text_bound,
    exclusive_path_lock,
    load_records,
    prepare_bound_directory,
    read_bytes_bound,
)

__all__ = [
    "RevisionApplicationError",
    "RevisionPlan",
    "RevisionRunError",
    "RevisionRunResult",
    "plan_revision",
    "recover_revision",
    "run_revision",
]


class RevisionApplicationError(JankiError):
    """A conjugation-deck revision cannot be planned or staged safely."""


class RevisionRunError(RevisionApplicationError):
    """A revision stopped after its paid-operation identity was allocated."""

    def __init__(
        self,
        message: str,
        *,
        operation_id: str,
        provider_dispatched: bool,
    ) -> None:
        self.operation_id = operation_id
        self.provider_dispatched = provider_dispatched
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class RevisionPlan:
    """The exact request and local state one owner confirmation can consume."""

    repository_root: Path
    deck_path: Path
    deck_relative_path: str
    deck_sha256: str
    selected_record_ids: tuple[str, ...]
    owner_instruction: str
    provider_plan: revision_provider.RevisionProviderPlan
    style_guide: str
    task_template: str
    user_turn: str
    canonical_context_fingerprint: str
    staging_path: Path
    staging_revision: str | None
    plan_fingerprint: str
    form: str
    current_form_note: str
    current_examples: Mapping[str, tuple[Mapping[str, str], ...]]
    canonical_context: Mapping[str, Mapping[str, Any]]

    @property
    def provider(self) -> str:
        return self.provider_plan.provider

    @property
    def model(self) -> str:
        return self.provider_plan.model

    @property
    def billing_class(self) -> str:
        return self.provider_plan.billing_class

    @property
    def billing_display(self) -> str:
        return self.provider_plan.billing_display

    @property
    def auth_metadata(self) -> Mapping[str, Any]:
        return self.provider_plan.auth_metadata

    @property
    def transport(self) -> Mapping[str, Any]:
        return self.provider_plan.transport

    @property
    def system_blocks(self) -> tuple[Mapping[str, Any], ...]:
        return self.provider_plan.system_blocks

    @property
    def schema(self) -> Any:
        return self.provider_plan.schema

    @property
    def response_schema_fingerprint(self) -> str:
        return self.provider_plan.response_schema_fingerprint

    @property
    def request_fingerprint(self) -> str:
        return self.provider_plan.request_fingerprint

    @property
    def can_dispatch(self) -> bool:
        """Whether the deterministic proposal target is still unoccupied."""
        return self.staging_revision is None


@dataclass(frozen=True, slots=True)
class RevisionRunResult:
    """One paid answer that is now a durable, unapproved staging proposal."""

    operation_id: str
    staging_path: Path
    request_fingerprint: str
    plan_fingerprint: str


def _report_progress(progress: Callable[[str], None] | None, label: str) -> None:
    if progress is not None:
        progress(label)


_SENT_EXAMPLE_FIELDS = (
    "japanese",
    "furigana",
    "english",
    "register",
)
_CURRENT_EXAMPLE_FIELDS = (
    "japanese",
    "furigana",
    "romaji",
    "english",
    "register",
    "audio",
    "spoken_japanese",
)


def _canonical_json(value: Any, *, pretty: bool = False) -> str:
    options: dict[str, Any] = {
        "ensure_ascii": False,
        "sort_keys": True,
    }
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return json.dumps(value, **options)


def _plain_json_value(value: Any) -> Any:
    """Copy immutable provider metadata into ordinary JSON containers."""
    if isinstance(value, Mapping):
        return {str(key): _plain_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_plain_json_value(item) for item in value]
    return value


def _fingerprint_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_json(raw: bytes, path: Path) -> Mapping[str, Any]:
    """Decode one request manifest without accepting duplicate keys or NaN."""

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RevisionApplicationError(
                    f"Revision manifest {path} repeats key {key!r}."
                )
            result[key] = value
        return result

    def invalid_constant(value: str) -> Any:
        raise RevisionApplicationError(
            f"Revision manifest {path} contains invalid JSON value {value!r}."
        )

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=object_pairs,
            parse_constant=invalid_constant,
        )
    except RevisionApplicationError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RevisionApplicationError(
            f"Could not parse revision manifest {path}: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise RevisionApplicationError(
            f"Revision manifest {path} must contain one JSON object."
        )
    return value


def _resolve_deck(config: ProjectConfig, deck: Path | str) -> Path:
    requested = Path(deck)
    target = (requested if requested.is_absolute() else config.deck_dir / requested).absolute()
    deck_root = config.deck_dir.absolute()
    try:
        target.relative_to(deck_root)
    except ValueError as exc:
        raise RevisionApplicationError(
            f"Revision target must be one deck under {deck_root}; nothing was sent."
        ) from exc
    if target.suffix.lower() not in {".yaml", ".yml"}:
        raise RevisionApplicationError(
            f"Revision target must be a YAML deck: {target}; nothing was sent."
        )
    return target


def _selected_ids(values: Sequence[str]) -> tuple[str, ...]:
    if not values:
        raise RevisionApplicationError(
            "Choose at least one record to revise; nothing was sent."
        )
    selected = tuple(values)
    if any(not isinstance(value, str) or not value.strip() for value in selected):
        raise RevisionApplicationError(
            "Every selected record id must be nonblank text; nothing was sent."
        )
    if len(selected) != len(set(selected)):
        raise RevisionApplicationError(
            "Each selected record id may appear only once; nothing was sent."
        )
    return selected


def _canonical_context(
    config: ProjectConfig,
    selected: tuple[str, ...],
    form: str,
) -> dict[str, Mapping[str, Any]]:
    records = load_records(config.normalized_file)
    by_id = {record.id: record for record in records}
    missing = [record_id for record_id in selected if record_id not in by_id]
    if missing:
        raise RevisionApplicationError(
            "Selected record(s) are missing from the canonical collection: "
            f"{', '.join(missing)}; nothing was sent."
        )
    selected_records = [by_id[record_id] for record_id in selected]
    computed = {
        record_id: card.result
        for card, record_id in pattern_cards.drill_cards(selected_records, form)
    }
    unconjugable = [record_id for record_id in selected if record_id not in computed]
    if unconjugable:
        raise RevisionApplicationError(
            f"Selected record(s) cannot currently be conjugated into {form}: "
            f"{', '.join(unconjugable)}; nothing was sent."
        )
    context: dict[str, Mapping[str, Any]] = {}
    for record_id in selected:
        record = by_id[record_id]
        context[record_id] = {
            "id": record.id,
            "expression": record.expression,
            "reading": record.reading,
            "meanings": list(record.meanings),
            "part_of_speech": record.part_of_speech,
            "verb_group": record.verb_group,
            "transitivity": record.transitivity,
            "target_conjugation": computed[record_id],
        }
    return context


def _user_turn(
    *,
    deck_name: str,
    form: str,
    form_note: str,
    selected: tuple[str, ...],
    instruction: str,
    current: Mapping[str, tuple[Mapping[str, str], ...]],
    context: Mapping[str, Mapping[str, Any]],
) -> str:
    cards: list[dict[str, Any]] = []
    for record_id in selected:
        sent_examples = [
            {field: example.get(field, "") for field in _SENT_EXAMPLE_FIELDS}
            for example in current[record_id]
        ]
        cards.append(
            {
                "record": context[record_id],
                "current_examples": sent_examples,
            }
        )
    return _canonical_json(
        {
            "owner_instruction": instruction,
            "deck": {
                "file": deck_name,
                "form": form,
                "current_form_note": form_note,
                "selected_cards": cards,
            },
        },
        pretty=True,
    ) + "\n"


def _staging_revision(path: Path) -> str | None:
    try:
        return _fingerprint_bytes(read_bytes_bound(path))
    except FileNotFoundError:
        return None
    except (DataError, OSError) as exc:
        raise RevisionApplicationError(
            f"Could not inspect proposed revision target {path}: {exc}; "
            "nothing was sent."
        ) from exc


def _plan_identity(
    *,
    deck_relative: str,
    deck_sha256: str,
    selected: Sequence[str],
    owner_instruction: str,
    provider: str,
    model: str,
    canonical_context_fingerprint: str,
    request_fingerprint: str,
    staging_relative: str,
    staging_revision: str | None,
) -> str:
    return prompts.fingerprint(
        _canonical_json(
            {
                "version": 1,
                "deck_path": deck_relative,
                "deck_sha256": deck_sha256,
                "selected_record_ids": list(selected),
                "owner_instruction_sha256": prompts.fingerprint(owner_instruction),
                "provider": provider,
                "model": model,
                "canonical_context_fingerprint": canonical_context_fingerprint,
                "request_fingerprint": request_fingerprint,
                "staging_path": staging_relative,
                "staging_revision": staging_revision,
            }
        )
    )


def plan_revision(
    config: ProjectConfig,
    deck: Path | str,
    selected_record_ids: Sequence[str],
    owner_instruction: str,
) -> RevisionPlan:
    """Read and fingerprint one exact paid revision without writing anything."""
    if not isinstance(owner_instruction, str) or not owner_instruction.strip():
        raise RevisionApplicationError(
            "Describe the exact change to make; nothing was sent."
        )
    selected = _selected_ids(selected_record_ids)
    chosen_provider = str(config.revise_provider).strip().lower()
    chosen_model = str(config.revise_model).strip()
    if not chosen_model:
        raise RevisionApplicationError(
            "Revision model must be nonblank; nothing was sent."
        )

    deck_path = _resolve_deck(config, deck)
    try:
        deck_content = pattern_cards.read_drill_deck_content(deck_path)
    except (DataError, OSError) as exc:
        raise RevisionApplicationError(
            f"Could not safely read rich drill deck {deck_path}: {exc}; "
            "nothing was sent."
        ) from exc
    if deck_content.revision.text is None:  # guarded by the public reader
        raise RevisionApplicationError(
            f"Revision deck does not exist: {deck_path}; nothing was sent."
        )
    selected_set = set(selected)
    ordered_selection = tuple(
        record_id
        for record_id in deck_content.record_ids
        if record_id in selected_set
    )
    extra = [
        record_id for record_id in selected if record_id not in deck_content.record_ids
    ]
    if extra or ordered_selection != selected:
        detail = (
            f"not deck members: {', '.join(extra)}"
            if extra
            else "the selection is not in deck.include_ids order"
        )
        raise RevisionApplicationError(
            f"Selected record ids must be an exact ordered subset of "
            f"{deck_path.name} ({detail}); nothing was sent."
        )
    deck_wire = deck_content.revision.text.encode("utf-8")
    deck_sha256 = _fingerprint_bytes(deck_wire)
    form = deck_content.form
    form_note = deck_content.form_note
    current = {
        record_id: tuple(
            {
                field: str(getattr(example, field))
                for field in _CURRENT_EXAMPLE_FIELDS
            }
            for example in deck_content.drill_examples[record_id]
        )
        for record_id in selected
    }
    context = _canonical_context(config, selected, form)

    style_guide = claude_client.read_style_guide(config.root)
    task_template = prompts.load(config.root, "revise-conjugation-deck")
    schema = ai_schema.conjugation_deck_revision_schema()
    blocks = tuple(claude_client.system_blocks(style_guide, task_template))
    user_turn = _user_turn(
        deck_name=deck_path.name,
        form=form,
        form_note=form_note,
        selected=selected,
        instruction=owner_instruction,
        current=current,
        context=context,
    )
    provider_plan = revision_provider.plan_provider(
        chosen_provider,
        model=chosen_model,
        style_guide=style_guide,
        task_template=task_template,
        system_blocks=blocks,
        user_turn=user_turn,
        schema=schema,
        effort=claude_client.effort_for(chosen_model),
    )
    request_fingerprint = provider_plan.request_fingerprint
    deck_relative = deck_path.relative_to(config.root.absolute()).as_posix()
    target_name = (
        f"revise-{prompts.fingerprint(deck_relative)[:12]}-"
        f"{request_fingerprint[:16]}.json"
    )
    staging_path = config.staging_dir / target_name
    staging_revision = _staging_revision(staging_path)
    canonical_context_fingerprint = prompts.fingerprint(_canonical_json(context))
    staging_relative = staging_path.relative_to(config.root.absolute()).as_posix()
    plan_fingerprint = _plan_identity(
        deck_relative=deck_relative,
        deck_sha256=deck_sha256,
        selected=selected,
        owner_instruction=owner_instruction,
        provider=provider_plan.provider,
        model=chosen_model,
        canonical_context_fingerprint=canonical_context_fingerprint,
        request_fingerprint=request_fingerprint,
        staging_relative=staging_relative,
        staging_revision=staging_revision,
    )
    return RevisionPlan(
        repository_root=config.root.resolve(),
        deck_path=deck_path,
        deck_relative_path=deck_relative,
        deck_sha256=deck_sha256,
        selected_record_ids=selected,
        owner_instruction=owner_instruction,
        provider_plan=provider_plan,
        style_guide=style_guide,
        task_template=task_template,
        user_turn=user_turn,
        canonical_context_fingerprint=canonical_context_fingerprint,
        staging_path=staging_path,
        staging_revision=staging_revision,
        plan_fingerprint=plan_fingerprint,
        form=form,
        current_form_note=form_note,
        current_examples=current,
        canonical_context=context,
    )


def _fresh_plan(config: ProjectConfig, expected: RevisionPlan) -> RevisionPlan:
    if expected.repository_root != config.root.resolve():
        raise RevisionApplicationError(
            "[revision-config-mismatch] this revision plan belongs to a different "
            "repository. Nothing was sent."
        )
    fresh = plan_revision(
        config,
        expected.deck_path,
        expected.selected_record_ids,
        expected.owner_instruction,
    )
    if fresh.request_fingerprint != expected.request_fingerprint:
        raise RevisionApplicationError(
            "[revision-request-stale] the selected deck content, canonical record "
            "context, prompt, model, provider, or owner instruction changed. "
            "Nothing was sent; reload the plan."
        )
    if fresh.plan_fingerprint != expected.plan_fingerprint:
        raise RevisionApplicationError(
            "[revision-plan-stale] the deck bytes or proposal staging state changed. "
            "Nothing was sent; reload the plan."
        )
    return fresh


def _proposal(parsed: Any, selected: tuple[str, ...]) -> dict[str, Any]:
    cards = list(getattr(parsed, "cards", []) or [])
    identifiers = tuple(str(getattr(card, "record_id", "")) for card in cards)
    if identifiers != selected:
        raise RevisionApplicationError(
            "The paid revision did not return every selected record exactly once "
            "in the rendered order. Its exact reply was captured, but no proposal "
            "was staged."
        )
    by_id: dict[str, list[dict[str, str]]] = {}
    for card in cards:
        examples: list[dict[str, str]] = []
        for example in list(getattr(card, "examples", []) or []):
            examples.append(
                {
                    "japanese": str(getattr(example, "japanese", "")),
                    "furigana": str(getattr(example, "furigana", "")),
                    "english": str(getattr(example, "english", "")),
                    "register": str(getattr(example, "speech_level", "")),
                }
            )
        by_id[str(card.record_id)] = examples
    return {
        "form_note": str(getattr(parsed, "form_note", "")),
        "drill_examples": by_id,
    }


def _request_manifest(plan: RevisionPlan, *, operation_id: str) -> dict[str, Any]:
    """The exact paid request and base state durable before dispatch."""
    return {
        "schema_version": 2,
        "kind": "conjugation_deck_revision",
        "state": "request",
        "operation_id": operation_id,
        "target": {
            "deck_path": plan.deck_relative_path,
            "deck_sha256": plan.deck_sha256,
            "form": plan.form,
            "selected_record_ids": list(plan.selected_record_ids),
            "staging_path": plan.staging_path.relative_to(
                plan.repository_root
            ).as_posix(),
        },
        "request": {
            "provider_plan": plan.provider_plan.persistent_manifest(),
            "owner_instruction": plan.owner_instruction,
            "style_guide": plan.style_guide,
            "task_template": plan.task_template,
            "system_blocks": _plain_json_value(plan.system_blocks),
            "user_turn": plan.user_turn,
            "plan_fingerprint": plan.plan_fingerprint,
            "canonical_context_fingerprint": plan.canonical_context_fingerprint,
        },
        "canonical_context": plan.canonical_context,
        "before": {
            "form_note": plan.current_form_note,
            "drill_examples": plan.current_examples,
        },
    }


def _manifest_text(value: Mapping[str, Any]) -> str:
    return _canonical_json(value, pretty=True) + "\n"


def _proposed_manifest(
    request_manifest: Mapping[str, Any], proposal: Mapping[str, Any]
) -> str:
    value = dict(request_manifest)
    value["state"] = "proposed"
    value["staged_at"] = date.today().isoformat()
    value["proposal"] = proposal
    return _manifest_text(value)


def _require_exact_keys(
    value: Any,
    keys: set[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        found = sorted(str(key) for key in value) if isinstance(value, Mapping) else []
        raise RevisionApplicationError(
            f"Revision request manifest has invalid {label} fields: {found}."
        )
    return value


def _request_manifest_for_recovery(
    config: ProjectConfig,
    operation_id: str,
) -> tuple[
    operations.OperationJournal,
    Any,
    Path,
    Mapping[str, Any],
    str,
    revision_provider.RevisionProviderPlan,
]:
    """Load and prove the exact request, journal identity, and captured reply."""
    journal = operations.OperationJournal.load(config.operations_file)
    held = journal.operations.get(operation_id)
    if held is None:
        raise RevisionApplicationError(
            f"No journaled revision operation {operation_id!r} to recover."
        )
    if held.kind != "revise":
        raise RevisionApplicationError(
            f"Operation {operation_id!r} is {held.kind!r}, not a revision."
        )
    if held.state != "result_captured" or held.artifact is None:
        raise RevisionApplicationError(
            f"Revision {operation_id!r} has no exact captured reply ready for "
            f"recovery; its state is {held.state!r}."
        )

    expected_name = (
        f"revise-{prompts.fingerprint(held.source_file)[:12]}-"
        f"{held.request_fp[:16]}.json"
    )
    manifest_path = config.staging_dir / expected_name
    try:
        manifest_wire = read_bytes_bound(manifest_path)
    except (FileNotFoundError, DataError, OSError) as exc:
        raise RevisionApplicationError(
            f"Could not read the durable request manifest for revision "
            f"{operation_id}: {exc}"
        ) from exc
    manifest_revision = _fingerprint_bytes(manifest_wire)
    manifest = _strict_json(manifest_wire, manifest_path)
    manifest_state = manifest.get("state")
    top_level = {
        "schema_version",
        "kind",
        "state",
        "operation_id",
        "target",
        "request",
        "canonical_context",
        "before",
    }
    if manifest_state == "proposed":
        top_level.update({"staged_at", "proposal"})
    _require_exact_keys(manifest, top_level, label="top-level")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("kind") != "conjugation_deck_revision"
        or manifest_state not in {"request", "proposed"}
        or manifest.get("operation_id") != operation_id
    ):
        raise RevisionApplicationError(
            f"Revision manifest {manifest_path} is not the recoverable result for "
            f"operation {operation_id}."
        )

    target = _require_exact_keys(
        manifest.get("target"),
        {
            "deck_path",
            "deck_sha256",
            "form",
            "selected_record_ids",
            "staging_path",
        },
        label="target",
    )
    request = _require_exact_keys(
        manifest.get("request"),
        {
            "provider_plan",
            "owner_instruction",
            "style_guide",
            "task_template",
            "system_blocks",
            "user_turn",
            "plan_fingerprint",
            "canonical_context_fingerprint",
        },
        label="request",
    )
    before = _require_exact_keys(
        manifest.get("before"),
        {"form_note", "drill_examples"},
        label="base content",
    )
    scalar_target = ("deck_path", "deck_sha256", "form", "staging_path")
    scalar_request = (
        "owner_instruction",
        "style_guide",
        "task_template",
        "user_turn",
        "plan_fingerprint",
        "canonical_context_fingerprint",
    )
    if any(not isinstance(target.get(key), str) for key in scalar_target) or any(
        not isinstance(request.get(key), str) for key in scalar_request
    ):
        raise RevisionApplicationError(
            "Revision request manifest contains a non-text identity field."
        )
    selected_raw = target.get("selected_record_ids")
    if not isinstance(selected_raw, list) or any(
        not isinstance(value, str) or not value.strip() for value in selected_raw
    ):
        raise RevisionApplicationError(
            "Revision request manifest has an invalid selected record list."
        )
    selected = tuple(selected_raw)
    if not selected or len(selected) != len(set(selected)):
        raise RevisionApplicationError(
            "Revision request manifest must bind a nonempty unique record selection."
        )
    context = manifest.get("canonical_context")
    examples = before.get("drill_examples")
    if (
        not isinstance(context, Mapping)
        or set(context) != set(selected)
        or not isinstance(examples, Mapping)
        or set(examples) != set(selected)
        or not isinstance(before.get("form_note"), str)
    ):
        raise RevisionApplicationError(
            "Revision request manifest does not bind its exact selected base content."
        )

    expected_staging = manifest_path.relative_to(config.root.absolute()).as_posix()
    if (
        target["deck_path"] != held.source_file
        or target["deck_sha256"] != held.source_sha256
        or target["staging_path"] != expected_staging
    ):
        raise RevisionApplicationError(
            "Revision request manifest does not match its journaled operation."
        )

    schema = ai_schema.conjugation_deck_revision_schema()
    expected_blocks = claude_client.system_blocks(
        request["style_guide"], request["task_template"]
    )
    if request["system_blocks"] != expected_blocks:
        raise RevisionApplicationError(
            "Revision request manifest has inconsistent system prompt blocks."
        )
    if request["canonical_context_fingerprint"] != prompts.fingerprint(
        _canonical_json(context)
    ):
        raise RevisionApplicationError(
            "Revision request manifest has an inconsistent content fingerprint."
        )
    rebuilt_turn = _user_turn(
        deck_name=Path(held.source_file).name,
        form=target["form"],
        form_note=before["form_note"],
        selected=selected,
        instruction=request["owner_instruction"],
        current={record_id: tuple(examples[record_id]) for record_id in selected},
        context={record_id: context[record_id] for record_id in selected},
    )
    if rebuilt_turn != request["user_turn"]:
        raise RevisionApplicationError(
            "Revision request manifest's exact owner instruction, selected base, "
            "and user turn do not agree."
        )
    provider_manifest = request["provider_plan"]
    if not isinstance(provider_manifest, Mapping):
        raise RevisionApplicationError(
            "Revision request manifest has invalid provider metadata."
        )
    provider_plan = revision_provider.provider_plan_from_manifest(
        provider_manifest,
        model=str(provider_manifest.get("model", "")),
        style_guide=request["style_guide"],
        task_template=request["task_template"],
        system_blocks=request["system_blocks"],
        user_turn=request["user_turn"],
        schema=schema,
    )
    if provider_plan.model != held.model or provider_plan.request_fingerprint != held.request_fp:
        raise RevisionApplicationError(
            "Revision provider metadata does not match its journaled operation."
        )
    rebuilt_plan = _plan_identity(
        deck_relative=target["deck_path"],
        deck_sha256=target["deck_sha256"],
        selected=selected,
        owner_instruction=request["owner_instruction"],
        provider=provider_plan.provider,
        model=provider_plan.model,
        canonical_context_fingerprint=request["canonical_context_fingerprint"],
        request_fingerprint=provider_plan.request_fingerprint,
        staging_relative=target["staging_path"],
        staging_revision=None,
    )
    if rebuilt_plan != request["plan_fingerprint"]:
        raise RevisionApplicationError(
            "Revision request manifest does not reproduce its authorized request."
        )

    reply = journal.read_reply(operation_id)
    if _fingerprint_bytes(reply) != held.artifact.content_sha256:
        raise RevisionApplicationError(
            f"Captured reply for revision {operation_id} does not match its journal receipt."
        )
    return journal, held, manifest_path, manifest, manifest_revision, provider_plan


def recover_revision(
    config: ProjectConfig,
    operation_id: str,
    *,
    progress: Callable[[str], None] | None = None,
) -> RevisionRunResult:
    """Parse one captured revise reply into its request manifest, without sending."""
    _report_progress(progress, "Reading the source")
    journal, held, manifest_path, manifest, manifest_revision, provider_plan = (
        _request_manifest_for_recovery(config, operation_id)
    )
    reply = journal.read_reply(operation_id)
    _report_progress(progress, "Checking the answer's shape")
    result = revision_provider.provider_for(provider_plan.provider).recover(
        provider_plan,
        reply,
    )
    parsed = getattr(result, "parsed", None)
    if parsed is None:
        stop_reason = str(getattr(result, "stop_reason", "") or "unknown")
        raise RevisionApplicationError(
            f"Captured reply for revision {operation_id} has no complete revision "
            f"({stop_reason})."
        )
    target = manifest["target"]
    selected = tuple(target["selected_record_ids"])
    proposal = _proposal(parsed, selected)
    if manifest["state"] == "proposed":
        if (
            not isinstance(manifest.get("staged_at"), str)
            or not manifest["staged_at"].strip()
            or manifest.get("proposal") != proposal
        ):
            raise RevisionApplicationError(
                f"Staged proposal for revision {operation_id} does not match its "
                "exact captured reply."
            )
        rendered = _manifest_text(manifest)
        if _fingerprint_bytes(rendered.encode("utf-8")) != manifest_revision:
            raise RevisionApplicationError(
                f"Staged proposal for revision {operation_id} is not its exact "
                "durable canonical manifest."
            )
    else:
        rendered = _proposed_manifest(manifest, proposal)
    _report_progress(progress, "Saving proposals")
    with exclusive_path_lock(manifest_path):
        journal.commit_result(
            operation_id,
            lambda: atomic_write_text_bound(
                manifest_path,
                rendered,
                expected_revision=manifest_revision,
            ),
        )
    return RevisionRunResult(
        operation_id=operation_id,
        staging_path=manifest_path,
        request_fingerprint=held.request_fp,
        plan_fingerprint=manifest["request"]["plan_fingerprint"],
    )


def run_revision(
    config: ProjectConfig,
    expected: RevisionPlan,
    *,
    client: Any | None = None,
    provider_env: Mapping[str, str] | None = None,
    provider_runner: Callable[..., Any] = subprocess.run,
    provider_which: Callable[..., str | None] = shutil.which,
    provider_spawn: revision_provider.Spawn = subprocess.Popen,
    api_call: Callable[..., Any] | None = None,
    progress: Callable[[str], None] | None = None,
) -> RevisionRunResult:
    """Consume one exact owner-confirmed plan and stage its paid answer."""
    _report_progress(progress, "Reading the source")
    fresh = _fresh_plan(config, expected)
    if not fresh.can_dispatch:
        raise RevisionApplicationError(
            f"A revision proposal already occupies {fresh.staging_path}. Review or "
            "archive it before authorizing another identical call; nothing was sent."
        )

    # Credential/login preparation can fail locally. It precedes authority so
    # such a failure never leaves an entry claiming a call may have been sent.
    provider = revision_provider.provider_for(fresh.provider)
    prepared_provider = provider.prepare(
        fresh.provider_plan,
        env=provider_env,
        runner=provider_runner,
        which=provider_which,
        client=client,
    )
    operation_id = str(uuid.uuid4())
    with contextlib.ExitStack() as binding_locks:
        for path in sorted(
            {
                expected.deck_path,
                config.normalized_file,
                expected.staging_path,
                config.root / prompts.DIRECTORY / "style-guide.md",
                config.root / prompts.DIRECTORY / "revise-conjugation-deck.md",
            },
            key=lambda item: str(item.absolute()),
        ):
            binding_locks.enter_context(exclusive_path_lock(path))

        # Client construction is intentionally outside these locks, then every
        # mutable request/target input is re-read while its writers are excluded.
        # The locks stay held through authority and durable request publication.
        fresh = _fresh_plan(config, expected)
        if not fresh.can_dispatch:
            raise RevisionApplicationError(
                f"A revision proposal already occupies {fresh.staging_path}. Review or "
                "archive it before authorizing another identical call; nothing was sent."
            )
        operations.prepare_artifact_store(config.operations_file)
        prepare_bound_directory(fresh.staging_path.parent)
        journal = operations.OperationJournal.load(config.operations_file)
        try:
            journal.authorize(
                operation_id,
                kind="revise",
                source_file=fresh.deck_relative_path,
                source_sha256=fresh.deck_sha256,
                request_fp=fresh.request_fingerprint,
                model=fresh.model,
            )
        except Exception as exc:  # noqa: BLE001 - journal authority may have landed
            raise RevisionRunError(
                f"Revision authority for {fresh.deck_relative_path} could not be "
                f"recorded safely: {exc} Inspect janki operations before retrying.",
                operation_id=operation_id,
                provider_dispatched=False,
            ) from exc

        request_manifest = _request_manifest(fresh, operation_id=operation_id)
        request_text = _manifest_text(request_manifest)
        request_revision = prompts.fingerprint(request_text)
        try:
            atomic_write_text_bound(
                fresh.staging_path,
                request_text,
                expected_absent=True,
            )
        except Exception as exc:  # noqa: BLE001 - publication may have landed
            raise operations.cancel_before_send(
                config.operations_file,
                operation_id,
                error=RevisionRunError,
                label="Revision",
                detail=(
                    "The durable revision request manifest could not be prepared."
                ),
                cause=exc,
            ) from exc

    try:
        journal.advance(operation_id, "dispatching")
    except Exception as exc:  # noqa: BLE001 - durable transition may have landed
        try:
            current = operations.OperationJournal.load(config.operations_file)
            held = current.operations.get(operation_id)
            if held is not None and held.state == "authorized":
                current.advance(
                    operation_id,
                    "canceled_before_send",
                    detail="The dispatch boundary could not be recorded.",
                )
        except JankiError:
            pass
        raise RevisionRunError(
            f"Revision {operation_id} was not dispatched because its dispatch "
            f"boundary could not be recorded safely: {exc}. Inspect janki "
            "operations before retrying.",
            operation_id=operation_id,
            provider_dispatched=False,
        ) from exc

    try:
        def capture_and_check(raw_reply: bytes) -> None:
            if not isinstance(raw_reply, bytes):
                raise operations.OperationError(
                    "Revision provider capture must supply exact response bytes."
                )
            journal.capture_result(
                operation_id,
                lambda: operations.capture_artifact(
                    config.operations_file,
                    operation_id,
                    raw_reply,
                ),
            )
            _report_progress(progress, "Checking the answer's shape")

        result = provider.dispatch(
            prepared_provider,
            capture=capture_and_check,
            spawn=provider_spawn,
            api_call=api_call,
        )
        captured = operations.OperationJournal.load(
            config.operations_file
        ).operations.get(operation_id)
        if captured is None or captured.state != "result_captured":
            raise operations.OperationError(
                f"Revision {operation_id} returned data without durably capturing "
                "the exact provider reply; no parsed value will be staged."
            )
        parsed = getattr(result, "parsed", None)
        if parsed is None:
            stop_reason = str(getattr(result, "stop_reason", "") or "unknown")
            refusal = getattr(result, "refusal", None)
            detail = ""
            if refusal is not None:
                category = str(getattr(refusal, "category", "") or "").strip()
                explanation = str(
                    getattr(refusal, "explanation", "") or ""
                ).strip()
                detail = ": " + " — ".join(
                    item for item in (category, explanation) if item
                )
            raise RevisionApplicationError(
                f"{fresh.model} returned no complete revision "
                f"({stop_reason}{detail}). Its exact reply was captured, but no "
                "proposal was staged."
            )
        proposal = _proposal(parsed, fresh.selected_record_ids)
        rendered = _proposed_manifest(request_manifest, proposal)
        _report_progress(progress, "Saving proposals")
        with exclusive_path_lock(fresh.staging_path):
            journal.commit_result(
                operation_id,
                lambda: atomic_write_text_bound(
                    fresh.staging_path,
                    rendered,
                    expected_revision=request_revision,
                ),
            )
    except Exception as exc:  # noqa: BLE001 - every post-dispatch failure is settled
        try:
            classify_dispatch_failure(config, journal, operation_id, exc)
        except JankiError as journal_error:
            raise RevisionRunError(
                f"Revision {operation_id} failed after dispatch: {exc} Janki could "
                f"not settle its journal entry: {journal_error}. This call may "
                "have been billed; do not retry until you inspect janki operations.",
                operation_id=operation_id,
                provider_dispatched=True,
            ) from exc
        raise RevisionRunError(
            f"Revision {operation_id} failed after dispatch: {exc}",
            operation_id=operation_id,
            provider_dispatched=True,
        ) from exc

    return RevisionRunResult(
        operation_id=operation_id,
        staging_path=fresh.staging_path,
        request_fingerprint=fresh.request_fingerprint,
        plan_fingerprint=fresh.plan_fingerprint,
    )
