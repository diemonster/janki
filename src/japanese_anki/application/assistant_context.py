"""Bounded, typed repository reads for an Assistant provider.

The provider never names a path.  It receives a catalog of opaque resource
identifiers and can ask this broker for one exact JSON snapshot, or perform a
literal substring search over the card values janki already parsed.  Only
application-owned readers open repository files; arbitrary filesystem reads
are deliberately not part of this interface.

Every returned value is the exact canonical JSON wire whose SHA-256 is carried
beside it.  Limits apply both to one result and cumulatively to one broker
instance (one Assistant turn).  A refusal consumes no budget.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any, Literal

from japanese_anki import kanji, ledger, operations, patterns, staging, status
from japanese_anki.application import promotion, revision_apply
from japanese_anki.application.detail import source_detail
from japanese_anki.application.journey import (
    SourceJourney,
    source_journeys,
    source_names,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.exporters import kanji_cards, pattern_cards
from japanese_anki.exporters.anki import (
    deck_kind,
    resolve_card_types,
    resolve_deck_records,
)
from japanese_anki.io import (
    DataError,
    load_records,
    load_structured,
    read_bytes_bound,
    records_revision,
)
from japanese_anki.models import ExampleSentence, VocabularyRecord
from japanese_anki.staging import LiveStaging, live_staging

__all__ = [
    "AssistantContextBroker",
    "AssistantContextError",
    "assistant_record_value",
    "ContextDisclosure",
    "ContextLimitError",
    "ContextLimits",
    "DeckContext",
    "ProposalContext",
]


ResourceKind = Literal[
    "deck",
    "card",
    "source",
    "status",
    "operations",
    "patterns",
    "kanji",
    "proposal",
    "artifacts",
]

_SCHEMA_VERSION = 1
_RESOURCE_ID_PREFIX = "resource_"
_RESOURCE_ID_HEX_LENGTH = 32
_DEFAULT_SEARCH_LIMIT = 20


class AssistantContextError(JankiError):
    """A repository context request could not be answered safely."""


class ContextLimitError(AssistantContextError):
    """A disclosure would exceed its result or turn budget."""


@dataclass(frozen=True, slots=True)
class ContextLimits:
    """Hard bounds for one result and one Assistant turn."""

    max_result_items: int = 1_024
    max_result_bytes: int = 512 * 1_024
    max_turn_items: int = 4_096
    max_turn_bytes: int = 2 * 1_024 * 1_024

    def __post_init__(self) -> None:
        for name in (
            "max_result_items",
            "max_result_bytes",
            "max_turn_items",
            "max_turn_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise AssistantContextError(f"Assistant context {name} must be a positive integer.")
        if self.max_turn_items < self.max_result_items:
            raise AssistantContextError(
                "Assistant context max_turn_items cannot be smaller than max_result_items."
            )
        if self.max_turn_bytes < self.max_result_bytes:
            raise AssistantContextError(
                "Assistant context max_turn_bytes cannot be smaller than max_result_bytes."
            )


@dataclass(frozen=True, slots=True)
class ContextDisclosure:
    """One exact provider-facing JSON value and its accounting."""

    resource_id: str
    kind: str
    wire: str
    sha256: str
    item_count: int
    utf8_bytes: int


@dataclass(frozen=True, slots=True)
class DeckContext:
    """A deck disclosure plus the safe editable card values behind its wire.

    ``records`` contain the exact learner-facing fields serialized into the
    disclosure.  Importer-private ``source.raw_fields`` and media paths are not
    retained, so handing these objects to an agent-context builder cannot
    widen what the broker authorized.
    """

    resource_id: str
    deck_kind: str
    configuration: Mapping[str, Any]
    records: tuple[VocabularyRecord, ...]
    disclosure: ContextDisclosure


@dataclass(frozen=True, slots=True)
class ProposalContext:
    """One freshly resolved live proposal behind an opaque catalog id.

    This is a local application boundary, not provider context.  The path is
    minted from the current staging census and has passed the same containment,
    no-symlink, regular-file, and proposal-kind checks used by snapshots.  The
    byte fingerprint comes from a bound no-follow read at that same boundary;
    an action must compare it with the bytes its ordinary service actually
    planned before rendering or writing anything.
    """

    resource_id: str
    proposal_kind: str
    path: Path
    proposal_sha256: str = ""


@dataclass(frozen=True, slots=True)
class _Resource:
    resource_id: str
    kind: ResourceKind
    title: str
    key: str
    deck_kind: str = ""
    available: bool = True
    subtype: str = ""


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _contains_literal(value: object, literal: str) -> bool:
    """Whether one parsed card value contains the exact string in a leaf."""

    if isinstance(value, str):
        return literal in value
    if isinstance(value, Mapping):
        return any(_contains_literal(item, literal) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        return any(_contains_literal(item, literal) for item in value)
    return False


def _opaque_id(kind: ResourceKind, key: str) -> str:
    digest = hashlib.sha256(
        f"janki-assistant-context-v{_SCHEMA_VERSION}\0{kind}\0{key}".encode()
    ).hexdigest()
    return _RESOURCE_ID_PREFIX + digest[:_RESOURCE_ID_HEX_LENGTH]


def _safe_name(value: str) -> str:
    """One inert final name, accepting either platform's separator syntax."""

    name = PurePath(value.replace("\\", "/")).name if value else ""
    return "" if name in {".", ".."} else name


def _display_record(record: VocabularyRecord) -> dict[str, Any]:
    """Card content without importer-private raw fields or source paths."""

    value = record.to_dict()
    value["audio"] = _safe_name(record.audio)
    value["image"] = _safe_name(record.image)
    value["examples"] = [_display_example(example) for example in record.examples]
    source = record.source
    value["source"] = {
        "type": source.type,
        "name": _safe_name(source.imported_from),
        "row": source.row,
    }
    return value


def assistant_record_value(record: VocabularyRecord) -> dict[str, Any]:
    """Return the safe learner-facing card value used by Assistant surfaces."""

    return _display_record(record)


def _context_record(record: VocabularyRecord) -> VocabularyRecord:
    """The editable card object corresponding exactly to ``_display_record``."""

    value = _display_record(record)
    value["source"] = {
        "type": record.source.type,
        "imported_from": (_safe_name(record.source.imported_from)),
        "row": record.source.row,
    }
    return VocabularyRecord.from_dict(value)


def _display_example(example: ExampleSentence) -> dict[str, str]:
    return {
        "japanese": example.japanese,
        "furigana": example.furigana,
        "romaji": example.romaji,
        "english": example.english,
        "audio": _safe_name(example.audio),
        "spoken_japanese": example.spoken_japanese,
        "register": example.register,
    }


_TEACHING_CONFIGURATION = (
    "name",
    "description",
    "kind",
    "form",
    "cards",
    "max_meanings",
    "include_ids",
    "exclude_ids",
    "include_tags",
    "exclude_tags",
    "intake_tag",
    "form_note",
    "document",
)


def _teaching_configuration(section: Mapping[str, Any], kind: str) -> dict[str, Any]:
    """The structural deck choices useful for explaining what it teaches.

    This whitelist is also the secrets boundary.  Unknown YAML keys, output
    paths, and collection paths are not part of an Assistant disclosure.
    """

    value = {
        key: section[key] for key in _TEACHING_CONFIGURATION if key in section and key != "kind"
    }
    document = value.get("document")
    if isinstance(document, str):
        value["document"] = _safe_name(document)
    value["kind"] = kind or "vocabulary"
    return value


def _pattern_card(card: pattern_cards.PatternCard) -> dict[str, Any]:
    return {
        "trigger": card.trigger,
        "result": card.result,
        "gloss": card.gloss,
        "examples": list(card.examples),
    }


def _display_pattern_set(entry: patterns.PatternSet, *, source_name: str) -> dict[str, Any]:
    value = entry.to_dict()
    value["source_name"] = _safe_name(source_name)
    # Prompt provenance is extensible. Only request-identity scalars cross the
    # boundary, so a future prompt body or local path cannot hitch a ride.
    provenance = value.get("prompt_provenance")
    if isinstance(provenance, Mapping):
        value["prompt_provenance"] = {
            str(key): item
            for key, item in provenance.items()
            if isinstance(item, str | int | bool)
            and (
                "fingerprint" in str(key)
                or str(key)
                in {
                    "mode",
                    "provider",
                    "model",
                    "response_schema_version",
                    "source_sha256",
                }
            )
        }
    return value


def _journey_value(journey: SourceJourney) -> dict[str, Any]:
    return {
        "name": _safe_name(journey.source),
        "state": journey.state,
        "next_action": journey.next_action,
        "grammar": journey.grammar,
        "card_count": journey.card_count,
        "held_count": journey.held_count,
        "example_review_count": journey.example_review_count,
        "invalid_count": journey.invalid_count,
        "deck_decision_count": journey.deck_decision_count,
        "has_completed_receipt": bool(journey.finish_receipt_ids),
    }


class AssistantContextBroker:
    """Resolve opaque project resources within one Assistant-turn budget."""

    def __init__(
        self,
        config: ProjectConfig,
        *,
        limits: ContextLimits | None = None,
    ) -> None:
        self.config = config
        self.limits = limits or ContextLimits()
        self.used_items = 0
        self.used_utf8_bytes = 0
        self._budget_lock = threading.Lock()
        self._resources: dict[str, _Resource] = {}
        self._root = config.root.absolute()
        # The one parse of the review queue this broker gets. Source names,
        # proposal kinds and the staged counts in the project status are three
        # projections of the same half-megabyte of YAML: reading it once costs
        # a third as much, and it is also the only way the catalog and the
        # status it discloses cannot describe two different directories.
        try:
            self._live_staging: tuple[LiveStaging, ...] | None = tuple(
                live_staging(config)
            )
        except (JankiError, OSError, UnicodeError, ValueError):
            # Unknown, not empty: every projection re-reads and reports for itself.
            self._live_staging = None
        try:
            self._preflight_project()
            self._discover_resources()
        except AssistantContextError:
            raise
        except (JankiError, OSError, UnicodeError, ValueError) as exc:
            raise AssistantContextError(
                f"Could not build the Assistant project catalog: {exc}"
            ) from exc

    # -- public provider boundary -----------------------------------------

    def catalog(self) -> ContextDisclosure:
        """Return every addressable project resource, without filesystem paths."""

        resources = [
            {
                "resource_id": resource.resource_id,
                "kind": resource.kind,
                "title": resource.title,
                **(
                    {
                        "deck_kind": resource.deck_kind or "unknown",
                        "available": resource.available,
                    }
                    if resource.kind == "deck"
                    else {}
                ),
                **({"proposal_kind": resource.subtype} if resource.kind == "proposal" else {}),
            }
            for resource in sorted(
                self._resources.values(),
                key=lambda item: (item.kind, item.title, item.resource_id),
            )
        ]
        return self._disclose(
            resource_id=_opaque_id("status", "project-catalog"),
            kind="catalog",
            data={"project": self.config.name, "resources": resources},
            item_count=len(resources),
        )

    def snapshot(self, resource_id: str) -> ContextDisclosure:
        """Read one catalog resource; an arbitrary string is never a path."""

        resource = self._resources.get(resource_id)
        if resource is None:
            raise AssistantContextError(
                f"Unknown Assistant resource {resource_id!r}; request a fresh catalog."
            )
        try:
            if resource.kind == "deck":
                return self.deck_context(resource_id).disclosure
            elif resource.kind == "card":
                data, count = self._card_snapshot(resource)
            elif resource.kind == "source":
                data, count = self._source_snapshot(resource)
            elif resource.kind == "status":
                data, count = self._status_snapshot()
            elif resource.kind == "operations":
                data, count = self._operations_snapshot()
            elif resource.kind == "patterns":
                data, count = self._patterns_snapshot()
            elif resource.kind == "kanji":
                data, count = self._kanji_snapshot()
            elif resource.kind == "proposal":
                data, count = self._proposal_snapshot(resource)
            else:
                data, count = self._artifacts_snapshot()
        except AssistantContextError:
            raise
        except (JankiError, OSError, UnicodeError, ValueError) as exc:
            raise AssistantContextError(
                f"Could not read Assistant resource {resource_id}: {exc}"
            ) from exc
        return self._disclose(
            resource_id=resource.resource_id,
            kind=resource.kind,
            data=data,
            item_count=count,
        )

    def resource_id_for_deck(self, deck: Path | str) -> str:
        """Resolve a trusted local deck scope against the configured census.

        This is for the local adapter, not a provider tool.  An unmatched
        value is never opened, resolved, or interpreted as a filesystem path.
        Relative scopes must be canonical repository-relative names without
        ``.`` or ``..`` components.
        """

        candidate = Path(deck)
        if not candidate.is_absolute():
            pure = PurePath(str(deck))
            if not pure.parts or any(part in {".", ".."} for part in pure.parts):
                raise AssistantContextError(
                    "Assistant deck scope must be a canonical project-relative path."
                )
            candidate = self._root / candidate
        candidate = candidate.absolute()
        configured = {
            path.absolute(): path.absolute().relative_to(self._root).as_posix()
            for path in status.deck_files(self.config)
        }
        key = configured.get(candidate)
        if key is None:
            raise AssistantContextError(
                "Assistant deck scope is not one currently configured deck."
            )
        resource_id = _opaque_id("deck", key)
        if resource_id not in self._resources:
            raise AssistantContextError(
                "The configured deck set changed; request a fresh project catalog."
            )
        return resource_id

    def deck_context(self, resource_id: str) -> DeckContext:
        """Return one debited deck snapshot without requiring JSON re-parsing."""

        resource = self._resources.get(resource_id)
        if resource is None or resource.kind != "deck":
            raise AssistantContextError(
                f"Unknown Assistant deck resource {resource_id!r}; request a fresh catalog."
            )
        try:
            data, count, records, actual_kind = self._deck_snapshot(resource)
        except AssistantContextError:
            raise
        except (JankiError, OSError, UnicodeError, ValueError) as exc:
            raise AssistantContextError(
                f"Could not read Assistant deck resource {resource_id}: {exc}"
            ) from exc
        disclosure = self._disclose(
            resource_id=resource.resource_id,
            kind=resource.kind,
            data=data,
            item_count=count,
        )
        configuration = data.get("configuration")
        if not isinstance(configuration, Mapping):
            raise AssertionError("deck snapshots always carry configuration")
        return DeckContext(
            resource_id=resource_id,
            deck_kind=actual_kind,
            configuration=dict(configuration),
            records=records,
            disclosure=disclosure,
        )

    def project_status(self) -> ContextDisclosure:
        """Return the debited aggregate status resource for an unfocused turn."""

        resource_id = _opaque_id("status", "project-status")
        if resource_id not in self._resources:
            raise AssistantContextError("Assistant project status is unavailable.")
        return self.snapshot(resource_id)

    def proposal_context(self, resource_id: str) -> ProposalContext:
        """Resolve one current live proposal without accepting a path as input."""

        resource = self._resources.get(resource_id)
        if resource is None or resource.kind != "proposal":
            raise AssistantContextError(
                f"Unknown Assistant proposal resource {resource_id!r}; request a fresh catalog."
            )
        try:
            path = self._proposal_path(resource)
            proposal_sha256 = hashlib.sha256(read_bytes_bound(path)).hexdigest()
        except AssistantContextError:
            raise
        except (JankiError, OSError, UnicodeError, ValueError) as exc:
            raise AssistantContextError(
                f"Could not resolve Assistant proposal resource {resource_id}: {exc}"
            ) from exc
        return ProposalContext(
            resource_id=resource_id,
            proposal_kind=resource.subtype,
            path=path,
            proposal_sha256=proposal_sha256,
        )

    def source_path(self, resource_id: str) -> Path:
        """Resolve one exact current source journey to its preserved inbox file.

        A source name learned from staging metadata is not a path.  It must
        still be the inert final name of exactly one current journey, and the
        resulting file must be a direct, regular, no-symlink child of the
        configured source inbox.
        """

        resource = self._resources.get(resource_id)
        if resource is None or resource.kind != "source":
            raise AssistantContextError(
                f"Unknown Assistant source resource {resource_id!r}; request a fresh catalog."
            )
        if not resource.key or _safe_name(resource.key) != resource.key:
            raise AssistantContextError("The selected source is not a basename-shaped inbox entry.")
        try:
            journeys, _warnings = source_journeys(self.config, live=self._live_staging)
        except (JankiError, OSError, UnicodeError, ValueError) as exc:
            raise AssistantContextError(
                f"Could not refresh the selected source journey: {exc}"
            ) from exc
        matches = [journey for journey in journeys if journey.source == resource.key]
        if len(matches) != 1:
            raise AssistantContextError(
                "The selected source journey changed or became ambiguous; request a fresh catalog."
            )

        inbox = self.config.scan_inbox.absolute()
        self._guard_directory(inbox, "source inbox")
        path = (inbox / resource.key).absolute()
        if path.parent != inbox:
            raise AssistantContextError("The selected source does not remain a direct inbox child.")
        self._guard_regular_file(path, "source")
        return path

    def search_cards(
        self,
        literal: str,
        *,
        limit: int = _DEFAULT_SEARCH_LIMIT,
    ) -> ContextDisclosure:
        """Search parsed card JSON by exact, case-sensitive Unicode substring.

        This deliberately performs no tokenization, normalization, reading
        choice, or other Japanese interpretation.
        """

        if not isinstance(literal, str) or not literal:
            raise AssistantContextError("Card search needs a nonempty literal string.")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise AssistantContextError("Card search limit must be a positive integer.")
        if limit > self.limits.max_result_items:
            raise ContextLimitError(
                "Card search limit exceeds the Assistant context result item limit."
            )

        records = self._current_records()
        matches: list[tuple[VocabularyRecord, dict[str, Any]]] = []
        for record in records:
            displayed = _display_record(record)
            if _contains_literal(displayed, literal):
                matches.append((record, displayed))
        shown = matches[:limit]
        cards = [
            {
                "resource_id": _opaque_id("card", record.id),
                "card": displayed,
            }
            for record, displayed in shown
        ]
        return self._disclose(
            resource_id=_opaque_id("card", f"search\0{literal}\0{limit}"),
            kind="card_search",
            data={
                "literal": literal,
                "cards": cards,
                "total_matches": len(matches),
                "truncated": len(matches) > len(shown),
            },
            item_count=len(cards),
        )

    # -- resource discovery ----------------------------------------------

    def _register(
        self,
        kind: ResourceKind,
        title: str,
        key: str,
        *,
        deck_kind_value: str = "",
        available: bool = True,
        subtype: str = "",
    ) -> None:
        resource_id = _opaque_id(kind, key)
        resource = _Resource(
            resource_id=resource_id,
            kind=kind,
            title=title,
            key=key,
            deck_kind=deck_kind_value,
            available=available,
            subtype=subtype,
        )
        previous = self._resources.setdefault(resource_id, resource)
        if previous != resource:
            raise AssistantContextError(
                "Assistant resource identifier collision; no catalog was disclosed."
            )

    def _discover_resources(self) -> None:
        for path in status.deck_files(self.config):
            title = path.stem
            kind = ""
            available = False
            try:
                self._guard_regular_file(path, "deck")
                raw = load_structured(path)
                section = raw.get("deck") if isinstance(raw, Mapping) else None
                if not isinstance(section, Mapping):
                    raise DataError(f"The deck section must be a mapping: {path}")
                kind = deck_kind(path) or "vocabulary"
                title = str(section.get("name") or path.stem)
                self._guard_deck_source(path, section, kind)
                available = True
            except (AssistantContextError, JankiError, OSError, UnicodeError, ValueError):
                # An invalid deck remains visible as its opaque configured
                # resource.  It refuses when selected, but it cannot make an
                # unrelated valid deck or project catalog disappear.
                pass
            _target, relative_path = self._relative_allowed(path, "deck")
            relative = relative_path.as_posix()
            self._register(
                "deck",
                title,
                relative,
                deck_kind_value=kind,
                available=available,
            )

        try:
            records = self._current_records()
        except (AssistantContextError, JankiError, OSError, UnicodeError, ValueError):
            records = []
        for record in records:
            record_title = record.expression
            if record.reading and record.reading != record.expression:
                record_title = f"{record.expression}（{record.reading}）"
            self._register("card", record_title, record.id)

        try:
            names = source_names(self.config, live=self._live_staging)
        except (JankiError, OSError, UnicodeError, ValueError):
            names = ()
        for name in names:
            self._register("source", _safe_name(name), name)

        for resource in self._discover_proposals():
            self._register(
                "proposal",
                resource.title,
                resource.key,
                subtype=resource.subtype,
            )

        self._register("status", "Project status", "project-status")
        self._register("operations", "Model-call operations", "model-call-operations")
        self._register("patterns", "Grammar pattern store", "pattern-store")
        self._register("kanji", "Kanji reference store", "kanji-store")
        self._register("artifacts", "Media and packages", "artifact-inventory")

    def _discover_proposals(self) -> list[_Resource]:
        """Readable direct staging proposals; bad siblings are omitted alone."""

        try:
            self._guard_directory(self.config.staging_dir, "staging directory", missing_ok=True)
            with os.scandir(self.config.staging_dir) as scan:
                entries = sorted(scan, key=lambda entry: entry.name)
        except (AssistantContextError, FileNotFoundError, OSError):
            return []
        parsed = {read.path.name: read for read in self._live_staging or ()}
        found: list[_Resource] = []
        for entry in entries:
            path = Path(entry.path)
            try:
                if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                    continue
                subtype, title = self._readable_proposal_identity(
                    path, read=parsed.get(entry.name)
                )
            except (AssistantContextError, JankiError, OSError, UnicodeError, ValueError):
                continue
            if not subtype:
                continue
            found.append(
                _Resource(
                    resource_id=_opaque_id("proposal", entry.name),
                    kind="proposal",
                    title=title,
                    key=entry.name,
                    subtype=subtype,
                )
            )
        return found

    def _readable_proposal_identity(
        self, path: Path, *, read: LiveStaging | None = None
    ) -> tuple[str, str]:
        """Classify one staging entry, reusing this broker's parse of it.

        ``read`` is this file as `live_staging` already parsed it, or `None`
        for one it did not cover — a `revise-*.json` plan, or a name the
        directory grew since. An entry it *could not* read is not a proposal:
        the symlink and regular-file guards above still run first either way.
        """
        self._guard_regular_file(path, "staging proposal")
        if path.suffix.lower() in {".yaml", ".yml"}:
            if read is None:
                records, meta = staging.read_staging(path)
            elif read.records is None or read.meta is None:
                return "", ""
            else:
                records, meta = read.records, read.meta
            if staging.CARD_REVISION_KEY in meta:
                promotion.staged_card_revision(
                    meta,
                    records,
                    require_owner_review=False,
                )
                subtype = "card_revision"
            elif staging.AI_ENRICHMENT_KEY in meta:
                promotion.staged_ai_enrichment(meta, [record.id for record in records])
                subtype = "ai_enrichment"
            else:
                subtype = "source_extraction"
            source = meta.get("source_file")
            title = _safe_name(source) if isinstance(source, str) else path.name
            return subtype, title or path.name
        if path.suffix.lower() == ".json" and path.name.startswith("revise-"):
            plan = revision_apply.plan_revision_apply(self.config, path)
            return "deck_revision", f"Deck revision: {_safe_name(plan.deck_relative_path)}"
        return "", ""

    # -- concrete snapshots ----------------------------------------------

    def _deck_path(self, resource: _Resource) -> Path:
        # ``resource.key`` was derived internally from a configured path.  The
        # caller never supplies it, and a fresh deck census must still contain
        # that exact lexical file before it is opened again.
        configured = {
            path.absolute().relative_to(self._root).as_posix(): path.absolute()
            for path in status.deck_files(self.config)
        }
        path = configured.get(resource.key)
        if path is None:
            raise AssistantContextError(
                "The configured deck set changed; request a fresh project catalog."
            )
        self._guard_regular_file(path, "deck")
        return path

    def _deck_snapshot(
        self, resource: _Resource
    ) -> tuple[dict[str, Any], int, tuple[VocabularyRecord, ...], str]:
        path = self._deck_path(resource)
        before = records_revision(path)
        if before.text is None:
            raise AssistantContextError("The selected deck no longer exists.")
        kind = deck_kind(path) or "vocabulary"
        raw = load_structured(path)
        section = raw.get("deck") if isinstance(raw, Mapping) else None
        if not isinstance(section, Mapping):
            raise DataError(f"The deck section must be a mapping: {path}")
        self._guard_deck_source(path, section, kind)

        if kind == "pattern":
            data, count = self._pattern_deck_snapshot(path, section, kind)
            context_records: tuple[VocabularyRecord, ...] = ()
        elif kind == "kanji":
            data, count = self._kanji_deck_snapshot(path, section, kind)
            context_records = ()
        elif kind == "conjugation":
            data, count, context_records = self._conjugation_deck_snapshot(path, before.text)
        else:
            resolved, records = resolve_deck_records(path)
            context_records = tuple(_context_record(record) for record in records)
            data = {
                "configuration": _teaching_configuration(resolved, kind),
                "enabled_card_types": resolve_card_types(resolved, self.config),
                "cards": [_display_record(record) for record in context_records],
            }
            count = max(1, len(records))

        after = records_revision(path)
        if after.text != before.text:
            raise AssistantContextError(
                "The selected deck changed while it was being read; request it again."
            )
        return data, count, context_records, kind

    def _pattern_deck_snapshot(
        self,
        path: Path,
        section: Mapping[str, Any],
        kind: str,
    ) -> tuple[dict[str, Any], int]:
        self._guard_optional_file(self.config.patterns_file, "pattern store")
        store = patterns.load_store(self.config.patterns_file)
        document = str(section.get("document") or "").strip()
        entry = store.get(document)
        if entry is None:
            raise AssistantContextError(
                f"The selected pattern deck names no readable document {document!r}."
            )
        cards = pattern_cards.cards_for(entry)
        return (
            {
                "configuration": _teaching_configuration(section, kind),
                "pattern_set": {
                    "source_name": _safe_name(entry.source),
                    "kind": entry.kind,
                    "title": entry.title,
                    "reviewed": entry.reviewed,
                    "patterns": [pattern.to_dict() for pattern in entry.patterns],
                },
                "cards": [_pattern_card(card) for card in cards],
            },
            max(1, len(cards)),
        )

    def _kanji_deck_snapshot(
        self,
        path: Path,
        section: Mapping[str, Any],
        kind: str,
    ) -> tuple[dict[str, Any], int]:
        """Disclose the curated character notes this deck already ships.

        Character notes are their own content type: they are keyed on the
        character rather than a vocabulary identity, so this snapshot reads the
        curated store instead of the canonical collection. Nothing here is a
        vocabulary record, and no word identity is implied by one.

        Through the exporter's own resolver, which is what a build reads: the
        deck's configured store, its named identities, and its enabled
        directions. Filtering the project's default store by the deck's ids
        instead answers a different question the moment a deck names another
        store, and disclosing notes a build never ships is exactly what this
        method exists not to do. ``_guard_deck_source`` has already proved that
        store's path before anything opens it.
        """

        deck = kanji_cards.resolve_kanji_deck_notes(path, self.config)
        characters = [
            {
                "record_id": note.id,
                "character": note.character,
                "meanings": list(note.meanings),
            }
            for note in deck.notes
        ]
        return (
            {
                "configuration": _teaching_configuration(section, kind),
                "character_notes": characters,
            },
            max(1, len(characters)),
        )

    def _conjugation_deck_snapshot(
        self,
        path: Path,
        deck_text: str,
    ) -> tuple[dict[str, Any], int, tuple[VocabularyRecord, ...]]:
        section = pattern_cards.conjugation_deck_section(path, deck_text)
        collection = pattern_cards.collection_for_section(path, self.config, section)
        self._guard_regular_file(collection, "conjugation source")
        records = load_records(collection)
        shipping = pattern_cards.shipping_records_for_section(path, section, records)
        context_records = tuple(_context_record(record) for record in shipping)
        form = str(section.get("form") or "te_form").strip()
        drills = pattern_cards.drill_cards(shipping, form)
        data: dict[str, Any] = {
            "configuration": _teaching_configuration(section, "conjugation"),
            "cards": [_display_record(record) for record in context_records],
            "drills": [
                {"record_id": record_id, "card": _pattern_card(card)} for card, record_id in drills
            ],
        }
        if section.get("drill_examples") is not None:
            content = pattern_cards.read_drill_deck_content(path)
            data["drill_examples"] = {
                record_id: [_display_example(example) for example in examples]
                for record_id, examples in content.drill_examples.items()
            }
        return data, max(1, len(drills)), context_records

    def _card_snapshot(self, resource: _Resource) -> tuple[dict[str, Any], int]:
        record = next(
            (record for record in self._current_records() if record.id == resource.key),
            None,
        )
        if record is None:
            raise AssistantContextError(
                "The selected card changed or disappeared; request a fresh catalog."
            )
        return {"card": _display_record(record)}, 1

    def _source_snapshot(self, resource: _Resource) -> tuple[dict[str, Any], int]:
        journeys, _warnings = source_journeys(self.config, live=self._live_staging)
        journey = next((item for item in journeys if item.source == resource.key), None)
        if journey is None:
            raise AssistantContextError(
                "The selected source changed or disappeared; request a fresh catalog."
            )
        value: dict[str, Any] = {"source": _journey_value(journey)}
        count = 1
        if journey.staging_path is not None:
            self._guard_regular_file(journey.staging_path, "staging source projection")
            detail = source_detail(
                self.config, journey.source, live=self._live_staging
            )
            if detail is not None:
                value["proposed_cards"] = [
                    {
                        "card": _display_record(card.record),
                        "authority": card.authority,
                        "hold_reason": card.hold_reason,
                    }
                    for card in detail.cards
                ]
                count += len(detail.cards)
                if detail.pattern_set is not None:
                    value["patterns"] = [
                        pattern.to_dict() for pattern in detail.pattern_set.patterns
                    ]
                    count += len(detail.pattern_set.patterns)
        return value, count

    def _status_snapshot(self) -> tuple[dict[str, Any], int]:
        universe = self._current_universe()
        if self.config.ledger_file.exists() or self.config.ledger_file.is_symlink():
            self._guard_regular_file(self.config.ledger_file, "ledger")
            ledger_wire: bytes | None = read_bytes_bound(self.config.ledger_file)
        else:
            self._guard_missing_parent(self.config.ledger_file, "ledger")
            ledger_wire = None
        book = ledger.load_snapshot(self.config.ledger_file, ledger_wire)
        staged, _warnings = status.collect_staged(
            self.config, parsed=self._live_staging
        )
        report = status.build_report(
            self.config,
            universe,
            book,
            word_provider=None,
            example_provider=None,
            staged=staged,
        )
        data = {
            "records": {
                "total": report.total,
                "normalized": report.normalized_count,
                "inline": report.inline_count,
                "by_source_type": dict(report.by_source),
                "ids": list(report.record_ids),
            },
            "decks": [
                {
                    "stem": deck.stem,
                    "cards": deck.total,
                    "never_exported": len(deck.unexported_ids),
                }
                for deck in report.decks
            ],
            "quality": {
                "missing_word_audio": list(report.missing_audio),
                "example_sentences": report.example_count,
                "missing_example_audio": [
                    {"record_id": record_id, "example_index": index}
                    for record_id, index in report.unvoiced_examples
                ],
                "stale_audio": list(report.stale_audio),
                "missing_enrichment": list(report.missing_enrichment),
                "missing_pitch_accent": report.missing_pitch_accent,
                "provisional": dict(report.provisional),
            },
            "staging": {
                "files": len(report.staged),
                "cards": report.staged_count,
            },
            "paid_operations_blocking": len(report.blocking_operations),
        }
        return data, max(1, report.total + len(report.decks) + len(report.staged))

    def _operations_snapshot(self) -> tuple[dict[str, Any], int]:
        if self.config.operations_file.exists() or self.config.operations_file.is_symlink():
            self._guard_regular_file(self.config.operations_file, "operation journal")
        else:
            self._guard_missing_parent(self.config.operations_file, "operation journal")
        journal = operations.OperationJournal.load(self.config.operations_file)
        rows = [
            {
                "operation_id": operation.operation_id,
                "kind": operation.kind,
                "state": operation.state,
                "source_name": _safe_name(operation.source_file),
                "model": operation.model,
                "authorized_at": operation.authorized_at,
                "updated_at": operation.updated_at,
                "money_may_have_been_spent": operation.money_may_have_been_spent,
                "blocks_spending": operation.blocks_spending,
                "needs_a_person": operation.needs_a_person,
                "has_captured_reply": operation.artifact is not None,
                "has_response_spool": operation.response_spool is not None,
                "cleanup_pending": operation.cleanup is not None,
            }
            for operation in sorted(
                journal.operations.values(),
                key=lambda item: (item.authorized_at, item.operation_id),
            )
        ]
        return {"operations": rows}, max(1, len(rows))

    def _patterns_snapshot(self) -> tuple[dict[str, Any], int]:
        self._guard_optional_file(self.config.patterns_file, "pattern store")
        store = patterns.load_store(self.config.patterns_file)
        rows = []
        for source, entry in sorted(store.items()):
            rows.append(_display_pattern_set(entry, source_name=source))
        return {"pattern_sets": rows}, max(1, len(rows))

    def _kanji_snapshot(self) -> tuple[dict[str, Any], int]:
        self._guard_optional_file(self.config.kanji_file, "kanji store")
        store = kanji.load_store(self.config.kanji_file)
        rows = [store.entries[key].to_dict() for key in sorted(store.entries)]
        return {"entries": rows}, max(1, len(rows))

    def _proposal_path(self, resource: _Resource) -> Path:
        if not resource.key or _safe_name(resource.key) != resource.key:
            raise AssistantContextError("The staging proposal binding is invalid.")
        self._guard_directory(self.config.staging_dir, "staging directory", missing_ok=True)
        path = self.config.staging_dir / resource.key
        self._guard_regular_file(path, "staging proposal")
        # Deliberately *not* the catalog's parse: this is the revalidation that
        # refuses a proposal whose kind changed since the catalog named it.
        subtype, _title = self._readable_proposal_identity(path)
        if subtype != resource.subtype:
            raise AssistantContextError(
                "The staging proposal changed kind; request a fresh catalog."
            )
        return path

    def _proposal_snapshot(self, resource: _Resource) -> tuple[dict[str, Any], int]:
        path = self._proposal_path(resource)
        before = records_revision(path)
        if resource.subtype == "deck_revision":
            data, count = self._deck_revision_snapshot(path)
        else:
            records, meta = staging.read_staging(path)
            data = {
                "proposal_kind": resource.subtype,
                "metadata": self._proposal_metadata(meta, resource.subtype),
                "cards": [_display_record(record) for record in records],
            }
            count = max(1, len(records))
        if records_revision(path).text != before.text:
            raise AssistantContextError(
                "The staging proposal changed while it was being read; request it again."
            )
        return data, count

    def _proposal_metadata(self, meta: Mapping[str, Any], subtype: str) -> dict[str, Any]:
        value: dict[str, Any] = {"proposal_kind": subtype}
        source = meta.get("source_file")
        if isinstance(source, str):
            value["source_name"] = _safe_name(source)
        for key in ("extracted_at", "model", "provider", "review_run_id"):
            item = meta.get(key)
            if isinstance(item, str | int | bool) or item is None:
                value[key] = item

        coverage = meta.get("coverage")
        if isinstance(coverage, Mapping):
            safe_coverage_keys = {
                "version",
                "status",
                "blocking",
                "source_fingerprint",
                "model_reported_unit_count",
                "observed_unit_count",
                "prose_candidate_count",
                "prose_coverage",
                "parsed_candidate_count",
                "canonical_record_count",
                "unusable_candidate_count",
                "duplicate_candidate_count",
                "collision_group_count",
                "candidate_accounting_fingerprint",
                "coverage_block_fingerprint",
            }
            value["coverage"] = {
                str(key): item
                for key, item in coverage.items()
                if key in safe_coverage_keys
                and (isinstance(item, str | int | bool) or item is None)
            }

        pattern = meta.get("pattern_set")
        if isinstance(pattern, Mapping):
            source_name = str(value.get("source_name") or "staging proposal")
            parsed = patterns.PatternSet.from_dict(source_name, dict(pattern))
            value["pattern_set"] = _display_pattern_set(parsed, source_name=source_name)

        provenance_key = (
            staging.CARD_REVISION_KEY
            if subtype == "card_revision"
            else staging.AI_ENRICHMENT_KEY
            if subtype == "ai_enrichment"
            else ""
        )
        provenance = meta.get(provenance_key) if provenance_key else None
        if isinstance(provenance, Mapping):
            allowed = {
                "version",
                "operation_id",
                "request_fingerprint",
                "provider",
                "attribution_provider",
                "model",
                "focus_resource_id",
                "fields",
            }
            value[provenance_key] = {
                str(key): item for key, item in provenance.items() if key in allowed
            }
        return value

    def _deck_revision_snapshot(self, path: Path) -> tuple[dict[str, Any], int]:
        plan = revision_apply.plan_revision_apply(self.config, path)

        def examples(
            values: Mapping[str, Sequence[ExampleSentence]],
        ) -> dict[str, list[dict[str, str]]]:
            return {
                record_id: [_display_example(example) for example in listed]
                for record_id, listed in sorted(values.items())
            }

        current_examples = examples(plan.current_drill_examples)
        proposed_examples = examples(plan.drill_examples)
        count = max(
            1,
            len(plan.selected_record_ids) + sum(len(items) for items in proposed_examples.values()),
        )
        return (
            {
                "proposal_kind": "deck_revision",
                "state": plan.state,
                "target_name": _safe_name(plan.deck_relative_path),
                "selected_record_ids": list(plan.selected_record_ids),
                "current": {
                    "form_note": plan.current_form_note,
                    "drill_examples": current_examples,
                },
                "proposed": {
                    "form_note": plan.form_note,
                    "drill_examples": proposed_examples,
                },
                "provenance": {
                    "operation_id": plan.operation_id,
                    "request_fingerprint": plan.request_fingerprint,
                    "provider": plan.provider,
                    "billing_class": plan.billing_class,
                    "request_bytes_sha256": plan.request_bytes_sha256,
                    "proposal_sha256": plan.proposal_sha256,
                    "plan_fingerprint": plan.plan_fingerprint,
                },
            },
            count,
        )

    def _artifacts_snapshot(self) -> tuple[dict[str, Any], int]:
        media, media_count = self._media_projection()
        packages, package_count = self._package_projection()
        return (
            {"media": media, "packages": packages},
            max(1, media_count + package_count),
        )

    def _media_projection(self) -> tuple[dict[str, Any], int]:
        self._guard_directory(
            self.config.media_dir,
            "media directory",
            missing_ok=True,
            allow_media_metadata=True,
        )
        counts: dict[str, int] = {}
        total_bytes = 0
        total_files = 0
        skipped_unsafe = 0
        if not self.config.media_dir.exists():
            return {"files": 0, "bytes": 0, "by_extension": {}}, 0
        stack = [self.config.media_dir]
        while stack:
            directory = stack.pop()
            with os.scandir(directory) as scan:
                entries = sorted(scan, key=lambda entry: entry.name)
            for entry in entries:
                try:
                    if entry.is_symlink():
                        skipped_unsafe += 1
                    elif entry.is_dir(follow_symlinks=False):
                        if entry.name not in {".pending", ".git"}:
                            stack.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        details = entry.stat(follow_symlinks=False)
                        suffix = Path(entry.name).suffix.lower() or "[none]"
                        counts[suffix] = counts.get(suffix, 0) + 1
                        total_files += 1
                        total_bytes += details.st_size
                    else:
                        skipped_unsafe += 1
                except OSError:
                    skipped_unsafe += 1
        return (
            {
                "files": total_files,
                "bytes": total_bytes,
                "by_extension": dict(sorted(counts.items())),
                "skipped_unsafe_entries": skipped_unsafe,
            },
            total_files,
        )

    def _package_projection(self) -> tuple[list[dict[str, Any]], int]:
        self._guard_directory(self.config.dist_dir, "package directory", missing_ok=True)
        if not self.config.dist_dir.exists():
            return [], 0
        rows: list[dict[str, Any]] = []
        with os.scandir(self.config.dist_dir) as scan:
            entries = sorted(scan, key=lambda entry: entry.name)
        for entry in entries:
            try:
                if (
                    entry.is_symlink()
                    or not entry.is_file(follow_symlinks=False)
                    or Path(entry.name).suffix.lower() not in {".apkg", ".html"}
                ):
                    continue
                details = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            rows.append(
                {
                    "name": _safe_name(entry.name),
                    "kind": (
                        "anki_package" if Path(entry.name).suffix.lower() == ".apkg" else "preview"
                    ),
                    "bytes": details.st_size,
                }
            )
        return rows, len(rows)

    # -- budgets and filesystem boundary ---------------------------------

    def _disclose(
        self,
        *,
        resource_id: str,
        kind: str,
        data: Mapping[str, Any],
        item_count: int,
    ) -> ContextDisclosure:
        envelope = {
            "schema_version": _SCHEMA_VERSION,
            "resource_id": resource_id,
            "kind": kind,
            "data": data,
        }
        wire = _canonical_json(envelope)
        encoded = wire.encode("utf-8")
        byte_count = len(encoded)
        with self._budget_lock:
            if item_count > self.limits.max_result_items:
                raise ContextLimitError(
                    "Assistant context result item limit exceeded "
                    f"({item_count} > {self.limits.max_result_items})."
                )
            if byte_count > self.limits.max_result_bytes:
                raise ContextLimitError(
                    "Assistant context result UTF-8 byte limit exceeded "
                    f"({byte_count} > {self.limits.max_result_bytes})."
                )
            if self.used_items + item_count > self.limits.max_turn_items:
                raise ContextLimitError(
                    "Assistant context turn item limit exceeded "
                    f"({self.used_items + item_count} > {self.limits.max_turn_items})."
                )
            if self.used_utf8_bytes + byte_count > self.limits.max_turn_bytes:
                raise ContextLimitError(
                    "Assistant context turn UTF-8 byte limit exceeded "
                    f"({self.used_utf8_bytes + byte_count} > {self.limits.max_turn_bytes})."
                )
            self.used_items += item_count
            self.used_utf8_bytes += byte_count
        return ContextDisclosure(
            resource_id=resource_id,
            kind=kind,
            wire=wire,
            sha256=hashlib.sha256(encoded).hexdigest(),
            item_count=item_count,
            utf8_bytes=byte_count,
        )

    def _preflight_project(self) -> None:
        self._guard_directory(self._root, "project root")
        self._guard_directory(self.config.deck_dir, "deck directory", missing_ok=True)

    def _readable_current_decks(self) -> list[Path]:
        """Bind each readable deck and its declared records source.

        This runs again for fresh snapshots.  A deck edited after catalog
        creation must not turn a formerly safe status/source read into a path
        traversal merely because its opaque identifier is still in memory.
        A bad sibling remains an unavailable catalog object; it cannot hide
        records from the normalized store or another independently safe deck.
        """

        readable: list[Path] = []
        for path in status.deck_files(self.config):
            try:
                self._guard_regular_file(path, "deck")
                raw = load_structured(path)
                section = raw.get("deck") if isinstance(raw, Mapping) else None
                if not isinstance(section, Mapping):
                    raise DataError(f"The deck section must be a mapping: {path}")
                raw_kind = section.get("kind")
                kind = raw_kind.strip().lower() if isinstance(raw_kind, str) else ""
                self._guard_deck_source(path, section, kind)
            except (AssistantContextError, JankiError, OSError, UnicodeError, ValueError):
                continue
            readable.append(path)
        return readable

    def _current_universe(self) -> status.RecordUniverse:
        self._preflight_project()
        self._guard_optional_file(self.config.normalized_file, "normalized cards")
        return status.collect_records(
            self.config,
            deck_paths=self._readable_current_decks(),
        )

    def _current_records(self) -> list[VocabularyRecord]:
        return self._current_universe().records

    def _guard_deck_source(
        self,
        deck_path: Path,
        section: Mapping[str, Any],
        kind: str,
    ) -> None:
        source = section.get("source")
        if source is not None and not isinstance(source, str):
            raise AssistantContextError(
                f"Deck source must be text before it can be disclosed: {deck_path.name}"
            )
        if source:
            source_path = Path(source)
            if not source_path.is_absolute():
                parts = PurePath(source).parts
                seen_named_component = False
                for part in parts:
                    if part in {"", "."}:
                        continue
                    if part == "..":
                        if seen_named_component:
                            raise AssistantContextError(
                                "Refusing a deck source whose path traverses back "
                                f"through a named component: {deck_path.name}"
                            )
                        continue
                    seen_named_component = True
                target = Path(os.path.normpath(deck_path.parent / source_path))
            else:
                target = source_path
            self._guard_optional_file(target, "deck source")
        elif kind == "conjugation":
            self._guard_optional_file(self.config.normalized_file, "conjugation source")
        elif kind == "kanji":
            # A character deck naming no store reads the project's curated one,
            # which must pass the same containment and no-follow proof as a
            # store it spells out.
            self._guard_optional_file(
                self.config.kanji_notes_file, "character note store"
            )

    def _relative_allowed(
        self,
        path: Path,
        label: str,
        *,
        allow_media_metadata: bool = False,
    ) -> tuple[Path, Path]:
        raw = Path(path)
        if not raw.is_absolute():
            raise AssistantContextError(
                f"Refusing non-absolute internal Assistant {label} binding."
            )
        # Lexically collapse parent components before the containment check.
        # `Path.absolute()` deliberately leaves them in place, allowing a path
        # that *looks* rooted to walk above the root during the later lstat.
        # Deck source syntax is checked separately so an internal `link/..`
        # cannot use symlink resolution to disagree with this normalization.
        target = Path(os.path.normpath(raw))
        try:
            relative = target.relative_to(self._root)
        except ValueError as exc:
            raise AssistantContextError(
                f"Refusing {label} that escapes the project: {target}"
            ) from exc
        parts = relative.parts
        if ".git" in parts:
            raise AssistantContextError(f"Refusing {label} under .git.")
        if ".pending" in parts:
            raise AssistantContextError(f"Refusing {label} under data/.pending.")
        assistant = self.config.assistant_dir.absolute()
        media = self.config.media_dir.absolute()
        if target == assistant or target.is_relative_to(assistant):
            raise AssistantContextError(f"Refusing {label} from Assistant reply artifacts.")
        if not allow_media_metadata and (target == media or target.is_relative_to(media)):
            raise AssistantContextError(f"Refusing {label} from media bytes.")
        lowered = {part.casefold() for part in parts}
        if lowered & {".env", "credentials", "secrets"}:
            raise AssistantContextError(f"Refusing {label} from a secrets path.")
        return target, relative

    def _guard_directory(
        self,
        path: Path,
        label: str,
        *,
        missing_ok: bool = False,
        allow_media_metadata: bool = False,
    ) -> None:
        target, _relative = self._relative_allowed(
            path, label, allow_media_metadata=allow_media_metadata
        )
        if not target.exists() and not target.is_symlink():
            if missing_ok:
                self._guard_missing_parent(
                    target,
                    label,
                    allow_media_metadata=allow_media_metadata,
                )
                return
            raise AssistantContextError(f"Assistant {label} does not exist: {target}")
        self._walk_components(
            target,
            label,
            final_directory=True,
            allow_media_metadata=allow_media_metadata,
        )

    def _guard_regular_file(self, path: Path, label: str) -> None:
        target, _relative = self._relative_allowed(path, label)
        self._walk_components(target, label, final_directory=False)
        # The bound reader opens every directory and the final name with
        # O_NOFOLLOW, so this is a real no-follow proof rather than `resolve()`
        # followed by an ordinary open.
        read_bytes_bound(target)

    def _guard_optional_file(self, path: Path, label: str) -> None:
        target, _relative = self._relative_allowed(path, label)
        if target.exists() or target.is_symlink():
            self._guard_regular_file(target, label)
        else:
            self._guard_missing_parent(target, label)

    def _guard_missing_parent(
        self,
        path: Path,
        label: str,
        *,
        allow_media_metadata: bool = False,
    ) -> None:
        target, _relative = self._relative_allowed(
            path, label, allow_media_metadata=allow_media_metadata
        )
        current = target.parent
        while current != self._root and not current.exists() and not current.is_symlink():
            current = current.parent
        self._walk_components(
            current,
            label,
            final_directory=True,
            allow_media_metadata=allow_media_metadata,
        )

    def _walk_components(
        self,
        target: Path,
        label: str,
        *,
        final_directory: bool,
        allow_media_metadata: bool = False,
    ) -> None:
        _target, relative = self._relative_allowed(
            target, label, allow_media_metadata=allow_media_metadata
        )
        current = self._root
        components: Sequence[str] = relative.parts
        for index, component in enumerate(components):
            current = current / component
            try:
                details = os.lstat(current)
            except OSError as exc:
                raise AssistantContextError(
                    f"Could not inspect Assistant {label}: {current}: {exc.strerror or exc}"
                ) from exc
            if stat.S_ISLNK(details.st_mode):
                raise AssistantContextError(f"Refusing symlink in Assistant {label}: {current}")
            final = index == len(components) - 1
            if not final or final_directory:
                if not stat.S_ISDIR(details.st_mode):
                    raise AssistantContextError(
                        f"Assistant {label} path component is not a directory: {current}"
                    )
            elif not stat.S_ISREG(details.st_mode):
                raise AssistantContextError(f"Assistant {label} is not a regular file: {current}")
