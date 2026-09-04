"""The optional ChatKit sidecar is wired to one exact revision application plan."""

from __future__ import annotations

import builtins
import hashlib
import http.client
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from japanese_anki import operations
from japanese_anki.application import (
    ANSWER_EMPTY,
    ANSWER_SAVED,
    ANSWER_UNAVAILABLE,
    FORGOTTEN,
    OUTCOME_UNKNOWN,
    DispatchFailure,
    ExtractionCompletionError,
    ExtractionDispatchError,
    ExtractionDispatchExpectation,
    ExtractionRevision,
    revision,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.models import ExampleSentence, VocabularyRecord
from japanese_anki.workbench import assistant_adapter, assistant_http
from japanese_anki.workbench import server as workbench_server
from japanese_anki.workbench.assistant import (
    AssistantDeckChoice,
    RevisionConfirmation,
    RevisionFinishConfirmation,
    RevisionRefusal,
    SourceExtractionConfirmation,
    SourceExtractionExecution,
    SourceExtractionPlan,
)
from japanese_anki.workbench.assistant_http import create_assistant_sidecar


def _config(tmp_path: Path, *, enabled: bool = True) -> ProjectConfig:
    (tmp_path / "janki.toml").write_text(
        f"[assistant]\nenabled = {'true' if enabled else 'false'}\n",
        encoding="utf-8",
    )
    return ProjectConfig.load(tmp_path)


def _adapter(
    config: ProjectConfig,
    deck_path: Path | None = None,
    *,
    record_ids: tuple[str, ...] = ("word:one",),
    label: str = "Potential Practice",
) -> assistant_adapter.RevisionAssistantAdapter:
    if deck_path is None:
        return assistant_adapter.RevisionAssistantAdapter(
            config=config,
            deck_choices=(),
            _targets=(),
        )
    scope = deck_path.absolute().relative_to(config.root.absolute()).as_posix()
    choice = AssistantDeckChoice(
        deck_id="test-deck-id",
        label=label,
        scope=scope,
        chat_supported=True,
        revision_supported=True,
    )
    return assistant_adapter.RevisionAssistantAdapter(
        config=config,
        deck_choices=(choice,),
        _targets=(
            assistant_adapter._RevisionDeckTarget(
                choice=choice,
                path=deck_path.absolute(),
                record_ids=record_ids,
            ),
        ),
    )


def _scope(adapter: assistant_adapter.RevisionAssistantAdapter) -> str:
    assert len(adapter.deck_choices) == 1
    return adapter.deck_choices[0].scope


def _prepare_deck_revision(
    adapter: assistant_adapter.RevisionAssistantAdapter,
    *,
    deck_scope: str,
    instruction: str,
) -> Any:
    """Exercise the live typed deck-revision planner without a model turn."""

    target = next(
        target for target in adapter._targets if target.choice.scope == deck_scope
    )
    application_plan, rendered = adapter._plan_deck_revision(
        ProjectConfig.load(adapter.config.root),
        target,
        instruction=instruction,
    )
    adapter._agent_plans[rendered.request_fingerprint] = (
        assistant_adapter._PreparedAgentAction(
            kind="revise_deck",
            focus_scope=deck_scope,
            instruction=instruction,
            target=rendered.target,
            plan=application_plan,
        )
    )
    return rendered


def _revision_plan(
    *,
    plan_fingerprint: str = "rendered-plan",
    request_fingerprint: str = "narrow-provider-request-fingerprint",
    selected_record_ids: tuple[str, ...] = ("word:one",),
    provider: str = "claude-code",
    billing_display: str = "Claude Max subscription via Claude Code",
    auth_metadata: dict[str, Any] | None = None,
    transport: dict[str, Any] | None = None,
) -> Any:
    return SimpleNamespace(
        plan_fingerprint=plan_fingerprint,
        request_fingerprint=request_fingerprint,
        selected_record_ids=selected_record_ids,
        provider=provider,
        billing_display=billing_display,
        auth_metadata=(
            auth_metadata
            if auth_metadata is not None
            else {"auth_method": "claude.ai", "subscription_type": "max"}
        ),
        model="claude-opus-5",
        transport=(
            transport
            if transport is not None
            else {"kind": "claude-code-cli", "cli_version": "2.1.246"}
        ),
        provider_plan=SimpleNamespace(request_bytes=b"exact-provider-request"),
        deck_relative_path="data/decks/potential.yaml",
        owner_instruction="Add examples.",
    )


def _agent_context_value(*, focus_resource_id: str | None = None) -> Any:
    wire = json.dumps(
        {
            "schema_version": 1,
            "focus_resource_id": focus_resource_id,
            "resources": ["resource_deck", "resource_first", "resource_second"],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return assistant_adapter.assistant_agent.AgentContext(
        wire=wire,
        fingerprint=hashlib.sha256(wire.encode("utf-8")).hexdigest(),
        resource_ids=("resource_deck", "resource_first", "resource_second"),
        focus_resource_id=focus_resource_id,
    )


def _agent_result(
    *,
    answer: str = "Here is the exact project answer.",
    intents: tuple[Any, ...] = (),
) -> Any:
    return SimpleNamespace(
        answer=answer,
        action_intents=intents,
        operation_id="agent-operation",
        manifest_path=Path("data/assistant/agent-operation.json"),
        request_fingerprint="agent-request",
    )


def _agent_intent(
    *,
    kind: str = "revise_cards",
    resource_ids: tuple[str, ...] = ("resource_deck",),
    record_ids: tuple[str, ...] = (),
    instruction: str = "Improve these cards.",
    options: dict[str, Any] | None = None,
) -> Any:
    return assistant_adapter.assistant_agent.AgentActionIntent(
        kind=kind,
        resource_ids=resource_ids,
        record_ids=record_ids,
        instruction=instruction,
        options_json=json.dumps(
            options or {},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def _disclosure(
    *,
    resource_id: str,
    kind: str,
    data: dict[str, Any],
) -> Any:
    wire = json.dumps(
        {
            "schema_version": 1,
            "resource_id": resource_id,
            "kind": kind,
            "data": data,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return SimpleNamespace(
        resource_id=resource_id,
        kind=kind,
        wire=wire,
        sha256=hashlib.sha256(wire.encode("utf-8")).hexdigest(),
        item_count=1,
        utf8_bytes=len(wire.encode("utf-8")),
    )


class _FakeContextBroker:
    """Small exact broker double for adapter target-binding tests."""

    records = (
        VocabularyRecord(id="word:second", expression="二", reading="に"),
        VocabularyRecord(id="word:first", expression="一", reading="いち"),
    )

    def __init__(self, _config: ProjectConfig) -> None:
        pass

    def catalog(self) -> Any:
        return _disclosure(
            resource_id="resource_catalog",
            kind="catalog",
            data={
                "resources": [
                    {"resource_id": "resource_deck", "kind": "deck"},
                    {"resource_id": "resource_first", "kind": "card"},
                    {"resource_id": "resource_second", "kind": "card"},
                    {"resource_id": "resource_outside", "kind": "card"},
                    {"resource_id": "resource_status", "kind": "status"},
                ]
            },
        )

    def resource_id_for_deck(self, _deck: Path | str) -> str:
        return "resource_deck"

    def deck_context(self, resource_id: str) -> Any:
        if resource_id != "resource_deck":
            raise JankiError("unknown deck resource")
        return SimpleNamespace(
            resource_id=resource_id,
            records=self.records,
            configuration={"name": "Vocabulary"},
            deck_kind="vocabulary",
        )

    def snapshot(self, resource_id: str) -> Any:
        record_ids = {
            "resource_first": "word:first",
            "resource_second": "word:second",
            "resource_outside": "word:outside",
        }
        return _disclosure(
            resource_id=resource_id,
            kind="card",
            data={"card": {"id": record_ids[resource_id]}},
        )


class _FakeSourceBroker:
    source = Path("/tmp/janki-test-inbox/lesson.pdf")

    def __init__(self, _config: ProjectConfig) -> None:
        pass

    def source_path(self, resource_id: str) -> Path:
        if resource_id != "resource_source":
            raise JankiError("Unknown Assistant source resource")
        return self.source


class _FakeCardRevisionPlan:
    def __init__(
        self,
        *,
        selected_record_ids: tuple[str, ...] = ("word:second", "word:first"),
        plan_fingerprint: str = "card-plan",
    ) -> None:
        self.plan_fingerprint = plan_fingerprint
        self.deck_relative_path = "data/decks/vocabulary.yaml"
        self.selected_record_ids = selected_record_ids
        self.current_records = tuple(
            VocabularyRecord(
                id=record_id,
                expression=f"expression-{index}",
                reading=f"reading-{index}",
            )
            for index, record_id in enumerate(selected_record_ids, start=1)
        )
        self.provider = "claude-code"
        self.billing_display = "Claude Max subscription via Claude Code"
        self.model = "claude-opus-5"
        self.request_fingerprint = "card-provider-request"
        self.provider_plan = SimpleNamespace(
            request_bytes=b"exact-card-revision-request",
            auth_metadata={
                "auth_method": "claude.ai",
                "subscription_type": "max",
            },
            transport={"kind": "claude-code-cli", "cli_version": "2.1.246"},
        )


class _FakeAiEnrichmentPlan:
    def __init__(self) -> None:
        self.plan_fingerprint = "e" * 64
        self.provider = "anthropic"
        self.model = "claude-opus-5"
        self.billing_display = "Anthropic API billing"
        self.canonical_relative_path = "data/normalized/vocabulary.json"
        self.canonical_sha256 = "7" * 64
        self.patterns_sha256 = "8" * 64
        self.force_fields = ()
        self.calls = (
            SimpleNamespace(
                operation_id="579fd68e-f84b-44f8-a949-800fc0d93eb7",
                record_id="word:second",
                record=VocabularyRecord(
                    id="word:second", expression="二", reading="に"
                ),
                record_sha256="1" * 64,
                input_fingerprint="2" * 64,
                request_fingerprint="3" * 64,
                request_manifest_path=Path(
                    "/tmp/janki/staging/ai-enrichment-first.request.json"
                ),
                staging_path=Path("/tmp/janki/staging/ai-enrichment-first.yaml"),
                provider_plan=SimpleNamespace(
                    request_bytes=b"first exact enrichment request",
                    auth_metadata={
                        "auth_method": "environment-api-key",
                        "api_key_source": "ANTHROPIC_API_KEY",
                    },
                    transport={"kind": "anthropic-messages-api"},
                ),
            ),
            SimpleNamespace(
                operation_id="3cb5026d-824e-4209-89b4-fd9c66f74966",
                record_id="word:first",
                record=VocabularyRecord(
                    id="word:first", expression="一", reading="いち"
                ),
                record_sha256="4" * 64,
                input_fingerprint="5" * 64,
                request_fingerprint="6" * 64,
                request_manifest_path=Path(
                    "/tmp/janki/staging/ai-enrichment-second.request.json"
                ),
                staging_path=Path("/tmp/janki/staging/ai-enrichment-second.yaml"),
                provider_plan=SimpleNamespace(
                    request_bytes=b"second exact enrichment request",
                    auth_metadata={
                        "auth_method": "environment-api-key",
                        "api_key_source": "ANTHROPIC_API_KEY",
                    },
                    transport={"kind": "anthropic-messages-api"},
                ),
            ),
        )


class _FakeAssistantActionPlan:
    def __init__(
        self,
        kind: str,
        *,
        fingerprint: str | None = None,
        audio_force: bool = False,
        audio_prune: bool = False,
    ) -> None:
        self.kind = kind
        self.fingerprint = fingerprint or ("a" if kind == "generate_audio" else "b") * 64
        configured_file = (
            "data/decks/vocabulary.yaml"
            if kind == "generate_audio"
            else "data/decks/potential.yaml"
        )
        if kind == "generate_audio":
            self.projection = {
                "target": {"configured_file": configured_file},
                "options": {
                    "words": True,
                    "examples": True,
                    "force": audio_force,
                    "prune": audio_prune,
                    "clip_classes_source": "deck-default",
                },
                "counts": {
                    "words": {
                        "total": 2,
                        "current": 1,
                        "recoverable": 0,
                        "provider_required": 1,
                    },
                    "examples": {
                        "total": 4,
                        "current": 1,
                        "recoverable": 2,
                        "provider_required": 1,
                    },
                },
                "providers": [
                    {
                        "name": "voicevox",
                        "access": "local-network",
                        "clip_kind": "words",
                        "voice": 13,
                        "speed": 0.75,
                        "provider_required": 1,
                    },
                    {
                        "name": "openai-realtime",
                        "access": "paid-network",
                        "clip_kind": "examples",
                        "voice": "cedar",
                        "speed": 0.8,
                        "provider_required": 1,
                    },
                ],
                "clips": [
                    {
                        "record_id": "word:first",
                        "kind": "word",
                        "target": "audio/first.wav",
                        "request_input": {"text": "話す", "voice": 13},
                        "content_fingerprint": "1" * 64,
                        "state": "provider_required",
                        "provider": "voicevox",
                        "billing_class": "local-network",
                        "recovery_sha256": None,
                    },
                    {
                        "record_id": "word:second",
                        "kind": "example",
                        "target": "audio/second.wav",
                        "request_input": {"text": "話せます。", "voice": "cedar"},
                        "content_fingerprint": "2" * 64,
                        "state": "recoverable",
                        "provider": "openai-realtime",
                        "billing_class": "paid-network",
                        "recovery_sha256": "3" * 64,
                    },
                ],
                "writes": {
                    "media_directory": "data/media/audio",
                    "ledger": "data/ledger.json",
                },
                "force_replacements": (
                    [
                        {
                            "record_id": "word:first",
                            "kind": "word",
                            "target": "janki-first.wav",
                            "existing": {
                                "file": "data/media/audio/janki-first.wav",
                                "entry_type": "regular-file",
                                "identity": [1, 2, 3, 4],
                                "bytes": 3,
                                "sha256": "f" * 64,
                            },
                        }
                    ]
                    if audio_force
                    else []
                ),
                "cleanup": {
                    "enabled": audio_prune,
                    "scope": (
                        "repository-wide-unreferenced-janki-audio"
                        if audio_prune
                        else "none"
                    ),
                    "media_files_removed": (
                        [
                            {
                                "file": "data/media/audio/janki-old.wav",
                                "entry_type": "regular-file",
                                "identity": [5, 6, 7, 8],
                                "bytes": 7,
                                "sha256": "e" * 64,
                            }
                        ]
                        if audio_prune
                        else []
                    ),
                    "ledger_audio_files_forgotten": (
                        ["janki-old.wav"] if audio_prune else []
                    ),
                    "missing_record_pending_audio_removed": [],
                },
            }
        else:
            self.projection = {
                "target": {
                    "configured_file": configured_file,
                    "card_count": 16,
                },
                "inputs": {
                    "deck": {
                        "file": "data/decks/potential.yaml",
                        "sha256": "9" * 64,
                    },
                    "source": {
                        "label": "canonical cards",
                        "file": "data/normalized/vocabulary.json",
                        "sha256": "c" * 64,
                    },
                    "sources": [
                        {
                            "label": "configured deck",
                            "file": "data/decks/potential.yaml",
                            "sha256": "9" * 64,
                        },
                        {
                            "label": "canonical cards",
                            "file": "data/normalized/vocabulary.json",
                            "sha256": "c" * 64,
                        },
                    ],
                    "templates": [
                        {
                            "label": "recognition front",
                            "file": "templates/front.html",
                            "sha256": "d" * 64,
                        },
                        {
                            "label": "recognition back",
                            "file": "templates/back.html",
                            "sha256": "e" * 64,
                        },
                    ],
                    "media": [
                        {
                            "label": "example audio",
                            "file": "data/media/audio/example.wav",
                            "sha256": "f" * 64,
                        }
                    ],
                },
                "writes": {
                    "package": "dist/potential.apkg",
                    "package_precondition": {
                        "state": "replace_exact",
                        "sha256": "8" * 64,
                        "identity": [23, 47],
                    },
                    "export_history": "data/ledger.json",
                },
            }


class _FakeDeckCreationPlan:
    def __init__(self, *, fingerprint: str = "6" * 64) -> None:
        self.fingerprint = fingerprint
        self.request = SimpleNamespace(name="Weekly Review")
        self.projection = {
            "target": {
                "configured_file": "data/decks/weekly-review.yaml",
                "future_package": "dist/weekly-review.apkg",
                "intake_tag": "deck:weekly-review",
                "deck_id": 1_702_000_001,
            },
            "definition": {
                "sha256": "7" * 64,
                "yaml": (
                    "deck:\n"
                    "  name: Weekly Review\n"
                    "  cards:\n"
                    "    recognition: true\n"
                    "    production: false\n"
                    "    reading: true\n"
                ),
            },
            "inputs": {"deck_set_sha256": "8" * 64},
        }


class _FakeStagingReviewPlan:
    def __init__(self, *, fingerprint: str = "3" * 64) -> None:
        self.fingerprint = fingerprint
        self.source_name = "lesson.pdf"
        self.projection = {
            "selection": {
                "records": [
                    {
                        "record_id": "word:食べる:たべる",
                        "expression": "食べる",
                        "reading": "たべる",
                        "examples": [
                            {
                                "register": "polite",
                                "japanese": "朝ご飯が食べられます。",
                                "furigana": "朝ご飯[あさごはん]が 食[た]べられます。",
                                "english": "I can eat breakfast.",
                                "romaji": "asagohan ga taberaremasu",
                                "spoken_japanese": "",
                            }
                        ],
                    }
                ],
                "review_patterns": True,
                "patterns": {
                    "title": "Potential verbs",
                    "patterns": [{"template": "る → られる"}],
                },
            },
            "snapshots": {
                "staging_sha256": "1" * 64,
                "patterns_sha256": "2" * 64,
            },
            "writes": {
                "card_review": ["word:食べる:たべる"],
                "pattern_review": "lesson.pdf",
            },
        }


class _FakeCardRevisionReviewPlan:
    def __init__(self, *, fingerprint: str = "c" * 64) -> None:
        self.fingerprint = fingerprint
        self.proposal_sha256 = "d" * 64
        self.proposal_path = Path("/tmp/janki-review/card-revision.yaml")
        self.record_ids = ("word:話す:はなす",)
        self.projection = {
            "selection": {
                "record_ids": list(self.record_ids),
                "changes": [
                    {
                        "record_id": self.record_ids[0],
                        "expression": "話す",
                        "field": "usage_notes",
                        "old_value": "Old note",
                        "proposed_value": "Reviewed new note",
                    }
                ],
            },
            "writes": {
                "durable_owner_review": True,
                "unselected_rows_removed": ["word:聞く:きく"],
            },
        }


class _FakeCardRevisionFinishPlan:
    def __init__(
        self,
        root: Path,
        *,
        instruction: str,
        fingerprint: str = "e" * 64,
        review: _FakeCardRevisionReviewPlan | None = None,
        promotion: Any | None = None,
    ) -> None:
        self.fingerprint = fingerprint
        self.resource_id = "resource_revision"
        self.instruction = instruction
        self.review = review
        self.promotion = promotion
        self.deck_path = root / "data" / "decks" / "lesson.yaml"
        self.record_ids = ("word:話す:はなす",)
        local_provider = SimpleNamespace(
            name="voicevox",
            access="local",
            voice=0,
            speed=1.0,
        )
        paid_provider = SimpleNamespace(
            name="openai-realtime",
            access="paid-network",
            voice="cedar",
            speed=0.8,
        )
        self.audio = SimpleNamespace(
            force=False,
            word_counts=SimpleNamespace(
                total=1,
                current=1,
                recoverable=0,
                provider_required=0,
            ),
            example_counts=SimpleNamespace(
                total=1,
                current=0,
                recoverable=0,
                provider_required=1,
            ),
            clips=(
                SimpleNamespace(
                    kind="word",
                    record_id=self.record_ids[0],
                    target="janki-word.wav",
                    state="current",
                    provider=local_provider,
                    request_input="話す",
                ),
                SimpleNamespace(
                    kind="example",
                    record_id=self.record_ids[0],
                    target="janki-example.wav",
                    state="provider-required",
                    provider=paid_provider,
                    request_input="日本語で話せます。",
                ),
            ),
        )
        package = root / "dist" / "lesson.apkg"
        deck_input = SimpleNamespace(
            label="configured deck",
            path=self.deck_path,
            sha256="1" * 64,
        )
        canonical_input = SimpleNamespace(
            label="canonical cards",
            path=root / "data" / "normalized" / "vocabulary.json",
            sha256="2" * 64,
        )
        template_input = SimpleNamespace(
            label="vocabulary template",
            path=root / "templates" / "vocabulary.html",
            sha256="3" * 64,
        )
        media_input = SimpleNamespace(
            label="package media",
            path=root / "data" / "media" / "audio" / "janki-example.wav",
            sha256=None,
        )
        self.build = SimpleNamespace(
            deck_name="Lesson deck",
            output_path=package,
            card_count=2,
            note_count=1,
            variant="recognition-production",
            card_types=("recognition", "production"),
            configuration_fingerprint="4" * 64,
            output_revision=None,
            output_identity=None,
            deck_input=deck_input,
            source_inputs=(canonical_input,),
            template_inputs=(template_input,),
            media_inputs=(media_input,),
        )
        self.projected_canonical_text = '[{"id":"word:話す:はなす"}]\n'
        self.record_path = root / "data" / "staging" / "done" / "revisions" / (
            f"card-finish-{fingerprint}.json"
        )


class _FakeAssignmentPlan:
    def __init__(self, *, fingerprint: str = "9" * 64) -> None:
        self.fingerprint = fingerprint
        self.service_fingerprint = "a" * 64
        self.source_name = "lesson.pdf"
        self.destination_name = "Lesson deck"
        self.proposal_path = Path("/tmp/janki-assignment/staging/lesson.pdf.yaml")
        self.projection = {
            "proposal": {
                "resource_id": "resource_proposal",
                "kind": "source_extraction",
                "source_name": "lesson.pdf",
                "configured_file": "data/staging/lesson.pdf.yaml",
                "sha256": "b" * 64,
            },
            "destination": {
                "resource_id": "resource_destination",
                "name": "Lesson deck",
                "stem": "lesson",
                "intake_tag": "lesson-intake",
                "configured_file": "data/decks/lesson.yaml",
            },
            "selection": {
                "record_ids": ["word:食べる:たべる"],
                "assignments": [
                    {
                        "record_id": "word:食べる:たべる",
                        "expression": "食べる",
                        "reading": "たべる",
                        "existing_owner": None,
                        "assigned_record_sha256": "c" * 64,
                        "prospective_record_sha256": "d" * 64,
                        "tags": {
                            "before": ["personal"],
                            "after": ["personal", "lesson-intake"],
                            "removed": [],
                            "added": ["lesson-intake"],
                        },
                        "proposal_occurrences": [
                            {
                                "source_name": "lesson.pdf",
                                "row": 7,
                                "source_type": "extract",
                            }
                        ],
                        "resulting_memberships": [
                            {
                                "deck": "Lesson deck",
                                "stem": "lesson",
                                "takes": True,
                                "refusal": None,
                            }
                        ],
                    }
                ],
            },
            "writes": {
                "staging_proposal": "data/staging/lesson.pdf.yaml",
                "canonical_cards": False,
                "deck_definition": False,
            },
            "service_fingerprint": self.service_fingerprint,
        }


class _FakePromotionPlan:
    def __init__(self, *, fingerprint: str = "4" * 64) -> None:
        self.fingerprint = fingerprint
        self.source = "lesson.pdf"
        self.service_fingerprint = "5" * 64
        self.projection = {
            "target": {"proposal_sha256": "6" * 64},
            "decision": {
                "state": "lands",
                "landing": [
                    {
                        "record_id": "word:食べる:たべる",
                        "destination": "data/normalized/vocabulary.json",
                    }
                ],
                "held": [
                    {
                        "record_id": "word:未定:みてい",
                        "reason": "identity decision required",
                    }
                ],
            },
            "writes": {
                "collection": "data/normalized/vocabulary.json",
                "archive": "data/staging/done/lesson.pdf.yaml",
                "ledger": "data/ledger.json",
            },
        }


def _agent_extraction_plan() -> SourceExtractionPlan:
    return SourceExtractionPlan(
        preparation_id="source-preparation",
        source_name="lesson.pdf",
        request_fingerprint="5" * 64,
        target="data/staging/lesson.pdf.yaml",
        effects=(
            "Send the whole lesson.pdf source to claude-opus-5",
            "Use automatic source-shape selection",
            "Propose vocabulary cards and grammar for owner review",
        ),
        disclosures=(
            "This is one paid Anthropic API call.",
            "Only the named source leaves this computer.",
        ),
        confirm_label="Send lesson.pdf using claude-opus-5 — paid API call",
        replaces=False,
    )


def _finish_plan(
    tmp_path: Path,
    *,
    fingerprint: str = "f" * 64,
    staging_name: str = "proposal.json",
) -> Any:
    current = (
        ExampleSentence(
            japanese="今は遊べません。",
            furigana="今[いま]は 遊[あそ]べません。",
            english="I cannot play now.",
            register="polite",
        ),
        ExampleSentence(
            japanese="今日は遊べない。",
            furigana="今日[きょう]は 遊[あそ]べない。",
            english="I cannot play today.",
            register="casual",
        ),
    )
    proposed = (
        ExampleSentence(
            japanese="明日は遊べます。",
            furigana="明日[あした]は 遊[あそ]べます。",
            english="I can play tomorrow.",
            register="polite",
        ),
        ExampleSentence(
            japanese="今日は遊べる。",
            furigana="今日[きょう]は 遊[あそ]べる。",
            english="I can play today.",
            register="casual",
        ),
    )
    revision_plan = SimpleNamespace(
        staging_path=tmp_path / "data" / "staging" / staging_name,
        deck_relative_path="data/decks/potential.yaml",
        current_form_note="Old potential note.",
        form_note="Potential expresses ability or possibility.",
        selected_record_ids=("word:one",),
        current_drill_examples={"word:one": current},
        drill_examples={"word:one": proposed},
    )
    counts = SimpleNamespace(
        total=2,
        current=0,
        recoverable=1,
        provider_required=1,
    )
    return SimpleNamespace(
        revision=revision_plan,
        audio=SimpleNamespace(
            example_provider=SimpleNamespace(
                name="openai-realtime",
                access="paid-network",
                settings={"model": "gpt-realtime-1.5"},
            ),
            example_counts=counts,
        ),
        build=SimpleNamespace(
            output_path=tmp_path / "dist" / "potential.apkg",
            card_count=16,
        ),
        authority={"exact": fingerprint},
        fingerprint=fingerprint,
    )


def _seeded_extraction_confirmation(
    tmp_path: Path,
) -> tuple[
    assistant_adapter.RevisionAssistantAdapter,
    SourceExtractionConfirmation,
    ProjectConfig,
]:
    config = _config(tmp_path)
    source = config.scan_inbox / "lesson.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"%PDF")
    adapter = _adapter(config, config.deck_dir / "potential.yaml")
    preparation_id = "prepared-source"
    expected = ExtractionDispatchExpectation(
        source=source,
        model="claude-opus-5",
        mode=None,
        source_sha256="a" * 64,
        request_fingerprint="b" * 64,
        replacement_revision=None,
        replacement_confirmed=False,
        staging_path=(config.staging_dir / "lesson.pdf.yaml").resolve(),
        patterns_path=config.patterns_file.resolve(),
        operations_path=config.operations_file.resolve(),
    )
    with adapter._plan_lock:
        adapter._extraction_expectations[preparation_id] = expected
    return (
        adapter,
        SourceExtractionConfirmation(
            preparation_id=preparation_id,
            source_name="lesson.pdf",
            expected_fingerprint="b" * 64,
        ),
        config,
    )


def test_disabled_workbench_never_imports_the_assistant_modules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in {
            "japanese_anki.workbench.assistant_adapter",
            "japanese_anki.workbench.assistant_http",
        } or name.startswith("chatkit"):
            raise AssertionError(f"disabled workbench imported {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    assert workbench_server._start_assistant_for(_config(tmp_path, enabled=False)) is None


def test_enabled_workbench_gives_assistant_the_configured_local_inbox(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    choices = (
        AssistantDeckChoice(
            deck_id="opaque-deck",
            label="Potential Practice",
            scope="data/decks/potential.yaml",
            chat_supported=True,
            revision_supported=True,
        ),
    )
    adapter = SimpleNamespace(
        deck_choices=choices,
    )
    expected_sidecar = object()
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        assistant_adapter,
        "discover_revision_adapter",
        lambda _config: (adapter, []),
    )

    def start(callbacks: Any, **kwargs: Any) -> object:
        captured.update(callbacks=callbacks, **kwargs)
        return expected_sidecar

    monkeypatch.setattr(assistant_http, "start_assistant_sidecar", start)

    result = workbench_server._start_assistant_for(config)

    assert result is expected_sidecar
    assert captured == {
        "callbacks": adapter,
        "deck_choices": choices,
        "inbox_root": config.scan_inbox,
    }


def test_enabled_workbench_starts_project_intake_without_a_revision_deck(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    expected_sidecar = object()
    captured: dict[str, Any] = {}
    monkeypatch.setattr(assistant_adapter.status, "deck_files", lambda _config: [])

    def start(callbacks: Any, **kwargs: Any) -> object:
        captured.update(callbacks=callbacks, **kwargs)
        return expected_sidecar

    monkeypatch.setattr(assistant_http, "start_assistant_sidecar", start)

    result = workbench_server._start_assistant_for(config)

    assert result is expected_sidecar
    assert captured["callbacks"].deck_choices == ()
    assert captured["deck_choices"] == ()
    assert captured["inbox_root"] == config.scan_inbox


def test_discovery_lists_one_supported_deck_without_exposing_its_path_as_the_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deck = tmp_path / "data" / "decks" / "potential.yaml"
    monkeypatch.setattr(assistant_adapter.status, "deck_files", lambda _config: [deck])
    monkeypatch.setattr(
        assistant_adapter,
        "resolve_deck_records",
        lambda _path: (
            {
                "kind": "conjugation",
                "name": "Brandon Japanese::Potential Practice",
                "drill_examples": {"word:one": []},
            },
            [],
        ),
    )
    monkeypatch.setattr(
        assistant_adapter,
        "read_drill_deck_content",
        lambda _path: SimpleNamespace(
            record_ids=("word:one",),
            drill_examples={"word:one": object()},
        ),
    )

    monkeypatch.setattr(assistant_adapter.secrets, "token_urlsafe", lambda _size: "opaque-deck")

    adapter, warnings = assistant_adapter.discover_revision_adapter(_config(tmp_path))

    assert warnings == ()
    assert adapter.deck_choices == (
        AssistantDeckChoice(
            deck_id="opaque-deck",
            label="Brandon Japanese::Potential Practice",
            scope="data/decks/potential.yaml",
            chat_supported=True,
            revision_supported=True,
        ),
    )
    assert adapter.resolve_deck_selection("opaque-deck") == adapter.deck_choices[0]


def test_discovery_lists_two_supported_decks_without_guessing_an_active_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "data" / "decks" / "potential.yaml"
    second = tmp_path / "data" / "decks" / "te-form.yaml"
    monkeypatch.setattr(assistant_adapter.status, "deck_files", lambda _config: [first, second])
    monkeypatch.setattr(
        assistant_adapter,
        "resolve_deck_records",
        lambda _path: (
            {"kind": "conjugation", "drill_examples": {"record": []}},
            [],
        ),
    )
    monkeypatch.setattr(
        assistant_adapter,
        "read_drill_deck_content",
        lambda path: SimpleNamespace(
            record_ids=(f"word:{path.stem}",),
            drill_examples={f"word:{path.stem}": object()},
        ),
    )
    opaque_ids = iter(("opaque-potential", "opaque-te-form"))
    monkeypatch.setattr(
        assistant_adapter.secrets,
        "token_urlsafe",
        lambda _size: next(opaque_ids),
    )

    adapter, warnings = assistant_adapter.discover_revision_adapter(_config(tmp_path))

    assert warnings == ()
    assert [(choice.deck_id, choice.scope) for choice in adapter.deck_choices] == [
        ("opaque-potential", "data/decks/potential.yaml"),
        ("opaque-te-form", "data/decks/te-form.yaml"),
    ]
    assert all(choice.revision_supported for choice in adapter.deck_choices)


def test_discovery_keeps_every_configured_deck_visible_with_exact_support_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rich = tmp_path / "data" / "decks" / "potential.yaml"
    plain = tmp_path / "data" / "decks" / "vocabulary.yaml"
    incomplete = tmp_path / "data" / "decks" / "te-form.yaml"
    monkeypatch.setattr(
        assistant_adapter.status,
        "deck_files",
        lambda _config: [rich, plain, incomplete],
    )

    def resolve(path: Path) -> tuple[dict[str, Any], list[Any]]:
        if path == rich:
            return (
                {
                    "kind": "conjugation",
                    "name": "Potential Practice",
                    "drill_examples": {"word:one": []},
                },
                [],
            )
        if path == incomplete:
            return (
                {
                    "kind": "conjugation",
                    "name": "Te-form Practice",
                },
                [],
            )
        return ({"kind": "vocabulary", "name": "Vocabulary"}, [])

    monkeypatch.setattr(assistant_adapter, "resolve_deck_records", resolve)
    monkeypatch.setattr(
        assistant_adapter,
        "read_drill_deck_content",
        lambda _path: SimpleNamespace(
            record_ids=("word:one",),
            drill_examples={"word:one": object()},
        ),
    )
    opaque_ids = iter(("opaque-rich", "opaque-plain", "opaque-incomplete"))
    monkeypatch.setattr(
        assistant_adapter.secrets,
        "token_urlsafe",
        lambda _size: next(opaque_ids),
    )

    adapter, warnings = assistant_adapter.discover_revision_adapter(_config(tmp_path))

    assert warnings == ()
    assert [choice.label for choice in adapter.deck_choices] == [
        "Potential Practice",
        "Vocabulary",
        "Te-form Practice",
    ]
    assert [choice.revision_supported for choice in adapter.deck_choices] == [
        True,
        False,
        False,
    ]
    assert [choice.chat_supported for choice in adapter.deck_choices] == [
        True,
        True,
        True,
    ]
    assert adapter.deck_choices[0].unavailable_reason is None
    assert "rich conjugation" in (adapter.deck_choices[1].unavailable_reason or "")
    assert "drill examples" in (adapter.deck_choices[2].unavailable_reason or "")
    assert adapter.resolve_deck_selection("opaque-plain") == adapter.deck_choices[1]
    with pytest.raises(RevisionRefusal, match="unknown"):
        adapter.resolve_deck_selection("invented")
    with pytest.raises(RevisionRefusal, match="unknown"):
        adapter.resolve_deck_selection("未知")


def test_discovery_disambiguates_duplicate_visible_deck_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "data" / "decks" / "first.yaml"
    second = tmp_path / "data" / "decks" / "second.yaml"
    monkeypatch.setattr(
        assistant_adapter.status,
        "deck_files",
        lambda _config: [first, second],
    )
    monkeypatch.setattr(
        assistant_adapter,
        "resolve_deck_records",
        lambda _path: (
            {
                "kind": "conjugation",
                "name": "Shared Practice",
                "drill_examples": {"word:one": []},
            },
            [],
        ),
    )
    monkeypatch.setattr(
        assistant_adapter,
        "read_drill_deck_content",
        lambda path: SimpleNamespace(
            record_ids=(f"word:{path.stem}",),
            drill_examples={f"word:{path.stem}": object()},
        ),
    )
    opaque_ids = iter(("opaque-first", "opaque-second"))
    monkeypatch.setattr(
        assistant_adapter.secrets,
        "token_urlsafe",
        lambda _size: next(opaque_ids),
    )

    adapter, warnings = assistant_adapter.discover_revision_adapter(_config(tmp_path))

    assert warnings == ()
    assert [choice.label for choice in adapter.deck_choices] == [
        "Shared Practice (first.yaml)",
        "Shared Practice (second.yaml)",
    ]
    assert [target.choice for target in adapter._targets] == list(adapter.deck_choices)


def test_discovery_refuses_a_deck_symlink_that_resolves_outside_the_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    deck_dir = project_root / "data" / "decks"
    deck_dir.mkdir(parents=True)
    outside = tmp_path / "outside.yaml"
    outside.write_text("kind: conjugation\n", encoding="utf-8")
    linked = deck_dir / "linked.yaml"
    linked.symlink_to(outside)
    monkeypatch.setattr(
        assistant_adapter.status,
        "deck_files",
        lambda _config: [linked],
    )

    adapter, warnings = assistant_adapter.discover_revision_adapter(
        _config(project_root)
    )

    assert adapter.deck_choices == ()
    assert adapter._targets == ()
    assert len(warnings) == 1
    assert "outside the project root" in warnings[0]


def test_discovery_lists_only_one_name_for_two_paths_to_the_same_deck(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    deck_dir = project_root / "data" / "decks"
    deck_dir.mkdir(parents=True)
    real = deck_dir / "potential.yaml"
    real.write_text("kind: conjugation\n", encoding="utf-8")
    alias = deck_dir / "alias.yaml"
    alias.symlink_to(real)
    monkeypatch.setattr(
        assistant_adapter.status,
        "deck_files",
        lambda _config: [real, alias],
    )
    monkeypatch.setattr(
        assistant_adapter,
        "_inspect_deck",
        lambda _path: ("Potential Practice", ("word:one",), None, None),
    )
    opaque_ids = iter(("opaque-real", "opaque-alias"))
    monkeypatch.setattr(
        assistant_adapter.secrets,
        "token_urlsafe",
        lambda _size: next(opaque_ids),
    )

    adapter, warnings = assistant_adapter.discover_revision_adapter(
        _config(project_root)
    )

    assert [choice.deck_id for choice in adapter.deck_choices] == ["opaque-real"]
    assert [choice.scope for choice in adapter.deck_choices] == [
        "data/decks/potential.yaml"
    ]
    assert len(warnings) == 1
    assert "resolves to an already listed deck" in warnings[0]


def test_discovery_skips_a_deck_without_a_visible_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deck = tmp_path / "data" / "decks" / "blank.yaml"
    monkeypatch.setattr(
        assistant_adapter.status,
        "deck_files",
        lambda _config: [deck],
    )
    monkeypatch.setattr(
        assistant_adapter,
        "_inspect_deck",
        lambda _path: ("   ", ("word:one",), None, None),
    )

    adapter, warnings = assistant_adapter.discover_revision_adapter(_config(tmp_path))

    assert adapter.deck_choices == ()
    assert adapter._targets == ()
    assert len(warnings) == 1
    assert "has no display name" in warnings[0]


def test_selection_refuses_a_fresh_revision_reader_failure_as_capability_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deck = tmp_path / "data" / "decks" / "potential.yaml"
    monkeypatch.setattr(assistant_adapter.status, "deck_files", lambda _config: [deck])
    monkeypatch.setattr(
        assistant_adapter,
        "resolve_deck_records",
        lambda _path: (
            {
                "kind": "conjugation",
                "name": "Potential Practice",
                "drill_examples": {"word:one": []},
            },
            [],
        ),
    )
    inspections = 0

    def read(_path: Path) -> Any:
        nonlocal inspections
        inspections += 1
        if inspections == 1:
            return SimpleNamespace(
                record_ids=("word:one",),
                drill_examples={"word:one": object()},
            )
        raise JankiError("deck changed after the selector was rendered")

    monkeypatch.setattr(assistant_adapter, "read_drill_deck_content", read)
    monkeypatch.setattr(assistant_adapter.secrets, "token_urlsafe", lambda _size: "opaque-deck")
    monkeypatch.setattr(
        assistant_adapter.revision,
        "plan_revision",
        lambda *_args, **_kwargs: pytest.fail("selection must not plan a revision"),
    )
    adapter, warnings = assistant_adapter.discover_revision_adapter(_config(tmp_path))

    assert warnings == ()
    with pytest.raises(RevisionRefusal, match="Assistant capabilities changed"):
        adapter.resolve_deck_selection("opaque-deck")
    assert inspections == 2


def test_selection_refuses_when_deck_capabilities_change_after_catalog_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deck = tmp_path / "data" / "decks" / "potential.yaml"
    monkeypatch.setattr(assistant_adapter.status, "deck_files", lambda _config: [deck])
    deck_configs = iter(
        (
            {
                "kind": "conjugation",
                "name": "Potential Practice",
                "drill_examples": {"word:one": []},
            },
            {"kind": "vocabulary", "name": "Potential Practice"},
        )
    )
    monkeypatch.setattr(
        assistant_adapter,
        "resolve_deck_records",
        lambda _path: (next(deck_configs), []),
    )
    monkeypatch.setattr(
        assistant_adapter,
        "read_drill_deck_content",
        lambda _path: SimpleNamespace(
            record_ids=("word:one",),
            drill_examples={"word:one": object()},
        ),
    )
    monkeypatch.setattr(
        assistant_adapter.secrets,
        "token_urlsafe",
        lambda _size: "opaque-deck",
    )

    adapter, warnings = assistant_adapter.discover_revision_adapter(_config(tmp_path))

    assert warnings == ()
    assert adapter.deck_choices[0].revision_supported is True
    with pytest.raises(RevisionRefusal, match="capabilities changed"):
        adapter.resolve_deck_selection("opaque-deck")


def test_discovery_keeps_an_unreadable_deck_visible_but_chat_inert(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deck = tmp_path / "data" / "decks" / "broken.yaml"
    monkeypatch.setattr(assistant_adapter.status, "deck_files", lambda _config: [deck])
    monkeypatch.setattr(
        assistant_adapter,
        "resolve_deck_records",
        lambda _path: (_ for _ in ()).throw(JankiError("invalid deck document")),
    )
    monkeypatch.setattr(
        assistant_adapter.secrets,
        "token_urlsafe",
        lambda _size: "opaque-broken",
    )

    adapter, warnings = assistant_adapter.discover_revision_adapter(_config(tmp_path))

    [choice] = adapter.deck_choices
    assert choice.label == "broken"
    assert choice.chat_supported is False
    assert choice.revision_supported is False
    assert "could not be read safely" in (choice.unavailable_reason or "")
    assert len(warnings) == 1
    assert "could not safely inspect" in warnings[0]
    with pytest.raises(RevisionRefusal, match="could not be read safely"):
        adapter.resolve_deck_selection("opaque-broken")
    with pytest.raises(RevisionRefusal, match="Could not inspect Assistant deck"):
        adapter.chat(
            deck_scope=choice.scope,
            history=(),
            message="Can you read this deck?",
            progress=lambda _label: None,
            preview=lambda _delta: None,
        )


def test_revision_only_deck_error_remains_selectable_for_chat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deck = tmp_path / "data" / "decks" / "broken-drill.yaml"
    monkeypatch.setattr(assistant_adapter.status, "deck_files", lambda _config: [deck])
    monkeypatch.setattr(
        assistant_adapter,
        "resolve_deck_records",
        lambda _path: (
            {
                "kind": "conjugation",
                "name": "Broken Drill",
                "drill_examples": {"word:outside-scope": []},
            },
            [],
        ),
    )
    monkeypatch.setattr(
        assistant_adapter,
        "read_drill_deck_content",
        lambda _path: (_ for _ in ()).throw(
            JankiError("drill example is outside include_ids")
        ),
    )
    monkeypatch.setattr(
        assistant_adapter.secrets,
        "token_urlsafe",
        lambda _size: "opaque-broken-drill",
    )

    adapter, warnings = assistant_adapter.discover_revision_adapter(_config(tmp_path))

    [choice] = adapter.deck_choices
    assert warnings == ()
    assert choice.chat_supported is True
    assert choice.revision_supported is False
    assert "Deck changes cannot safely read" in (choice.unavailable_reason or "")
    assert adapter.resolve_deck_selection(choice.deck_id) == choice


def test_selection_refuses_a_deck_whose_full_revision_scope_changed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deck = tmp_path / "data" / "decks" / "potential.yaml"
    monkeypatch.setattr(assistant_adapter.status, "deck_files", lambda _config: [deck])
    monkeypatch.setattr(
        assistant_adapter,
        "resolve_deck_records",
        lambda _path: (
            {
                "kind": "conjugation",
                "name": "Potential Practice",
                "drill_examples": {"word:one": []},
            },
            [],
        ),
    )
    reads = iter(
        (
            SimpleNamespace(
                record_ids=("word:one",),
                drill_examples={"word:one": object()},
            ),
            SimpleNamespace(
                record_ids=("word:one", "word:two"),
                drill_examples={"word:one": object(), "word:two": object()},
            ),
        )
    )
    monkeypatch.setattr(
        assistant_adapter,
        "read_drill_deck_content",
        lambda _path: next(reads),
    )
    monkeypatch.setattr(assistant_adapter.secrets, "token_urlsafe", lambda _size: "opaque-deck")
    adapter, warnings = assistant_adapter.discover_revision_adapter(_config(tmp_path))

    assert warnings == ()
    with pytest.raises(RevisionRefusal, match="scope changed"):
        adapter.resolve_deck_selection("opaque-deck")


def test_chat_refuses_a_path_that_has_no_startup_choice(
    tmp_path: Path,
) -> None:
    adapter = _adapter(
        _config(tmp_path),
        tmp_path / "data" / "decks" / "potential.yaml",
    )
    with pytest.raises(RevisionRefusal, match="not one currently configured deck"):
        adapter.chat(
            deck_scope="data/decks/invented.yaml",
            history=(),
            message="Which deck?",
            progress=lambda _label: None,
            preview=lambda _delta: None,
        )


def test_adapter_plans_the_complete_stored_order_and_displays_application_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deck = tmp_path / "data" / "decks" / "potential.yaml"
    selected = ("word:second", "word:first")
    adapter = _adapter(_config(tmp_path), deck, record_ids=selected)
    calls: list[tuple[Path, tuple[str, ...], str]] = []
    application_plan = _revision_plan(
        plan_fingerprint="full-application-plan-fingerprint",
        request_fingerprint="narrow-provider-request-fingerprint",
        selected_record_ids=selected,
    )

    def fake_plan(
        _config: Any,
        target: Path,
        record_ids: tuple[str, ...],
        instruction: str,
    ) -> Any:
        calls.append((target, tuple(record_ids), instruction))
        return application_plan

    monkeypatch.setattr(assistant_adapter.revision, "plan_revision", fake_plan)

    rendered = _prepare_deck_revision(
        adapter,
        deck_scope="data/decks/potential.yaml",
        instruction="Add examples.",
    )

    assert calls == [(deck, selected, "Add examples.")]
    assert rendered.request_fingerprint == "full-application-plan-fingerprint"
    assert "word:second, word:first" in rendered.effects[0]
    wire = "\n".join((*rendered.effects, *rendered.disclosures))
    assert "claude-code" in wire
    assert "Billing: Claude Max subscription via Claude Code" in wire
    assert "Authentication: auth method=claude.ai, subscription type=max" in wire
    assert "Model: claude-opus-5" in wire
    assert "Claude Code version: 2.1.246" in wire
    assert (
        "Exact provider request bytes: 22 bytes; request bytes SHA-256 "
        f"{hashlib.sha256(b'exact-provider-request').hexdigest()}"
    ) in wire
    assert ("Provider request identity: narrow-provider-request-fingerprint") in wire
    assert "Anthropic" not in wire
    assert "API" not in wire


@pytest.mark.parametrize("focused", [False, True])
def test_adapter_routes_unfocused_and_focused_turns_through_exact_agent_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    focused: bool,
) -> None:
    deck = tmp_path / "data" / "decks" / "potential.yaml"
    adapter = _adapter(_config(tmp_path), deck)
    deck_scope = _scope(adapter) if focused else ""
    contexts: list[str] = []
    planned = object()
    observed: list[tuple[Any, str, tuple[tuple[str, str], ...]]] = []

    def context_for(_config: ProjectConfig, *, deck_scope: str) -> Any:
        contexts.append(deck_scope)
        return _agent_context_value(
            focus_resource_id="resource_deck" if deck_scope else None
        )

    def plan_agent(
        _config: ProjectConfig,
        *,
        context: Any,
        message: str,
        history: tuple[tuple[str, str], ...],
    ) -> Any:
        observed.append((context, message, history))
        return planned

    def run_agent(
        _config: ProjectConfig,
        received: Any,
        *,
        progress: Any,
        preview: Any,
    ) -> Any:
        assert received is planned
        assert received is not None
        progress("Writing answer")
        return _agent_result(answer="This is the exact repository answer.")

    monkeypatch.setattr(assistant_adapter, "_agent_context", context_for)
    monkeypatch.setattr(assistant_adapter.assistant_agent, "plan_agent", plan_agent)
    monkeypatch.setattr(
        assistant_adapter.assistant_agent,
        "run_agent",
        run_agent,
    )
    monkeypatch.setattr(
        assistant_adapter.revision,
        "plan_revision",
        lambda *_args, **_kwargs: pytest.fail(
            "ordinary agent prose must not become a legacy revision"
        ),
    )

    progress: list[str] = []
    reply = adapter.chat(
        deck_scope=deck_scope,
        history=(("user", "earlier"), ("assistant", "Earlier answer.")),
        message="What is in my library?",
        progress=progress.append,
        preview=lambda _delta: None,
    )

    # One build per message: the turn dispatches the exact context it was
    # planned and fingerprinted against, so there is no second read to differ.
    assert contexts == [deck_scope]
    assert len(observed) == 1
    context, message, history = observed[0]
    assert context.focus_resource_id == ("resource_deck" if focused else None)
    assert message == "What is in my library?"
    assert history == (("user", "earlier"), ("assistant", "Earlier answer."))
    assert progress == ["Writing answer"]
    assert reply.text == "This is the exact repository answer."
    assert reply.action is None
    assert reply.action_instruction is None


def test_adapter_forwards_preview_deltas_from_the_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The adapter is a pipe for streamed prose, not a place it can be dropped."""
    deck = tmp_path / "data" / "decks" / "potential.yaml"
    adapter = _adapter(_config(tmp_path), deck)

    def run_agent(
        _config: ProjectConfig,
        _received: Any,
        *,
        progress: Any,
        preview: Any,
    ) -> Any:
        del progress
        preview("partial")
        return _agent_result(answer="This is the exact repository answer.")

    monkeypatch.setattr(
        assistant_adapter,
        "_agent_context",
        lambda _config, *, deck_scope: _agent_context_value(focus_resource_id=None),
    )
    monkeypatch.setattr(
        assistant_adapter.assistant_agent,
        "plan_agent",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(assistant_adapter.assistant_agent, "run_agent", run_agent)
    deltas: list[str] = []

    reply = adapter.chat(
        deck_scope="",
        history=(),
        message="What is in my library?",
        progress=lambda _label: None,
        preview=deltas.append,
    )

    assert deltas == ["partial"]
    assert reply.text == "This is the exact repository answer."


def test_revise_cards_resolves_every_deck_record_in_canonical_deck_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    deck = config.deck_dir / "vocabulary.yaml"
    adapter = _adapter(
        config,
        deck,
        record_ids=("legacy:drill-only",),
        label="Vocabulary",
    )
    context = _agent_context_value(focus_resource_id="resource_deck")
    agent_plan = object()
    result = _agent_result(intents=(_agent_intent(),))
    planned: list[tuple[Path, tuple[str, ...], str, str | None]] = []
    card_plan = _FakeCardRevisionPlan()

    monkeypatch.setattr(assistant_adapter, "_agent_context", lambda *_args, **_kwargs: context)
    monkeypatch.setattr(
        assistant_adapter.assistant_agent,
        "plan_agent",
        lambda *_args, **_kwargs: agent_plan,
    )
    monkeypatch.setattr(
        assistant_adapter.assistant_agent,
        "run_agent",
        lambda _config, received, **_kwargs: (
            result if received is agent_plan else pytest.fail("wrong agent plan")
        ),
    )
    monkeypatch.setattr(
        assistant_adapter.assistant_context,
        "AssistantContextBroker",
        _FakeContextBroker,
    )

    def plan_cards(
        _config: ProjectConfig,
        target: Path,
        record_ids: tuple[str, ...],
        instruction: str,
        *,
        focus_resource_id: str | None,
    ) -> Any:
        planned.append((target, record_ids, instruction, focus_resource_id))
        return card_plan

    monkeypatch.setattr(
        assistant_adapter.card_revision,
        "plan_card_revision",
        plan_cards,
    )

    reply = adapter.chat(
        deck_scope=_scope(adapter),
        history=(),
        message="Improve every card in this deck.",
        progress=lambda _label: None,
        preview=lambda _delta: None,
    )

    assert planned == [
        (
            deck,
            ("word:second", "word:first"),
            "Improve these cards.",
            "resource_deck",
        )
    ]
    assert reply.text == "Here is the exact project answer."
    assert reply.action is not None
    assert reply.action.request_fingerprint == "card-plan"
    assert reply.action.target == "data/decks/vocabulary.yaml"
    assert "word:second, word:first" in reply.action.effects[0]
    wire = "\n".join((*reply.action.effects, *reply.action.disclosures))
    assert "Authentication: auth method=claude.ai, subscription type=max" in wire
    assert "Claude Code version: 2.1.246" in wire
    for index, record in enumerate(card_plan.current_records, start=1):
        exact_card = json.dumps(
            assistant_adapter.assistant_context.assistant_record_value(record),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        assert f"Selected current card {index}: {exact_card}" in wire
    assert reply.action_instruction == "Improve these cards."
    assert adapter._agent_plans["card-plan"].plan is card_plan


@pytest.mark.parametrize(
    "intent",
    [
        _agent_intent(record_ids=("word:first", "word:outside")),
        _agent_intent(
            resource_ids=(
                "resource_deck",
                "resource_first",
                "resource_outside",
            )
        ),
    ],
    ids=("invented-record-id", "card-resource-from-another-deck"),
)
def test_revise_cards_binds_selected_ids_and_card_resources_to_the_deck(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    intent: Any,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config, config.deck_dir / "vocabulary.yaml")
    monkeypatch.setattr(
        assistant_adapter.assistant_context,
        "AssistantContextBroker",
        _FakeContextBroker,
    )

    with pytest.raises(RevisionRefusal, match="not members of the selected deck"):
        adapter._resolve_card_revision_target(
            config,
            intent=intent,
            deck_scope=_scope(adapter),
        )


def test_one_confirmed_agent_card_plan_dispatches_once_and_stops_in_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    deck = config.deck_dir / "vocabulary.yaml"
    adapter = _adapter(config, deck, label="Vocabulary")
    card_plan = _FakeCardRevisionPlan()
    resolved = assistant_adapter._ResolvedCardRevisionTarget(
        target=adapter._targets[0],
        resource_id="resource_deck",
        record_ids=card_plan.selected_record_ids,
        focus_resource_id="resource_deck",
    )
    monkeypatch.setattr(
        assistant_adapter.RevisionAssistantAdapter,
        "_resolve_card_revision_target",
        lambda _self, *_args, **_kwargs: resolved,
    )
    monkeypatch.setattr(
        assistant_adapter.card_revision,
        "plan_card_revision",
        lambda *_args, **_kwargs: card_plan,
    )
    monkeypatch.setattr(
        assistant_adapter.card_revision,
        "CardRevisionPlan",
        _FakeCardRevisionPlan,
    )
    runs: list[Any] = []

    def run_cards(
        fresh: ProjectConfig,
        received: Any,
        *,
        progress: Any,
    ) -> Any:
        assert fresh.root == config.root
        runs.append(received)
        progress("Saving proposals")
        return SimpleNamespace(staging_path=config.staging_dir / "revision.yaml")

    monkeypatch.setattr(
        assistant_adapter.card_revision,
        "run_card_revision",
        run_cards,
    )
    reply = adapter._prepare_agent_intent(
        config,
        result=_agent_result(intents=(_agent_intent(),)),
        deck_scope=_scope(adapter),
    )
    assert reply.action is not None
    progress: list[str] = []
    confirmation = RevisionConfirmation(
        capability="one-use-capability",
        deck_scope=_scope(adapter),
        instruction="Improve these cards.",
        expected_fingerprint=reply.action.request_fingerprint,
        target=reply.action.target,
    )

    execution = adapter.consume_replan_and_execute(
        confirmation,
        progress=progress.append,
    )

    assert runs == [card_plan]
    assert progress == ["Saving proposals"]
    assert "data/staging/revision.yaml" in execution.message
    assert "Canonical cards are unchanged" in execution.message
    assert execution.review_required is not None
    assert "Review the exact proposed Japanese" in execution.review_required
    assert "separate exact actions" in execution.review_required
    assert "Apply and finish" not in execution.review_required
    with pytest.raises(RevisionRefusal, match="missing, stale, already consumed"):
        adapter.consume_replan_and_execute(
            confirmation,
            progress=lambda _label: None,
        )
    assert runs == [card_plan]


def test_one_confirmed_enrichment_batch_resolves_exact_cards_and_stops_in_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    plan = _FakeAiEnrichmentPlan()
    planned: list[tuple[tuple[str, ...], str | None]] = []
    monkeypatch.setattr(
        assistant_adapter.assistant_context,
        "AssistantContextBroker",
        _FakeContextBroker,
    )

    def plan_enrichment(
        fresh: ProjectConfig,
        record_ids: tuple[str, ...],
        *,
        focus_resource_id: str | None,
    ) -> Any:
        assert fresh.root == config.root
        planned.append((record_ids, focus_resource_id))
        return plan

    monkeypatch.setattr(
        assistant_adapter.ai_enrichment,
        "plan_ai_enrichment",
        plan_enrichment,
    )
    intent = _agent_intent(
        kind="enrich_cards",
        resource_ids=("resource_deck", "resource_second", "resource_first"),
        instruction="Fill the missing rich fields on these cards.",
    )

    reply = adapter._prepare_agent_intent(
        config,
        result=_agent_result(intents=(intent,)),
        deck_scope="",
    )

    assert planned == [(("word:second", "word:first"), "resource_deck")]
    assert reply.action is not None
    assert reply.action.request_fingerprint == "e" * 64
    assert reply.action.target == "2 canonical cards"
    rendered = "\n".join(reply.action.effects)
    assert "2 paid enrichment calls" in rendered
    assert "word:second" in rendered
    assert "word:first" in rendered
    assert "579fd68e-f84b-44f8-a949-800fc0d93eb7" in rendered
    assert "3cb5026d-824e-4209-89b4-fd9c66f74966" in rendered
    assert "ai-enrichment-first.yaml" in rendered
    assert "ai-enrichment-second.yaml" in rendered
    assert "Anthropic API billing" in rendered
    for call in plan.calls:
        exact_card = json.dumps(
            assistant_adapter.assistant_context.assistant_record_value(call.record),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        assert f"Exact current card fields: {exact_card}" in rendered
    assert reply.action.confirm_label == "Enrich and stage these exact cards"

    monkeypatch.setattr(
        assistant_adapter.ai_enrichment,
        "AiEnrichmentPlan",
        _FakeAiEnrichmentPlan,
    )
    runs: list[Any] = []

    def run_enrichment(
        fresh: ProjectConfig,
        received: Any,
        *,
        progress: Any,
    ) -> Any:
        assert fresh.root == config.root
        runs.append(received)
        progress("Checking the answer's shape")
        return SimpleNamespace(
            results=(
                SimpleNamespace(
                    state="staged",
                    record_id="word:second",
                    staging_path=config.staging_dir / "ai-enrichment-first.yaml",
                ),
                SimpleNamespace(
                    state="staged",
                    record_id="word:first",
                    staging_path=config.staging_dir / "ai-enrichment-second.yaml",
                ),
            )
        )

    monkeypatch.setattr(
        assistant_adapter.ai_enrichment,
        "run_ai_enrichment",
        run_enrichment,
    )
    confirmation = RevisionConfirmation(
        capability="one-use-capability",
        deck_scope="",
        instruction=intent.instruction,
        expected_fingerprint=reply.action.request_fingerprint,
        target=reply.action.target,
    )
    progress: list[str] = []

    execution = adapter.consume_replan_and_execute(
        confirmation,
        progress=progress.append,
    )

    assert runs == [plan]
    assert progress == ["Checking the answer's shape"]
    assert "2 AI-enrichment proposal(s)" in execution.message
    assert "Canonical cards are unchanged" in execution.message
    assert execution.review_required is not None
    assert "review" in execution.review_required.casefold()
    with pytest.raises(RevisionRefusal, match="missing, stale, already consumed"):
        adapter.consume_replan_and_execute(
            confirmation,
            progress=lambda _label: None,
        )
    assert runs == [plan]


def test_single_enrichment_result_prepares_one_exact_apply_and_finish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    plan = _FakeAiEnrichmentPlan()
    plan.calls = (plan.calls[0],)
    monkeypatch.setattr(
        assistant_adapter.assistant_context,
        "AssistantContextBroker",
        _FakeContextBroker,
    )
    monkeypatch.setattr(
        assistant_adapter.ai_enrichment,
        "plan_ai_enrichment",
        lambda *_args, **_kwargs: plan,
    )
    monkeypatch.setattr(
        assistant_adapter.ai_enrichment,
        "AiEnrichmentPlan",
        _FakeAiEnrichmentPlan,
    )
    staged_path = config.staging_dir / "ai-enrichment-first.yaml"
    monkeypatch.setattr(
        assistant_adapter.ai_enrichment,
        "run_ai_enrichment",
        lambda *_args, **_kwargs: SimpleNamespace(
            results=(
                SimpleNamespace(
                    state="staged",
                    record_id="word:second",
                    staging_path=staged_path,
                ),
            )
        ),
    )
    prepared: list[tuple[Path, str, str]] = []

    def prepare_finish(
        _self: Any,
        _fresh: ProjectConfig,
        *,
        staging_path: Path,
        record_id: str,
        instruction: str,
    ) -> Any:
        prepared.append((staging_path, record_id, instruction))
        return assistant_adapter.StagedContentFinishReview(
            preparation_id="finish-preparation",
            request_fingerprint="f" * 64,
            target="data/decks/vocabulary.yaml",
            effects=("Review exact current and proposed examples",),
            disclosures=("One receipt-backed aggregate",),
        )

    monkeypatch.setattr(
        assistant_adapter.RevisionAssistantAdapter,
        "_prepare_ai_enrichment_finish",
        prepare_finish,
    )
    intent = _agent_intent(
        kind="enrich_cards",
        resource_ids=("resource_deck", "resource_second"),
        instruction="Fill this card's missing rich fields.",
    )
    reply = adapter._prepare_agent_intent(
        config,
        result=_agent_result(intents=(intent,)),
        deck_scope="",
    )
    assert reply.action is not None
    execution = adapter.consume_replan_and_execute(
        RevisionConfirmation(
            capability="one-use-capability",
            deck_scope="",
            instruction=intent.instruction,
            expected_fingerprint=reply.action.request_fingerprint,
            target=reply.action.target,
        ),
        progress=lambda _label: None,
    )

    assert prepared == [(staged_path, "word:second", intent.instruction)]
    assert execution.finish is not None
    assert execution.finish.confirm_label == "Apply and finish"
    assert execution.review_required is None
    assert not execution.complete


def test_unfocused_enrichment_requires_a_deck_before_paid_planning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    monkeypatch.setattr(
        assistant_adapter.assistant_context,
        "AssistantContextBroker",
        _FakeContextBroker,
    )
    monkeypatch.setattr(
        assistant_adapter.ai_enrichment,
        "plan_ai_enrichment",
        lambda *_args, **_kwargs: pytest.fail("must refuse before paid planning"),
    )

    with pytest.raises(RevisionRefusal, match="active deck|deck resource"):
        adapter._prepare_agent_intent(
            config,
            result=_agent_result(
                intents=(
                    _agent_intent(
                        kind="enrich_cards",
                        resource_ids=("resource_second",),
                    ),
                )
            ),
            deck_scope="",
        )


def test_explicit_enrichment_deck_overrides_conversational_focus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)

    class CrossDeckBroker(_FakeContextBroker):
        def catalog(self) -> Any:
            return _disclosure(
                resource_id="resource_catalog",
                kind="catalog",
                data={
                    "resources": [
                        {"resource_id": "resource_active", "kind": "deck"},
                        {"resource_id": "resource_target", "kind": "deck"},
                    ]
                },
            )

        def resource_id_for_deck(self, _deck: Path | str) -> str:
            return "resource_active"

        def deck_context(self, resource_id: str) -> Any:
            assert resource_id in {"resource_active", "resource_target"}
            return SimpleNamespace(records=self.records, deck_kind="vocabulary")

    monkeypatch.setattr(
        assistant_adapter.assistant_context,
        "AssistantContextBroker",
        CrossDeckBroker,
    )
    intent = _agent_intent(
        kind="enrich_cards",
        resource_ids=("resource_target",),
    )

    record_ids, focus = adapter._resolve_enrichment_record_ids(
        config,
        intent=intent,
        deck_scope="data/decks/active.yaml",
    )

    assert record_ids == ("word:second", "word:first")
    assert focus == "resource_target"


def test_enrichment_refuses_explicit_non_vocabulary_destination_before_planning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)

    class ConjugationTargetBroker(_FakeContextBroker):
        def catalog(self) -> Any:
            return _disclosure(
                resource_id="resource_catalog",
                kind="catalog",
                data={
                    "resources": [
                        {"resource_id": "resource_conjugation", "kind": "deck"}
                    ]
                },
            )

        def deck_context(self, resource_id: str) -> Any:
            assert resource_id == "resource_conjugation"
            return SimpleNamespace(records=self.records, deck_kind="conjugation")

    monkeypatch.setattr(
        assistant_adapter.assistant_context,
        "AssistantContextBroker",
        ConjugationTargetBroker,
    )

    with pytest.raises(RevisionRefusal, match="exact vocabulary deck"):
        adapter._resolve_enrichment_record_ids(
            config,
            intent=_agent_intent(
                kind="enrich_cards",
                resource_ids=("resource_conjugation",),
            ),
            deck_scope="",
        )


def test_enrichment_refuses_non_vocabulary_active_focus_before_planning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)

    class ConjugationFocusBroker(_FakeContextBroker):
        def deck_context(self, resource_id: str) -> Any:
            assert resource_id == "resource_deck"
            return SimpleNamespace(records=self.records, deck_kind="conjugation")

    monkeypatch.setattr(
        assistant_adapter.assistant_context,
        "AssistantContextBroker",
        ConjugationFocusBroker,
    )

    with pytest.raises(RevisionRefusal, match="exact vocabulary deck"):
        adapter._resolve_enrichment_record_ids(
            config,
            intent=_agent_intent(
                kind="enrich_cards",
                resource_ids=("resource_second",),
            ),
            deck_scope="data/decks/conjugation.yaml",
        )


def test_enrichment_refuses_an_unfocused_bare_record_id_before_planning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    monkeypatch.setattr(
        assistant_adapter.assistant_context,
        "AssistantContextBroker",
        _FakeContextBroker,
    )
    monkeypatch.setattr(
        assistant_adapter.ai_enrichment,
        "plan_ai_enrichment",
        lambda *_args, **_kwargs: pytest.fail(
            "an unfocused bare record id must not reach enrichment planning"
        ),
    )
    intent = _agent_intent(
        kind="enrich_cards",
        resource_ids=(),
        record_ids=("word:first",),
        instruction="Fill the missing rich fields on this card.",
    )

    with pytest.raises(RevisionRefusal, match="opaque card or deck resource"):
        adapter._prepare_agent_intent(
            config,
            result=_agent_result(intents=(intent,)),
            deck_scope="",
        )


@pytest.mark.parametrize(
    "intent",
    [
        _agent_intent(
            kind="enrich_cards",
            resource_ids=("resource_first",),
            instruction="Replace this card's examples.",
            options={"search_limit": 1},
        ),
        _agent_intent(
            kind="enrich_cards",
            resource_ids=("resource_status",),
            instruction="Fill this card's missing fields.",
        ),
    ],
)
def test_enrichment_refuses_replacement_options_and_noncard_resources_before_planning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    intent: Any,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    monkeypatch.setattr(
        assistant_adapter.assistant_context,
        "AssistantContextBroker",
        _FakeContextBroker,
    )
    monkeypatch.setattr(
        assistant_adapter.ai_enrichment,
        "plan_ai_enrichment",
        lambda *_args, **_kwargs: pytest.fail(
            "an invalid enrichment intent must not reach paid-call planning"
        ),
    )

    with pytest.raises(RevisionRefusal, match="fills missing|only canonical card"):
        adapter._prepare_agent_intent(
            config,
            result=_agent_result(intents=(intent,)),
            deck_scope="",
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("deck_scope", "data/decks/other.yaml"),
        ("target", "data/decks/other.yaml"),
        ("expected_fingerprint", "tampered-fingerprint"),
    ],
)
def test_agent_card_confirmation_refuses_tampered_focus_target_or_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: str,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config, config.deck_dir / "vocabulary.yaml")
    card_plan = _FakeCardRevisionPlan()
    prepared = assistant_adapter._PreparedAgentAction(
        kind="revise_cards",
        focus_scope=_scope(adapter),
        instruction="Improve these cards.",
        target=card_plan.deck_relative_path,
        plan=card_plan,
    )
    adapter._agent_plans[card_plan.plan_fingerprint] = prepared
    monkeypatch.setattr(
        assistant_adapter.card_revision,
        "CardRevisionPlan",
        _FakeCardRevisionPlan,
    )
    monkeypatch.setattr(
        assistant_adapter.card_revision,
        "run_card_revision",
        lambda *_args, **_kwargs: pytest.fail("a tampered binding must not dispatch"),
    )
    values = {
        "capability": "one-use-capability",
        "deck_scope": _scope(adapter),
        "instruction": "Improve these cards.",
        "expected_fingerprint": card_plan.plan_fingerprint,
        "target": card_plan.deck_relative_path,
    }
    values[field] = replacement

    with pytest.raises(RevisionRefusal, match="stale|matches the rendered"):
        adapter.consume_replan_and_execute(
            RevisionConfirmation(**values),
            progress=lambda _label: None,
        )
    assert adapter._agent_plans[card_plan.plan_fingerprint] is prepared


def test_card_action_planner_failure_keeps_the_answer_and_reports_no_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config, config.deck_dir / "vocabulary.yaml")
    planned = object()
    context = _agent_context_value(focus_resource_id="resource_deck")
    resolved = assistant_adapter._ResolvedCardRevisionTarget(
        target=adapter._targets[0],
        resource_id="resource_deck",
        record_ids=("word:first",),
        focus_resource_id="resource_deck",
    )
    monkeypatch.setattr(assistant_adapter, "_agent_context", lambda *_args, **_kwargs: context)
    monkeypatch.setattr(
        assistant_adapter.assistant_agent,
        "plan_agent",
        lambda *_args, **_kwargs: planned,
    )
    monkeypatch.setattr(
        assistant_adapter.assistant_agent,
        "run_agent",
        lambda *_args, **_kwargs: _agent_result(
            answer="I found the requested cards.",
            intents=(_agent_intent(),),
        ),
    )
    monkeypatch.setattr(
        assistant_adapter.RevisionAssistantAdapter,
        "_resolve_card_revision_target",
        lambda _self, *_args, **_kwargs: resolved,
    )
    monkeypatch.setattr(
        assistant_adapter.card_revision,
        "plan_card_revision",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            JankiError("the canonical deck changed before planning")
        ),
    )

    reply = adapter.chat(
        deck_scope=_scope(adapter),
        history=(),
        message="Improve this card.",
        progress=lambda _label: None,
        preview=lambda _delta: None,
    )

    assert reply.text.startswith("I found the requested cards.\n\n")
    assert "the canonical deck changed before planning" in reply.text
    assert reply.text.endswith("Nothing was changed.")
    assert reply.action is None
    assert adapter._agent_plans == {}


def test_operation_intent_plans_and_reads_exact_recovery_without_another_model_call(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    journal = operations.OperationJournal.load(config.operations_file)
    journal.authorize(
        "op-1",
        kind="assistant_chat",
        source_file="data/assistant/request.json",
        source_sha256="a" * 64,
        request_fp="b" * 64,
        model="claude-opus-5",
    )
    journal.advance("op-1", "dispatching")
    payload = b'{"answer":"recover me"}'
    journal.capture_result(
        "op-1",
        lambda: operations.capture_artifact(
            config.operations_file,
            "op-1",
            payload,
        ),
    )
    broker = assistant_adapter.assistant_context.AssistantContextBroker(config)
    catalog = json.loads(broker.catalog().wire)["data"]["resources"]
    operations_resource = next(
        item["resource_id"] for item in catalog if item["kind"] == "operations"
    )

    reply = adapter._prepare_agent_intent(
        config,
        result=_agent_result(
            answer="I can show the exact recovery reply.",
            intents=(
                _agent_intent(
                    kind="manage_operation",
                    resource_ids=(operations_resource,),
                    instruction="Show reply for operation op-1.",
                    options={
                        "operation_id": "op-1",
                        "operation_action": "show_reply",
                    },
                ),
            ),
        ),
        deck_scope="",
        owner_message="Show reply for operation op-1.",
    )

    assert reply.action is not None
    assert reply.action.confirm_label == "Show this exact private reply"
    assert "Do not change or settle" in "\n".join(reply.action.effects)
    execution = adapter.consume_replan_and_execute(
        RevisionConfirmation(
            capability="owner-click",
            deck_scope="",
            instruction="Show reply for operation op-1.",
            expected_fingerprint=reply.action.request_fingerprint,
            target="Paid operation op-1",
        ),
        progress=lambda _label: None,
    )
    assert execution.complete
    assert execution.remember_in_chat_context is False
    assert hashlib.sha256(payload).hexdigest() in execution.message
    assert "recover me" in execution.message


def test_local_operation_manager_recovers_captured_assistant_turn_without_raw_reply_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    operation_id = "55555555-5555-4555-8555-555555555555"
    raw_private_reply = b"RAW-PRIVATE-ENVELOPE-MUST-NOT-ENTER-CHAT-CONTEXT"
    journal = operations.OperationJournal.load(config.operations_file)
    journal.authorize(
        operation_id,
        kind="assistant_agent",
        source_file="janki-project",
        source_sha256="a" * 64,
        request_fp="b" * 64,
        model="claude-opus-5",
    )
    journal.advance(operation_id, "dispatching")
    captured = journal.capture_result(
        operation_id,
        lambda: operations.capture_artifact(
            config.operations_file,
            operation_id,
            raw_private_reply,
        ),
    )
    manifest_path = config.assistant_dir / f"{operation_id}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_wire = json.dumps(
        {
            "schema_version": 1,
            "kind": "assistant_agent",
            "state": "request",
            "operation_id": operation_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    manifest_path.write_bytes(manifest_wire)

    def recover_agent(
        fresh_config: ProjectConfig,
        selected_operation_id: str,
        **_kwargs: object,
    ) -> object:
        manifest_path.write_bytes(manifest_wire + b"\nrecovered")
        operations.OperationJournal.load(fresh_config.operations_file).advance(
            selected_operation_id,
            "committed",
        )
        return SimpleNamespace(
            manifest_path=manifest_path,
            answer="Recovered decoded answer",
            action_intents=(SimpleNamespace(kind="revise_cards"),),
        )

    monkeypatch.setattr(
        assistant_adapter.assistant_operations.assistant_agent,
        "recover_agent",
        recover_agent,
    )

    reply = adapter.prepare_operation_action(
        operation_id=operation_id,
        action="recover",
        accept_paid_output_loss=False,
        deck_scope="",
    )

    assert reply.action is not None
    rendered = "\n".join(reply.action.effects)
    assert captured.artifact is not None
    assert captured.artifact.content_sha256 in rendered
    assert hashlib.sha256(manifest_wire).hexdigest() in rendered
    assert "commit the captured Assistant turn" in rendered
    execution = adapter.consume_replan_and_execute(
        RevisionConfirmation(
            capability="owner-click",
            deck_scope="",
            instruction=f"recover paid operation {operation_id}",
            expected_fingerprint=reply.action.request_fingerprint,
            target=f"Paid operation {operation_id}",
        ),
        progress=lambda _label: None,
    )

    assert execution.complete
    assert execution.remember_in_chat_context is False
    assert "Recovered decoded answer" in execution.message
    assert "1 typed action intent" in execution.message
    assert "was not prepared or executed" in execution.message
    assert raw_private_reply.decode() not in execution.message
    assert manifest_path.name in execution.message
    assert hashlib.sha256(manifest_wire + b"\nrecovered").hexdigest() in execution.message


def test_whole_deck_revision_uses_one_typed_plan_without_requiring_focus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    deck = config.deck_dir / "potential.yaml"
    adapter = _adapter(config, deck)
    planned = _revision_plan()
    monkeypatch.setattr(
        assistant_adapter.assistant_context,
        "AssistantContextBroker",
        _FakeContextBroker,
    )
    monkeypatch.setattr(
        assistant_adapter.revision,
        "plan_revision",
        lambda *_args, **_kwargs: planned,
    )

    reply = adapter._prepare_agent_intent(
        config,
        result=_agent_result(
            answer="I prepared the exact teaching-content revision.",
            intents=(
                _agent_intent(
                    kind="revise_deck",
                    resource_ids=("resource_deck",),
                    instruction="Add clearer potential-form teaching examples.",
                ),
            ),
        ),
        deck_scope="",
    )

    assert reply.action is not None
    assert reply.action.request_fingerprint == "rendered-plan"
    assert reply.action.target == "data/decks/potential.yaml"
    assert reply.action_instruction == "Add clearer potential-form teaching examples."
    prepared = adapter._agent_plans["rendered-plan"]
    assert prepared.kind == "revise_deck"
    assert prepared.focus_scope == ""
    assert prepared.plan is planned


def test_confirmed_whole_deck_revision_delegates_once_to_the_shared_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config, config.deck_dir / "potential.yaml")
    plan = _revision_plan()
    monkeypatch.setattr(assistant_adapter.revision, "RevisionPlan", SimpleNamespace)
    prepared = assistant_adapter._PreparedAgentAction(
        kind="revise_deck",
        focus_scope="",
        instruction="Improve the deck examples.",
        target="data/decks/potential.yaml",
        plan=plan,
    )
    adapter._agent_plans["rendered-plan"] = prepared
    delegated: list[Any] = []
    expected = SimpleNamespace(message="revision staged", finish=None, complete=False)

    def stage(_self: Any, received: Any, *, progress: Any) -> Any:
        delegated.append(received)
        progress("Reading the source")
        return expected

    monkeypatch.setattr(
        assistant_adapter.RevisionAssistantAdapter,
        "_stage_deck_revision",
        stage,
    )
    progress: list[str] = []

    result = adapter.consume_replan_and_execute(
        RevisionConfirmation(
            capability="one-use",
            deck_scope="",
            instruction="Improve the deck examples.",
            expected_fingerprint="rendered-plan",
            target="data/decks/potential.yaml",
        ),
        progress=progress.append,
    )

    assert result is expected
    assert delegated == [plan]
    assert progress == ["Reading the source"]
    assert adapter._agent_plans == {}


def test_equal_plans_prepared_by_two_threads_each_keep_one_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config, config.deck_dir / "potential.yaml")
    plan = _revision_plan()
    monkeypatch.setattr(assistant_adapter.revision, "RevisionPlan", SimpleNamespace)
    prepared = assistant_adapter._PreparedAgentAction(
        kind="revise_deck",
        focus_scope="",
        instruction="Improve the deck examples.",
        target="data/decks/potential.yaml",
        plan=plan,
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(
            executor.map(
                lambda _index: adapter._remember_agent_plan(
                    "rendered-plan",
                    prepared,
                ),
                range(2),
            )
        )
    delegated: list[Any] = []

    def stage(_self: Any, received: Any, *, progress: Any) -> Any:
        delegated.append(received)
        return SimpleNamespace(message="revision staged", finish=None, complete=False)

    monkeypatch.setattr(
        assistant_adapter.RevisionAssistantAdapter,
        "_stage_deck_revision",
        stage,
    )
    confirmation = RevisionConfirmation(
        capability="one-use",
        deck_scope="",
        instruction="Improve the deck examples.",
        expected_fingerprint="rendered-plan",
        target="data/decks/potential.yaml",
    )

    adapter.consume_replan_and_execute(confirmation, progress=lambda _label: None)
    assert adapter._agent_plans["rendered-plan"] is prepared
    adapter.consume_replan_and_execute(confirmation, progress=lambda _label: None)

    assert delegated == [plan, plan]
    assert adapter._agent_plans == {}
    assert adapter._agent_plan_duplicates == {}


def test_agent_plan_cache_evicts_the_oldest_rendered_capability_at_its_bound(
    tmp_path: Path,
) -> None:
    adapter = _adapter(_config(tmp_path))
    prepared = assistant_adapter._PreparedAgentAction(
        kind="revise_deck",
        focus_scope="",
        instruction="Improve the deck examples.",
        target="data/decks/potential.yaml",
        plan=_revision_plan(),
    )

    for index in range(257):
        adapter._remember_agent_plan(f"{index:064x}", prepared)

    assert len(adapter._agent_plans) == 256
    assert f"{0:064x}" not in adapter._agent_plans
    assert adapter._agent_plans[f"{1:064x}"] is prepared
    assert adapter._agent_plans[f"{256:064x}"] is prepared
    assert adapter._agent_plan_duplicates == {}


def test_inspect_resources_renders_three_exact_local_snapshots_without_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    disclosures = {
        "resource_deck": _disclosure(
            resource_id="resource_deck",
            kind="deck",
            data={"name": "Weekly Review", "cards": 16},
        ),
        "resource_status": _disclosure(
            resource_id="resource_status",
            kind="status",
            data={"staged": 2, "unfinished_operations": 0},
        ),
        "resource_source": _disclosure(
            resource_id="resource_source",
            kind="source",
            data={"name": "lesson.pdf", "state": "staged"},
        ),
    }
    snapshots: list[str] = []

    class Broker:
        def __init__(self, fresh: ProjectConfig) -> None:
            assert fresh.root == config.root

        def snapshot(self, resource_id: str) -> Any:
            snapshots.append(resource_id)
            return disclosures[resource_id]

    monkeypatch.setattr(
        assistant_adapter.assistant_context,
        "AssistantContextBroker",
        Broker,
    )
    intent = _agent_intent(
        kind="inspect_resources",
        resource_ids=("resource_deck", "resource_status", "resource_source"),
        instruction="Show these exact local resources.",
    )

    reply = adapter._prepare_agent_intent(
        config,
        result=_agent_result(
            answer="I requested the exact local facts.",
            intents=(intent,),
        ),
        deck_scope="",
    )

    assert snapshots == ["resource_deck", "resource_status", "resource_source"]
    assert reply.action is None
    assert reply.action_instruction is None
    assert adapter._agent_plans == {}
    assert reply.text.startswith(
        "I requested the exact local facts.\n\n### Local repository result"
    )
    for disclosure in disclosures.values():
        assert f"SHA-256 `{disclosure.sha256}`" in reply.text
        value = json.loads(disclosure.wire)
        rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
        assert rendered in reply.text


@pytest.mark.parametrize(
    "intent",
    [
        _agent_intent(
            kind="inspect_resources",
            resource_ids=(),
            instruction="Inspect.",
        ),
        _agent_intent(
            kind="inspect_resources",
            resource_ids=("one", "two", "three", "four"),
            instruction="Inspect.",
        ),
        _agent_intent(
            kind="inspect_resources",
            resource_ids=("resource_deck",),
            record_ids=("word:first",),
            instruction="Inspect.",
        ),
        _agent_intent(
            kind="inspect_resources",
            resource_ids=("resource_deck",),
            instruction="Inspect.",
            options={"search_limit": 2},
        ),
    ],
    ids=("no-target", "too-many-targets", "write-card-target", "options"),
)
def test_inspect_resources_refuses_invalid_targets_and_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    intent: Any,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)

    class Broker:
        def __init__(self, _fresh: ProjectConfig) -> None:
            pass

        def snapshot(self, _resource_id: str) -> Any:
            pytest.fail("invalid inspection must not read a resource")

    monkeypatch.setattr(
        assistant_adapter.assistant_context,
        "AssistantContextBroker",
        Broker,
    )

    with pytest.raises(RevisionRefusal, match="one to three|write-target|write options"):
        adapter._prepare_agent_intent(
            config,
            result=_agent_result(intents=(intent,)),
            deck_scope="",
        )


def test_inspect_resources_preserves_an_unknown_resource_refusal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)

    class Broker:
        def __init__(self, _fresh: ProjectConfig) -> None:
            pass

        def snapshot(self, resource_id: str) -> Any:
            raise JankiError(f"Unknown Assistant resource {resource_id!r}")

    monkeypatch.setattr(
        assistant_adapter.assistant_context,
        "AssistantContextBroker",
        Broker,
    )

    result = _agent_result(
        answer="I need the exact local resource.",
        intents=(
            _agent_intent(
                kind="inspect_resources",
                resource_ids=("invented",),
                instruction="Inspect it.",
            ),
        ),
    )
    monkeypatch.setattr(
        assistant_adapter,
        "_agent_context",
        lambda *_args, **_kwargs: _agent_context_value(),
    )
    monkeypatch.setattr(
        assistant_adapter.assistant_agent,
        "plan_agent",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        assistant_adapter.assistant_agent,
        "run_agent",
        lambda *_args, **_kwargs: result,
    )

    reply = adapter.chat(
        deck_scope="",
        history=(),
        message="Inspect that resource.",
        progress=lambda _label: None,
        preview=lambda _delta: None,
    )

    assert "I need the exact local resource." in reply.text
    assert "Unknown Assistant resource 'invented'" in reply.text
    assert reply.text.endswith("Nothing was changed.")
    assert reply.action is None


def test_search_cards_forwards_exact_literal_and_limit_and_renders_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    disclosure = _disclosure(
        resource_id="resource_search",
        kind="card",
        data={
            "literal": "食",
            "matches": [
                {"id": "word:食べる:たべる", "expression": "食べる"},
                {"id": "word:食事:しょくじ", "expression": "食事"},
            ],
        },
    )
    searches: list[tuple[str, int]] = []

    class Broker:
        def __init__(self, fresh: ProjectConfig) -> None:
            assert fresh.root == config.root

        def search_cards(self, literal: str, *, limit: int) -> Any:
            searches.append((literal, limit))
            return disclosure

    monkeypatch.setattr(
        assistant_adapter.assistant_context,
        "AssistantContextBroker",
        Broker,
    )
    intent = _agent_intent(
        kind="search_cards",
        resource_ids=(),
        instruction="Find cards containing this exact text.",
        options={"search_literal": "食", "search_limit": 7},
    )

    reply = adapter._prepare_agent_intent(
        config,
        result=_agent_result(
            answer="I requested an exact local card search.",
            intents=(intent,),
        ),
        deck_scope="",
    )

    assert searches == [("食", 7)]
    assert reply.action is None
    assert reply.action_instruction is None
    assert adapter._agent_plans == {}
    assert f"SHA-256 `{disclosure.sha256}`" in reply.text
    assert "word:食べる:たべる" in reply.text
    assert '"expression": "食事"' in reply.text


@pytest.mark.parametrize(
    "intent",
    [
        _agent_intent(
            kind="search_cards",
            resource_ids=("resource_deck",),
            instruction="Search.",
            options={"search_literal": "食"},
        ),
        _agent_intent(
            kind="search_cards",
            resource_ids=(),
            instruction="Search.",
            options={"search_literal": "食", "search_limit": 0},
        ),
        _agent_intent(
            kind="search_cards",
            resource_ids=(),
            instruction="Search.",
            options={"search_literal": "食", "search_limit": 21},
        ),
        _agent_intent(
            kind="search_cards",
            resource_ids=(),
            instruction="Search.",
            options={"search_literal": "食", "search_limit": True},
        ),
        _agent_intent(
            kind="search_cards",
            resource_ids=(),
            instruction="Search.",
            options={"search_limit": 5},
        ),
        _agent_intent(
            kind="search_cards",
            resource_ids=(),
            instruction="Search.",
            options={"search_literal": "食", "deck": "resource_deck"},
        ),
    ],
    ids=(
        "resource-target",
        "zero-limit",
        "over-limit",
        "boolean-limit",
        "missing-literal",
        "extra-option",
    ),
)
def test_search_cards_refuses_resource_targets_invalid_limits_and_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    intent: Any,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)

    class Broker:
        def __init__(self, _fresh: ProjectConfig) -> None:
            pass

        def search_cards(self, _literal: str, *, limit: int) -> Any:
            pytest.fail(f"invalid search must not reach the broker ({limit=})")

    monkeypatch.setattr(
        assistant_adapter.assistant_context,
        "AssistantContextBroker",
        Broker,
    )

    with pytest.raises(RevisionRefusal, match="resource target|options are invalid|limit"):
        adapter._prepare_agent_intent(
            config,
            result=_agent_result(intents=(intent,)),
            deck_scope="",
        )


def test_assign_cards_renders_exact_tag_ownership_and_staging_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    plan = _FakeAssignmentPlan()
    intent = _agent_intent(
        kind="assign_cards",
        resource_ids=("resource_proposal",),
        record_ids=("word:食べる:たべる",),
        instruction="Assign this staged card to Lesson deck.",
        options={"destination_resource_id": "resource_destination"},
    )
    planned: list[tuple[str, str, tuple[str, ...], str]] = []

    def plan_assignment(
        fresh: ProjectConfig,
        *,
        proposal_resource_id: str,
        destination_resource_id: str,
        record_ids: tuple[str, ...],
        instruction: str,
    ) -> Any:
        assert fresh.root == config.root
        planned.append(
            (
                proposal_resource_id,
                destination_resource_id,
                record_ids,
                instruction,
            )
        )
        return plan

    monkeypatch.setattr(
        assistant_adapter.assistant_assignment,
        "plan_assignment",
        plan_assignment,
    )

    reply = adapter._prepare_agent_intent(
        config,
        result=_agent_result(intents=(intent,)),
        deck_scope="",
    )

    assert planned == [
        (
            "resource_proposal",
            "resource_destination",
            ("word:食べる:たべる",),
            "Assign this staged card to Lesson deck.",
        )
    ]
    assert reply.action is not None
    assert reply.action.request_fingerprint == "9" * 64
    assert reply.action.target == "lesson.pdf"
    assert reply.action.confirm_label == "Assign these exact staged cards"
    assert reply.action.progress_label == "Saving deck assignments"
    wire = "\n".join((*reply.action.effects, *reply.action.disclosures))
    assert "Assign 食べる（たべる） [word:食べる:たべる] to Lesson deck" in wire
    assert '"before": ["personal"]' in wire
    assert '"after": ["personal", "lesson-intake"]' in wire
    assert '"takes": true' in wire
    assert '"source_name": "lesson.pdf"' in wire
    assert "b" * 64 in wire
    assert "a" * 64 in wire
    assert "c" * 64 in wire
    assert "d" * 64 in wire
    assert '"staging_proposal": "data/staging/lesson.pdf.yaml"' in wire
    assert "changes only the displayed ownership tags" in wire
    assert "does not approve Japanese" in wire
    assert adapter._agent_plans["9" * 64].plan is plan


@pytest.mark.parametrize(
    ("resource_ids", "record_ids", "options"),
    [
        ((), ("word:first",), {"destination_resource_id": "resource_deck"}),
        (
            ("resource_proposal",),
            (),
            {"destination_resource_id": "resource_deck"},
        ),
        (("resource_proposal",), ("word:first",), {}),
        (
            ("resource_proposal",),
            ("word:first",),
            {"destination_resource_id": "resource_deck", "review_patterns": True},
        ),
        (
            ("resource_proposal",),
            ("word:first",),
            {"destination_resource_id": 7},
        ),
    ],
)
def test_assign_cards_refuses_incomplete_or_extra_closed_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resource_ids: tuple[str, ...],
    record_ids: tuple[str, ...],
    options: dict[str, Any],
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    intent = _agent_intent(
        kind="assign_cards",
        resource_ids=resource_ids,
        record_ids=record_ids,
        instruction="Assign selected cards.",
        options=options,
    )
    monkeypatch.setattr(
        assistant_adapter.assistant_assignment,
        "plan_assignment",
        lambda *_args, **_kwargs: pytest.fail(
            "an invalid assignment intent must refuse before planning"
        ),
    )

    with pytest.raises(RevisionRefusal, match="exactly one source-extraction"):
        adapter._prepare_agent_intent(
            config,
            result=_agent_result(intents=(intent,)),
            deck_scope="",
        )


def test_one_assign_cards_confirmation_executes_the_exact_plan_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    plan = _FakeAssignmentPlan()
    instruction = "Assign this staged card to Lesson deck."
    adapter._agent_plans[plan.fingerprint] = assistant_adapter._PreparedAgentAction(
        kind="assign_cards",
        focus_scope="",
        instruction=instruction,
        target=plan.source_name,
        plan=plan,
    )
    monkeypatch.setattr(
        assistant_adapter.assistant_assignment,
        "AssistantAssignmentPlan",
        _FakeAssignmentPlan,
    )
    delegated: list[Any] = []

    def execute(fresh: ProjectConfig, received: Any) -> Any:
        assert fresh.root == config.root
        delegated.append(received)
        return SimpleNamespace(
            plan=received,
            assigned_record_ids=("word:食べる:たべる",),
        )

    monkeypatch.setattr(
        assistant_adapter.assistant_assignment,
        "execute_assignment",
        execute,
    )
    confirmation = RevisionConfirmation(
        capability="one-use",
        deck_scope="",
        instruction=instruction,
        expected_fingerprint=plan.fingerprint,
        target=plan.source_name,
    )

    execution = adapter.consume_replan_and_execute(
        confirmation,
        progress=lambda _label: pytest.fail(
            "the local assignment service reports no provider progress"
        ),
    )

    assert delegated == [plan]
    assert execution.complete is True
    assert "word:食べる:たべる" in execution.message
    assert "Lesson deck" in execution.message
    assert "Canonical cards and deck definitions are unchanged" in execution.message
    with pytest.raises(RevisionRefusal, match="missing, stale, already consumed"):
        adapter.consume_replan_and_execute(
            confirmation,
            progress=lambda _label: None,
        )
    assert delegated == [plan]


@pytest.mark.parametrize(
    ("field", "replacement"),
    [("target", "other.pdf"), ("expected_fingerprint", "0" * 64)],
)
def test_assign_cards_confirmation_binds_target_and_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: str,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    plan = _FakeAssignmentPlan()
    instruction = "Assign this staged card to Lesson deck."
    adapter._agent_plans[plan.fingerprint] = assistant_adapter._PreparedAgentAction(
        kind="assign_cards",
        focus_scope="",
        instruction=instruction,
        target=plan.source_name,
        plan=plan,
    )
    monkeypatch.setattr(
        assistant_adapter.assistant_assignment,
        "AssistantAssignmentPlan",
        _FakeAssignmentPlan,
    )
    monkeypatch.setattr(
        assistant_adapter.assistant_assignment,
        "execute_assignment",
        lambda *_args, **_kwargs: pytest.fail(
            "a tampered assignment must not execute"
        ),
    )
    values = {
        "capability": "one-use",
        "deck_scope": "",
        "instruction": instruction,
        "expected_fingerprint": plan.fingerprint,
        "target": plan.source_name,
    }
    values[field] = replacement

    with pytest.raises(RevisionRefusal, match="stale|matches the rendered"):
        adapter.consume_replan_and_execute(
            RevisionConfirmation(**values),
            progress=lambda _label: None,
        )


def test_assign_cards_executor_independently_rechecks_the_plan_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    plan = _FakeAssignmentPlan()
    instruction = "Assign this staged card to Lesson deck."
    prepared = assistant_adapter._PreparedAgentAction(
        kind="assign_cards",
        focus_scope="",
        instruction=instruction,
        target=plan.source_name,
        plan=plan,
    )
    monkeypatch.setattr(
        assistant_adapter.assistant_assignment,
        "AssistantAssignmentPlan",
        _FakeAssignmentPlan,
    )
    monkeypatch.setattr(
        assistant_adapter.assistant_assignment,
        "execute_assignment",
        lambda *_args, **_kwargs: pytest.fail(
            "a mismatched assignment fingerprint must not execute"
        ),
    )

    with pytest.raises(RevisionRefusal, match="no longer matches"):
        adapter._consume_agent_action(
            RevisionConfirmation(
                capability="one-use",
                deck_scope="",
                instruction=instruction,
                expected_fingerprint="0" * 64,
                target=plan.source_name,
            ),
            prepared,
            progress=lambda _label: None,
        )


def test_staging_review_card_renders_exact_japanese_patterns_and_owner_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    plan = _FakeStagingReviewPlan()
    intent = _agent_intent(
        kind="review_staging",
        resource_ids=("resource_proposal",),
        record_ids=("word:食べる:たべる",),
        instruction="Approve this exact card and its grammar patterns.",
        options={"review_patterns": True},
    )
    planned: list[tuple[str, tuple[str, ...], bool]] = []

    def plan_review(
        fresh: ProjectConfig,
        *,
        proposal_resource_id: str,
        record_ids: tuple[str, ...],
        review_patterns: bool,
    ) -> Any:
        assert fresh.root == config.root
        planned.append((proposal_resource_id, record_ids, review_patterns))
        return plan

    monkeypatch.setattr(
        assistant_adapter.assistant_staging_review,
        "plan_staging_review",
        plan_review,
    )
    monkeypatch.setattr(
        assistant_adapter.assistant_context.AssistantContextBroker,
        "proposal_context",
        lambda *_args: SimpleNamespace(proposal_kind="source_extraction"),
    )
    reply = adapter._prepare_agent_intent(
        config,
        result=_agent_result(intents=(intent,)),
        deck_scope="",
    )

    assert planned == [
        ("resource_proposal", ("word:食べる:たべる",), True)
    ]
    assert reply.action is not None
    assert reply.action.request_fingerprint == "3" * 64
    assert reply.action.target == "lesson.pdf"
    assert reply.action.confirm_label == "Approve this exact review"
    assert reply.action.progress_label == "Saving review"
    wire = "\n".join((*reply.action.effects, *reply.action.disclosures))
    assert "食べる（たべる） [word:食べる:たべる]" in wire
    assert "japanese: 朝ご飯が食べられます。" in wire
    assert "furigana: 朝ご飯[あさごはん]が 食[た]べられます。" in wire
    assert "english: I can eat breakfast." in wire
    assert '"template": "る → られる"' in wire
    assert "1" * 64 in wire
    assert "2" * 64 in wire
    assert '"pattern_review": "lesson.pdf"' in wire
    assert "owner's review decision" in wire
    assert adapter._agent_plans["3" * 64].plan is plan


def test_card_revision_review_is_one_apply_and_finish_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    review = _FakeCardRevisionReviewPlan()
    instruction = "Approve this exact revised card."
    plan = _FakeCardRevisionFinishPlan(
        config.root,
        instruction=instruction,
        review=review,
    )
    intent = _agent_intent(
        kind="review_staging",
        resource_ids=("resource_revision",),
        record_ids=review.record_ids,
        instruction=instruction,
        options={},
    )
    monkeypatch.setattr(
        assistant_adapter.assistant_context.AssistantContextBroker,
        "proposal_context",
        lambda *_args: SimpleNamespace(proposal_kind="card_revision"),
    )
    monkeypatch.setattr(
        assistant_adapter.card_revision_finish,
        "plan_card_revision_finish",
        lambda *_args, **_kwargs: plan,
    )
    monkeypatch.setattr(
        assistant_adapter.assistant_card_revision_review,
        "AssistantCardRevisionReviewPlan",
        _FakeCardRevisionReviewPlan,
    )
    monkeypatch.setattr(
        assistant_adapter.card_revision_finish,
        "CardRevisionFinishPlan",
        _FakeCardRevisionFinishPlan,
    )

    reply = adapter._prepare_agent_intent(
        config,
        result=_agent_result(intents=(intent,)),
        deck_scope="",
    )

    assert reply.action is not None
    assert reply.action.target == "data/decks/lesson.yaml"
    assert reply.action.confirm_label == "Apply and finish"
    assert reply.action.progress_label == "Preparing finish"
    rendered = "\n".join((*reply.action.effects, *reply.action.disclosures))
    assert "old \"Old note\" → new \"Reviewed new note\"" in rendered
    assert "word:聞く:きく" in rendered
    assert "Audio word" in rendered
    assert "Audio example" in rendered
    assert "voice 0" in rendered
    assert "openai-realtime (paid-network)" in rendered
    assert "日本語で話せます。" in rendered
    assert "Require the package output to remain absent" in rendered
    assert "Build 2 card(s) from 1 note(s)" in rendered
    assert "At most 1 paid audio-provider call" in rendered
    assert "authority receipt is saved before review or promotion" in rendered
    assert adapter._agent_plans[plan.fingerprint].plan is plan

    delegated: list[Any] = []
    progress: list[str] = []

    def execute(
        _config: ProjectConfig,
        received: Any,
        *,
        progress: Any,
    ) -> Any:
        delegated.append(received)
        for label in (
            "Applying reviewed cards",
            "Creating card audio",
            "Building Anki package",
            "Saving finish receipt",
        ):
            progress(label)
        return SimpleNamespace(
            succeeded=True,
            state="complete",
            receipt_id=received.fingerprint,
            output_path=received.build.output_path,
            card_count=received.build.card_count,
            package_sha256="9" * 64,
            audio=None,
        )

    monkeypatch.setattr(
        assistant_adapter.card_revision_finish,
        "execute_card_revision_finish",
        execute,
    )
    confirmation = RevisionConfirmation(
        capability="one-use",
        deck_scope="",
        instruction=instruction,
        expected_fingerprint=plan.fingerprint,
        target="data/decks/lesson.yaml",
    )
    execution = adapter.consume_replan_and_execute(
        confirmation,
        progress=progress.append,
    )
    assert delegated == [plan]
    assert progress == [
        "Applying reviewed cards",
        "Creating card audio",
        "Building Anki package",
        "Saving finish receipt",
    ]
    assert execution.complete is True
    assert "created their selected audio" in execution.message
    assert "Finish receipt: " + plan.fingerprint in execution.message
    with pytest.raises(RevisionRefusal, match="missing, stale, already consumed"):
        adapter.consume_replan_and_execute(
            confirmation,
            progress=lambda _label: None,
        )
    assert delegated == [plan]


def test_card_apply_and_finish_failure_names_the_durable_resume_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    instruction = "Apply and finish this exact review."
    plan = _FakeCardRevisionFinishPlan(
        config.root,
        instruction=instruction,
        review=_FakeCardRevisionReviewPlan(),
    )
    target = "data/decks/lesson.yaml"
    adapter._agent_plans[plan.fingerprint] = assistant_adapter._PreparedAgentAction(
        kind="review_staging",
        focus_scope="",
        instruction=instruction,
        target=target,
        plan=plan,
    )
    monkeypatch.setattr(
        assistant_adapter.card_revision_finish,
        "CardRevisionFinishPlan",
        _FakeCardRevisionFinishPlan,
    )
    monkeypatch.setattr(
        assistant_adapter.card_revision_finish,
        "execute_card_revision_finish",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            JankiError("package publisher stopped")
        ),
    )
    monkeypatch.setattr(
        assistant_adapter.card_revision_finish,
        "inspect_card_revision_finish",
        lambda *_args, **_kwargs: SimpleNamespace(
            receipt_id=plan.fingerprint,
            state="audio_complete",
        ),
    )

    with pytest.raises(RevisionRefusal) as caught:
        adapter.consume_replan_and_execute(
            RevisionConfirmation(
                capability="one-use",
                deck_scope="",
                instruction=instruction,
                expected_fingerprint=plan.fingerprint,
                target=target,
            ),
            progress=lambda _label: None,
        )

    text = str(caught.value)
    assert "package publisher stopped" in text
    assert "Durable receipt " + plan.fingerprint in text
    assert "state audio_complete" in text
    assert "resume that exact receipt" in text


def test_card_revision_review_rejects_pattern_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    monkeypatch.setattr(
        assistant_adapter.assistant_context.AssistantContextBroker,
        "proposal_context",
        lambda *_args: SimpleNamespace(proposal_kind="card_revision"),
    )
    intent = _agent_intent(
        kind="review_staging",
        resource_ids=("resource_revision",),
        record_ids=("word:話す:はなす",),
        instruction="Approve revision and patterns.",
        options={"review_patterns": True},
    )
    with pytest.raises(RevisionRefusal, match="do not carry grammar-pattern"):
        adapter._prepare_agent_intent(
            config,
            result=_agent_result(intents=(intent,)),
            deck_scope="",
        )


def test_source_staging_review_still_requires_explicit_pattern_choice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    monkeypatch.setattr(
        assistant_adapter.assistant_context.AssistantContextBroker,
        "proposal_context",
        lambda *_args: SimpleNamespace(proposal_kind="source_extraction"),
    )
    monkeypatch.setattr(
        assistant_adapter.assistant_staging_review,
        "plan_staging_review",
        lambda *_args, **_kwargs: _FakeStagingReviewPlan(),
    )
    intent = _agent_intent(
        kind="review_staging",
        resource_ids=("resource_proposal",),
        record_ids=("word:話す:はなす",),
        instruction="Review this source proposal.",
        options={},
    )

    with pytest.raises(RevisionRefusal, match="explicit.*grammar-pattern decision"):
        adapter._prepare_agent_intent(
            config,
            result=_agent_result(intents=(intent,)),
            deck_scope="",
        )


def test_promotion_card_renders_exact_landing_holds_writes_and_authority_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    plan = _FakePromotionPlan()
    intent = _agent_intent(
        kind="promote_staging",
        resource_ids=("resource_proposal",),
        instruction="Promote this reviewed proposal.",
    )
    planned: list[tuple[str, str]] = []

    def plan_promotion(
        fresh: ProjectConfig,
        resource_id: str,
        instruction: str,
    ) -> Any:
        assert fresh.root == config.root
        planned.append((resource_id, instruction))
        return plan

    monkeypatch.setattr(
        assistant_adapter.assistant_promotion,
        "plan_promotion_action",
        plan_promotion,
    )

    reply = adapter._prepare_agent_intent(
        config,
        result=_agent_result(intents=(intent,)),
        deck_scope="",
    )

    assert planned == [("resource_proposal", "Promote this reviewed proposal.")]
    assert reply.action is not None
    assert reply.action.request_fingerprint == "4" * 64
    assert reply.action.target == "lesson.pdf"
    assert reply.action.confirm_label == "Promote this exact proposal"
    assert reply.action.progress_label == "Checking the reviewed proposal"
    wire = "\n".join((*reply.action.effects, *reply.action.disclosures))
    assert "state lands with 1 landing card(s) and 1 held card(s)" in wire
    assert "word:食べる:たべる" in wire
    assert "word:未定:みてい" in wire
    assert "identity decision required" in wire
    assert "data/normalized/vocabulary.json" in wire
    assert "data/staging/done/lesson.pdf.yaml" in wire
    assert "data/ledger.json" in wire
    assert "6" * 64 in wire
    assert "5" * 64 in wire
    assert "review, identity, deck, and coverage decisions" in wire
    assert adapter._agent_plans["4" * 64].plan is plan


def test_reviewed_card_revision_promotion_routes_to_the_same_finish_aggregate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    instruction = "Apply and finish this reviewed card revision."
    promotion = _FakePromotionPlan()
    promotion.proposal_kind = "card_revision"
    finish = _FakeCardRevisionFinishPlan(
        config.root,
        instruction=instruction,
        review=None,
        promotion=promotion,
    )
    finish.build.output_revision = "8" * 64
    finish.build.output_identity = (11, 12)
    intent = _agent_intent(
        kind="promote_staging",
        resource_ids=("resource_revision",),
        instruction=instruction,
    )
    calls: list[tuple[str, str, str]] = []

    def plan_promotion(
        _config: ProjectConfig,
        resource_id: str,
        received_instruction: str,
    ) -> Any:
        calls.append(("promotion", resource_id, received_instruction))
        return promotion

    def plan_finish(
        _config: ProjectConfig,
        resource_id: str,
        received_instruction: str,
    ) -> Any:
        calls.append(("finish", resource_id, received_instruction))
        return finish

    monkeypatch.setattr(
        assistant_adapter.assistant_promotion,
        "plan_promotion_action",
        plan_promotion,
    )
    monkeypatch.setattr(
        assistant_adapter.card_revision_finish,
        "plan_card_revision_finish",
        plan_finish,
    )
    monkeypatch.setattr(
        assistant_adapter.card_revision_finish,
        "CardRevisionFinishPlan",
        _FakeCardRevisionFinishPlan,
    )

    reply = adapter._prepare_agent_intent(
        config,
        result=_agent_result(intents=(intent,)),
        deck_scope="",
    )

    assert calls == [
        ("promotion", "resource_revision", instruction),
        ("finish", "resource_revision", instruction),
    ]
    assert reply.action is not None
    assert reply.action.confirm_label == "Apply and finish"
    assert reply.action.progress_label == "Preparing finish"
    assert reply.action.target == "data/decks/lesson.yaml"
    rendered = "\n".join((*reply.action.effects, *reply.action.disclosures))
    assert "Exact card landing" in rendered
    assert "Create selected-card example audio" in rendered
    assert "Build 2 card(s) from 1 note(s)" in rendered
    assert "Replace only the exact existing package" in rendered
    assert "device/inode 11/12" in rendered
    assert adapter._agent_plans[finish.fingerprint].plan is finish


@pytest.mark.parametrize(
    ("kind", "resource_ids", "record_ids", "options", "message"),
    [
        ("review_staging", (), (), {"review_patterns": False}, "exactly one proposal"),
        (
            "review_staging",
            ("resource_proposal",),
            (),
            {"review_patterns": "yes"},
            "explicit true-or-false",
        ),
        ("promote_staging", (), (), {}, "exactly one reviewed proposal"),
        (
            "promote_staging",
            ("resource_proposal",),
            ("word:first",),
            {},
            "cannot carry inferred",
        ),
        (
            "promote_staging",
            ("resource_proposal",),
            (),
            {"review_patterns": True},
            "cannot carry inferred",
        ),
    ],
)
def test_review_and_promotion_refuse_invalid_targets_records_and_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    resource_ids: tuple[str, ...],
    record_ids: tuple[str, ...],
    options: dict[str, Any],
    message: str,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    intent = _agent_intent(
        kind=kind,
        resource_ids=resource_ids,
        record_ids=record_ids,
        instruction="Apply this exact owner decision.",
        options=options,
    )
    monkeypatch.setattr(
        assistant_adapter.assistant_staging_review,
        "plan_staging_review",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid review intent must refuse before planning"
        ),
    )
    monkeypatch.setattr(
        assistant_adapter.assistant_promotion,
        "plan_promotion_action",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid promotion intent must refuse before planning"
        ),
    )

    with pytest.raises(RevisionRefusal, match=message):
        adapter._prepare_agent_intent(
            config,
            result=_agent_result(intents=(intent,)),
            deck_scope="",
        )


@pytest.mark.parametrize("kind", ["review_staging", "promote_staging"])
def test_review_and_promotion_preserve_application_target_refusals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    intent = _agent_intent(
        kind=kind,
        resource_ids=("resource_card",),
        instruction="Use this invalid target.",
        options={"review_patterns": False} if kind == "review_staging" else {},
    )
    error = JankiError("the resource is not a compatible staging proposal")
    monkeypatch.setattr(
        assistant_adapter.assistant_staging_review,
        "plan_staging_review",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )
    monkeypatch.setattr(
        assistant_adapter.assistant_promotion,
        "plan_promotion_action",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )
    if kind == "review_staging":
        monkeypatch.setattr(
            assistant_adapter.assistant_context.AssistantContextBroker,
            "proposal_context",
            lambda *_args: SimpleNamespace(proposal_kind="source_extraction"),
        )

    with pytest.raises(RevisionRefusal, match="not a compatible staging proposal"):
        adapter._prepare_agent_intent(
            config,
            result=_agent_result(intents=(intent,)),
            deck_scope="",
        )


def test_staging_review_preserves_an_invalid_record_selection_refusal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    calls: list[tuple[str, tuple[str, ...], bool]] = []

    def refuse(
        _config: ProjectConfig,
        *,
        proposal_resource_id: str,
        record_ids: tuple[str, ...],
        review_patterns: bool,
    ) -> Any:
        calls.append((proposal_resource_id, record_ids, review_patterns))
        raise JankiError("record 'word:outside' is not in this staging proposal")

    monkeypatch.setattr(
        assistant_adapter.assistant_staging_review,
        "plan_staging_review",
        refuse,
    )
    monkeypatch.setattr(
        assistant_adapter.assistant_context.AssistantContextBroker,
        "proposal_context",
        lambda *_args: SimpleNamespace(proposal_kind="source_extraction"),
    )
    intent = _agent_intent(
        kind="review_staging",
        resource_ids=("resource_proposal",),
        record_ids=("word:outside",),
        instruction="Review this card.",
        options={"review_patterns": False},
    )

    with pytest.raises(RevisionRefusal, match="not in this staging proposal"):
        adapter._prepare_agent_intent(
            config,
            result=_agent_result(intents=(intent,)),
            deck_scope="",
        )
    assert calls == [("resource_proposal", ("word:outside",), False)]


def test_one_staging_review_confirmation_delegates_exact_plan_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    plan = _FakeStagingReviewPlan()
    instruction = "Approve this exact card and its grammar patterns."
    adapter._agent_plans[plan.fingerprint] = assistant_adapter._PreparedAgentAction(
        kind="review_staging",
        focus_scope="",
        instruction=instruction,
        target=plan.source_name,
        plan=plan,
    )
    monkeypatch.setattr(
        assistant_adapter.assistant_staging_review,
        "AssistantStagingReviewPlan",
        _FakeStagingReviewPlan,
    )
    delegated: list[Any] = []

    def execute(fresh: ProjectConfig, received: Any) -> Any:
        assert fresh.root == config.root
        delegated.append(received)
        return SimpleNamespace(
            plan=received,
            outcome=SimpleNamespace(
                accepted_record_ids=("word:食べる:たべる",),
                pattern_reviewed=True,
            ),
        )

    monkeypatch.setattr(
        assistant_adapter.assistant_staging_review,
        "execute_staging_review",
        execute,
    )
    confirmation = RevisionConfirmation(
        capability="one-use",
        deck_scope="",
        instruction=instruction,
        expected_fingerprint=plan.fingerprint,
        target=plan.source_name,
    )

    execution = adapter.consume_replan_and_execute(
        confirmation,
        progress=lambda _label: pytest.fail("review service reports no fake progress"),
    )

    assert delegated == [plan]
    assert execution.complete is True
    assert "word:食べる:たべる" in execution.message
    assert "grammar pattern set was also marked reviewed" in execution.message
    with pytest.raises(RevisionRefusal, match="missing, stale, already consumed"):
        adapter.consume_replan_and_execute(
            confirmation,
            progress=lambda _label: None,
        )
    assert delegated == [plan]


def test_staging_review_partial_write_surfaces_exact_landed_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    plan = _FakeStagingReviewPlan()
    instruction = "Approve this exact review."
    adapter._agent_plans[plan.fingerprint] = assistant_adapter._PreparedAgentAction(
        kind="review_staging",
        focus_scope="",
        instruction=instruction,
        target=plan.source_name,
        plan=plan,
    )
    monkeypatch.setattr(
        assistant_adapter.assistant_staging_review,
        "AssistantStagingReviewPlan",
        _FakeStagingReviewPlan,
    )
    outcome = SimpleNamespace(
        accepted_record_ids=("word:食べる:たべる",),
        pattern_reviewed=False,
    )
    error = assistant_adapter.assistant_staging_review.AssistantStagingReviewPartialError(
        "pattern store changed after card authority landed",
        outcome,
    )
    monkeypatch.setattr(
        assistant_adapter.assistant_staging_review,
        "execute_staging_review",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )

    with pytest.raises(RevisionRefusal) as caught:
        adapter.consume_replan_and_execute(
            RevisionConfirmation(
                capability="one-use",
                deck_scope="",
                instruction=instruction,
                expected_fingerprint=plan.fingerprint,
                target=plan.source_name,
            ),
            progress=lambda _label: None,
        )

    text = str(caught.value)
    assert "only partly completed" in text
    assert "card approvals word:食べる:たべる" in text
    assert "pattern review not saved" in text
    assert "pattern store changed after card authority landed" in text
    assert "Refresh before any retry" in text


def test_promotion_confirmation_delegates_exact_plan_and_surfaces_ledger_partial_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    plan = _FakePromotionPlan()
    instruction = "Promote this reviewed proposal."
    adapter._agent_plans[plan.fingerprint] = assistant_adapter._PreparedAgentAction(
        kind="promote_staging",
        focus_scope="",
        instruction=instruction,
        target=plan.source,
        plan=plan,
    )
    monkeypatch.setattr(
        assistant_adapter.assistant_promotion,
        "AssistantPromotionPlan",
        _FakePromotionPlan,
    )
    delegated: list[Any] = []

    def execute(
        fresh: ProjectConfig,
        received: Any,
        *,
        progress: Any,
    ) -> Any:
        assert fresh.root == config.root
        delegated.append(received)
        progress("Saving canonical cards")
        return SimpleNamespace(
            plan=received,
            result=SimpleNamespace(
                state="canonical_landed_ledger_pending",
                promoted_ids=("word:食べる:たべる",),
                archive_path=None,
                ledger_error="ledger replace failed after canonical save",
            ),
        )

    monkeypatch.setattr(
        assistant_adapter.assistant_promotion,
        "execute_promotion_action",
        execute,
    )
    progress: list[str] = []
    confirmation = RevisionConfirmation(
        capability="one-use",
        deck_scope="",
        instruction=instruction,
        expected_fingerprint=plan.fingerprint,
        target=plan.source,
    )

    execution = adapter.consume_replan_and_execute(
        confirmation,
        progress=progress.append,
    )

    assert delegated == [plan]
    assert progress == ["Saving canonical cards"]
    assert execution.complete is True
    assert "Promotion state canonical_landed_ledger_pending" in execution.message
    assert "Canonical card ids: word:食べる:たべる" in execution.message
    assert "Review archive: none" in execution.message
    assert "Ledger follow-up is incomplete" in execution.message
    assert "ledger replace failed after canonical save" in execution.message
    with pytest.raises(RevisionRefusal, match="missing, stale, already consumed"):
        adapter.consume_replan_and_execute(
            confirmation,
            progress=lambda _label: None,
        )
    assert delegated == [plan]


@pytest.mark.parametrize("kind", ["review_staging", "promote_staging"])
@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("target", "another.pdf"),
        ("expected_fingerprint", "0" * 64),
    ],
)
def test_review_and_promotion_confirmations_bind_target_and_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    field: str,
    replacement: str,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    plan: Any = (
        _FakeStagingReviewPlan() if kind == "review_staging" else _FakePromotionPlan()
    )
    target = plan.source_name if kind == "review_staging" else plan.source
    instruction = "Apply this exact owner decision."
    adapter._agent_plans[plan.fingerprint] = assistant_adapter._PreparedAgentAction(
        kind=kind,
        focus_scope="",
        instruction=instruction,
        target=target,
        plan=plan,
    )
    monkeypatch.setattr(
        assistant_adapter.assistant_staging_review,
        "AssistantStagingReviewPlan",
        _FakeStagingReviewPlan,
    )
    monkeypatch.setattr(
        assistant_adapter.assistant_promotion,
        "AssistantPromotionPlan",
        _FakePromotionPlan,
    )
    monkeypatch.setattr(
        assistant_adapter.assistant_staging_review,
        "execute_staging_review",
        lambda *_args, **_kwargs: pytest.fail("tampered review must not execute"),
    )
    monkeypatch.setattr(
        assistant_adapter.assistant_promotion,
        "execute_promotion_action",
        lambda *_args, **_kwargs: pytest.fail("tampered promotion must not execute"),
    )
    values = {
        "capability": "one-use",
        "deck_scope": "",
        "instruction": instruction,
        "expected_fingerprint": plan.fingerprint,
        "target": target,
    }
    values[field] = replacement

    with pytest.raises(RevisionRefusal, match="stale|matches the rendered"):
        adapter.consume_replan_and_execute(
            RevisionConfirmation(**values),
            progress=lambda _label: None,
        )


def test_audio_action_card_renders_exact_counts_providers_writes_and_billing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    action_plan = _FakeAssistantActionPlan("generate_audio")
    intent = _agent_intent(
        kind="generate_audio",
        instruction="Generate every missing clip for this deck.",
    )
    calls: list[Any] = []

    def plan_action(fresh: ProjectConfig, received: Any) -> Any:
        assert fresh.root == config.root
        calls.append(received)
        return action_plan

    monkeypatch.setattr(
        assistant_adapter.assistant_actions,
        "plan_action",
        plan_action,
    )

    reply = adapter._prepare_agent_intent(
        config,
        result=_agent_result(intents=(intent,)),
        deck_scope="",
    )

    assert calls == [intent]
    assert reply.action is not None
    assert reply.action.request_fingerprint == "a" * 64
    assert reply.action.target == "data/decks/vocabulary.yaml"
    assert reply.action.confirm_label == "Generate this exact audio"
    assert reply.action.progress_label == "Preparing audio"
    wire = "\n".join((*reply.action.effects, *reply.action.disclosures))
    assert "Words: 2 exact clip(s) — 1 current, 0 recoverable" in wire
    assert "Examples: 4 exact clip(s) — 1 current, 2 recoverable" in wire
    assert "voicevox · local-network · voice 13 · speed 0.75" in wire
    assert "openai-realtime · paid-network · voice cedar · speed 0.8" in wire
    assert "data/media/audio" in wire
    assert "data/ledger.json" in wire
    assert "may make 1 paid audio provider call(s)" in wire
    assert "separately journaled before dispatch" in wire
    assert "Clip classes: words and examples (deck default)" in wire
    assert "Force regeneration is off" in wire
    assert "Pruning is off" in wire
    assert "audio/first.wav" in wire
    assert '"text": "話す"' in wire
    assert "content SHA-256 " + "1" * 64 in wire
    assert "audio/second.wav" in wire
    assert "recovery SHA-256 " + "3" * 64 in wire
    assert adapter._agent_plans["a" * 64].plan is action_plan


def test_audio_action_card_names_force_replacements_and_exact_prune_deletions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    action_plan = _FakeAssistantActionPlan(
        "generate_audio",
        audio_force=True,
        audio_prune=True,
    )
    intent = _agent_intent(
        kind="generate_audio",
        instruction=(
            "Regenerate this audio and prune repository-wide unreferenced clips."
        ),
        options={"audio_force": True, "audio_prune": True},
    )
    monkeypatch.setattr(
        assistant_adapter.assistant_actions,
        "plan_action",
        lambda _fresh, received: (
            action_plan
            if received is intent
            else pytest.fail("audio action planned another intent")
        ),
    )

    reply = adapter._prepare_agent_intent(
        config,
        result=_agent_result(intents=(intent,)),
        deck_scope="",
        owner_message=intent.instruction,
    )

    assert reply.action is not None
    wire = "\n".join((*reply.action.effects, *reply.action.disclosures))
    assert "Force regeneration is on" in wire
    assert "data/media/audio/janki-first.wav" in wire
    assert "content SHA-256 " + "f" * 64 in wire
    assert "Pruning is on for repository-wide unreferenced janki audio" in wire
    assert "Permanently delete: data/media/audio/janki-old.wav" in wire
    assert "content SHA-256 " + "e" * 64 in wire
    assert "Remove ledger audio references: [\"janki-old.wav\"]" in wire
    assert "destructive" in wire.casefold()


@pytest.mark.parametrize(
    ("option", "required_word"),
    [
        ("audio_force", "regenerate"),
        ("audio_prune", "prune"),
    ],
)
def test_audio_destructive_options_require_current_owner_words_before_planning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    option: str,
    required_word: str,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    intent = _agent_intent(
        kind="generate_audio",
        instruction="Generate missing audio.",
        options={option: True},
    )
    monkeypatch.setattr(
        assistant_adapter.assistant_actions,
        "plan_action",
        lambda *_args, **_kwargs: pytest.fail(
            "an untrusted destructive option must refuse before action planning"
        ),
    )

    with pytest.raises(RevisionRefusal, match=required_word):
        adapter._prepare_agent_intent(
            config,
            result=_agent_result(intents=(intent,)),
            deck_scope="",
            owner_message="Please generate the missing audio.",
        )


def test_build_action_card_renders_exact_inputs_output_and_locality(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    action_plan = _FakeAssistantActionPlan("build_deck")
    intent = _agent_intent(
        kind="build_deck",
        instruction="Build the current package.",
    )
    monkeypatch.setattr(
        assistant_adapter.assistant_actions,
        "plan_action",
        lambda fresh, received: (
            action_plan
            if fresh.root == config.root and received is intent
            else pytest.fail("build action planned another target")
        ),
    )

    reply = adapter._prepare_agent_intent(
        config,
        result=_agent_result(intents=(intent,)),
        deck_scope="data/decks/potential.yaml",
    )

    assert reply.action is not None
    assert reply.action.request_fingerprint == "b" * 64
    assert reply.action.target == "data/decks/potential.yaml"
    assert reply.action.confirm_label == "Build this exact deck"
    assert reply.action.progress_label == "Preparing build"
    wire = "\n".join(reply.action.effects)
    assert "Build all 16 card(s)" in wire
    assert f"Configured deck: data/decks/potential.yaml · SHA-256 {'9' * 64}" in wire
    assert f"Canonical cards: data/normalized/vocabulary.json · SHA-256 {'c' * 64}" in wire
    assert f"Recognition front: templates/front.html · SHA-256 {'d' * 64}" in wire
    assert f"Recognition back: templates/back.html · SHA-256 {'e' * 64}" in wire
    assert f"Example audio: data/media/audio/example.wav · SHA-256 {'f' * 64}" in wire
    assert (
        f"Replace only the existing package at dist/potential.apkg: SHA-256 "
        f"{'8' * 64} · device/inode 23/47"
    ) in wire
    assert "Write export history to data/ledger.json" in wire
    assert "Write the Anki package to dist/potential.apkg" in wire
    assert reply.action.disclosures == (
        "This is a local build and makes no model or audio-provider call.",
        "Janki re-reads and hashes every bound input before accepting the package.",
    )
    assert adapter._agent_plans["b" * 64].plan is action_plan


def test_build_action_card_names_an_output_that_must_remain_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    action_plan = _FakeAssistantActionPlan("build_deck")
    action_plan.projection["writes"]["package_precondition"] = {
        "state": "absent",
        "sha256": None,
        "identity": None,
    }
    intent = _agent_intent(
        kind="build_deck",
        instruction="Build the current package.",
    )
    monkeypatch.setattr(
        assistant_adapter.assistant_actions,
        "plan_action",
        lambda _fresh, _received: action_plan,
    )

    reply = adapter._prepare_agent_intent(
        config,
        result=_agent_result(intents=(intent,)),
        deck_scope="",
    )

    assert reply.action is not None
    assert (
        "Require dist/potential.apkg to remain absent until this build publishes it"
        in reply.action.effects
    )


def test_create_deck_explicit_options_render_the_exact_local_plan_card(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    creation_plan = _FakeDeckCreationPlan()
    intent = _agent_intent(
        kind="create_deck",
        resource_ids=(),
        instruction="Create a deck for this week's review.",
        options={
            "deck_name": "Weekly Review",
            "card_directions": ["recognition", "reading"],
        },
    )
    planned: list[Any] = []

    def plan_creation(fresh: ProjectConfig, request: Any) -> Any:
        assert fresh.root == config.root
        planned.append(request)
        return creation_plan

    monkeypatch.setattr(
        assistant_adapter.assistant_deck_creation,
        "plan_deck_creation",
        plan_creation,
    )

    reply = adapter._prepare_agent_intent(
        config,
        result=_agent_result(intents=(intent,)),
        deck_scope="",
        owner_message=(
            "Create a deck named Weekly Review with recognition and reading cards."
        ),
    )

    assert len(planned) == 1
    request = planned[0]
    assert request.name == "Weekly Review"
    assert request.recognition is True
    assert request.production is False
    assert request.reading is True
    assert request.instruction == "Create a deck for this week's review."
    assert reply.action is not None
    assert reply.action.request_fingerprint == "6" * 64
    assert reply.action.target == "data/decks/weekly-review.yaml"
    assert reply.action.confirm_label == "Create this exact deck"
    assert reply.action.progress_label == "Preparing deck"
    assert reply.action.effects == (
        "Create the configured study deck 'Weekly Review'",
        "Enable card directions: recognition, reading",
        "Assign Anki deck id 1702000001 and intake tag deck:weekly-review",
        "Reserve its future package path at dist/weekly-review.apkg",
        f"Bind the current configured-deck set: SHA-256 {'8' * 64}",
        (
            f"Write this exact UTF-8 YAML (SHA-256 {'7' * 64}):\n"
            "deck:\n"
            "  name: Weekly Review\n"
            "  cards:\n"
            "    recognition: true\n"
            "    production: false\n"
            "    reading: true\n"
        ),
    )
    assert reply.action.disclosures == (
        "This local action makes no model or audio-provider call.",
        "The new deck definition is written once; existing decks and cards are unchanged.",
    )
    assert adapter._agent_plans["6" * 64].plan is creation_plan


@pytest.mark.parametrize(
    ("directions", "message"),
    [
        ([], "Choose at least one"),
        (["recognition", "listening"], "Choose at least one"),
        (["recognition", "recognition"], "unique card directions"),
    ],
    ids=("missing", "extra", "duplicate"),
)
def test_create_deck_refuses_missing_extra_or_duplicate_directions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    directions: list[str],
    message: str,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    intent = _agent_intent(
        kind="create_deck",
        resource_ids=(),
        instruction="Create this deck.",
        options={
            "deck_name": "Weekly Review",
            "card_directions": directions,
        },
    )
    monkeypatch.setattr(
        assistant_adapter.assistant_deck_creation,
        "plan_deck_creation",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid directions must refuse before deck planning"
        ),
    )

    with pytest.raises(RevisionRefusal, match=message):
        adapter._prepare_agent_intent(
            config,
            result=_agent_result(intents=(intent,)),
            deck_scope="",
        )


@pytest.mark.parametrize(
    ("owner_message", "message"),
    [
        (
            "Create a deck with recognition and reading cards.",
            "deck name must appear verbatim",
        ),
        (
            "Create a deck named Weekly Review with recognition cards.",
            "explicitly name every requested card direction",
        ),
    ],
    ids=("model-invented-name", "model-invented-direction"),
)
def test_create_deck_refuses_owner_only_values_invented_by_the_model_before_planning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    owner_message: str,
    message: str,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    intent = _agent_intent(
        kind="create_deck",
        resource_ids=(),
        instruction="Create a deck for this week's review.",
        options={
            "deck_name": "Weekly Review",
            "card_directions": ["recognition", "reading"],
        },
    )
    monkeypatch.setattr(
        assistant_adapter.assistant_deck_creation,
        "plan_deck_creation",
        lambda *_args, **_kwargs: pytest.fail(
            "model-invented owner values must refuse before deck planning"
        ),
    )

    with pytest.raises(RevisionRefusal, match=message):
        adapter._prepare_agent_intent(
            config,
            result=_agent_result(intents=(intent,)),
            deck_scope="",
            owner_message=owner_message,
        )

    assert adapter._agent_plans == {}


def test_one_create_deck_confirmation_delegates_the_exact_plan_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    creation_plan = _FakeDeckCreationPlan()
    instruction = "Create a deck for this week's review."
    intent = _agent_intent(
        kind="create_deck",
        resource_ids=(),
        instruction=instruction,
        options={
            "deck_name": "Weekly Review",
            "card_directions": ["recognition", "reading"],
        },
    )
    monkeypatch.setattr(
        assistant_adapter.assistant_deck_creation,
        "plan_deck_creation",
        lambda *_args, **_kwargs: creation_plan,
    )
    monkeypatch.setattr(
        assistant_adapter.assistant_deck_creation,
        "AssistantDeckCreationPlan",
        _FakeDeckCreationPlan,
    )
    reply = adapter._prepare_agent_intent(
        config,
        result=_agent_result(intents=(intent,)),
        deck_scope="",
        owner_message=(
            "Create a deck named Weekly Review with recognition and reading cards."
        ),
    )
    assert reply.action is not None
    delegated: list[Any] = []

    def execute(fresh: ProjectConfig, received: Any) -> Any:
        assert fresh.root == config.root
        delegated.append(received)
        return SimpleNamespace(
            plan=received,
            result=SimpleNamespace(path=config.deck_dir / "weekly-review.yaml"),
        )

    monkeypatch.setattr(
        assistant_adapter.assistant_deck_creation,
        "execute_deck_creation",
        execute,
    )
    confirmation = RevisionConfirmation(
        capability="one-use-capability",
        deck_scope="",
        instruction=instruction,
        expected_fingerprint=reply.action.request_fingerprint,
        target=reply.action.target,
    )

    execution = adapter.consume_replan_and_execute(
        confirmation,
        progress=lambda _label: pytest.fail("deck creation reports no fake progress"),
    )

    assert delegated == [creation_plan]
    assert execution.complete is True
    assert execution.message == (
        "Created 'Weekly Review' at data/decks/weekly-review.yaml. It is ready to "
        "receive explicitly assigned cards."
    )
    with pytest.raises(RevisionRefusal, match="missing, stale, already consumed"):
        adapter.consume_replan_and_execute(
            confirmation,
            progress=lambda _label: None,
        )
    assert delegated == [creation_plan]


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("target", "data/decks/another.yaml"),
        ("expected_fingerprint", "0" * 64),
    ],
)
def test_create_deck_confirmation_binds_target_and_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: str,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    creation_plan = _FakeDeckCreationPlan()
    instruction = "Create a deck for this week's review."
    adapter._agent_plans[creation_plan.fingerprint] = assistant_adapter._PreparedAgentAction(
        kind="create_deck",
        focus_scope="",
        instruction=instruction,
        target="data/decks/weekly-review.yaml",
        plan=creation_plan,
    )
    monkeypatch.setattr(
        assistant_adapter.assistant_deck_creation,
        "AssistantDeckCreationPlan",
        _FakeDeckCreationPlan,
    )
    monkeypatch.setattr(
        assistant_adapter.assistant_deck_creation,
        "execute_deck_creation",
        lambda *_args, **_kwargs: pytest.fail(
            "a tampered creation action must not execute"
        ),
    )
    values = {
        "capability": "one-use-capability",
        "deck_scope": "",
        "instruction": instruction,
        "expected_fingerprint": creation_plan.fingerprint,
        "target": "data/decks/weekly-review.yaml",
    }
    values[field] = replacement

    with pytest.raises(RevisionRefusal, match="stale|matches the rendered"):
        adapter.consume_replan_and_execute(
            RevisionConfirmation(**values),
            progress=lambda _label: None,
        )


def test_extract_source_intent_routes_one_opaque_source_to_the_existing_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    extraction = _agent_extraction_plan()
    intent = _agent_intent(
        kind="extract_source",
        resource_ids=("resource_source",),
        instruction="Extract this source into proposed cards.",
    )
    prepared_paths: list[Path] = []
    monkeypatch.setattr(
        assistant_adapter.assistant_context,
        "AssistantContextBroker",
        _FakeSourceBroker,
    )

    def prepare(_self: Any, *, source_path: Path) -> SourceExtractionPlan:
        prepared_paths.append(source_path)
        return extraction

    monkeypatch.setattr(
        assistant_adapter.RevisionAssistantAdapter,
        "prepare_source_extraction",
        prepare,
    )

    reply = adapter._prepare_agent_intent(
        config,
        result=_agent_result(intents=(intent,)),
        deck_scope="",
    )

    assert prepared_paths == [_FakeSourceBroker.source]
    assert reply.action is not None
    assert reply.action.request_fingerprint == "5" * 64
    assert reply.action.target == "data/staging/lesson.pdf.yaml"
    assert reply.action.effects == extraction.effects
    assert reply.action.disclosures == extraction.disclosures
    assert reply.action.confirm_label == extraction.confirm_label
    assert reply.action.progress_label == "Preparing pages"
    assert reply.action_instruction == "Extract this source into proposed cards."
    assert adapter._agent_plans["5" * 64].plan is extraction


@pytest.mark.parametrize(
    "intent",
    [
        _agent_intent(
            kind="extract_source",
            resource_ids=("resource_card",),
            instruction="Extract this.",
        ),
        _agent_intent(
            kind="extract_source",
            resource_ids=("resource_source",),
            record_ids=("word:first",),
            instruction="Extract this.",
        ),
        _agent_intent(
            kind="extract_source",
            resource_ids=("resource_source",),
            instruction="Extract this.",
            options={"mode": "table"},
        ),
    ],
    ids=("card-target", "card-ids", "inferred-options"),
)
def test_extract_source_refuses_non_source_targets_cards_and_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    intent: Any,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    monkeypatch.setattr(
        assistant_adapter.assistant_context,
        "AssistantContextBroker",
        _FakeSourceBroker,
    )
    monkeypatch.setattr(
        assistant_adapter.RevisionAssistantAdapter,
        "prepare_source_extraction",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid source intent must refuse before extraction planning"
        ),
    )

    with pytest.raises(
        RevisionRefusal,
        match="Unknown Assistant source resource|exactly one preserved source resource",
    ):
        adapter._prepare_agent_intent(
            config,
            result=_agent_result(intents=(intent,)),
            deck_scope="",
        )


def test_generic_extraction_confirmation_delegates_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    extraction = _agent_extraction_plan()
    instruction = "Extract this source into proposed cards."
    adapter._agent_plans[extraction.request_fingerprint] = (
        assistant_adapter._PreparedAgentAction(
            kind="extract_source",
            focus_scope="",
            instruction=instruction,
            target=extraction.target,
            plan=extraction,
        )
    )
    delegated: list[Any] = []

    def consume(
        _self: Any,
        confirmation: SourceExtractionConfirmation,
        *,
        progress: Any,
    ) -> SourceExtractionExecution:
        delegated.append(confirmation)
        progress("Preparing pages")
        return SourceExtractionExecution(
            message="Extraction saved 12 card proposal(s) for owner review."
        )

    monkeypatch.setattr(
        assistant_adapter.RevisionAssistantAdapter,
        "consume_replan_and_extract",
        consume,
    )
    confirmation = RevisionConfirmation(
        capability="one-use-capability",
        deck_scope="",
        instruction=instruction,
        expected_fingerprint=extraction.request_fingerprint,
        target=extraction.target,
    )
    progress: list[str] = []

    execution = adapter.consume_replan_and_execute(
        confirmation,
        progress=progress.append,
    )

    assert delegated == [
        SourceExtractionConfirmation(
            preparation_id="source-preparation",
            source_name="lesson.pdf",
            expected_fingerprint="5" * 64,
        )
    ]
    assert progress == ["Preparing pages"]
    assert execution.message == "Extraction saved 12 card proposal(s) for owner review."
    assert execution.complete is True
    with pytest.raises(RevisionRefusal, match="missing, stale, already consumed"):
        adapter.consume_replan_and_execute(
            confirmation,
            progress=lambda _label: None,
        )
    assert len(delegated) == 1


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("target", "data/staging/another.yaml"),
        ("expected_fingerprint", "0" * 64),
    ],
)
def test_generic_extraction_confirmation_binds_target_and_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: str,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    extraction = _agent_extraction_plan()
    instruction = "Extract this source into proposed cards."
    adapter._agent_plans[extraction.request_fingerprint] = (
        assistant_adapter._PreparedAgentAction(
            kind="extract_source",
            focus_scope="",
            instruction=instruction,
            target=extraction.target,
            plan=extraction,
        )
    )
    monkeypatch.setattr(
        assistant_adapter.RevisionAssistantAdapter,
        "consume_replan_and_extract",
        lambda *_args, **_kwargs: pytest.fail(
            "a tampered generic extraction must not dispatch"
        ),
    )
    values = {
        "capability": "one-use-capability",
        "deck_scope": "",
        "instruction": instruction,
        "expected_fingerprint": extraction.request_fingerprint,
        "target": extraction.target,
    }
    values[field] = replacement

    with pytest.raises(RevisionRefusal, match="stale|matches the rendered"):
        adapter.consume_replan_and_execute(
            RevisionConfirmation(**values),
            progress=lambda _label: None,
        )


@pytest.mark.parametrize("kind", ["generate_audio", "build_deck"])
def test_one_action_confirmation_delegates_the_exact_plan_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    action_plan = _FakeAssistantActionPlan(kind)
    instruction = "Generate the exact audio." if kind == "generate_audio" else "Build it."
    intent = _agent_intent(kind=kind, instruction=instruction)
    monkeypatch.setattr(
        assistant_adapter.assistant_actions,
        "plan_action",
        lambda *_args, **_kwargs: action_plan,
    )
    monkeypatch.setattr(
        assistant_adapter.assistant_actions,
        "AssistantActionPlan",
        _FakeAssistantActionPlan,
    )
    reply = adapter._prepare_agent_intent(
        config,
        result=_agent_result(intents=(intent,)),
        deck_scope="",
    )
    assert reply.action is not None
    delegated: list[Any] = []

    def execute(
        fresh: ProjectConfig,
        received: Any,
        *,
        progress: Any,
    ) -> Any:
        assert fresh.root == config.root
        delegated.append(received)
        progress("Executing exact action")
        if kind == "build_deck":
            result = SimpleNamespace(
                output_path=config.dist_dir / "potential.apkg",
                card_count=16,
                package_sha256="9" * 64,
            )
        else:
            result = SimpleNamespace(
                state="complete",
                file_count=4,
                up_to_date=2,
                warnings=("One local voice used its configured fallback.",),
                pending_recovery=False,
                stopped_by=None,
            )
        return SimpleNamespace(plan=received, result=result)

    monkeypatch.setattr(
        assistant_adapter.assistant_actions,
        "execute_action",
        execute,
    )
    confirmation = RevisionConfirmation(
        capability="one-use-capability",
        deck_scope="",
        instruction=instruction,
        expected_fingerprint=reply.action.request_fingerprint,
        target=reply.action.target,
    )
    progress: list[str] = []

    result = adapter.consume_replan_and_execute(
        confirmation,
        progress=progress.append,
    )

    assert delegated == [action_plan]
    assert progress == ["Executing exact action"]
    assert result.complete is True
    if kind == "build_deck":
        assert "Built 16 card(s) at dist/potential.apkg" in result.message
        assert "9" * 64 in result.message
    else:
        assert "Audio state complete: generated 4 clip(s)" in result.message
        assert "2 were already current" in result.message
        assert "configured fallback" in result.message
    with pytest.raises(RevisionRefusal, match="missing, stale, already consumed"):
        adapter.consume_replan_and_execute(
            confirmation,
            progress=lambda _label: None,
        )
    assert delegated == [action_plan]


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("target", "data/decks/other.yaml"),
        ("expected_fingerprint", "0" * 64),
    ],
)
def test_audio_action_confirmation_binds_target_and_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: str,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    action_plan = _FakeAssistantActionPlan("generate_audio")
    instruction = "Generate the exact audio."
    prepared = assistant_adapter._PreparedAgentAction(
        kind="generate_audio",
        focus_scope="",
        instruction=instruction,
        target="data/decks/vocabulary.yaml",
        plan=action_plan,
    )
    adapter._agent_plans[action_plan.fingerprint] = prepared
    monkeypatch.setattr(
        assistant_adapter.assistant_actions,
        "AssistantActionPlan",
        _FakeAssistantActionPlan,
    )
    monkeypatch.setattr(
        assistant_adapter.assistant_actions,
        "execute_action",
        lambda *_args, **_kwargs: pytest.fail("a tampered action must not execute"),
    )
    values = {
        "capability": "one-use-capability",
        "deck_scope": "",
        "instruction": instruction,
        "expected_fingerprint": action_plan.fingerprint,
        "target": "data/decks/vocabulary.yaml",
    }
    values[field] = replacement

    with pytest.raises(RevisionRefusal, match="stale|matches the rendered"):
        adapter.consume_replan_and_execute(
            RevisionConfirmation(**values),
            progress=lambda _label: None,
        )


def test_incomplete_audio_reports_durable_recovery_truth_and_never_claims_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config)
    action_plan = _FakeAssistantActionPlan("generate_audio")
    instruction = "Generate the exact audio."
    adapter._agent_plans[action_plan.fingerprint] = assistant_adapter._PreparedAgentAction(
        kind="generate_audio",
        focus_scope="",
        instruction=instruction,
        target="data/decks/vocabulary.yaml",
        plan=action_plan,
    )
    monkeypatch.setattr(
        assistant_adapter.assistant_actions,
        "AssistantActionPlan",
        _FakeAssistantActionPlan,
    )
    calls: list[Any] = []

    def execute(*_args: Any, **_kwargs: Any) -> Any:
        calls.append(action_plan)
        return SimpleNamespace(
            plan=action_plan,
            result=SimpleNamespace(
                state="generation-stopped",
                file_count=3,
                up_to_date=1,
                warnings=(),
                pending_recovery=True,
                stopped_by="The paid provider connection closed after capture.",
            ),
        )

    monkeypatch.setattr(
        assistant_adapter.assistant_actions,
        "execute_action",
        execute,
    )

    with pytest.raises(RevisionRefusal) as caught:
        adapter.consume_replan_and_execute(
            RevisionConfirmation(
                capability="one-use-capability",
                deck_scope="",
                instruction=instruction,
                expected_fingerprint=action_plan.fingerprint,
                target="data/decks/vocabulary.yaml",
            ),
            progress=lambda _label: None,
        )

    text = str(caught.value)
    assert calls == [action_plan]
    assert "Audio state generation-stopped" in text
    assert "generated 3 clip(s); 1 were already current" in text
    assert "Paid bytes remain recoverable" in text
    assert "do not repeat the request blindly" in text
    assert "paid provider connection closed after capture" in text
    assert "complete" not in text.casefold()


def test_adapter_prepares_then_dispatches_one_exact_saved_source_only_after_click(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    source = config.scan_inbox / "lesson.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"%PDF")
    target = SimpleNamespace(
        source_sha256="a" * 64,
        staging_path=config.staging_dir / "lesson.yaml",
        patterns_path=config.patterns_file,
        provenance={"request_fingerprint": "b" * 64},
    )
    consent = SimpleNamespace(
        sendable=True,
        target=target,
        refusal="",
        busy="",
        name="lesson.pdf",
        model="claude-opus-5",
        mode=None,
        replacement_revision=None,
        replaces=None,
        replaces_cards=0,
        replaces_state="",
        replaces_grammar="",
        sends_known_words=False,
    )
    monkeypatch.setattr(
        assistant_adapter,
        "describe_extraction",
        lambda fresh, path, *, mode: (
            consent
            if fresh.root == config.root and path == source and mode is None
            else pytest.fail("adapter described a different source")
        ),
    )
    dispatched: list[Any] = []

    def dispatch(fresh: Any, expected: Any, *, progress: Any) -> Any:
        assert fresh.root == config.root
        dispatched.append(expected)
        for label in (
            "Preparing pages",
            "Reading the source",
            "Checking the answer's shape",
            "Saving proposals",
        ):
            progress(label)
        return SimpleNamespace(
            target=config.staging_dir / "lesson.yaml",
            records=12,
        )

    monkeypatch.setattr(assistant_adapter, "dispatch_extraction", dispatch)
    adapter = _adapter(config)

    plan = adapter.prepare_source_extraction(source_path=source)

    assert dispatched == []
    assert plan.source_name == "lesson.pdf"
    assert "whole lesson.pdf" in " ".join(plan.effects)
    assert "page ranges" in " ".join(plan.effects)
    assert "paid Anthropic API call" in " ".join(plan.disclosures)
    assert "Claude Pro or Max does not pay" in " ".join(plan.disclosures)
    assert plan.replaces is False
    assert plan.confirm_label == "Send lesson.pdf using claude-opus-5 — paid API call"

    seen_progress: list[str] = []
    result = adapter.consume_replan_and_extract(
        SourceExtractionConfirmation(
            preparation_id=plan.preparation_id,
            source_name=plan.source_name,
            expected_fingerprint=plan.request_fingerprint,
        ),
        progress=seen_progress.append,
    )

    assert len(dispatched) == 1
    expected = dispatched[0]
    assert expected.source == source
    assert expected.request_fingerprint == "b" * 64
    assert expected.source_sha256 == "a" * 64
    assert expected.replacement_confirmed is False
    assert expected.staging_path == (config.staging_dir / "lesson.yaml").resolve()
    assert expected.patterns_path == config.patterns_file.resolve()
    assert expected.operations_path == config.operations_file.resolve()
    assert seen_progress == [
        "Preparing pages",
        "Reading the source",
        "Checking the answer's shape",
        "Saving proposals",
    ]
    assert "12 card proposal(s)" in result.message

    with pytest.raises(RevisionRefusal, match="already used"):
        adapter.consume_replan_and_extract(
            SourceExtractionConfirmation(
                preparation_id=plan.preparation_id,
                source_name=plan.source_name,
                expected_fingerprint=plan.request_fingerprint,
            ),
            progress=lambda _label: None,
        )
    assert len(dispatched) == 1


def test_adapter_replacement_button_is_the_only_event_that_grants_force(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    source = config.scan_inbox / "lesson.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"%PDF")
    revision_snapshot = ExtractionRevision(
        staging_sha256="c" * 64,
        pattern_entry_sha256="d" * 64,
        pattern_reviewed=True,
        pattern_has_patterns=True,
    )
    consent = SimpleNamespace(
        sendable=True,
        target=SimpleNamespace(
            source_sha256="a" * 64,
            staging_path=config.staging_dir / "lesson.yaml",
            patterns_path=config.patterns_file,
            provenance={"request_fingerprint": "b" * 64},
        ),
        refusal="",
        busy="",
        name="lesson.pdf",
        model="claude-opus-5",
        mode=None,
        replacement_revision=revision_snapshot,
        replaces=config.staging_dir / "lesson.yaml",
        replaces_cards=18,
        replaces_state="Cards need edits",
        replaces_grammar="Grammar reviewed",
        sends_known_words=False,
    )
    monkeypatch.setattr(
        assistant_adapter,
        "describe_extraction",
        lambda _config, _path, *, mode: consent,
    )
    dispatched: list[Any] = []
    monkeypatch.setattr(
        assistant_adapter,
        "dispatch_extraction",
        lambda _config, expected, *, progress: (
            dispatched.append(expected)
            or SimpleNamespace(target=config.staging_dir / "lesson.yaml", records=18)
        ),
    )
    adapter = _adapter(config, tmp_path / "data" / "decks" / "potential.yaml")

    plan = adapter.prepare_source_extraction(source_path=source)

    assert plan.replaces is True
    assert plan.confirm_label.startswith("Replace the named review")
    plan_text = " ".join(plan.effects)
    assert "18 cards, Cards need edits" in plan_text
    assert "Grammar reviewed" in plan_text
    assert "not recoverable" in plan_text
    with adapter._plan_lock:
        rendered = adapter._extraction_expectations[plan.preparation_id]
    assert rendered.replacement_revision == revision_snapshot
    assert rendered.replacement_confirmed is False

    adapter.consume_replan_and_extract(
        SourceExtractionConfirmation(
            preparation_id=plan.preparation_id,
            source_name=plan.source_name,
            expected_fingerprint=plan.request_fingerprint,
        ),
        progress=lambda _label: None,
    )

    assert len(dispatched) == 1
    assert dispatched[0].replacement_revision == revision_snapshot
    assert dispatched[0].replacement_confirmed is True


@pytest.mark.parametrize(
    ("phase", "expected_text"),
    [
        ("binding", "no paid request was made"),
        ("preparation", "no paid request was made"),
        ("authorization", "may contain unused authority"),
    ],
)
def test_adapter_preserves_each_pre_dispatch_failure_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    expected_text: str,
) -> None:
    adapter, confirmation, _config = _seeded_extraction_confirmation(tmp_path)
    error = ExtractionDispatchError(JankiError("pre-dispatch refusal"), phase=phase)

    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(assistant_adapter, "dispatch_extraction", refuse)

    with pytest.raises(RevisionRefusal) as caught:
        adapter.consume_replan_and_extract(
            confirmation,
            progress=lambda _label: None,
        )

    text = str(caught.value).casefold()
    assert expected_text in text
    assert "nothing was sent" in text
    if phase == "authorization":
        assert "janki operations" in text


@pytest.mark.parametrize(
    ("failure", "expected_text"),
    [
        (
            DispatchFailure("operation-saved", ANSWER_SAVED),
            "operations --show-reply operation-saved",
        ),
        (
            DispatchFailure("operation-empty", ANSWER_EMPTY),
            "captured provider reply contains no answer",
        ),
        (
            DispatchFailure("operation-unavailable", ANSWER_UNAVAILABLE),
            "exact recovery bytes are unavailable",
        ),
        (
            DispatchFailure("operation-forgotten", FORGOTTEN),
            "already forgotten",
        ),
        (
            DispatchFailure("operation-cleanup", FORGOTTEN, cleanup_pending=True),
            "operations --forget operation-cleanup",
        ),
        (
            DispatchFailure(
                "operation-unknown",
                OUTCOME_UNKNOWN,
                money_may_have_been_spent=True,
            ),
            "retry may pay twice",
        ),
    ],
)
def test_adapter_preserves_each_dispatched_recovery_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: DispatchFailure,
    expected_text: str,
) -> None:
    adapter, confirmation, _config = _seeded_extraction_confirmation(tmp_path)
    error = ExtractionDispatchError(
        JankiError("provider refused"),
        phase="dispatch",
        operation_id=failure.operation_id,
        failure=failure,
    )

    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(assistant_adapter, "dispatch_extraction", refuse)

    with pytest.raises(RevisionRefusal) as caught:
        adapter.consume_replan_and_extract(
            confirmation,
            progress=lambda _label: None,
        )

    text = str(caught.value).casefold()
    assert expected_text in text
    assert failure.operation_id in text
    assert "may have been billed" in text


def test_adapter_preserves_a_dispatch_journal_failure_without_inviting_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, confirmation, _config = _seeded_extraction_confirmation(tmp_path)
    error = ExtractionDispatchError(
        JankiError("provider connection ended"),
        phase="dispatch",
        operation_id="operation-journal",
        journal_error=operations.OperationError("journal unreadable"),
    )

    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(assistant_adapter, "dispatch_extraction", refuse)

    with pytest.raises(RevisionRefusal) as caught:
        adapter.consume_replan_and_extract(
            confirmation,
            progress=lambda _label: None,
        )

    text = str(caught.value).casefold()
    assert "operation-journal" in text
    assert "journal unreadable" in text
    assert "could not settle" in text
    assert "do not retry" in text


def test_adapter_reports_saved_proposals_after_pattern_store_completion_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, confirmation, config = _seeded_extraction_confirmation(tmp_path)
    staging_path = config.staging_dir / "lesson.yaml"
    cause = ExtractionCompletionError(JankiError("pattern store refused"), staging_path)
    error = ExtractionDispatchError(
        cause,
        phase="completion",
        operation_id="operation-completion",
    )

    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(assistant_adapter, "dispatch_extraction", refuse)

    with pytest.raises(RevisionRefusal) as caught:
        adapter.consume_replan_and_extract(
            confirmation,
            progress=lambda _label: None,
        )

    text = str(caught.value)
    assert str(staging_path) in text
    assert "proposals are saved" in text
    assert "do not repeat extraction" in text


def test_adapter_refuses_blind_retry_after_final_operation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, confirmation, _config = _seeded_extraction_confirmation(tmp_path)
    error = ExtractionDispatchError(
        operations.OperationError("final journal save refused"),
        phase="completion",
        operation_id="operation-final-save",
    )

    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(assistant_adapter, "dispatch_extraction", refuse)

    with pytest.raises(RevisionRefusal) as caught:
        adapter.consume_replan_and_extract(
            confirmation,
            progress=lambda _label: None,
        )

    text = str(caught.value).casefold()
    assert "operation-final-save" in text
    assert "could not prove" in text
    assert "do not retry" in text
    assert "janki operations" in text


def test_assistant_page_does_not_render_provider_disclosure(tmp_path: Path) -> None:
    config_path = tmp_path / "janki.toml"
    config_path.write_text(
        '[assistant]\nenabled = true\nprovider = "claude-code"\nmodel = "claude-opus-5"\n',
        encoding="utf-8",
    )
    config = ProjectConfig.load(tmp_path)
    adapter = _adapter(config, config.deck_dir / "potential.yaml")
    sidecar = create_assistant_sidecar(
        adapter,
        deck_choices=adapter.deck_choices,
        session_token="assistant-session-token-000000000000",
    )
    sidecar.start()
    try:
        connection = http.client.HTTPConnection(
            "127.0.0.1",
            sidecar.server.server_address[1],
            timeout=3,
        )
        connection.request("GET", sidecar.server.shell_path)
        response = connection.getresponse()
        body = response.read().decode("utf-8")
        connection.close()

        assert response.status == 200
        assert "Ask uses claude-code" not in body
        assert "Claude Pro/Max subscription" not in body
        assert "claude-opus-5" not in body
        assert "data/assistant" not in body
    finally:
        sidecar.close()


def test_adapter_renders_the_same_confirmation_shape_for_anthropic_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deck = tmp_path / "data" / "decks" / "potential.yaml"
    adapter = _adapter(_config(tmp_path), deck)
    application_plan = _revision_plan(
        provider="anthropic-api",
        billing_display="Anthropic API billing",
        auth_metadata={
            "auth_method": "environment-api-key",
            "api_key_source": "ANTHROPIC_API_KEY",
        },
        transport={"kind": "anthropic-messages-api"},
    )
    monkeypatch.setattr(
        assistant_adapter.revision,
        "plan_revision",
        lambda *_args, **_kwargs: application_plan,
    )

    rendered = _prepare_deck_revision(
        adapter,
        deck_scope="data/decks/potential.yaml",
        instruction="Add examples.",
    )
    wire = "\n".join((*rendered.effects, *rendered.disclosures))

    assert "anthropic-api" in wire
    assert "Billing: Anthropic API billing" in wire
    assert "Authentication: auth method=environment-api-key" in wire
    assert "api key source=ANTHROPIC_API_KEY" in wire
    assert "Claude Code version" not in wire
    assert "Provider request identity: narrow-provider-request-fingerprint" in wire


def test_adapter_reloads_the_on_disk_provider_before_rendering_a_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "janki.toml"
    config_path.write_text(
        '[ai]\nrevise_provider = "claude-code"\nrevise_model = "claude-opus-5"\n',
        encoding="utf-8",
    )
    config = ProjectConfig.load(tmp_path)
    adapter = _adapter(config, config.deck_dir / "potential.yaml")
    config_path.write_text(
        '[ai]\nrevise_provider = "anthropic-api"\nrevise_model = "claude-opus-5"\n',
        encoding="utf-8",
    )
    application_plan = _revision_plan(
        provider="anthropic-api",
        billing_display="Anthropic API billing",
        auth_metadata={
            "auth_method": "environment-api-key",
            "api_key_source": "ANTHROPIC_API_KEY",
        },
        transport={"kind": "anthropic-messages-api"},
    )
    observed: list[str] = []

    def fresh_plan(fresh_config: ProjectConfig, *_args: Any, **_kwargs: Any) -> Any:
        observed.append(fresh_config.revise_provider)
        return application_plan

    monkeypatch.setattr(assistant_adapter.revision, "plan_revision", fresh_plan)

    rendered = _prepare_deck_revision(
        adapter,
        deck_scope=_scope(adapter),
        instruction="Add examples.",
    )

    assert observed == ["anthropic-api"]
    wire = "\n".join((*rendered.effects, *rendered.disclosures))
    assert "Billing: Anthropic API billing" in wire
    assert "Authentication: auth method=environment-api-key" in wire


def test_adapter_refuses_a_stale_confirmation_before_run_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deck = tmp_path / "data" / "decks" / "potential.yaml"
    adapter = _adapter(_config(tmp_path), deck)
    application_plan = _revision_plan()
    monkeypatch.setattr(
        assistant_adapter.revision,
        "plan_revision",
        lambda *_args, **_kwargs: application_plan,
    )
    monkeypatch.setattr(
        assistant_adapter.revision,
        "run_revision",
        lambda *_args, **_kwargs: pytest.fail("a stale binding must not execute"),
    )
    _prepare_deck_revision(
        adapter,
        deck_scope=_scope(adapter),
        instruction="Add examples.",
    )

    with pytest.raises(RevisionRefusal, match="stale"):
        adapter.consume_replan_and_execute(
            RevisionConfirmation(
                capability="one-use",
                deck_scope=_scope(adapter),
                instruction="Add examples.",
                expected_fingerprint="different-plan",
                target="data/decks/potential.yaml",
            ),
            progress=lambda _label: None,
        )


def test_adapter_delegates_replan_and_dispatch_to_run_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deck = tmp_path / "data" / "decks" / "potential.yaml"
    adapter = _adapter(_config(tmp_path), deck)
    application_plan = _revision_plan()
    monkeypatch.setattr(
        assistant_adapter.revision,
        "plan_revision",
        lambda *_args, **_kwargs: application_plan,
    )
    runs: list[Any] = []

    def fake_run(_config: Any, expected: Any, *, progress: Any) -> Any:
        runs.append(expected)
        progress("Reading the source")
        progress("Checking the answer's shape")
        progress("Saving proposals")
        return SimpleNamespace(staging_path=tmp_path / "data" / "staging" / "proposal.json")

    monkeypatch.setattr(assistant_adapter.revision, "run_revision", fake_run)
    finish_plan = _finish_plan(tmp_path)
    monkeypatch.setattr(
        assistant_adapter.revision_finish,
        "plan_revision_finish",
        lambda _config, staging_path: (
            finish_plan
            if staging_path == finish_plan.revision.staging_path
            else pytest.fail("finish planned another staged proposal")
        ),
    )
    _prepare_deck_revision(
        adapter,
        deck_scope=_scope(adapter),
        instruction="Add examples.",
    )
    progress: list[str] = []

    result = adapter.consume_replan_and_execute(
        RevisionConfirmation(
            capability="one-use",
            deck_scope=_scope(adapter),
            instruction="Add examples.",
            expected_fingerprint="rendered-plan",
            target="data/decks/potential.yaml",
        ),
        progress=progress.append,
    )

    assert runs == [application_plan]
    assert progress == [
        "Reading the source",
        "Checking the answer's shape",
        "Saving proposals",
    ]
    assert "data/staging/proposal.json" in result.message
    assert result.finish is not None
    assert result.finish.request_fingerprint == "f" * 64
    assert result.finish.target == "data/decks/potential.yaml"
    assert result.finish.current_form_note == "Old potential note."
    assert result.finish.proposed_form_note == ("Potential expresses ability or possibility.")
    assert result.finish.records[0].record_id == "word:one"
    assert result.finish.records[0].proposed_examples[0].japanese == "明日は遊べます。"
    assert result.finish.audio_provider_required == 1
    assert result.finish.output_path == "dist/potential.apkg"
    assert result.finish.card_count == 16
    assert result.finish_unavailable is None


def test_staged_revision_survives_finish_planning_failure_without_inviting_rebill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deck = tmp_path / "data" / "decks" / "potential.yaml"
    adapter = _adapter(_config(tmp_path), deck)
    application_plan = _revision_plan()
    staging_path = tmp_path / "data" / "staging" / "paid-proposal.json"
    monkeypatch.setattr(
        assistant_adapter.revision,
        "plan_revision",
        lambda *_args, **_kwargs: application_plan,
    )
    monkeypatch.setattr(
        assistant_adapter.revision,
        "run_revision",
        lambda *_args, **_kwargs: SimpleNamespace(staging_path=staging_path),
    )
    monkeypatch.setattr(
        assistant_adapter.revision_finish,
        "plan_revision_finish",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(JankiError("build template is missing")),
    )
    _prepare_deck_revision(
        adapter,
        deck_scope=_scope(adapter),
        instruction="Add examples.",
    )

    result = adapter.consume_replan_and_execute(
        RevisionConfirmation(
            capability="revision-capability",
            deck_scope=_scope(adapter),
            instruction="Add examples.",
            expected_fingerprint="rendered-plan",
            target="data/decks/potential.yaml",
        ),
        progress=lambda _label: None,
    )

    assert result.finish is None
    assert "data/staging/paid-proposal.json" in result.message
    assert "Do not repeat the paid revision call" in result.message
    assert result.finish_unavailable is not None
    assert "build template is missing" in result.finish_unavailable
    assert "staged proposal remains the deliverable" in result.finish_unavailable
    assert adapter._finish_plans == {}


def test_adapter_replans_exact_finish_then_executes_shared_aggregate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config, config.deck_dir / "potential.yaml")
    expected = _finish_plan(tmp_path)
    with adapter._plan_lock:
        adapter._finish_plans["finish-preparation"] = expected
    planned: list[Path] = []

    def fresh_plan(_config: ProjectConfig, staging_path: Path) -> Any:
        planned.append(staging_path)
        return expected

    monkeypatch.setattr(
        assistant_adapter.revision_finish,
        "plan_revision_finish",
        fresh_plan,
    )
    executed: list[Any] = []

    def execute(_config: ProjectConfig, plan: Any, *, progress: Any) -> Any:
        executed.append(plan)
        for phase in (
            "Preparing finish",
            "Applying reviewed revision",
            "Creating example audio",
            "Building Anki package",
            "Saving finish receipt",
        ):
            progress(phase)
        return SimpleNamespace(
            receipt_id="f" * 64,
            state="complete",
            deck_path=config.deck_dir / "potential.yaml",
            output_path=tmp_path / "dist" / "potential.apkg",
            package_sha256="a" * 64,
            card_count=16,
        )

    monkeypatch.setattr(
        assistant_adapter.revision_finish,
        "execute_revision_finish",
        execute,
    )
    phases: list[str] = []

    result = adapter.consume_replan_and_finish(
        RevisionFinishConfirmation(
            preparation_id="finish-preparation",
            expected_fingerprint="f" * 64,
            target="data/decks/potential.yaml",
        ),
        progress=phases.append,
    )

    assert planned == [expected.revision.staging_path]
    assert executed == [expected]
    assert phases == [
        "Preparing finish",
        "Applying reviewed revision",
        "Creating example audio",
        "Building Anki package",
        "Saving finish receipt",
    ]
    assert result.state == "complete"
    assert result.receipt_id == "f" * 64
    assert result.output_path == "dist/potential.apkg"
    assert result.package_sha256 == "a" * 64


def test_adapter_consumes_finish_plan_and_refuses_staged_or_build_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config, config.deck_dir / "potential.yaml")
    expected = _finish_plan(tmp_path)
    changed = _finish_plan(tmp_path, fingerprint="e" * 64)
    with adapter._plan_lock:
        adapter._finish_plans["finish-preparation"] = expected
    monkeypatch.setattr(
        assistant_adapter.revision_finish,
        "plan_revision_finish",
        lambda *_args, **_kwargs: changed,
    )
    monkeypatch.setattr(
        assistant_adapter.revision_finish,
        "execute_revision_finish",
        lambda *_args, **_kwargs: pytest.fail("drift must not execute"),
    )
    confirmation = RevisionFinishConfirmation(
        preparation_id="finish-preparation",
        expected_fingerprint="f" * 64,
        target="data/decks/potential.yaml",
    )

    with pytest.raises(RevisionRefusal, match="changed after you reviewed") as drift:
        adapter.consume_replan_and_finish(
            confirmation,
            progress=lambda _label: None,
        )
    assert "Return to the Workbench" in str(drift.value)
    assert "reopen the existing staged proposal" in str(drift.value)
    assert "do not repeat the paid revise call" in str(drift.value)
    assert "prepare and review a fresh revision" not in str(drift.value)
    with pytest.raises(RevisionRefusal, match="already used"):
        adapter.consume_replan_and_finish(
            confirmation,
            progress=lambda _label: None,
        )


def test_adapter_reports_inspected_recovery_state_after_finish_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config, config.deck_dir / "potential.yaml")
    expected = _finish_plan(tmp_path)
    with adapter._plan_lock:
        adapter._finish_plans["finish-preparation"] = expected
    monkeypatch.setattr(
        assistant_adapter.revision_finish,
        "plan_revision_finish",
        lambda *_args, **_kwargs: expected,
    )
    monkeypatch.setattr(
        assistant_adapter.revision_finish,
        "execute_revision_finish",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(JankiError("audio provider disconnected")),
    )
    inspected: list[str] = []

    def inspect(_config: ProjectConfig, receipt_id: str) -> Any:
        inspected.append(receipt_id)
        return SimpleNamespace(
            receipt_id=receipt_id,
            state="revision_applied",
            deck_path=config.deck_dir / "potential.yaml",
            output_path=tmp_path / "dist" / "potential.apkg",
            package_sha256=None,
            card_count=None,
        )

    monkeypatch.setattr(
        assistant_adapter.revision_finish,
        "inspect_revision_finish",
        inspect,
    )

    result = adapter.consume_replan_and_finish(
        RevisionFinishConfirmation(
            preparation_id="finish-preparation",
            expected_fingerprint="f" * 64,
            target="data/decks/potential.yaml",
        ),
        progress=lambda _label: None,
    )

    assert inspected == ["f" * 64]
    assert result.state == "revision_applied"
    assert "audio provider disconnected" in result.message
    assert "reviewed revision is applied and archived" in result.message
    assert "Resume only receipt" in result.message


def test_adapter_reloads_the_on_disk_provider_before_confirmation_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "janki.toml"
    config_path.write_text(
        '[ai]\nrevise_provider = "claude-code"\nrevise_model = "claude-opus-5"\n',
        encoding="utf-8",
    )
    config = ProjectConfig.load(tmp_path)
    deck = config.deck_dir / "potential.yaml"
    adapter = _adapter(config, deck)
    application_plan = _revision_plan(provider="claude-code")
    monkeypatch.setattr(
        assistant_adapter.revision,
        "plan_revision",
        lambda *_args, **_kwargs: application_plan,
    )
    _prepare_deck_revision(
        adapter,
        deck_scope=_scope(adapter),
        instruction="Add examples.",
    )
    config_path.write_text(
        '[ai]\nrevise_provider = "anthropic-api"\nrevise_model = "claude-opus-5"\n',
        encoding="utf-8",
    )

    def refuse_switched_provider(
        fresh_config: ProjectConfig,
        _expected: Any,
        *,
        progress: Any,
    ) -> Any:
        del progress
        assert fresh_config.revise_provider == "anthropic-api"
        raise revision.RevisionApplicationError(
            "[revision-request-stale] the configured provider changed; nothing was sent"
        )

    monkeypatch.setattr(
        assistant_adapter.revision,
        "run_revision",
        refuse_switched_provider,
    )

    with pytest.raises(RevisionRefusal, match="request-stale"):
        adapter.consume_replan_and_execute(
            RevisionConfirmation(
                capability="one-use",
                deck_scope=_scope(adapter),
                instruction="Add examples.",
                expected_fingerprint="rendered-plan",
                target="data/decks/potential.yaml",
            ),
            progress=lambda _label: None,
        )

    assert not config.operations_file.exists()
    assert not config.staging_dir.exists()


def test_serve_starts_assistant_first_and_closes_it_after_main_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    events: list[str] = []

    class FakeAssistant:
        url = "http://127.0.0.1:41001/assistant-token/"

        def close(self) -> None:
            events.append("assistant-close")

    assistant = FakeAssistant()

    def start(_config: Any) -> FakeAssistant:
        events.append("assistant-start")
        return assistant

    monkeypatch.setattr(workbench_server, "_start_assistant_for", start)
    session = SimpleNamespace(token="main-workbench-token")

    def open_session(_config: Any, *, assistant_url: str) -> Any:
        events.append("session-open")
        assert assistant_url == assistant.url
        return session

    monkeypatch.setattr(workbench_server.WorkbenchSession, "open", staticmethod(open_session))

    class FakeMainServer:
        expected_host = "127.0.0.1:41002"

        def serve_forever(self) -> None:
            events.append("main-serve")
            raise KeyboardInterrupt

        def server_close(self) -> None:
            events.append("main-close")

    monkeypatch.setattr(workbench_server, "make_server", lambda _session: FakeMainServer())

    url = workbench_server.serve(_config(tmp_path), open_browser=False)

    assert url == "http://127.0.0.1:41002/main-workbench-token/"
    assert events == [
        "assistant-start",
        "session-open",
        "main-serve",
        "main-close",
        "assistant-close",
    ]
    output = capsys.readouterr().out
    assert f"Workbench: {url}" in output
    assert f"Assistant: {assistant.url}" in output
    assert "main-workbench-token" not in assistant.url


def test_expected_application_refusal_becomes_chatkit_refusal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter(
        _config(tmp_path),
        tmp_path / "data" / "decks" / "potential.yaml",
    )
    monkeypatch.setattr(
        assistant_adapter.revision,
        "plan_revision",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            revision.RevisionApplicationError("nothing was sent")
        ),
    )

    with pytest.raises(RevisionRefusal, match="nothing was sent"):
        _prepare_deck_revision(
            adapter,
            deck_scope=_scope(adapter),
            instruction="Add examples.",
        )


def test_plan_deck_revision_surfaces_any_local_janki_refusal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter(
        _config(tmp_path),
        tmp_path / "data" / "decks" / "potential.yaml",
    )
    monkeypatch.setattr(
        assistant_adapter.revision,
        "plan_revision",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            JankiError("prompt unavailable; no provider was contacted")
        ),
    )

    with pytest.raises(RevisionRefusal, match="no provider was contacted"):
        _prepare_deck_revision(
            adapter,
            deck_scope=_scope(adapter),
            instruction="Add examples.",
        )


def test_execute_revision_surfaces_any_local_janki_refusal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deck = tmp_path / "data" / "decks" / "potential.yaml"
    adapter = _adapter(_config(tmp_path), deck)
    application_plan = _revision_plan()
    monkeypatch.setattr(
        assistant_adapter.revision,
        "plan_revision",
        lambda *_args, **_kwargs: application_plan,
    )
    monkeypatch.setattr(
        assistant_adapter.revision,
        "run_revision",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            JankiError("missing provider key; no provider was contacted")
        ),
    )
    _prepare_deck_revision(
        adapter,
        deck_scope=_scope(adapter),
        instruction="Add examples.",
    )

    with pytest.raises(RevisionRefusal, match="no provider was contacted"):
        adapter.consume_replan_and_execute(
            RevisionConfirmation(
                capability="one-use",
                deck_scope=_scope(adapter),
                instruction="Add examples.",
                expected_fingerprint="rendered-plan",
                target="data/decks/potential.yaml",
            ),
            progress=lambda _label: None,
        )
