"""ChatKit controller for explicitly selected, thread-local deck work.

This module deliberately contains no model client. Every composer message goes
only to the injected read-only chat callback. A separate one-use action may turn
that exact message into a revision plan, whose confirmation remains bound to a
fresh application-side re-plan and ``request_fingerprint`` comparison.

``chatkit`` is imported only by :func:`create_assistant_core`.  Importing the
ordinary workbench therefore does not make ChatKit a runtime requirement when
the assistant sidecar is disabled.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, TypeVar

from japanese_anki.errors import JankiError

__all__ = [
    "AssistantDeckChoice",
    "AssistantCore",
    "AssistantRequestContext",
    "ChatReply",
    "RevisionCallbacks",
    "RevisionConfirmation",
    "RevisionExecution",
    "RevisionExampleReview",
    "RevisionFinishConfirmation",
    "RevisionFinishExecution",
    "RevisionFinishReview",
    "RevisionRecordReview",
    "RevisionPlan",
    "RevisionRefusal",
    "ScopedMemoryStore",
    "SourceExtractionConfirmation",
    "SourceExtractionExecution",
    "SourceExtractionPlan",
    "create_assistant_core",
]


_PREPARE_ACTION = "janki.revision.prepare"
_CONFIRM_ACTION = "janki.revision.confirm"
_FINISH_ACTION = "janki.revision.finish"
_EXTRACT_CONFIRM_ACTION = "janki.extraction.confirm"
_SELECT_DECK_ACTION = "janki.deck.select"
_SHOW_DECKS_MESSAGE = "Choose an active deck"
_CAPABILITIES_MESSAGE = "What can Janki help me do here?"
_CHANGES_HELP_MESSAGE = "Explain how I can prepare and confirm a deck change."
_SOURCE_HELP_MESSAGE = "Explain how to make cards here from an attached PDF or photo."
_ASSISTANT_SCOPE = "janki-project"
_ACTIVE_DECK_ID_KEY = "janki_active_deck_id"
_ACTIVE_DECK_SCOPE_KEY = "janki_active_deck_scope"
_SELECTION_EPOCH_KEY = "janki_deck_selection_epoch"
_FINGERPRINT_DISPLAY_CHARS = 32
_MAX_CHAT_HISTORY_ENTRIES = 12
_MAX_CHAT_HISTORY_BYTES = 24_000
_CHAT_PROGRESS_LABELS = frozenset(
    {
        "Preparing answer",
        "Writing answer",
        "Saving answer",
    }
)
_PROGRESS_LABELS = frozenset(
    {
        "Preparing revision",
        "Reading the source",
        "Checking the answer's shape",
        "Saving proposals",
    }
)
_FINISH_PROGRESS_LABELS = frozenset(
    {
        "Preparing finish",
        "Applying reviewed revision",
        "Creating example audio",
        "Building Anki package",
        "Saving finish receipt",
    }
)
_EXTRACTION_PROGRESS_LABELS = frozenset(
    {
        "Preparing pages",
        "Reading the source",
        "Checking the answer's shape",
        "Saving proposals",
    }
)
_T = TypeVar("_T")


def _bounded_chat_history(
    entries: list[tuple[str, str]] | tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    def wire_size(items: list[tuple[str, str]]) -> int:
        return len(
            json.dumps(
                items,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )

    bounded = [item for item in entries if wire_size([item]) <= _MAX_CHAT_HISTORY_BYTES][
        -_MAX_CHAT_HISTORY_ENTRIES:
    ]
    while bounded and wire_size(bounded) > _MAX_CHAT_HISTORY_BYTES:
        del bounded[0]
    return tuple(bounded)


@dataclass(frozen=True, slots=True)
class ChatReply:
    """One non-mutating assistant answer rendered as ordinary prose."""

    text: str


@dataclass(frozen=True, slots=True)
class AssistantDeckChoice:
    """One server-discovered deck shown in the thread-local selector.

    ``deck_id`` is the only target value accepted from the browser. ``scope``
    stays server-side and is passed to application callbacks only after the
    opaque id has been resolved through the immutable startup catalog.
    """

    deck_id: str
    label: str
    scope: str
    revision_supported: bool
    unavailable_reason: str | None = None


@dataclass(frozen=True, slots=True)
class RevisionPlan:
    """Exact, side-effect-free revision plan rendered before owner consent."""

    request_fingerprint: str
    target: str
    effects: tuple[str, ...]
    disclosures: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RevisionConfirmation:
    """The exact plan binding consumed by the confirmation action.

    ``consume_replan_and_execute`` must derive a fresh plan from ``deck_scope``
    and ``instruction`` and refuse unless its fingerprint equals
    ``expected_fingerprint``.  The sidecar consumes ``capability`` first, so a
    failed or disconnected attempt is never silently replayed.
    """

    capability: str
    deck_scope: str
    instruction: str
    expected_fingerprint: str
    target: str


@dataclass(frozen=True, slots=True)
class RevisionExampleReview:
    """One exact sentence value shown without interpreting its Japanese."""

    register: str
    japanese: str
    furigana: str
    english: str


@dataclass(frozen=True, slots=True)
class RevisionRecordReview:
    """Current and proposed examples for one exact selected record."""

    record_id: str
    current_examples: tuple[RevisionExampleReview, ...]
    proposed_examples: tuple[RevisionExampleReview, ...]


@dataclass(frozen=True, slots=True)
class RevisionFinishReview:
    """Exact reviewed bytes and aggregate consequences shown before authority."""

    preparation_id: str
    request_fingerprint: str
    target: str
    current_form_note: str
    proposed_form_note: str
    records: tuple[RevisionRecordReview, ...]
    audio_provider: str
    audio_model: str
    audio_access: str
    audio_total: int
    audio_current: int
    audio_recoverable: int
    audio_provider_required: int
    output_path: str
    card_count: int


@dataclass(frozen=True, slots=True)
class RevisionExecution:
    """Durable staged proposal plus its fresh exact finish review."""

    message: str
    finish: RevisionFinishReview | None
    finish_unavailable: str | None = None


@dataclass(frozen=True, slots=True)
class RevisionFinishConfirmation:
    """One server-held finish plan consumed by the owner's review action."""

    preparation_id: str
    expected_fingerprint: str
    target: str


@dataclass(frozen=True, slots=True)
class RevisionFinishExecution:
    """Truthful durable aggregate state after Apply and finish."""

    message: str
    receipt_id: str
    state: str
    target: str
    output_path: str
    package_sha256: str | None = None
    card_count: int | None = None


@dataclass(frozen=True, slots=True)
class SourceExtractionPlan:
    """Exact read-only extraction facts rendered after local source intake."""

    preparation_id: str
    source_name: str
    request_fingerprint: str
    target: str
    effects: tuple[str, ...]
    disclosures: tuple[str, ...]
    confirm_label: str
    replaces: bool = False


@dataclass(frozen=True, slots=True)
class SourceExtractionConfirmation:
    """One server-held plan binding consumed by the paid extraction click."""

    preparation_id: str
    source_name: str
    expected_fingerprint: str


@dataclass(frozen=True, slots=True)
class SourceExtractionExecution:
    """Durable result of a confirmed source extraction."""

    message: str


class RevisionRefusal(RuntimeError):
    """A safe refusal that may be shown to the owner without a server error."""


class RevisionCallbacks(Protocol):
    """Conversation and revision boundary used by the ChatKit controller."""

    def resolve_deck_selection(self, deck_id: str) -> AssistantDeckChoice:
        """Freshly validate one opaque startup-catalog choice without inference."""

    def chat(
        self,
        *,
        deck_scope: str,
        history: tuple[tuple[str, str], ...],
        message: str,
        progress: Callable[[str], None],
    ) -> ChatReply | Awaitable[ChatReply]:
        """Answer one message without planning or mutating a deck."""

    def prepare_revision(
        self,
        *,
        deck_scope: str,
        instruction: str,
    ) -> RevisionPlan | Awaitable[RevisionPlan]:
        """Return a fresh, read-only plan for the exact instruction."""

    def consume_replan_and_execute(
        self,
        confirmation: RevisionConfirmation,
        *,
        progress: Callable[[str], None],
    ) -> RevisionExecution | Awaitable[RevisionExecution]:
        """Re-plan, compare, authorize, stage, and prepare exact owner review."""

    def consume_replan_and_finish(
        self,
        confirmation: RevisionFinishConfirmation,
        *,
        progress: Callable[[str], None],
    ) -> RevisionFinishExecution | Awaitable[RevisionFinishExecution]:
        """Re-plan, compare, and execute one reviewed aggregate finish."""

    def prepare_source_extraction(
        self,
        *,
        source_path: Path,
    ) -> SourceExtractionPlan | Awaitable[SourceExtractionPlan]:
        """Describe one saved source without dispatching a provider call."""

    def consume_replan_and_extract(
        self,
        confirmation: SourceExtractionConfirmation,
        *,
        progress: Callable[[str], None],
    ) -> SourceExtractionExecution | Awaitable[SourceExtractionExecution]:
        """Re-plan, authorize, dispatch, and durably stage one extraction."""


class AssistantAttachmentStore(Protocol):
    """Local intake boundary used by the optional ChatKit attachment path."""

    def assert_ready(self, attachment_id: str) -> None:
        """Refuse unless the exact registered upload finished successfully."""

    def commit_attachment(self, attachment_id: str) -> Any:
        """Preserve the uploaded bytes in the immutable source inbox."""


@dataclass(frozen=True, slots=True)
class AssistantRequestContext:
    """Per-request scope passed through every ChatKit store operation."""

    deck_scope: str
    request_origin: str | None = None


@dataclass(frozen=True, slots=True)
class AssistantCore:
    """A ChatKit server and its project-scoped isolation context."""

    server: Any
    context: AssistantRequestContext
    store: ScopedMemoryStore
    attachment_store: AssistantAttachmentStore | None = None


@dataclass(frozen=True, slots=True)
class _PlanBinding:
    confirmation: RevisionConfirmation
    thread_id: str
    widget_item_id: str
    deck_id: str
    selection_epoch: int


@dataclass(frozen=True, slots=True)
class _MessageBinding:
    message: str
    thread_id: str
    widget_item_id: str
    deck_id: str
    deck_scope: str
    selection_epoch: int


@dataclass(frozen=True, slots=True)
class _ExtractionBinding:
    confirmation: SourceExtractionConfirmation
    thread_id: str
    widget_item_id: str


@dataclass(frozen=True, slots=True)
class _FinishBinding:
    confirmation: RevisionFinishConfirmation
    thread_id: str
    widget_item_id: str
    deck_id: str
    selection_epoch: int


@dataclass(frozen=True, slots=True)
class _SelectorBinding:
    thread_id: str
    widget_item_id: str
    selection_epoch: int
    deck_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class _ThreadSelection:
    choice: AssistantDeckChoice
    epoch: int


class ScopedMemoryStore:
    """Small in-memory ChatKit store that refuses cross-sidecar contexts.

    Threads are convenience state, not paid-call provenance. The injected
    application services own durable chat records, journaling, and staging.
    """

    _SCOPE_KEY = "janki_deck_scope"

    def __init__(
        self,
        deck_scope: str,
        *,
        attachment_store: AssistantAttachmentStore | None = None,
    ) -> None:
        if not deck_scope.strip():
            raise ValueError("deck_scope must not be blank")
        self.deck_scope = deck_scope
        self._threads: dict[str, Any] = {}
        self._items: dict[str, list[Any]] = {}
        self._attachments: dict[str, Any] = {}
        self._attachment_store = attachment_store
        self._lock: asyncio.Lock | None = None

    def _context_ok(self, context: AssistantRequestContext) -> None:
        if context.deck_scope != self.deck_scope:
            raise PermissionError("The ChatKit thread belongs to another deck scope.")

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    @staticmethod
    def _copy(value: _T) -> _T:
        copier = getattr(value, "model_copy", None)
        return copier(deep=True) if copier is not None else value

    @staticmethod
    def _page(data: list[Any], *, has_more: bool, after: str | None) -> Any:
        from chatkit.types import Page

        return Page(data=data, has_more=has_more, after=after)

    @staticmethod
    def _not_found(message: str) -> Exception:
        from chatkit.store import NotFoundError

        return NotFoundError(message)

    def generate_thread_id(self, context: AssistantRequestContext) -> str:
        self._context_ok(context)
        from chatkit.store import default_generate_id

        return default_generate_id("thread")

    def generate_item_id(
        self,
        item_type: Any,
        thread: Any,
        context: AssistantRequestContext,
    ) -> str:
        self._context_ok(context)
        self._assert_thread_scope(thread)
        from chatkit.store import default_generate_id

        return default_generate_id(item_type)

    def _assert_thread_scope(self, thread: Any) -> None:
        metadata = getattr(thread, "metadata", {})
        if metadata.get(self._SCOPE_KEY) == self.deck_scope:
            return
        saved = self._threads.get(getattr(thread, "id", ""))
        if saved is not None and saved.metadata.get(self._SCOPE_KEY) == self.deck_scope:
            return
        raise PermissionError("The ChatKit thread is not bound to this deck.")

    async def load_thread(
        self,
        thread_id: str,
        context: AssistantRequestContext,
    ) -> Any:
        self._context_ok(context)
        async with self._get_lock():
            thread = self._threads.get(thread_id)
            if thread is None:
                raise self._not_found(f"Unknown thread: {thread_id}")
            self._assert_thread_scope(thread)
            return self._copy(thread)

    async def save_thread(
        self,
        thread: Any,
        context: AssistantRequestContext,
    ) -> None:
        self._context_ok(context)
        thread.metadata = {**thread.metadata, self._SCOPE_KEY: self.deck_scope}
        saved = self._copy(thread)
        async with self._get_lock():
            self._threads[saved.id] = saved
            self._items.setdefault(saved.id, [])

    @staticmethod
    def _page_slice(
        values: list[Any],
        *,
        after: str | None,
        limit: int,
        order: str,
    ) -> tuple[list[Any], bool, str | None]:
        if limit < 1:
            raise ValueError("limit must be positive")
        if order not in {"asc", "desc"}:
            raise ValueError("order must be asc or desc")
        ordered = values if order == "asc" else list(reversed(values))
        start = 0
        if after is not None:
            positions = [index for index, item in enumerate(ordered) if item.id == after]
            if not positions:
                raise KeyError(after)
            start = positions[0] + 1
        selected = ordered[start : start + limit]
        has_more = start + len(selected) < len(ordered)
        cursor = selected[-1].id if has_more and selected else None
        return selected, has_more, cursor

    async def load_thread_items(
        self,
        thread_id: str,
        after: str | None,
        limit: int,
        order: str,
        context: AssistantRequestContext,
    ) -> Any:
        self._context_ok(context)
        async with self._get_lock():
            thread = self._threads.get(thread_id)
            if thread is None:
                raise self._not_found(f"Unknown thread: {thread_id}")
            self._assert_thread_scope(thread)
            try:
                selected, has_more, cursor = self._page_slice(
                    self._items[thread_id],
                    after=after,
                    limit=limit,
                    order=order,
                )
            except KeyError as error:
                raise self._not_found(f"Unknown item cursor: {error.args[0]}") from error
            return self._page(
                [self._copy(item) for item in selected],
                has_more=has_more,
                after=cursor,
            )

    async def load_threads(
        self,
        limit: int,
        after: str | None,
        order: str,
        context: AssistantRequestContext,
    ) -> Any:
        self._context_ok(context)
        async with self._get_lock():
            values = sorted(
                self._threads.values(),
                key=lambda thread: (thread.created_at, thread.id),
            )
            try:
                selected, has_more, cursor = self._page_slice(
                    values,
                    after=after,
                    limit=limit,
                    order=order,
                )
            except KeyError as error:
                raise self._not_found(f"Unknown thread cursor: {error.args[0]}") from error
            return self._page(
                [self._copy(thread) for thread in selected],
                has_more=has_more,
                after=cursor,
            )

    async def add_thread_item(
        self,
        thread_id: str,
        item: Any,
        context: AssistantRequestContext,
    ) -> None:
        self._context_ok(context)
        async with self._get_lock():
            if thread_id not in self._threads:
                raise self._not_found(f"Unknown thread: {thread_id}")
            if item.thread_id != thread_id:
                raise PermissionError("The item belongs to another thread.")
            if any(saved.id == item.id for saved in self._items[thread_id]):
                raise ValueError(f"Duplicate thread item: {item.id}")
            self._items[thread_id].append(self._copy(item))

    async def save_item(
        self,
        thread_id: str,
        item: Any,
        context: AssistantRequestContext,
    ) -> None:
        self._context_ok(context)
        async with self._get_lock():
            if thread_id not in self._threads:
                raise self._not_found(f"Unknown thread: {thread_id}")
            if item.thread_id != thread_id:
                raise PermissionError("The item belongs to another thread.")
            items = self._items[thread_id]
            for index, saved in enumerate(items):
                if saved.id == item.id:
                    items[index] = self._copy(item)
                    break
            else:
                items.append(self._copy(item))

    async def load_item(
        self,
        thread_id: str,
        item_id: str,
        context: AssistantRequestContext,
    ) -> Any:
        self._context_ok(context)
        async with self._get_lock():
            for item in self._items.get(thread_id, []):
                if item.id == item_id:
                    return self._copy(item)
        raise self._not_found(f"Unknown item: {item_id}")

    async def delete_thread(
        self,
        thread_id: str,
        context: AssistantRequestContext,
    ) -> None:
        self._context_ok(context)
        async with self._get_lock():
            if thread_id not in self._threads:
                raise self._not_found(f"Unknown thread: {thread_id}")
            del self._threads[thread_id]
            del self._items[thread_id]

    async def delete_thread_item(
        self,
        thread_id: str,
        item_id: str,
        context: AssistantRequestContext,
    ) -> None:
        self._context_ok(context)
        async with self._get_lock():
            items = self._items.get(thread_id)
            if items is None:
                raise self._not_found(f"Unknown thread: {thread_id}")
            for index, item in enumerate(items):
                if item.id == item_id:
                    del items[index]
                    return
        raise self._not_found(f"Unknown item: {item_id}")

    async def save_attachment(self, attachment: Any, context: AssistantRequestContext) -> None:
        self._context_ok(context)
        if self._attachment_store is None:
            raise PermissionError("Attachments are disabled for the workbench assistant.")
        saved = self._copy(attachment)
        async with self._get_lock():
            existing = self._attachments.get(saved.id)
            if (
                existing is not None
                and existing.thread_id is not None
                and saved.thread_id != existing.thread_id
            ):
                raise PermissionError("The attachment belongs to another thread.")
            self._attachments[saved.id] = saved

    async def load_attachment(
        self,
        attachment_id: str,
        context: AssistantRequestContext,
    ) -> Any:
        self._context_ok(context)
        if self._attachment_store is None:
            raise self._not_found(f"Attachments are disabled: {attachment_id}")
        async with self._get_lock():
            attachment = self._attachments.get(attachment_id)
            if attachment is None:
                raise self._not_found(f"Unknown attachment: {attachment_id}")
            saved = self._copy(attachment)
        self._attachment_store.assert_ready(attachment_id)
        return saved

    async def delete_attachment(
        self,
        attachment_id: str,
        context: AssistantRequestContext,
    ) -> None:
        self._context_ok(context)
        if self._attachment_store is None:
            raise PermissionError("Attachments are disabled for the workbench assistant.")
        async with self._get_lock():
            if attachment_id not in self._attachments:
                raise self._not_found(f"Unknown attachment: {attachment_id}")
            del self._attachments[attachment_id]


def _validate_chat_reply(reply: Any) -> ChatReply:
    if not isinstance(reply, ChatReply):
        raise TypeError("chat must return ChatReply")
    if not reply.text.strip():
        raise ValueError("The chat reply must not be blank.")
    return reply


def _validate_plan(plan: Any) -> RevisionPlan:
    if not isinstance(plan, RevisionPlan):
        raise TypeError("prepare_revision must return RevisionPlan")
    if not plan.request_fingerprint.strip():
        raise ValueError("The revision plan has no request fingerprint.")
    if not plan.target.strip():
        raise ValueError("The revision plan has no target.")
    if not plan.effects or any(not effect.strip() for effect in plan.effects):
        raise ValueError("The revision plan must name every intended effect.")
    if any(not disclosure.strip() for disclosure in plan.disclosures):
        raise ValueError("Revision-plan disclosures must not be blank.")
    return plan


def _validate_source_extraction_plan(plan: Any) -> SourceExtractionPlan:
    if not isinstance(plan, SourceExtractionPlan):
        raise TypeError("prepare_source_extraction must return SourceExtractionPlan")
    required = (
        plan.preparation_id,
        plan.source_name,
        plan.request_fingerprint,
        plan.target,
        plan.confirm_label,
    )
    if any(not value.strip() for value in required):
        raise ValueError("The source extraction plan is incomplete.")
    if not plan.effects or any(not effect.strip() for effect in plan.effects):
        raise ValueError("The source extraction plan must name every intended effect.")
    if any(not disclosure.strip() for disclosure in plan.disclosures):
        raise ValueError("Source extraction disclosures must not be blank.")
    if plan.replaces and "replace" not in plan.confirm_label.casefold():
        raise ValueError("A replacement extraction button must name the replacement.")
    return plan


def _validate_execution(result: Any) -> RevisionExecution:
    if not isinstance(result, RevisionExecution):
        raise TypeError("consume_replan_and_execute must return RevisionExecution")
    if not result.message.strip():
        raise ValueError("The revision result message must not be blank.")
    if result.finish is None:
        if result.finish_unavailable is None or not result.finish_unavailable.strip():
            raise ValueError("A staged revision without a finish review must explain why.")
    else:
        if result.finish_unavailable is not None:
            raise ValueError("A staged revision cannot be both reviewable and unavailable.")
        _validate_finish_review(result.finish)
    return result


def _staged_finish_unavailable(error: Exception) -> str:
    detail = str(error) or type(error).__name__
    return (
        "Janki could not render its Apply and finish review after the paid "
        f"revision was staged: {detail} The staged proposal remains the durable "
        "deliverable. Repair the local review path and reopen this exact proposal; "
        "do not repeat the paid revision call."
    )


def _is_lower_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _validate_review_example(
    example: Any,
    *,
    require_furigana: bool,
) -> RevisionExampleReview:
    if not isinstance(example, RevisionExampleReview):
        raise TypeError("Revision review examples must be RevisionExampleReview values.")
    required = (
        example.register,
        example.japanese,
        example.english,
    )
    if require_furigana:
        required = (*required, example.furigana)
    if any(not value.strip() for value in required):
        raise ValueError("Revision review examples must be complete.")
    return example


def _validate_finish_review(review: Any) -> RevisionFinishReview:
    if not isinstance(review, RevisionFinishReview):
        raise TypeError("The staged revision must include a RevisionFinishReview.")
    if any(
        not value.strip()
        for value in (
            review.preparation_id,
            review.request_fingerprint,
            review.target,
            review.audio_provider,
            review.audio_model,
            review.audio_access,
            review.output_path,
        )
    ):
        raise ValueError("The revision finish review is incomplete.")
    if not _is_lower_sha256(review.request_fingerprint):
        raise ValueError("The revision finish fingerprint is malformed.")
    if not review.records or any(
        not isinstance(item, RevisionRecordReview) for item in review.records
    ):
        raise ValueError("The revision finish review needs record values.")
    if len({item.record_id for item in review.records}) != len(review.records):
        raise ValueError("The revision finish review needs unique record ids.")
    for item in review.records:
        if not item.record_id.strip():
            raise ValueError("The revision finish record review is incomplete.")
        if len(item.current_examples) != 2 or len(item.proposed_examples) != 2:
            raise ValueError("Every revision finish record needs two exact examples.")
        for example in item.current_examples:
            _validate_review_example(example, require_furigana=False)
        for example in item.proposed_examples:
            _validate_review_example(example, require_furigana=True)
    counts = (
        review.audio_total,
        review.audio_current,
        review.audio_recoverable,
        review.audio_provider_required,
    )
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
        raise ValueError("Revision finish audio counts must be nonnegative integers.")
    if review.audio_total != sum(counts[1:]):
        raise ValueError("Revision finish audio counts do not add up.")
    if (
        isinstance(review.card_count, bool)
        or not isinstance(review.card_count, int)
        or review.card_count <= 0
    ):
        raise ValueError("Revision finish must name a positive package card count.")
    return review


def _validate_finish_execution(result: Any) -> RevisionFinishExecution:
    if not isinstance(result, RevisionFinishExecution):
        raise TypeError("consume_replan_and_finish must return RevisionFinishExecution")
    if any(
        not value.strip()
        for value in (
            result.message,
            result.receipt_id,
            result.state,
            result.target,
            result.output_path,
        )
    ):
        raise ValueError("The revision finish result is incomplete.")
    if not _is_lower_sha256(result.receipt_id):
        raise ValueError("The revision finish receipt id is malformed.")
    if result.state not in {
        "authorized",
        "revision_applied",
        "audio_complete",
        "complete",
    }:
        raise ValueError("The revision finish result has an unknown durable state.")
    if result.package_sha256 is not None and not _is_lower_sha256(result.package_sha256):
        raise ValueError("The revision finish package hash is malformed.")
    if result.card_count is not None and (
        isinstance(result.card_count, bool)
        or not isinstance(result.card_count, int)
        or result.card_count <= 0
    ):
        raise ValueError("The revision finish card count is malformed.")
    if result.state == "complete" and (result.package_sha256 is None or result.card_count is None):
        raise ValueError("A completed revision finish needs its package receipt.")
    return result


def _validate_source_extraction_execution(result: Any) -> SourceExtractionExecution:
    if not isinstance(result, SourceExtractionExecution):
        raise TypeError("consume_replan_and_extract must return SourceExtractionExecution")
    if not result.message.strip():
        raise ValueError("The extraction result message must not be blank.")
    return result


def _confirmation_body(
    *,
    title: str,
    target: str,
    effects: tuple[str, ...],
    disclosures: tuple[str, ...],
    fingerprint_label: str,
    fingerprint: str,
    one_use_note: str,
    instruction: str | None = None,
) -> dict[str, Any]:
    """Build one readable, full-width confirmation body without changing its facts."""

    sections: list[dict[str, Any]] = [
        {"type": "Title", "value": title},
        {
            "type": "Col",
            "gap": 1,
            "width": "100%",
            "minWidth": 0,
            "children": [
                {
                    "type": "Text",
                    "value": "Target",
                    "size": "sm",
                    "color": "secondary",
                    "weight": "semibold",
                },
                {
                    "type": "Text",
                    "value": target,
                    "weight": "semibold",
                    "width": "100%",
                },
            ],
        },
    ]
    if instruction is not None:
        sections.append(
            {
                "type": "Col",
                "gap": 1,
                "width": "100%",
                "minWidth": 0,
                "children": [
                    {
                        "type": "Text",
                        "value": "Your instruction",
                        "size": "sm",
                        "color": "secondary",
                        "weight": "semibold",
                    },
                    {"type": "Text", "value": instruction, "width": "100%"},
                ],
            }
        )
    effect_rows = [
        {
            "type": "Row",
            "gap": 2,
            "align": "start",
            "width": "100%",
            "children": [
                {
                    "type": "Text",
                    "value": "•",
                    "width": "1rem",
                    "color": "secondary",
                    "weight": "semibold",
                },
                {
                    "type": "Col",
                    "flex": 1,
                    "minWidth": 0,
                    "children": [{"type": "Text", "value": effect, "width": "100%"}],
                },
            ],
        }
        for effect in effects
    ]
    sections.extend(
        [
            {"type": "Divider", "spacing": 2},
            {
                "type": "Col",
                "gap": 2,
                "width": "100%",
                "minWidth": 0,
                "children": [
                    {
                        "type": "Text",
                        "value": "What confirming does",
                        "weight": "semibold",
                    },
                    {
                        "type": "Col",
                        "id": "confirmation-effects",
                        "gap": 3,
                        "width": "100%",
                        "minWidth": 0,
                        "children": effect_rows,
                    },
                ],
            },
        ]
    )
    if disclosures:
        sections.extend(
            [
                {"type": "Divider", "spacing": 2},
                {
                    "type": "Col",
                    "gap": 2,
                    "width": "100%",
                    "minWidth": 0,
                    "children": [
                        {
                            "type": "Text",
                            "value": "Before you confirm",
                            "weight": "semibold",
                        },
                        {
                            "type": "Col",
                            "id": "confirmation-disclosures",
                            "gap": 2,
                            "width": "100%",
                            "minWidth": 0,
                            "children": [
                                {
                                    "type": "Text",
                                    "value": disclosure,
                                    "color": "secondary",
                                    "width": "100%",
                                }
                                for disclosure in disclosures
                            ],
                        },
                    ],
                },
            ]
        )
    sections.extend(
        [
            {"type": "Divider", "spacing": 2},
            {
                "type": "Col",
                "gap": 1,
                "width": "100%",
                "minWidth": 0,
                "children": [
                    {
                        "type": "Text",
                        "value": fingerprint_label,
                        "size": "xs",
                        "color": "tertiary",
                        "weight": "semibold",
                    },
                    {
                        "type": "Col",
                        "id": "confirmation-fingerprint",
                        "gap": 0,
                        "width": "100%",
                        "minWidth": 0,
                        "children": [
                            {
                                "type": "Text",
                                "value": fingerprint[offset : offset + _FINGERPRINT_DISPLAY_CHARS],
                                "size": "xs",
                                "color": "tertiary",
                                "width": "100%",
                            }
                            for offset in range(
                                0,
                                len(fingerprint),
                                _FINGERPRINT_DISPLAY_CHARS,
                            )
                        ],
                    },
                    {
                        "type": "Text",
                        "value": one_use_note,
                        "size": "xs",
                        "color": "tertiary",
                        "width": "100%",
                    },
                ],
            },
        ]
    )
    return {
        "type": "Col",
        "gap": 4,
        "width": "100%",
        "minWidth": 0,
        "children": sections,
    }


def _review_examples_body(
    title: str,
    examples: tuple[RevisionExampleReview, ...],
) -> dict[str, Any]:
    return {
        "type": "Col",
        "gap": 2,
        "width": "100%",
        "minWidth": 0,
        "children": [
            {"type": "Text", "value": title, "weight": "semibold"},
            *[
                {
                    "type": "Col",
                    "gap": 1,
                    "width": "100%",
                    "minWidth": 0,
                    "children": [
                        {
                            "type": "Text",
                            "value": example.register.title(),
                            "size": "xs",
                            "color": "secondary",
                            "weight": "semibold",
                        },
                        {
                            "type": "Text",
                            "value": example.japanese,
                            "weight": "semibold",
                            "width": "100%",
                        },
                        {
                            "type": "Text",
                            "value": example.furigana,
                            "color": "secondary",
                            "width": "100%",
                        },
                        {
                            "type": "Text",
                            "value": example.english,
                            "width": "100%",
                        },
                    ],
                }
                for example in examples
            ],
        ],
    }


def _finish_review_body(review: RevisionFinishReview) -> dict[str, Any]:
    counts = (
        ("Total clips", review.audio_total),
        ("Already current", review.audio_current),
        ("Recoverable exact clips", review.audio_recoverable),
        ("Provider calls required", review.audio_provider_required),
    )
    phases = (
        "Preparing finish",
        "Applying reviewed revision",
        "Creating example audio",
        "Building Anki package",
        "Saving finish receipt",
    )
    children: list[dict[str, Any]] = [
        {"type": "Title", "value": "Review this exact revision"},
        {
            "type": "Col",
            "gap": 1,
            "width": "100%",
            "minWidth": 0,
            "children": [
                {
                    "type": "Text",
                    "value": "Target",
                    "size": "sm",
                    "color": "secondary",
                    "weight": "semibold",
                },
                {
                    "type": "Text",
                    "value": review.target,
                    "weight": "semibold",
                    "width": "100%",
                },
            ],
        },
        {"type": "Divider", "spacing": 2},
        {
            "type": "Col",
            "id": "revision-form-note-review",
            "gap": 2,
            "width": "100%",
            "minWidth": 0,
            "children": [
                {"type": "Text", "value": "Form note", "weight": "semibold"},
                {
                    "type": "Text",
                    "value": "Current",
                    "size": "xs",
                    "color": "secondary",
                    "weight": "semibold",
                },
                {
                    "type": "Text",
                    "value": review.current_form_note or "Empty",
                    "width": "100%",
                },
                {
                    "type": "Text",
                    "value": "Proposed",
                    "size": "xs",
                    "color": "secondary",
                    "weight": "semibold",
                },
                {
                    "type": "Text",
                    "value": review.proposed_form_note or "Empty",
                    "width": "100%",
                },
            ],
        },
    ]
    for record in review.records:
        children.extend(
            [
                {"type": "Divider", "spacing": 2},
                {
                    "type": "Col",
                    "gap": 3,
                    "width": "100%",
                    "minWidth": 0,
                    "children": [
                        {
                            "type": "Text",
                            "value": record.record_id,
                            "weight": "semibold",
                            "width": "100%",
                        },
                        _review_examples_body(
                            "Current polite/casual examples",
                            record.current_examples,
                        ),
                        _review_examples_body(
                            "Proposed polite/casual examples",
                            record.proposed_examples,
                        ),
                    ],
                },
            ]
        )
    children.extend(
        [
            {"type": "Divider", "spacing": 2},
            {
                "type": "Col",
                "id": "revision-finish-consequences",
                "gap": 2,
                "width": "100%",
                "minWidth": 0,
                "children": [
                    {
                        "type": "Text",
                        "value": "Audio and package consequences",
                        "weight": "semibold",
                    },
                    {
                        "type": "Text",
                        "value": (
                            f"Example audio: {review.audio_provider} · "
                            f"{review.audio_model} · {review.audio_access}"
                        ),
                        "width": "100%",
                    },
                    *[
                        {
                            "type": "Text",
                            "value": f"{label}: {value}",
                            "width": "100%",
                        }
                        for label, value in counts
                    ],
                    {
                        "type": "Text",
                        "value": (
                            "Only these reviewed examples are voiced. Existing "
                            "current clips are reused; unrelated media is not pruned."
                        ),
                        "color": "secondary",
                        "width": "100%",
                    },
                    {
                        "type": "Text",
                        "value": (
                            f"Anki package: {review.output_path} · cards: {review.card_count}"
                        ),
                        "width": "100%",
                    },
                ],
            },
            {"type": "Divider", "spacing": 2},
            {
                "type": "Col",
                "id": "revision-finish-phases",
                "gap": 2,
                "width": "100%",
                "minWidth": 0,
                "children": [
                    {
                        "type": "Text",
                        "value": "After one confirmation",
                        "weight": "semibold",
                    },
                    *[{"type": "Text", "value": f"• {phase}", "width": "100%"} for phase in phases],
                    {
                        "type": "Text",
                        "value": (
                            "No provider-invented percentage is shown. Interrupted "
                            "work keeps one exact durable finish receipt."
                        ),
                        "color": "secondary",
                        "width": "100%",
                    },
                ],
            },
            {"type": "Divider", "spacing": 2},
            {
                "type": "Col",
                "id": "revision-finish-fingerprint",
                "gap": 0,
                "width": "100%",
                "minWidth": 0,
                "children": [
                    {
                        "type": "Text",
                        "value": "Finish fingerprint",
                        "size": "xs",
                        "color": "tertiary",
                        "weight": "semibold",
                    },
                    *[
                        {
                            "type": "Text",
                            "value": review.request_fingerprint[
                                offset : offset + _FINGERPRINT_DISPLAY_CHARS
                            ],
                            "size": "xs",
                            "color": "tertiary",
                            "width": "100%",
                        }
                        for offset in range(
                            0,
                            len(review.request_fingerprint),
                            _FINGERPRINT_DISPLAY_CHARS,
                        )
                    ],
                    {
                        "type": "Text",
                        "value": (
                            "Apply and finish is one-use. At the click, janki "
                            "re-plans and refuses if the reviewed proposal, audio, "
                            "or build inputs changed."
                        ),
                        "size": "xs",
                        "color": "tertiary",
                        "width": "100%",
                    },
                ],
            },
        ]
    )
    return {
        "type": "Col",
        "gap": 4,
        "width": "100%",
        "minWidth": 0,
        "children": children,
    }


async def _call_callback(function: Callable[..., _T], *args: Any, **kwargs: Any) -> _T:
    if inspect.iscoroutinefunction(function):
        return await function(*args, **kwargs)
    result = await asyncio.to_thread(function, *args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


def create_assistant_core(
    callbacks: RevisionCallbacks,
    *,
    deck_choices: tuple[AssistantDeckChoice, ...],
    attachment_store: AssistantAttachmentStore | None = None,
) -> AssistantCore:
    """Create the project-scoped deterministic ChatKit server.

    Calling this function is the point at which the optional ``chatkit``
    dependency is imported.
    """

    from chatkit.server import ChatKitServer
    from chatkit.types import (
        AssistantMessageContent,
        AssistantMessageItem,
        ErrorEvent,
        NoticeEvent,
        ProgressUpdateEvent,
        StreamOptions,
        ThreadItemDoneEvent,
        WidgetItem,
    )
    from chatkit.widgets import DynamicWidgetRoot

    choices = tuple(deck_choices)
    if any(not isinstance(choice, AssistantDeckChoice) for choice in choices):
        raise TypeError("deck_choices must contain AssistantDeckChoice values")
    if any(
        not choice.deck_id.strip() or not choice.label.strip() or not choice.scope.strip()
        for choice in choices
    ):
        raise ValueError("Every assistant deck choice must be complete.")
    if len({choice.deck_id for choice in choices}) != len(choices):
        raise ValueError("Assistant deck choice ids must be unique.")
    if len({choice.scope for choice in choices}) != len(choices):
        raise ValueError("Assistant deck choice scopes must be unique.")
    if any(
        choice.revision_supported == bool(choice.unavailable_reason)
        for choice in choices
    ):
        raise ValueError(
            "Supported decks cannot have an unavailable reason, and unsupported "
            "decks must have one."
        )
    choices_by_id = {choice.deck_id: choice for choice in choices}

    store = ScopedMemoryStore(_ASSISTANT_SCOPE, attachment_store=attachment_store)
    context = AssistantRequestContext(deck_scope=_ASSISTANT_SCOPE)

    class _JankiChatKitServer(ChatKitServer[AssistantRequestContext]):
        def __init__(self) -> None:
            super().__init__(store=store, attachment_store=attachment_store)
            self._selectors: dict[str, _SelectorBinding] = {}
            self._messages: dict[str, _MessageBinding] = {}
            self._plans: dict[str, _PlanBinding] = {}
            self._finishes: dict[str, _FinishBinding] = {}
            self._extractions: dict[str, _ExtractionBinding] = {}
            self._histories: dict[str, list[tuple[str, str]]] = {}
            self._history_locks: dict[str, asyncio.Lock] = {}
            self._selection_locks: dict[str, asyncio.Lock] = {}
            self._busy_threads: set[str] = set()
            self._durable_tasks: set[asyncio.Task[Any]] = set()

        def _track_durable_task(
            self,
            thread_id: str,
            task: asyncio.Task[Any],
        ) -> Callable[[], None]:
            """Hold the thread busy through its final result and action bindings."""

            response_released = False
            self._busy_threads.add(thread_id)
            self._durable_tasks.add(task)

            def task_done(done: asyncio.Task[Any]) -> None:
                self._durable_tasks.discard(done)
                if response_released:
                    self._busy_threads.discard(thread_id)
                if not done.cancelled():
                    # Retrieve exceptions even when the browser disconnected and
                    # no response generator remains to await the durable task.
                    done.exception()

            task.add_done_callback(task_done)

            def release_response() -> None:
                nonlocal response_released
                response_released = True
                if task.done():
                    self._busy_threads.discard(thread_id)

            return release_response

        @staticmethod
        def _selection(thread: Any) -> _ThreadSelection | None:
            metadata = getattr(thread, "metadata", {})
            deck_id = metadata.get(_ACTIVE_DECK_ID_KEY)
            deck_scope = metadata.get(_ACTIVE_DECK_SCOPE_KEY)
            epoch = metadata.get(_SELECTION_EPOCH_KEY, 0)
            if deck_id is None and deck_scope is None and epoch == 0:
                return None
            if (
                not isinstance(deck_id, str)
                or not isinstance(deck_scope, str)
                or isinstance(epoch, bool)
                or not isinstance(epoch, int)
                or epoch < 1
            ):
                raise RevisionRefusal("This thread's active-deck state is invalid.")
            choice = choices_by_id.get(deck_id)
            if (
                choice is None
                or not choice.revision_supported
                or choice.scope != deck_scope
            ):
                raise RevisionRefusal("This thread's active deck is no longer available.")
            return _ThreadSelection(choice=choice, epoch=epoch)

        @staticmethod
        def _binding_is_current(
            selection: _ThreadSelection | None,
            *,
            deck_id: str,
            selection_epoch: int,
        ) -> bool:
            return (
                selection is not None
                and selection.choice.deck_id == deck_id
                and selection.epoch == selection_epoch
            )

        def _invalidate_thread_state(self, thread_id: str) -> None:
            self._histories.pop(thread_id, None)
            for bindings in (
                self._selectors,
                self._messages,
                self._plans,
                self._finishes,
            ):
                stale = [
                    capability
                    for capability, binding in bindings.items()
                    if binding.thread_id == thread_id
                ]
                for capability in stale:
                    del bindings[capability]

        @staticmethod
        def _local_help(message: str) -> str | None:
            if message == _CAPABILITIES_MESSAGE:
                return (
                    "Choose an active deck, ask read-only questions, or attach one "
                    "PDF or photo for local source intake. A supported deck can also "
                    "turn an exact message into a reviewed change proposal."
                )
            if message == _CHANGES_HELP_MESSAGE:
                return (
                    "Choose an active deck, send the requested change, then use the "
                    "explicit deck-change action under Janki's answer. Janki shows "
                    "one exact plan before any revision is sent."
                )
            if message == _SOURCE_HELP_MESSAGE:
                return (
                    "Use the attachment button to add one PDF or photo. Janki saves "
                    "it locally first, then shows a separate exact extraction plan; "
                    "the source is not sent unless you confirm that plan."
                )
            return None

        @staticmethod
        def _selector_widget(
            *,
            capability: str,
            active: _ThreadSelection | None,
        ) -> Any:
            rows: list[dict[str, Any]] = []
            for choice in choices:
                is_active = (
                    active is not None and active.choice.deck_id == choice.deck_id
                )
                if is_active:
                    badge = {
                        "type": "Badge",
                        "label": "Active",
                        "color": "success",
                        "variant": "soft",
                    }
                elif choice.revision_supported:
                    badge = {
                        "type": "Badge",
                        "label": "Chat + changes",
                        "color": "info",
                        "variant": "soft",
                    }
                else:
                    badge = {
                        "type": "Badge",
                        "label": "Unsupported",
                        "color": "secondary",
                        "variant": "soft",
                    }
                details = [
                    {
                        "type": "Text",
                        "value": choice.label,
                        "weight": "semibold",
                        "width": "100%",
                    },
                    badge,
                ]
                if choice.unavailable_reason:
                    details.append(
                        {
                            "type": "Text",
                            "value": choice.unavailable_reason,
                            "size": "sm",
                            "color": "secondary",
                            "width": "100%",
                        }
                    )
                row: dict[str, Any] = {
                    "type": "ListViewItem",
                    "gap": 3,
                    "align": "center",
                    "children": [
                        {
                            "type": "Box",
                            "direction": "column",
                            "gap": 1,
                            "flex": 1,
                            "minWidth": 0,
                            "children": details,
                        }
                    ],
                }
                if choice.revision_supported and not is_active:
                    row["onClickAction"] = {
                        "type": _SELECT_DECK_ACTION,
                        "payload": {
                            "capability": capability,
                            "deck_id": choice.deck_id,
                        },
                        "handler": "server",
                        "loadingBehavior": "container",
                        "streaming": True,
                    }
                rows.append(row)
            return DynamicWidgetRoot.model_validate(
                {
                    "type": "ListView",
                    "limit": "auto",
                    "status": {
                        "text": "Choose the active deck",
                        "icon": "book-open",
                    },
                    "children": rows,
                }
            )

        def _selector_event(
            self,
            thread: Any,
            request_context: AssistantRequestContext,
        ) -> Any:
            try:
                active = self._selection(thread)
            except RevisionRefusal:
                # A selector is the recovery surface for stale catalog metadata.
                # Old deck actions are unusable while this state is unselected;
                # the next valid selection invalidates them before resetting its
                # epoch. Project-scoped extraction remains independent.
                metadata = dict(getattr(thread, "metadata", {}))
                metadata.pop(_ACTIVE_DECK_ID_KEY, None)
                metadata.pop(_ACTIVE_DECK_SCOPE_KEY, None)
                metadata.pop(_SELECTION_EPOCH_KEY, None)
                thread.metadata = metadata
                active = None
            capability = secrets.token_urlsafe(32)
            item_id = store.generate_item_id("message", thread, request_context)
            self._selectors[capability] = _SelectorBinding(
                thread_id=thread.id,
                widget_item_id=item_id,
                selection_epoch=active.epoch if active is not None else 0,
                deck_ids=frozenset(
                    choice.deck_id for choice in choices if choice.revision_supported
                ),
            )
            return ThreadItemDoneEvent(
                item=WidgetItem(
                    id=item_id,
                    thread_id=thread.id,
                    created_at=datetime.now(),
                    widget=self._selector_widget(
                        capability=capability,
                        active=active,
                    ),
                    copy_text=None,
                )
            )

        def get_stream_options(self, thread: Any, request_context: Any) -> Any:
            del thread, request_context
            return StreamOptions(allow_cancel=False)

        def _message_event(self, thread: Any, text: str) -> Any:
            return ThreadItemDoneEvent(
                item=AssistantMessageItem(
                    id=store.generate_item_id("message", thread, context),
                    thread_id=thread.id,
                    created_at=datetime.now(),
                    content=[AssistantMessageContent(text=text)],
                )
            )

        @staticmethod
        def _message_input(message: Any) -> tuple[str, tuple[Any, ...]]:
            if message is None:
                raise RevisionRefusal("Send one message to begin.")
            attachments = tuple(message.attachments)
            if len(attachments) > 1:
                raise RevisionRefusal("Attach one source at a time.")
            if attachments and attachment_store is None:
                raise RevisionRefusal("Attachments are disabled on this assistant.")
            if message.quoted_text and message.quoted_text.strip():
                raise RevisionRefusal("Quoted context is disabled; send one direct message.")
            content = tuple(message.content)
            if len(content) > 1:
                raise RevisionRefusal("Send at most one plain-text message.")
            text = ""
            if content:
                part = content[0]
                if getattr(part, "type", None) != "input_text":
                    raise RevisionRefusal("Only a plain-text message is accepted.")
                text = part.text.strip()
            if not attachments and not text:
                raise RevisionRefusal("The message must not be blank.")
            return text, attachments

        @staticmethod
        def _plan_widget(
            plan: RevisionPlan,
            *,
            instruction: str,
            capability: str,
        ) -> Any:
            return DynamicWidgetRoot.model_validate(
                {
                    "type": "Card",
                    "size": "full",
                    "children": [
                        _confirmation_body(
                            title="Confirm this exact revision",
                            target=plan.target,
                            instruction=instruction,
                            effects=plan.effects,
                            disclosures=plan.disclosures,
                            fingerprint_label="Request fingerprint",
                            fingerprint=plan.request_fingerprint,
                            one_use_note=(
                                "Confirm is one-use. At the click, janki re-plans and "
                                "refuses if the target or request changed."
                            ),
                        )
                    ],
                    "confirm": {
                        "label": "Confirm exact revision",
                        "action": {
                            "type": _CONFIRM_ACTION,
                            "payload": {
                                "capability": capability,
                                "request_fingerprint": plan.request_fingerprint,
                            },
                            "handler": "server",
                            "loadingBehavior": "container",
                            "streaming": True,
                        },
                    },
                }
            )

        @staticmethod
        def _source_extraction_widget(
            plan: SourceExtractionPlan,
            *,
            capability: str,
        ) -> Any:
            return DynamicWidgetRoot.model_validate(
                {
                    "type": "Card",
                    "size": "full",
                    "children": [
                        _confirmation_body(
                            title="Confirm this exact extraction",
                            target=plan.target,
                            effects=plan.effects,
                            disclosures=plan.disclosures,
                            fingerprint_label="Request fingerprint",
                            fingerprint=plan.request_fingerprint,
                            one_use_note=(
                                "Confirm is one-use. At the click, janki re-plans "
                                "from the saved source and refuses if the request "
                                "or any named replacement changed."
                            ),
                        )
                    ],
                    "confirm": {
                        "label": plan.confirm_label,
                        "action": {
                            "type": _EXTRACT_CONFIRM_ACTION,
                            "payload": {
                                "capability": capability,
                                "request_fingerprint": plan.request_fingerprint,
                            },
                            "handler": "server",
                            "loadingBehavior": "container",
                            "streaming": True,
                        },
                    },
                }
            )

        @staticmethod
        def _finish_widget(
            review: RevisionFinishReview,
            *,
            capability: str,
        ) -> Any:
            return DynamicWidgetRoot.model_validate(
                {
                    "type": "Card",
                    "size": "full",
                    "children": [_finish_review_body(review)],
                    "confirm": {
                        "label": "Apply and finish",
                        "action": {
                            "type": _FINISH_ACTION,
                            "payload": {
                                "capability": capability,
                                "request_fingerprint": review.request_fingerprint,
                            },
                            "handler": "server",
                            "loadingBehavior": "container",
                            "streaming": True,
                        },
                    },
                }
            )

        @staticmethod
        def _prepare_widget(*, capability: str) -> Any:
            return DynamicWidgetRoot.model_validate(
                {
                    "type": "Card",
                    "size": "full",
                    "children": [
                        {
                            "type": "Text",
                            "value": (
                                "This answer is read-only. Use this action only if "
                                "you want the exact message above treated as a deck "
                                "revision instruction."
                            ),
                            "color": "secondary",
                        },
                        {
                            "type": "Button",
                            "label": "Prepare this message as a deck change",
                            "onClickAction": {
                                "type": _PREPARE_ACTION,
                                "payload": {"capability": capability},
                                "handler": "server",
                                "loadingBehavior": "container",
                                "streaming": True,
                            },
                        },
                    ],
                }
            )

        async def respond(
            self,
            thread: Any,
            input_user_message: Any,
            request_context: AssistantRequestContext,
        ) -> Any:
            if request_context.deck_scope != _ASSISTANT_SCOPE:
                yield ErrorEvent(
                    message="The requested assistant scope was refused.",
                    allow_retry=False,
                )
                return
            try:
                message, attachments = self._message_input(input_user_message)
            except RevisionRefusal as error:
                yield NoticeEvent(level="warning", message=str(error))
                return

            if attachments:
                attachment = attachments[0]
                try:
                    intake = await asyncio.to_thread(
                        attachment_store.commit_attachment,
                        attachment.id,
                    )
                except JankiError as error:
                    yield NoticeEvent(
                        level="warning",
                        title="Source not saved",
                        message=str(error),
                    )
                    return
                state = "Saved" if intake.stored else "Already present"
                try:
                    extraction_plan = _validate_source_extraction_plan(
                        await _call_callback(
                            callbacks.prepare_source_extraction,
                            source_path=intake.path,
                        )
                    )
                except RevisionRefusal as error:
                    yield self._message_event(
                        thread,
                        (
                            f"{state}: {intake.path.name}. The source stayed local "
                            "and was not sent to Claude or any other model."
                        ),
                    )
                    yield NoticeEvent(
                        level="warning",
                        title="Extraction plan unavailable",
                        message=str(error),
                    )
                    return
                yield self._message_event(
                    thread,
                    (
                        f"{state}: {intake.path.name}. The source stayed local and "
                        "was not sent to Claude or any other model. "
                        + (
                            "The text accompanying this upload was not used as an "
                            "extraction instruction. "
                            if message
                            else ""
                        )
                        + "Review the exact extraction plan below."
                    ),
                )
                capability = secrets.token_urlsafe(32)
                item_id = store.generate_item_id("message", thread, request_context)
                confirmation = SourceExtractionConfirmation(
                    preparation_id=extraction_plan.preparation_id,
                    source_name=extraction_plan.source_name,
                    expected_fingerprint=extraction_plan.request_fingerprint,
                )
                self._extractions[capability] = _ExtractionBinding(
                    confirmation=confirmation,
                    thread_id=thread.id,
                    widget_item_id=item_id,
                )
                yield ThreadItemDoneEvent(
                    item=WidgetItem(
                        id=item_id,
                        thread_id=thread.id,
                        created_at=datetime.now(),
                        widget=self._source_extraction_widget(
                            extraction_plan,
                            capability=capability,
                        ),
                        copy_text=None,
                    )
                )
                return

            if message == _SHOW_DECKS_MESSAGE:
                yield self._selector_event(thread, request_context)
                return

            local_help = self._local_help(message)
            if local_help is not None:
                yield self._message_event(thread, local_help)
                yield self._selector_event(thread, request_context)
                return

            try:
                selection = self._selection(thread)
            except RevisionRefusal as error:
                yield NoticeEvent(level="warning", message=str(error))
                yield self._selector_event(thread, request_context)
                return
            if selection is None:
                yield NoticeEvent(
                    level="warning",
                    title="Choose an active deck",
                    message=(
                        "Janki did not send this message to a model. Choose a "
                        "supported deck below, then send the question again."
                    ),
                )
                yield self._selector_event(thread, request_context)
                return
            if thread.id in self._busy_threads:
                yield NoticeEvent(
                    level="warning",
                    message="Wait for this thread's current operation to finish.",
                )
                return

            progress_queue: asyncio.Queue[str] = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def report_progress(label: str) -> None:
                normalized = label.strip()
                if normalized not in _CHAT_PROGRESS_LABELS:
                    raise ValueError("The chat service reported an unknown progress state.")
                loop.call_soon_threadsafe(progress_queue.put_nowait, normalized)

            async def execute_chat() -> ChatReply:
                history_lock = self._history_locks.setdefault(thread.id, asyncio.Lock())
                async with history_lock:
                    history = _bounded_chat_history(self._histories.get(thread.id, ()))
                    reply = _validate_chat_reply(
                        await _call_callback(
                            callbacks.chat,
                            deck_scope=selection.choice.scope,
                            history=history,
                            message=message,
                            progress=report_progress,
                        )
                    )
                    self._histories[thread.id] = list(
                        _bounded_chat_history(
                            [*history, ("user", message), ("assistant", reply.text)]
                        )
                    )
                    return reply

            execution_task = asyncio.create_task(execute_chat())
            release_busy = self._track_durable_task(thread.id, execution_task)
            try:
                try:
                    while not execution_task.done():
                        try:
                            label = await asyncio.wait_for(
                                progress_queue.get(), timeout=0.1
                            )
                        except TimeoutError:
                            continue
                        yield ProgressUpdateEvent(text=label, icon="write")
                    while not progress_queue.empty():
                        yield ProgressUpdateEvent(
                            text=progress_queue.get_nowait(),
                            icon="write",
                        )
                    reply = await asyncio.shield(execution_task)
                except RevisionRefusal as error:
                    yield NoticeEvent(
                        level="danger", title="Answer refused", message=str(error)
                    )
                    return

                yield self._message_event(thread, reply.text)
                capability = secrets.token_urlsafe(32)
                item_id = store.generate_item_id("message", thread, request_context)
                self._messages[capability] = _MessageBinding(
                    message=message,
                    thread_id=thread.id,
                    widget_item_id=item_id,
                    deck_id=selection.choice.deck_id,
                    deck_scope=selection.choice.scope,
                    selection_epoch=selection.epoch,
                )
                yield ThreadItemDoneEvent(
                    item=WidgetItem(
                        id=item_id,
                        thread_id=thread.id,
                        created_at=datetime.now(),
                        widget=self._prepare_widget(capability=capability),
                        copy_text=None,
                    )
                )
            finally:
                release_busy()

        async def action(
            self,
            thread: Any,
            action: Any,
            sender: Any,
            request_context: AssistantRequestContext,
        ) -> Any:
            if request_context.deck_scope != _ASSISTANT_SCOPE:
                yield ErrorEvent(message="That assistant action was refused.", allow_retry=False)
                return
            if action.type == _SELECT_DECK_ACTION:
                payload = action.payload
                capability = payload.get("capability") if isinstance(payload, dict) else None
                selector_binding = (
                    self._selectors.pop(capability, None)
                    if isinstance(capability, str)
                    else None
                )
                deck_id = payload.get("deck_id") if isinstance(payload, dict) else None
                if (
                    not isinstance(payload, dict)
                    or set(payload) != {"capability", "deck_id"}
                    or not isinstance(capability, str)
                    or not isinstance(deck_id, str)
                    or selector_binding is None
                    or selector_binding.thread_id != thread.id
                    or sender is None
                    or sender.id != selector_binding.widget_item_id
                    or deck_id not in selector_binding.deck_ids
                ):
                    yield ErrorEvent(
                        message=(
                            "This deck selector is missing, stale, already used, "
                            "tampered with, or belongs elsewhere."
                        ),
                        allow_retry=False,
                    )
                    return
                selection_lock = self._selection_locks.setdefault(
                    thread.id,
                    asyncio.Lock(),
                )
                async with selection_lock:
                    if thread.id in self._busy_threads:
                        yield ErrorEvent(
                            message=(
                                "The active deck cannot change while this thread's "
                                "current operation is still running."
                            ),
                            allow_retry=False,
                        )
                        return
                    try:
                        current = self._selection(thread)
                    except RevisionRefusal as error:
                        yield ErrorEvent(message=str(error), allow_retry=False)
                        return
                    current_epoch = current.epoch if current is not None else 0
                    if selector_binding.selection_epoch != current_epoch:
                        yield ErrorEvent(
                            message="This deck selector is stale or belongs elsewhere.",
                            allow_retry=False,
                        )
                        return
                    choice = choices_by_id[deck_id]
                    if current is not None and current.choice.deck_id == deck_id:
                        yield ErrorEvent(
                            message="That deck is already active in this thread.",
                            allow_retry=False,
                        )
                        return
                    if not choice.revision_supported:
                        yield ErrorEvent(
                            message=choice.unavailable_reason
                            or "That deck is not available for Assistant changes.",
                            allow_retry=False,
                        )
                        return
                    try:
                        fresh_choice = await _call_callback(
                            callbacks.resolve_deck_selection,
                            deck_id,
                        )
                    except (JankiError, RevisionRefusal, OSError, ValueError) as error:
                        yield ErrorEvent(
                            message=f"That deck is no longer available: {error}",
                            allow_retry=False,
                        )
                        return
                    if (
                        not isinstance(fresh_choice, AssistantDeckChoice)
                        or fresh_choice.deck_id != choice.deck_id
                        or fresh_choice.scope != choice.scope
                        or not fresh_choice.revision_supported
                    ):
                        yield ErrorEvent(
                            message="That deck changed after the selector was rendered.",
                            allow_retry=False,
                        )
                        return
                    if thread.id in self._busy_threads:
                        yield ErrorEvent(
                            message=(
                                "The active deck cannot change while this thread's "
                                "current operation is still running."
                            ),
                            allow_retry=False,
                        )
                        return
                    self._invalidate_thread_state(thread.id)
                    next_epoch = current_epoch + 1
                    thread.metadata = {
                        **getattr(thread, "metadata", {}),
                        _ACTIVE_DECK_ID_KEY: choice.deck_id,
                        _ACTIVE_DECK_SCOPE_KEY: choice.scope,
                        _SELECTION_EPOCH_KEY: next_epoch,
                    }
                    yield self._message_event(
                        thread,
                        (
                            f"Active deck: {choice.label}. Deck-specific conversation "
                            "context and older deck-specific action cards in this "
                            "thread were cleared."
                        ),
                    )
                    yield self._selector_event(thread, request_context)
                return

            if action.type == _EXTRACT_CONFIRM_ACTION:
                payload = action.payload
                if not isinstance(payload, dict) or set(payload) != {
                    "capability",
                    "request_fingerprint",
                }:
                    yield ErrorEvent(
                        message="The extraction confirmation payload was refused.",
                        allow_retry=False,
                    )
                    return
                capability = payload.get("capability")
                fingerprint = payload.get("request_fingerprint")
                if not isinstance(capability, str) or not isinstance(fingerprint, str):
                    yield ErrorEvent(
                        message="The extraction confirmation payload was refused.",
                        allow_retry=False,
                    )
                    return
                extraction_binding = self._extractions.pop(capability, None)
                if (
                    extraction_binding is None
                    or extraction_binding.thread_id != thread.id
                    or sender is None
                    or sender.id != extraction_binding.widget_item_id
                    or fingerprint != extraction_binding.confirmation.expected_fingerprint
                ):
                    yield ErrorEvent(
                        message=(
                            "This extraction confirmation is missing, stale, already "
                            "used, or belongs elsewhere."
                        ),
                        allow_retry=False,
                    )
                    return
                if thread.id in self._busy_threads:
                    yield ErrorEvent(
                        message="Wait for this thread's current operation to finish.",
                        allow_retry=False,
                    )
                    return

                progress_queue: asyncio.Queue[str] = asyncio.Queue()
                loop = asyncio.get_running_loop()

                def report_extraction_progress(label: str) -> None:
                    normalized = label.strip()
                    if normalized not in _EXTRACTION_PROGRESS_LABELS:
                        raise ValueError(
                            "The extraction service reported an unknown progress state."
                        )
                    loop.call_soon_threadsafe(progress_queue.put_nowait, normalized)

                async def execute_extraction() -> SourceExtractionExecution:
                    result = await _call_callback(
                        callbacks.consume_replan_and_extract,
                        extraction_binding.confirmation,
                        progress=report_extraction_progress,
                    )
                    return _validate_source_extraction_execution(result)

                execution_task = asyncio.create_task(execute_extraction())
                release_busy = self._track_durable_task(thread.id, execution_task)
                try:
                    try:
                        while not execution_task.done():
                            try:
                                label = await asyncio.wait_for(
                                    progress_queue.get(), timeout=0.1
                                )
                            except TimeoutError:
                                continue
                            yield ProgressUpdateEvent(text=label, icon="write")
                        while not progress_queue.empty():
                            yield ProgressUpdateEvent(
                                text=progress_queue.get_nowait(),
                                icon="write",
                            )
                        result = await asyncio.shield(execution_task)
                    except RevisionRefusal as error:
                        yield NoticeEvent(
                            level="danger",
                            title="Extraction refused",
                            message=str(error),
                        )
                        return
                    yield self._message_event(thread, result.message)
                finally:
                    release_busy()
                return

            if action.type == _FINISH_ACTION:
                payload = action.payload
                if not isinstance(payload, dict):
                    yield ErrorEvent(
                        message="The finish confirmation payload was refused.",
                        allow_retry=False,
                    )
                    return
                capability = payload.get("capability")
                if not isinstance(capability, str):
                    yield ErrorEvent(
                        message="The finish confirmation payload was refused.",
                        allow_retry=False,
                    )
                    return

                # A recognizable capability is consumed before any binding or
                # fingerprint check. A tampered click can never be repaired and
                # replayed into the authority the owner originally saw.
                finish_binding = self._finishes.pop(capability, None)
                fingerprint = payload.get("request_fingerprint")
                try:
                    current_selection = self._selection(thread)
                except RevisionRefusal:
                    current_selection = None
                if (
                    set(payload) != {"capability", "request_fingerprint"}
                    or not isinstance(fingerprint, str)
                    or finish_binding is None
                    or finish_binding.thread_id != thread.id
                    or sender is None
                    or sender.id != finish_binding.widget_item_id
                    or fingerprint != finish_binding.confirmation.expected_fingerprint
                    or not self._binding_is_current(
                        current_selection,
                        deck_id=finish_binding.deck_id,
                        selection_epoch=finish_binding.selection_epoch,
                    )
                    or finish_binding.confirmation.target
                    != current_selection.choice.scope
                ):
                    yield ErrorEvent(
                        message=(
                            "This Apply and finish action is missing, stale, already "
                            "used, or belongs elsewhere."
                        ),
                        allow_retry=False,
                    )
                    return
                if thread.id in self._busy_threads:
                    yield ErrorEvent(
                        message="Wait for this thread's current operation to finish.",
                        allow_retry=False,
                    )
                    return

                progress_queue: asyncio.Queue[str] = asyncio.Queue()
                loop = asyncio.get_running_loop()

                def report_finish_progress(label: str) -> None:
                    normalized = label.strip()
                    if normalized not in _FINISH_PROGRESS_LABELS:
                        raise ValueError(
                            "The revision finish service reported an unknown progress state."
                        )
                    loop.call_soon_threadsafe(progress_queue.put_nowait, normalized)

                async def execute_finish() -> RevisionFinishExecution:
                    result = await _call_callback(
                        callbacks.consume_replan_and_finish,
                        finish_binding.confirmation,
                        progress=report_finish_progress,
                    )
                    return _validate_finish_execution(result)

                execution_task = asyncio.create_task(execute_finish())
                release_busy = self._track_durable_task(thread.id, execution_task)
                try:
                    try:
                        while not execution_task.done():
                            try:
                                label = await asyncio.wait_for(
                                    progress_queue.get(), timeout=0.1
                                )
                            except TimeoutError:
                                continue
                            yield ProgressUpdateEvent(text=label, icon="write")
                        while not progress_queue.empty():
                            yield ProgressUpdateEvent(
                                text=progress_queue.get_nowait(),
                                icon="write",
                            )
                        finish_result = await asyncio.shield(execution_task)
                    except RevisionRefusal as error:
                        yield NoticeEvent(
                            level="danger",
                            title="Apply and finish refused",
                            message=str(error),
                        )
                        return
                    yield self._message_event(thread, finish_result.message)
                finally:
                    release_busy()
                return

            if action.type == _PREPARE_ACTION:
                payload = action.payload
                if not isinstance(payload, dict) or set(payload) != {"capability"}:
                    yield ErrorEvent(
                        message="The prepare-action payload was refused.",
                        allow_retry=False,
                    )
                    return
                capability = payload.get("capability")
                if not isinstance(capability, str):
                    yield ErrorEvent(
                        message="The prepare-action payload was refused.",
                        allow_retry=False,
                    )
                    return
                message_binding = self._messages.pop(capability, None)
                try:
                    current_selection = self._selection(thread)
                except RevisionRefusal:
                    current_selection = None
                if (
                    message_binding is None
                    or message_binding.thread_id != thread.id
                    or sender is None
                    or sender.id != message_binding.widget_item_id
                    or not self._binding_is_current(
                        current_selection,
                        deck_id=message_binding.deck_id,
                        selection_epoch=message_binding.selection_epoch,
                    )
                    or message_binding.deck_scope
                    != current_selection.choice.scope
                ):
                    yield ErrorEvent(
                        message=(
                            "This prepare action is missing, stale, already used, "
                            "or belongs elsewhere."
                        ),
                        allow_retry=False,
                    )
                    return
                if thread.id in self._busy_threads:
                    yield ErrorEvent(
                        message="Wait for this thread's current operation to finish.",
                        allow_retry=False,
                    )
                    return
                self._busy_threads.add(thread.id)
                try:
                    try:
                        plan = _validate_plan(
                            await _call_callback(
                                callbacks.prepare_revision,
                                deck_scope=message_binding.deck_scope,
                                instruction=message_binding.message,
                            )
                        )
                    except RevisionRefusal as error:
                        yield NoticeEvent(level="warning", message=str(error))
                        return
                    if plan.target != message_binding.deck_scope:
                        yield ErrorEvent(
                            message="The revision plan targeted a different deck.",
                            allow_retry=False,
                        )
                        return
                    capability = secrets.token_urlsafe(32)
                    item_id = store.generate_item_id("message", thread, request_context)
                    confirmation = RevisionConfirmation(
                        capability=capability,
                        deck_scope=message_binding.deck_scope,
                        instruction=message_binding.message,
                        expected_fingerprint=plan.request_fingerprint,
                        target=plan.target,
                    )
                    self._plans[capability] = _PlanBinding(
                        confirmation=confirmation,
                        thread_id=thread.id,
                        widget_item_id=item_id,
                        deck_id=message_binding.deck_id,
                        selection_epoch=message_binding.selection_epoch,
                    )
                    yield ThreadItemDoneEvent(
                        item=WidgetItem(
                            id=item_id,
                            thread_id=thread.id,
                            created_at=datetime.now(),
                            widget=self._plan_widget(
                                plan,
                                instruction=message_binding.message,
                                capability=capability,
                            ),
                            copy_text=None,
                        )
                    )
                    return
                finally:
                    self._busy_threads.discard(thread.id)

            if action.type != _CONFIRM_ACTION:
                yield ErrorEvent(message="That assistant action was refused.", allow_retry=False)
                return
            payload = action.payload
            if not isinstance(payload, dict) or set(payload) != {
                "capability",
                "request_fingerprint",
            }:
                yield ErrorEvent(message="The confirmation payload was refused.", allow_retry=False)
                return
            capability = payload.get("capability")
            fingerprint = payload.get("request_fingerprint")
            if not isinstance(capability, str) or not isinstance(fingerprint, str):
                yield ErrorEvent(message="The confirmation payload was refused.", allow_retry=False)
                return

            binding = self._plans.pop(capability, None)
            try:
                current_selection = self._selection(thread)
            except RevisionRefusal:
                current_selection = None
            if (
                binding is None
                or binding.thread_id != thread.id
                or sender is None
                or sender.id != binding.widget_item_id
                or fingerprint != binding.confirmation.expected_fingerprint
                or not self._binding_is_current(
                    current_selection,
                    deck_id=binding.deck_id,
                    selection_epoch=binding.selection_epoch,
                )
                or binding.confirmation.deck_scope != current_selection.choice.scope
                or binding.confirmation.target != current_selection.choice.scope
            ):
                yield ErrorEvent(
                    message=(
                        "This confirmation is missing, stale, already used, or belongs elsewhere."
                    ),
                    allow_retry=False,
                )
                return
            if thread.id in self._busy_threads:
                yield ErrorEvent(
                    message="Wait for this thread's current operation to finish.",
                    allow_retry=False,
                )
                return

            progress_queue: asyncio.Queue[str] = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def report_progress(label: str) -> None:
                normalized = label.strip()
                if normalized not in _PROGRESS_LABELS:
                    raise ValueError("The revision service reported an unknown progress state.")
                loop.call_soon_threadsafe(progress_queue.put_nowait, normalized)

            async def execute() -> RevisionExecution:
                result = await _call_callback(
                    callbacks.consume_replan_and_execute,
                    binding.confirmation,
                    progress=report_progress,
                )
                try:
                    return _validate_execution(result)
                except RevisionRefusal:
                    raise
                except Exception as error:
                    if (
                        not isinstance(result, RevisionExecution)
                        or not isinstance(result.message, str)
                        or not result.message.strip()
                    ):
                        raise
                    return RevisionExecution(
                        message=result.message,
                        finish=None,
                        finish_unavailable=_staged_finish_unavailable(error),
                    )

            execution_task = asyncio.create_task(execute())
            release_busy = self._track_durable_task(thread.id, execution_task)
            try:
                yield ProgressUpdateEvent(text="Preparing revision", icon="write")

                try:
                    while not execution_task.done():
                        try:
                            label = await asyncio.wait_for(
                                progress_queue.get(), timeout=0.1
                            )
                        except TimeoutError:
                            continue
                        yield ProgressUpdateEvent(text=label, icon="write")
                    while not progress_queue.empty():
                        yield ProgressUpdateEvent(
                            text=progress_queue.get_nowait(), icon="write"
                        )
                    result = await asyncio.shield(execution_task)
                except RevisionRefusal as error:
                    yield NoticeEvent(
                        level="danger", title="Revision refused", message=str(error)
                    )
                    return

                yield self._message_event(thread, result.message)
                finish = result.finish
                if finish is None:
                    yield NoticeEvent(
                        level="warning",
                        title="Apply and finish review unavailable",
                        message=(
                            result.finish_unavailable
                            or "The finish plan is unavailable."
                        ),
                    )
                    return
                capability = secrets.token_urlsafe(32)
                item_id = store.generate_item_id("message", thread, request_context)
                finish_confirmation = RevisionFinishConfirmation(
                    preparation_id=finish.preparation_id,
                    expected_fingerprint=finish.request_fingerprint,
                    target=finish.target,
                )
                try:
                    finish_widget = self._finish_widget(
                        finish,
                        capability=capability,
                    )
                    finish_event = ThreadItemDoneEvent(
                        item=WidgetItem(
                            id=item_id,
                            thread_id=thread.id,
                            created_at=datetime.now(),
                            widget=finish_widget,
                            copy_text=None,
                        )
                    )
                except Exception as error:
                    yield NoticeEvent(
                        level="warning",
                        title="Apply and finish review unavailable",
                        message=_staged_finish_unavailable(error),
                    )
                    return
                self._finishes[capability] = _FinishBinding(
                    confirmation=finish_confirmation,
                    thread_id=thread.id,
                    widget_item_id=item_id,
                    deck_id=binding.deck_id,
                    selection_epoch=binding.selection_epoch,
                )
                yield finish_event
            finally:
                release_busy()

    server = _JankiChatKitServer()
    return AssistantCore(
        server=server,
        context=context,
        store=store,
        attachment_store=attachment_store,
    )
