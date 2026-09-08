"""ChatKit controller for repository-wide Japanese-library conversation.

This module deliberately contains no model client. Every composer message goes
only to the injected chat callback. A selected deck is optional thread-local
focus, not authority: an unfocused callback receives an empty ``deck_scope``.
Every mutation arrives as a typed action attached to the ordinary reply; there
is no second "turn this message into a change" route.

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
from typing import Any, Literal, Protocol, TypeVar
from urllib.parse import urlsplit

from japanese_anki.errors import JankiError

__all__ = [
    "CAPABILITIES_MESSAGE",
    "MANAGE_OPERATIONS_MESSAGE",
    "RESUME_KANJI_MESSAGE",
    "SHOW_DECKS_MESSAGE",
    "SOURCE_HELP_MESSAGE",
    "AssistantDeckChoice",
    "AssistantCore",
    "AssistantRequestContext",
    "ChatReply",
    "KanjiFinishActionChoice",
    "KanjiFinishChoice",
    "KanjiFinishResumption",
    "OperationActionChoice",
    "OperationChoice",
    "RevisionCallbacks",
    "RevisionConfirmation",
    "RevisionExecution",
    "RevisionExampleReview",
    "RevisionFinishConfirmation",
    "RevisionFinishExecution",
    "RevisionFinishReview",
    "StagedContentFinishReview",
    "RevisionRecordReview",
    "RevisionPlan",
    "RevisionRefusal",
    "ScopedMemoryStore",
    "SourceExtractionConfirmation",
    "SourceExtractionExecution",
    "SourceExtractionPlan",
    "create_assistant_core",
]


_CONFIRM_ACTION = "janki.revision.confirm"
_FINISH_ACTION = "janki.revision.finish"
_EXTRACT_CONFIRM_ACTION = "janki.extraction.confirm"
_SELECT_DECK_ACTION = "janki.deck.select"
_PREPARE_OPERATION_ACTION = "janki.operation.prepare"
_RESUME_KANJI_ACTION = "janki.kanji.resume"
_ALL_LIBRARY_DECK_ID = "janki:all-library"
SHOW_DECKS_MESSAGE = "Choose a deck to focus on"
CAPABILITIES_MESSAGE = "Show me what I can do with my Japanese library"
SOURCE_HELP_MESSAGE = "How do I add study material?"
MANAGE_OPERATIONS_MESSAGE = "Manage model calls"
RESUME_KANJI_MESSAGE = "Resume kanji cards"
_ASSISTANT_SCOPE = "janki-project"
_ACTIVE_DECK_ID_KEY = "janki_active_deck_id"
_ACTIVE_DECK_SCOPE_KEY = "janki_active_deck_scope"
_SELECTION_EPOCH_KEY = "janki_deck_selection_epoch"
_FINGERPRINT_DISPLAY_CHARS = 32
_MAX_CHAT_HISTORY_ENTRIES = 12
_MAX_CHAT_HISTORY_BYTES = 24_000
_DECK_SELECTOR_MESSAGES = frozenset(
    {
        "change active deck",
        "change the active deck",
        "choose a deck",
        "choose a deck to focus on",
        "choose an active deck",
        "choose new study content",
        "i want to work on a different deck",
        "let's pick new study content",
        "let's work on a different deck",
        "pick a deck",
        "pick a new deck",
        "pick new study content",
        "select a deck",
        "select the active deck",
        "switch deck",
        "switch decks",
        "switch to a different deck",
        "switch to a new deck",
        "switch to another deck",
        "work on a different deck",
    }
)
_CHAT_PROGRESS_LABELS = frozenset(
    {
        "Preparing answer",
        "Writing answer",
        "Staging proposed changes",
        "Saving answer",
    }
)
_PROGRESS_LABELS = frozenset(
    {
        # The Assistant agent's own labels: `run_agent` reports them on the chat
        # route and `recover_agent` reports the same ones on the action route.
        "Preparing answer",
        "Writing answer",
        "Saving answer",
        "Preparing revision",
        "Reading the source",
        "Checking the answer's shape",
        "Saving proposals",
        "Preparing audio",
        "Creating audio",
        "Saving audio",
        "Cleaning up audio",
        "Preparing build",
        "Preparing deck",
        "Preparing pages",
        "Saving review",
        "Saving deck assignments",
        "Removing staged cards",
        "Saving staged identity",
        "Saving coverage approval",
        "Checking the reviewed proposal",
        "Checking readings",
        "Saving reviewed cards",
        "Checking an earlier model call",
        "Deleting canonical cards",
        "Deleting deck definition",
        "Preparing finish",
        "Applying reviewed cards",
        "Creating card audio",
        "Writing character notes",
        "Building Anki package",
        "Saving finish receipt",
    }
)
_FINISH_PROGRESS_LABELS = frozenset(
    {
        "Preparing finish",
        "Applying reviewed revision",
        "Applying reviewed cards",
        "Creating example audio",
        "Creating card audio",
        "Writing character notes",
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


def _is_deck_selector_message(message: str) -> bool:
    normalized = " ".join(
        message.translate(str.maketrans({"‘": "'", "’": "'"})).casefold().split()
    ).rstrip(".!?")
    return normalized in _DECK_SELECTOR_MESSAGES


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
    """One Assistant answer and, when requested, one exact protected plan.

    The ordinary provider call may describe a closed action intent, but only
    the local application adapter can turn that intent into ``action``.  The
    controller renders that already-planned action directly; it never treats
    arbitrary prose as an instruction or grants the model a capability.
    """

    text: str
    action: RevisionPlan | None = None
    action_instruction: str | None = None


@dataclass(frozen=True, slots=True)
class AssistantDeckChoice:
    """One server-discovered deck shown in the optional-focus selector.

    ``deck_id`` is the only target value accepted from the browser. ``scope``
    stays server-side and is passed to application callbacks only after the
    opaque id has been resolved through the immutable startup catalog.
    ``revision_supported`` describes only the specialized rich-conjugation
    whole-deck pass. Generic canonical-card revision is resolved separately
    for every readable vocabulary-backed deck, so this flag does not create a
    presentation capability class.
    """

    deck_id: str
    label: str
    scope: str
    chat_supported: bool
    revision_supported: bool
    unavailable_reason: str | None = None


@dataclass(frozen=True, slots=True)
class OperationActionChoice:
    """One exact operation-journal action offered by the local application."""

    action: Literal["recover", "show_reply", "end", "forget"]
    label: str
    accept_paid_output_loss: bool = False


@dataclass(frozen=True, slots=True)
class OperationChoice:
    """Safe operation facts rendered without asking the conversation model."""

    operation_id: str
    kind: str
    state: str
    source_name: str
    model: str
    authorized_at: str
    blocks_spending: bool
    money_may_have_been_spent: bool
    has_captured_reply: bool
    has_response_spool: bool
    cleanup_pending: bool
    actions: tuple[OperationActionChoice, ...]


@dataclass(frozen=True, slots=True)
class KanjiFinishActionChoice:
    """One action a durable character-note receipt still offers."""

    action: Literal["resume", "download"]
    label: str


@dataclass(frozen=True, slots=True)
class KanjiFinishChoice:
    """One durable character-note receipt found on disk after a restart.

    This is not a plan and carries no capability: the receipt itself is the
    owner's already-recorded confirmation, so resuming it needs no fresh
    consent and no replacement batch.
    """

    receipt_id: str
    state: str
    deck_name: str
    target: str
    characters: str
    detail: str
    actions: tuple[KanjiFinishActionChoice, ...]


@dataclass(frozen=True, slots=True)
class KanjiFinishResumption:
    """Truthful durable state after resuming one receipt in the thread."""

    message: str
    receipt_id: str
    state: str
    complete: bool


@dataclass(frozen=True, slots=True)
class RevisionPlan:
    """Exact, side-effect-free revision plan rendered before owner consent."""

    request_fingerprint: str
    target: str
    effects: tuple[str, ...]
    disclosures: tuple[str, ...] = ()
    confirm_label: str = "Confirm exact action"
    progress_label: str = "Preparing revision"
    #: One local link to the exact cards this plan renders, when Janki could
    #: draw them. Looking at a preview is not an approval, and a plan without
    #: one is complete: its written effects remain the decision.
    preview_url: str | None = None
    preview_label: str = "Preview these cards"


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
    #: The reviewed cards as Anki draws them, rendered from the exact proposed
    #: content already staged. Never an approval, and never required.
    preview_url: str | None = None
    preview_label: str = "Preview these cards"
    #: Why there is no link, when Janki tried and could not draw them. The
    #: review itself stays complete and usable; this says what is missing
    #: instead of leaving a silent gap where the cards would be.
    preview_unavailable_message: str | None = None


@dataclass(frozen=True, slots=True)
class StagedContentFinishReview:
    """One exact generic staging review and all post-review consequences."""

    preparation_id: str
    request_fingerprint: str
    target: str
    effects: tuple[str, ...]
    disclosures: tuple[str, ...]
    confirm_label: str = "Apply and finish"
    preview_url: str | None = None
    preview_label: str = "Preview these cards"


@dataclass(frozen=True, slots=True)
class RevisionExecution:
    """Durable staged proposal plus its fresh exact finish review."""

    message: str
    finish: RevisionFinishReview | StagedContentFinishReview | None
    finish_unavailable: str | None = None
    review_required: str | None = None
    complete: bool = False
    remember_in_chat_context: bool = True


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

    def list_operation_choices(self) -> tuple[OperationChoice, ...]:
        """Return current blocking and recovery-bearing paid operations."""

    def prepare_operation_action(
        self,
        *,
        operation_id: str,
        action: str,
        accept_paid_output_loss: bool,
        deck_scope: str,
    ) -> ChatReply | Awaitable[ChatReply]:
        """Freshly plan one owner-selected journal action without a model call."""

    def list_kanji_finish_choices(
        self,
    ) -> tuple[KanjiFinishChoice, ...] | Awaitable[tuple[KanjiFinishChoice, ...]]:
        """Return the durable character-note receipts still worth acting on."""

    def resume_kanji_finish(
        self,
        *,
        receipt_id: str,
        action: str,
        progress: Callable[[str], None],
    ) -> KanjiFinishResumption | Awaitable[KanjiFinishResumption]:
        """Continue or re-offer one durable receipt under its recorded authority."""

    def chat(
        self,
        *,
        deck_scope: str,
        history: tuple[tuple[str, str], ...],
        message: str,
        progress: Callable[[str], None],
        preview: Callable[[str], None],
    ) -> ChatReply | Awaitable[ChatReply]:
        """Answer one message without planning or mutating a deck."""

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
    deck_id: str | None
    selection_epoch: int
    progress_label: str


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
    deck_id: str | None
    selection_epoch: int


@dataclass(frozen=True, slots=True)
class _SelectorBinding:
    thread_id: str
    widget_item_id: str
    selection_epoch: int
    deck_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class _OperationSelectorBinding:
    thread_id: str
    widget_item_id: str
    deck_id: str | None
    selection_epoch: int
    actions: frozenset[tuple[str, str, bool]]


@dataclass(frozen=True, slots=True)
class _KanjiResumeBinding:
    """Which rendered receipt rows this exact widget may act on.

    The binding is a UI anti-replay guard, not authority: the durable receipt
    carries the owner's confirmation, and the application service revalidates
    it before continuing.
    """

    thread_id: str
    widget_item_id: str
    actions: frozenset[tuple[str, str]]


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
        # Delete means ensure-absent. ChatKit removes pending items it never
        # stored — a streamed message that did not finish is the common case —
        # so an id this store never held is the state the caller asked for.

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
    if reply.action is None:
        if reply.action_instruction is not None:
            raise ValueError(
                "A chat reply without an action cannot carry an action instruction."
            )
    else:
        _validate_plan(reply.action)
        if (
            not isinstance(reply.action_instruction, str)
            or not reply.action_instruction.strip()
        ):
            raise ValueError("A planned Assistant action needs its exact instruction.")
    return reply


def _validate_kanji_finish_choices(value: Any) -> tuple[KanjiFinishChoice, ...]:
    if not isinstance(value, tuple) or any(
        not isinstance(choice, KanjiFinishChoice) for choice in value
    ):
        raise TypeError(
            "list_kanji_finish_choices must return KanjiFinishChoice values"
        )
    receipt_ids: set[str] = set()
    for choice in value:
        if any(
            not item.strip()
            for item in (choice.receipt_id, choice.state, choice.target, choice.detail)
        ):
            raise ValueError("Every character-note receipt choice must be complete.")
        if choice.receipt_id in receipt_ids:
            raise ValueError("Character-note receipt choices must have unique ids.")
        receipt_ids.add(choice.receipt_id)
        seen: set[str] = set()
        for action in choice.actions:
            if (
                not isinstance(action, KanjiFinishActionChoice)
                or action.action not in {"resume", "download"}
                or not action.label.strip()
                or action.action in seen
            ):
                raise ValueError(
                    "Every character-note receipt action must be valid and unique."
                )
            seen.add(action.action)
        if not seen:
            raise ValueError("A character-note receipt with no action is not offered.")
    return value


def _validate_kanji_resumption(value: Any) -> KanjiFinishResumption:
    if not isinstance(value, KanjiFinishResumption):
        raise TypeError("resume_kanji_finish must return KanjiFinishResumption")
    if not value.message.strip() or not value.receipt_id.strip():
        raise ValueError("A resumed character-note receipt must report its state.")
    if not value.state.strip() or not isinstance(value.complete, bool):
        raise ValueError("A resumed character-note receipt must report its state.")
    return value


def _validate_operation_choices(value: Any) -> tuple[OperationChoice, ...]:
    if not isinstance(value, tuple) or any(
        not isinstance(choice, OperationChoice) for choice in value
    ):
        raise TypeError("list_operation_choices must return OperationChoice values")
    operation_ids: set[str] = set()
    for choice in value:
        required = (
            choice.operation_id,
            choice.kind,
            choice.state,
            choice.source_name,
            choice.model,
            choice.authorized_at,
        )
        if any(not item.strip() for item in required):
            raise ValueError("Every paid-operation choice must be complete.")
        if choice.operation_id in operation_ids:
            raise ValueError("Paid-operation choices must have unique ids.")
        operation_ids.add(choice.operation_id)
        if any(
            not isinstance(flag, bool)
            for flag in (
                choice.blocks_spending,
                choice.money_may_have_been_spent,
                choice.has_captured_reply,
                choice.has_response_spool,
                choice.cleanup_pending,
            )
        ):
            raise ValueError("Paid-operation status flags must be true or false.")
        seen_actions: set[str] = set()
        for action in choice.actions:
            if (
                not isinstance(action, OperationActionChoice)
                or action.action not in {"recover", "show_reply", "end", "forget"}
                or not action.label.strip()
                or not isinstance(action.accept_paid_output_loss, bool)
                or action.action in seen_actions
            ):
                raise ValueError("Every paid-operation action must be valid and unique.")
            seen_actions.add(action.action)
    return value


def _validate_plan(plan: Any) -> RevisionPlan:
    if not isinstance(plan, RevisionPlan):
        raise TypeError("A planned Assistant reply must carry RevisionPlan")
    if not plan.request_fingerprint.strip():
        raise ValueError("The revision plan has no request fingerprint.")
    if not plan.target.strip():
        raise ValueError("The revision plan has no target.")
    if not plan.effects or any(not effect.strip() for effect in plan.effects):
        raise ValueError("The revision plan must name every intended effect.")
    if any(not disclosure.strip() for disclosure in plan.disclosures):
        raise ValueError("Revision-plan disclosures must not be blank.")
    if not plan.confirm_label.strip():
        raise ValueError("The action plan must have a confirmation label.")
    if plan.progress_label not in _PROGRESS_LABELS:
        raise ValueError("The action plan has an unknown progress state.")
    _validate_preview_link(plan.preview_url, plan.preview_label)
    return plan


def _validate_preview_link(url: str | None, label: str) -> None:
    """Refuse a preview link that is not a plain loopback HTTP URL.

    A preview link is written into a widget the owner clicks, so this parses
    the URL rather than matching its prefix: ``http://localhost:80@evil/`` has
    the right first characters and a different host entirely. This is a
    loopback-shape check, not proof of a particular origin — the exact port
    belongs to the running sidecar, which mints these URLs itself.
    """

    if url is None:
        return
    if not url.strip() or not label.strip():
        raise ValueError("A card-preview link needs its exact URL and label.")
    try:
        parsed = urlsplit(url)
        port = parsed.port
        hostname = parsed.hostname
    except ValueError as exc:
        raise ValueError(f"The card-preview link is not a usable URL: {exc}") from exc
    if (
        parsed.scheme != "http"
        or parsed.username is not None
        or parsed.password is not None
        or hostname not in {"127.0.0.1", "localhost", "::1"}
        or port is None
    ):
        raise ValueError(
            "A card-preview link must be a loopback http:// URL with a port and "
            "no embedded credentials."
        )


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
    if not isinstance(result.remember_in_chat_context, bool):
        raise ValueError("The revision result context choice must be true or false.")
    if result.finish is None:
        explanations = tuple(
            value
            for value in (result.finish_unavailable, result.review_required)
            if value is not None and value.strip()
        )
        expected_explanations = 0 if result.complete else 1
        if len(explanations) != expected_explanations:
            raise ValueError(
                "An unfinished action without a finish review must carry exactly "
                "one unavailability or review-required explanation; a completed "
                "action must carry neither."
            )
    else:
        if (
            result.finish_unavailable is not None
            or result.review_required is not None
            or result.complete
        ):
            raise ValueError(
                "An action with a follow-up finish review cannot also be complete "
                "or unavailable."
            )
        if isinstance(result.finish, RevisionFinishReview):
            _validate_finish_review(result.finish)
        else:
            _validate_staged_content_finish_review(result.finish)
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
    _validate_preview_link(review.preview_url, review.preview_label)
    if review.preview_unavailable_message is not None:
        if not review.preview_unavailable_message.strip():
            raise ValueError("An absent card preview must say why it is absent.")
        if review.preview_url is not None:
            raise ValueError(
                "A revision finish review cannot both offer a card preview and "
                "explain why there is none."
            )
    return review


def _validate_staged_content_finish_review(
    review: Any,
) -> StagedContentFinishReview:
    if not isinstance(review, StagedContentFinishReview):
        raise TypeError(
            "The staged action must include a supported exact finish review."
        )
    if any(
        not value.strip()
        for value in (
            review.preparation_id,
            review.request_fingerprint,
            review.target,
            review.confirm_label,
        )
    ):
        raise ValueError("The staged content finish review is incomplete.")
    if not _is_lower_sha256(review.request_fingerprint):
        raise ValueError("The staged content finish fingerprint is malformed.")
    if not review.effects or any(not value.strip() for value in review.effects):
        raise ValueError(
            "The staged content finish must render every exact consequence."
        )
    if any(not value.strip() for value in review.disclosures):
        raise ValueError("Staged content finish disclosures must not be blank.")
    if review.confirm_label != "Apply and finish":
        raise ValueError(
            "A staged content aggregate must name its one Apply and finish action."
        )
    _validate_preview_link(review.preview_url, review.preview_label)
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
        "promoted",
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


def _preview_link_body(url: str, label: str) -> dict[str, Any]:
    """One link to the actual cards, above the written consequences.

    The owner's preferred review is the cards themselves (``docs/DESIGN.md``),
    so it comes first — but opening it is not the decision, which is why the
    exact effects below it are unchanged and the confirm button is elsewhere.
    """

    return {
        "type": "Col",
        "id": "confirmation-preview",
        "gap": 1,
        "width": "100%",
        "minWidth": 0,
        "children": [
            {"type": "Markdown", "value": f"[{label}]({url})"},
            {
                "type": "Text",
                "value": (
                    "Opens the real cards in a new tab. Looking is not approving."
                ),
                "size": "xs",
                "color": "tertiary",
                "width": "100%",
            },
        ],
    }


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
    preview_url: str | None = None,
    preview_label: str = "Preview these cards",
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
    if preview_url is not None:
        sections.append(_preview_link_body(preview_url, preview_label))
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
        *(
            (_preview_link_body(review.preview_url, review.preview_label),)
            if review.preview_url is not None
            else ()
        ),
        *(
            (
                {
                    "type": "Col",
                    "id": "revision-finish-preview-unavailable",
                    "gap": 1,
                    "width": "100%",
                    "minWidth": 0,
                    "children": [
                        {
                            "type": "Text",
                            "value": (
                                "Janki could not draw these cards for review: "
                                f"{review.preview_unavailable_message}"
                            ),
                            "color": "secondary",
                            "width": "100%",
                        }
                    ],
                },
            )
            if review.preview_unavailable_message is not None
            else ()
        ),
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
        AssistantMessageContentPartAdded,
        AssistantMessageContentPartDone,
        AssistantMessageContentPartTextDelta,
        AssistantMessageItem,
        ErrorEvent,
        NoticeEvent,
        ProgressUpdateEvent,
        StreamOptions,
        ThreadItemAddedEvent,
        ThreadItemDoneEvent,
        ThreadItemRemovedEvent,
        ThreadItemUpdatedEvent,
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
    if any(choice.deck_id == _ALL_LIBRARY_DECK_ID for choice in choices):
        raise ValueError("Assistant deck choice ids cannot use the all-library id.")
    if len({choice.scope for choice in choices}) != len(choices):
        raise ValueError("Assistant deck choice scopes must be unique.")
    if any(not choice.chat_supported and not choice.unavailable_reason for choice in choices):
        raise ValueError(
            "Every unreadable deck choice must explain why focus is unavailable."
        )
    choices_by_id = {choice.deck_id: choice for choice in choices}

    store = ScopedMemoryStore(_ASSISTANT_SCOPE, attachment_store=attachment_store)
    context = AssistantRequestContext(deck_scope=_ASSISTANT_SCOPE)

    class _JankiChatKitServer(ChatKitServer[AssistantRequestContext]):
        def __init__(self) -> None:
            super().__init__(store=store, attachment_store=attachment_store)
            self._selectors: dict[str, _SelectorBinding] = {}
            self._operation_selectors: dict[str, _OperationSelectorBinding] = {}
            self._kanji_resumes: dict[str, _KanjiResumeBinding] = {}
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

        async def _remember_assistant_result(self, thread_id: str, message: str) -> None:
            """Keep a completed application action visible to the next model turn."""

            history_lock = self._history_locks.setdefault(thread_id, asyncio.Lock())
            async with history_lock:
                history = _bounded_chat_history(self._histories.get(thread_id, ()))
                self._histories[thread_id] = list(
                    _bounded_chat_history([*history, ("assistant", message)])
                )

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
                raise RevisionRefusal("This thread's deck-focus state is invalid.")
            choice = choices_by_id.get(deck_id)
            if (
                choice is None
                or not choice.chat_supported
                or choice.scope != deck_scope
            ):
                raise RevisionRefusal("This thread's deck focus is no longer available.")
            return _ThreadSelection(choice=choice, epoch=epoch)

        @staticmethod
        def _binding_is_current(
            selection: _ThreadSelection | None,
            *,
            deck_id: str | None,
            selection_epoch: int,
        ) -> bool:
            if deck_id is None:
                return selection is None and selection_epoch == 0
            return (
                selection is not None
                and selection.choice.deck_id == deck_id
                and selection.epoch == selection_epoch
            )

        def _invalidate_thread_state(self, thread_id: str) -> None:
            self._histories.pop(thread_id, None)
            for bindings in (
                self._selectors,
                self._operation_selectors,
                self._kanji_resumes,
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

        def _clear_focus(self, thread: Any) -> None:
            """Clear stale focus plus every context/action derived from it."""

            self._invalidate_thread_state(thread.id)
            metadata = dict(getattr(thread, "metadata", {}))
            metadata.pop(_ACTIVE_DECK_ID_KEY, None)
            metadata.pop(_ACTIVE_DECK_SCOPE_KEY, None)
            metadata.pop(_SELECTION_EPOCH_KEY, None)
            thread.metadata = metadata

        @staticmethod
        def _local_help(message: str) -> str | None:
            if message == CAPABILITIES_MESSAGE:
                return (
                    "You can use Janki across the whole Japanese library; choosing a "
                    "deck only gives the conversation a convenient focus.\n\n"
                    "- **Read and find:** inspect configured decks, canonical cards, "
                    "preserved sources, staging proposals, patterns, kanji, operation "
                    "status, media, and built packages.\n\n"
                    "- **Change study content:** revise selected canonical cards in any "
                    "readable vocabulary-backed deck. Rich conjugation decks also support "
                    "their specialized whole-deck teaching-content revision. Model-written "
                    "Japanese always stops in staging for your review.\n\n"
                    "- **Run the deck workflow:** create a deck; attach one PDF or photo "
                    "for local intake; extract it; review or organize staged cards; "
                    "promote reviewed proposals; generate audio; and build Anki packages "
                    "through exact action plans.\n\n"
                    f"- **Pick work back up:** say “{RESUME_KANJI_MESSAGE}” to list the "
                    "kanji card batches you already confirmed, and finish or download "
                    "one without confirming it again.\n\n"
                    "Janki uses bounded application actions rather than giving the model a "
                    "raw shell, arbitrary filesystem, Git, or network access."
                )
            if message == SOURCE_HELP_MESSAGE:
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
            all_library_details: list[dict[str, Any]] = [
                {
                    "type": "Text",
                    "value": "All library",
                    "weight": "semibold",
                    "width": "100%",
                },
                {
                    "type": "Text",
                    "value": (
                        "Clear the active deck focus"
                        if active is not None
                        else "No deck focus; use the whole Japanese library"
                    ),
                    "size": "sm",
                    "color": "secondary",
                    "width": "100%",
                },
            ]
            if active is None:
                all_library_details.append(
                    {
                        "type": "Badge",
                        "label": "Focused",
                        "color": "success",
                        "variant": "soft",
                    }
                )
            all_library_row: dict[str, Any] = {
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
                        "children": all_library_details,
                    }
                ],
            }
            if active is not None:
                all_library_row["onClickAction"] = {
                    "type": _SELECT_DECK_ACTION,
                    "payload": {
                        "capability": capability,
                        "deck_id": _ALL_LIBRARY_DECK_ID,
                    },
                    "handler": "server",
                    "loadingBehavior": "container",
                    "streaming": True,
                }
            rows: list[dict[str, Any]] = [all_library_row]
            for choice in choices:
                is_active = (
                    active is not None and active.choice.deck_id == choice.deck_id
                )
                if is_active:
                    badge = {
                        "type": "Badge",
                        "label": "Focused",
                        "color": "success",
                        "variant": "soft",
                    }
                elif choice.chat_supported:
                    badge = None
                else:
                    badge = {
                        "type": "Badge",
                        "label": "Unavailable",
                        "color": "secondary",
                        "variant": "soft",
                    }
                details: list[dict[str, Any]] = [
                    {
                        "type": "Text",
                        "value": choice.label,
                        "weight": "semibold",
                        "width": "100%",
                    }
                ]
                if badge is not None:
                    details.append(badge)
                if not choice.chat_supported and choice.unavailable_reason:
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
                if choice.chat_supported and not is_active:
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
                        "text": "Optional deck focus",
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
                # A selector is the recovery surface for stale focus metadata.
                # Project-scoped extraction remains independent.
                self._clear_focus(thread)
                active = None
            capability = secrets.token_urlsafe(32)
            item_id = store.generate_item_id("message", thread, request_context)
            self._selectors[capability] = _SelectorBinding(
                thread_id=thread.id,
                widget_item_id=item_id,
                selection_epoch=active.epoch if active is not None else 0,
                deck_ids=frozenset(
                    {
                        _ALL_LIBRARY_DECK_ID,
                        *(choice.deck_id for choice in choices if choice.chat_supported),
                    }
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

        @staticmethod
        def _operation_selector_widget(
            operation_choices: tuple[OperationChoice, ...],
            *,
            capability: str,
        ) -> Any:
            rows: list[dict[str, Any]] = []
            for choice in operation_choices:
                status = "Blocks new model calls" if choice.blocks_spending else "Recovery only"
                evidence = []
                if choice.has_captured_reply:
                    evidence.append("captured reply")
                if choice.has_response_spool:
                    evidence.append("response frames")
                if choice.cleanup_pending:
                    evidence.append("cleanup pending")
                details = (
                    f"{choice.kind} · {choice.state} · {choice.source_name} · "
                    f"{choice.model} · {status}"
                )
                if evidence:
                    details += " · " + ", ".join(evidence)
                rows.append(
                    {
                        "type": "ListViewItem",
                        "gap": 2,
                        "children": [
                            {
                                "type": "Box",
                                "direction": "column",
                                "gap": 1,
                                "minWidth": 0,
                                "children": [
                                    {
                                        "type": "Text",
                                        "value": choice.operation_id,
                                        "weight": "semibold",
                                        "width": "100%",
                                    },
                                    {
                                        "type": "Text",
                                        "value": details,
                                        "size": "sm",
                                        "color": "secondary",
                                        "width": "100%",
                                    },
                                ],
                            }
                        ],
                    }
                )
                for option in choice.actions:
                    rows.append(
                        {
                            "type": "ListViewItem",
                            "gap": 2,
                            "children": [
                                {
                                    "type": "Text",
                                    "value": option.label,
                                    "width": "100%",
                                    "color": (
                                        "danger"
                                        if option.accept_paid_output_loss
                                        else "primary"
                                    ),
                                }
                            ],
                            "onClickAction": {
                                "type": _PREPARE_OPERATION_ACTION,
                                "payload": {
                                    "capability": capability,
                                    "operation_id": choice.operation_id,
                                    "action": option.action,
                                    "accept_paid_output_loss": (
                                        option.accept_paid_output_loss
                                    ),
                                },
                                "handler": "server",
                                "loadingBehavior": "container",
                                "streaming": True,
                            },
                        }
                    )
            return DynamicWidgetRoot.model_validate(
                {
                    "type": "ListView",
                    "limit": "auto",
                    "status": {
                        "text": "Model-call recovery",
                        "icon": "keys",
                    },
                    "children": rows,
                }
            )

        def _operation_selector_event(
            self,
            thread: Any,
            request_context: AssistantRequestContext,
            operation_choices: tuple[OperationChoice, ...],
        ) -> Any:
            selection = self._selection(thread)
            capability = secrets.token_urlsafe(32)
            item_id = store.generate_item_id("message", thread, request_context)
            allowed = frozenset(
                (
                    choice.operation_id,
                    option.action,
                    option.accept_paid_output_loss,
                )
                for choice in operation_choices
                for option in choice.actions
            )
            self._operation_selectors[capability] = _OperationSelectorBinding(
                thread_id=thread.id,
                widget_item_id=item_id,
                deck_id=selection.choice.deck_id if selection is not None else None,
                selection_epoch=selection.epoch if selection is not None else 0,
                actions=allowed,
            )
            return ThreadItemDoneEvent(
                item=WidgetItem(
                    id=item_id,
                    thread_id=thread.id,
                    created_at=datetime.now(),
                    widget=self._operation_selector_widget(
                        operation_choices,
                        capability=capability,
                    ),
                    copy_text=None,
                )
            )

        @staticmethod
        def _kanji_finish_widget(
            finish_choices: tuple[KanjiFinishChoice, ...],
            *,
            capability: str,
        ) -> Any:
            rows: list[dict[str, Any]] = []
            for choice in finish_choices:
                rows.append(
                    {
                        "type": "ListViewItem",
                        "gap": 2,
                        "children": [
                            {
                                "type": "Box",
                                "direction": "column",
                                "gap": 1,
                                "minWidth": 0,
                                "children": [
                                    {
                                        "type": "Text",
                                        "value": (
                                            f"{choice.deck_name} · {choice.characters}"
                                            if choice.deck_name
                                            else choice.characters
                                        ),
                                        "weight": "semibold",
                                        "width": "100%",
                                    },
                                    {
                                        "type": "Text",
                                        "value": choice.detail,
                                        "size": "sm",
                                        "color": "secondary",
                                        "width": "100%",
                                    },
                                ],
                            }
                        ],
                    }
                )
                for option in choice.actions:
                    rows.append(
                        {
                            "type": "ListViewItem",
                            "gap": 2,
                            "children": [
                                {
                                    "type": "Text",
                                    "value": option.label,
                                    "width": "100%",
                                    "color": "primary",
                                }
                            ],
                            "onClickAction": {
                                "type": _RESUME_KANJI_ACTION,
                                "payload": {
                                    "capability": capability,
                                    "receipt_id": choice.receipt_id,
                                    "action": option.action,
                                },
                                "handler": "server",
                                "loadingBehavior": "container",
                                "streaming": True,
                            },
                        }
                    )
            return DynamicWidgetRoot.model_validate(
                {
                    "type": "ListView",
                    "limit": "auto",
                    "status": {
                        "text": "Unfinished kanji cards",
                        "icon": "book-open",
                    },
                    "children": rows,
                }
            )

        def _kanji_finish_selector_event(
            self,
            thread: Any,
            request_context: AssistantRequestContext,
            finish_choices: tuple[KanjiFinishChoice, ...],
        ) -> Any:
            capability = secrets.token_urlsafe(32)
            item_id = store.generate_item_id("message", thread, request_context)
            self._kanji_resumes[capability] = _KanjiResumeBinding(
                thread_id=thread.id,
                widget_item_id=item_id,
                actions=frozenset(
                    (choice.receipt_id, option.action)
                    for choice in finish_choices
                    for option in choice.actions
                ),
            )
            return ThreadItemDoneEvent(
                item=WidgetItem(
                    id=item_id,
                    thread_id=thread.id,
                    created_at=datetime.now(),
                    widget=self._kanji_finish_widget(
                        finish_choices,
                        capability=capability,
                    ),
                    copy_text=None,
                )
            )

        async def _list_kanji_finish_choices(self) -> tuple[KanjiFinishChoice, ...]:
            return _validate_kanji_finish_choices(
                await _call_callback(callbacks.list_kanji_finish_choices)
            )

        async def _list_operation_choices(self) -> tuple[OperationChoice, ...]:
            return _validate_operation_choices(
                await _call_callback(callbacks.list_operation_choices)
            )

        def _current_operation_selector_event(
            self,
            thread: Any,
            request_context: AssistantRequestContext,
            operation_choices: tuple[OperationChoice, ...],
        ) -> Any:
            try:
                return self._operation_selector_event(
                    thread,
                    request_context,
                    operation_choices,
                )
            except RevisionRefusal:
                self._clear_focus(thread)
                return self._operation_selector_event(
                    thread,
                    request_context,
                    operation_choices,
                )

        def get_stream_options(self, thread: Any, request_context: Any) -> Any:
            del thread, request_context
            return StreamOptions(allow_cancel=False)

        async def handle_stream_cancelled(
            self,
            thread: Any,
            pending_items: Any,
            context: Any,
        ) -> None:
            """Keep nothing a canceled stream had only started to say.

            ChatKit's default persists unfinished assistant messages. Here the
            only unfinished item is the live preview of a turn whose validated
            answer is the durable record, so saving it would put text into the
            transcript that no Assistant manifest ever answered with.
            """
            del thread, pending_items, context

        def _message_item(
            self, thread: Any, text: str | None, *, item_id: str | None = None
        ) -> Any:
            """One assistant message item; ``text`` of ``None`` opens an empty one.

            The empty shape is the item a live preview streams into, finished
            later under the same id with the answer the schema returned.
            """
            return AssistantMessageItem(
                id=item_id or store.generate_item_id("message", thread, context),
                thread_id=thread.id,
                created_at=datetime.now(),
                content=[] if text is None else [AssistantMessageContent(text=text)],
            )

        def _message_event(
            self, thread: Any, text: str, *, item_id: str | None = None
        ) -> Any:
            return ThreadItemDoneEvent(
                item=self._message_item(thread, text, item_id=item_id)
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
                            title="Confirm this exact action",
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
                            preview_url=plan.preview_url,
                            preview_label=plan.preview_label,
                        )
                    ],
                    "confirm": {
                        "label": plan.confirm_label,
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
            review: RevisionFinishReview | StagedContentFinishReview,
            *,
            capability: str,
        ) -> Any:
            if isinstance(review, StagedContentFinishReview):
                body = _confirmation_body(
                    title="Review this exact content and finish",
                    target=review.target,
                    effects=review.effects,
                    disclosures=review.disclosures,
                    fingerprint_label="Finish fingerprint",
                    fingerprint=review.request_fingerprint,
                    one_use_note=(
                        "Apply and finish is one-use. At the click, Janki re-plans "
                        "and refuses if the reviewed proposal, audio, or build "
                        "inputs changed."
                    ),
                    preview_url=review.preview_url,
                    preview_label=review.preview_label,
                )
                confirm_label = review.confirm_label
            else:
                body = _finish_review_body(review)
                confirm_label = "Apply and finish"
            return DynamicWidgetRoot.model_validate(
                {
                    "type": "Card",
                    "size": "full",
                    "children": [body],
                    "confirm": {
                        "label": confirm_label,
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

            if _is_deck_selector_message(message):
                if not choices:
                    yield NoticeEvent(
                        level="info",
                        title="No deck available to focus",
                        message=(
                            "No readable deck is available to focus right now. You "
                            "can still ask about the Japanese library or attach a "
                            "source."
                        ),
                    )
                    return
                yield self._selector_event(thread, request_context)
                return

            if message == MANAGE_OPERATIONS_MESSAGE:
                try:
                    operation_choices = await self._list_operation_choices()
                except (JankiError, RevisionRefusal, OSError, TypeError, ValueError) as error:
                    yield NoticeEvent(
                        level="danger",
                        title="Model-call recovery unavailable",
                        message=str(error),
                    )
                    return
                if not operation_choices:
                    yield self._message_event(
                        thread,
                        "No earlier model call is blocking spending or waiting for recovery.",
                    )
                    return
                yield self._message_event(
                    thread,
                    (
                        "Choose one exact recovery action below. Janki plans it "
                        "locally, so this status and recovery surface does not make "
                        "another model call."
                    ),
                )
                yield self._current_operation_selector_event(
                    thread,
                    request_context,
                    operation_choices,
                )
                return

            if message == RESUME_KANJI_MESSAGE:
                try:
                    finish_choices = await self._list_kanji_finish_choices()
                except (JankiError, RevisionRefusal, OSError, TypeError, ValueError) as error:
                    yield NoticeEvent(
                        level="danger",
                        title="Kanji card recovery unavailable",
                        message=str(error),
                    )
                    return
                if not finish_choices:
                    yield self._message_event(
                        thread,
                        "No kanji card batch is waiting to finish or download.",
                    )
                    return
                yield self._message_event(
                    thread,
                    (
                        "These kanji card batches are already confirmed and saved. "
                        "Resuming one continues that exact receipt locally — it asks "
                        "for nothing again, looks nothing up again, and makes no "
                        "model call."
                    ),
                )
                yield self._kanji_finish_selector_event(
                    thread,
                    request_context,
                    finish_choices,
                )
                return

            local_help = self._local_help(message)
            if local_help is not None:
                yield self._message_event(thread, local_help)
                return

            try:
                operation_choices = await self._list_operation_choices()
            except (JankiError, RevisionRefusal, OSError, TypeError, ValueError) as error:
                yield NoticeEvent(
                    level="danger",
                    title="Model-call recovery unavailable",
                    message=(
                        "Janki did not make a model call because the status of its "
                        f"earlier model calls could not be checked safely: {error}"
                    ),
                )
                return
            selection_lock = self._selection_locks.setdefault(
                thread.id,
                asyncio.Lock(),
            )
            async with selection_lock:
                fresh_metadata = (
                    await store.load_thread(thread.id, request_context)
                ).metadata
                current_metadata = dict(thread.metadata)
                for key in (
                    _ACTIVE_DECK_ID_KEY,
                    _ACTIVE_DECK_SCOPE_KEY,
                    _SELECTION_EPOCH_KEY,
                ):
                    if key in fresh_metadata:
                        current_metadata[key] = fresh_metadata[key]
                    else:
                        current_metadata.pop(key, None)
                if current_metadata != thread.metadata:
                    thread.metadata = current_metadata
            blocking_operations = tuple(
                choice for choice in operation_choices if choice.blocks_spending
            )
            if blocking_operations:
                yield NoticeEvent(
                    level="warning",
                    title="Earlier model call needs attention",
                    message=(
                        "Janki did not make a new model call. An earlier model call "
                        "left an unsettled result that needs your decision first. "
                        "Choose one exact action below; local deck selection remains "
                        "available. This settles that one earlier call and is not a "
                        "confirmation for each question."
                    ),
                )
                yield self._current_operation_selector_event(
                    thread,
                    request_context,
                    blocking_operations,
                )
                return

            try:
                selection = self._selection(thread)
            except RevisionRefusal as error:
                self._clear_focus(thread)
                yield NoticeEvent(
                    level="info",
                    title="Deck focus cleared",
                    message=f"{error} Janki continued without a deck focus.",
                )
                selection = None
            if thread.id in self._busy_threads:
                yield NoticeEvent(
                    level="warning",
                    message="Wait for this thread's current operation to finish.",
                )
                return

            # One queue, not two: a preview delta and a progress label are the
            # same turn's narration, and separate queues would let a drain
            # reorder them against each other.
            turn_queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def report_progress(label: str) -> None:
                normalized = label.strip()
                if normalized not in _CHAT_PROGRESS_LABELS:
                    raise ValueError("The chat service reported an unknown progress state.")
                loop.call_soon_threadsafe(
                    turn_queue.put_nowait, ("progress", normalized)
                )

            def report_preview(delta: str) -> None:
                loop.call_soon_threadsafe(turn_queue.put_nowait, ("preview", delta))

            async def execute_chat() -> ChatReply:
                history_lock = self._history_locks.setdefault(thread.id, asyncio.Lock())
                async with history_lock:
                    history = _bounded_chat_history(self._histories.get(thread.id, ()))
                    reply = _validate_chat_reply(
                        await _call_callback(
                            callbacks.chat,
                            deck_scope=(selection.choice.scope if selection else ""),
                            history=history,
                            message=message,
                            progress=report_progress,
                            preview=report_preview,
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
            preview_id: str | None = None

            def narrate(batch: list[tuple[str, str]]) -> list[Any]:
                """Render one drained batch, keeping its exact arrival order."""
                nonlocal preview_id
                rendered: list[Any] = []
                for kind, payload in batch:
                    if kind == "progress":
                        rendered.append(
                            ProgressUpdateEvent(text=payload, icon="write")
                        )
                        continue
                    if preview_id is None:
                        preview_id = store.generate_item_id(
                            "message", thread, request_context
                        )
                        rendered.append(
                            ThreadItemAddedEvent(
                                item=self._message_item(
                                    thread, None, item_id=preview_id
                                )
                            )
                        )
                        rendered.append(
                            ThreadItemUpdatedEvent(
                                item_id=preview_id,
                                update=AssistantMessageContentPartAdded(
                                    content_index=0,
                                    content=AssistantMessageContent(text=""),
                                ),
                            )
                        )
                    rendered.append(
                        ThreadItemUpdatedEvent(
                            item_id=preview_id,
                            update=AssistantMessageContentPartTextDelta(
                                content_index=0,
                                delta=payload,
                            ),
                        )
                    )
                return rendered

            def drain(first: tuple[str, str] | None = None) -> list[tuple[str, str]]:
                batch = [] if first is None else [first]
                while not turn_queue.empty():
                    batch.append(turn_queue.get_nowait())
                return batch

            try:
                try:
                    while not execution_task.done():
                        try:
                            item = await asyncio.wait_for(
                                turn_queue.get(), timeout=0.1
                            )
                        except TimeoutError:
                            continue
                        for event in narrate(drain(item)):
                            yield event
                    for event in narrate(drain()):
                        yield event
                    reply = await asyncio.shield(execution_task)
                except RevisionRefusal as error:
                    if preview_id is not None:
                        # The unvalidated preview is not an answer; it leaves the
                        # transcript before the refusal explains why.
                        yield ThreadItemRemovedEvent(item_id=preview_id)
                    yield NoticeEvent(
                        level="danger", title="Answer refused", message=str(error)
                    )
                    try:
                        operation_choices = await self._list_operation_choices()
                    except (
                        JankiError,
                        RevisionRefusal,
                        OSError,
                        TypeError,
                        ValueError,
                    ):
                        return
                    blocking_operations = tuple(
                        choice
                        for choice in operation_choices
                        if choice.blocks_spending
                    )
                    if blocking_operations:
                        yield NoticeEvent(
                            level="warning",
                            title="Earlier model call needs attention",
                            message=(
                                "An earlier or just-started model call now blocks "
                                "further model calls. Choose one exact action below; "
                                "Janki will not recover, end, or discard it without "
                                "your confirmation."
                            ),
                        )
                        yield self._current_operation_selector_event(
                            thread,
                            request_context,
                            blocking_operations,
                        )
                    return
                except Exception:
                    # A refusal is not the only way a turn ends. Every other
                    # failure leaves the same unvalidated preview on screen,
                    # where the error reported next would stand beside half an
                    # answer that no longer belongs to any turn.
                    if preview_id is not None:
                        yield ThreadItemRemovedEvent(item_id=preview_id)
                    raise

                if preview_id is None:
                    yield self._message_event(thread, reply.text)
                else:
                    # The validated answer replaces the preview whole: the
                    # streamed prose is the model's first pass at it, not a
                    # prefix of what the schema finally returned.
                    yield ThreadItemUpdatedEvent(
                        item_id=preview_id,
                        update=AssistantMessageContentPartDone(
                            content_index=0,
                            content=AssistantMessageContent(text=reply.text),
                        ),
                    )
                    yield self._message_event(
                        thread, reply.text, item_id=preview_id
                    )
                plan = reply.action
                if plan is not None:
                    instruction = reply.action_instruction
                    if instruction is None:  # guarded by _validate_chat_reply
                        raise AssertionError("validated planned replies carry an instruction")
                    capability = secrets.token_urlsafe(32)
                    item_id = store.generate_item_id("message", thread, request_context)
                    confirmation = RevisionConfirmation(
                        capability=capability,
                        deck_scope=(selection.choice.scope if selection else ""),
                        instruction=instruction,
                        expected_fingerprint=plan.request_fingerprint,
                        target=plan.target,
                    )
                    self._plans[capability] = _PlanBinding(
                        confirmation=confirmation,
                        thread_id=thread.id,
                        widget_item_id=item_id,
                        deck_id=(selection.choice.deck_id if selection else None),
                        selection_epoch=(selection.epoch if selection else 0),
                        progress_label=plan.progress_label,
                    )
                    yield ThreadItemDoneEvent(
                        item=WidgetItem(
                            id=item_id,
                            thread_id=thread.id,
                            created_at=datetime.now(),
                            widget=self._plan_widget(
                                plan,
                                instruction=instruction,
                                capability=capability,
                            ),
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
            if action.type == _RESUME_KANJI_ACTION:
                payload = action.payload
                capability = (
                    payload.get("capability") if isinstance(payload, dict) else None
                )
                receipt_id = (
                    payload.get("receipt_id") if isinstance(payload, dict) else None
                )
                receipt_action = (
                    payload.get("action") if isinstance(payload, dict) else None
                )
                binding = (
                    self._kanji_resumes.get(capability)
                    if isinstance(capability, str)
                    else None
                )
                if (
                    not isinstance(payload, dict)
                    or set(payload) != {"capability", "receipt_id", "action"}
                    or not isinstance(receipt_id, str)
                    or receipt_action not in {"resume", "download"}
                    or binding is None
                    or binding.thread_id != thread.id
                    or sender is None
                    or sender.id != binding.widget_item_id
                    or (receipt_id, receipt_action) not in binding.actions
                ):
                    if isinstance(capability, str):
                        self._kanji_resumes.pop(capability, None)
                    yield ErrorEvent(
                        message=(
                            "This kanji card recovery action is missing, stale, "
                            "already used, tampered with, or belongs elsewhere."
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
                self._kanji_resumes.pop(capability, None)

                progress_queue: asyncio.Queue[str] = asyncio.Queue()
                loop = asyncio.get_running_loop()

                def report_resume_progress(label: str) -> None:
                    normalized = label.strip()
                    if normalized not in _FINISH_PROGRESS_LABELS:
                        raise ValueError(
                            "The character-note finish reported an unknown progress state."
                        )
                    loop.call_soon_threadsafe(progress_queue.put_nowait, normalized)

                async def execute_resume() -> KanjiFinishResumption:
                    result = _validate_kanji_resumption(
                        await _call_callback(
                            callbacks.resume_kanji_finish,
                            receipt_id=receipt_id,
                            action=receipt_action,
                            progress=report_resume_progress,
                        )
                    )
                    await self._remember_assistant_result(thread.id, result.message)
                    return result

                execution_task = asyncio.create_task(execute_resume())
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
                        resumed = await asyncio.shield(execution_task)
                    except RevisionRefusal as error:
                        yield NoticeEvent(
                            level="danger",
                            title="Kanji card recovery refused",
                            message=str(error),
                        )
                        return
                    yield self._message_event(thread, resumed.message)
                    if not resumed.complete:
                        yield NoticeEvent(
                            level="warning",
                            title="Kanji cards still unfinished",
                            message=(
                                f"Receipt {resumed.receipt_id} remains in state "
                                f"{resumed.state}. Its already-written work is kept; "
                                "resume the same receipt again."
                            ),
                        )
                finally:
                    release_busy()
                return
            if action.type == _PREPARE_OPERATION_ACTION:
                payload = action.payload
                if not isinstance(payload, dict) or set(payload) != {
                    "capability",
                    "operation_id",
                    "action",
                    "accept_paid_output_loss",
                }:
                    yield ErrorEvent(
                        message="The paid-operation action was refused.",
                        allow_retry=False,
                    )
                    return
                capability = payload.get("capability")
                operation_id = payload.get("operation_id")
                operation_action = payload.get("action")
                accept_loss = payload.get("accept_paid_output_loss")
                binding = (
                    self._operation_selectors.get(capability)
                    if isinstance(capability, str)
                    else None
                )
                try:
                    current_selection = self._selection(thread)
                except RevisionRefusal:
                    current_selection = None
                requested = (operation_id, operation_action, accept_loss)
                if (
                    not isinstance(operation_id, str)
                    or operation_action
                    not in {"recover", "show_reply", "end", "forget"}
                    or not isinstance(accept_loss, bool)
                    or binding is None
                    or binding.thread_id != thread.id
                    or sender is None
                    or sender.id != binding.widget_item_id
                    or requested not in binding.actions
                    or not self._binding_is_current(
                        current_selection,
                        deck_id=binding.deck_id,
                        selection_epoch=binding.selection_epoch,
                    )
                ):
                    if isinstance(capability, str):
                        self._operation_selectors.pop(capability, None)
                    yield ErrorEvent(
                        message=(
                            "This paid-operation selector is missing, stale, "
                            "already used, tampered with, or belongs elsewhere."
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
                self._operation_selectors.pop(capability, None)
                deck_scope = (
                    current_selection.choice.scope
                    if current_selection is not None
                    else ""
                )
                try:
                    reply = _validate_chat_reply(
                        await _call_callback(
                            callbacks.prepare_operation_action,
                            operation_id=operation_id,
                            action=operation_action,
                            accept_paid_output_loss=accept_loss,
                            deck_scope=deck_scope,
                        )
                    )
                except (JankiError, RevisionRefusal, OSError, TypeError, ValueError) as error:
                    yield NoticeEvent(
                        level="danger",
                        title="Paid-operation action unavailable",
                        message=str(error),
                    )
                    return
                plan = reply.action
                instruction = reply.action_instruction
                if plan is None or instruction is None:
                    yield ErrorEvent(
                        message="The local paid-operation planner returned no exact action.",
                        allow_retry=False,
                    )
                    return
                yield self._message_event(thread, reply.text)
                confirm_capability = secrets.token_urlsafe(32)
                item_id = store.generate_item_id("message", thread, request_context)
                confirmation = RevisionConfirmation(
                    capability=confirm_capability,
                    deck_scope=deck_scope,
                    instruction=instruction,
                    expected_fingerprint=plan.request_fingerprint,
                    target=plan.target,
                )
                self._plans[confirm_capability] = _PlanBinding(
                    confirmation=confirmation,
                    thread_id=thread.id,
                    widget_item_id=item_id,
                    deck_id=(
                        current_selection.choice.deck_id
                        if current_selection is not None
                        else None
                    ),
                    selection_epoch=(
                        current_selection.epoch if current_selection is not None else 0
                    ),
                    progress_label=plan.progress_label,
                )
                yield ThreadItemDoneEvent(
                    item=WidgetItem(
                        id=item_id,
                        thread_id=thread.id,
                        created_at=datetime.now(),
                        widget=self._plan_widget(
                            plan,
                            instruction=instruction,
                            capability=confirm_capability,
                        ),
                        copy_text=None,
                    )
                )
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
                                "The deck focus cannot change while this thread's "
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
                    if deck_id == _ALL_LIBRARY_DECK_ID:
                        if current is None:
                            yield ErrorEvent(
                                message="All library is already active in this thread.",
                                allow_retry=False,
                            )
                            return
                        self._clear_focus(thread)
                        yield self._message_event(
                            thread,
                            (
                                "Continuing with all library. Deck-focused conversation "
                                "context and older focus-specific action cards in this "
                                "thread were cleared."
                            ),
                        )
                        yield self._selector_event(thread, request_context)
                        return
                    choice = choices_by_id[deck_id]
                    if current is not None and current.choice.deck_id == deck_id:
                        yield ErrorEvent(
                            message="That deck is already active in this thread.",
                            allow_retry=False,
                        )
                        return
                    if not choice.chat_supported:
                        yield ErrorEvent(
                            message=choice.unavailable_reason
                            or "That deck is not available for Assistant chat.",
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
                        or fresh_choice.chat_supported != choice.chat_supported
                    ):
                        yield ErrorEvent(
                            message="That deck changed after the selector was rendered.",
                            allow_retry=False,
                        )
                        return
                    if thread.id in self._busy_threads:
                        yield ErrorEvent(
                            message=(
                                "The deck focus cannot change while this thread's "
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
                            f"Focused on: {choice.label}. Focus-specific conversation "
                            "context and older focus-specific action cards in this "
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
                extraction_binding = self._extractions.get(capability)
                if (
                    extraction_binding is None
                    or extraction_binding.thread_id != thread.id
                    or sender is None
                    or sender.id != extraction_binding.widget_item_id
                    or fingerprint != extraction_binding.confirmation.expected_fingerprint
                ):
                    self._extractions.pop(capability, None)
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
                self._extractions.pop(capability, None)

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
                    checked = _validate_source_extraction_execution(result)
                    await self._remember_assistant_result(thread.id, checked.message)
                    return checked

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
                finish_binding = self._finishes.get(capability)
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
                ):
                    self._finishes.pop(capability, None)
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
                self._finishes.pop(capability, None)

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
                    checked = _validate_finish_execution(result)
                    await self._remember_assistant_result(thread.id, checked.message)
                    return checked

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

            binding = self._plans.get(capability)
            try:
                current_selection = self._selection(thread)
            except RevisionRefusal:
                current_selection = None
            current_scope = (
                current_selection.choice.scope if current_selection is not None else ""
            )
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
                or binding.confirmation.deck_scope != current_scope
            ):
                self._plans.pop(capability, None)
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
            self._plans.pop(capability, None)

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
                    checked = _validate_execution(result)
                except RevisionRefusal:
                    raise
                except Exception as error:
                    if (
                        not isinstance(result, RevisionExecution)
                        or not isinstance(result.message, str)
                        or not result.message.strip()
                    ):
                        raise
                    checked = RevisionExecution(
                        message=result.message,
                        finish=None,
                        finish_unavailable=_staged_finish_unavailable(error),
                    )
                if checked.remember_in_chat_context:
                    await self._remember_assistant_result(thread.id, checked.message)
                return checked

            execution_task = asyncio.create_task(execute())
            release_busy = self._track_durable_task(thread.id, execution_task)
            try:
                yield ProgressUpdateEvent(text=binding.progress_label, icon="write")

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
                    if result.complete:
                        return
                    if result.review_required is not None:
                        yield NoticeEvent(
                            level="info",
                            title="Review the proposed card changes",
                            message=result.review_required,
                        )
                        return
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
