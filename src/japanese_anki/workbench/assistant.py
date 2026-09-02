"""ChatKit controller for conversation over one explicitly selected deck.

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
    "AssistantCore",
    "AssistantRequestContext",
    "ChatReply",
    "RevisionCallbacks",
    "RevisionConfirmation",
    "RevisionExecution",
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
_EXTRACT_CONFIRM_ACTION = "janki.extraction.confirm"
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

    bounded = [
        item
        for item in entries
        if wire_size([item]) <= _MAX_CHAT_HISTORY_BYTES
    ][-_MAX_CHAT_HISTORY_ENTRIES:]
    while bounded and wire_size(bounded) > _MAX_CHAT_HISTORY_BYTES:
        del bounded[0]
    return tuple(bounded)


@dataclass(frozen=True, slots=True)
class ChatReply:
    """One non-mutating assistant answer rendered as ordinary prose."""

    text: str


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
class RevisionExecution:
    """Durable result of a confirmed revision operation."""

    message: str = (
        "The revision proposal is ready. Refresh the workbench dashboard to review it."
    )
    review_url: str | None = None


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
        """Re-plan, compare, authorize, execute, and durably stage the result."""

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
    """A ChatKit server and the context bound to its sole deck scope."""

    server: Any
    context: AssistantRequestContext
    store: ScopedMemoryStore
    attachment_store: AssistantAttachmentStore | None = None


@dataclass(frozen=True, slots=True)
class _PlanBinding:
    confirmation: RevisionConfirmation
    thread_id: str
    widget_item_id: str


@dataclass(frozen=True, slots=True)
class _MessageBinding:
    message: str
    thread_id: str
    widget_item_id: str


@dataclass(frozen=True, slots=True)
class _ExtractionBinding:
    confirmation: SourceExtractionConfirmation
    thread_id: str
    widget_item_id: str


class ScopedMemoryStore:
    """Small in-memory ChatKit store that refuses cross-deck contexts.

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
    if result.review_url is not None and result.review_url != "":
        raise ValueError(
            "Review links remain disabled until the sidecar has its own "
            "least-authority review route."
        )
    if not result.message.strip():
        raise ValueError("The revision result message must not be blank.")
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
                    "children": [
                        {"type": "Text", "value": effect, "width": "100%"}
                    ],
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
    deck_scope: str,
    attachment_store: AssistantAttachmentStore | None = None,
) -> AssistantCore:
    """Create the scoped deterministic ChatKit server.

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

    store = ScopedMemoryStore(deck_scope, attachment_store=attachment_store)
    context = AssistantRequestContext(deck_scope=deck_scope)

    class _JankiChatKitServer(ChatKitServer[AssistantRequestContext]):
        def __init__(self) -> None:
            super().__init__(store=store, attachment_store=attachment_store)
            self._messages: dict[str, _MessageBinding] = {}
            self._plans: dict[str, _PlanBinding] = {}
            self._extractions: dict[str, _ExtractionBinding] = {}
            self._histories: dict[str, list[tuple[str, str]]] = {}
            self._history_locks: dict[str, asyncio.Lock] = {}
            self._durable_tasks: set[asyncio.Task[Any]] = set()

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
            if request_context.deck_scope != deck_scope:
                yield ErrorEvent(message="The requested deck scope was refused.", allow_retry=False)
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
                    history = _bounded_chat_history(
                        self._histories.get(thread.id, ())
                    )
                    reply = _validate_chat_reply(
                        await _call_callback(
                            callbacks.chat,
                            deck_scope=deck_scope,
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
            self._durable_tasks.add(execution_task)

            def forget_chat(done: asyncio.Task[Any]) -> None:
                self._durable_tasks.discard(done)
                if not done.cancelled():
                    done.exception()

            execution_task.add_done_callback(forget_chat)
            try:
                while not execution_task.done():
                    try:
                        label = await asyncio.wait_for(progress_queue.get(), timeout=0.1)
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
                yield NoticeEvent(level="danger", title="Answer refused", message=str(error))
                return

            yield self._message_event(thread, reply.text)
            capability = secrets.token_urlsafe(32)
            item_id = store.generate_item_id("message", thread, request_context)
            self._messages[capability] = _MessageBinding(
                message=message,
                thread_id=thread.id,
                widget_item_id=item_id,
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

        async def action(
            self,
            thread: Any,
            action: Any,
            sender: Any,
            request_context: AssistantRequestContext,
        ) -> Any:
            if request_context.deck_scope != deck_scope:
                yield ErrorEvent(message="That assistant action was refused.", allow_retry=False)
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
                    or fingerprint
                    != extraction_binding.confirmation.expected_fingerprint
                ):
                    yield ErrorEvent(
                        message=(
                            "This extraction confirmation is missing, stale, already "
                            "used, or belongs elsewhere."
                        ),
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
                self._durable_tasks.add(execution_task)

                def forget_extraction(done: asyncio.Task[Any]) -> None:
                    self._durable_tasks.discard(done)
                    if not done.cancelled():
                        done.exception()

                execution_task.add_done_callback(forget_extraction)
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
                if (
                    message_binding is None
                    or message_binding.thread_id != thread.id
                    or sender is None
                    or sender.id != message_binding.widget_item_id
                ):
                    yield ErrorEvent(
                        message=(
                            "This prepare action is missing, stale, already used, "
                            "or belongs elsewhere."
                        ),
                        allow_retry=False,
                    )
                    return
                try:
                    plan = _validate_plan(
                        await _call_callback(
                            callbacks.prepare_revision,
                            deck_scope=deck_scope,
                            instruction=message_binding.message,
                        )
                    )
                except RevisionRefusal as error:
                    yield NoticeEvent(level="warning", message=str(error))
                    return
                capability = secrets.token_urlsafe(32)
                item_id = store.generate_item_id("message", thread, request_context)
                confirmation = RevisionConfirmation(
                    capability=capability,
                    deck_scope=deck_scope,
                    instruction=message_binding.message,
                    expected_fingerprint=plan.request_fingerprint,
                    target=plan.target,
                )
                self._plans[capability] = _PlanBinding(
                    confirmation=confirmation,
                    thread_id=thread.id,
                    widget_item_id=item_id,
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
            if (
                binding is None
                or binding.thread_id != thread.id
                or sender is None
                or sender.id != binding.widget_item_id
                or fingerprint != binding.confirmation.expected_fingerprint
            ):
                yield ErrorEvent(
                    message=(
                        "This confirmation is missing, stale, already used, "
                        "or belongs elsewhere."
                    ),
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
                return _validate_execution(result)

            execution_task = asyncio.create_task(execute())
            self._durable_tasks.add(execution_task)

            def forget_task(done: asyncio.Task[Any]) -> None:
                self._durable_tasks.discard(done)
                if done.cancelled():
                    return
                # Retrieve exceptions even when the browser disconnected and no
                # action generator remains to await the durable operation.
                done.exception()

            execution_task.add_done_callback(forget_task)
            yield ProgressUpdateEvent(text="Preparing revision", icon="write")

            try:
                while not execution_task.done():
                    try:
                        label = await asyncio.wait_for(progress_queue.get(), timeout=0.1)
                    except TimeoutError:
                        continue
                    yield ProgressUpdateEvent(text=label, icon="write")
                while not progress_queue.empty():
                    yield ProgressUpdateEvent(text=progress_queue.get_nowait(), icon="write")
                result = await asyncio.shield(execution_task)
            except RevisionRefusal as error:
                yield NoticeEvent(level="danger", title="Revision refused", message=str(error))
                return

            yield self._message_event(thread, result.message)

    server = _JankiChatKitServer()
    return AssistantCore(
        server=server,
        context=context,
        store=store,
        attachment_store=attachment_store,
    )
