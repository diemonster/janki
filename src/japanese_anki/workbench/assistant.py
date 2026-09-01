"""Deterministic ChatKit controller for one explicitly selected deck.

This module deliberately contains no OpenAI model client.  One user message is
an instruction to the injected revision planner; the returned plan is rendered
verbatim and the only mutating route is its one-use confirmation action.  The
application callback remains responsible for re-planning under its repository
locks and comparing ``request_fingerprint`` before it dispatches anything.

``chatkit`` is imported only by :func:`create_assistant_core`.  Importing the
ordinary workbench therefore does not make ChatKit a runtime requirement when
the assistant sidecar is disabled.
"""

from __future__ import annotations

import asyncio
import inspect
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, TypeVar

__all__ = [
    "AssistantCore",
    "AssistantRequestContext",
    "RevisionCallbacks",
    "RevisionConfirmation",
    "RevisionExecution",
    "RevisionPlan",
    "RevisionRefusal",
    "OwnerActionConfirmation",
    "OwnerActionExecution",
    "OwnerActionPlan",
    "ScopedMemoryStore",
    "create_assistant_core",
]


_CONFIRM_ACTION = "janki.revision.confirm"
_PREPARE_ACTIONS = {
    "apply": "janki.revision.apply.prepare",
    "audio": "janki.revision.audio.prepare",
    "build": "janki.revision.build.prepare",
}
_FOLLOWUP_CONFIRM_ACTIONS = {
    "apply": "janki.revision.apply.confirm",
    "audio": "janki.revision.audio.confirm",
    "build": "janki.revision.build.confirm",
}
_PROGRESS_LABELS = frozenset(
    {
        "Preparing revision",
        "Reading the source",
        "Checking the answer's shape",
        "Saving proposals",
        "Preparing proposal",
        "Re-reading proposal",
        "Applying revision",
        "Archiving proposal",
        "Preparing audio",
        "Creating audio",
        "Saving audio",
        "Cleaning up audio",
        "Preparing deck",
        "Building package",
        "Saving package",
    }
)
_T = TypeVar("_T")


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
    next_action: str | None = None
    continuation_context: str | None = None


@dataclass(frozen=True, slots=True)
class OwnerActionPlan:
    """Exact read-only plan for applying, voicing, or building the deck."""

    kind: str
    fingerprint: str
    target: str
    title: str
    effects: tuple[str, ...]
    disclosures: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class OwnerActionConfirmation:
    """One exact follow-up plan consumed before its application service runs."""

    capability: str
    deck_scope: str
    kind: str
    expected_fingerprint: str
    target: str


@dataclass(frozen=True, slots=True)
class OwnerActionExecution:
    """Durable outcome of one separately confirmed follow-up action."""

    message: str
    next_action: str | None = None
    continuation_context: str | None = None


class RevisionRefusal(RuntimeError):
    """A safe refusal that may be shown to the owner without a server error."""


class RevisionCallbacks(Protocol):
    """Application boundary used by the deterministic ChatKit controller."""

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

    def prepare_followup(
        self,
        kind: str,
        *,
        deck_scope: str,
        continuation_context: str,
    ) -> OwnerActionPlan | Awaitable[OwnerActionPlan]:
        """Prepare one exact, side-effect-free post-revision action."""

    def consume_followup(
        self,
        confirmation: OwnerActionConfirmation,
        *,
        progress: Callable[[str], None],
    ) -> OwnerActionExecution | Awaitable[OwnerActionExecution]:
        """Consume and execute one exact apply, audio, or build plan."""


@dataclass(frozen=True, slots=True)
class AssistantRequestContext:
    """Per-request scope passed through every ChatKit store operation."""

    deck_scope: str


@dataclass(frozen=True, slots=True)
class AssistantCore:
    """A ChatKit server and the context bound to its sole deck scope."""

    server: Any
    context: AssistantRequestContext
    store: ScopedMemoryStore


@dataclass(frozen=True, slots=True)
class _PlanBinding:
    confirmation: RevisionConfirmation | OwnerActionConfirmation
    thread_id: str
    widget_item_id: str


@dataclass(frozen=True, slots=True)
class _ContinuationBinding:
    kind: str
    context: str
    thread_id: str
    widget_item_id: str


class ScopedMemoryStore:
    """Small in-memory ChatKit store that refuses cross-deck contexts.

    Threads are convenience state, not paid-call provenance.  The injected
    revision application service owns durable journaling and staging.
    """

    _SCOPE_KEY = "janki_deck_scope"

    def __init__(self, deck_scope: str) -> None:
        if not deck_scope.strip():
            raise ValueError("deck_scope must not be blank")
        self.deck_scope = deck_scope
        self._threads: dict[str, Any] = {}
        self._items: dict[str, list[Any]] = {}
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
        del attachment
        self._context_ok(context)
        raise PermissionError("Attachments are disabled for the workbench assistant.")

    async def load_attachment(
        self,
        attachment_id: str,
        context: AssistantRequestContext,
    ) -> Any:
        self._context_ok(context)
        raise self._not_found(f"Attachments are disabled: {attachment_id}")

    async def delete_attachment(
        self,
        attachment_id: str,
        context: AssistantRequestContext,
    ) -> None:
        del attachment_id
        self._context_ok(context)
        raise PermissionError("Attachments are disabled for the workbench assistant.")


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
    if result.next_action not in {None, "apply"}:
        raise ValueError("A revision may offer only the separate apply action.")
    if (result.next_action is None) != (result.continuation_context is None):
        raise ValueError("A revision continuation must bind its durable result.")
    if result.continuation_context is not None and not result.continuation_context.strip():
        raise ValueError("A revision continuation context must not be blank.")
    return result


def _validate_owner_plan(plan: Any, expected_kind: str) -> OwnerActionPlan:
    if not isinstance(plan, OwnerActionPlan):
        raise TypeError("prepare_followup must return OwnerActionPlan")
    if plan.kind != expected_kind or plan.kind not in _PREPARE_ACTIONS:
        raise ValueError("The owner-action plan has the wrong operation kind.")
    if not plan.fingerprint.strip() or not plan.target.strip() or not plan.title.strip():
        raise ValueError("The owner-action plan is missing its exact identity.")
    if not plan.effects or any(not effect.strip() for effect in plan.effects):
        raise ValueError("The owner-action plan must name every intended effect.")
    if any(not disclosure.strip() for disclosure in plan.disclosures):
        raise ValueError("Owner-action disclosures must not be blank.")
    return plan


def _validate_owner_execution(
    result: Any,
    *,
    kind: str,
) -> OwnerActionExecution:
    if not isinstance(result, OwnerActionExecution):
        raise TypeError("consume_followup must return OwnerActionExecution")
    expected_next = {"apply": "audio", "audio": "build", "build": None}[kind]
    if result.next_action not in {None, expected_next}:
        raise ValueError("The owner-action result offered an invalid next action.")
    if (result.next_action is None) != (result.continuation_context is None):
        raise ValueError("An owner-action continuation must bind its durable result.")
    if result.continuation_context is not None and not result.continuation_context.strip():
        raise ValueError("An owner-action continuation context must not be blank.")
    if not result.message.strip():
        raise ValueError("The owner-action result message must not be blank.")
    return result


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

    store = ScopedMemoryStore(deck_scope)
    context = AssistantRequestContext(deck_scope=deck_scope)

    class _RevisionChatKitServer(ChatKitServer[AssistantRequestContext]):
        def __init__(self) -> None:
            super().__init__(store=store, attachment_store=None)
            self._plans: dict[str, _PlanBinding] = {}
            self._continuations: dict[str, _ContinuationBinding] = {}
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
        def _instruction(message: Any) -> str:
            if message is None:
                raise RevisionRefusal("Send one revision instruction to begin.")
            if message.attachments:
                raise RevisionRefusal("Attachments are disabled on this assistant.")
            if message.quoted_text and message.quoted_text.strip():
                raise RevisionRefusal("Quoted context is disabled; send one direct instruction.")
            if len(message.content) != 1:
                raise RevisionRefusal("Send exactly one plain-text revision instruction.")
            part = message.content[0]
            if getattr(part, "type", None) != "input_text":
                raise RevisionRefusal("Only a plain-text revision instruction is accepted.")
            instruction = part.text.strip()
            if not instruction:
                raise RevisionRefusal("The revision instruction must not be blank.")
            return instruction

        @staticmethod
        def _plan_widget(
            plan: RevisionPlan,
            *,
            instruction: str,
            capability: str,
        ) -> Any:
            children: list[dict[str, Any]] = [
                {"type": "Title", "value": "Confirm this exact revision"},
                {"type": "Text", "value": f"Target: {plan.target}", "weight": "semibold"},
                {"type": "Text", "value": "Your instruction"},
                {"type": "Text", "value": instruction},
                {"type": "Divider"},
                {"type": "Text", "value": "This confirmation will:"},
            ]
            children.extend({"type": "Text", "value": f"• {effect}"} for effect in plan.effects)
            if plan.disclosures:
                children.append({"type": "Divider"})
                children.extend(
                    {"type": "Text", "value": disclosure, "color": "secondary"}
                    for disclosure in plan.disclosures
                )
            children.extend(
                [
                    {"type": "Divider"},
                    {
                        "type": "Text",
                        "value": f"Request fingerprint: {plan.request_fingerprint}",
                        "size": "xs",
                        "color": "tertiary",
                    },
                    {
                        "type": "Text",
                        "value": (
                            "Confirm is one-use. At the click, janki re-plans and refuses "
                            "if the target or request changed."
                        ),
                        "size": "xs",
                        "color": "tertiary",
                    },
                ]
            )
            return DynamicWidgetRoot.model_validate(
                {
                    "type": "Card",
                    "children": children,
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
        def _owner_plan_widget(plan: OwnerActionPlan, *, capability: str) -> Any:
            children: list[dict[str, Any]] = [
                {"type": "Title", "value": plan.title},
                {"type": "Text", "value": f"Target: {plan.target}", "weight": "semibold"},
                {"type": "Divider"},
                {"type": "Text", "value": "This confirmation will:"},
            ]
            children.extend({"type": "Text", "value": f"• {effect}"} for effect in plan.effects)
            if plan.disclosures:
                children.append({"type": "Divider"})
                children.extend(
                    {"type": "Text", "value": disclosure, "color": "secondary"}
                    for disclosure in plan.disclosures
                )
            children.extend(
                [
                    {"type": "Divider"},
                    {
                        "type": "Text",
                        "value": f"Plan fingerprint: {plan.fingerprint}",
                        "size": "xs",
                        "color": "tertiary",
                    },
                    {
                        "type": "Text",
                        "value": (
                            "Confirm is one-use. janki re-plans under its ordinary locks "
                            "and refuses if any bound input changed."
                        ),
                        "size": "xs",
                        "color": "tertiary",
                    },
                ]
            )
            return DynamicWidgetRoot.model_validate(
                {
                    "type": "Card",
                    "children": children,
                    "confirm": {
                        "label": f"Confirm exact {plan.kind}",
                        "action": {
                            "type": _FOLLOWUP_CONFIRM_ACTIONS[plan.kind],
                            "payload": {
                                "capability": capability,
                                "plan_fingerprint": plan.fingerprint,
                            },
                            "handler": "server",
                            "loadingBehavior": "container",
                            "streaming": True,
                        },
                    },
                }
            )

        def _next_action_event(self, thread: Any, kind: str, action_context: str) -> Any:
            if kind not in _PREPARE_ACTIONS:
                raise ValueError("Unknown assistant continuation.")
            capability = secrets.token_urlsafe(32)
            item_id = store.generate_item_id("message", thread, context)
            self._continuations[capability] = _ContinuationBinding(
                kind=kind,
                context=action_context,
                thread_id=thread.id,
                widget_item_id=item_id,
            )
            labels = {
                "apply": "Review exact proposal apply plan",
                "audio": "Review exact example-audio plan",
                "build": "Review exact deck build plan",
            }
            preparation_disclosures = {
                "apply": (
                    "Preparing this plan renders the current and proposed form note "
                    "and selected Japanese/English examples in the OpenAI-hosted "
                    "ChatKit UI. It does not apply them."
                ),
                "audio": (
                    "Preparing this plan renders the exact Japanese audio inputs in "
                    "the OpenAI-hosted ChatKit UI. It does not contact the audio provider."
                ),
                "build": (
                    "Preparing this plan renders package targets and bound hashes. "
                    "It does not build the package."
                ),
            }
            return ThreadItemDoneEvent(
                item=WidgetItem(
                    id=item_id,
                    thread_id=thread.id,
                    created_at=datetime.now(),
                    widget=DynamicWidgetRoot.model_validate(
                        {
                            "type": "Card",
                            "children": [
                                {
                                    "type": "Text",
                                    "value": (
                                        "The prior action finished durably. The next action "
                                        "has not been planned or authorized."
                                    ),
                                },
                                {
                                    "type": "Text",
                                    "value": preparation_disclosures[kind],
                                    "color": "secondary",
                                },
                                {
                                    "type": "Button",
                                    "label": labels[kind],
                                    "onClickAction": {
                                        "type": _PREPARE_ACTIONS[kind],
                                        "payload": {"continuation": capability},
                                        "handler": "server",
                                    },
                                },
                            ],
                        }
                    ),
                    copy_text=None,
                )
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
                instruction = self._instruction(input_user_message)
                plan = _validate_plan(
                    await _call_callback(
                        callbacks.prepare_revision,
                        deck_scope=deck_scope,
                        instruction=instruction,
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
                instruction=instruction,
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
                        instruction=instruction,
                        capability=capability,
                    ),
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
            prepare_kind = next(
                (kind for kind, name in _PREPARE_ACTIONS.items() if action.type == name),
                None,
            )
            if prepare_kind is not None:
                payload = action.payload
                if not isinstance(payload, dict) or set(payload) != {"continuation"}:
                    yield ErrorEvent(
                        message="The prepare-action payload was refused.",
                        allow_retry=False,
                    )
                    return
                continuation = payload.get("continuation")
                if not isinstance(continuation, str):
                    yield ErrorEvent(
                        message="The prepare-action payload was refused.",
                        allow_retry=False,
                    )
                    return
                next_binding = self._continuations.pop(continuation, None)
                if (
                    next_binding is None
                    or next_binding.kind != prepare_kind
                    or next_binding.thread_id != thread.id
                    or sender is None
                    or sender.id != next_binding.widget_item_id
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
                    plan = _validate_owner_plan(
                        await _call_callback(
                            callbacks.prepare_followup,
                            prepare_kind,
                            deck_scope=deck_scope,
                            continuation_context=next_binding.context,
                        ),
                        prepare_kind,
                    )
                except RevisionRefusal as error:
                    yield NoticeEvent(level="warning", message=str(error))
                    return
                capability = secrets.token_urlsafe(32)
                item_id = store.generate_item_id("message", thread, request_context)
                confirmation = OwnerActionConfirmation(
                    capability=capability,
                    deck_scope=deck_scope,
                    kind=prepare_kind,
                    expected_fingerprint=plan.fingerprint,
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
                        widget=self._owner_plan_widget(plan, capability=capability),
                        copy_text=None,
                    )
                )
                return

            confirm_kind = next(
                (
                    kind
                    for kind, name in _FOLLOWUP_CONFIRM_ACTIONS.items()
                    if action.type == name
                ),
                None,
            )
            if confirm_kind is not None:
                payload = action.payload
                if not isinstance(payload, dict) or set(payload) != {
                    "capability",
                    "plan_fingerprint",
                }:
                    yield ErrorEvent(
                        message="The confirmation payload was refused.",
                        allow_retry=False,
                    )
                    return
                capability = payload.get("capability")
                fingerprint = payload.get("plan_fingerprint")
                if not isinstance(capability, str) or not isinstance(fingerprint, str):
                    yield ErrorEvent(
                        message="The confirmation payload was refused.",
                        allow_retry=False,
                    )
                    return
                binding = self._plans.pop(capability, None)
                confirmation = None if binding is None else binding.confirmation
                if (
                    not isinstance(confirmation, OwnerActionConfirmation)
                    or confirmation.kind != confirm_kind
                    or binding is None
                    or binding.thread_id != thread.id
                    or sender is None
                    or sender.id != binding.widget_item_id
                    or fingerprint != confirmation.expected_fingerprint
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
                        raise ValueError(
                            "The application service reported an unknown progress state."
                        )
                    loop.call_soon_threadsafe(progress_queue.put_nowait, normalized)

                async def execute_followup() -> OwnerActionExecution:
                    result = await _call_callback(
                        callbacks.consume_followup,
                        confirmation,
                        progress=report_progress,
                    )
                    return _validate_owner_execution(result, kind=confirm_kind)

                execution_task = asyncio.create_task(execute_followup())
                self._durable_tasks.add(execution_task)

                def forget_followup(done: asyncio.Task[Any]) -> None:
                    self._durable_tasks.discard(done)
                    if not done.cancelled():
                        done.exception()

                execution_task.add_done_callback(forget_followup)
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
                    result = await asyncio.shield(execution_task)
                except RevisionRefusal as error:
                    yield NoticeEvent(
                        level="danger",
                        title=f"{confirm_kind.title()} refused",
                        message=str(error),
                    )
                    return
                yield self._message_event(thread, result.message)
                if result.next_action is not None:
                    assert result.continuation_context is not None
                    yield self._next_action_event(
                        thread,
                        result.next_action,
                        result.continuation_context,
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
                or not isinstance(binding.confirmation, RevisionConfirmation)
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
            if result.next_action is not None:
                assert result.continuation_context is not None
                yield self._next_action_event(
                    thread,
                    result.next_action,
                    result.continuation_context,
                )

    server = _RevisionChatKitServer()
    return AssistantCore(server=server, context=context, store=store)
