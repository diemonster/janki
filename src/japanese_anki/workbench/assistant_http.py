"""Isolated loopback HTTP origin for the workbench ChatKit surface.

The hosted ChatKit JavaScript never runs in the main workbench authority
origin. This sidecar has its own random path token, one repository-scoped
in-memory store with optional thread-local deck focus, and no access to the
workbench session or CSRF secrets. Opaque one-use capabilities bind deck-focus
changes and exact confirmed actions to the server-rendered item that created
them.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import json
import queue
import re
import secrets
import threading
from collections.abc import Coroutine
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from japanese_anki.localhttp import MAX_BODY_BYTES, LocalOnlyHandler, LocalOnlyServer, bind_loopback
from japanese_anki.workbench.assistant import (
    CAPABILITIES_MESSAGE,
    MANAGE_EXTRACTION_BATCHES_MESSAGE,
    MANAGE_OPERATIONS_MESSAGE,
    RESUME_KANJI_MESSAGE,
    SHOW_DECKS_MESSAGE,
    SOURCE_HELP_MESSAGE,
    AssistantCore,
    AssistantDeckChoice,
    RevisionCallbacks,
    create_assistant_core,
)
from japanese_anki.workbench.assistant_attachments import (
    MAX_ASSISTANT_ATTACHMENT_BYTES,
    AssistantAttachmentError,
    LocalAssistantAttachmentStore,
)
from japanese_anki.workbench.assistant_packages import (
    AssistantPackageError,
    LocalAssistantPackageStore,
)
from japanese_anki.workbench.assistant_previews import (
    AssistantPreviewError,
    LocalAssistantPreviewStore,
)
from japanese_anki.workbench.source_part_editor import (
    LocalSourcePartEditorStore,
    SourcePartEditorError,
)

__all__ = [
    "AssistantHTTPServer",
    "AssistantSidecar",
    "AsyncLoopBridge",
    "create_assistant_sidecar",
    "start_assistant_sidecar",
]


_CHATKIT_CDN = "https://cdn.platform.openai.com"
_CHATKIT_SCRIPT = f"{_CHATKIT_CDN}/deployments/chatkit/chatkit.js"
# The hosted component installs one fixed bootstrap style in this origin's
# document before it creates its iframe.  Permit that exact content without
# widening the isolated origin to arbitrary inline styles.
_CHATKIT_STYLE_SOURCE = "'sha256-G2shiuZXM1qoGNHm7OQ6u7Ye45SO8f8LKO17q0kfGvw='"
_TOKEN = re.compile(r"^[A-Za-z0-9_-]{32,}$")
_DENIED_OPERATIONS = frozenset(
    {
        "input.transcribe",
        "items.feedback",
        "threads.add_client_tool_output",
        "threads.add_structured_input",
        "threads.retry_after_item",
        "threads.sync_custom_action",
    }
)
_SENTINEL = object()


class AsyncLoopBridge:
    """One long-lived asyncio loop shared by every threaded HTTP request."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._closed = False
        self._state_lock = threading.Lock()
        self._background: set[concurrent.futures.Future[Any]] = set()
        self._thread = threading.Thread(
            target=self._run,
            name="janki-chatkit-asyncio",
            daemon=False,
        )
        self._thread.start()
        self._ready.wait()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()
        self._loop.run_until_complete(self._loop.shutdown_asyncgens())
        self._loop.run_until_complete(self._loop.shutdown_default_executor())
        self._loop.close()

    def submit(
        self,
        coroutine: Coroutine[Any, Any, Any],
        *,
        durable: bool = False,
    ) -> concurrent.futures.Future[Any]:
        with self._state_lock:
            if self._closed:
                coroutine.close()
                raise RuntimeError("The ChatKit event loop is closed.")
            future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
            if durable:
                self._background.add(future)

                def finished(done: concurrent.futures.Future[Any]) -> None:
                    # Always retrieve an exception: a disconnected browser no
                    # longer has a handler waiting on this future.
                    with contextlib.suppress(concurrent.futures.CancelledError):
                        done.exception()
                    with self._state_lock:
                        self._background.discard(done)

                future.add_done_callback(finished)
            return future

    def call(self, coroutine: Coroutine[Any, Any, Any]) -> Any:
        return self.submit(coroutine).result()

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            pending = tuple(self._background)
        # A confirmed action owns a durable provider/journal transaction.  Do
        # not cancel it because its page or the surrounding server closed.
        for future in pending:
            # The application callback owns durable refusal/recovery state.
            with contextlib.suppress(Exception):
                future.result()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()


class AssistantHTTPServer(LocalOnlyServer):
    """Loopback server carrying the repository-scoped ChatKit sidecar."""

    assistant_core: AssistantCore
    bridge: AsyncLoopBridge
    session_token: str
    shell_path: str
    api_path: str
    script_path: str
    style_path: str
    upload_path_prefix: str | None
    download_path_prefix: str
    preview_path_prefix: str
    editor_path_prefix: str
    attachment_store: LocalAssistantAttachmentStore | None
    package_store: LocalAssistantPackageStore | None
    preview_store: LocalAssistantPreviewStore | None
    source_part_editor_store: LocalSourcePartEditorStore | None
    deck_choices: tuple[AssistantDeckChoice, ...]


class _AssistantHandler(LocalOnlyHandler):
    server: AssistantHTTPServer
    error_title = "janki assistant"
    content_security_policy = (
        "default-src 'none'; "
        f"script-src 'self' {_CHATKIT_CDN}; "
        f"style-src 'self' {_CHATKIT_STYLE_SOURCE}; connect-src 'self'; "
        f"frame-src {_CHATKIT_CDN}; "
        f"img-src data: {_CHATKIT_CDN}; "
        f"font-src {_CHATKIT_CDN}; "
        "base-uri 'none'; form-action 'none'; frame-ancestors 'none'; "
        "object-src 'none'"
    )

    def _start_response(
        self,
        status: int,
        *,
        content_type: str = "text/html; charset=utf-8",
        content_length: int | None = None,
        content_disposition: str | None = None,
        content_security_policy: str | None = None,
    ) -> None:
        """Use no-referrer on this third-party-script isolation origin.

        ``content_security_policy`` overrides the shell policy for exactly one
        response. A rendered card preview ships the policy its own bytes were
        built for; the ChatKit shell's policy is never widened to accommodate
        it.
        """

        self.send_response(status)
        self.send_header("Content-Type", content_type)
        if content_length is not None:
            self.send_header("Content-Length", str(content_length))
        if content_disposition is not None:
            self.send_header("Content-Disposition", content_disposition)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header(
            "Content-Security-Policy",
            content_security_policy or self.content_security_policy,
        )
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if not self._request_is_local():
            self._error(403, "The request host or origin was refused.")
            return
        if self.path == self.server.shell_path:
            self._send(200, _shell_html(self.server))
        elif self.path == self.server.script_path:
            self._send(
                200,
                _application_javascript(self.server),
                content_type="text/javascript; charset=utf-8",
            )
        elif self.path == self.server.style_path:
            self._send(200, _stylesheet(), content_type="text/css; charset=utf-8")
        elif self.path.startswith(self.server.download_path_prefix):
            self._send_package(self.path.removeprefix(self.server.download_path_prefix))
        elif self.path.startswith(self.server.preview_path_prefix):
            self._send_preview(self.path.removeprefix(self.server.preview_path_prefix))
        elif self.path.startswith(self.server.editor_path_prefix):
            self._send_source_part_editor(
                self.path.removeprefix(self.server.editor_path_prefix)
            )
        else:
            self._error(404, "This assistant route does not exist.")

    def _send_preview(self, token: str) -> None:
        """Serve one retained preview document under its own exact policy.

        This is a read. It consumes no capability, writes no receipt, and
        touches no repository file: the bytes were rendered before the link
        existed and are served back unchanged.
        """

        store = self.server.preview_store
        if store is None or not token or not _TOKEN.fullmatch(token):
            self._error(404, "This assistant route does not exist.")
            return
        try:
            snapshot = store.read(token)
        except AssistantPreviewError as error:
            self._error(409, str(error))
            return
        self._start_response(
            200,
            content_type="text/html; charset=utf-8",
            content_length=len(snapshot.html),
            content_security_policy=snapshot.content_security_policy,
        )
        self.wfile.write(snapshot.html)
        self.close_connection = True

    def _send_source_part_editor(self, token: str) -> None:
        """Serve one open region editor under its own exact policy.

        A read, exactly as a card preview is: the contact sheet was rendered
        before the link existed and is served back unchanged. Nothing here
        publishes a part.
        """

        store = self.server.source_part_editor_store
        if store is None or not token or not _TOKEN.fullmatch(token):
            self._error(404, "This assistant route does not exist.")
            return
        try:
            document = store.read(token)
        except SourcePartEditorError as error:
            self._error(409, str(error))
            return
        self._start_response(
            200,
            content_type="text/html; charset=utf-8",
            content_length=len(document.html),
            content_security_policy=document.content_security_policy,
        )
        self.wfile.write(document.html)
        self.close_connection = True

    def _source_part_editor_action(self, token: str) -> None:
        """One owner control from an open editor: render a plan, or publish it.

        Preparing source parts is ordinary local work — an explicit request and
        the owner's own page or region choice authorize it — so this route
        consumes no capability and asks for no second confirmation. It also
        grants nothing else: the editor's session decides which source may be
        named, and publication still binds the exact plan fingerprint the owner
        reviewed.
        """

        store = self.server.source_part_editor_store
        if store is None or not token or not _TOKEN.fullmatch(token):
            self._error(404, "This assistant route does not exist.")
            return
        body = self._read_json_body()
        if body is None:
            return
        try:
            parsed = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._error(400, "The source-part request was not valid JSON.")
            return
        try:
            payload = store.act(token, parsed)
        except SourcePartEditorError as error:
            self._send_json(409, {"ok": False, "error": str(error)})
            return
        except Exception as error:  # noqa: BLE001 - one owner-facing refusal
            self._send_json(409, {"ok": False, "error": str(error)})
            return
        self._send_json(200, payload)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._start_response(
            status,
            content_type="application/json; charset=utf-8",
            content_length=len(encoded),
        )
        self.wfile.write(encoded)
        self.close_connection = True

    def _send_package(self, token: str) -> None:
        """Serve only the exact package a complete finish receipt still proves."""

        store = self.server.package_store
        if store is None or not token or not _TOKEN.fullmatch(token):
            self._error(404, "This assistant route does not exist.")
            return
        try:
            filename, payload = store.read(token)
        except AssistantPackageError as error:
            self._error(409, str(error))
            return
        self._start_response(
            200,
            content_type="application/octet-stream",
            content_length=len(payload),
            content_disposition=f'attachment; filename="{filename}"',
        )
        self.wfile.write(payload)
        self.close_connection = True

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path != self.server.api_path and not self.path.startswith(
            self.server.editor_path_prefix
        ):
            self._error(404, "This assistant route does not exist.")
            return
        request_origin = self._exact_origin()
        if not self._request_is_local() or request_origin is None:
            self._error(403, "The request host or origin was refused.")
            return
        if self.path != self.server.api_path:
            self._source_part_editor_action(
                self.path.removeprefix(self.server.editor_path_prefix)
            )
            return
        body = self._read_json_body()
        if body is None:
            return
        try:
            parsed = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._error(400, "The ChatKit request was not valid JSON.")
            return
        refusal = _request_refusal(
            parsed,
            attachment_store=self.server.attachment_store,
        )
        if refusal is not None:
            self._error(400, refusal)
            return

        try:
            result = self.server.bridge.call(
                self.server.assistant_core.server.process(
                    body,
                    replace(
                        self.server.assistant_core.context,
                        request_origin=request_origin,
                    ),
                )
            )
        except AssistantAttachmentError as error:
            self._error(409, str(error))
            return
        except Exception:
            self._error(400, "The ChatKit request was refused.")
            return

        stream = getattr(result, "json_events", None)
        if stream is None:
            payload = getattr(result, "json", None)
            if not isinstance(payload, bytes):
                self._error(500, "The ChatKit response was invalid.")
                return
            self._send(200, payload, content_type="application/json")
            return
        self._stream_sse(stream)

    def do_PUT(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        prefix = self.server.upload_path_prefix
        attachment_store = self.server.attachment_store
        if (
            prefix is None
            or attachment_store is None
            or not self.path.startswith(prefix)
        ):
            self._error(404, "This assistant route does not exist.")
            return
        if not self._request_is_local() or not self._has_exact_origin():
            self._error(403, "The request host or origin was refused.")
            return
        coordinates = self.path.removeprefix(prefix).split("/")
        if (
            len(coordinates) != 2
            or not coordinates[0]
            or not coordinates[1]
            or any(character in coordinates[1] for character in "?#")
        ):
            self._error(404, "This assistant route does not exist.")
            return
        declared_content_length = self._upload_content_length()
        if declared_content_length is None:
            return
        try:
            attachment_store.accept_upload_stream(
                coordinates[0],
                coordinates[1],
                self.rfile,
                declared_content_length=declared_content_length,
            )
        except AssistantAttachmentError as error:
            self._error(409, str(error))
            return
        self._start_response(204, content_length=0)

    def _has_exact_origin(self) -> bool:
        return self._exact_origin() is not None

    def _exact_origin(self) -> str | None:
        origins = self.headers.get_all("Origin") or []
        hosts = self.headers.get_all("Host") or []
        if len(hosts) != 1 or origins != [f"http://{hosts[0]}"]:
            return None
        return origins[0]

    def _read_json_body(self) -> bytes | None:
        if self.headers.get_all("Transfer-Encoding"):
            self._error(400, "Transfer-encoded request bodies are refused.")
            return None
        if self.headers.get_all("Content-Encoding"):
            self._error(400, "Encoded request bodies are refused.")
            return None
        content_types = self.headers.get_all("Content-Type") or []
        if len(content_types) != 1 or content_types[0].lower() != "application/json":
            self._error(415, "ChatKit accepts exactly application/json.")
            return None
        lengths = self.headers.get_all("Content-Length") or []
        if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit():
            self._error(411, "One decimal Content-Length is required.")
            return None
        length = int(lengths[0])
        if length < 1:
            self._error(400, "The ChatKit request body is empty.")
            return None
        if length > MAX_BODY_BYTES:
            self._error(413, "The ChatKit request body is too large.")
            return None
        try:
            body = self.rfile.read(length)
        except OSError:
            self._error(408, "The ChatKit request body timed out.")
            return None
        if len(body) != length:
            self._error(400, "The ChatKit request body ended early.")
            return None
        return body

    def _upload_content_length(self) -> int | None:
        if self.headers.get_all("Transfer-Encoding"):
            self._error(400, "Transfer-encoded uploads are refused.")
            return None
        if self.headers.get_all("Content-Encoding"):
            self._error(400, "Encoded uploads are refused.")
            return None
        lengths = self.headers.get_all("Content-Length") or []
        if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit():
            self._error(411, "One decimal Content-Length is required.")
            return None
        length = int(lengths[0])
        if length < 1:
            self._error(400, "The attachment upload is empty.")
            return None
        if length > MAX_ASSISTANT_ATTACHMENT_BYTES:
            limit_mib = MAX_ASSISTANT_ATTACHMENT_BYTES // (1024 * 1024)
            self._error(
                413,
                f"The attachment upload exceeds Janki's {limit_mib} MiB limit.",
            )
            return None
        return length

    def _stream_sse(self, stream: Any) -> None:
        chunks: queue.Queue[Any] = queue.Queue()

        async def pump() -> None:
            try:
                async for chunk in stream:
                    chunks.put(bytes(chunk))
            except Exception:
                chunks.put(
                    b'data: {"type":"error","code":"custom",'
                    b'"message":"The assistant stream failed.","allow_retry":false}\n\n'
                )
            finally:
                chunks.put(_SENTINEL)

        self.server.bridge.submit(pump(), durable=True)
        self._start_response(200, content_type="text/event-stream; charset=utf-8")
        while True:
            chunk = chunks.get()
            if chunk is _SENTINEL:
                return
            try:
                self.wfile.write(chunk)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                # The pump remains retained on the long-lived loop.  In
                # particular, a confirmed action continues through its durable
                # application callback even though this handler is gone.
                return


def _request_refusal(
    payload: Any,
    *,
    attachment_store: LocalAssistantAttachmentStore | None = None,
) -> str | None:
    if not isinstance(payload, dict):
        return "The ChatKit request must be a JSON object."
    operation = payload.get("type")
    if not isinstance(operation, str):
        return "The ChatKit request has no operation type."
    if operation in _DENIED_OPERATIONS:
        return "That ChatKit capability is disabled."
    if operation in {"attachments.create", "attachments.delete"}:
        if attachment_store is None:
            return "Attachments are disabled on this assistant."
        return None
    if operation not in {
        "attachments.create",
        "attachments.delete",
        "items.list",
        "threads.add_user_message",
        "threads.create",
        "threads.custom_action",
        "threads.delete",
        "threads.get_by_id",
        "threads.list",
        "threads.update",
    }:
        return "That ChatKit operation is not supported."
    if operation not in {"threads.create", "threads.add_user_message"}:
        return None
    params = payload.get("params")
    message = params.get("input") if isinstance(params, dict) else None
    if not isinstance(message, dict):
        return "The ChatKit message shape was refused."
    attachments = message.get("attachments")
    if not isinstance(attachments, list):
        return "The message attachment list was refused."
    if attachments and attachment_store is None:
        return "Attachments are disabled on this assistant."
    if len(attachments) > 1 or any(not isinstance(item, str) for item in attachments):
        return "Attach one registered source at a time."
    if attachments:
        try:
            attachment_store.assert_ready(attachments[0])
        except AssistantAttachmentError as error:
            return str(error)
    quoted_text = message.get("quoted_text")
    if quoted_text is not None and quoted_text != "":
        return "Quoted context is disabled on this assistant."
    content = message.get("content")
    if not isinstance(content, list) or len(content) > 1:
        return "Send at most one plain-text message."
    text = ""
    if content:
        if (
            not isinstance(content[0], dict)
            or content[0].get("type") != "input_text"
            or not isinstance(content[0].get("text"), str)
        ):
            return "Send at most one plain-text message."
        text = content[0]["text"].strip()
    if not attachments and not text:
        return "Send one nonblank plain-text message."
    inference = message.get("inference_options")
    if not isinstance(inference, dict):
        return "The ChatKit inference options were refused."
    if inference.get("model") is not None or inference.get("tool_choice") is not None:
        return "Client-selected models and tools are disabled."
    return None


def _shell_html(server: AssistantHTTPServer) -> str:
    introduction = (
        "Ask about your Japanese library in ordinary language, optionally focus this "
        "conversation on one deck, or attach one PDF or photo for local source "
        "intake. A deck focus narrows context; it does not limit what Janki can do."
    )
    assistant_label = "janki Japanese-library assistant"
    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        '<meta name=viewport content="width=device-width,initial-scale=1">'
        "<title>Janki</title>"
        f'<link rel=stylesheet href="{server.style_path}">'
        f'<script src="{_CHATKIT_SCRIPT}" async></script>'
        f'<script src="{server.script_path}" defer></script>'
        "</head><body><main>"
        f"<header><h1>Janki</h1><p>{introduction}</p>"
        "<p class=boundary>This isolated page loads OpenAI's hosted ChatKit UI. It does "
        "not receive the main workbench session or its CSRF authority.</p></header>"
        f'<openai-chatkit id="janki-chat" aria-label="{assistant_label}"></openai-chatkit>'
        '<section id="janki-chat-error" class="chat-error" role="alert" hidden>'
        "<h2>Chat interface did not load</h2>"
        "<p>The local workbench is still running. Check your connection to "
        "OpenAI's hosted ChatKit service, then reload this page.</p></section>"
        "<noscript>JavaScript is required for the ChatKit interface.</noscript>"
        "</main></body></html>"
    )


def _application_javascript(server: AssistantHTTPServer) -> str:
    api_path = json.dumps(server.api_path)
    upload_path_prefix = json.dumps(server.upload_path_prefix)
    frame_title = "janki Japanese-library assistant"
    placeholder = "Ask Janki or attach a source"
    greeting = "What would you like to do?"
    starter_prompts = f"""[
        {{
          label: "Explore my library",
          prompt: {json.dumps(CAPABILITIES_MESSAGE)},
          icon: "book-open",
        }},
        {{
          label: "Focus on a deck",
          prompt: {json.dumps(SHOW_DECKS_MESSAGE)},
          icon: "notebook",
        }},
        {{
          label: "Add study material",
          prompt: {json.dumps(SOURCE_HELP_MESSAGE)},
          icon: "document",
        }},
        {{
          label: "Manage model calls",
          prompt: {json.dumps(MANAGE_OPERATIONS_MESSAGE)},
          icon: "keys",
        }},
        {{
          label: "Resume kanji cards",
          prompt: {json.dumps(RESUME_KANJI_MESSAGE)},
          icon: "book-open",
        }},
        {{
          label: "Manage extraction batches",
          prompt: {json.dumps(MANAGE_EXTRACTION_BATCHES_MESSAGE)},
          icon: "book-open",
        }},
      ]"""
    if server.attachment_store is None:
        attachments = "{ enabled: false }"
        upload_strategy = ""
    else:
        attachments = """{
        enabled: true,
        maxSize: 128 * 1024 * 1024,
        maxCount: 1,
        accept: {
          "application/pdf": [".pdf"],
          "image/jpeg": [".jpg", ".jpeg"],
          "image/png": [".png"],
          "image/heic": [".heic"],
          "image/heif": [".heif"],
        },
      }"""
        upload_strategy = '\n      uploadStrategy: { type: "two_phase" },'
    return f'''"use strict";
(async () => {{
  await new Promise((resolve, reject) => {{
    const timer = window.setTimeout(() => reject(new Error("ChatKit load timed out")), 10000);
    customElements.whenDefined("openai-chatkit").then(
      () => {{
        window.clearTimeout(timer);
        resolve();
      }},
      (error) => {{
        window.clearTimeout(timer);
        reject(error);
      }}
    );
  }});
  const chat = document.getElementById("janki-chat");
  const apiPath = {api_path};
  const uploadPathPrefix = {upload_path_prefix};
  const apiURL = new URL(apiPath, window.location.origin).href;
  const localFetch = (input, init = {{}}) => {{
    const isRequest = input instanceof Request;
    const raw = isRequest ? input.url : String(input);
    const target = new URL(raw, window.location.href);
    const isUpload = uploadPathPrefix !== null &&
      target.pathname.startsWith(uploadPathPrefix);
    if (target.origin !== window.location.origin ||
        (target.pathname !== apiPath && !isUpload)) {{
      throw new Error("ChatKit attempted an unscoped request.");
    }}
    return window.fetch(isRequest ? input : target, {{
      ...init,
      credentials: "omit",
      referrerPolicy: "no-referrer",
    }});
  }};
  chat.setOptions({{
    api: {{
      url: apiURL,
      domainKey: "domain_pk_localhost_dev",
      fetch: localFetch,{upload_strategy}
    }},
    frameTitle: {json.dumps(frame_title)},
    header: {{ enabled: false }},
    history: {{ enabled: false }},
    threadItemActions: {{
      feedback: false,
      retry: false,
    }},
    composer: {{
      placeholder: {json.dumps(placeholder)},
      attachments: {attachments},
      tools: [],
    }},
    startScreen: {{
      greeting: {json.dumps(greeting)},
      prompts: {starter_prompts},
    }},
  }});
}})().catch(() => {{
  document.body.dataset.chatkitFailed = "true";
  const failure = document.getElementById("janki-chat-error");
  if (failure !== null) {{
    failure.hidden = false;
  }}
}});
'''


def _stylesheet() -> str:
    return """
:root { color-scheme: light dark; font-family: system-ui, sans-serif; }
* { box-sizing: border-box; }
body { margin: 0; background: Canvas; color: CanvasText; }
main { width: min(100%, 72rem); min-height: 100vh; margin: 0 auto; padding: 1rem; }
header { width: min(100%, 48rem); margin: 0 auto 1rem; }
h1 { margin-bottom: .35rem; }
p { margin: .35rem 0; line-height: 1.45; }
.boundary { color: GrayText; font-size: .875rem; }
.deck-scope { margin-top: .8rem; }
.deck-scope code { color: GrayText; font-size: .8rem; overflow-wrap: anywhere; }
openai-chatkit { display: block; width: 100%; height: min(76vh, 54rem); min-height: 32rem; }
[data-chatkit-failed="true"] openai-chatkit { display: none; }
.chat-error { width: min(100%, 48rem); margin: 2rem auto; padding: 1rem 1.25rem;
  border: 1px solid color-mix(in srgb, CanvasText 20%, transparent); border-radius: .75rem; }
.chat-error h2 { margin: 0 0 .5rem; font-size: 1.1rem; }
""".strip()


@dataclass(slots=True)
class AssistantSidecar:
    """Running isolated assistant origin and its orderly shutdown boundary."""

    server: AssistantHTTPServer
    bridge: AsyncLoopBridge
    thread: threading.Thread | None = None
    _closed: bool = field(default=False, init=False)

    @property
    def origin(self) -> str:
        return f"http://{self.server.expected_host}"

    @property
    def url(self) -> str:
        return f"{self.origin}{self.server.shell_path}"

    @property
    def api_url(self) -> str:
        return f"{self.origin}{self.server.api_path}"

    def start(self) -> AssistantSidecar:
        if self._closed:
            raise RuntimeError("The assistant sidecar is closed.")
        if self.thread is not None:
            raise RuntimeError("The assistant sidecar is already running.")
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            name="janki-chatkit-http",
            daemon=False,
        )
        self.thread.start()
        return self

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.thread is not None:
            self.server.shutdown()
            self.thread.join()
        self.server.server_close()
        self.bridge.close()
        if self.server.attachment_store is not None:
            self.server.attachment_store.close()

    def __enter__(self) -> AssistantSidecar:
        return self if self.thread is not None else self.start()

    def __exit__(self, *exc: Any) -> None:
        del exc
        self.close()


def create_assistant_sidecar(
    callbacks: RevisionCallbacks,
    *,
    deck_choices: tuple[AssistantDeckChoice, ...],
    session_token: str | None = None,
    inbox_root: Path | None = None,
) -> AssistantSidecar:
    """Bind one isolated sidecar, optionally with local source intake."""

    token = session_token or secrets.token_urlsafe(32)
    if not _TOKEN.fullmatch(token):
        raise ValueError("The assistant session token is not a safe URL segment.")
    bridge = AsyncLoopBridge()
    server: AssistantHTTPServer | None = None
    attachment_store: LocalAssistantAttachmentStore | None = None
    try:
        server = AssistantHTTPServer(("127.0.0.1", 0), _AssistantHandler)
        bind_loopback(server)
        root = f"/{token}/"
        upload_path_prefix = f"{root}attachments/" if inbox_root is not None else None
        download_path_prefix = f"{root}packages/"
        preview_path_prefix = f"{root}previews/"
        editor_path_prefix = f"{root}source-parts/"
        if upload_path_prefix is not None:
            attachment_store = LocalAssistantAttachmentStore(
                inbox_root=inbox_root,
                upload_prefix=(
                    f"http://{server.expected_host}{upload_path_prefix}"
                ),
            )
        # The adapter owns the repository config, so it builds the store this
        # origin serves. A callback set without finished deliverables simply
        # offers no downloads.
        binder = getattr(callbacks, "bind_package_downloads", None)
        package_store: LocalAssistantPackageStore | None = None
        if callable(binder):
            package_store = binder(
                f"http://{server.expected_host}{download_path_prefix}"
            )
            if not isinstance(package_store, LocalAssistantPackageStore):
                raise TypeError(
                    "bind_package_downloads must return a package store."
                )
        # Card previews are held by the same adapter, for the same reason: the
        # isolated origin may hold opaque tokens, never repository paths.
        preview_binder = getattr(callbacks, "bind_preview_links", None)
        preview_store: LocalAssistantPreviewStore | None = None
        if callable(preview_binder):
            preview_store = preview_binder(
                f"http://{server.expected_host}{preview_path_prefix}"
            )
            if not isinstance(preview_store, LocalAssistantPreviewStore):
                raise TypeError("bind_preview_links must return a preview store.")
        # Region editors are held by the same adapter, for the same reason, and
        # carry its owner-only plan and publish routes.
        editor_binder = getattr(callbacks, "bind_source_part_editors", None)
        source_part_editor_store: LocalSourcePartEditorStore | None = None
        if callable(editor_binder):
            source_part_editor_store = editor_binder(
                f"http://{server.expected_host}{editor_path_prefix}"
            )
            if not isinstance(source_part_editor_store, LocalSourcePartEditorStore):
                raise TypeError(
                    "bind_source_part_editors must return a source-part editor store."
                )
        core = create_assistant_core(
            callbacks,
            deck_choices=deck_choices,
            attachment_store=attachment_store,
        )
        server.assistant_core = core
        server.bridge = bridge
        server.session_token = token
        server.shell_path = root
        server.api_path = f"{root}chatkit"
        server.script_path = f"{root}application.js"
        server.style_path = f"{root}application.css"
        server.upload_path_prefix = upload_path_prefix
        server.download_path_prefix = download_path_prefix
        server.preview_path_prefix = preview_path_prefix
        server.editor_path_prefix = editor_path_prefix
        server.attachment_store = attachment_store
        server.package_store = package_store
        server.preview_store = preview_store
        server.source_part_editor_store = source_part_editor_store
        server.deck_choices = tuple(deck_choices)
        return AssistantSidecar(server=server, bridge=bridge)
    except Exception:
        if attachment_store is not None:
            attachment_store.close()
        if server is not None:
            server.server_close()
        bridge.close()
        raise


def start_assistant_sidecar(
    callbacks: RevisionCallbacks,
    *,
    deck_choices: tuple[AssistantDeckChoice, ...],
    session_token: str | None = None,
    inbox_root: Path | None = None,
) -> AssistantSidecar:
    """Bind and start the isolated ChatKit sidecar."""

    return create_assistant_sidecar(
        callbacks,
        deck_choices=deck_choices,
        session_token=session_token,
        inbox_root=inbox_root,
    ).start()
