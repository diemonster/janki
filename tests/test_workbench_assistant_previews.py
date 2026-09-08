"""Rendered card previews are offered as immutable bytes behind opaque tokens.

A preview is a *view*: it carries no authority, consumes no consent, writes no
receipt, and re-reading one changes nothing. These tests hold the store's
bounds and the isolated origin's route to that shape, including the exact
pairing of the renderer's own content-security policy with its exact bytes.
"""

from __future__ import annotations

import hashlib
import http.client
from pathlib import Path

import pytest

from japanese_anki.card_preview import CardPreview, PreviewCard
from japanese_anki.config import ProjectConfig
from japanese_anki.workbench import assistant_adapter, assistant_previews
from japanese_anki.workbench.assistant_http import (
    _AssistantHandler,
    create_assistant_sidecar,
)

_POLICY = (
    "default-src 'none'; img-src data:; media-src data:; style-src 'unsafe-inline'; "
    "script-src 'sha256-Pm7c9lLmzUCFPnKZKLBLPPzMUKGGqjNJJ0Kb1kIJHOw='; "
    "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)


def _config(tmp_path: Path) -> ProjectConfig:
    (tmp_path / "janki.toml").write_text(
        "[assistant]\nenabled = true\n\n"
        "[paths]\n"
        'normalized_file = "data/normalized/vocabulary.json"\n'
        'deck_dir = "data/decks"\n'
        'dist_dir = "dist"\n'
        'staging_dir = "data/staging"\n'
        'kanji_notes_file = "data/kanji_notes.json"\n',
        encoding="utf-8",
    )
    normalized = tmp_path / "data/normalized/vocabulary.json"
    normalized.parent.mkdir(parents=True)
    normalized.write_text("[]\n", encoding="utf-8")
    return ProjectConfig.load(tmp_path)


def _preview(body: str = "one card") -> CardPreview:
    html = f"<!doctype html><title>preview</title><p>{body}</p>".encode()
    return CardPreview(
        deck_name="Genki II Kanji",
        deck_kind="kanji",
        directions=("recognition",),
        note_count=1,
        card_count=1,
        new_note_count=1,
        deck_note_count=2,
        deck_card_count=2,
        cards=(
            PreviewCard(
                record_id="kanji:理",
                label="理",
                template="Kanji Recognition",
                direction="recognition",
                question_html="<div>理</div>",
                answer_html="<div>logic</div>",
                is_new=True,
            ),
        ),
        html=html,
        sha256=hashlib.sha256(html).hexdigest(),
        content_security_policy=_POLICY,
    )


def _store(**kwargs: object) -> assistant_previews.LocalAssistantPreviewStore:
    return assistant_previews.LocalAssistantPreviewStore(
        preview_prefix="http://127.0.0.1:9/token/previews/",
        **kwargs,
    )


def _tree(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_an_offer_binds_the_exact_bytes_and_policy_behind_an_opaque_token() -> None:
    store = _store()
    preview = _preview()

    offer = store.offer(
        preview,
        label="Preview these cards",
        plan_fingerprint="a" * 64,
        focus_scope="data/decks/genki-ii-kanji.yaml",
        thread_id="thread_1",
    )

    assert offer.url == f"http://127.0.0.1:9/token/previews/{offer.token}"
    assert offer.byte_count == len(preview.html)
    assert offer.sha256 == preview.sha256
    assert offer.card_count == 1
    assert offer.token not in preview.html.decode("utf-8")

    snapshot = store.read(offer.token)
    assert snapshot.html == preview.html
    assert snapshot.content_security_policy == _POLICY
    assert snapshot.plan_fingerprint == "a" * 64
    assert snapshot.focus_scope == "data/decks/genki-ii-kanji.yaml"
    assert snapshot.thread_id == "thread_1"
    # Opening a preview is not a one-use capability: a reload reads it again.
    assert store.read(offer.token).html == preview.html


def test_bytes_that_disagree_with_their_own_hash_are_refused() -> None:
    store = _store()
    preview = _preview()
    tampered = CardPreview(
        **{
            **{
                field: getattr(preview, field)
                for field in (
                    "deck_name",
                    "deck_kind",
                    "directions",
                    "note_count",
                    "card_count",
                    "new_note_count",
                    "deck_note_count",
                    "deck_card_count",
                    "cards",
                    "content_security_policy",
                )
            },
            "html": b"<!doctype html><title>other</title>",
            "sha256": preview.sha256,
        }
    )

    with pytest.raises(assistant_previews.AssistantPreviewError, match="SHA-256"):
        store.offer(tampered, label="Preview these cards")


def test_a_preview_without_its_own_policy_is_refused() -> None:
    store = _store()
    preview = _preview()
    unpaired = CardPreview(
        **{
            **{
                field: getattr(preview, field)
                for field in (
                    "deck_name",
                    "deck_kind",
                    "directions",
                    "note_count",
                    "card_count",
                    "new_note_count",
                    "deck_note_count",
                    "deck_card_count",
                    "cards",
                    "html",
                    "sha256",
                )
            },
            "content_security_policy": "",
        }
    )

    with pytest.raises(assistant_previews.AssistantPreviewError, match="policy"):
        store.offer(unpaired, label="Preview these cards")


def test_the_store_is_bounded_and_evicts_its_oldest_preview() -> None:
    store = _store()
    tokens = [
        store.offer(_preview(f"card {index}"), label="Preview these cards").token
        for index in range(assistant_previews.MAX_PREVIEWS + 1)
    ]

    with pytest.raises(assistant_previews.AssistantPreviewError):
        store.read(tokens[0])
    assert store.read(tokens[-1]).html == _preview(
        f"card {assistant_previews.MAX_PREVIEWS}"
    ).html


def test_an_oversized_preview_is_refused_rather_than_truncated() -> None:
    store = _store()
    huge = b"<!doctype html>" + b"x" * assistant_previews.MAX_PREVIEW_BYTES
    preview = _preview()
    oversized = CardPreview(
        **{
            **{
                field: getattr(preview, field)
                for field in (
                    "deck_name",
                    "deck_kind",
                    "directions",
                    "note_count",
                    "card_count",
                    "new_note_count",
                    "deck_note_count",
                    "deck_card_count",
                    "cards",
                    "content_security_policy",
                )
            },
            "html": huge,
            "sha256": hashlib.sha256(huge).hexdigest(),
        }
    )

    with pytest.raises(assistant_previews.AssistantPreviewError, match="too large"):
        store.offer(oversized, label="Preview these cards")


def test_an_expired_link_explains_recovery_without_promising_a_free_chat_turn() -> None:
    store = _store()
    offer = store.offer(_preview(), label="Preview these cards")
    for index in range(assistant_previews.MAX_PREVIEWS):
        store.offer(_preview(f"card {index}"), label="Preview these cards")

    with pytest.raises(assistant_previews.AssistantPreviewError) as error:
        store.read(offer.token)

    message = str(error.value)
    assert message == assistant_previews.PREVIEW_UNAVAILABLE_MESSAGE
    assert "expired" in message
    # Truthful about what asking again costs: rendering is local, but the
    # message that asks for it is an ordinary Assistant turn.
    assert "ordinary Assistant" in message
    assert "no model" not in message


def test_the_isolated_origin_serves_a_preview_under_the_renderers_own_policy(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    adapter = assistant_adapter.RevisionAssistantAdapter(
        config=config,
        deck_choices=(),
        _targets=(),
    )
    sidecar = create_assistant_sidecar(
        adapter,
        deck_choices=(),
        session_token="assistant-session-token-000000000000",
    )
    sidecar.start()
    try:
        store = sidecar.server.preview_store
        assert store is not None
        preview = _preview()
        offer = store.offer(preview, label="Preview these cards")
        assert offer.url.startswith(f"http://{sidecar.server.expected_host}")
        before = _tree(tmp_path)

        def fetch(path: str) -> tuple[int, bytes, str | None, str | None]:
            connection = http.client.HTTPConnection(
                "127.0.0.1",
                sidecar.server.server_address[1],
                timeout=3,
            )
            connection.request("GET", path)
            response = connection.getresponse()
            body = response.read()
            policy = response.getheader("Content-Security-Policy")
            content_type = response.getheader("Content-Type")
            connection.close()
            return response.status, body, policy, content_type

        route = f"{sidecar.server.preview_path_prefix}{offer.token}"
        status, body, policy, content_type = fetch(route)
        assert status == 200
        assert body == preview.html
        assert policy == _POLICY
        assert content_type == "text/html; charset=utf-8"

        # A browser reload in this same running process still resolves.
        assert fetch(route)[:3] == (200, preview.html, _POLICY)

        # The ChatKit shell keeps its own unwidened policy.
        shell_status, _shell, shell_policy, _type = fetch(sidecar.server.shell_path)
        assert shell_status == 200
        assert shell_policy == _AssistantHandler.content_security_policy
        assert "'unsafe-inline'" not in (shell_policy or "")

        # Reading a preview writes no repository file and no receipt.
        assert _tree(tmp_path) == before
    finally:
        sidecar.close()


def test_an_unknown_token_and_a_foreign_host_are_refused(tmp_path: Path) -> None:
    config = _config(tmp_path)
    adapter = assistant_adapter.RevisionAssistantAdapter(
        config=config,
        deck_choices=(),
        _targets=(),
    )
    sidecar = create_assistant_sidecar(
        adapter,
        deck_choices=(),
        session_token="assistant-session-token-000000000000",
    )
    sidecar.start()
    try:
        port = sidecar.server.server_address[1]
        prefix = sidecar.server.preview_path_prefix

        def fetch(path: str, host: str | None = None) -> tuple[int, bytes]:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
            headers = {"Host": host} if host is not None else {}
            connection.request("GET", path, headers=headers)
            response = connection.getresponse()
            body = response.read()
            connection.close()
            return response.status, body

        unknown = "u" * 43
        status, body = fetch(f"{prefix}{unknown}")
        assert status == 409
        assert "expired" in body.decode("utf-8")

        assert fetch(f"{prefix}not-a-token")[0] == 404
        assert fetch(prefix)[0] == 404
        assert fetch(f"{prefix}{unknown}", host="example.com")[0] == 403
    finally:
        sidecar.close()
