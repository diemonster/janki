"""What the Assistant's card previews actually render, and what they may not do.

Every preview here is a read. It draws proposed content the owning service has
already prepared — never by applying it, never by asking a dictionary or a
model again — and looking at one consumes no capability. The renderer is
exercised through its frozen contract; where a test needs to prove *which*
bytes were handed to it, it records the exact call instead of guessing from
the rendered document.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_application_assistant_kanji_notes import _note

from japanese_anki import card_preview, kanji_notes
from japanese_anki.application import card_revision_finish, character_notes, revision_apply
from japanese_anki.card_preview import CardPreview, CardPreviewError, PreviewCard
from japanese_anki.config import ProjectConfig
from japanese_anki.workbench import assistant_adapter
from japanese_anki.workbench.assistant import RevisionRefusal, _validate_preview_link

_PREFIX = "http://127.0.0.1:9931/session/previews/"
_POLICY = "default-src 'none'; img-src data:; style-src 'unsafe-inline'"


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


def _adapter(
    config: ProjectConfig,
    *,
    previews: bool = True,
) -> assistant_adapter.RevisionAssistantAdapter:
    adapter = assistant_adapter.RevisionAssistantAdapter(
        config=config,
        deck_choices=(),
        _targets=(),
    )
    if previews:
        adapter.bind_preview_links(_PREFIX)
    return adapter


def _kanji_deck(config: ProjectConfig, *, characters: tuple[str, ...] = ("説",)) -> Path:
    config.deck_dir.mkdir(parents=True, exist_ok=True)
    path = config.deck_dir / "genki-ii-kanji.yaml"
    include = ", ".join(f"kanji:{character}" for character in characters)
    path.write_text(
        "deck:\n"
        "  name: Genki II Kanji\n"
        "  kind: kanji\n"
        "  deck_id: 1500000002\n"
        "  model_id: 1500000102\n"
        f"  include_ids: [{include}]\n",
        encoding="utf-8",
    )
    kanji_notes.save_notes(
        config.kanji_notes_file,
        {character: _note(character, ("explanation",)) for character in characters},
    )
    return path


def _resource_id(config: ProjectConfig, deck: Path) -> str:
    broker = assistant_adapter.assistant_context.AssistantContextBroker(config)
    return broker.resource_id_for_deck(deck)


def _intent(kind: str, **values: Any) -> Any:
    options = values.pop("options", None)
    return assistant_adapter.assistant_agent.AgentActionIntent(
        kind=kind,
        resource_ids=tuple(values.pop("resource_ids", ())),
        record_ids=tuple(values.pop("record_ids", ())),
        instruction="Show me these cards.",
        options_json=json.dumps(options or {}, sort_keys=True, separators=(",", ":")),
    )


def _prepare(
    adapter: assistant_adapter.RevisionAssistantAdapter,
    config: ProjectConfig,
    intent: Any,
) -> Any:
    return adapter._prepare_agent_intent(
        config,
        result=SimpleNamespace(answer="Here they are.", action_intents=(intent,)),
        deck_scope="",
    )


def _preview(cards: int = 1, note_count: int = 1) -> CardPreview:
    html = b"<!doctype html><title>preview</title><p>rendered</p>"
    return CardPreview(
        deck_name="Genki II Kanji",
        deck_kind="kanji",
        directions=("recognition",),
        note_count=note_count,
        card_count=cards,
        new_note_count=0,
        deck_note_count=note_count,
        deck_card_count=cards,
        cards=tuple(
            PreviewCard(
                record_id=f"kanji:card{index}",
                label="説",
                template="Kanji Recognition",
                direction="recognition",
                question_html="<div>説</div>",
                answer_html="<div>explanation</div>",
                is_new=False,
            )
            for index in range(cards)
        ),
        html=html,
        sha256=hashlib.sha256(html).hexdigest(),
        content_security_policy=_POLICY,
    )


def _recording_renderer(
    monkeypatch: pytest.MonkeyPatch,
    *,
    preview: CardPreview | None = None,
    error: Exception | None = None,
    unavailable: str | None = None,
) -> list[dict[str, Any]]:
    """Stand in for the renderer while recording its exact frozen arguments."""

    calls: list[dict[str, Any]] = []

    def render(
        config: ProjectConfig,
        deck_path: Path,
        *,
        proposed: Any = (),
        new_record_ids: Any = (),
        scope_record_ids: Any = None,
        subtitle: str = "",
    ) -> CardPreview:
        calls.append(
            {
                "config": config,
                "deck_path": deck_path,
                "proposed": tuple(proposed),
                "new_record_ids": tuple(new_record_ids),
                "scope_record_ids": scope_record_ids,
                "subtitle": subtitle,
            }
        )
        if error is not None:
            raise error
        return preview or _preview()

    monkeypatch.setattr(
        assistant_adapter,
        "card_preview",
        SimpleNamespace(
            ProposedText=card_preview.ProposedText,
            CardPreviewError=CardPreviewError,
            preview_unavailable=lambda: unavailable,
            render_card_preview=render,
        ),
    )
    return calls


def _tree(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_preview_cards_renders_one_existing_deck_and_plans_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    deck = _kanji_deck(config)
    adapter = _adapter(config)
    calls = _recording_renderer(monkeypatch, preview=_preview(cards=2, note_count=2))
    before = _tree(tmp_path)

    reply = _prepare(
        adapter,
        config,
        _intent("preview_cards", resource_ids=(_resource_id(config, deck),)),
    )

    # A read: no capability, no cached plan, no confirmation widget.
    assert reply.action is None
    assert adapter._agent_plans == {}
    assert adapter._agent_plan_duplicates == {}
    assert _tree(tmp_path) == before

    assert _PREFIX in reply.text
    assert "2 note(s)" in reply.text
    assert "2 card(s)" in reply.text
    assert "recognition" in reply.text
    assert "kanji deck" in reply.text

    [call] = calls
    assert call["deck_path"] == deck.resolve()
    assert call["proposed"] == ()
    assert call["scope_record_ids"] is None


def test_preview_cards_narrows_to_the_exact_disclosed_card_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    deck = _kanji_deck(config, characters=("説", "理"))
    adapter = _adapter(config)
    calls = _recording_renderer(
        monkeypatch,
        preview=CardPreview(
            **{
                **{
                    field: getattr(_preview(), field)
                    for field in (
                        "deck_name",
                        "deck_kind",
                        "directions",
                        "note_count",
                        "card_count",
                        "new_note_count",
                        "cards",
                        "html",
                        "sha256",
                        "content_security_policy",
                    )
                },
                "deck_note_count": 2,
                "deck_card_count": 2,
            }
        ),
    )

    reply = _prepare(
        adapter,
        config,
        _intent(
            "preview_cards",
            resource_ids=(_resource_id(config, deck),),
            record_ids=("kanji:説",),
        ),
    )

    assert calls[0]["scope_record_ids"] == ("kanji:説",)
    # The selection and the whole deck are reported as different numbers.
    assert "1 note(s)" in reply.text
    assert "the whole deck holds 2 note(s)" in reply.text


def test_preview_cards_refuses_options_extra_targets_and_model_supplied_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    deck = _kanji_deck(config)
    adapter = _adapter(config)
    calls = _recording_renderer(monkeypatch)
    resource_id = _resource_id(config, deck)

    with pytest.raises(RevisionRefusal, match="no other options"):
        _prepare(
            adapter,
            config,
            _intent(
                "preview_cards",
                resource_ids=(resource_id,),
                options={"search_limit": 3},
            ),
        )
    with pytest.raises(RevisionRefusal, match="exactly one deck resource"):
        _prepare(adapter, config, _intent("preview_cards", resource_ids=()))
    with pytest.raises(RevisionRefusal):
        _prepare(
            adapter,
            config,
            _intent(
                "preview_cards",
                resource_ids=("data/decks/genki-ii-kanji.yaml",),
            ),
        )
    assert calls == []


def test_preview_cards_reports_the_renderers_own_refusal_for_unknown_card_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    deck = _kanji_deck(config)
    adapter = _adapter(config)
    _recording_renderer(
        monkeypatch,
        error=CardPreviewError("kanji:none is not in this deck"),
    )

    with pytest.raises(RevisionRefusal, match="kanji:none is not in this deck"):
        _prepare(
            adapter,
            config,
            _intent(
                "preview_cards",
                resource_ids=(_resource_id(config, deck),),
                record_ids=("kanji:none",),
            ),
        )


def _character_plan(
    config: ProjectConfig,
    deck: Path,
    *,
    characters: tuple[str, ...],
    deck_created: bool,
) -> character_notes.CharacterNotesPlan:
    """One real prepared batch: real notes, real proposed-file bindings."""

    notes = tuple(_note(character, ("explanation",)) for character in characters)
    store = dict(kanji_notes.load_notes(config.kanji_notes_file))
    store.update({note.character: note for note in notes})
    notes_after = kanji_notes.render_notes(store)
    notes_before = config.kanji_notes_file.read_bytes()
    return character_notes.CharacterNotesPlan(
        project_root=config.root.resolve(),
        deck_path=deck.resolve(),
        deck_name="Genki II Kanji",
        output_path=(config.dist_dir / "genki-ii-kanji.apkg").resolve(),
        deck_created=deck_created,
        characters=characters,
        directions=("recognition",),
        notes=notes,
        deck_record_ids=tuple(f"kanji:{character}" for character in characters),
        files=(
            character_notes.ProposedFile(
                label="character notes",
                path=config.kanji_notes_file.resolve(),
                before_sha256=hashlib.sha256(notes_before).hexdigest(),
                after_text=notes_after,
            ),
            character_notes.ProposedFile(
                label="kanji reference store",
                path=config.kanji_file.resolve(),
                before_sha256=None,
                after_text='{"schema_version": 1, "characters": {}}',
            ),
            character_notes.ProposedFile(
                label="jpdb reading facts",
                path=config.jpdb_readings_file.resolve(),
                before_sha256=None,
                after_text='{"schema_version": 1, "characters": {}}',
            ),
            character_notes.ProposedFile(
                label="character deck",
                path=deck.resolve(),
                before_sha256=None,
                after_text=deck.read_text(encoding="utf-8"),
            ),
        ),
        looked_up=characters,
        fetched_readings=(),
        fingerprint="b" * 64,
    )


def test_a_prepared_kanji_batch_previews_only_the_bytes_its_deck_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    deck = _kanji_deck(config)
    adapter = _adapter(config)
    calls = _recording_renderer(monkeypatch)
    # 説 already has a note; 理 is the one this batch actually creates. The
    # deck itself is new, which does not make an existing note new.
    service = _character_plan(
        config,
        deck,
        characters=("説", "理"),
        deck_created=True,
    )
    before = _tree(tmp_path)

    offer, problem = adapter._offer_kanji_preview(
        config,
        SimpleNamespace(service_plan=service, fingerprint="c" * 64),
        deck_scope="",
    )

    assert problem is None
    assert offer is not None
    assert offer.url.startswith(_PREFIX)
    [call] = calls
    assert call["deck_path"] == deck.resolve()
    # Only the exporter's own inputs are overlaid: the refreshable dictionary
    # caches this batch would also write are not deck inputs.
    assert {item.path for item in call["proposed"]} == {
        config.kanji_notes_file.resolve(),
        deck.resolve(),
    }
    assert any(
        item.path == config.kanji_notes_file.resolve() and "理" in item.text
        for item in call["proposed"]
    )
    assert call["new_record_ids"] == ("kanji:理",)
    assert call["scope_record_ids"] == ("kanji:説", "kanji:理")
    # Rendering a proposal applies nothing.
    assert _tree(tmp_path) == before
    assert adapter._preview_store is not None
    assert adapter._preview_store.read(offer.token).plan_fingerprint == "c" * 64


def test_a_kanji_batch_claims_no_addition_when_the_note_store_moved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    deck = _kanji_deck(config)
    adapter = _adapter(config)
    calls = _recording_renderer(monkeypatch)
    service = _character_plan(config, deck, characters=("理",), deck_created=False)
    kanji_notes.save_notes(
        config.kanji_notes_file,
        {"説": _note("説", ("explanation",)), "頼": _note("頼", ("request",))},
    )

    offer, problem = adapter._offer_kanji_preview(
        config,
        SimpleNamespace(service_plan=service, fingerprint="c" * 64),
        deck_scope="",
    )
    assert (offer is not None, problem) == (True, None)
    assert calls[0]["new_record_ids"] == ()


def test_without_a_renderer_the_kanji_confirmation_keeps_its_written_sides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    deck = _kanji_deck(config)
    adapter = _adapter(config)
    _recording_renderer(monkeypatch, unavailable="Install the preview extra.")
    service = _character_plan(config, deck, characters=("理",), deck_created=False)

    offer, problem = adapter._offer_kanji_preview(
        config,
        SimpleNamespace(service_plan=service, fingerprint="c" * 64),
        deck_scope="",
    )

    # The confirmation stays usable and says exactly what could not be drawn.
    assert offer is None
    assert problem == "Install the preview extra."
    assert assistant_adapter.RevisionAssistantAdapter._preview_disclosure(problem) == (
        "Janki could not draw these cards for review: Install the preview extra. "
        "The exact effects above are unchanged and still describe what confirming "
        "does.",
    )


def _card_finish_plan(
    config: ProjectConfig,
    deck: Path,
    *,
    projected: str,
) -> card_revision_finish.CardRevisionFinishPlan:
    """A real finish plan carrying the projection its service already made."""

    return card_revision_finish.CardRevisionFinishPlan(
        repository_root=config.root.resolve(),
        proposal_kind="ai_enrichment",
        resource_id="proposal-1",
        instruction="Fill the missing meanings.",
        review=None,
        promotion=None,
        deck_path=deck.resolve(),
        record_ids=("word:話す:はなす",),
        audio=None,
        build=None,
        projected_canonical_text=projected,
        projected_audio_text=projected,
        finish_directory=(config.root / "data/finish").resolve(),
        record_path=(config.root / "data/finish/card-finish.json").resolve(),
        authority={},
        fingerprint="d" * 64,
    )


def test_a_reviewed_card_finish_previews_the_projected_canonical_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    deck = _kanji_deck(config)
    adapter = _adapter(config)
    calls = _recording_renderer(monkeypatch)
    projected = json.dumps([{"id": "word:話す:はなす", "meanings": ["to speak"]}])
    plan = _card_finish_plan(config, deck, projected=projected)
    before = _tree(tmp_path)

    offer, problem = adapter._offer_card_finish_preview(config, plan, deck_scope="")

    assert problem is None
    assert offer is not None
    [call] = calls
    assert call["deck_path"] == deck.resolve()
    # The proposed collection, not the one on disk, and not the post-audio
    # projection whose clips do not exist yet.
    assert call["proposed"] == (
        card_preview.ProposedText(
            path=config.normalized_file.resolve(),
            text=projected,
        ),
    )
    assert call["new_record_ids"] == ()
    assert call["scope_record_ids"] == ("word:話す:はなす",)
    assert _tree(tmp_path) == before
    assert config.normalized_file.read_text(encoding="utf-8") == "[]\n"


def _apply_plan(deck: Path, *, intended: str) -> revision_apply.RevisionApplyPlan:
    return revision_apply.RevisionApplyPlan(
        repository_root=deck.parent.parent.parent.resolve(),
        staging_path=deck.parent / "staged.json",
        live_sha256="a" * 64,
        proposal_sha256="b" * 64,
        state="staged",
        operation_id="op-1",
        request_fingerprint="c" * 64,
        provider="claude-code",
        billing_class="subscription",
        auth_metadata={},
        transport_metadata={},
        cli_version=None,
        request_bytes_sha256="d" * 64,
        model="claude-opus-5",
        deck_path=deck.resolve(),
        deck_relative_path="data/decks/drills.yaml",
        deck_base_sha256="e" * 64,
        selected_record_ids=("word:話す:はなす",),
        canonical_context_fingerprint="f" * 64,
        current_form_note="old note",
        current_drill_examples={},
        form_note="new note",
        drill_examples={},
        intended_deck_text=intended,
        intended_deck_sha256="0" * 64,
        archive_path=deck.parent / "archive.json",
        plan_fingerprint="1" * 64,
    )


def test_a_reviewed_deck_revision_previews_its_exact_intended_deck_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    deck = _kanji_deck(config)
    adapter = _adapter(config)
    calls = _recording_renderer(monkeypatch)
    intended = "deck:\n  name: Revised\n"
    original = deck.read_bytes()

    offer, problem = adapter._offer_deck_revision_preview(
        config,
        _apply_plan(deck, intended=intended),
        deck_scope="",
    )

    assert problem is None
    assert offer is not None
    [call] = calls
    assert call["deck_path"] == deck.resolve()
    assert call["proposed"] == (
        card_preview.ProposedText(path=deck.resolve(), text=intended),
    )
    assert deck.read_bytes() == original


def test_viewing_a_preview_never_consumes_the_plan_it_decorates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    deck = _kanji_deck(config)
    adapter = _adapter(config)
    _recording_renderer(monkeypatch)
    service = _character_plan(config, deck, characters=("理",), deck_created=False)
    offer, _problem = adapter._offer_kanji_preview(
        config,
        SimpleNamespace(service_plan=service, fingerprint="c" * 64),
        deck_scope="",
    )
    assert offer is not None
    store = adapter._preview_store
    assert store is not None

    # A browser reload in this same process reads the same document again.
    first = store.read(offer.token)
    second = store.read(offer.token)
    assert first.html == second.html
    assert first.sha256 == second.sha256


def test_a_preview_link_must_be_a_loopback_url_without_userinfo() -> None:
    _validate_preview_link(f"{_PREFIX}token", "Preview these cards")
    _validate_preview_link(None, "Preview these cards")
    for hostile in (
        "http://localhost:80@external.example/steal",
        "http://127.0.0.1.example.com:8000/x",
        "https://127.0.0.1:9931/x",
        "http://127.0.0.1:notaport/x",
        "javascript:alert(1)",
        "http://[::1]x:80/",
    ):
        with pytest.raises(ValueError, match="preview link"):
            _validate_preview_link(hostile, "Preview these cards")
