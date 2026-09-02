"""Bound owner application of staged conjugation-deck revisions."""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from japanese_anki import ai_schema, claude_client, ledger
from japanese_anki.application import revision, revision_apply, revision_provider
from japanese_anki.config import ProjectConfig
from japanese_anki.exporters import pattern_cards
from japanese_anki.io import DataError
from japanese_anki.models import VocabularyRecord


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _fixture(tmp_path: Path) -> tuple[ProjectConfig, Path, Path]:
    (tmp_path / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "vocabulary.json"\n'
        'deck_dir = "decks"\n'
        'ledger_file = "ledger.json"\n'
        'staging_dir = "staging"\n'
        'media_dir = "media"\n',
        encoding="utf-8",
    )
    config = ProjectConfig.load(tmp_path)
    config.deck_dir.mkdir(parents=True)
    config.staging_dir.mkdir(parents=True)
    records = [
        VocabularyRecord(
            id="word:話す:はなす",
            expression="話す",
            reading="はなす",
            meanings=["to speak"],
            part_of_speech="verb",
            verb_group="godan",
        ),
        VocabularyRecord(
            id="word:読む:よむ",
            expression="読む",
            reading="よむ",
            meanings=["to read"],
            part_of_speech="verb",
            verb_group="godan",
        ),
    ]
    config.normalized_file.write_text(
        json.dumps([record.to_dict() for record in records], ensure_ascii=False),
        encoding="utf-8",
    )
    deck = config.deck_dir / "potential.yaml"
    deck.write_text(
        "# keep comment\n"
        "deck:\n"
        "  kind: conjugation\n"
        "  form: potential\n"
        "  name: Potential\n"
        "  deck_id: 19\n"
        "  model_id: 20\n"
        "  custom: 'keep'\n"
        "  form_note: 'Old note.'\n"
        "  include_ids:\n"
        "    - word:話す:はなす\n"
        "    - word:読む:よむ\n"
        "  drill_examples:\n"
        "    word:話す:はなす:\n"
        "      - japanese: 話せます。\n"
        "        furigana: '話[はな]せます。'\n"
        "        english: I can speak.\n"
        "        register: polite\n"
        "        audio: audio/old.wav\n"
        "      - japanese: 話せる？\n"
        "        furigana: '話[はな]せる？'\n"
        "        english: Can you speak?\n"
        "        register: casual\n"
        "        spoken_japanese: 話せる？\n"
        "    word:読む:よむ:\n"
        "      - japanese: 読めます。\n"
        "        furigana: '読[よ]めます。'\n"
        "        english: I can read.\n"
        "        register: polite\n"
        "      - japanese: 読める？\n"
        "        furigana: '読[よ]める？'\n"
        "        english: Can you read?\n"
        "        register: casual\n",
        encoding="utf-8",
    )
    staging = config.staging_dir / "revise-test.json"
    selected = ("word:話す:はなす",)
    context = revision._canonical_context(config, selected, "potential")
    context_fp = _sha(_canonical(context))
    before = {
        "form_note": "Old note.",
        "drill_examples": {
            "word:話す:はなす": [
                {
                    "japanese": "話せます。",
                    "furigana": "話[はな]せます。",
                    "romaji": "",
                    "english": "I can speak.",
                    "register": "polite",
                    "audio": "audio/old.wav",
                    "spoken_japanese": "",
                },
                {
                    "japanese": "話せる？",
                    "furigana": "話[はな]せる？",
                    "romaji": "",
                    "english": "Can you speak?",
                    "register": "casual",
                    "audio": "",
                    "spoken_japanese": "話せる？",
                },
            ]
        },
    }
    style_guide = "style"
    task_template = "task"
    model = "claude-opus-5"
    system_blocks = claude_client.system_blocks(style_guide, task_template)
    user_turn = revision._user_turn(
        deck_name=deck.name,
        form="potential",
        form_note=before["form_note"],
        selected=selected,
        instruction="Improve the examples.",
        current={
            record_id: tuple(before["drill_examples"][record_id])
            for record_id in selected
        },
        context=context,
    )
    provider_plan = revision_provider.plan_provider(
        revision_provider.ANTHROPIC_API_PROVIDER,
        model=model,
        style_guide=style_guide,
        task_template=task_template,
        system_blocks=system_blocks,
        user_turn=user_turn,
        schema=ai_schema.conjugation_deck_revision_schema(),
    )
    request_fp = provider_plan.request_fingerprint
    deck_sha = _sha(deck.read_text(encoding="utf-8"))
    plan_fp = revision._plan_identity(
        deck_relative="decks/potential.yaml",
        deck_sha256=deck_sha,
        selected=selected,
        owner_instruction="Improve the examples.",
        provider=provider_plan.provider,
        model=model,
        canonical_context_fingerprint=context_fp,
        request_fingerprint=request_fp,
        staging_relative="staging/revise-test.json",
        staging_revision=None,
    )
    manifest = {
        "schema_version": 2,
        "kind": "conjugation_deck_revision",
        "state": "proposed",
        "operation_id": "11111111-1111-4111-8111-111111111111",
        "target": {
            "deck_path": "decks/potential.yaml",
            "deck_sha256": deck_sha,
            "form": "potential",
            "selected_record_ids": ["word:話す:はなす"],
            "staging_path": "staging/revise-test.json",
        },
        "request": {
            "provider_plan": provider_plan.persistent_manifest(),
            "owner_instruction": "Improve the examples.",
            "style_guide": style_guide,
            "task_template": task_template,
            "system_blocks": system_blocks,
            "user_turn": user_turn,
            "plan_fingerprint": plan_fp,
            "canonical_context_fingerprint": context_fp,
        },
        "canonical_context": context,
        "before": before,
        "staged_at": "2026-08-31",
        "proposal": {
            "form_note": "New note.",
            "drill_examples": {
                "word:話す:はなす": [
                    {
                        "japanese": "話せます。",
                        "furigana": "話[はな]せます。",
                        "english": "I am able to speak.",
                        "register": "polite",
                    },
                    {
                        "japanese": "日本語が話せる？",
                        "furigana": "日本語[にほんご]が 話[はな]せる？",
                        "english": "Can you speak Japanese?",
                        "register": "casual",
                    },
                ]
            },
        },
    }
    staging.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config.operations_file.parent.mkdir(parents=True, exist_ok=True)
    config.operations_file.write_text(
        json.dumps(
            {
                "version": 1,
                "operations": {
                    manifest["operation_id"]: {
                        "kind": "revise",
                        "state": "committed",
                        "source_file": "decks/potential.yaml",
                        "source_sha256": manifest["target"]["deck_sha256"],
                        "request_fp": manifest["request"]["provider_plan"][
                            "request_fingerprint"
                        ],
                        "model": manifest["request"]["provider_plan"]["model"],
                        "authorized_at": "2026-08-31T12:00:00+00:00",
                        "updated_at": "2026-08-31T12:01:00+00:00",
                        "artifact": {
                            "relative_name": (f".pending/{manifest['operation_id']}.json"),
                            "directory_identity": [1, 2],
                            "entry_state": [1, 3, 4, 5, 6],
                            "sha256": "9" * 64,
                            "terminal_marker": None,
                        },
                    }
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return config, deck, staging


def _replace_with_claude_code_plan(
    config: ProjectConfig,
    staging: Path,
) -> None:
    manifest = json.loads(staging.read_text(encoding="utf-8"))
    request = manifest["request"]
    target = manifest["target"]
    model = request["provider_plan"]["model"]
    provider_plan = revision_provider._claude_code_plan(
        model=model,
        system_blocks=request["system_blocks"],
        user_turn=request["user_turn"],
        schema=ai_schema.conjugation_deck_revision_schema(),
        auth={
            "auth_method": "claude.ai",
            "api_provider": "firstParty",
            "subscription_type": "max",
            "api_key_source": None,
        },
        version="2.1.246",
    )
    request["provider_plan"] = provider_plan.persistent_manifest()
    request["plan_fingerprint"] = revision._plan_identity(
        deck_relative=target["deck_path"],
        deck_sha256=target["deck_sha256"],
        selected=tuple(target["selected_record_ids"]),
        owner_instruction=request["owner_instruction"],
        provider=provider_plan.provider,
        model=provider_plan.model,
        canonical_context_fingerprint=request["canonical_context_fingerprint"],
        request_fingerprint=provider_plan.request_fingerprint,
        staging_relative=target["staging_path"],
        staging_revision=None,
    )
    staging.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    journal = json.loads(config.operations_file.read_text(encoding="utf-8"))
    operation = journal["operations"][manifest["operation_id"]]
    operation["request_fp"] = provider_plan.request_fingerprint
    operation["model"] = provider_plan.model
    config.operations_file.write_text(
        json.dumps(journal, indent=2) + "\n",
        encoding="utf-8",
    )


def test_revision_apply_lands_archives_and_is_exactly_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, deck, staging = _fixture(tmp_path)

    def contacted_provider(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("applying a captured proposal contacted its provider")

    monkeypatch.setattr(claude_client, "prepare_paid_client", contacted_provider)
    ledger_before = config.ledger_file.read_bytes() if config.ledger_file.exists() else None
    plan = revision_apply.plan_revision_apply(config, staging)
    assert plan.current_form_note == "Old note."
    assert tuple(plan.current_drill_examples) == ("word:話す:はなす",)
    assert plan.current_drill_examples["word:話す:はなす"][0].audio == "audio/old.wav"
    assert plan.provider == "anthropic-api"
    assert plan.billing_class == "anthropic-platform-api"
    assert plan.billing_display == "Anthropic API billing"
    assert plan.auth_metadata["auth_method"] == "environment-api-key"
    assert plan.transport_metadata["kind"] == "anthropic-messages-api"
    assert plan.cli_version is None
    stored = json.loads(staging.read_text(encoding="utf-8"))
    assert plan.request_bytes_sha256 == stored["request"]["provider_plan"][
        "request_bytes_sha256"
    ]

    result = revision_apply.execute_revision_apply(config, plan)

    assert result.recovered is False
    assert not staging.exists()
    assert result.archive_path.exists()
    archived = json.loads(result.archive_path.read_text(encoding="utf-8"))
    assert archived["state"] == "accepted"
    assert archived["acceptance"]["authority"] == "repository-owner"
    text = deck.read_text(encoding="utf-8")
    assert "# keep comment" in text
    assert "custom: 'keep'" in text
    assert "New note." in text
    assert "I am able to speak." in text
    assert "audio/old.wav" in text
    assert "spoken_japanese" not in text
    assert "word:読む:よむ" in text
    ledger_after = config.ledger_file.read_bytes() if config.ledger_file.exists() else None
    assert ledger_after == ledger_before
    assert not config.media_dir.exists()

    repeated = revision_apply.execute_revision_apply(config, plan)
    assert repeated.recovered is True
    assert repeated.deck_sha256 == result.deck_sha256


def test_revision_apply_recovery_plan_exactly_matches_live_accepted_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _deck, staging = _fixture(tmp_path)
    proposed = revision_apply.plan_revision_apply(config, staging)
    real_unlink = revision_apply.atomic_unlink_bound

    def interrupt(_path: Path, *, expected_revision: str) -> None:
        raise DataError(f"interrupted {expected_revision}")

    monkeypatch.setattr(revision_apply, "atomic_unlink_bound", interrupt)
    with pytest.raises(DataError, match="interrupted"):
        revision_apply.execute_revision_apply(config, proposed)
    accepted = revision_apply.plan_revision_apply(config, staging)
    real_unlink(staging, expected_revision=accepted.live_sha256)

    recovered = revision_apply.plan_revision_apply_recovery(
        config,
        staging,
        archive_path=accepted.archive_path,
        plan_fingerprint=accepted.plan_fingerprint,
    )

    assert recovered == accepted


def test_revision_apply_recovery_refuses_missing_archive(tmp_path: Path) -> None:
    config, _deck, staging = _fixture(tmp_path)
    proposed = revision_apply.plan_revision_apply(config, staging)
    staging.unlink()

    with pytest.raises(revision_apply.RevisionApplyError, match="archive no longer exists"):
        revision_apply.plan_revision_apply_recovery(
            config,
            staging,
            archive_path=proposed.archive_path,
            plan_fingerprint=proposed.plan_fingerprint,
        )


def test_revision_apply_recovery_refuses_divergent_plan_fingerprint(
    tmp_path: Path,
) -> None:
    config, _deck, staging = _fixture(tmp_path)
    plan = revision_apply.plan_revision_apply(config, staging)
    result = revision_apply.execute_revision_apply(config, plan)

    with pytest.raises(revision_apply.RevisionApplyError, match="accepted plan"):
        revision_apply.plan_revision_apply_recovery(
            config,
            staging,
            archive_path=result.archive_path,
            plan_fingerprint="f" * 64,
        )


def test_revision_apply_recovery_refuses_divergent_archive_content(tmp_path: Path) -> None:
    config, _deck, staging = _fixture(tmp_path)
    plan = revision_apply.plan_revision_apply(config, staging)
    result = revision_apply.execute_revision_apply(config, plan)
    archived = json.loads(result.archive_path.read_text(encoding="utf-8"))
    archived["proposal"]["form_note"] = "Divergent archive."
    result.archive_path.write_text(
        json.dumps(archived, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(revision_apply.RevisionApplyError, match="proposal content is corrupt"):
        revision_apply.plan_revision_apply_recovery(
            config,
            staging,
            archive_path=result.archive_path,
            plan_fingerprint=plan.plan_fingerprint,
        )


def test_revision_apply_recovery_refuses_a_lexical_same_name_archive(
    tmp_path: Path,
) -> None:
    config, _deck, staging = _fixture(tmp_path)
    plan = revision_apply.plan_revision_apply(config, staging)
    result = revision_apply.execute_revision_apply(config, plan)
    replacement_root = result.archive_path.parent / "replacement"
    replacement_root.mkdir()
    replacement = replacement_root / result.archive_path.name
    replacement.write_bytes(result.archive_path.read_bytes())

    with pytest.raises(revision_apply.RevisionApplyError, match="logical proposal archive"):
        revision_apply.plan_revision_apply_recovery(
            config,
            staging,
            archive_path=replacement,
            plan_fingerprint=plan.plan_fingerprint,
        )


def test_revision_apply_wrapper_owns_global_lock_once_and_locked_executor_does_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _deck, staging = _fixture(tmp_path)
    plan = revision_apply.plan_revision_apply(config, staging)
    real_lock = revision_apply.exclusive_path_lock
    global_lock = config.root / ".janki-audio-operation"
    acquired: list[Path] = []

    @contextlib.contextmanager
    def observed_lock(path: Path):
        if path == global_lock:
            acquired.append(path)
        with real_lock(path):
            yield

    monkeypatch.setattr(revision_apply, "exclusive_path_lock", observed_lock)
    revision_apply.execute_revision_apply(config, plan)
    assert acquired == [global_lock]

    recovery = revision_apply.plan_revision_apply_recovery(
        config,
        staging,
        archive_path=plan.archive_path,
        plan_fingerprint=plan.plan_fingerprint,
    )
    revision_apply.execute_revision_apply_locked(config, recovery)
    assert acquired == [global_lock]


def test_completed_revision_recovery_plan_executes_idempotently(tmp_path: Path) -> None:
    config, _deck, staging = _fixture(tmp_path)
    plan = revision_apply.plan_revision_apply(config, staging)
    first = revision_apply.execute_revision_apply(config, plan)
    recovery = revision_apply.plan_revision_apply_recovery(
        config,
        staging,
        archive_path=first.archive_path,
        plan_fingerprint=plan.plan_fingerprint,
    )

    second = revision_apply.execute_revision_apply(config, recovery)
    third = revision_apply.execute_revision_apply(config, recovery)

    assert second == third
    assert second.recovered is True


def test_revision_apply_purely_reconstructs_claude_subscription_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, deck, staging = _fixture(tmp_path)
    _replace_with_claude_code_plan(config, staging)

    def contacted_provider(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("applying a captured proposal contacted its provider")

    monkeypatch.setattr(revision_provider, "_probe_claude", contacted_provider)
    monkeypatch.setattr(claude_client, "prepare_paid_client", contacted_provider)

    plan = revision_apply.plan_revision_apply(config, staging)

    assert plan.provider == "claude-code"
    assert plan.billing_class == "claude-subscription"
    assert plan.billing_display == "Claude Max subscription via Claude Code"
    assert plan.auth_metadata["subscription_type"] == "max"
    assert plan.transport_metadata["kind"] == "claude-code-cli"
    assert plan.cli_version == "2.1.246"
    assert plan.model == "claude-opus-5"
    stored = json.loads(staging.read_text(encoding="utf-8"))
    assert plan.request_bytes_sha256 == stored["request"]["provider_plan"][
        "request_bytes_sha256"
    ]

    result = revision_apply.execute_revision_apply(config, plan)

    assert result.recovered is False
    assert "New note." in deck.read_text(encoding="utf-8")
    repeated = revision_apply.execute_revision_apply(config, plan)
    assert repeated.recovered is True


def test_revision_apply_refuses_stored_provider_manifest_drift(tmp_path: Path) -> None:
    config, deck, staging = _fixture(tmp_path)
    manifest = json.loads(staging.read_text(encoding="utf-8"))
    manifest["request"]["provider_plan"]["billing_display"] = "Unbound billing"
    staging.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    before = deck.read_bytes()

    with pytest.raises(
        revision_apply.RevisionApplyError,
        match="invalid provider plan fields",
    ):
        revision_apply.plan_revision_apply(config, staging)

    assert deck.read_bytes() == before


def test_revision_apply_reports_progress_before_each_transaction_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _deck, staging = _fixture(tmp_path)
    plan = revision_apply.plan_revision_apply(config, staging)
    events: list[str] = []
    real_prepare = revision_apply.prepare_bound_directory
    real_read_optional = revision_apply._read_optional
    real_save = pattern_cards.save_drill_deck_content
    real_archive = revision_apply.atomic_write_bytes_bound
    reads = 0

    def prepare(path: Path) -> None:
        assert events == ["Preparing proposal"]
        real_prepare(path)

    def read_optional(path: Path) -> tuple[bytes | None, str | None]:
        nonlocal reads
        if reads == 0:
            assert path == staging
            assert events == ["Preparing proposal", "Re-reading proposal"]
        reads += 1
        return real_read_optional(path)

    def save(*args: object, **kwargs: object) -> object:
        assert events[-1] == "Applying revision"
        return real_save(*args, **kwargs)

    def archive(*args: object, **kwargs: object) -> None:
        assert events[-1] == "Archiving proposal"
        real_archive(*args, **kwargs)

    monkeypatch.setattr(revision_apply, "prepare_bound_directory", prepare)
    monkeypatch.setattr(revision_apply, "_read_optional", read_optional)
    monkeypatch.setattr(pattern_cards, "save_drill_deck_content", save)
    monkeypatch.setattr(revision_apply, "atomic_write_bytes_bound", archive)

    revision_apply.execute_revision_apply(config, plan, progress=events.append)

    assert events == [
        "Preparing proposal",
        "Re-reading proposal",
        "Applying revision",
        "Archiving proposal",
    ]


def test_completed_revision_refuses_a_tampered_archive_on_retry(tmp_path: Path) -> None:
    config, _deck, staging = _fixture(tmp_path)
    plan = revision_apply.plan_revision_apply(config, staging)
    result = revision_apply.execute_revision_apply(config, plan)
    archived = json.loads(result.archive_path.read_text(encoding="utf-8"))
    archived["target"]["selected_record_ids"] = ["word:読む:よむ"]
    result.archive_path.write_text(
        json.dumps(archived, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(revision_apply.RevisionApplyError, match="confirmed plan"):
        revision_apply.execute_revision_apply(config, plan)


def test_revision_apply_refuses_proposal_drift_before_acceptance(tmp_path: Path) -> None:
    config, deck, staging = _fixture(tmp_path)
    plan = revision_apply.plan_revision_apply(config, staging)
    before = deck.read_bytes()
    staging.write_text(
        staging.read_text(encoding="utf-8").replace("New note.", "Changed later."),
        encoding="utf-8",
    )

    with pytest.raises(revision_apply.RevisionApplyError, match="plan-stale"):
        revision_apply.execute_revision_apply(config, plan)

    assert deck.read_bytes() == before


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    [
        ("request_fingerprint", "f" * 64),
        ("provider", "claude-code"),
        ("billing_class", "different-billing-class"),
        ("auth_metadata", {"auth_method": "different"}),
        ("transport_metadata", {"kind": "different"}),
        ("cli_version", "9.9.9"),
        ("request_bytes_sha256", "e" * 64),
        ("model", "different-model"),
    ],
    ids=[
        "request",
        "provider",
        "billing-class",
        "auth",
        "transport",
        "cli-version",
        "request-bytes",
        "model",
    ],
)
def test_revision_apply_refuses_changed_provider_identity_in_the_confirmation(
    tmp_path: Path,
    field_name: str,
    changed_value: object,
) -> None:
    config, deck, staging = _fixture(tmp_path)
    plan = revision_apply.plan_revision_apply(config, staging)
    changed = dataclasses.replace(plan, **{field_name: changed_value})
    before = deck.read_bytes()

    with pytest.raises(revision_apply.RevisionApplyError, match="plan-stale"):
        revision_apply.execute_revision_apply(config, changed)

    assert deck.read_bytes() == before
    assert json.loads(staging.read_text(encoding="utf-8"))["state"] == "proposed"


def test_revision_apply_refuses_request_metadata_that_does_not_reproduce_authority(
    tmp_path: Path,
) -> None:
    config, deck, staging = _fixture(tmp_path)
    manifest = json.loads(staging.read_text(encoding="utf-8"))
    manifest["request"]["owner_instruction"] = "A different instruction."
    staging.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    before = deck.read_bytes()
    plan = revision_apply.plan_revision_apply(config, staging)

    with pytest.raises(
        revision_apply.RevisionApplyError,
        match="revision-request-provenance",
    ):
        revision_apply.execute_revision_apply(config, plan)

    assert deck.read_bytes() == before
    assert json.loads(staging.read_text(encoding="utf-8"))["state"] == "proposed"


def test_revision_apply_refuses_a_paid_plan_fingerprint_that_does_not_reproduce(
    tmp_path: Path,
) -> None:
    config, deck, staging = _fixture(tmp_path)
    manifest = json.loads(staging.read_text(encoding="utf-8"))
    manifest["request"]["plan_fingerprint"] = "0" * 64
    staging.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    before = deck.read_bytes()
    plan = revision_apply.plan_revision_apply(config, staging)

    with pytest.raises(
        revision_apply.RevisionApplyError,
        match="revision-request-provenance",
    ):
        revision_apply.execute_revision_apply(config, plan)

    assert deck.read_bytes() == before
    assert json.loads(staging.read_text(encoding="utf-8"))["state"] == "proposed"


def test_revision_apply_refuses_current_normalized_context_drift_before_deck_write(
    tmp_path: Path,
) -> None:
    config, deck, staging = _fixture(tmp_path)
    plan = revision_apply.plan_revision_apply(config, staging)
    before = deck.read_bytes()
    records = json.loads(config.normalized_file.read_text(encoding="utf-8"))
    records[0]["meanings"] = ["changed canonical meaning"]
    config.normalized_file.write_text(
        json.dumps(records, ensure_ascii=False), encoding="utf-8"
    )

    with pytest.raises(
        revision_apply.RevisionApplyError,
        match="revision-canonical-context-stale",
    ):
        revision_apply.execute_revision_apply(config, plan)

    assert deck.read_bytes() == before
    assert json.loads(staging.read_text(encoding="utf-8"))["state"] == "proposed"


def test_revision_apply_reloads_context_while_holding_the_normalized_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _deck, staging = _fixture(tmp_path)
    plan = revision_apply.plan_revision_apply(config, staging)
    real_lock = revision_apply.exclusive_path_lock
    real_context = revision._canonical_context
    normalized_held = False

    @contextlib.contextmanager
    def observed_lock(path: Path):
        nonlocal normalized_held
        is_normalized = path.absolute() == config.normalized_file.absolute()
        with real_lock(path):
            if is_normalized:
                normalized_held = True
            try:
                yield
            finally:
                if is_normalized:
                    normalized_held = False

    def observed_context(*args: object, **kwargs: object) -> object:
        assert normalized_held, "canonical context must reload under its file lock"
        return real_context(*args, **kwargs)

    monkeypatch.setattr(revision_apply, "exclusive_path_lock", observed_lock)
    monkeypatch.setattr(revision, "_canonical_context", observed_context)

    revision_apply.execute_revision_apply(config, plan)


def test_accepted_revision_recovery_ignores_later_normalized_context_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, deck, staging = _fixture(tmp_path)
    plan = revision_apply.plan_revision_apply(config, staging)
    real_save = pattern_cards.save_drill_deck_content

    def land_then_interrupt(*args: object, **kwargs: object) -> object:
        real_save(*args, **kwargs)
        raise OSError("crash after canonical write")

    monkeypatch.setattr(pattern_cards, "save_drill_deck_content", land_then_interrupt)
    with pytest.raises(OSError, match="crash after canonical write"):
        revision_apply.execute_revision_apply(config, plan)
    records = json.loads(config.normalized_file.read_text(encoding="utf-8"))
    records[0]["meanings"] = ["changed after acceptance"]
    config.normalized_file.write_text(
        json.dumps(records, ensure_ascii=False), encoding="utf-8"
    )

    monkeypatch.setattr(pattern_cards, "save_drill_deck_content", real_save)
    recovery = revision_apply.plan_revision_apply(config, staging)
    result = revision_apply.execute_revision_apply(config, recovery)

    assert result.recovered is True
    assert not staging.exists()
    assert result.archive_path.exists()
    assert "New note." in deck.read_text(encoding="utf-8")


def test_accepted_revision_recovers_after_deck_write_without_new_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, deck, staging = _fixture(tmp_path)
    plan = revision_apply.plan_revision_apply(config, staging)
    real_save = pattern_cards.save_drill_deck_content

    def land_then_interrupt(*args: object, **kwargs: object) -> object:
        real_save(*args, **kwargs)
        raise OSError("crash after canonical write")

    monkeypatch.setattr(pattern_cards, "save_drill_deck_content", land_then_interrupt)
    with pytest.raises(OSError, match="crash after canonical write"):
        revision_apply.execute_revision_apply(config, plan)
    assert json.loads(staging.read_text(encoding="utf-8"))["state"] == "accepted"
    assert _sha(deck.read_text(encoding="utf-8")) == plan.intended_deck_sha256

    monkeypatch.setattr(pattern_cards, "save_drill_deck_content", real_save)
    recovery = revision_apply.plan_revision_apply(config, staging)
    result = revision_apply.execute_revision_apply(config, recovery)
    assert result.recovered is True
    assert not staging.exists()
    assert result.archive_path.exists()


def test_accepted_revision_recovers_after_archive_before_live_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _deck, staging = _fixture(tmp_path)
    plan = revision_apply.plan_revision_apply(config, staging)
    real_unlink = revision_apply.atomic_unlink_bound

    def interrupt(_path: Path, *, expected_revision: str) -> None:
        raise DataError(f"interrupted {expected_revision}")

    monkeypatch.setattr(revision_apply, "atomic_unlink_bound", interrupt)
    with pytest.raises(DataError, match="interrupted"):
        revision_apply.execute_revision_apply(config, plan)
    accepted = revision_apply.plan_revision_apply(config, staging)
    assert accepted.state == "accepted"
    assert accepted.archive_path.exists()

    monkeypatch.setattr(revision_apply, "atomic_unlink_bound", real_unlink)
    result = revision_apply.execute_revision_apply(config, accepted)
    assert result.recovered is True
    assert not staging.exists()


def test_accepted_revision_refuses_proposal_edits_after_owner_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _deck, staging = _fixture(tmp_path)
    plan = revision_apply.plan_revision_apply(config, staging)

    def interrupt(_path: Path, *, expected_revision: str) -> None:
        raise DataError(f"interrupted {expected_revision}")

    monkeypatch.setattr(revision_apply, "atomic_unlink_bound", interrupt)
    with pytest.raises(DataError, match="interrupted"):
        revision_apply.execute_revision_apply(config, plan)
    accepted = json.loads(staging.read_text(encoding="utf-8"))
    accepted["proposal"]["form_note"] = "Changed after acceptance."
    staging.write_text(
        json.dumps(accepted, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(revision_apply.RevisionApplyError, match="proposal content is corrupt"):
        revision_apply.plan_revision_apply(config, staging)


def test_revision_apply_refuses_pending_audio_for_the_target_deck(tmp_path: Path) -> None:
    config, deck, staging = _fixture(tmp_path)
    plan = revision_apply.plan_revision_apply(config, staging)
    book = ledger.Ledger(path=config.ledger_file)
    owner = pattern_cards.drill_audio_owner_id(19, "potential", "word:話す:はなす")
    identity = book._pending_audio_identity(
        owner,
        of="example",
        target="pending.wav",
        request_input="話せます。",
        forced_accent=False,
        content_fp="a" * 64,
        provider="openai-realtime",
        voice="cedar",
        speed=1.0,
        settings={},
    )
    key = book._pending_audio_key(identity)
    book.record_pending_audio(
        owner,
        of="example",
        target="pending.wav",
        request_input="話せます。",
        forced_accent=False,
        content_fp="a" * 64,
        provider="openai-realtime",
        voice="cedar",
        speed=1.0,
        settings={},
        staged_file=f".pending/{key}-{'b' * 64}.stage",
        staged_sha256="b" * 64,
    )
    book.save()
    before = deck.read_bytes()

    with pytest.raises(revision_apply.RevisionApplyError, match="pending paid audio"):
        revision_apply.execute_revision_apply(config, plan)

    assert deck.read_bytes() == before
    assert json.loads(staging.read_text(encoding="utf-8"))["state"] == "proposed"


def test_revision_apply_refuses_duplicate_json_and_path_escape(tmp_path: Path) -> None:
    config, _deck, staging = _fixture(tmp_path)
    original = staging.read_text(encoding="utf-8")
    duplicated = original.replace(
        '"state": "proposed",',
        '"state": "proposed",\n  "state": "proposed",',
    )
    staging.write_text(duplicated, encoding="utf-8")
    with pytest.raises(revision_apply.RevisionApplyError, match="repeats key 'state'"):
        revision_apply.plan_revision_apply(config, staging)

    escaped = original.replace("decks/potential.yaml", "../outside.yaml")
    staging.write_text(escaped, encoding="utf-8")
    with pytest.raises(revision_apply.RevisionApplyError, match="traverse"):
        revision_apply.plan_revision_apply(config, staging)


def test_revision_apply_requires_matching_committed_paid_provenance(tmp_path: Path) -> None:
    config, _deck, staging = _fixture(tmp_path)
    journal = json.loads(config.operations_file.read_text(encoding="utf-8"))
    operation = next(iter(journal["operations"].values()))
    operation["model"] = "different-model"
    config.operations_file.write_text(json.dumps(journal), encoding="utf-8")

    with pytest.raises(revision_apply.RevisionApplyError, match="journal identity"):
        revision_apply.plan_revision_apply(config, staging)


def test_revision_apply_requires_the_journaled_provider_request_identity(
    tmp_path: Path,
) -> None:
    config, _deck, staging = _fixture(tmp_path)
    journal = json.loads(config.operations_file.read_text(encoding="utf-8"))
    operation = next(iter(journal["operations"].values()))
    operation["request_fp"] = "0" * 64
    config.operations_file.write_text(json.dumps(journal), encoding="utf-8")

    with pytest.raises(revision_apply.RevisionApplyError, match="journal identity"):
        revision_apply.plan_revision_apply(config, staging)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value["request"].__setitem__(
                "canonical_context_fingerprint",
                str(value["request"]["canonical_context_fingerprint"]).upper(),
            ),
            "SHA-256",
        ),
        (
            lambda value: value["proposal"]["drill_examples"]["word:話す:はなす"][0].__setitem__(
                "english", ""
            ),
            "nonblank japanese, furigana, and english",
        ),
        (
            lambda value: value["target"].__setitem__("deck_path", "decks/nested/potential.yaml"),
            "direct configured deck file",
        ),
        (
            lambda value: value["before"]["drill_examples"]["word:話す:はなす"][0].__setitem__(
                "english", "Different old value."
            ),
            "base values no longer match",
        ),
    ],
    ids=["uppercase-sha", "blank-example", "nested-deck", "wrong-before"],
)
def test_revision_apply_strictly_refuses_malformed_authority_inputs(
    tmp_path: Path, mutate: object, message: str
) -> None:
    config, _deck, staging = _fixture(tmp_path)
    value = json.loads(staging.read_text(encoding="utf-8"))
    assert callable(mutate)
    mutate(value)
    staging.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(revision_apply.RevisionApplyError, match=message):
        revision_apply.plan_revision_apply(config, staging)


def test_revision_apply_refuses_a_symlinked_direct_deck(tmp_path: Path) -> None:
    config, deck, staging = _fixture(tmp_path)
    outside = tmp_path / "outside.yaml"
    outside.write_bytes(deck.read_bytes())
    deck.unlink()
    deck.symlink_to(outside)

    with pytest.raises(revision_apply.RevisionApplyError, match="symlink"):
        revision_apply.plan_revision_apply(config, staging)


def test_revision_apply_refuses_a_direct_file_not_in_the_configured_deck_set(
    tmp_path: Path,
) -> None:
    config, deck, staging = _fixture(tmp_path)
    unconfigured = config.deck_dir / "potential.txt"
    unconfigured.write_bytes(deck.read_bytes())
    value = json.loads(staging.read_text(encoding="utf-8"))
    value["target"]["deck_path"] = "decks/potential.txt"
    staging.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    journal = json.loads(config.operations_file.read_text(encoding="utf-8"))
    operation = next(iter(journal["operations"].values()))
    operation["source_file"] = "decks/potential.txt"
    config.operations_file.write_text(json.dumps(journal), encoding="utf-8")

    with pytest.raises(revision_apply.RevisionApplyError, match="configured deck"):
        revision_apply.plan_revision_apply(config, staging)
