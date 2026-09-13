"""One closed download registry: a token binds one kind and one receipt.

`docs/ASSISTANT_STUDY_JOBS_CONTRACTS.md` §7.11 turns the kanji-only download
store into a closed resolver registry over the two finish services that own a
completed package. These cases pin the registry itself: which service a kind
dispatches to, that an unknown kind refuses before either service is asked,
that a token carries its kind as well as its receipt id, and that a fresh
process mints a new token from the same durable receipt.

**Unit level.** The receipts here are simple fake inspector results carrying
exactly what `KanjiFinishResult` and `StudyFinishResult` already expose —
`succeeded`, `receipt_id`, `state`, `output_path`, `package_sha256` — so the
dispatch and the byte proof are the subject rather than either finish service.
The real kanji journey is covered against real receipts in
`tests/test_workbench_assistant_kanji.py`; the real study journey is still owed
and nothing here stands in for it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest

from japanese_anki.application import kanji_finish, study_finish
from japanese_anki.config import ProjectConfig
from japanese_anki.workbench import assistant_packages

_PREFIX = "http://127.0.0.1:9/token/packages/"
_KANJI_BYTES = b"one exact kanji package"
_STUDY_BYTES = b"one exact study package"
_RECEIPT = "a" * 64


def _config(tmp_path: Path) -> ProjectConfig:
    (tmp_path / "janki.toml").write_text(
        '[paths]\ndist_dir = "dist"\n', encoding="utf-8"
    )
    return ProjectConfig.load(tmp_path)


def _store(config: ProjectConfig) -> assistant_packages.LocalAssistantPackageStore:
    return assistant_packages.LocalAssistantPackageStore(
        config=config,
        download_prefix=_PREFIX,
    )


@dataclass(frozen=True)
class _FakeReceipt:
    """Only what both finish services' public results already expose."""

    receipt_id: str
    state: str
    output_path: Path
    package_sha256: str | None

    @property
    def succeeded(self) -> bool:
        return self.state == "complete"


def _package(config: ProjectConfig, name: str, payload: bytes) -> Path:
    config.dist_dir.mkdir(parents=True, exist_ok=True)
    path = config.dist_dir / name
    path.write_bytes(payload)
    return path


def _complete(config: ProjectConfig, name: str, payload: bytes) -> _FakeReceipt:
    path = _package(config, name, payload)
    return _FakeReceipt(
        receipt_id=_RECEIPT,
        state="complete",
        output_path=path,
        package_sha256=hashlib.sha256(payload).hexdigest(),
    )


def _install_inspectors(
    monkeypatch: pytest.MonkeyPatch,
    *,
    kanji: Mapping[str, _FakeReceipt] | None = None,
    study: Mapping[str, _FakeReceipt] | None = None,
) -> list[tuple[str, str]]:
    """Record every dispatch, and answer only from the asked kind's receipts."""

    kanji_receipts = dict(kanji or {})
    study_receipts = dict(study or {})
    asked: list[tuple[str, str]] = []

    def inspect_kanji(_config: ProjectConfig, receipt_id: str) -> _FakeReceipt:
        asked.append(("kanji_finish", receipt_id))
        if receipt_id not in kanji_receipts:
            raise kanji_finish.KanjiFinishError(
                f"No character-note finish receipt {receipt_id}."
            )
        return kanji_receipts[receipt_id]

    def inspect_study(_config: ProjectConfig, receipt_id: str) -> _FakeReceipt:
        asked.append(("study_finish", receipt_id))
        if receipt_id not in study_receipts:
            raise study_finish.StudyFinishError(
                f"No study finish receipt {receipt_id}."
            )
        return study_receipts[receipt_id]

    monkeypatch.setattr(kanji_finish, "inspect_kanji_finish", inspect_kanji)
    monkeypatch.setattr(study_finish, "inspect_study_finish", inspect_study)
    return asked


def test_an_unknown_kind_refuses_without_asking_either_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A kind outside the closed table is refused, not guessed at."""

    config = _config(tmp_path)
    asked = _install_inspectors(
        monkeypatch, kanji={_RECEIPT: _complete(config, "kanji.apkg", _KANJI_BYTES)}
    )
    store = _store(config)

    with pytest.raises(assistant_packages.AssistantPackageError) as refusal:
        store.offer(kind="revision_finish", receipt_id=_RECEIPT)

    assert "revision_finish" in str(refusal.value)
    assert asked == []


def test_a_receipt_alone_cannot_mint_a_download_without_its_kind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """There is no default kind: the caller states which service owns it."""

    config = _config(tmp_path)
    asked = _install_inspectors(
        monkeypatch, kanji={_RECEIPT: _complete(config, "kanji.apkg", _KANJI_BYTES)}
    )
    store = _store(config)

    with pytest.raises(TypeError):
        store.offer(receipt_id=_RECEIPT)  # type: ignore[call-arg]

    assert asked == []


def test_the_kanji_kind_dispatches_only_to_the_kanji_inspector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    asked = _install_inspectors(
        monkeypatch,
        kanji={_RECEIPT: _complete(config, "genki-ii-kanji.apkg", _KANJI_BYTES)},
        study={_RECEIPT: _complete(config, "week-two.apkg", _STUDY_BYTES)},
    )
    store = _store(config)

    offer = store.offer(kind="kanji_finish", receipt_id=_RECEIPT)

    assert offer.filename == "genki-ii-kanji.apkg"
    assert offer.byte_count == len(_KANJI_BYTES)
    assert offer.sha256 == hashlib.sha256(_KANJI_BYTES).hexdigest()
    assert offer.url == f"{_PREFIX}{offer.token}"
    assert store.read(offer.token) == ("genki-ii-kanji.apkg", _KANJI_BYTES)
    assert asked == [("kanji_finish", _RECEIPT), ("kanji_finish", _RECEIPT)]


def test_the_study_kind_dispatches_only_to_the_study_inspector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unit level: a fake study result, not a real study finish journey."""

    config = _config(tmp_path)
    asked = _install_inspectors(
        monkeypatch,
        kanji={_RECEIPT: _complete(config, "genki-ii-kanji.apkg", _KANJI_BYTES)},
        study={_RECEIPT: _complete(config, "week-two.apkg", _STUDY_BYTES)},
    )
    store = _store(config)

    offer = store.offer(kind="study_finish", receipt_id=_RECEIPT)

    assert offer.filename == "week-two.apkg"
    assert offer.byte_count == len(_STUDY_BYTES)
    assert offer.sha256 == hashlib.sha256(_STUDY_BYTES).hexdigest()
    assert store.read(offer.token) == ("week-two.apkg", _STUDY_BYTES)
    assert asked == [("study_finish", _RECEIPT), ("study_finish", _RECEIPT)]


def test_a_token_binds_the_exact_kind_and_receipt_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two kinds holding the same receipt id are still two different packages."""

    config = _config(tmp_path)
    asked = _install_inspectors(
        monkeypatch,
        kanji={_RECEIPT: _complete(config, "genki-ii-kanji.apkg", _KANJI_BYTES)},
        study={_RECEIPT: _complete(config, "week-two.apkg", _STUDY_BYTES)},
    )
    store = _store(config)
    kanji_offer = store.offer(kind="kanji_finish", receipt_id=_RECEIPT)
    study_offer = store.offer(kind="study_finish", receipt_id=_RECEIPT)
    asked.clear()

    assert kanji_offer.token != study_offer.token
    assert store.read(kanji_offer.token) == ("genki-ii-kanji.apkg", _KANJI_BYTES)
    assert store.read(study_offer.token) == ("week-two.apkg", _STUDY_BYTES)
    assert asked == [("kanji_finish", _RECEIPT), ("study_finish", _RECEIPT)]


@pytest.mark.parametrize(
    ("kind", "state"),
    [("kanji_finish", "applied"), ("study_finish", "packaged")],
)
def test_an_unfinished_receipt_of_either_kind_offers_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    state: str,
) -> None:
    config = _config(tmp_path)
    unfinished = _FakeReceipt(
        receipt_id=_RECEIPT,
        state=state,
        output_path=_package(config, "week-two.apkg", _STUDY_BYTES),
        package_sha256=None,
    )
    asked = _install_inspectors(
        monkeypatch, kanji={_RECEIPT: unfinished}, study={_RECEIPT: unfinished}
    )
    store = _store(config)

    with pytest.raises(
        assistant_packages.AssistantPackageError, match=f"state {state}"
    ):
        store.offer(kind=kind, receipt_id=_RECEIPT)

    assert asked == [(kind, _RECEIPT)]


@pytest.mark.parametrize("kind", ["kanji_finish", "study_finish"])
def test_every_read_reproves_the_bytes_against_that_receipts_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    """The token is not the proof; the receipt's own digest is, on every read."""

    config = _config(tmp_path)
    receipt = _complete(config, "week-two.apkg", _STUDY_BYTES)
    _install_inspectors(
        monkeypatch, kanji={_RECEIPT: receipt}, study={_RECEIPT: receipt}
    )
    store = _store(config)
    offer = store.offer(kind=kind, receipt_id=_RECEIPT)
    assert store.read(offer.token) == ("week-two.apkg", _STUDY_BYTES)

    receipt.output_path.write_bytes(b"a replaced package")

    with pytest.raises(
        assistant_packages.AssistantPackageError, match="changed after it was finished"
    ):
        store.read(offer.token)


@pytest.mark.parametrize("kind", ["kanji_finish", "study_finish"])
def test_a_package_name_that_is_not_a_plain_apkg_download_name_refuses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    config = _config(tmp_path)
    receipt = _complete(config, "week two.apkg", _STUDY_BYTES)
    _install_inspectors(
        monkeypatch, kanji={_RECEIPT: receipt}, study={_RECEIPT: receipt}
    )
    store = _store(config)

    with pytest.raises(
        assistant_packages.AssistantPackageError, match="plain .apkg download name"
    ):
        store.offer(kind=kind, receipt_id=_RECEIPT)


def test_a_restarted_store_remints_from_the_durable_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tokens are process-local handles; the receipt is what survives."""

    config = _config(tmp_path)
    asked = _install_inspectors(
        monkeypatch, study={_RECEIPT: _complete(config, "week-two.apkg", _STUDY_BYTES)}
    )
    first = _store(config)
    stale = first.offer(kind="study_finish", receipt_id=_RECEIPT).token

    restarted = _store(ProjectConfig.load(tmp_path))
    asked.clear()

    with pytest.raises(
        assistant_packages.AssistantPackageError, match="unknown or expired"
    ):
        restarted.read(stale)
    assert asked == []

    minted = restarted.offer(kind="study_finish", receipt_id=_RECEIPT)

    assert minted.token != stale
    assert restarted.read(minted.token) == ("week-two.apkg", _STUDY_BYTES)
    assert asked == [("study_finish", _RECEIPT), ("study_finish", _RECEIPT)]
