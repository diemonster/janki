"""One real-browser table journey, with every application provider replaced.

The HTTP suites prove exact controller outcomes.  This test covers the part
they cannot: Chrome's form submission, Origin header, focus model, responsive
layout, dark colour behavior, and the DOM it constructs from the HTML.  Its
project configuration and destination deck are fixture setup, not a claim that
the workbench creates a project.  The request observer covers this workbench
page; it does not instrument Chrome's own background processes.
"""

from __future__ import annotations

import ipaddress
import re
import shutil
import socket
import threading
from pathlib import Path

import pytest
from test_application_journey import _project
from test_audio_cmd import FakeVoice
from test_promote import FakeJpdb, client_for
from test_workbench import _request, _running
from test_workbench_dispatch import PDF, _FakeCall, _install_fake
from test_workbench_promotion import _write_deck

from conftest import seed_prompts
from japanese_anki import jpdb, kanji, operations
from japanese_anki.application import audio as audio_application
from japanese_anki.config import ProjectConfig
from japanese_anki.io import load_records
from japanese_anki.kanji import KanjiInfo
from japanese_anki.workbench import WorkbenchSession

playwright_api = pytest.importorskip(
    "playwright.sync_api",
    reason="the optional real-browser test driver is not installed",
)
PlaywrightError = playwright_api.Error
Request = playwright_api.Request
sync_playwright = playwright_api.sync_playwright


class _WordVoice(FakeVoice):
    name = "voicevox"


class _ExampleVoice(FakeVoice):
    name = "openai"
    suffix = ".mp3"
    launch_hint = "use the fake OpenAI transport"

    def __init__(self) -> None:
        super().__init__(audio=b"ID3 fake mp3", voice="onyx")  # type: ignore[arg-type]
        self.settings = {"model": "gpt-4o-mini-tts", "instructions": ""}


def _launch_chrome(playwright: object):
    """Use an installed browser without making gates download one.

    Absence is a supported skip.  A browser that exists but cannot launch is a
    failed browser gate, not another spelling of absence.
    """
    chromium = playwright.chromium  # type: ignore[attr-defined]
    candidates: list[Path] = []
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        executable = shutil.which(name)
        if executable:
            candidates.append(Path(executable))
    candidates.extend(
        path
        for path in (
            Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
            Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
            Path(chromium.executable_path),
        )
        if path.is_file()
    )
    installed = tuple(dict.fromkeys(path.resolve() for path in candidates))
    if not installed:
        pytest.skip("no installed Chrome/Chromium for the workbench browser test")

    failures = []
    for executable in installed:
        try:
            return chromium.launch(headless=True, executable_path=str(executable))
        except PlaywrightError as exc:
            failures.append(f"{executable}: {exc}")
    pytest.fail("installed Chrome/Chromium could not launch:\n" + "\n".join(failures))


def _deny_non_loopback_egress(
    monkeypatch: pytest.MonkeyPatch,
) -> list[str]:
    """Fail closed if server-side code tries to reach any non-loopback network."""
    blocked: list[str] = []
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_getaddrinfo = socket.getaddrinfo
    unix_family = getattr(socket, "AF_UNIX", None)

    def allowed(sock: socket.socket, address: object) -> bool:
        if unix_family is not None and sock.family == unix_family:
            return True
        if sock.family not in {socket.AF_INET, socket.AF_INET6}:
            return False
        if not isinstance(address, tuple) or not address:
            return False
        host = str(address[0]).split("%", 1)[0]
        if host.casefold() in {"localhost", "ip6-localhost"}:
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False

    def guarded_connect(sock: socket.socket, address: object) -> None:
        if not allowed(sock, address):
            blocked.append(repr(address))
            raise AssertionError(f"browser test blocked non-loopback egress to {address!r}")
        original_connect(sock, address)  # type: ignore[arg-type]

    def guarded_connect_ex(sock: socket.socket, address: object) -> int:
        if not allowed(sock, address):
            blocked.append(repr(address))
            raise AssertionError(f"browser test blocked non-loopback egress to {address!r}")
        return original_connect_ex(sock, address)  # type: ignore[arg-type]

    def guarded_getaddrinfo(host: object, *args: object, **kwargs: object) -> object:
        text = str(host).split("%", 1)[0]
        try:
            local = ipaddress.ip_address(text).is_loopback
        except ValueError:
            local = text.casefold() in {"localhost", "ip6-localhost"}
        if not local:
            blocked.append(repr(host))
            raise AssertionError(
                f"browser test blocked non-loopback name resolution for {host!r}"
            )
        return original_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    return blocked


def test_server_egress_guard_allows_only_loopback_and_unix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the guard itself without making a real network connection."""
    connected: list[tuple[int, object]] = []
    resolved: list[object] = []

    def observe_connect(sock: socket.socket, address: object) -> None:
        connected.append((sock.family, address))

    def observe_connect_ex(sock: socket.socket, address: object) -> int:
        connected.append((sock.family, address))
        return 0

    def observe_getaddrinfo(
        host: object, *_args: object, **_kwargs: object
    ) -> list[object]:
        resolved.append(host)
        return []

    monkeypatch.setattr(socket.socket, "connect", observe_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", observe_connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", observe_getaddrinfo)
    blocked = _deny_non_loopback_egress(monkeypatch)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as ipv4:
        ipv4.connect(("127.0.0.1", 3210))
        assert ipv4.connect_ex(("localhost", 3211)) == 0
        with pytest.raises(AssertionError, match="non-loopback egress"):
            ipv4.connect(("192.0.2.1", 443))
        with pytest.raises(AssertionError, match="non-loopback egress"):
            ipv4.connect_ex(("198.51.100.2", 443))

    assert socket.getaddrinfo("localhost", 80) == []
    with pytest.raises(AssertionError, match="non-loopback name resolution"):
        socket.getaddrinfo("provider.invalid", 443)

    unix_family = getattr(socket, "AF_UNIX", None)
    if unix_family is not None:
        with socket.socket(unix_family, socket.SOCK_STREAM) as unix:
            unix.connect("/not/opened/by-the-observer")

    assert resolved == ["localhost"]
    assert (socket.AF_INET, ("127.0.0.1", 3210)) in connected
    assert (socket.AF_INET, ("localhost", 3211)) in connected
    if unix_family is not None:
        assert (unix_family, "/not/opened/by-the-observer") in connected
    assert blocked == [
        "('192.0.2.1', 443)",
        "('198.51.100.2', 443)",
        "'provider.invalid'",
    ]


def _dictionary() -> FakeJpdb:
    parses = {
        "走る": [1, 11, "走る", "はしる", ["LHLL"], 101, ["vi", "v5r"]],
        "食べる": [2, 22, "食べる", "たべる", ["LHH"], 202, ["vt", "v1"]],
        "飲む": [3, 33, "飲む", "のむ", ["LH"], 303, ["vt", "v5m"]],
    }
    senses = {
        (row[0], row[1]): {"reading": row[3], "alt_sids": []}
        for row in parses.values()
    }
    return FakeJpdb(parses, senses)


def test_table_source_reaches_a_built_deck_in_real_chrome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The corpus begins with no sources or records. Project configuration and a
    # destination deck are prerequisites supplied by the fixture; there is no
    # workbench project-creation flow.
    _project(tmp_path)
    seed_prompts(tmp_path)
    shutil.copytree(Path(__file__).parents[1] / "templates", tmp_path / "templates")
    config_path = tmp_path / "janki.toml"
    config_path.write_text(
        config_path.read_text(encoding="utf-8")
        + '\n[tts]\nsentence_provider = "openai"\n',
        encoding="utf-8",
    )
    _write_deck(tmp_path)

    provider_entered = threading.Event()
    allow_provider_answer = threading.Event()
    extraction = _FakeCall(
        "table_exhaustive",
        entered=provider_entered,
        allow_answer=allow_provider_answer,
    )
    _install_fake(monkeypatch, extraction)
    monkeypatch.setattr(
        "japanese_anki.workbench.server.claude_client.prepare_paid_client",
        lambda: object(),
    )
    dictionary = _dictionary()
    monkeypatch.setattr(jpdb, "api_key_from_env", lambda: "fixture-key")
    monkeypatch.setattr(jpdb, "JpdbClient", lambda *_a, **_kw: client_for(dictionary))
    monkeypatch.setattr(kanji, "fetch_kanji", lambda character: KanjiInfo(character))

    word_voice = _WordVoice()
    example_voice = _ExampleVoice()
    monkeypatch.setattr(
        audio_application,
        "resolve_word_provider",
        lambda _config, _chosen: word_voice,
    )
    monkeypatch.setattr(
        audio_application,
        "resolve_sentence_provider",
        lambda _config, _chosen, _words: example_voice,
    )
    for key in ("ANTHROPIC_API_KEY", "JPDB_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    blocked_egress = _deny_non_loopback_egress(monkeypatch)

    session = WorkbenchSession.open(ProjectConfig.load(tmp_path))
    server, _thread = _running(session)
    assert server.server_address[0] == "127.0.0.1"
    assert ipaddress.ip_address(server.server_address[0]).is_loopback
    workbench_page_requests: list[tuple[str, str, str]] = []
    try:
        with sync_playwright() as playwright:
            browser = _launch_chrome(playwright)
            try:
                context = browser.new_context(
                    viewport={"width": 500, "height": 900},
                    color_scheme="dark",
                )
                page = context.new_page()

                def remember(request: Request) -> None:
                    workbench_page_requests.append(
                        (request.method, request.url, request.headers.get("origin", ""))
                    )

                page.on("request", remember)
                origin = f"http://127.0.0.1:{server.server_address[1]}"
                dashboard = f"{origin}/{session.token}/"
                page.goto(dashboard)

                assert page.evaluate("matchMedia('(prefers-color-scheme: dark)').matches")
                appearance = page.evaluate(
                    """() => {
                      const root = getComputedStyle(document.documentElement);
                      const body = getComputedStyle(document.body);
                      const luminance = value => {
                        const values = value.match(/[\\d.]+/g)?.slice(0, 3).map(Number);
                        if (!values || values.length !== 3) return null;
                        const channels = values.map(channel => {
                          const normalized = channel / 255;
                          return normalized <= 0.04045
                            ? normalized / 12.92
                            : ((normalized + 0.055) / 1.055) ** 2.4;
                        });
                        return 0.2126 * channels[0]
                          + 0.7152 * channels[1]
                          + 0.0722 * channels[2];
                      };
                      return {
                        scheme: root.colorScheme,
                        background: body.backgroundColor,
                        foreground: body.color,
                        backgroundLuminance: luminance(body.backgroundColor),
                        foregroundLuminance: luminance(body.color),
                      };
                    }"""
                )
                assert appearance["scheme"] == "light dark"
                assert appearance["background"] != appearance["foreground"]
                assert appearance["backgroundLuminance"] < appearance["foregroundLuminance"]
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                assert page.evaluate("getComputedStyle(document.body).paddingLeft") == "12px"
                page.set_viewport_size({"width": 800, "height": 900})
                page.wait_for_function("innerWidth === 800")
                assert page.evaluate("getComputedStyle(document.body).paddingLeft") == "24px"
                page.set_viewport_size({"width": 500, "height": 900})
                page.wait_for_function("innerWidth === 500")
                page.locator("#source-file").set_input_files(
                    {
                        "name": "lesson.pdf",
                        "mimeType": "application/pdf",
                        "buffer": PDF,
                    }
                )
                preview = page.locator("#intake-pages iframe")
                assert preview.get_attribute("sandbox") == ""
                assert preview.get_attribute("referrerpolicy") == "no-referrer"
                page.get_by_role("button", name="Save permanent copy").click()
                assert (tmp_path / "inbox" / "lesson.pdf").read_bytes() == PDF

                first_action = page.get_by_role(
                    "link", name="See what reading this would send"
                )
                keyboard_reached_action = False
                for _attempt in range(20):
                    page.keyboard.press("Tab")
                    if first_action.evaluate("node => document.activeElement === node"):
                        keyboard_reached_action = True
                        break
                assert keyboard_reached_action
                focus_style = first_action.evaluate(
                    """node => {
                      const style = getComputedStyle(node);
                      return {
                        kind: style.outlineStyle,
                        width: parseFloat(style.outlineWidth),
                      };
                    }"""
                )
                assert focus_style["kind"] != "none"
                assert focus_style["width"] >= 3

                first_action.click()
                exact_model = session.config.extract_model
                paid = page.get_by_role("button", name="paid API call")
                assert " ".join(paid.inner_text().split()) == (
                    f"Send lesson.pdf to Anthropic using {exact_model} to propose "
                    "vocabulary cards and grammar — paid API call"
                )
                disclosure = page.locator("main").inner_text()
                for named_fact in (
                    "lesson.pdf",
                    "Anthropic",
                    exact_model,
                    "propose vocabulary cards and grammar",
                    "paid API call",
                ):
                    assert named_fact in disclosure

                # Keep the first exact form submission in flight, then submit the
                # same rendered one-use authority through Chrome. A literal
                # dblclick abandons the streamed first response during navigation
                # and is timing-dependent; this preserves the duplicate POST and
                # lets the browser render its refusal deterministically.
                encoded_form = paid.evaluate(
                    """button => {
                      const fields = new FormData(button.form);
                      fields.append(button.name, button.value);
                      return new URLSearchParams(fields).toString();
                    }"""
                )
                first_response: list[tuple[int, dict[str, str], bytes]] = []

                def submit_first() -> None:
                    first_response.append(
                        _request(
                            server,
                            "POST",
                            f"/{session.token}/extract/lesson.pdf",
                            headers={
                                "Content-Type": "application/x-www-form-urlencoded",
                                "Origin": origin,
                            },
                            body=encoded_form.encode("utf-8"),
                        )
                    )

                first_submit = threading.Thread(target=submit_first)
                first_submit.start()
                assert provider_entered.wait(5), (
                    first_response[0][2].decode("utf-8", errors="replace")
                    if first_response
                    else "the first submission neither reached the provider nor replied"
                )
                paid.click()
                assert "already been used" in page.locator("body").inner_text()
                allow_provider_answer.set()
                first_submit.join(3)
                assert not first_submit.is_alive()
                assert first_response and first_response[0][0] == 200
                assert b"Review proposed cards" in first_response[0][2]
                page.goto(f"{origin}/{session.token}/source/lesson.pdf")

                ruby = page.locator("#card-1 h3 ruby")
                ruby_parts = ruby.evaluate(
                    """node => ({
                      expression: Array.from(node.childNodes)
                        .filter(child => child.nodeType === Node.TEXT_NODE)
                        .map(child => child.textContent).join(''),
                      reading: node.querySelector(':scope > rt')?.textContent || '',
                    })"""
                )
                assert ruby_parts == {"expression": "走る", "reading": "はしる"}

                long_meaning = "long-English-" + "W" * 320
                page.get_by_role("link", name="Correct these cards").click()
                page.get_by_label("Meaning in this lesson (one per line)").first.fill(
                    long_meaning
                )
                page.get_by_role("button", name="Save these corrections").click()
                wrapped = page.get_by_text(long_meaning, exact=True)
                wrapping = wrapped.evaluate(
                    """node => ({
                      ownWidth: node.scrollWidth <= node.clientWidth,
                      pageWidth: node.getBoundingClientRect().right <= innerWidth,
                    })"""
                )
                assert wrapping == {"ownWidth": True, "pageWidth": True}

                approvals = page.locator('input[name="record"]')
                assert approvals.count() == 3
                for index in range(approvals.count()):
                    approvals.nth(index).check()
                page.get_by_role("button", name="Save the approvals I ticked").click()

                for index in range(3):
                    page.locator(
                        f'button[form="assign-{index}"][name="destination"]'
                    ).click()

                assert page.locator("ruby").count() > 0
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                page.get_by_role("link", name="Preview adding these cards").click()
                page.get_by_label(
                    "I compared the named source with the coverage account above."
                ).check()
                page.get_by_label("Why the account is complete").fill(
                    "I compared the three fixture rows with the coverage account."
                )
                page.get_by_role(
                    "button", name="I compared the source rows myself"
                ).click()
                page.get_by_role(
                    "button", name="Add 3 cards to your collection"
                ).click()

                page.get_by_role(
                    "button", name="Check dictionary facts with jpdb"
                ).click()
                page.get_by_role("button", name="Save these dictionary facts").click()
                page.get_by_role("button", name="Add the missing kanji reference").click()
                page.get_by_role("button", name="Create words audio").click()
                page.get_by_role(
                    "button",
                    name=re.compile(
                        r"Create example sentence audio for lesson\.pdf with "
                        r"OpenAI gpt-4o-mini-tts .* paid network call"
                    ),
                ).click()
                page.get_by_role("link", name="Preview Lesson deck").click()
                assert page.get_by_text("exact finish preview").is_visible()
                page.get_by_role(
                    "button", name="Build these complete study decks"
                ).click()
                assert page.get_by_text(
                    "Built every receipted study deck"
                ).is_visible(), page.locator("body").inner_text()

                attacker = context.new_page()
                attacker.set_content(
                    f'<form method="post" action="{origin}/{session.token}/decks/new">'
                    '<button type="submit">attack</button></form>'
                )
                attacker.get_by_role("button", name="attack").click()
                assert "exact localhost origin" in attacker.locator("body").inner_text()

                assert extraction.calls and len(extraction.calls) == 1
                assert len(load_records(session.config.normalized_file)) == 3
                assert len(word_voice.said) == 3
                assert len(example_voice.said) == 6
                assert (tmp_path / "dist" / "lesson.apkg").is_file()
                journal = operations.OperationJournal.load(session.config.operations_file)
                assert {entry.state for entry in journal.operations.values()} == {"committed"}
                assert all(
                    url.startswith(origin + "/") or url.startswith("blob:")
                    for _method, url, _request_origin in workbench_page_requests
                )
                post_origins = [
                    request_origin
                    for method, url, request_origin in workbench_page_requests
                    if method == "POST" and url.startswith(origin + "/")
                ]
                assert post_origins and set(post_origins) == {origin}
                assert blocked_egress == []
            finally:
                browser.close()
    finally:
        allow_provider_answer.set()
        server.shutdown()
        server.server_close()
