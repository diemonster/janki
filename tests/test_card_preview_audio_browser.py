"""The delivered package's sentence audio, played in a real browser.

`docs/DESIGN.md` states what a finished study job has to prove about sound:
"Completion proves canonical sentence links, current media, sound references
and bytes inside the package, and **playable audio in the final HTML preview**."
Every other check in this repository stops one step short of that last clause.
`tests/test_card_preview.py` proves the preview *draws a control* and inlines
the archive's bytes; the audio and build suites prove bytes exist in the
package; `tests/test_workbench_browser.py` drives Chrome over the workbench.
None of them ever asked a browser to decode a clip and play it, so a page that
carried an unplayable payload — a wrong MIME type, a truncated member, a
control pointing at another card's clip — passed every one of them.

This file asks exactly that question, once, end to end:

* real WAV bytes (a finite 0.75-second 16-bit PCM tone, built here by `wave`
  rather than a fake `ID3`/`RIFF` literal) are written into a real project,
* the owning exporter builds a real `.apkg` through genanki and the archive is
  opened to confirm those exact bytes are the member the media index names,
* `render_packaged_card_preview` draws that proven archive — the same seam a
  finished job's final preview uses — scoped to **one** of the deck's two
  notes,
* the exported HTML is opened in installed Chrome over `file://`, Show Answer
  is clicked, and the *selected sentence's* own control is clicked the way the
  owner clicks it: a real mouse event on the native play button,
* and the clip is then observed playing: metadata loaded, a finite duration of
  0.75s, `currentTime` strictly advancing at moments the element itself reported
  as neither paused nor ended, `ended` reached, no `MediaError`, and the word
  clip beside it still untouched at zero.

The advancing evidence is taken *in the page*, inside the event listener, and
read back only once the clip has finished. A driver that waits for an in-page
event and then evaluates the element again is asking a second process a question
with a deadline: 0.75 seconds of audio can end before that read lands, and a
perfectly good playthrough then reports itself paused at its final frame. What
the element was doing at an event is known at the event or not at all.

A resolved `play()` promise is not evidence and is never used here: nothing in
this test calls `play()`, constructs an `Audio`, stubs a player or asserts on
the absence of an exception.

**Mandatory, so absence fails.** The frozen browser helpers in
`tests/test_card_preview.py` and `tests/test_workbench_browser.py` skip when no
browser is installed, which is right for the checks they guard — they are about
layout and flip state, and the exported bytes carry those claims too. It is not
right here: "the audio plays" has no non-browser spelling, so a skip would
report the one thing that cannot be proven elsewhere as proven. Playwright and
an installed Chrome/Chromium are therefore requirements, and their absence is a
failure naming what to install. Nothing here downloads a browser.

Offline by construction and independent of any other module: nothing is
imported from another test file (an imported helper would not bring its
module's autouse fixtures with it anyway), the page is a self-contained
`file://` document whose own policy is `default-src 'none'` with `data:` media,
and the test asserts the browser issued no request off that file. The
directory's `conftest.py` guards — no real Anki collection, no billed client,
no installed Claude CLI — apply to this module as they do to every other.

Nothing here reads Japanese. The Japanese is ordinary fixture input; the
assertions are about bytes, identities and what a media element did.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import json
import math
import shutil
import struct
import wave
import zipfile
from pathlib import Path
from typing import Any

import pytest

from japanese_anki import card_preview
from japanese_anki.config import ProjectConfig
from japanese_anki.models import ExampleSentence, VocabularyRecord

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: The clip under test. 0.75s is what S7's acceptance names, and it is short
#: enough that a real browser really does play the whole of it inside a test.
SENTENCE_SECONDS = 0.75
#: The word clip on the same card, deliberately a *different* length. A control
#: that played the word instead of the sentence would otherwise look identical.
WORD_SECONDS = 0.25
#: The other note's sentence clip — longer again, and never in this page at all.
OTHER_SENTENCE_SECONDS = 1.5

SAMPLE_RATE = 24_000

SENTENCE_CLIP = "janki-sentence-0750.wav"
WORD_CLIP = "janki-word-0250.wav"
OTHER_SENTENCE_CLIP = "janki-other-sentence-1500.wav"

SELECTED_ID = "word:話す:はなす"
WHOLE_DECK_ONLY_ID = "word:見る:みる"
SELECTED_SENTENCE_ENGLISH = "I speak with my wife every day."
WHOLE_DECK_ONLY_ENGLISH = "I watch a film on Sunday."

#: How long a 0.75s clip may take to finish, wall clock, before the browser is
#: telling us something other than "it played". Generous: the assertion that
#: matters is that it ended at all, not how promptly.
PLAYBACK_TIMEOUT_MS = 20_000


# --- real audio bytes ----------------------------------------------------------


def _wav_bytes(seconds: float, hertz: int) -> bytes:
    """A finite, decodable, *real* WAV: 16-bit PCM mono at 24 kHz.

    Written with `wave` rather than hand-rolled, so the header is a real header
    and the frame count is the duration: a browser reads 0.75 out of these
    bytes because they are 18000 frames at 24000 Hz, not because a test said so.
    A tone rather than silence, so a decoder that dropped the data chunk would
    not be indistinguishable from one that played it.
    """
    frames = round(SAMPLE_RATE * seconds)
    samples = bytearray()
    for index in range(frames):
        value = int(12000 * math.sin(2 * math.pi * hertz * index / SAMPLE_RATE))
        samples += struct.pack("<h", value)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(SAMPLE_RATE)
        stream.writeframes(bytes(samples))
    return buffer.getvalue()


SENTENCE_WAV = _wav_bytes(SENTENCE_SECONDS, 440)
WORD_WAV = _wav_bytes(WORD_SECONDS, 660)
OTHER_SENTENCE_WAV = _wav_bytes(OTHER_SENTENCE_SECONDS, 880)


def _b64(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")


def _wav_duration(payload: bytes) -> float:
    with wave.open(io.BytesIO(payload), "rb") as stream:
        return stream.getnframes() / stream.getframerate()


# --- one real project, one real package ----------------------------------------


def _project(root: Path) -> ProjectConfig:
    """A scratch project holding this checkout's real templates and nothing else."""
    (root / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "vocabulary.json"\n'
        'deck_dir = "decks"\n'
        'media_dir = "media"\n'
        'template_dir = "templates/japanese-study"\n'
        'dist_dir = "dist"\n',
        encoding="utf-8",
    )
    shutil.copytree(PROJECT_ROOT / "templates", root / "templates")
    (root / "decks").mkdir()
    (root / "media").mkdir()
    (root / "vocabulary.json").write_text("[]", encoding="utf-8")
    return ProjectConfig.load(root)


def _records() -> list[VocabularyRecord]:
    """Two notes: the one this preview selects, and one only the deck holds.

    The second note exists so "the clip that played belongs to the selected
    sentence" is a claim with something to be wrong about. Its own sentence clip
    is a third distinct length, and its bytes must never reach the page.
    """
    return [
        VocabularyRecord(
            id=SELECTED_ID,
            expression="話す",
            reading="はなす",
            furigana="話[はな]す",
            meanings=["to speak"],
            part_of_speech="verb",
            verb_group="godan",
            audio=WORD_CLIP,
            tags=["lesson"],
            examples=[
                ExampleSentence(
                    japanese="毎日妻と話します。",
                    furigana="毎日[まいにち] 妻[つま]と 話[はな]します。",
                    english=SELECTED_SENTENCE_ENGLISH,
                    audio=SENTENCE_CLIP,
                    register="polite",
                )
            ],
        ),
        VocabularyRecord(
            id=WHOLE_DECK_ONLY_ID,
            expression="見る",
            reading="みる",
            furigana="見[み]る",
            meanings=["to see"],
            part_of_speech="verb",
            verb_group="ichidan",
            tags=["lesson"],
            examples=[
                ExampleSentence(
                    japanese="日曜日に映画を見ます。",
                    furigana="日曜日[にちようび]に 映画[えいが]を 見[み]ます。",
                    english=WHOLE_DECK_ONLY_ENGLISH,
                    audio=OTHER_SENTENCE_CLIP,
                    register="polite",
                )
            ],
        ),
    ]


def _build_real_package(root: Path) -> tuple[ProjectConfig, Path, str]:
    """Write the clips, build the deck's real `.apkg`, and prove the bytes landed.

    The archive check is not decoration: everything after it is a claim about
    *this* package's media, so the member the media index names for the sentence
    has to be byte-identical to the fixture before a browser is asked anything.
    """
    config = _project(root)
    for name, payload in (
        (SENTENCE_CLIP, SENTENCE_WAV),
        (WORD_CLIP, WORD_WAV),
        (OTHER_SENTENCE_CLIP, OTHER_SENTENCE_WAV),
    ):
        (root / "media" / name).write_bytes(payload)
    (root / "vocabulary.json").write_text(
        json.dumps([record.to_dict() for record in _records()], ensure_ascii=False),
        encoding="utf-8",
    )
    deck = root / "decks" / "lesson.yaml"
    deck.write_text(
        "deck:\n"
        "  name: Lesson\n"
        '  source: "../vocabulary.json"\n'
        "  cards:\n"
        "    recognition: true\n"
        "    production: true\n"
        "    reading: true\n",
        encoding="utf-8",
    )

    from japanese_anki.exporters.anki import build_deck

    package = root / "dist" / "lesson.apkg"
    build_deck(deck, config, output_path=package)
    digest = hashlib.sha256(package.read_bytes()).hexdigest()

    with zipfile.ZipFile(package) as archive:
        index = json.loads(archive.read("media").decode("utf-8"))
        by_name = {str(name): str(member) for member, name in index.items()}
        assert SENTENCE_CLIP in by_name, sorted(by_name)
        packaged = archive.read(by_name[SENTENCE_CLIP])
    assert packaged == SENTENCE_WAV, "the archive carries the exact fixture clip"
    return config, package, digest


# --- the browser this gate requires --------------------------------------------


def _required_playwright() -> Any:
    """Playwright, as a requirement. No `importorskip`: absence is a failure."""
    try:
        import playwright.sync_api as sync_api
    except ImportError as exc:  # pragma: no cover - reported as the failure below
        pytest.fail(
            "Sentence-audio playback can only be proven in a browser, so this "
            "test is mandatory rather than optional. Playwright is missing: "
            f"{exc}. Install it with: python -m pip install -e '.[dev]'"
        )
    return sync_api


def _required_installed_chrome(playwright: Any) -> Any:
    """An *installed* Chrome/Chromium, as a requirement, and never a download.

    The same search the frozen helpers do — PATH, the two macOS applications,
    and any browser Playwright already installed — with the one difference this
    gate needs: absence fails instead of skipping, and nothing calls
    `playwright install`.
    """
    chromium = playwright.chromium
    candidates: list[Path] = []
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))
    known = [
        Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
        Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
    ]
    # No previously installed Playwright browser is an ordinary absence here,
    # not an error: the two application paths and PATH are the usual answer.
    with contextlib.suppress(Exception):
        known.append(Path(chromium.executable_path))
    candidates.extend(path for path in known if path.is_file())
    installed = tuple(dict.fromkeys(path.resolve() for path in candidates))
    if not installed:
        pytest.fail(
            "This gate proves the delivered package's sentence audio actually "
            "plays, which needs a real browser. No installed Chrome/Chromium was "
            "found on PATH or in /Applications. Install one — this test never "
            "downloads a browser — and run it again."
        )
    failures = []
    for executable in installed:
        try:
            return chromium.launch(headless=True, executable_path=str(executable))
        except Exception as exc:  # noqa: BLE001 - reported below by name
            failures.append(f"{executable}: {exc}")
    pytest.fail("installed Chrome/Chromium could not launch:\n" + "\n".join(failures))


# --- what the page is asked, in the page ---------------------------------------

#: Every media event the element under test can tell us about, recorded with the
#: element's own state *at that event*: the `currentTime` it reached, and whether
#: it was paused or ended at that moment. Installed *before* the click, so "it
#: advanced while it was playing" is a sequence this test watched happen.
#:
#: The three belong together, in the listener. A record with `paused: false` and
#: `ended: false` is the element mid-playback by its own account, and a later
#: such record at a greater `currentTime` is advancement however long afterwards
#: this process gets around to reading the array.
_OBSERVE = """
(selector) => {
  const element = document.querySelector(selector);
  window.__jankiAudio = {records: []};
  const names = ["loadstart", "loadedmetadata", "play", "playing", "timeupdate",
                 "pause", "ended", "error", "stalled", "abort", "emptied"];
  for (const name of names) {
    element.addEventListener(name, () => {
      window.__jankiAudio.records.push({
        name: name,
        currentTime: element.currentTime,
        paused: element.paused,
        ended: element.ended,
      });
    });
  }
  return true;
}
"""

#: One element's live state, read before the click and after the clip is over —
#: never as a mid-playback sample, which is the record's job. The clip's own
#: bytes are compared *in the page* and reported as two booleans: a 0.75-second
#: WAV is a 48 kB data URI, and carrying it back into the assertions twice would
#: bury the observation this test exists to record under its own fixture.
_STATE = """
({selector, expected, other}) => {
  const element = document.querySelector(selector);
  const src = element.getAttribute("src") || "";
  return {
    currentTime: element.currentTime,
    duration: element.duration,
    paused: element.paused,
    ended: element.ended,
    readyState: element.readyState,
    networkState: element.networkState,
    error: element.error ? {code: element.error.code, message: element.error.message}
                         : null,
    srcPrefix: src.slice(0, 24),
    srcLength: src.length,
    srcIsThisClip: expected !== "" && src.includes(expected),
    srcIsTheOtherClip: other !== "" && src.includes(other),
  };
}
"""

_RECORDED = "() => window.__jankiAudio"

_SECTION_TEXT = """
(selector) => {
  const element = document.querySelector(selector);
  const section = element.closest("section.example");
  return section ? section.textContent : "";
}
"""


def test_the_selected_sentences_packaged_clip_really_plays_in_a_browser(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One real clip, one real click, and the playback that follows it.

    The single end-to-end claim `docs/DESIGN.md` makes about a finished job's
    preview and nothing else in this repository checks: the sentence audio in
    the final HTML preview is audio a browser can play.
    """
    unavailable = card_preview.preview_unavailable()
    if unavailable is not None:
        pytest.fail(
            "This gate draws real cards out of a real package, so the preview "
            f"extras are a requirement rather than an option: {unavailable}"
        )
    sync_api = _required_playwright()

    config, package, digest = _build_real_package(tmp_path)

    preview = card_preview.render_packaged_card_preview(
        config,
        package,
        package_sha256=digest,
        deck_name="Lesson",
        directions=("recognition", "production", "reading"),
        scope_record_ids=[SELECTED_ID],
        subtitle="The cards this study job delivered",
    )
    page_path = card_preview.write_card_preview(preview, tmp_path / "preview.html")
    body = preview.html.decode("utf-8")

    # The selection is one of two notes, so the page can only be playing the
    # selected sentence's clip: the other note's bytes are not in the document.
    assert preview.package_sha256 == digest
    assert preview.note_count == 1 and preview.deck_note_count == 2
    assert {card.record_id for card in preview.cards} == {SELECTED_ID}
    assert _b64(SENTENCE_WAV) in body, "the selected sentence's clip is inlined"
    assert _b64(WORD_WAV) in body, "and so is its word clip, for the contrast"
    assert _b64(OTHER_SENTENCE_WAV) not in body
    assert WHOLE_DECK_ONLY_ENGLISH not in body

    card = '[data-preview-card][data-index="0"]'
    sentence_selector = f"{card} [data-preview-answer] .example-audio audio"
    word_selector = f"{card} [data-preview-answer] .audio:not(.example-audio) audio"

    requests: list[str] = []
    observations: dict[str, Any] = {}
    with sync_api.sync_playwright() as playwright:
        browser = _required_installed_chrome(playwright)
        try:
            page = browser.new_page()
            page.on("request", lambda request: requests.append(request.url))
            page.goto(page_path.as_uri())
            page.wait_for_selector("body[data-preview-ready='true']")

            # The page is the one that was rendered, opened on its question.
            assert page.get_attribute(card, "data-record-id") == SELECTED_ID
            assert page.get_attribute(card, "data-template") == "Recognition"
            assert not page.locator(f"{card} [data-preview-answer]").is_visible()
            assert page.locator(sentence_selector).count() == 1
            assert page.locator(word_selector).count() == 1

            # The owner's own interaction: Show Answer, then the control.
            page.click("#preview-flip")
            assert page.locator(f"{card} [data-preview-answer]").is_visible()
            sentence = page.locator(sentence_selector)
            assert sentence.is_visible()

            # This element is the selected *sentence's* control: it carries that
            # clip's exact bytes, sits inside that sentence's own block, and is
            # not the word control beside it.
            sentence_state = {
                "selector": sentence_selector,
                "expected": _b64(SENTENCE_WAV),
                "other": _b64(WORD_WAV),
            }
            word_state = {
                "selector": word_selector,
                "expected": _b64(WORD_WAV),
                "other": _b64(SENTENCE_WAV),
            }
            before = page.evaluate(_STATE, sentence_state)
            word_before = page.evaluate(_STATE, word_state)
            assert before["srcIsThisClip"] and not before["srcIsTheOtherClip"]
            assert before["srcPrefix"].startswith("data:audio/")
            assert word_before["srcIsThisClip"] and not word_before["srcIsTheOtherClip"]
            section_text = page.evaluate(_SECTION_TEXT, sentence_selector)
            assert SELECTED_SENTENCE_ENGLISH in section_text
            assert before["currentTime"] == 0 and before["paused"] is True
            assert before["error"] is None

            page.evaluate(_OBSERVE, sentence_selector)
            box = sentence.bounding_box()
            assert box is not None and box["width"] > 40 and box["height"] > 20
            # A real mouse click on the native play button of the real control,
            # at the left of the element where Chrome draws it. Not `play()`:
            # a fulfilled promise is not playback.
            sentence.click(position={"x": 16.0, "y": box["height"] / 2})

            # Let it play to the end, then read what the page recorded. This
            # process has no mid-clip deadline to meet: the evidence that it was
            # running, and where, was written by the listener as it ran.
            try:
                page.wait_for_function(
                    "() => window.__jankiAudio.records.some((r) => r.name === 'ended')",
                    timeout=PLAYBACK_TIMEOUT_MS,
                )
            except sync_api.Error as exc:
                recorded = page.evaluate(_RECORDED)
                state = page.evaluate(_STATE, sentence_state)
                pytest.fail(
                    "the selected sentence's clip did not play through in the "
                    f"browser: {exc}\nrecorded={json.dumps(recorded)}\n"
                    f"state={json.dumps(state)}"
                )

            after = page.evaluate(_STATE, sentence_state)
            recorded = page.evaluate(_RECORDED)
            word_after = page.evaluate(_STATE, word_state)
            observations = {
                "records": recorded["records"],
                "after": after,
                "word_control_after": word_after,
                "requests": requests,
            }
        finally:
            browser.close()

    records = observations["records"]
    events = [record["name"] for record in records]
    times = [record["currentTime"] for record in records]
    after = observations["after"]
    word_after = observations["word_control_after"]

    # Nothing failed to decode, and no media error was raised at any point.
    assert after["error"] is None, after["error"]
    assert "error" not in events and "abort" not in events, events
    assert "emptied" not in events, events

    # A finite duration, and the one these 0.75 seconds of frames encode. This
    # is what separates "the sentence played" from "something played": the word
    # clip on the same card is 0.25s and the other note's is 1.5s.
    assert math.isfinite(after["duration"]), after["duration"]
    assert after["duration"] == pytest.approx(SENTENCE_SECONDS, abs=0.02), after
    assert _wav_duration(SENTENCE_WAV) == SENTENCE_SECONDS
    assert after["duration"] != pytest.approx(WORD_SECONDS, abs=0.02)

    # It really ran, and it ran forwards: play, playing, timeupdate, ended, with
    # `currentTime` strictly greater at a later moment the element still called
    # itself playing. One such pair is the claim — a `playing` at 0 and a later
    # non-paused, non-ended tick past it is playback advancing, and demanding
    # several intermediate ticks would only re-introduce a dependence on how
    # often a browser happens to fire them.
    assert "play" in events and "playing" in events and "ended" in events
    assert "timeupdate" in events, events
    assert times == sorted(times), records
    playing_moments = [
        record for record in records if not record["paused"] and not record["ended"]
    ]
    advancing = [
        (earlier, later)
        for index, earlier in enumerate(playing_moments)
        for later in playing_moments[index + 1 :]
        if later["currentTime"] > earlier["currentTime"]
    ]
    assert advancing, records
    assert advancing[0][1]["currentTime"] > 0, advancing[0]

    # And it ran all the way out: the `ended` event was dispatched at the end of
    # the clip, not at some earlier point the element gave up at.
    ended_record = next(record for record in records if record["name"] == "ended")
    assert ended_record["currentTime"] == pytest.approx(after["duration"], abs=0.05), records
    assert after["currentTime"] == pytest.approx(after["duration"], abs=0.05), after
    assert after["ended"] is True and after["paused"] is True
    assert after["readyState"] >= 2, after

    # The clip beside it never moved, so what played was this sentence's own
    # control rather than the card's word audio.
    assert word_after["currentTime"] == 0 and word_after["paused"] is True
    assert word_after["error"] is None

    # Offline: a self-contained file, and the browser went nowhere for it. The
    # clip is a `data:` URI, which is already in the document and is not a
    # request at all — so "no request but the page itself" is the whole traffic.
    assert requests, "the page itself was fetched, so requests were observed"
    assert all(url.startswith("file://") for url in requests), requests
    assert preview.content_security_policy.startswith("default-src 'none';")
    assert "media-src data:;" in preview.content_security_policy

    with capsys.disabled():
        print(
            "\nbrowser sentence-audio observations: "
            + json.dumps(observations, ensure_ascii=False)
        )
