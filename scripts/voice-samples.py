#!/usr/bin/env python3
"""Audition VOICEVOX voices with janki's own pipeline.

Generates one clip per speaker — a word with its accent *forced*, and a full
sentence — plus an HTML page to compare them, so what you hear is what a card
will sound like rather than a generic TTS demo. The word matters: 話す through
`/accent_phrases?is_kana=true` carries its real accent (ハナ'ス), while the same
word left to the engine comes out as 箸.

    python3 scripts/voice-samples.py                 # every speaker
    python3 scripts/voice-samples.py --male          # only the male voices
    python3 scripts/voice-samples.py --word 橋 --reading はし --pattern LHL

Set your pick as `[tts] voicevox_speaker` in janki.toml, then re-voice with
`janki audio --words --examples --force` — voice and speed are not part of the
content fingerprint, so nothing is stale without it.

**Why it restarts the engine.** VOICEVOX loads a model per speaker on demand and
keeps it, so sampling the whole roster grows unboundedly: a 2 GiB VM gets about
nine speakers in before the container is OOM-killed mid-run (exit 137). Cycling
the container every few speakers costs a couple of seconds and is cheaper than
discovering that at speaker 9 of 43. Give the VM more memory and you can raise
`--batch`, or pass `--no-restart` if you are not running it in a container.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import pathlib
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from japanese_anki.pitch import to_aquestalk  # noqa: E402
from japanese_anki.tts.voicevox import VoicevoxProvider, urllib_transport  # noqa: E402

#: Speakers VOICEVOX's own character notes describe as male. Marked in the page
#: rather than filtered out of it — the rest are there to listen to.
MALE = {11, 12, 13, 21, 42, 52, 53, 89, 94, 99, 100, 118, 122}


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "speaker"


def restart(container: str, base_url: str) -> bool:
    """Cycle the engine and wait for it. False if it did not come back."""
    result = subprocess.run(
        ["docker", "restart", container], env=os.environ, capture_output=True, timeout=180
    )
    if result.returncode != 0:
        # Not fatal on its own — the engine may be running outside this
        # container — but silence here is how a wrong --container turns into a
        # two-minute stall with no explanation.
        detail = result.stderr.decode(errors="replace").strip() or "docker restart failed"
        print(f"  {container}: {detail}", file=sys.stderr)
    provider = VoicevoxProvider(base_url=base_url)
    for _ in range(60):
        if provider.available():
            return True
        time.sleep(2)
    return False


def synth(base_url: str, text: str, speaker: int, *, kana: bool, speed: float) -> bytes:
    """One clip, by the same route `janki audio` takes."""
    if kana:
        _, body = urllib_transport(
            "POST",
            f"{base_url}/accent_phrases?text={urllib.parse.quote(text)}"
            f"&is_kana=true&speaker={speaker}",
        )
        forced, plain = json.loads(body), text.replace("'", "")
    else:
        forced, plain = None, text
    _, body = urllib_transport(
        "POST", f"{base_url}/audio_query?text={urllib.parse.quote(plain)}&speaker={speaker}"
    )
    query = json.loads(body)
    if forced is not None:
        query["accent_phrases"] = forced
    if speed != 1.0:
        query["speedScale"] = speed
    _, wav = urllib_transport("POST", f"{base_url}/synthesis?speaker={speaker}", query)
    return wav


def write_page(out: pathlib.Path, rows: list[dict], word: str) -> pathlib.Path:
    def cell(base: str, kind: str, speed: str, label: str) -> str:
        name = f"{base}-{kind}-{speed}.wav"
        if not (out / name).exists():
            return "<td></td>"
        return (
            f'<td><div class="lbl">{html.escape(label)}</div>'
            f'<audio controls preload="none" src="{html.escape(name)}"></audio></td>'
        )

    body = []
    for row in sorted(rows, key=lambda r: (r["id"] not in MALE, r["id"])):
        tag = '<span class="tag male">male</span>' if row["id"] in MALE else ""
        extra = f'{row["styles"]} styles' if row["styles"] > 1 else ""
        body.append(
            f'<tr data-male="{str(row["id"] in MALE).lower()}">'
            f'<th><span class="jp">{html.escape(row["name"])}</span> {tag}<br>'
            f'<code>voicevox_speaker = {row["id"]}</code>'
            f'<div class="note">{html.escape(extra)}</div></th>'
            + cell(row["base"], "word", "1.0", f"{word} · 1.0×")
            + cell(row["base"], "word", "0.85", f"{word} · 0.85×")
            + cell(row["base"], "sentence", "0.85", "sentence · 0.85×")
            + "</tr>"
        )

    page = out / "index.html"
    page.write_text(
        f"""<!doctype html>
<meta charset="utf-8"><title>janki — VOICEVOX voices</title>
<style>
 body {{ font-family: -apple-system, sans-serif; max-width: 1050px;
        margin: 2rem auto; padding: 0 1rem; }}
 table {{ border-collapse: collapse; width: 100%; }}
 th, td {{ border-bottom: 1px solid #ddd; padding: .55rem .5rem;
           vertical-align: top; text-align: left; }}
 th {{ width: 15rem; font-weight: normal; }}
 .jp {{ font-size: 1.15rem; font-weight: 600; }}
 .note {{ opacity: .6; font-size: .78rem; margin-top: .2rem; }}
 .lbl {{ font-size: .72rem; opacity: .65; margin-bottom: .15rem; }}
 .tag {{ font-size: .68rem; padding: .1rem .35rem; border-radius: 3px;
         vertical-align: middle; }}
 .male {{ background: #d8e8ff; color: #17457a; }}
 audio {{ width: 185px; }}
 code {{ background: #f0f0f0; padding: .1rem .3rem; border-radius: 3px; font-size: .82rem; }}
 @media (prefers-color-scheme: dark) {{
   body {{ background:#111; color:#eee; }} th, td {{ border-color:#3a3a3a; }}
   code {{ background:#2a2a2a; }} .male {{ background:#1e3a5f; color:#bcd8ff; }}
 }}
</style>
<h1>VOICEVOX voices — {len(rows)} speakers</h1>
<p>Every clip goes through janki's forced-accent path, so <b>{html.escape(word)}</b>
carries its real accent rather than the engine's guess — this is what a card will
sound like. One <em>talk</em> style per speaker.</p>
<p><label><input type="checkbox" id="only-male"> male voices only</label></p>
<table><tbody>{"".join(body)}</tbody></table>
<script>
document.getElementById('only-male').addEventListener('change', e => {{
  for (const row of document.querySelectorAll('tbody tr'))
    row.style.display = (!e.target.checked || row.dataset.male === 'true') ? '' : 'none';
}});
</script>
""",
        encoding="utf-8",
    )
    return page


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path.home() / "Desktop" / "janki-voice-samples")
    parser.add_argument("--url", default="http://localhost:50021")
    parser.add_argument("--word", default="話す")
    parser.add_argument("--reading", default="はなす")
    parser.add_argument("--pattern", default="LHLL",
                        help="jpdb accent pattern for --reading (one per kana, plus the particle)")
    parser.add_argument("--sentence", default="毎日、妻と日本語で話します。")
    parser.add_argument("--male", action="store_true", help="Only the male voices.")
    parser.add_argument("--batch", type=int, default=5,
                        help="Restart the engine every N speakers (0 to never).")
    parser.add_argument("--container", default="janki-voicevox")
    parser.add_argument("--no-restart", action="store_true")
    args = parser.parse_args()

    kana = to_aquestalk(args.reading, args.pattern)
    args.out.mkdir(parents=True, exist_ok=True)
    cycle = 0 if args.no_restart else args.batch

    try:
        with urllib.request.urlopen(f"{args.url}/speakers", timeout=30) as response:
            speakers = json.load(response)
    except (OSError, ValueError) as exc:
        # ValueError too: a non-JSON body — a proxy or captive portal
        # answering on the engine's port — makes `json.load` raise
        # JSONDecodeError, which is one. (A `--url` typo is usually already an
        # OSError: `localhost:50021` parses as scheme `localhost` and comes
        # back as URLError. Only a colon-less `--url localhost` is a bare
        # ValueError. Checked, 2026-08-08.)
        print(f"No VOICEVOX engine at {args.url}: {exc}", file=sys.stderr)
        print(VoicevoxProvider(base_url=args.url).launch_hint, file=sys.stderr)
        return 1

    targets = []
    for speaker in speakers:
        talk = [s for s in speaker["styles"] if s.get("type", "talk") == "talk"]
        if talk and (not args.male or talk[0]["id"] in MALE):
            targets.append((speaker["name"], talk[0]["id"], len(speaker["styles"])))

    rows, attempted, skipped = [], 0, []
    for name, sid, n_styles in targets:
        if cycle and attempted and attempted % cycle == 0 and not restart(
            args.container, args.url
        ):
            print("  engine did not come back; stopping", file=sys.stderr)
            break
        attempted += 1
        base = f"{slug(name)}-{sid}"
        wanted = [
            (f"{base}-word-1.0.wav", kana, True, 1.0),
            (f"{base}-word-0.85.wav", kana, True, 0.85),
            (f"{base}-sentence-0.85.wav", args.sentence, False, 0.85),
        ]
        # Clear this speaker's slots first. `cell()` only asks whether the file
        # exists, so a leftover from an earlier run — a different --word, even —
        # would be embedded under this run's label, which is the one thing the
        # page must not do.
        for filename, *_ in wanted:
            (args.out / filename).unlink(missing_ok=True)

        ok = False
        for attempt in (1, 2):
            try:
                for filename, text, forced, speed in wanted:
                    (args.out / filename).write_bytes(
                        synth(args.url, text, sid, kana=forced, speed=speed))
                ok = True
                break
            except Exception as exc:  # noqa: BLE001 - any engine failure is retryable once
                if attempt == 2 or cycle == 0 or not restart(args.container, args.url):
                    print(f"  skipped {name} ({sid}): {exc}", file=sys.stderr)
                    break
        if not ok:
            # Counted as skipped rather than done: reporting a speaker complete
            # because the loop reached the bottom is how a run that voiced nine
            # speakers printed 43/43 and rendered a page claiming all of them.
            for filename, *_ in wanted:
                (args.out / filename).unlink(missing_ok=True)
            skipped.append(f"{name} ({sid})")
            continue
        rows.append({"name": name, "id": sid, "base": base, "styles": n_styles})
        print(f"  {len(rows)}/{len(targets)} {name} ({sid})")

    page = write_page(args.out, rows, args.word)
    print(f"\n{len(rows)} of {len(targets)} speakers -> {page}")
    if skipped:
        print(f"skipped {len(skipped)}: {', '.join(skipped)}", file=sys.stderr)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
