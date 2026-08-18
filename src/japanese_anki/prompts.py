"""The files janki sends to a model, and the one function that reads them.

Every substantive task instruction lives in `prompts/` as Markdown, one file
per pass and input shape, and Anthropic receives it **byte for byte**. There is
no placeholder syntax, comment convention, or conditional section. The only
Python-side instructions are transport structure: terse schema field labels
and Codex's JSON-only/no-tools preamble. Neither carries Japanese policy.

**This module is a file read, not a renderer.** A prompt that would need a
branch in its *instruction prose* is two prompts, which is why extraction's
three modes are three complete files rather than one file plus three rule
blocks. Composing a record's own data into the user turn stays in Python: that
is data, not instruction. The user turn, schema labels, and Codex transport
preamble are the deliberately visible non-file inputs.

**Nothing is cached.** A prompt is re-read from disk on every call, so editing
a file changes the next run with no rebuild and no way for an edit to be
silently stale. The cost is a few kilobytes of file I/O against a network call
to a language model.

**A missing file is an error, never an empty string.** Silently sending no
instructions would produce plausible output that ignores every rule this
directory exists to state — the expensive kind of wrong, because it looks
like success.

**Every model answer records which prompt produced it.** Extraction carries
the source template, style, user turn, and wire-schema fingerprints into
staging, its archive, and the pattern store. Bare-word enrichment records the
provider plus its normalized transport prompt and wire schema in the full
request identity used by live ledger entries, batch descriptors, and large-run
staging metadata. Coverage approval records its own instruction fingerprint.
Together those artifacts can be traced to exact text through `git log prompts/`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from japanese_anki.errors import JankiError

__all__ = [
    "DIRECTORY",
    "PromptError",
    "fingerprint",
    "load",
    "path_for",
    "request_fingerprint",
    "schema_fingerprint",
]

#: Where the templates live, relative to the project root. Top-level rather
#: than under `docs/` (these are operational inputs, not reading material) or
#: `templates/` (that is the card HTML).
DIRECTORY = Path("prompts")

#: Characters that occupy a file without instructing anything.
#:
#: Stripped in **one** pass together with whitespace. Two passes — strip
#: whitespace, then strip the marks — let `"\u200b \u200b"` through, because
#: the space survives the first and the marks survive the second.
#:
#: The whitespace half is taken from `str`'s own definition rather than a
#: hand-written list of spaces, which is how an earlier attempt dropped
#: U+001C–U+001F, U+0085, U+2028 and U+2029: they are whitespace to `str.strip`
#: and were absent from the literal, so a file holding one alone started
#: passing a guard it used to fail. Bounded at U+3000, the highest whitespace
#: code point — scanning all 0x110000 cost 54 ms at import, a quarter of
#: janki's cold start, to rediscover the same 29 characters.
#:
#: The rest are the Cc and Cf ranges that render as nothing. This does not
#: catch every invisible character Unicode has — U+3164 HANGUL FILLER and
#: U+2800 BRAILLE PATTERN BLANK are letters and symbols by category, and a file
#: of one of those still reads as content. The guard is for a truncated or
#: emptied prompt, not for an adversary.
_INVISIBLE = (
    "".join(chr(c) for c in range(0x3001) if chr(c).isspace())
    + "".join(chr(c) for c in range(0x00, 0x20))
    + "\x7f\u00ad\u061c\u180e"
    + "".join(chr(c) for c in range(0x200B, 0x2010))
    + "".join(chr(c) for c in range(0x2060, 0x2065))
    + "\ufeff\ufff9\ufffa\ufffb"
)


class PromptError(JankiError):
    """A prompt file could not be read."""


def path_for(root: Path, name: str) -> Path:
    """The file ``name`` names, under ``root``.

    ``name`` is a bare stem — ``"enrich-bare-word"`` — because a caller that
    could pass a path could pass one outside the directory, and the point of
    this module is that every instruction sent to a model is a file someone can
    find by name.
    """
    return Path(root) / DIRECTORY / f"{name}.md"


def load(root: Path, name: str) -> str:
    """The text of prompt ``name``, exactly as it will be sent.

    Read fresh every time. No normalization, no stripping: a trailing newline
    the author put there is part of the prompt, and a loader that tidied the
    text would make the file and the request differ in a way nobody could see.
    """
    path = path_for(root, name)
    try:
        # Bytes, then decode. `read_text` translates CRLF to LF, which would
        # make "sent byte for byte" false on any checkout with `core.autocrlf`
        # on and — worse — make the recorded sha-256 match no version of the
        # file in `git log prompts/`.
        raw = path.read_bytes()
    except OSError as exc:
        raise PromptError(
            f"Could not read the prompt at {path}: {exc}. Every model call sends "
            "one, so janki will not run a pass without it."
        ) from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PromptError(
            f"The prompt at {path} is not UTF-8: {exc}. Prompts are Markdown "
            "and are sent as written, so janki will not guess an encoding."
        ) from exc
    # `\ufeff` and other zero-width marks are truthy under `str.strip`, so a
    # prompt truncated to its byte-order mark alone would slip past an
    # `if not text.strip()` guard and buy a full paid pass with no
    # instructions — the exact failure this refuses.
    if not text.strip(_INVISIBLE):
        # An empty file is the failure this module exists to prevent, arriving
        # by a different door: a truncated or emptied prompt would buy a full
        # paid pass with no instructions and report success.
        raise PromptError(
            f"The prompt at {path} is empty. A pass with no instructions costs "
            "the same as one with them and returns something that looks like an "
            "answer, so janki will not run it."
        )
    return text


def fingerprint(text: str) -> str:
    """The sha-256 of a prompt's exact bytes.

    Over the text rather than the file, so a caller that assembled a user turn
    from a template plus record data can fingerprint what it actually sent.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _schema_value(schema: Any) -> Any:
    """Return the JSON-schema value a model request actually carries."""
    if hasattr(schema, "model_json_schema"):
        return schema.model_json_schema()
    return schema


def _canonical_json(value: Any) -> str:
    """Serialize structural prompt data without depending on dict insertion order."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def schema_fingerprint(schema: Any) -> str:
    """Identify the canonical response schema sent with a structured call."""
    return fingerprint(_canonical_json(_schema_value(schema)))


def request_fingerprint(
    *,
    provider: str,
    style_guide: str,
    task_template: str,
    user_turn: str,
    transport_prompt: Any,
    schema: Any,
) -> str:
    """Identify every prompt channel and the response contract for one call.

    Named channels prevent concatenation collisions (``"ab" + "c"`` versus
    ``"a" + "bc"``). Prompt text remains exact — including line endings —
    while only the schema's object-key order is canonicalized.
    """
    return fingerprint(
        _canonical_json(
            {
                "provider": provider,
                "schema": _schema_value(schema),
                "style_guide": style_guide,
                "task_template": task_template,
                "transport_prompt": transport_prompt,
                "user_turn": user_turn,
            }
        )
    )
