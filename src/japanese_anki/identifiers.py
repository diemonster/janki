from __future__ import annotations

import hashlib
import unicodedata


def normalize_identity_part(value: str) -> str:
    """Normalize text used in durable record identities."""
    return unicodedata.normalize("NFKC", value).strip()


# Every Unicode range that holds Han characters a Japanese record can carry,
# inclusive. Kanji live in six blocks across three planes, not one contiguous
# BMP run: 𠮟 (U+20B9F, the JIS2004 form of しかる) and 𩸽 (U+29E3D, hokke) are
# ordinary Japanese words that Shirabe exports, and both sit above U+FFFF.
_HAN_RANGES: tuple[tuple[int, int], ...] = (
    (0x3005, 0x3005),  # 々, the iteration mark: kanji for identity purposes
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0x20000, 0x2EBEF),  # CJK Unified Ideographs Extensions B through I
    (0x2F800, 0x2FA1F),  # CJK Compatibility Ideographs Supplement
)


def contains_kanji(value: str) -> bool:
    """Whether ``value`` holds a Han character.

    One definition for the whole project: the importer decides with it whether
    a row may be imported at all, and the validator reports with it what got in
    anyway. Two copies would drift, and a copy that is too narrow does not miss
    a warning — it mints ``word:<kanji>:<kanji>``, an ID that can never be
    corrected without orphaning Anki review history.

    Iteration is over code points (Python strings iterate that way, so a
    supplementary-plane kanji is one character here rather than two surrogate
    halves) and the ranges cover every plane kanji live in.
    """
    return any(any(low <= code <= high for low, high in _HAN_RANGES) for code in map(ord, value))


def stable_record_id(expression: str, reading: str = "") -> str:
    expression_part = normalize_identity_part(expression)
    reading_part = normalize_identity_part(reading)
    return f"word:{expression_part}:{reading_part}"


def short_fingerprint(*values: str, length: int = 12) -> str:
    payload = "\x1f".join(normalize_identity_part(value) for value in values)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]
