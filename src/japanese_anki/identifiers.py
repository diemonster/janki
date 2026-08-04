from __future__ import annotations

import hashlib
import unicodedata


def normalize_identity_part(value: str) -> str:
    """Normalize text used in durable record identities."""
    return unicodedata.normalize("NFKC", value).strip()


def stable_record_id(expression: str, reading: str = "") -> str:
    expression_part = normalize_identity_part(expression)
    reading_part = normalize_identity_part(reading)
    return f"word:{expression_part}:{reading_part}"


def short_fingerprint(*values: str, length: int = 12) -> str:
    payload = "\x1f".join(normalize_identity_part(value) for value in values)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]
