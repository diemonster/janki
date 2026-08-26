from __future__ import annotations

import html
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from japanese_anki.errors import JankiError
from japanese_anki.exporters.anki import resolve_deck_records
from japanese_anki.models import VocabularyRecord


class PreviewError(JankiError):
    pass


def _resolved_preview(
    deck_path: Path,
    requested_ids: tuple[str, ...] | None,
) -> tuple[dict[str, Any], tuple[VocabularyRecord, ...]]:
    """Resolve the whole deck, then optionally project one exact ID scope."""
    deck_config, resolved = resolve_deck_records(deck_path.resolve())
    if requested_ids is None:
        return deck_config, tuple(resolved)

    by_id: dict[str, VocabularyRecord] = {}
    for record in resolved:
        if record.id in by_id:
            raise PreviewError(
                f"The deck resolves card ID {record.id!r} more than once."
            )
        by_id[record.id] = record
    missing = tuple(record_id for record_id in requested_ids if record_id not in by_id)
    if missing:
        names = ", ".join(repr(record_id) for record_id in missing)
        raise PreviewError(
            f"The requested card IDs are not in the resolved deck: {names}."
        )
    return deck_config, tuple(by_id[record_id] for record_id in requested_ids)


def resolve_preview_records(
    deck_path: Path,
    record_ids: Sequence[str],
) -> tuple[VocabularyRecord, ...]:
    """Return exact deck-resolved record versions in requested order.

    Resolving the complete deck applies its real selectors and inline
    overrides. Missing IDs therefore refuse instead of previewing a canonical
    record that this deck would not actually build. No files are written.
    """
    if isinstance(record_ids, str | bytes | bytearray) or not isinstance(
        record_ids, Sequence
    ):
        raise PreviewError("Preview card IDs must be a sequence of IDs, not text.")
    requested_ids = tuple(record_ids)
    if not requested_ids:
        raise PreviewError("Preview needs at least one card ID.")
    if any(
        not isinstance(record_id, str) or not record_id.strip()
        for record_id in requested_ids
    ):
        raise PreviewError("Preview card IDs must be nonblank strings.")
    seen: set[str] = set()
    for record_id in requested_ids:
        if record_id in seen:
            raise PreviewError(f"Preview card ID {record_id!r} is repeated.")
        seen.add(record_id)
    _deck_config, records = _resolved_preview(deck_path, requested_ids)
    return records


def build_preview(deck_path: Path, output_path: Path) -> Path:
    deck_config, records = _resolved_preview(deck_path, None)
    title = str(deck_config.get("name", deck_path.stem))
    cards = []
    for record in records:
        # The same two slots the card has, filled by the same rules. With one
        # slot and the exporter's rule, a record whose examples are all casual
        # previewed blank while the built card showed the sentence under
        # "Casually" — a preview that disagrees with the deck it previews.
        example = record.main_example()
        casual = record.example_in("casual")
        meanings = "<br>".join(html.escape(item) for item in record.meanings)
        tags = " ".join(f"<span>{html.escape(tag)}</span>" for tag in record.tags)
        casual_block = (
            f"""
              <div class="example-casual">
                <span class="register">Casually</span>
                <div class="example-ja">{html.escape(casual.japanese)}</div>
                <div class="example-en">{html.escape(casual.english)}</div>
              </div>
            """
            if casual.japanese
            else ""
        )
        cards.append(
            f"""
            <article class="note">
              <div class="expression">{html.escape(record.expression)}</div>
              <div class="reading">{html.escape(record.reading)}</div>
              <div class="meanings">{meanings}</div>
              <div class="example-ja">{html.escape(example.japanese)}</div>
              <div class="example-en">{html.escape(example.english)}</div>
              {casual_block}
              <div class="tags">{tags}</div>
            </article>
            """
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)} preview</title>
<style>
body {{
  font-family: -apple-system, BlinkMacSystemFont, sans-serif;
  max-width: 900px;
  margin: 2rem auto;
  padding: 0 1rem;
  background: #f4f4f4;
  color: #181818;
}}
.note {{
  background: white;
  border: 1px solid #ddd;
  border-radius: 12px;
  margin: 1rem 0;
  padding: 1.25rem;
}}
.expression {{
  font-size: 2.2rem;
  font-family: "Hiragino Sans", "Yu Gothic", sans-serif;
}}
.reading {{ font-size: 1.1rem; margin-top: .25rem; }}
.meanings {{ font-size: 1.15rem; margin: 1rem 0; }}
.example-ja {{ font-size: 1.3rem; margin-top: 1rem; }}
.example-en {{ opacity: .8; margin-top: .3rem; }}
.tags span {{
  display: inline-block;
  margin: .75rem .3rem 0 0;
  padding: .15rem .45rem;
  border-radius: 999px;
  background: #eee;
  font-size: .8rem;
}}
@media (prefers-color-scheme: dark) {{
  body {{ background: #111; color: #eee; }}
  .note {{ background: #1e1e1e; border-color: #444; }}
  .tags span {{ background: #333; }}
}}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<p>{len(records)} notes</p>
{''.join(cards)}
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    return output_path
