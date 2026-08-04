from __future__ import annotations

import html
from pathlib import Path

from japanese_anki.exporters.anki import resolve_deck_records


def build_preview(deck_path: Path, output_path: Path) -> Path:
    deck_config, records = resolve_deck_records(deck_path.resolve())
    title = str(deck_config.get("name", deck_path.stem))
    cards = []
    for record in records:
        example = record.first_example
        meanings = "<br>".join(html.escape(item) for item in record.meanings)
        tags = " ".join(f"<span>{html.escape(tag)}</span>" for tag in record.tags)
        cards.append(
            f"""
            <article class="note">
              <div class="expression">{html.escape(record.expression)}</div>
              <div class="reading">{html.escape(record.reading)}</div>
              <div class="meanings">{meanings}</div>
              <div class="example-ja">{html.escape(example.japanese)}</div>
              <div class="example-en">{html.escape(example.english)}</div>
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
