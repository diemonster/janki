# Import Shirabe Prompt

Use this prompt with Codex in VS Code after placing a new export in
`data/inbox/shirabe/`:

> Inspect the newest Shirabe CSV export. Preserve the raw file and every unknown
> source column. Import it into the normalized vocabulary database, report added,
> updated, duplicate, and ambiguous rows, and do not overwrite existing curated
> furigana, examples, conjugations, or usage notes. Add or update tests if the
> export schema differs from existing fixtures. Run validation and the full test
> suite afterward.
