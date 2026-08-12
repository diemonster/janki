# Hardening pilot reports

This directory contains reviewed measurements from real or synthetic deck
pilots. One file describes one pilot. Name the file `<id>.yaml`. Use a stable,
lowercase slug for the ID.

Run this command after you edit a finding or a pilot report:

```text
janki harden status
```

Use `--format json` when another tool must read the result. The command is
read-only. It does not rewrite YAML, comments, or approval records. M7.2 checks
link syntax. M7.3 will check that case and oracle targets exist.

## Common rules

- Use schema version `1`.
- Use lowercase slugs for IDs, stages, phases, repair codes, and archetypes.
- Use a lowercase, 64-character SHA-256 value for each fingerprint.
- Use repository-relative locators. Do not use an absolute path, a URL, `~`,
  `..`, or a Windows drive.
- Do not add a field that this document does not define. Unknown fields cause
  an error.
- Do not use a generated package in `dist/` as durable evidence.

## Finding catalog

The file `quality/findings.yaml` contains systemic defects. A
content-specific correction belongs only in a pilot count.

Each finding has these required fields:

```yaml
- id: missing-second-column
  state: open
  pipeline_stage: extraction
  source_archetypes:
    - native-pdf-table
  symptom: The second vocabulary column is absent from staging.
  invariant: Each selected table cell has one recorded disposition.
  evidence:
    - fingerprint: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
      locator: quality/cases/missing-second-column/case.yaml
  recurrences: []
  case_ids: []
```

`recurrences` uses the same `fingerprint` and `locator` fields as `evidence`.
Use it only for later evidence of the same systemic defect.

A finding can have one of these states:

- `open`: Do not add a fix, deferral, risk, or approval field.
- `fixed`: Add at least one `case_ids` value and add `fix_ref`. The fix
  reference is a repository-relative locator, such as a commit hash or file
  locator. M7.3 will check case existence and reverse links.
- `deferred`: Add `deferral_reason`. Do not add fix or risk fields.
- `accepted-risk`: Add `risk`, `reason`, and `approval`. Do not add fix or
  deferral fields.

An accepted-risk approval has this form:

```yaml
approval:
  authority: repository-owner
  finding_id: accepted-layout
  approved_at: 2026-08-12
  risk_content_fingerprint: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
  risk: One rare annotation can remain held.
  reason: No safe deterministic rule exists.
```

Only the repository owner can approve an accepted risk. The approval must
repeat the exact finding ID, risk, and reason. The fingerprint binds the
finding ID, pipeline stage, source archetypes, symptom, invariant, evidence,
recurrences, case IDs, risk, and reason. It does not include the state,
approval block, or date. `risk_content_fingerprint(finding)` in
`japanese_anki.hardening` computes the value. A content change makes an old
approval invalid.

To prepare an approval, first add the `risk` and `reason` without an approval
block. Run `janki harden status`. The error gives the normalized risk-content
fingerprint. Give the finding ID, fingerprint, risk, and reason to the
repository owner. Add the approval block only after the owner approves those
exact values.

## Pilot report

A pilot file has this form:

```yaml
version: 1
id: native-table-pilot
source_fingerprint: cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
source_archetype: native-pdf-table
redistributable: false
unit_oracle_id: native-table-units
eval_case_id: native-table-eval
counts:
  extracted: 20
  omitted: 1
  held: 2
  promoted: 17
  rejected: 0
corrections:
  content_specific: 3
  systemic: 1
repair_applications:
  - code: normalize-romaji
    version: v1
    count: 4
false_positive_repairs:
  - code: normalize-romaji
    version: v1
    count: 1
model_fingerprints:
  - phase: extraction
    provider: anthropic
    model: claude-sonnet-4-5
    fingerprint: dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd
prompt_fingerprints:
  - phase: extraction
    kind: system-prompt
    fingerprint: eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
finding_ids:
  - missing-second-column
pipeline:
  extract: true
  coverage_review: true
  promote: true
  dictionary_enrich: true
  ai_enrich: true
  audio: true
  final_review: false
  build: false
notes: Final review found one unresolved reading.
```

Counts are non-negative integers. A repair entry count must be positive. A
systemic correction or a repair false positive must link at least one finding.
A complete pilot must have at least one model fingerprint and one prompt
fingerprint.

The M7 source-archetype cells are:

- `native-pdf-table`
- `mixed-layout-pdf`
- `scan`
- `camera`
- `vertical-text`
- `dense-annotation`
- `handwritten`
- `printed-ruby`
- `bilingual-layout`

`janki harden status` reports which cells have no pilot. M7.6 can add reviewed
cells if field use shows a distinct source type.
