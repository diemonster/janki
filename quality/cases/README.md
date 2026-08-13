# Hardening cases

Each child directory is one offline hardening case. `case.yaml` selects one
registered runner. It cannot contain a shell command or a Python import path.

The manifest declares every file in the directory and pins its SHA-256. The
loader rejects extra files, missing files, path traversal, symlinks, and hash
changes. A fixture with `fixture_basis: agent-created-synthetic` contains only
data that an agent created for the test. Owner-provided and source-derived
bundled files need the exact repository-owner redistribution approval that
`docs/HARDENING.md` defines. The source `basis` states the exact license or
permission. For a synthetic source, it is `agent-created-synthetic`.

`private_inbox_ref` keeps source bytes under `data/inbox/`. Offline replay can
hash that file, but it does not decode it or send it. A separate exact owner
approval is necessary before a later live evaluation can send it to the named
provider and model scope.

Run all gating cases with:

```console
janki harden replay
```

Name case IDs to run a specific case, including a non-gating case for an open
finding.

## Manifest fields

`case.yaml` has these common fields:

- `id` must match the directory name.
- `purpose` is `regression` or `coverage`.
- `gating` is true only when the case must pass in the default replay.
- `runner` selects a name from the registry below.
- `pipeline_boundary` must match that runner.
- `source_archetype` selects one M7 source-archetype value.
- `fixtures` names and hashes every file beside `case.yaml`.
- `runner_input` and `oracle` each name one declared JSON fixture.
- `fixture_basis` is `agent-created-synthetic`, `owner-provided`, or
  `source-derived`.
- `redistributable` states whether a source artifact can be committed.

A regression case has one `finding_id`. The finding must link back to the case.
A coverage case has one `pilot_id` and one approved `unit_oracle_id`. The pilot
and oracle must link back to the case.

## Registered runners

The runner names and production boundaries are:

- `input-provenance` → durable inbox selection and copy behavior
- `candidate-response` → extraction and normalization
- `staging-promote` → staging round-trip and promotion
- `validation-qc` → record validation and deterministic QC repair
- `render-build` → finished-record rendering and package build
- `dictionary-enrichment` → canned jpdb data and KANJIDIC evidence
- `ai-enrichment` → canned structured AI data, sentence parse evidence, and
  command no-change reporting

Runner input is strict JSON. Unknown keys, duplicate keys, non-finite numbers,
unused canned responses, and malformed records are errors. An oracle stores
only stable facts such as IDs, diagnostic codes, field changes, and selected
rendered fields.

## Source forms

A case that reaches a model boundary has one `source` mapping.

`bundled_fixture` names one declared file in the case. It requires
`redistributable: true`. A synthetic file uses the basis
`agent-created-synthetic`. An owner-provided or source-derived file states its
license or permission in `basis`. Its `redistribution_approval` repeats that
basis, the case ID, and the exact artifact fingerprint.

`private_inbox_ref` names one file under `data/inbox/` and requires
`redistributable: false`. Its `live_eval` mapping states a provider, model
scope, and purpose `hardening-eval`. An absent nested `approval` means that
consent is false. An approval must repeat the case, source, provider, model
scope, and purpose. It also records the owner's date and reason.
