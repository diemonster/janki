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
