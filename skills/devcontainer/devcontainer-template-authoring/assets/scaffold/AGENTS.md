# AGENTS.md

Agent entrypoint for this Dev Container Templates repository. Read this
before making changes.

## Purpose

This repository publishes Dev Container Templates — parameterized
`.devcontainer` starting points that users apply **once** into their
projects and then own. Templates are distributed via an OCI registry; the
repository itself never ships to consumers.

## Structure

```
src/<id>/
  devcontainer-template.json  Manifest; `id` MUST equal the directory name
  .devcontainer/              The payload copied into the user's project
  NOTES.md                    Optional; appended to the generated README
  README.md                   AUTO-GENERATED at release — never edit by hand
test/<id>/test.sh             Assertions run inside the applied container
.github/actions/smoke-test/   Shared action: substitute defaults -> up -> test
justfile                      Local mirror of the CI commands
```

Template files may contain `${templateOption:<key>}` placeholders,
replaced with the user's chosen values at apply time (plain text
substitution — no logic).

## Core conventions

- Every option in `devcontainer-template.json` must have a working
  `default`; applying with zero input must yield a working container (CI
  enforces this — the smoke test substitutes defaults only).
- Keep the payload minimal: only files the user should own after apply.
  Reusable install logic belongs in a Feature referenced by full public
  OCI address, not in template files.
- Bump `version` (semver) for every change you want published; releasing
  an unchanged version is a no-op.
- Commit messages: Conventional Commits, scope = template id.

## Commands

| Task | Command |
|---|---|
| Smoke-test one template | `just smoke <id>` |
| Validate manifests | `just validate` |
| Shell lint | `just lint` |

## CI/CD

| Workflow | Trigger | Purpose |
|---|---|---|
| `test.yaml` | push/PR | Smoke-test changed templates (defaults applied) |
| `release.yaml` | manual | Publish to GHCR, then regenerate READMEs via PR |

Both use the shared `.github/actions/smoke-test` action; change test
mechanics in its `smoke.sh`, not in the workflows.

## Adding a template

1. Create `src/<id>/` (manifest + `.devcontainer/` payload) and
   `test/<id>/test.sh` asserting the applied container works.
2. `just smoke <id>` passes locally (needs Docker).
3. Open a PR; CI smoke-tests only your template. After merge, run the
   release workflow to publish.
