# Publishing templates and the CI pipeline

Read this when publishing templates, wiring CI, or listing the collection
on containers.dev. Distribution spec:
<https://raw.githubusercontent.com/devcontainers/spec/main/docs/specs/devcontainer-templates-distribution.md>

## What publishing produces

Each template becomes a tarball `devcontainer-template-<id>.tgz` pushed
as an OCI artifact to `<registry>/<namespace>/<id>` with semver tags
(`:1`, `:1.2`, `:1.2.3`, `:latest`); the namespace also receives an
auto-generated `devcontainer-collection.json` (tag `latest`). Publishing
is keyed on `version` in the manifest — an existing version is skipped.

## Manual publish and consumption

```bash
npx -y @devcontainers/cli@0.87.0 templates publish \
  -r ghcr.io -n <owner>/<repo> ./src
```

Consumers apply with:

```bash
npx -y @devcontainers/cli@0.87.0 templates apply \
  -t ghcr.io/<owner>/<repo>/<id> -a '{"<option>": "<value>"}'
```

or through "Add Dev Container Configuration Files" in VS Code once the
collection is indexed. GHCR packages default to **private** — flip each
package (and the collection package) to public after first publish, or
consumers get auth errors that look like a missing template.

## The scaffold's pipeline

- `test.yaml` (push/PR): `detect-changed-templates` builds path filters
  from `src/*/`, then the shared `smoke-test` action runs per changed
  template: substitute default option values into a temp copy (exactly
  what `templates apply` does), `devcontainer up`, then run
  `test-project/test.sh` inside via `devcontainer exec`. No
  `continue-on-error` — a failing template fails the check.
- `release.yaml` (manual, main only): job 1 publishes via
  `devcontainers/action@v1` (`publish-templates: "true"`,
  `base-path-to-templates: ./src`; only `packages: write`); job 2
  regenerates `src/<id>/README.md` (`generate-docs: "true"`, appending
  NOTES.md) and opens a docs PR (`contents: write` +
  `pull-requests: write`).

Adaptation points: registry/namespace (non-GHCR needs a login step and
CLI publish), and action pinning — `@v1`/`@v4`/`@v3` are moving tags; pin
to commit SHAs when policy requires immutable actions.

## Release procedure

1. Bump `version` in the changed template's manifest within the same PR.
2. Merge to main with a green smoke test.
3. Run the `release` workflow; verify the new tags exist; merge the
   automated docs PR.

## Optional: containers.dev discoverability

To appear on containers.dev/templates and in VS Code's picker, PR your
collection into `collection-index.yml` at
<https://github.com/devcontainers/devcontainers.github.io>. `templates
apply` by explicit OCI ref works without registration.
