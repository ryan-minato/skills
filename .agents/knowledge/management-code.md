# Management Code

Read this before writing or changing a script under `scripts/` or
`.agents/skills/change-workflow/scripts/`, a workflow step or composite
action under `.github/`, a `justfile` recipe, a pre-commit hook, or a tool
version. The Python that public skills ship follows
`.agents/knowledge/skill-quality.md` `## Script assets` instead.

## Scope

Management code never ships in the product. Here it is `scripts/` (with
the shared `scripts/lib/`), `.agents/skills/change-workflow/scripts/`, the
`run:` steps of `.github/workflows/*.yml` and `.github/actions/*/action.yml`,
the `justfile` recipes, and the pre-commit hooks.

- Scripts are TypeScript run by Deno. Shell stays in recipes and workflow
  steps, for simple operations; logic longer than a few lines goes into a
  script.
- Each script is written for its role here and owned here. Nothing keeps
  it identical to a skill's bundled script or script asset;
  `scripts/spec_changes.ts` is this repository's own port and binds the
  `openspec-workflow` skill's asset to nothing.

## Script shape

- `scripts/<name>.ts`, executable, first line
  `#!/usr/bin/env -S deno run --no-prompt <permissions>`. Callers run
  `./scripts/<name>.ts` from the repository root.
- A header comment states what the script does, its modes, and who calls it.
- All input comes through flags; nothing prompts. `--help` (`-h`) prints
  the usage to stdout and exits 0.
- Exit 0 on success or no findings, 1 on findings, 2 on a usage error or a
  failed interface the script depends on.
- Data goes to stdout (JSON where a later step consumes it); diagnostics go
  to stderr. A message says what is wrong, why it matters, and how to fix it.
- Idempotent by default: agents and workflows retry.
- Export the pure functions and `main(argv)`; the entry is guarded by
  `if (import.meta.main) Deno.exit(await main(Deno.args))`.
- Shared helpers live in `scripts/lib/` (`repo.ts`: the root, `run`,
  `runChecked`, `CommandError`, `UsageError`; `args.ts`: `parseArgs`).
  Add to it only what two scripts share. It imports nothing remote.
- Remote modules come only through the `deno.json` import map, `jsr:@std/*`
  by `^1` range; the standard library first, a mature library before a
  reimplementation.
- `just lint` runs `deno lint`, `deno fmt --check` (120 columns, 4 spaces),
  and `deno check`; the `deno.json` include globs name only the management
  code, never `skills/`, `openspec/`, or a generated skill.

## Permissions

The shebang is the script's permission statement; grant only what it uses.

- `--allow-read=<paths>`, scoped to repository-relative paths. Unscoped
  `--allow-read` only for a script that reads a caller-named file such as
  an event payload.
- `--allow-run=<command>,...` naming each command. Treat a grant as
  unconfined for that command: `gh` can do whatever the login can.
- `--allow-env=<VAR>,...`, `--allow-write=<paths>`, each named.
- `--allow-net=<host>` only for a script that calls an API directly.
- `--no-prompt` always, so a missing permission fails instead of waiting.
- Never `-A` or `--allow-all`, in a shebang, recipe, hook, or step.

`just test` runs the tests with broad grants; a script's own grants are
proved by running it through its shebang in its real invocation.

## Failure handling

Readability first, and an unexpected failure is left to fail.

- At every interface — an external command, `gh` output, an API response,
  a file another step wrote, YAML or JSON input — check what the code
  relies on (exit status, parseable output, shape, documented fields) and
  fail with a message naming the interface, the value received, and the
  fix, before anything is built on it. `runChecked` does this for
  commands; a `CommandError` exits 2.
- No fallback hides an unexpected failure: no default standing in for a
  failed read, no `?? []` over a field the interface must provide, no
  `catch` that swallows, catches everything, or only restates the error.
  A `catch` names the one error it expects, as `exists()` in
  `scripts/lib/repo.ts` does with `Deno.errors.NotFound`, and rethrows
  the rest.
- A value the data contract documents as possibly null is domain logic.
  Structural safety, idempotence, and a narrow retry on a known-transient
  error stay.
- A check collects every finding and exits 1 once; an unexpected failure
  exits 2 and is never reported as a finding or as success.
- Shell hides no failure: no failed command read as a false condition, no
  `|| true` wider than the exit it was written for, no pipe that drops a
  status nobody meant to drop. Choose the shell options each step needs.

Deferring a failure by design is outside these rules — a report step that
runs after a failure, a gate that decides on collected results, a reply
posted before the run fails. The deferral is written explicitly where it
happens, and the failure still reaches where the design sends it.

## Tests

- `<name>_test.ts` beside every script; `just test` runs them and
  `just check` runs `just test`.
- Cover the decision logic, every finding and exit path, and argument
  errors.
- Commands that reach GitHub go through an injectable runner; tests stub
  `gh`, including its failures, and never touch the network or the live
  repository.
- Fixture repositories and files live under `Deno.makeTempDirSync()` and
  are removed afterwards.
- A new rule in a check lands with a fixture that breaks it.

## Workflow security

The `workflows` check of `scripts/validate_harness.ts` enforces these over
`.github/workflows/*.yml` and `.github/actions/*/action.yml`, naming file
and line:

- Every workflow and every job declares `permissions`: `permissions: {}`
  at the top, the grants on the job.
- Every external action is pinned to a full commit SHA with a version
  comment (`# v1.2.3`).
- Every `actions/checkout` sets `persist-credentials: false`.
- Every job sets `timeout-minutes`.
- No `${{ github.event.* }}` or `${{ github.head_ref }}` expression in a
  `run:` line or a `script:` input; pass the value through `env:`.
- The privileged triggers `pull_request_target`, `issue_comment`, and
  `issues` appear only in the workflows `.agents/knowledge/github-checks.md`
  `## Privileged workflows` lists. Their jobs never check out, fetch, or
  run pull request content, and use no action but the checkout and
  `./.github/actions/setup-tools` with no tool inputs.
- No script a privileged job runs imports a remote module.
- No workflow uses `workflow_run`.
- No `pull_request` workflow grants a write permission.

Held by review, not by the check:

- Every script a job holding a write token runs imports only local
  modules: `spec_changes.ts`, `spec_approval.ts`, `sync_labels.ts`,
  `sync_issue_metadata.ts`, and `check_pr_policy.ts`. With no lockfile, a
  remote module would load unpinned beside the token.
- The token reaches only the steps that call the API (`GH_TOKEN` in those
  steps' `env:`, never the job's).
- `./.github/actions/setup-tools` installs the latest Deno, and just, the
  Python tools, or OpenTofu only when a job asks through an input; a job
  holding a write permission asks for nothing beyond Deno.
- The exact bounds of `spec-labels.yml`, the one label-writing workflow,
  are asserted by a test beside `scripts/spec_approval.ts`.

## Tool versions

- Deno, just, pre-commit, ruff, and OpenTofu are never pinned: the dev
  container features and `./.github/actions/setup-tools` install the
  latest release, OpenTofu through `just install-opentofu`, checked
  against that release's SHA256SUMS; the `tools` check fails an OpenTofu
  setup action or version pin in `.github/`, `.devcontainer/`, the
  `justfile`, `.pre-commit-config.yaml`, or `deno.json`. A release that
  turns a check red is the early signal; the fix is its own change, and
  one that touches public skills is a change of those skills.
- OpenSpec is the one pin, `openspec_version` in the `justfile`, because
  the generated `openspec-*` skills must equal that CLI's output byte for
  byte. `just install-tools` installs it through Deno into `.tools/bin`. A
  bump changes the pin and commits the output of `just spec-sync` with it.
  The `tools` check fails any other file that names an OpenSpec version,
  except the `generatedBy` stamps, which must equal the pin.
- Actions are pinned by commit SHA with a version comment; Dependabot
  updates them, `.github/actions/*` included (`.github/dependabot.yml`),
  and the repository settings require full-length SHAs and an action
  allowlist (`.agents/knowledge/github-settings.md`).
- Pre-commit hook revisions are pinned because pre-commit requires a
  `rev`. The ruff hooks are `repo: local` and call the installed ruff;
  Dependabot's `pre-commit` ecosystem updates the other revisions.
- No lockfile is committed: `deno.lock` (`"lock": false`; the `tools`
  check fails a tracked one), the dev container feature lockfile, and the
  skills lockfile are ignored.

## Update this file when

- A script, recipe, or hook departs from the shape, permission, or test
  rules here.
- The `workflows` or `tools` check gains or loses a rule.
- A tool gains or loses a pin, or a lockfile is committed.
- A workflow gains a write token, or a privileged workflow is added.
