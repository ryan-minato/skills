# GitHub harness CI

Read when a GitHub-hosted project needs a CI workflow that turns its existing
local quality commands into pull-request gates. CI mirrors the local feedback
loop; it does not choose the project's tests or linters.

1. Inventory the task runner, setup command, fast checks, runtime versions,
   lockfiles, generated artifacts, existing workflows, permissions, and expected
   runtime.
2. Run the exact local commands first. Do not automate a red or undocumented
   command.
3. Start from
   [the workflow shape](assets/workflow-harness-checks.yml), then replace
   every placeholder and remove unused jobs.
4. Use first-party actions by default. Any third-party action requires explicit
   opt-in and a full commit-SHA pin.
5. Match runtime matrices to supported project versions; do not invent a broad
   matrix. Cache only immutable dependency inputs.
6. Read [ci-monorepo-paths.md](ci-monorepo-paths.md) when path filters or
   several independent packages are involved. Read
   [ci-slow-checks.md](ci-slow-checks.md) when a check exceeds the normal PR
   feedback budget.
7. Add the selected checks and required project-side activation to the durable
   harness, using [the AGENTS.md section](assets/agents-md-ci-section.md)
   only as a reworked starting shape.

Done when: the workflow runs the same passing commands as local development,
has least-privilege permissions, and the project records which checks humans
must require before merge.
