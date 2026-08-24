# CI for Multi-Package Repositories

Read when the repository hosts more than one independently checked
package or app.

## Principle

Each package keeps its own check commands, and CI runs a package's
checks only when that package — or something it depends on — changed.
One monolithic pipeline that rebuilds everything on every merge request
burns the runtime budget and hides which package actually broke.

## Procedure

1. Map the packages: for each, its directory, its check commands, and
   which other packages' changes must also trigger it.
2. Fetch the current mechanisms from <https://docs.gitlab.com/ci/> — the
   docs name what exists today for scoping work to changed paths
   (`rules:changes`, `include:`, parent-child pipelines, the CI/CD
   Catalog's components at <https://gitlab.com/explore/catalog>). Choose
   after reading, not from memory.
3. One job per package, named after it, running that package's local
   commands verbatim from its directory.
4. Path-scoped jobs interact badly with blocking-merge settings: a job
   skipped by its rule may leave the merge gate unsatisfiable, or
   silently absent. Verify the currently documented interaction before
   making any path-scoped job blocking.
5. Record the package-to-job map in the AGENTS.md deposit so a failure
   points straight to a directory and a command.
