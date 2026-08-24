# CI for Multi-Package Repositories

Read when the repository hosts more than one independently checked
package or app.

## Principle

Each package keeps its own check commands, and CI runs a package's
checks only when that package — or something it depends on — changed.
One monolithic job that rebuilds everything on every pull request burns
the runtime budget and hides which package actually broke.

## Procedure

1. Map the packages: for each, its directory, its check commands, and
   which other packages' changes must also trigger it.
2. Fetch the current mechanisms from
   <https://docs.github.com/en/actions> — the docs name what exists
   today for scoping work to changed paths (trigger path filters,
   conditional jobs, matrix strategies, reusable workflows). Choose
   after reading, not from memory.
3. One job per package, named after it, running that package's local
   commands verbatim from its directory.
4. Path-filtered jobs interact badly with required status checks: a job
   skipped by its filter may leave a required check forever pending, or
   silently absent. The docs cover the current supported way to combine
   the two — verify it live before marking any path-filtered job
   required.
5. Record the package-to-job map in the AGENTS.md deposit so a failure
   points straight to a directory and a command.
