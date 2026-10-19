# Harness Self-Checks

Read when entropy management is rated medium or thick, or when a harness
validator, mirror check, or entrypoint budget check is planned. Defects are
kept out of the trunk by CI, not by asking; harness defects — expired
paths, dead commands, mirrored files that diverged — are the same class and
get the same treatment. Harness drift is among the most expensive defect
classes and, without this, the only one with no hard boundary.

## Two tiers

**Reference integrity** costs almost nothing and belongs in every
long-lived project:

- every path the entrypoint and the knowledge files name exists;
- every command a document names exists as a task-runner recipe or a
  script. This requires one command entry point (`just`, `make`, a package
  manager's scripts) so documents name recipes and arguments live in one
  place — the root fix for command drift;
- every pair of mirrored files is identical, for example a README in two
  languages.

**Structural compliance** costs a little more:

- skill directory layout, frontmatter fields, and description lengths
  follow the convention;
- the specification tool's strict validation runs in CI;
- the entrypoint has a length budget, and exceeding it fails or warns, so
  it does not grow back into an encyclopedia.

Add the second tier when the project has skills or specifications; the
first alone serves a small long-lived repository. Keep the tiers apart in
the plan: a project without skills or specifications gets the
reference-integrity checks only, with the structural tier — the entrypoint
budget included — named as deferred and its trigger stated.

## Rules for the checks

- A failure names the fix: which pair diverged, which path is dead, which
  file to edit. Guessing at the harness is more dangerous than guessing at
  product code.
- The checks are management code, not product code: readable first, no
  defensive fallbacks, and an unexpected condition fails loudly. A check
  that skips a file it cannot parse and exits zero is worse than no check,
  because it manufactures the belief that something was checked.
- The project's checks never run a script inside a skill's directory. A
  skill's script is written for the agent that runs the skill and evolves
  on the skill's schedule; sharing it lets a skill edit silently change CI.
  Where the same logic is needed, each side keeps its own copy, as
  deliberate duplication.
- Run them where the project's other checks run — the task runner's check
  recipe, the pre-commit hook when they are fast, CI always.

## Optional: canary tasks

A project with heavy harness investment may keep a few representative
historical tasks and rerun them after a major harness change, to catch a
rewrite that made agents worse. This never enters CI: CI usually lacks the
credentials to run an agent, and repeated runs are costly. It is a manual
action, triggered on demand, and small projects skip it.
