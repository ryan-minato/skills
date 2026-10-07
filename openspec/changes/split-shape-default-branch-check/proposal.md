## Why

`openspec-workflow` installs a check job whose push branch runs `check --all` without `--shape`. The script therefore defaults to the combined shape, and on a split-shape project `spec / check` (`spec:check` on GitLab) fails every push to the default branch from the moment a specification request merges its record until the implementation request archives it. The script, `references/github.md`, and the commit that introduced `--shape` (4619653) all say a split project's default branch warns instead. The spec's push clause and the two job assets say it fails. Only the split shape is affected, and it cannot be used as installed.

## What Changes

- `openspec-workflow`:
  - **Push check.** The produced check job passes the recorded shape to its push branch as it already does to its request branch: `check --all --shape {{REQUEST_SHAPE}}` in `assets/github/job-spec-check.yml` and `assets/gitlab/ci-spec-jobs.yml`. The job comments state the push rule for each shape.
  - **The rule.** On a push to the default branch, a change outside the archive directory fails under the combined shape. Under the split shape it is a warning naming the change, whether or not a task of it is ticked.
  - **References.** `references/github.md` (`## Ready-state rules`, `## Verification after installing`) and `references/gitlab.md` (the placeholder line, `## Verification after installing`) say that both branches of the job take the shape, and how to check the push branch after installing. `## Ready-state rules` no longer calls `combined` the default the asset ships: the job passes the recorded shape explicitly under either shape.
  - **Help text.** The `--all` help of `assets/spec_changes.py` and of the bundled `scripts/spec_changes.py` names both shapes. Neither script's logic changes: both already warn under `--all --shape split`.
  - **SKILL.md.** `## Ready, then the freeze` no longer says the integration branch never holds an unarchived change without qualification. It says so for the combined shape, and says a split project's integration branch holds approved records until their implementation requests archive them. `## Commands and labels on a request` says combined projects pass `--shape combined`, not that they leave the flag at its default.
  - **Spec.** One ADDED requirement states the rule for each shape, with its scenarios. The installation requirement's push clause and both `Script:` requirements' `--all` clauses, which state the combined rule alone, name the split exception.

## Skills touched

- `sdd/openspec-workflow` (modified): the default-branch check follows the recorded request shape. One requirement is added (`Behavior: The default-branch check follows the recorded request shape`). Three existing requirements change one clause each: the installation requirement and both `Script:` requirements.

## Installed behavior

An agent that installs the automation on a split-shape project produces a check job whose push branch warns about an approved, unarchived record instead of failing on it, and tells the user the default branch holds such records until their implementation requests archive them. Combined-shape installs behave as before; the push branch now names `--shape combined` explicitly. This corrects wrongly restrictive installed behavior → `fix`.

## Impact

- The companion repository change `split-shape-default-branch-check-harness` is required. `.agents/knowledge/harness-maintenance.md` registers `job-spec-check.yml` against the `checks / spec` job of this repository's `checks.yml`, and the asset's command line against this repository's `scripts/spec_changes.py`. So `checks.yml` pins `--shape combined` on its push step, the repository script's `--all` help follows the asset's, and the knowledge files that quote the push command follow.
- No change to the skill's description, symlink, `marketplace.json`, the `sdd` README pair, or `skills/sdd/CONTEXT.md`.
- A project that installed the automation earlier keeps the old push line until it changes it. Installed copies are never updated in place.

## Non-goals

- `spec-kit-workflow`. Its `check --all` checks only that each feature has a specification and a plan, which a split specification request carries, so it has no shape-dependent rule.
- Failing the split push check on an unarchived change that has a ticked task. The request check already refuses such a request; a stricter push rule is a possible follow-up.
- `meta-spec-workflow`'s `references/contract-design.md` (`## Archiving`) and its deposited contract template `assets/spec-workflow.md` (`## Archive executor`), which keep the unqualified "never holds an unarchived record" wording. Proposed as one follow-up issue.
- The logic of either script, and the pull-request side of the check.

## Tracked work

Issue #99.
