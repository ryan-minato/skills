# Reading the Evidence

Read when the detection sweep supports more than one model, contradicts the
model the user names, or cannot separate a live convention from an abandoned
one.

## What each signal proves

| Signal | Supports | What falsifies it |
|---|---|---|
| A branch named `develop`, `dev`, or `integration` with recent commits | git flow, or a local two-integration-branch variant | no commits in months, or every merge into it comes from one automation account |
| `release/*`, `*-stable`, or `v2.x` branches that still receive commits after a later version shipped | GitLab Flow release branches, or git flow | the branches are frozen at their release tag and only exist as archives |
| Branches named after deployment targets (`staging`, `pre-prod`, `uat`, `production`) | GitLab Flow environment branches | the names appear only in CI job names, not as refs |
| A `production` branch that only ever receives merges from the default branch | GitLab Flow production branch | it also receives direct commits, which makes it an environment chain or an ad-hoc branch |
| Only short-lived topic branches, all merged into the default branch | GitHub Flow | long-lived branches exist but were filtered out because they are not checked out locally |
| Deploy jobs keyed to a branch name in CI configuration | that branch is load-bearing, whatever its history looks like | the job is disabled, `when: manual` and never run, or gated on a variable that is never set |
| Tags on the default branch only | releases are cut from the default branch | tags also exist on stable branches, which means parallel maintenance |

## Rules for resolving a contradiction

Recency beats existence. A ref's last commit date and the date of the last
merge into it decide whether it is a convention or an artifact. Rank candidate
models by what the repository did in its most recent release cycle, not by what
it did once.

CI configuration is the strongest statement of intent the repository contains.
A branch that triggers a deployment is part of the model even when its commit
history looks quiet; a branch nothing references is a leftover even when it is
busy.

Platform protection settings outrank local heuristics. Read the protected
branch and protected tag lists from the host when access allows: the team
protected exactly the refs it considers load-bearing, and that list is a direct
answer to a question the checkout can only imply.

The merge-commit ratio on the default branch reveals the merge method, not the
model. A history with no merge commits means rebase or squash is already in
force — carry that finding into the merge-method question rather than
proposing a change to it.

## When evidence and the user disagree

State the conflict with its evidence, then ask. Do not silently adopt either
side. The common cause is a model that was adopted, partly abandoned, and never
cleaned up, which means the answer is a decision about the leftovers rather
than a detection failure.

Record the outcome as one of: the model in force, the refs that are artifacts
and may be deleted, and the refs whose status the user could not resolve.

## When there is nothing to detect

An empty repository, or one with a single branch and no tags, has no model to
preserve. Say so explicitly and move to selection. Absence of evidence is a
clean starting point, not an argument for the simplest model — the selection
rules still run against the project's stated release and deployment plans.
