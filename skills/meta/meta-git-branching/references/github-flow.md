# GitHub Flow

Read when GitHub Flow is the selected model. It is the default for projects
with one deployed version and no fixed chain of deployment targets.

## The contract

- The default branch is always deployable. Nothing else in the model works
  without this.
- Every change gets its own short-lived branch off the current default branch.
- The change is reviewed as a pull or merge request against the default branch.
- Merging is the release trigger: what lands is what ships, or what the next
  scheduled deploy ships.
- The branch is deleted on merge.

There are no other long-lived branches. Resist adding one; each addition
converts the project to a GitLab Flow variant, and that conversion should be a
decision rather than a drift.

## What the model demands in return

Checks must be fast and trustworthy enough that the team merges on them. A
suite that takes an hour, or that fails randomly, pushes work off the default
branch and the model collapses into ad-hoc long-lived branches.

Incomplete work ships dark. Merging a half-finished change to a deployable
branch requires a feature flag, an unreferenced code path, or a scope small
enough to finish in one branch. Decide which of the three the project uses and
record it, because this is the rule teams most often discover only after
breaking production.

Recovery is roll-forward. A bad deploy is fixed by another merge, not by a
parallel branch. Where a project cannot roll forward quickly, the production
branch variant in the GitLab Flow reference is the honest choice instead.

## Naming and lifetime

Short-lived branch names carry the change, not the person: `add-retry-backoff`,
`fix-token-refresh`. Where the host creates the branch from an issue, take the
name it generates and stop discussing the format — a generated name that
everyone gets for free beats a documented one that half the team types by hand.

Set deletion on merge at the repository level so the branch list stays a list of
work in progress.

## Releases

Releases are tags on the default branch. Choose the version scheme in the
convention round and tag from the default branch only; a tag on any other ref
means a second maintained line exists and the model has already changed.

## Hotfixes

A hotfix is an ordinary change on an ordinary branch. It is urgent, not
structurally different. Do not add a hotfix branch class: the moment a hotfix
needs a different path, the project has more than one deployed version and
belongs in a GitLab Flow variant.

## Failure signs

Watch for these after the model is in place; each means the contract is being
worked around rather than followed.

- Branches living longer than a few days, accumulating merges from the default
  branch.
- A `develop`, `staging`, or `next` branch appearing without a decision.
- Releases cut from a commit that is not the tip of the default branch.
- The team stops merging because the default branch is "not stable right now".
