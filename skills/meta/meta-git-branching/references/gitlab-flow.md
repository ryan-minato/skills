# GitLab Flow

Read when a GitLab Flow variant is the selected model. The model is a family:
GitHub Flow's short-lived branches plus exactly one of the three structures
below. Pick one. Combining the environment chain with release branches is
possible but doubles the refs that must stay honest, so adopt the second only
when the project genuinely maintains parallel versions *and* stages deployments.

The variants are not tied to GitLab as a host; they work on any platform with
merge requests. GitLab's current documentation describes the same three
structures under different headings — a single web service, branch per
environment, and long-lived release branches — so the terms are
interchangeable when reading its material.

## Variant 1 — production branch

Use when the default branch is deployable but the team does not deploy on every
merge: a release window, a manual approval, or an app-store review sits between
merge and production.

- The default branch stays the integration branch and receives every change.
- A `production` branch reflects what is deployed.
- Deploying means merging the default branch into `production`. Nothing is
  committed to `production` directly.
- The merge commit, or a tag on it, is the record of when each change went live.

This is the smallest step away from GitHub Flow. Choose it before an
environment chain whenever there is only one gate.

## Variant 2 — environment branches

Use when a change reaches production by traversing a fixed chain of deployment
targets and each hop is a decision someone makes: default branch → `staging` →
`pre-prod` → `production`, with as many intermediate branches as there are
environments.

The invariant is direction: **commits only flow downstream**. Each hop is a
merge request from the upstream branch into the next one, which is what makes
"everything in production was tested in every earlier environment" true rather
than aspirational.

Never merge a downstream branch back upstream. It looks like a harmless sync and
it silently promotes whatever else that branch carries.

Hotfixes are the documented exception, and they take a specific shape: develop
the fix on a feature branch, merge that branch into `production` through a merge
request, then merge the same feature branch into the other branches. Where the
downstream branches need manual testing first, send separate merge requests from
the feature branch to each of them. The fix reaches every branch from one
source; it does not travel up the chain.

Name the branches after the environments they deploy, exactly as the team says
them out loud, and make each name match the deploy job that consumes it.

## Variant 3 — release branches

Use when the project maintains several versions at once: an older release still
receives fixes after a newer one has shipped. This is the variant for software
other people install and pin.

- Cut stable branches from the default branch, and cut them as late as
  possible. Every day a stable branch does not exist is a day nothing has to be
  maintained twice.
- Name them by the minor version — `2-3-stable`, `2-4-stable`, or the project's
  existing shape. The name has to be predictable enough for automation to
  match.
- **Upstream first.** A fix merges into the default branch first, then is
  cherry-picked into each stable branch that needs it. Fixing the stable branch
  first and forgetting to forward-port reintroduces the bug in the next
  release — the whole reason the policy exists.
- Every fix that lands on a stable branch gets a new patch tag. Tags, not
  branch tips, are what users consume.
- After a release is announced, only serious bug fixes go to its branch. Write
  down what the project counts as serious; leaving it to judgment is what turns
  a stable branch back into a development branch.
- Keep the number of simultaneously maintained branches and their support
  window explicit, and put both in the deposited contract. An unstated support
  window never ends.

Cherry-pick messages must name the commit they came from. Two hashes for one
change is the cost of the model; an unlinked pair is the cost plus a mystery.

## Choosing between the variants

One deploy gate, one version → variant 1. Several ordered environments, one
version → variant 2. Several maintained versions → variant 3, regardless of how
deployment works.
