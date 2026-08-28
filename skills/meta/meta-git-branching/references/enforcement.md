# Platform Enforcement

Read before configuring protection rules or merge methods on GitHub or GitLab.
The goal is to move as much of the agreed contract as possible from convention
into enforcement, and to record honestly which rules did not make it.

## Resolve the mechanism before configuring it

Protection features are the fastest-moving part of both platforms, and what is
available depends on plan, tier, instance version, and permissions. Resolve the
current mechanism and its exact option names from the platform's own
documentation or CLI help at the time of the build. Never assume a feature
exists because it existed on another project, and never read a permission error
as proof that a feature is absent.

Where the catalog's platform builder for the host is present —
`meta-github-workflow` or `meta-gitlab-workflow` — it owns the mechanics of
protection settings and the settings register. Supply it with the branch
contract and let it configure; do not build a second, competing register.

## Establish the host before assigning tiers

A checkout with no configured remote is not proof that the project is unhosted.
The entrypoint, the CI configuration, the contribution documents, and the
project's own knowledge files routinely name a host the local clone does not
have. Reconcile them before deciding anything: where the checkout and the
project's documents disagree, ask rather than trusting `git remote`.

Getting this wrong is expensive in one direction only. Recording every rule as
convention because the remote list was empty parks the whole contract at the
weakest tier and attaches upgrade triggers whose condition has already been
met, so nothing ever fires them.

## What to enforce

For every long-lived branch the model defines:

- Block force-push and deletion. This is the rule that protects the model
  itself, and it is available everywhere.
- Require the merge request path: no direct pushes, at least on the default
  branch and on any branch a deploy job consumes.
- Require the checks that make the branch's promise true. GitHub Flow's
  deployable default branch and an environment chain's "tested upstream" claim
  are both enforced here or nowhere.
- Require linear history where the contract says the branch is linear;
  otherwise a single merge-commit merge silently ends the property.

For tags, protect the release tag pattern so a published version cannot be
moved. A tag that can be re-pointed is not a release.

## Merge method

Set the repository or project to allow exactly the methods the contract uses,
and disable the rest. A method that cannot be selected cannot be selected by
accident, and this is the highest-value setting on the page.

The recommended split — rebase merge for branches inside the main repository,
squash merge for outside contributions — cannot be enforced by contributor
origin on either platform: the setting is repository-wide, so both methods stay
enabled and the choice sits with whoever merges. Record the split as a
convention with a named owner, and put it in the contribution instructions
reviewers actually read. Where only one method may be enabled, keep squash and
document that internal branches squash too.

Read the platform's own naming before configuring. Fast-forward-style merges,
semi-linear history, and rebase merges are three different settings with
overlapping descriptions, and picking the wrong one produces a history the
contract did not describe.

## Record the tier, not just the intent

Every rule in the deposited contract carries one of three tiers:

- `enforced` — the platform rejects the violation. Note the exact readback
  command or settings path.
- `advisory` — a check reports the violation but the action still succeeds.
- `convention` — nothing but review catches it.

For every rule that is not `enforced`, pre-compute the upgrade trigger: the
concrete event that should turn it into a hard rule, such as the repository
becoming public, the plan changing, or the first violation reaching production.
An advisory rule with no upgrade trigger becomes permanent by default.

## Before any remote write

Confirm the user's approval covers this specific change, apply it
non-interactively, then read the setting back and compare it with what was
approved. Protection changes are silent when they fail partially — a ruleset
that applied to no branches because its pattern matched nothing looks exactly
like success.
