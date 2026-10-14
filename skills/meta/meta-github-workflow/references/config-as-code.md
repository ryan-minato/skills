# Platform Settings as Code

Read when Terraform or OpenTofu is installed, when the user wants the
repository's settings described as code, or before writing
`.github/policy.tf` or any other tf file for the target. The
configuration-repository case adds [config-repo.md](config-repo.md).

## When tf is delivered

Probe with `command -v tofu terraform`. When neither is on the path,
deliver no tf: record the settings in `platform-settings.md` with their
readback commands as before, and record "Platform settings as tf: no" in
its harness options. When one is installed, offer `.github/policy.tf` and
record the choice. One file serves both tools; the commands below say
`tofu`, and a Terraform project runs `terraform` instead.

`platform-settings.md` stays either way: rework `assets/platform-settings.md`
into the target's knowledge on every build, tf or not. Its "Described by" column names
the rows a tf file declares, and the rows no provider covers keep their
readback as the only check. Resolve resources and their arguments from the
provider's registry documentation at use: coverage lags the platform
(newer ruleset rules, issue types and fields), and a list written down here
would be stale.

## What policy.tf declares

Start from `assets/policy.tf` and keep only what the
harness or a requirement the user stated depends on: typically the
description, the visibility, Issues when intake and triage run on them, the
merge methods the branch contract settled, and the approval or merge policy
the organization leaves unset. Every declaration carries a comment saying
what it sets and why. Never declare a setting only to keep its current
value: that turns the file into an inventory nobody can justify line by
line. Leave it out and name it in the reset note instead.

Anything left out may take any value, and an apply resets it to the
provider's value. A read-only plan that imported this library's own
repository with only `name` and `has_issues` declared would have
re-enabled merge commits, turned branch deletion on merge off, cleared the
description, and dropped the Discussions setting (provider 6.13.0, OpenTofu 1.13.0,
2026-10-14). That is the accepted contract, not a defect: declare every
setting the harness depends on, and tell the user that undeclared ones may
be reset. With every value declared as it stood, the same plan was a clean
import with no change.

Pin the provider with a version constraint and commit the lock file
(`.github/.terraform.lock.hcl`, written by `tofu init`); CI initializes with
`-lockfile=readonly`.

## No state

- Declare no `backend` block. Every run starts from an empty state and
  learns the platform's current values through `import` blocks: one per
  resource, with the import ID format from that resource's documentation.
  Import blocks need Terraform 1.5 or OpenTofu 1.6; leave `for_each` off
  them so both floors hold.
- A resource that only has a platform-assigned ID (a ruleset) is created
  once without an import block. The plan guard lists the create, and the
  ID goes into an import block in the next change.
- Never commit state to git: the apply and the commit are not atomic, there
  is no lock, a protected default branch rejects the push, and plaintext
  lands in history. GitHub has no state storage, and an external backend
  would add an account and a credential to manage GitHub alone.

## Removal is not deletion

Removing a declaration and its import block stops managing the object;
with no state, nothing is destroyed. Two things could still delete:

- **Authoritative resources**, which own a whole set and remove what their
  list omits — read each resource's documentation before using it.
  `github_issue_labels` and `github_repository_collaborators` are of this
  kind. Never use one; labels stay in the label file.
- **Replacement**, when a changed argument forces the object to be
  recreated.

Deliver `assets/tf_plan_guard.py` as
`scripts/tf_plan_guard.py`. It reads `tofu show -json` of a saved plan and
exits 1 on any delete or replace. It also lists every create that imports
nothing and every update to an undeclared attribute, so the reviewer sees
each reset. Tell the user the target owns the delivered copy.

## Who applies

A repository administrator plans and applies `policy.tf` locally with
their own credentials. Project CI gets
`assets/workflow-policy-check.yml`
alone: format and validation against the committed lock, no secret, and
`contents: read`. Always deliver it with `policy.tf`. It installs OpenTofu
from the official release by checksum, so its only setting is the version
the probe reported, and it needs no third-party action. Never put an organization-level credential, or any token
that can change rules, into a project's workflows. When the user asks for
that, decline and offer the local recipes, or the organization's
configuration repository.

An apply is a remote write. It runs only with the user's explicit
approval, after the plan and the guard output have been shown to them.

### Local credentials

Add recipes to the target's existing task runner — a `justfile`, a
`Makefile`, `package.json` scripts, whatever the project already uses —
that hand the GitHub CLI login to tf for one run. The administrator logs in
once with `gh auth login`, and nothing new is stored.

```just
# Plan .github/policy.tf with the gh login; the guard lists what would change
policy-plan:
    t="$(gh auth token)" && cd .github && GITHUB_TOKEN="$t" tofu plan -input=false -out=plan.out
    cd .github && tofu show -json plan.out | python3 ../scripts/tf_plan_guard.py -

# Apply exactly the plan the user reviewed from policy-plan
policy-apply:
    test -f .github/plan.out || { echo "policy-apply: run just policy-plan and review it first" >&2; exit 1; }
    t="$(gh auth token)" && cd .github && GITHUB_TOKEN="$t" tofu apply -input=false plan.out
```

```make
policy-plan:
	@t="$$(gh auth token)" && cd .github && GITHUB_TOKEN="$$t" tofu plan -input=false -out=plan.out
	cd .github && tofu show -json plan.out | python3 ../scripts/tf_plan_guard.py -

policy-apply:
	test -f .github/plan.out || { echo "policy-apply: run make policy-plan and review it first" >&2; exit 1; }
	@t="$$(gh auth token)" && cd .github && GITHUB_TOKEN="$$t" tofu apply -input=false plan.out
```

- Pass the token explicitly. The provider documents a `gh auth token`
  fallback, but only in its legacy client, and it says the fallback will be
  removed (provider documentation read 2026-10-14).
- Read the token inside the recipe's shell, and stop when the read fails:
  `t="$(gh auth token)" && …` never runs tf without a token. `just` and
  `make` echo a recipe line before running it, and only shell substitution
  keeps the value out of that echo. Make's `$(shell gh auth token)` expands before the echo and
  prints the token. Never write the token to a file, `.env` included.
- Ignore `plan.out` and `plan.json` in `.gitignore`: a saved plan can hold
  values the reviewer never meant to publish.
- Repository settings need repository admin. The organization's
  configuration repository needs `admin:org` on the login: run
  `gh auth refresh -s admin:org` before its first local plan.

## Review is the check

Write this rule into the target's knowledge file itself — a comment in
the tf file is not enough:

> Review is the check on `policy.tf`. A change whose declarations do not
> say what they set and why is rejected; the plan, with the guard's
> listings, is read with it.

No static check beyond format and validation stands in for that review.

## Ownership

Every setting has exactly one owner:

- **The organization's configuration repository** owns the security
  baseline and the team workflow, through organization-level mechanisms
  such as organization rulesets, code security configurations, and Actions
  policies. It never declares a single repository.
- **`policy.tf`** owns the repository's description, its visibility, and
  the approval and merge policy the organization does not set. A repository
  ruleset layers on the organization's rulesets, and the most restrictive
  rule wins, so declare only the addition.

Where an organization-owned security setting can only be set on the
repository resource (`security_and_analysis` on `github_repository`), list
the attribute under `ignore_changes` with a comment naming the
configuration repository as its owner. `ignore_changes` serves only that
case and an attribute whose reset cannot be undone, each with its reason.
