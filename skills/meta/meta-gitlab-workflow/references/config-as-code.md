# Platform Settings as Code

Read when Terraform or OpenTofu is installed, when the user wants the
project's settings described as code, or before writing
`.gitlab/policy.tf` or any other tf file for the target. The
configuration-repository case adds [config-repo.md](config-repo.md).

## When tf is delivered

Probe with `command -v tofu terraform`. When neither is on the path,
deliver no tf: record the settings in `platform-settings.md` as before, and
record "Platform settings as tf: no" in its harness options. When one is
installed, offer `.gitlab/policy.tf` and record the choice. One file serves
both tools; the commands below say `tofu`, and a Terraform project runs
`terraform` instead.

`platform-settings.md` stays either way: rework `assets/platform-settings.md`
into the target's knowledge on every build, tf or not. Its "Described by" column names
the rows a tf file declares, and the rows no provider covers keep their
readback as the only check. Resolve resources, their arguments, and their
import ID formats from the `gitlabhq/gitlab` provider's registry
documentation at use; the provider tracks GitLab's API, and a list written
here would go stale.

## What policy.tf declares

Start from `assets/policy.tf` and keep only what the harness or a
requirement the user stated depends on: typically the description, the
visibility, issues and merge requests, the merge method the branch contract
settled, the pipeline and discussion merge checks, the default branch's
protection, and the approval rules the group leaves unset. Every
declaration carries a comment saying what it sets and why. Never declare a setting only to keep its current
value: that turns the file into an inventory nobody can justify line by
line. Leave it out and name it in the reset note instead.

Anything left out may take any value, and an apply resets it to the
provider's value. That is the accepted contract: declare every setting the
harness depends on, and tell the user that undeclared ones may be reset.
The first plan against a real project shows which undeclared attributes
would change; read it with the user before the first apply, and record any
attribute that differs on every plan (a perpetual diff) with its handling
and date in the target's knowledge. This builder has not yet run that plan
against a live GitLab project.

Pin the provider with a version constraint and commit the lock file
(`.gitlab/.terraform.lock.hcl`, written by `tofu init`); CI initializes with
`-lockfile=readonly`.

## No state

- Declare no `backend` block — not even GitLab-managed Terraform state,
  though it is free and built in. Every run starts from an empty state and
  learns the current values through `import` blocks: one per resource,
  using the import ID format in that resource's documentation. Import blocks
  need Terraform 1.5 or OpenTofu 1.6; leave `for_each` off them so both
  floors hold.
- When the user asks for GitLab-managed state, keep the configuration
  stateless and say why. Every run imports the current settings, so state
  would add nothing but a second source of truth. Removing a declaration
  only stops managing the object, while with state it would delete it.
- A resource that only has a platform-assigned ID, such as an approval
  rule, is created once without an import block. The plan guard lists the
  create, and the ID goes into an import block in the next change.

## Removal is not deletion

Removing a declaration and its import block stops managing the object;
with no state, nothing is destroyed. Two things could still delete:

- **Resources that own a whole set** and remove what their list omits.
  Read each resource's documentation before using it, and never use one.
  Project labels stay in the label file.
- **Replacement**, when a changed argument forces the object to be
  recreated.

Deliver `assets/tf_plan_guard.py` as `scripts/tf_plan_guard.py`. It reads
`tofu show -json` of a saved plan and exits 1 on any delete or replace. It
also lists every create that imports nothing and every update to an
undeclared attribute, so the reviewer sees each reset. Tell the user the
target owns the delivered copy.

## Who applies

A maintainer or owner plans and applies `policy.tf` locally with their own
login. A group access token lives only in the group's configuration
repository; never offer one in any project's variables, the project's own
included. The project's pipeline gets `assets/gitlab-ci-policy-check.yml`
alone: format and validation against the committed lock, reading no
variable that holds a token. Always deliver it with `policy.tf`. When the
image's digest cannot be resolved where you work, pin the image by tag and
record the digest pin as an open item in `platform-settings.md` — never
drop the job. Never put a group access token where a
project's pipeline can read it, and never in a group-level CI/CD variable:
every project in the group inherits those. When the user asks for that,
decline and offer the local recipes or the group's configuration
repository.

An apply is a remote write. It runs only with the user's explicit
approval, after the plan and the guard output have been shown to them.

### Local credentials

Add recipes to the target's existing task runner — a `justfile`, a
`Makefile`, whatever the project already uses — that hand glab's stored
login to tf for one run. The maintainer logs in once with
`glab auth login`, using OAuth (`--web`) or a personal token; the provider
accepts both as `GITLAB_TOKEN`. Read the token at each run, so the token
glab currently holds is the one used.

Facts about glab 1.122.0, checked 2026-10-14:

- `glab auth` has no `token` subcommand.
- `glab config get token --host HOST` prints the token when glab stored it
  in its configuration file. It prints nothing when glab used the operating
  system keyring, which glab prefers whenever one is available.
- `glab auth status --hostname HOST --show-token` prints a line
  `Token found in …: <token>`, and refuses `--output json`. Its exact
  wording for keyring storage was not checked.

The recipe tries the configuration file, falls back to the status line,
and fails with a message naming glab when neither yields a token. If tf
then reports the token as expired, run any glab API command, such as
`glab api user`, so glab can refresh its OAuth token, and run the recipe
again.

```just
gitlab_host := "gitlab.example.com"

# Print glab's token for the host; the recipes below capture it, nothing shows it
[private]
glab-token:
    #!/usr/bin/env bash
    set -euo pipefail
    t="$(glab config get token --host {{gitlab_host}})"
    if [ -z "$t" ]; then
        auth_out="$(glab auth status --hostname {{gitlab_host}} --show-token 2>&1)" \
            || { echo "glab is not logged in to {{gitlab_host}}; run glab auth login" >&2; exit 1; }
        t="$(printf '%s\n' "$auth_out" | sed -n '/Token found in/{s/.*Token found in [^:]*: //p;q;}')"
    fi
    [ -n "$t" ] || { echo "glab holds no token for {{gitlab_host}}" >&2; exit 1; }
    printf '%s' "$t"

# Plan .gitlab/policy.tf with the glab login; the guard lists what would change
policy-plan:
    t="$(just glab-token)" && cd .gitlab && GITLAB_TOKEN="$t" GITLAB_BASE_URL="https://{{gitlab_host}}/api/v4/" tofu plan -input=false -out=plan.out
    cd .gitlab && tofu show -json plan.out | python3 ../scripts/tf_plan_guard.py -

# Apply exactly the plan the user reviewed from policy-plan
policy-apply:
    test -f .gitlab/plan.out || { echo "policy-apply: run just policy-plan and review it first" >&2; exit 1; }
    t="$(just glab-token)" && cd .gitlab && GITLAB_TOKEN="$t" GITLAB_BASE_URL="https://{{gitlab_host}}/api/v4/" tofu apply -input=false plan.out
```

`just` echoes `t="$(just glab-token)"` literally, and the shebang helper is
not echoed at all. In a `Makefile`, join the token lines with `; \` into the
same recipe line as the tf command, and write each shell `$` as `$$`:

```make
GITLAB_HOST := gitlab.example.com
GLAB_TOKEN_READ = t="$$(glab config get token --host $(GITLAB_HOST))"; \
	[ -n "$$t" ] || t="$$(glab auth status --hostname $(GITLAB_HOST) --show-token 2>&1 | sed -n '/Token found in/{s/.*Token found in [^:]*: //p;q;}')"; \
	[ -n "$$t" ] || { echo "glab holds no token for $(GITLAB_HOST); run glab auth login" >&2; exit 1; }
BASE_URL = https://$(GITLAB_HOST)/api/v4/

policy-plan:
	@$(GLAB_TOKEN_READ); cd .gitlab && GITLAB_TOKEN="$$t" GITLAB_BASE_URL="$(BASE_URL)" tofu plan -input=false -out=plan.out
	cd .gitlab && tofu show -json plan.out | python3 ../scripts/tf_plan_guard.py -

policy-apply:
	@test -f .gitlab/plan.out || { echo "policy-apply: run make policy-plan and review it first" >&2; exit 1; }
	@$(GLAB_TOKEN_READ); cd .gitlab && GITLAB_TOKEN="$$t" GITLAB_BASE_URL="$(BASE_URL)" tofu apply -input=false plan.out
```

- Never read the token with Make's `$(shell …)`. Make expands it before
  echoing the recipe line, which prints the token.
- Never write the token to a file, `.env` included, and never echo the
  status output, which contains it.
- Ignore `plan.out` and `plan.json` in `.gitignore`.
- On a self-managed instance, `GITLAB_BASE_URL` is required and ends with
  a slash.

## Review is the check

Write this rule into the target's knowledge file itself — a comment in
the tf file is not enough:

> Review is the check on `policy.tf`. A change whose declarations do not
> say what they set and why is rejected; the plan, with the guard's
> listings, is read with it.

No static check beyond format and validation stands in for that review.

## Ownership

Every setting has exactly one owner:

- **The group's configuration repository** owns the security baseline and
  the team workflow, through group-level settings such as merge request
  approval settings, push rules, and group labels. It never declares a
  single project.
- **`policy.tf`** owns the project's description, its visibility, and the
  approval and merge policy the group does not set. It declares only what
  it adds, such as one extra approval rule, and nothing the group already
  sets.

Where a group-owned setting can only be set on the project resource, list
that attribute under `ignore_changes` with a comment naming the
configuration repository as its owner. `ignore_changes` serves only that
case and an attribute whose reset cannot be undone, each with its reason.
