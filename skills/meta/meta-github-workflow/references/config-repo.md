# Organization Configuration Repository

Read when the workspace is, or is to become, the repository that holds an
organization's settings as code. Every rule of
[config-as-code.md](config-as-code.md) applies here too — minimal
declarations with reasons, the plan guard, and review as the check —
except that this repository keeps state by default (`## State` below).
This file adds what is specific to the organization's repository.

## Eligibility

- **Organizations only.** A personal account has no organization-level
  rules to own. When the owner is a personal account, offer each
  repository's `.github/policy.tf` instead and build no configuration
  repository.
- **A separate private repository**, named by the user. Never the
  organization's `.github` repository: it is usually public and serves the
  default community health files, so settings kept there would publish the
  organization's security posture.
- It holds `assets/config-repo.tf` reworked
  into its root, the committed lock file, `scripts/tf_plan_guard.py`, the
  workflow its sync mode needs, and knowledge reachable from `AGENTS.md`,
  including `platform-settings.md` with the sync mode, the state backend,
  and every credential row — in every sync mode, the fully local one
  included.

## What it declares

Only organization-level mechanisms: organization rulesets, code security
configurations, Actions policies, and organization settings — the security
baseline, the team workflow, and every rule the organization can set
uniformly. When several repositories need the same rule, declare one
organization ruleset targeting them, not one repository ruleset each:
every managed resource counts against a state backend that caps or bills
them. It never declares a single repository's own attributes, since each
repository's `.github/policy.tf` owns those, and it never declares
labels, which stay in each repository's label file. Issue types and issue
fields follow [org-configuration.md](org-configuration.md); a provider
resource replaces its script only when the registry documents one.

State the blast radius before every approval: an apply here reaches every
repository in the organization. That approval is separate from any
repository-level one.

## State

The state backend is a harness option. Propose the default for the
installed tool, let the user decide, and record the choice in
`platform-settings.md` under harness options:

| Tool | Default | Alternatives |
|---|---|---|
| Terraform | HCP Terraform, workspace in local execution mode | stateless tf; a third-party backend |
| OpenTofu | a backend the user names: a third-party service that supports OpenTofu, or object storage the organization controls | stateless tf |

- **HCP Terraform.** Declare a `cloud` block naming the HCP Terraform
  organization and the workspace, and set the workspace's execution mode
  to Local. HCP Terraform then stores and locks state only, and plans and
  applies still run in Actions or on an administrator's machine, under the
  App tokens and the approvals below. The Free plan allows 500
  managed resources; data sources and `terraform_data` do not count
  (HashiCorp's documentation, checked 2026-10-15). State that limit, and
  say why the organization prefers one organization-level resource.
- **OpenTofu.** Do not default to HCP Terraform, and tell the user why in
  those words: OpenTofu's documentation promises no compatibility with it,
  and reports disagree (checked 2026-10-15). Ask the user to name a backend, or to choose stateless tf.
  Until they name one — including when they said to take every default —
  build stateless tf, which needs no credential and deletes nothing, and
  record the backend in `platform-settings.md` as an open question. When
  the user still wants HCP Terraform, say so and have them test it on a
  workspace that manages nothing first.
- **With state, removing a declaration deletes the object.** Say so to the
  user when you set up state, and again in the same turn whenever the user
  asks to stop managing something: deleting its declaration alone would
  delete it. Run the guard with `--allow-delete`: it lists every delete and
  replace for the approver and passes. Only the saved plan the approver saw
  is applied. To stop managing an object but keep it, replace its
  declaration with a `removed` block:

  ```hcl
  removed {
    from = github_organization_ruleset.example
    lifecycle {
      destroy = false
    }
  }
  ```

  `removed` blocks need Terraform 1.7: raise `required_version` to
  `>= 1.7.0` when you add one. OpenTofu 1.13.0 plans this block as
  `forget` and the guard passes it (checked 2026-10-15).
- **Keep the `import` blocks.** Give every object that may already exist on
  the platform one: it adopts the object into a new or lost state and does
  nothing once the object is in state.
- **Stateless tf**, the alternative: no backend, and every rule of
  `config-as-code.md` applies, including the guard's default mode and
  `--allow-create` for a first creation.

## Sync mode

The sync mode is a harness option. Propose the default, let the user
decide, and record the choice in `platform-settings.md` under harness
options:

| Mode | CI does | An administrator does | Credential in CI |
|---|---|---|---|
| CI plans and applies | plan on pull requests; on the default branch, save a plan and apply it after the environment's reviewers approve | reviews the saved plan and approves | the plan App's key (repository secret); the apply App's key (apply environment secret); the state backend's tokens split the same way |
| CI plans, the administrator applies | plan on pull requests | applies locally through the recipes in [config-as-code.md](config-as-code.md) | the plan App's key; the state backend's plan token |
| Fully local | nothing that needs a credential (format and validation only) | plans and applies locally | none |

The default depends on whether a second person can be required to approve
the apply:

- **CI plans and applies** is the default only when the private
  configuration repository can have environment required reviewers. GitHub
  offers them on private repositories with GitHub Enterprise only; on Free,
  Pro, and Team they exist for public repositories only (GitHub
  documentation, checked 2026-10-15). Read the plan from the probe.
- **CI plans, the administrator applies** is the default everywhere else.
  Say why: without required reviewers, a merge would apply with write
  credentials and nobody approving. Never deliver the apply job without
  them.
- **One organization owner.** Say that no second person can review a
  change or approve an apply, propose CI plans with the owner applying
  locally, and record the accepted risk with its date in
  `platform-settings.md`. Prevented self-review would block every apply
  of a single owner, so do not offer the CI apply as a way around it.

For the local steps, the administrator's `gh` login needs `admin:org`
(`gh auth refresh -s admin:org`). With state, pass `--allow-delete` as the
recipes' guard argument; with HCP Terraform, the administrator runs
`terraform login` once, which keeps their own token in Terraform's
credentials file.

Those recipes are written for `.github/policy.tf`; adapt them before
delivering them here. Run every step in the repository root, where this
repository's tf lives, instead of `.github`, and call the guard as
`scripts/tf_plan_guard.py`. With state, drop the
`rm -f terraform.tfstate terraform.tfstate.backup` parts: the backend holds
the state, and only stateless tf needs a fresh empty one. Keep removing
`plan.out` as the recipes do.

## The CI gate

Start from
`assets/workflow-config-repo.yml`:

- Plan on `pull_request`, never `pull_request_target`, so a fork's pull
  request gets no secret. With state, the pull request plan reads state
  without taking its lock (`-lock=false`).
- On a push to the default branch, one job plans, runs the guard, writes
  its output to the job summary, and uploads the saved plan with a short
  retention. The apply job is bound to an environment whose required
  reviewers are organization owners, with **prevent self-review** on, and
  whose deployment branches are limited to the default branch; it
  downloads that plan and applies it, so a second person approves the plan
  they read. A plan made stale by another apply is refused; run the
  workflow again.
- Serialize runs with a concurrency group that never cancels.
- Pin the provider and commit its lock file, and initialize with
  `-lockfile=readonly`.
- Protect the default branch so it takes changes only through pull
  requests with at least one approving review. GitHub never counts the
  author's own approval, so a change and its apply each need a second
  person.
- Agents draft tf changes and pull requests. They never approve the
  environment, and they never run an apply without the user's explicit
  approval.

## Credentials for CI

Use GitHub Apps owned by the organization. An installation token is minted
per run and expires after one hour, so no long-lived token sits in CI. The
private key is the credential to protect: a pull request runs its own
workflow, so everyone who can push a branch to the configuration
repository can read every secret the pull request's job is given, and mint
any token that key allows. Narrowing the minted token to read does not
narrow the key, so the plan and the apply use separate Apps.

1. Create the **plan App** under the organization's developer settings,
   with no webhook and only the read level of each permission the
   declared resources need: an organization ruleset needs organization
   **Administration: read**. Look up any other resource's endpoint in
   GitHub's App-permission reference.
2. Install it on the organization, with repository access limited to the
   configuration repository. Generate a private key and store it as a
   **repository** secret of the configuration repository, and the App's
   client ID as a repository variable; an environment secret would not
   reach the plan job, which runs on pull requests outside any
   environment. Everyone who can push to the configuration repository can
   read what this App reads: keep its collaborators to the organization's
   owners, and say so in `platform-settings.md`.
3. Only in the "CI plans and applies" mode, create the **apply App** the
   same way with the write level of the same permissions (organization
   **Administration: write** for a ruleset). Store its private key as a
   secret of the **apply environment**, and its client ID as a variable of
   that environment. A job reads an environment secret only after a
   required reviewer approves it (GitHub documentation, checked
   2026-10-15), so no pull request and no unapproved run reaches the write
   key.
4. **Never store either key as an organization secret**: an organization
   secret can be shared with other repositories, whose workflows would then
   hold a credential that reads or changes organization rules.
5. Record each App, its permissions, each secret's name and scope (never
   its value), and the key-rotation owner in `platform-settings.md`.
6. With HCP Terraform, Terraform reads the API token for that host from
   `TF_TOKEN_app_terraform_io`. Keep the HCP Terraform organization to this
   workspace, so the token reaches nothing else.
   - On a plan with teams, use two team tokens: one whose team can only
     read this workspace's state, stored as a **repository** secret for the
     plan jobs, which plan with `-lock=false`; and one whose team can write
     state, stored as a secret of the apply environment. Whether a
     read-only state token suffices for such a plan was not checked;
     confirm it on a workspace that manages nothing.
   - On the Free plan, which has no teams, use the user token of a
     dedicated HCP Terraform account as a repository secret, and record that
     the plan jobs hold a token that can write state.
   - HashiCorp's token documentation says organization tokens cannot start
     runs, and whether they can write state was not confirmed (read
     2026-10-15), so do not use one. Confirm the choice in that
     documentation before creating the token.

   Record in `platform-settings.md`, never the value, each token's type, as
   above; its expiry, set when created and at most one year; and its
   rotation owner, the organization owner who creates it unless the user
   names someone else. A third-party backend's credential follows the same
   rule.

A fine-grained personal access token is the fallback only when an App is
impossible. It belongs to a person and lives long, so record its owner and
expiry in the same row. Never propose a broader personal token to make
automation work.
