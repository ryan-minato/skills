# Organization Configuration Repository

Read when the workspace is, or is to become, the repository that holds an
organization's settings as code. Every rule of
[config-as-code.md](config-as-code.md) applies here too: minimal
declarations with reasons, no state, removal is not deletion, the plan
guard, and review as the check. This file adds what is specific to the
organization's repository.

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
  workflow its sync mode needs, and knowledge reachable from `AGENTS.md`.

## What it declares

Only organization-level mechanisms: organization rulesets, code security
configurations, Actions policies, and organization settings — the security
baseline and the team workflow. It never declares a single repository,
since each repository's `.github/policy.tf` owns that, and it never declares
labels, which stay in each repository's label file. Issue types and issue
fields follow [org-configuration.md](org-configuration.md); a provider
resource replaces its script only when the registry documents one.

State the blast radius before every approval: an apply here reaches every
repository in the organization. That approval is separate from any
repository-level one.

## Sync mode

The sync mode is a harness option. Propose the default, let the user
decide, and record the choice in `platform-settings.md` under harness
options:

| Mode | CI does | An administrator does | Credential in CI |
|---|---|---|---|
| CI plans and applies (default) | plan on pull requests; apply on the default branch after the environment's reviewers approve | reviews and approves | App installation token |
| CI plans, the administrator applies | plan on pull requests | applies locally through the recipes in [config-as-code.md](config-as-code.md) | App installation token |
| Fully local | nothing that needs a credential (format and validation only) | plans and applies locally | none |

For the local steps, the administrator's `gh` login needs `admin:org`
(`gh auth refresh -s admin:org`).

## The CI gate

Start from
`assets/workflow-config-repo.yml`:

- Plan on `pull_request`, never `pull_request_target`, so a fork's pull
  request gets no secret.
- Apply only from a push to the default branch, in a job bound to an
  environment whose required reviewers are organization owners and whose
  deployment branches are limited to the default branch.
- Plan, run the guard, and apply that saved plan in one job. Serialize runs
  with a concurrency group that never cancels.
- Pin the provider and commit its lock file, and initialize with
  `-lockfile=readonly`.
- Protect the default branch so it takes changes only through pull
  requests.
- Agents draft tf changes and pull requests. They never approve the
  environment, and they never run an apply without the user's explicit
  approval.

## Credentials for CI

Use a GitHub App owned by the organization. Its installation token is
minted per run and expires after one hour, so no long-lived token sits in
CI.

1. Create the App under the organization's developer settings, with no
   webhook. Grant only the permissions the declared resources need: an
   organization ruleset needs organization **Administration: write**.
   Look up any other resource's endpoint in GitHub's App-permission
   reference.
2. Install it on the organization, with repository access limited to the
   configuration repository.
3. Generate a private key and store it as a secret of the configuration
   repository or of its environment, and store the App's client ID as a
   variable there. **Never store the key as an organization secret**: an
   organization secret can be shared with other repositories, whose
   workflows would then hold a credential that changes organization rules.
4. Record the App, its permissions, the secret's name (never its value),
   and the key-rotation owner in `platform-settings.md`.

A fine-grained personal access token is the fallback only when an App is
impossible. It belongs to a person and lives long, so record its owner and
expiry in the same row. Never propose a broader personal token to make
automation work.
