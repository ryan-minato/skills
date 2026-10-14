# Group Configuration Repository

Read when the workspace is, or is to become, the project that holds a
group's settings as code. Every rule of [config-as-code.md](config-as-code.md)
applies here too: minimal declarations with reasons, no state (GitLab-managed
state included), removal is not deletion, the plan guard, and review as the
check. This file adds what is specific to the group's repository.

## Eligibility

- **Groups only.** A personal namespace has no group-level rules to own.
  When the project lives in a personal namespace, offer its
  `.gitlab/policy.tf` instead and build no configuration repository.
- **A separate private project**, named by the user. It holds:
  - `assets/config-repo.tf`, reworked into its root;
  - the committed lock file;
  - `scripts/tf_plan_guard.py`;
  - the pipeline its sync mode needs;
  - knowledge reachable from `AGENTS.md`.

## What it declares

It declares only group-level settings: the security baseline and the team
workflow, such as group merge request approval settings, push rules,
two-factor requirements, and **group labels**. Group labels are group
configuration and are declared here. Project labels stay in each project's
label file. It never declares a single project, because each project's
`.gitlab/policy.tf` owns that.

State the blast radius before every approval: an apply here reaches every
project in the group and its subgroups. That approval is separate from any
project-level one.

## Sync mode

The sync mode is a harness option. Propose the default, let the user
decide, and record the choice in `platform-settings.md` under harness
options:

| Mode | CI does | A maintainer does | Token in CI |
|---|---|---|---|
| CI plans and applies (default) | plan on merge requests; apply on the default branch as a manual job | reviews the plan and runs the apply job | group access tokens |
| CI plans, the maintainer applies | plan on merge requests | applies locally through the recipes in [config-as-code.md](config-as-code.md) | a read-only group access token |
| Fully local | format and validation only | plans and applies locally | none |

**GitLab.com Free** has no group access tokens (they need Premium or
Ultimate on GitLab.com, and any tier on Self-Managed or Dedicated; checked
2026-10-14). There, only the fully local mode exists. Say so plainly, and
never propose a personal access token, or a bot user's personal token, as a
substitute: it carries a person's whole authority.

## The CI gate

Start from `assets/gitlab-ci-config-repo.yml`:

- **Plan on merge request pipelines.** A fork's merge request pipeline runs
  in the fork and gets none of this project's variables.
- **Apply only on the protected default branch**, as a manual job. Only
  users allowed to merge to that branch can run a manual job. On Premium or
  Ultimate, a protected environment narrows the job to its allowed
  deployers and approval rules. Protected environments need Premium or
  Ultimate; checked 2026-10-14.
- **Plan, guard, and apply the same plan in one job.** `resource_group`
  serializes applies.
- **Pin the provider and commit its lock file**, and initialize with
  `-lockfile=readonly`. Set `GITLAB_BASE_URL` from `CI_API_V4_URL` with a
  trailing slash.
- **Agents draft tf changes and merge requests.** They never run the apply
  job, and they never apply locally without the user's explicit approval.

## Credentials for CI

1. A group Owner creates the group access tokens under the group's
   settings. They are valid for at most 365 days by default, and an
   administrator on GitLab 17.6 or later can change that limit:
   - for applies, a token with `api` and the least role the declared
     settings need;
   - for merge request plans, a token with `read_api` and the least role
     that can read them.
2. Store both as CI/CD variables of **this project only**, as the asset
   names them:
   - the apply token as masked and protected;
   - the plan token as masked and not protected, so merge request pipelines
     can read it.

   **Never store either one as a group variable.** Every project in the
   group inherits group variables, and their pipelines would then hold a
   credential that changes group rules.
3. In `platform-settings.md`, record each token's name, scopes, role,
   expiry, and rotation owner, never its value. Before the token exists,
   write the expiry as "at most 365 days after creation, set when created",
   and the rotation owner as the group Owner who creates it unless the user
   names someone else. GitLab can email owners before a token expires; the
   owner rotates it and updates the variable.
