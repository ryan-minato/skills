# Group Configuration Repository

Read when the workspace is, or is to become, the project that holds a
group's settings as code. Every rule of [config-as-code.md](config-as-code.md)
applies here too — minimal declarations with reasons, the plan guard, and
review as the check — except that this project keeps state by default
(`## State` below). This file adds what is specific to the group's
repository.

## Eligibility

- **Groups only.** A personal namespace has no group-level rules to own.
  When the project lives in a personal namespace, offer its
  `.gitlab/policy.tf` instead and build no configuration repository.
- **A separate private project**, named by the user. It holds:
  - `assets/config-repo.tf`, reworked into its root;
  - the committed lock file;
  - `scripts/tf_plan_guard.py`;
  - the pipeline its sync mode needs;
  - knowledge reachable from `AGENTS.md`, including `platform-settings.md`
    with the sync mode, the state backend, and every credential row, in
    every sync mode.

## What it declares

It declares only group-level settings: the security baseline and the team
workflow, such as group merge request approval settings, push rules,
two-factor requirements, and **group labels**. Group labels are group
configuration and are declared here. Project labels stay in each project's
label file. When several projects need the same rule, declare it once at
group level rather than once per project. It never declares a single
project's own attributes, because each project's `.gitlab/policy.tf` owns
those.

State the blast radius before every approval: an apply here reaches every
project in the group and its subgroups. That approval is separate from any
project-level one.

## State

The state backend is a harness option. Propose the default, let the user
decide, and record the choice in `platform-settings.md` under harness
options. The default is this project's GitLab-managed Terraform state,
which every tier provides and which works with Terraform and OpenTofu
alike (GitLab documentation, checked 2026-10-15). The alternatives are
stateless tf and a third-party backend.

- **GitLab-managed state.** Declare an empty `backend "http" {}` block and
  give its settings through the environment, so nothing in the file names
  a credential. In CI, `assets/gitlab-ci-config-repo.yml` sets the state
  address under `${CI_API_V4_URL}/projects/${CI_PROJECT_ID}/terraform/state/`,
  the lock and unlock addresses, the user `gitlab-ci-token`, and the job
  token as the password. The job token carries the role of the user who
  started the job: reading state needs Developer, and locking and writing
  it need Maintainer.
- **Locally**, a maintainer sets the same `TF_HTTP_*` variables in the
  recipe's shell, with their GitLab username and the token glab holds,
  read as in [config-as-code.md](config-as-code.md). GitLab documents a
  personal access token with the `api` scope for this; whether an OAuth
  token from `glab auth login --web` works here was not checked. If it is
  refused, log glab in with a personal access token instead, and never
  write the token to a file.
- **With state, removing a declaration deletes the object.** Say so to the
  user when you set up state, and again in the same turn whenever the user
  asks to stop managing something: deleting its declaration alone would
  delete it. Run the guard with `--allow-delete`: it lists every delete and
  replace for the approver and passes. Only the saved plan the approver saw
  is applied. To stop managing an object but keep it, replace its
  declaration with a `removed` block:

  ```hcl
  removed {
    from = gitlab_group_label.example
    lifecycle {
      destroy = false
    }
  }
  ```

  `removed` blocks need Terraform 1.7. OpenTofu 1.13.0 plans this block as
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

| Mode | CI does | A maintainer does | Token in CI |
|---|---|---|---|
| CI plans and applies (default) | plan on merge requests; on the default branch, save a plan, and apply it from a manual job | reviews the saved plan and runs the apply job | group access tokens; the job token for state |
| CI plans, the maintainer applies | plan on merge requests | applies locally through the recipes in [config-as-code.md](config-as-code.md) | a read-only group access token |
| Fully local | format and validation only | plans and applies locally | none |

**GitLab.com Free** has no group access tokens (they need Premium or
Ultimate on GitLab.com, and any tier on Self-Managed or Dedicated; checked
2026-10-14). There, only the fully local mode exists. Say so plainly, and
never propose a personal access token, or a bot user's personal token, as a
CI substitute: it carries a person's whole authority. The state can still
be this project's GitLab-managed state, reached with each maintainer's own
glab login.

## The CI gate

Start from `assets/gitlab-ci-config-repo.yml`:

- **Plan on merge request pipelines.** A fork's merge request pipeline runs
  in the fork and gets none of this project's variables, as long as the
  project setting "Run pipelines in the parent project for merge requests
  from forks" stays off; never enable it here. A merge request from a
  branch of this project runs its own CI under the plan token, so everyone
  who can push a branch here can read what that token reads: keep the
  project's members to the group's Owners, and say so in
  `platform-settings.md`. With state, the merge request plan reads state
  without taking its lock (`-lock=false`).
- **Save the plan on the default branch.** A plan job runs the guard and
  keeps the saved plan as a short-lived artifact.
- **Apply only on the protected default branch**, as a manual job that
  applies that saved plan, so whoever runs it approves the plan they
  read. Only
  users allowed to merge to that branch can run a manual job. On Premium or
  Ultimate, a protected environment narrows the job to its allowed
  deployers and approval rules. Protected environments need Premium or
  Ultimate; checked 2026-10-14.
- **`resource_group` serializes applies.** A plan made stale by another
  apply is refused; run the pipeline again.
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
     can read it; it is read-only, and the members rule above bounds who
     can run code under it.

   **Never store either one as a group variable.** Every project in the
   group inherits group variables, and their pipelines would then hold a
   credential that changes group rules.
3. In `platform-settings.md`, record each token's name, scopes, role,
   expiry, and rotation owner, never its value. Before the token exists,
   write the expiry as "at most 365 days after creation, set when created",
   and the rotation owner as the group Owner who creates it unless the user
   names someone else. GitLab can email owners before a token expires; the
   owner rotates it and updates the variable.
