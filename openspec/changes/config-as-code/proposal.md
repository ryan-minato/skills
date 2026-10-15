## Why

`meta-github-workflow` and `meta-gitlab-workflow` keep a target's platform configuration as a hand-maintained record: `platform-settings.md` rows with readback commands, plus a label sync the maintainer runs by hand. Nothing describes the configuration in a form a plan can compare against the platform. As a result, drift is found by reading rows one by one, and an organization or group has no supported way to keep its configuration in a repository of its own. Terraform and OpenTofu are mature tools for this, so the builders should offer them.

## What Changes

The rules below apply to both builders. Where GitLab differs, the difference is stated.

- **Configuration as code, when the tool is installed.** When Terraform or OpenTofu is installed, the builder offers to deliver `.github/policy.tf` (GitLab: `.gitlab/policy.tf`).
  - The file declares only the settings the harness and the user's stated requirements need, each with its reason. An attribute left out may take any value, including the provider's default.
  - Without the tool, delivery stays as it is today.
  - `platform-settings.md` stays in both cases. It records:
    - what no provider covers;
    - which rows a tf file describes;
    - the harness options chosen.
- **State where CI applies organization-wide rules; none for a repository's own attributes.**
  - `policy.tf` stays stateless and holds only enumerable, idempotent repository attributes: name, description, visibility, template, archive state, merge methods, branch deletion, and whether issues, discussions, and the wiki are on. Rules the organization does not set uniformly also stay there. Every managed resource has an `import` block, every run starts from an empty state, and removing a declaration only stops managing the object.
  - The configuration repository keeps state by default. On GitHub with Terraform, the default is HCP Terraform in local execution mode, storing state only. With OpenTofu, which does not promise compatibility with HCP Terraform, the default is a backend the user names. On GitLab, the default is the configuration project's GitLab-managed state, available on every tier. Stateless tf and third-party backends are the alternatives.
  - With state, removing a declaration deletes the object. The plan guard lists every delete and replace for approval, and only a saved plan a human approved is applied. A `removed` block with `destroy = false` stops managing an object without deleting it.
  - Without state, the plan guard fails on any delete or replace. In both modes it lists every change to an undeclared attribute.
  - The builder never uses a resource that rewrites or deletes objects outside its own list.
- **`policy.tf` is applied by an administrator, never by project CI.**
  - An administrator plans and applies `policy.tf` locally with their own credentials.
  - The project's CI runs `fmt` and `validate` without any credential. No static check of the tf files replaces review: a reviewer rejects a change whose declarations do not say what they do and why.
  - Project CI never holds a credential that can change organization- or group-level rules.
- **Local runs reuse the CLI login.** Where an administrator plans or applies locally, the builder adds recipes to the target's task runner (`justfile`, `Makefile`, or whichever the project uses).
  - GitHub: the recipe passes `gh auth token` as `GITHUB_TOKEN`. The provider's own `gh` fallback exists only in its legacy client, so the recipe does not rely on it.
  - GitLab: the recipe passes glab's stored OAuth or personal login as `GITLAB_TOKEN`, read at each run.
  - The apply recipe plans to a file, runs the guard, and applies only that file. No token is written to a file or printed.
- **One owner per setting, with the organization carrying most of it.**
  - The organization's or group's configuration repository owns the security baseline, the team workflow, and every rule that can be set uniformly. It prefers one organization- or group-level resource over one per repository, because HCP Terraform counts each managed resource against the organization's plan (500 on the Free plan, checked 2026-10-15).
  - `policy.tf` owns the repository's enumerable attributes and any rule the organization does not set uniformly.
  - Where a security setting can only be set per repository, `policy.tf` excludes that attribute with `ignore_changes` and states its owner.
- **New capability: the workspace is a configuration repository.** A new reference in each builder covers:
  - what the configuration repository is: a separate private repository, never GitHub's `.github` repository, and only for an organization or a group;
  - the sync mode, a harness option:
    - CI plans and applies (the default);
    - CI plans and an administrator applies locally;
    - everything runs locally;
  - when CI applies, the gate: plan on pull or merge requests with no secret reaching a fork; apply only from the default branch, through a protected environment with required reviewers;
  - the state backend, a harness option with the defaults above;
  - the credentials when CI is chosen, scoped to the configuration repository alone:
    - GitHub: an organization-installed GitHub App whose installation token is minted per run;
    - GitLab: a group access token, which on GitLab.com needs Premium or Ultimate and expires within 365 days, so its rotation owner is recorded;
  - GitLab.com Free: the fully local mode is the only one available, and the reference says so.
- **Labels stay out of tf and sync with the least privilege.**
  - The label file keeps its role.
  - The sync mode is a harness option: CI (the default) or manual.
  - On GitHub, CI sync runs with `issues: write` on the default token.
  - On GitLab, CI sync needs a project access token stored on the project; on GitLab.com Free, sync is manual.
  - Pruning labels absent from the file is a harness option and is off by default.
  - GitLab group labels belong to the configuration repository's tf.
- **New management assets.**
  - Starting-shape tf for `policy.tf` and for the configuration repository.
  - A plan-guard script, with a delete-listing mode for configuration repositories that keep state.
  - A label-sync workflow (GitHub) or job (GitLab).
  - A GitLab label-sync script, delivered to the target for the first time.
- **Descriptions.** Each builder's description states that it also handles an organization's or group's configuration repository, so the harness entry routes that case to it.

## Skills touched

- `meta/meta-github-workflow` (modified): adds the description trigger, configuration as code, the plan guard, setting ownership, the configuration-repository capability, label sync options, and the newly delivered scripts.
- `meta/meta-gitlab-workflow` (modified): the same, plus the GitLab-specific credential tiers and the newly delivered label sync.

## Installed behavior

- An agent building a GitHub or GitLab harness can now describe a repository's own settings as stateless tf, and can build an organization's or group's configuration repository that keeps its state in HCP Terraform, GitLab, or a backend the user names. It now offers label sync in CI, and offers both sync modes, label pruning, and the configuration-repository sync mode as harness options with stated defaults. → `feat`.
- Where the tool is absent, the agent does exactly what it did before.

## Impact

- Both `SKILL.md` descriptions change. Each stays within the 1024-character limit.
- The rows for both builders in `skills/meta/README.md` and `README.zh.md` add the new capability. No symlink changes, and `marketplace.json` lists no description. The proof is `just validate`.
- `meta/CONTEXT.md` is unchanged. tf assets are starting shapes, and the new Python assets are script assets under the existing rule.
- `meta-agent-authority` is unchanged. Apply is a remote write that needs human approval, and the conservative default already covers it.
- The companion repository change `config-as-code-harness` adds an OpenTofu `validate` of the tf assets to this repository's checks.

## Non-goals

- Managing secrets, variables, or tokens with tf.
- State committed to git, and state for a repository's own `policy.tf`.
- A configuration-repository mode for personal accounts or personal namespaces.
- Changing the label taxonomy rules, the issue-type and issue-field handling in `org-configuration.md`, or the security baseline itself.
- Listing provider resources as a static inventory. Resource names are resolved from the provider registry when used.
- Running tf against a live organization or group in this repository's checks.

## Tracked work

No issue. The goal was settled with the maintainer in conversation on 2026-10-14. The state model was revised with the maintainer on 2026-10-15: the configuration repository keeps state by default, and deletion with state needs an approved plan.
