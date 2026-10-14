## ADDED Requirements

### Requirement: Trigger: description
The description SHALL load the builder for building, replacing, or systematically repairing a GitHub-hosted repository's lifecycle harness, including the case where the repository is an organization's configuration repository, and SHALL not load it for a single operational question or for infrastructure code unrelated to GitHub settings.

#### Scenario: Organization configuration repository
- **WHEN** the user says "This repo will hold our GitHub organization's settings as Terraform and keep them in sync from Actions. Set up its harness."
- **THEN** the skill loads

#### Scenario: Repository harness
- **WHEN** the user says "Set up the whole GitHub workflow for this new repository: templates, checks, rulesets, labels."
- **THEN** the skill loads

#### Scenario: Unrelated Terraform near-miss
- **WHEN** the user says "Write a Terraform module that creates an S3 bucket and an IAM role for our app."
- **THEN** the skill does not load

#### Scenario: One-off label near-miss
- **WHEN** the user says "Add a `needs-triage` label to this repository."
- **THEN** the skill does not load

### Requirement: Behavior: Needed platform settings are declared as minimal stateless tf when the tool is installed
When Terraform or OpenTofu is installed, the builder SHALL offer `.github/policy.tf`, which declares only the repository settings that the harness or a requirement the user stated depends on, gives each declaration its reason, declares no backend, and has an `import` block for every resource; it SHALL keep `platform-settings.md`, recording there which rows the tf file describes, what no provider covers, and each harness option chosen; and it SHALL record in the target's knowledge that review is the check on `policy.tf`: a change whose declarations do not make their effect and reason readable is rejected.

#### Scenario: Tool installed
- **WHEN** `tofu` or `terraform` is on the path, the design is approved, and the harness needs pull requests and Issues enabled on the repository
- **THEN** the delivered `policy.tf` declares those settings, each with a comment naming the need it serves; it carries no `backend` block and one `import` block per resource; and `platform-settings.md` names the rows `policy.tf` describes

#### Scenario: Unreadable change rejected
- **WHEN** the builder delivers `policy.tf`
- **THEN** the knowledge file states that a reviewer rejects a `policy.tf` change whose declarations do not say what they set and why, and no check beyond format and validation stands in for that review

#### Scenario: Tool absent
- **WHEN** neither `tofu` nor `terraform` is on the path
- **THEN** no tf file is delivered, and the remote settings are recorded in `platform-settings.md` with readback commands as before

#### Scenario: Setting nobody needs
- **WHEN** the repository has the wiki enabled and neither the harness nor the user depends on it
- **THEN** `policy.tf` does not declare the wiki setting, and the builder says that undeclared settings may be reset to the provider's default by an apply

### Requirement: Behavior: Removing a declaration stops managing the object and never deletes it
The builder SHALL deliver tf that never deletes or replaces a platform object: it SHALL use no resource that deletes or rewrites objects outside its own list, and SHALL deliver a plan guard that fails on any delete or replace action and lists every change to an attribute the configuration does not declare.

#### Scenario: Declaration removed
- **WHEN** a ruleset declaration and its `import` block are removed from `policy.tf` and the plan guard reads the resulting plan
- **THEN** the plan contains no delete action and the guard passes, because the ruleset is no longer managed

#### Scenario: Replace in the plan
- **WHEN** the plan guard reads a plan whose resource changes include `["delete", "create"]`
- **THEN** the guard exits non-zero and names the resource

#### Scenario: Authoritative resource requested
- **WHEN** the user asks the builder to manage every repository label through tf
- **THEN** the builder keeps labels out of tf, points to the label file and its sync, and writes no resource that would delete labels missing from a list

### Requirement: Behavior: Project CI never holds a credential that can change organization rules
The builder SHALL have `policy.tf` planned and applied by a repository administrator with their own credentials and SHALL give the project's CI only credential-free checks of the tf files: format and validation.

#### Scenario: Project CI for policy.tf
- **WHEN** the builder delivers the CI check for `policy.tf`
- **THEN** the job runs format and validation, references no secret, declares no write permission, and the knowledge file says an administrator applies `policy.tf` locally

#### Scenario: Asked to apply from project CI
- **WHEN** the user asks for the repository's own workflow to apply `policy.tf` on merge with an organization token
- **THEN** the builder declines to put an organization-level credential in the project's CI, and offers the local apply or the configuration repository instead

### Requirement: Behavior: Local plan and apply reuse the GitHub CLI login through the task runner
Where an administrator plans or applies locally, the builder SHALL add recipes to the target's existing task runner (such as a `justfile` or `Makefile`) that pass the GitHub CLI's stored login to the tf command line as `GITHUB_TOKEN` read from `gh auth token` inside the recipe's shell, rather than relying on the provider's own CLI fallback; the apply recipe SHALL plan to a file, run the plan guard on it, and apply only that file; and no recipe SHALL write the token to a file or print it.

#### Scenario: Recipes for a justfile project
- **WHEN** the target uses a `justfile`, `policy.tf` is delivered, and the administrator is logged in with `gh auth login`
- **THEN** the delivered recipes set `GITHUB_TOKEN` from `gh auth token` within the recipe's shell command, the apply recipe runs plan with `-out`, the guard on that plan's JSON, and apply of that plan file in order, and no token value appears in the recipe text, its echoed command, or any file

#### Scenario: Provider fallback considered
- **WHEN** the agent considers leaving the provider without a token so it falls back to `gh auth token` by itself
- **THEN** it passes the token explicitly instead, saying the provider's GitHub CLI fallback is limited to its legacy client

#### Scenario: Login lacks the organization scope
- **WHEN** the workspace is a configuration repository applied locally and `gh auth status` shows no `admin:org` scope
- **THEN** the builder names `gh auth refresh -s admin:org` as the step before the first plan and writes nothing that stores a token

### Requirement: Behavior: Each platform setting has exactly one owner
The builder SHALL assign every managed setting to exactly one of the organization's configuration repository or the repository's `policy.tf`, keeping the security baseline and team workflow in organization-level mechanisms and leaving `policy.tf` the repository's description, visibility, and the approval and merge policy the organization does not set.

#### Scenario: Organization baseline exists
- **WHEN** the organization's configuration repository enforces an organization ruleset and the repository needs one extra required review
- **THEN** `policy.tf` declares a repository ruleset for that review alone, and declares nothing the organization ruleset already sets

#### Scenario: Security switch settable only per repository
- **WHEN** a security setting the organization owns can only be set on the repository resource
- **THEN** `policy.tf` lists that attribute under `ignore_changes` with a comment naming the configuration repository as its owner

### Requirement: Behavior: An organization configuration repository is built with a chosen sync mode
When the workspace is an organization's configuration repository, the builder SHALL load its configuration-repository reference, SHALL present the sync mode as a harness option defaulting to CI plan and apply, and SHALL scope any CI credential to that repository alone.

#### Scenario: Default sync mode chosen
- **WHEN** the user accepts the default sync mode for a private configuration repository
- **THEN** the delivered workflow plans on pull requests with no secret available to fork pull requests and applies only from the default branch through an environment with required reviewers; the credential is a GitHub App installation token minted per run from a private key stored as a secret of that repository or its environment; and the provider version is pinned with its lock file committed

#### Scenario: Fully local sync mode chosen
- **WHEN** the user chooses the fully local sync mode
- **THEN** no workflow holds an organization credential, and the knowledge file says administrators plan and apply locally with their own credentials

#### Scenario: Credential storage requested at organization level
- **WHEN** the user proposes storing the App private key as an organization secret available to all repositories
- **THEN** the builder declines and stores it on the configuration repository or its environment only, saying that an organization secret can reach other repositories' workflows

#### Scenario: Personal account or the .github repository
- **WHEN** the owner is a personal account, or the user proposes the `.github` repository as the configuration repository
- **THEN** the builder offers no configuration-repository mode for the personal account and offers `policy.tf` instead; for the `.github` repository, it recommends a separate private repository because the `.github` repository is usually public

### Requirement: Behavior: Labels sync with the least privilege and never prune by default
The builder SHALL keep labels in the label file rather than in tf, SHALL present the label sync mode as a harness option defaulting to a workflow that holds only `issues: write`, and SHALL present pruning of labels absent from the file as a harness option that is off by default.

#### Scenario: Default label sync
- **WHEN** the user accepts the default label sync mode
- **THEN** the delivered workflow runs on pushes to the default branch that change the label file, declares `issues: write` as its only write permission, uses no secret, and does not prune

#### Scenario: Pruning chosen
- **WHEN** the user turns pruning on
- **THEN** the choice is recorded in `platform-settings.md`, and the builder states that deleting a label removes it from every issue and pull request

## MODIFIED Requirements

### Requirement: Behavior: Scripts delivered to the target come from management assets the target owns
The builder SHALL deliver every script it gives the target — the commit check, the taxonomy check, the label sync committed beside `labels.json`, the tf plan guard when tf is delivered, and the run-log digest, next-version, and project-field helpers of the durable project skill — from its management assets rather than from the scripts the builder runs itself, with no rule, check, or instruction binding a delivered script to one of those (an asset may still match one where the role needs the same code); SHALL change only an asset's marked settings; and SHALL tell the user that the target owns the delivered scripts.

#### Scenario: Label taxonomy committed
- **WHEN** the approved taxonomy is committed to the target as `labels.json`
- **THEN** the label sync script delivered beside it comes from the builder's management asset, and nothing delivered to the target requires it to stay identical to a file in the builder's `scripts/`

#### Scenario: Durable project skill with CI diagnosis
- **WHEN** the builder delivers the durable project skill with the Actions-diagnosis branch selected
- **THEN** the run-log digest inside the project skill comes from the management asset, and the project skill runs it from the project's own path

#### Scenario: SemVer not chosen
- **WHEN** the project did not choose SemVer and did not opt into Projects
- **THEN** no next-version or project-field helper is delivered

#### Scenario: tf delivered
- **WHEN** the builder delivers `policy.tf`
- **THEN** the plan guard comes from the builder's management asset, and the user is told the target owns it
