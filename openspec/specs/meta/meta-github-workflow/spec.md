# meta/meta-github-workflow Specification

## Purpose
Governs what an agent that loaded the `meta-github-workflow` builder observably does when it consumes platform-worded deposits and delivers a paradigm-neutral GitHub lifecycle base whose templates and project skill carry registered extension slots for a paradigm builder to fill.

## Requirements

### Requirement: Behavior: Platform-worded deposits are implemented, not translated again
The builder SHALL read the workflow deposit at `.agents/knowledge/github-workflow.md` (or the path the entrypoint records), implement the objects it names, and append label semantics, tracking conventions, and mechanics to that same file instead of creating a separate planning file.

#### Scenario: Deposit already present
- **WHEN** the target carries `.agents/knowledge/github-workflow.md` naming its objects
- **THEN** the builder creates no separate planning knowledge file and records its additions in the existing one

### Requirement: Behavior: Templates and the project skill carry paradigm-neutral extension slots
The delivered pull request template, issue forms, and project skill SHALL contain no paradigm-specific line, field, or step, SHALL keep the headings, step positions, and field ids the builder's slot list names so that a paradigm builder can insert its lines by structure alone, SHALL contain no placeholder or anchor comment once delivered, and the builder SHALL hand off to the builder whose description claims the contract the entrypoint points to.

#### Scenario: Specification contract present at build time
- **WHEN** the target carries a specification contract and the builder delivers the base
- **THEN** the delivered template and project skill contain no specification line or step, every slot heading and field id is present, and the hand-off names the paradigm shaping as the next step

#### Scenario: No paradigm contract
- **WHEN** the target carries no paradigm contract
- **THEN** the delivered files contain no placeholder and no anchor comment, and the slot structures are still present

#### Scenario: Security line survives a slot fill
- **WHEN** a paradigm builder later inserts items into the checklist slot
- **THEN** the security item's wording is unchanged and the checklist check passes

### Requirement: Behavior: A required check is named in the ruleset only after it has run on the default branch
The builder SHALL sequence a new required check as workflow first, live and green on the default branch, then the ruleset entry, and SHALL say that a check the platform has never observed blocks every pull request.

#### Scenario: New check named
- **WHEN** a new workflow job is to become a required check
- **THEN** the builder merges the workflow and confirms a run on the default branch before editing the ruleset

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

#### Scenario: Guard for a configuration repository with state
- **WHEN** the configuration repository keeps state
- **THEN** its workflow runs the same delivered plan guard in its delete-listing mode, not a separate script

### Requirement: Behavior: Delivered workflows and the taxonomy check follow the management-code rules
The workflows the builder delivers SHALL hide no unexpected failure behind a fallback and SHALL mark every deliberate deferral where it is written, the aggregator gate being the designed point that decides on the needed jobs' results and fails whenever one failed or was cancelled; the taxonomy check SHALL declare its YAML dependency in PEP 723 inline metadata and run through `uv run`, the builder recommending uv for the target's harness where a developer environment lacks it and presenting dependency locking as the user's decision, defaulted by the job's risk.

#### Scenario: Workflows delivered
- **WHEN** the builder delivers the checks, commit-check, taxonomy, and tag-check workflows
- **THEN** no delivered step reads an unexpected failure as success, every deliberate deferral is marked where it is written, and the gate step, run as its workflow runs it with a needs result containing a failed job, exits non-zero

#### Scenario: Taxonomy check without local uv
- **WHEN** the target's developers do not have uv installed
- **THEN** the builder recommends adding uv to the target's harness for the taxonomy check instead of bundling a YAML parser, says that locking is optional for the read-only taxonomy job, and writes no lock the user did not choose

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

### Requirement: Behavior: A repository's own settings are declared as minimal stateless tf when the tool is installed
When Terraform or OpenTofu is installed, the builder SHALL offer `.github/policy.tf`, which declares only the repository settings that the harness or a requirement the user stated depends on, gives each declaration its reason, declares no backend, and has an `import` block for every resource. It SHALL limit `policy.tf` to enumerable, idempotent repository attributes, such as the name, description, visibility, template flag, archive state, merge methods, branch deletion on merge, and whether issues, discussions, and the wiki are enabled, plus any rule the harness needs that the organization does not set uniformly; it SHALL keep `platform-settings.md`, recording there which rows the tf file describes, what no provider covers, and each harness option chosen; and it SHALL record in the target's knowledge that review is the check on `policy.tf`: a change whose declarations do not make their effect and reason readable is rejected.

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

### Requirement: Behavior: Without state, removing a declaration stops managing the object and never deletes it
For stateless tf, which is `policy.tf` and any configuration repository that chose no state backend, the builder SHALL deliver tf that never deletes or replaces a platform object. It SHALL use no resource that deletes or rewrites objects outside its own list. It SHALL run the plan guard in its default mode, which fails on any delete or replace action and lists every change to an attribute the configuration does not declare.

#### Scenario: Declaration removed
- **WHEN** a ruleset declaration and its `import` block are removed from `policy.tf` and the plan guard reads the resulting plan
- **THEN** the plan contains no delete action and the guard passes, because the ruleset is no longer managed

#### Scenario: Replace in the plan
- **WHEN** the plan guard reads a plan whose resource changes include `["delete", "create"]`
- **THEN** the guard exits non-zero and names the resource

#### Scenario: Authoritative resource requested
- **WHEN** the user asks the builder to manage every repository label through tf
- **THEN** the builder keeps labels out of tf, points to the label file and its sync, and writes no resource that would delete labels missing from a list

### Requirement: Behavior: With state, a delete is applied only from a saved plan a human approved
In a configuration repository with a state backend, removing a declaration deletes the object. The builder SHALL say so. It SHALL run the plan guard in its delete-listing mode, which lists every delete and replace for approval and passes. It SHALL apply only the saved plan that a human approved after seeing that list. To stop managing an object without deleting it, it SHALL replace the declaration with a `removed` block whose lifecycle sets `destroy = false`.

#### Scenario: Declaration removed with state
- **WHEN** an organization ruleset's declaration is removed from a configuration repository that keeps state
- **THEN** the plan shows the delete, the guard lists it among the deletes to approve and passes, and the apply runs only that saved plan after the environment's required reviewers approve it

#### Scenario: Stop managing without deleting
- **WHEN** the user wants the configuration repository to stop managing an organization ruleset but keep it on the platform
- **THEN** the builder replaces its declaration with a `removed` block with `destroy = false`, and the plan shows no delete

### Requirement: Behavior: Project CI never holds a credential that can change organization rules
The builder SHALL have `policy.tf` planned and applied by a repository administrator with their own credentials and SHALL give the project's CI only credential-free checks of the tf files: format and validation.

#### Scenario: Project CI for policy.tf
- **WHEN** the builder delivers the CI check for `policy.tf`
- **THEN** the job runs format and validation, references no secret, declares no write permission, and the knowledge file says an administrator applies `policy.tf` locally

#### Scenario: Asked to apply from project CI
- **WHEN** the user asks for the repository's own workflow to apply `policy.tf` on merge with an organization token
- **THEN** the builder declines to put an organization-level credential in the project's CI, and offers the local apply or the configuration repository instead

### Requirement: Behavior: Local plan and apply reuse the GitHub CLI login through the task runner
Where an administrator plans or applies locally, the builder SHALL add recipes to the target's existing task runner (such as a `justfile` or `Makefile`) that pass the GitHub CLI's stored login to the tf command line as `GITHUB_TOKEN` read from `gh auth token` inside the recipe's shell, rather than relying on the provider's own CLI fallback; the plan recipe SHALL save the plan to a file and run the plan guard on it, removing a plan the guard rejects, and the apply recipe SHALL apply only that saved file; and no recipe SHALL write the token to a file or print it.

#### Scenario: Recipes for a justfile project
- **WHEN** the target uses a `justfile`, `policy.tf` is delivered, and the administrator is logged in with `gh auth login`
- **THEN** the delivered recipes set `GITHUB_TOKEN` from `gh auth token` within the recipe's shell command, the plan recipe runs plan with `-out` and the guard on that plan's JSON, the apply recipe applies only that plan file, and no token value appears in the recipe text, its echoed command, or any file

#### Scenario: Provider fallback considered
- **WHEN** the agent considers leaving the provider without a token so it falls back to `gh auth token` by itself
- **THEN** it passes the token explicitly instead, saying the provider's GitHub CLI fallback is limited to its legacy client

#### Scenario: Login lacks the organization scope
- **WHEN** the workspace is a configuration repository applied locally and `gh auth status` shows no `admin:org` scope
- **THEN** the builder names `gh auth refresh -s admin:org` as the step before the first plan and writes nothing that stores a token

### Requirement: Behavior: Each platform setting has exactly one owner
The builder SHALL assign every managed setting to exactly one of the organization's configuration repository or the repository's `policy.tf`. It SHALL keep the security baseline, the team workflow, and every rule the organization can set uniformly in organization-level mechanisms, and SHALL recommend that the configuration repository carry most managed resources. It SHALL prefer one organization-level resource over one resource per repository, because a state backend that bills or caps managed resources counts each one. It SHALL leave `policy.tf` the repository's enumerable, idempotent attributes and any rule the organization does not set uniformly.

#### Scenario: Organization baseline exists
- **WHEN** the organization's configuration repository enforces an organization ruleset and the repository needs one extra required review
- **THEN** `policy.tf` declares a repository ruleset for that review alone, and declares nothing the organization ruleset already sets

#### Scenario: Same rule needed by several repositories
- **WHEN** several repositories each need the same extra required review and the configuration repository keeps its state in HCP Terraform
- **THEN** the builder recommends one organization ruleset targeting those repositories in the configuration repository, not one repository ruleset per repository, saying that each managed resource counts against the HCP Terraform organization's resource limit

#### Scenario: Security switch settable only per repository
- **WHEN** a security setting the organization owns can only be set on the repository resource
- **THEN** `policy.tf` lists that attribute under `ignore_changes` with a comment naming the configuration repository as its owner

### Requirement: Behavior: An organization configuration repository is built with a chosen sync mode
When the workspace is an organization's configuration repository, the builder SHALL load its configuration-repository reference, SHALL present the sync mode as a harness option, and SHALL scope any CI credential to that repository alone. The default sync mode is CI plan and apply only where the private configuration repository can require reviewers on an environment; elsewhere it is CI plans with an administrator applying locally, and no workflow then holds a credential that can write organization settings. A CI credential that a pull request's workflow can reach SHALL grant only reads, because whoever can push a branch can read every secret that workflow is given; the write credential SHALL be reachable only by the apply job after a reviewer other than the run's initiator approves it, and the default branch SHALL take changes only through pull requests with at least one approving review. It SHALL present the state backend as a harness option:
- when the tool is Terraform, the default is HCP Terraform with the workspace in local execution mode, so it stores only state and runs happen in Actions or on an administrator's machine;
- when the tool is OpenTofu, the default is a backend the user names, either a third-party service that supports OpenTofu or object storage the organization controls, because OpenTofu does not promise compatibility with HCP Terraform;
- in both cases, the alternatives are stateless tf and a third-party backend.

#### Scenario: Default sync mode chosen
- **WHEN** the user accepts the default sync mode for a private configuration repository whose plan provides environment required reviewers on private repositories
- **THEN** the delivered workflow plans on pull requests with no secret available to fork pull requests; on the default branch it saves a plan, and the apply job behind an environment whose required reviewers are organization owners, with self-review prevented, applies only that saved plan; the plan jobs mint their token from a plan App that holds only read permissions, whose private key is a repository secret, and the builder records that everyone who can push to the configuration repository can read what that App reads; the apply job mints its token from a separate App with write permissions, whose private key is a secret of the apply environment only; the default branch takes changes only through pull requests with at least one approving review; and the provider version is pinned with its lock file committed

#### Scenario: No environment required reviewers on the plan
- **WHEN** the private configuration repository's plan does not provide environment required reviewers on private repositories and the user accepts the default sync mode
- **THEN** the default is CI plans with an administrator applying locally, the builder says why, and no workflow holds a credential that can write organization settings

#### Scenario: One organization owner
- **WHEN** the organization has a single owner
- **THEN** the builder says that no second person can review a change or approve an apply, proposes CI plans with the owner applying locally, and records the accepted risk in `platform-settings.md`

#### Scenario: Default state backend with Terraform
- **WHEN** `terraform` is installed and the user accepts the default state backend
- **THEN** the configuration declares a `cloud` block naming the HCP Terraform organization and workspace, the workspace runs in local execution mode, the HCP Terraform API token is a secret of the configuration repository only with its type, expiry, and rotation owner recorded in `platform-settings.md`; where the HCP Terraform plan has teams, the plan jobs use a token that can only read state and the apply job a write token kept as a secret of the apply environment, and otherwise the builder records that the token the plan jobs hold can write state; and the builder states the Free plan's managed-resource limit with the date it was checked

#### Scenario: OpenTofu installed
- **WHEN** only `tofu` is installed
- **THEN** the builder does not default to HCP Terraform, says that OpenTofu does not promise compatibility with it, asks the user to name a third-party or self-controlled backend or to choose stateless tf, and records the choice in `platform-settings.md`

#### Scenario: Fully local sync mode chosen
- **WHEN** the user chooses the fully local sync mode
- **THEN** no workflow holds an organization credential, and the knowledge file says administrators plan and apply locally with their own credentials

#### Scenario: Credential storage requested at organization level
- **WHEN** the user proposes storing the App private key as an organization secret available to all repositories
- **THEN** the builder declines and stores it on the configuration repository only, saying that an organization secret can reach other repositories' workflows

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
