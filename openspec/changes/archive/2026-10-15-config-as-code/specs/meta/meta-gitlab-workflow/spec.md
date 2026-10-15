## ADDED Requirements

### Requirement: Trigger: description
The description SHALL load the builder for building, replacing, or systematically repairing a GitLab-hosted project's lifecycle harness, including the case where the project is a group's configuration repository, and SHALL not load it for a single operational question or for infrastructure code unrelated to GitLab settings.

#### Scenario: Group configuration repository
- **WHEN** the user says "This project will hold our GitLab group's settings as OpenTofu code and apply them from CI. Set up its harness."
- **THEN** the skill loads

#### Scenario: Project harness
- **WHEN** the user says "Set up the complete GitLab workflow for this new project: templates, pipelines, protected branches, labels."
- **THEN** the skill loads

#### Scenario: Unrelated infrastructure near-miss
- **WHEN** the user says "Use OpenTofu's GitLab-managed state to deploy our Kubernetes cluster."
- **THEN** the skill does not load

#### Scenario: One-off label near-miss
- **WHEN** the user says "Create a `blocked` label in this project."
- **THEN** the skill does not load

### Requirement: Behavior: A project's own settings are declared as minimal stateless tf when the tool is installed
When Terraform or OpenTofu is installed, the builder SHALL offer `.gitlab/policy.tf`, which declares only the project settings that the harness or a requirement the user stated depends on, gives each declaration its reason, declares no backend (GitLab-managed state included), and has an `import` block for every resource. It SHALL limit `policy.tf` to enumerable, idempotent project attributes, such as the name, description, visibility, archive state, merge method, source-branch removal, the merge checks, and whether issues and the wiki are enabled, plus any rule the harness needs that the group does not set uniformly; it SHALL keep `platform-settings.md`, recording there which rows the tf file describes, what no provider covers, and each harness option chosen; and it SHALL record in the target's knowledge that review is the check on `policy.tf`: a change whose declarations do not make their effect and reason readable is rejected.

#### Scenario: Tool installed
- **WHEN** `tofu` or `terraform` is on the path, the design is approved, and the harness needs merge requests and a protected default branch
- **THEN** the delivered `policy.tf` declares those settings, each with a comment naming the need it serves; it carries no `backend` block and one `import` block per resource; and `platform-settings.md` names the rows `policy.tf` describes

#### Scenario: Unreadable change rejected
- **WHEN** the builder delivers `policy.tf`
- **THEN** the knowledge file states that a reviewer rejects a `policy.tf` change whose declarations do not say what they set and why, and no check beyond format and validation stands in for that review

#### Scenario: Tool absent
- **WHEN** neither `tofu` nor `terraform` is on the path
- **THEN** no tf file is delivered, and the remote settings are recorded in `platform-settings.md` as before

#### Scenario: GitLab-managed state offered
- **WHEN** the user asks to keep the state of the project's `policy.tf` in the project's GitLab-managed Terraform state
- **THEN** the builder keeps `policy.tf` stateless, explaining that its settings are enumerable and idempotent, so every run imports them, and that GitLab-managed state is the default of the group's configuration repository

### Requirement: Behavior: Without state, removing a declaration stops managing the object and never deletes it
For stateless tf, which is `policy.tf` and any configuration repository that chose no state backend, the builder SHALL deliver tf that never deletes or replaces a platform object. It SHALL use no resource that deletes or rewrites objects outside its own list. It SHALL run the plan guard in its default mode, which fails on any delete or replace action and lists every change to an attribute the configuration does not declare.

#### Scenario: Declaration removed
- **WHEN** a branch-protection declaration and its `import` block are removed from `policy.tf` and the plan guard reads the resulting plan
- **THEN** the plan contains no delete action and the guard passes, because the protection is no longer managed

#### Scenario: Replace in the plan
- **WHEN** the plan guard reads a plan whose resource changes include `["create", "delete"]`
- **THEN** the guard exits non-zero and names the resource

### Requirement: Behavior: With state, a delete is applied only from a saved plan a human approved
In a configuration repository with a state backend, removing a declaration deletes the object. The builder SHALL say so. It SHALL run the plan guard in its delete-listing mode, which lists every delete and replace for approval and passes. It SHALL apply only the saved plan that a human approved after seeing that list. To stop managing an object without deleting it, it SHALL replace the declaration with a `removed` block whose lifecycle sets `destroy = false`.

#### Scenario: Declaration removed with state
- **WHEN** a group label's declaration is removed from a configuration repository that keeps state
- **THEN** the default-branch pipeline's plan job saves a plan that shows the delete, the guard lists it among the deletes to approve and passes, and the manual apply job applies only that saved plan

#### Scenario: Stop managing without deleting
- **WHEN** the user wants the configuration repository to stop managing a group setting but keep it on the platform
- **THEN** the builder replaces its declaration with a `removed` block with `destroy = false`, and the plan shows no delete

### Requirement: Behavior: Project CI never holds a credential that can change group rules
The builder SHALL have `policy.tf` planned and applied by a project maintainer or owner with their own credentials, and SHALL give the project's pipeline only credential-free checks of the tf files: format and validation.

#### Scenario: Project pipeline for policy.tf
- **WHEN** the builder delivers the pipeline job for `policy.tf`
- **THEN** the job runs format and validation and reads no CI/CD variable holding a token, and the knowledge file says a maintainer applies `policy.tf` locally

#### Scenario: Group token offered to the project
- **WHEN** the user proposes a group-level CI/CD variable holding a group access token so every project can apply its own `policy.tf`
- **THEN** the builder declines, saying that group variables are inherited by every project in the group, and offers the local apply or the configuration repository instead

### Requirement: Behavior: Local plan and apply reuse the glab login through the task runner
Where a maintainer plans or applies locally, the builder SHALL add recipes to the target's existing task runner (such as a `justfile` or `Makefile`) that pass glab's stored OAuth or personal login for the instance to the tf command line as `GITLAB_TOKEN`, with `GITLAB_BASE_URL` set for a self-managed instance, read inside the recipe's shell at each run so a refreshed OAuth token is used; the plan recipe SHALL save the plan to a file and run the plan guard on it, removing a plan the guard rejects, and the apply recipe SHALL apply only that saved file; and no recipe SHALL write the token to a file or print it.

#### Scenario: Recipes for a Makefile project
- **WHEN** the target uses a `Makefile`, `policy.tf` is delivered, and the maintainer is logged in with `glab auth login` using OAuth
- **THEN** the delivered targets read the token from glab within the recipe's shell command at each run, set `GITLAB_TOKEN` (and `GITLAB_BASE_URL` on a self-managed instance) for the tf command only, the plan target runs plan with `-out` and the guard, the apply target applies only that plan file, and no token value appears in the Makefile, its echoed command, or any file

#### Scenario: Make expansion would print the token
- **WHEN** the agent drafts the token read as a Make-level `$(shell …)` expansion
- **THEN** it rewrites it as a shell substitution inside the recipe line, because Make echoes a recipe line after expanding its variables

### Requirement: Behavior: Each platform setting has exactly one owner
The builder SHALL assign every managed setting to exactly one of the group's configuration repository or the project's `policy.tf`. It SHALL keep the security baseline, the team workflow, and every rule the group can set uniformly in group-level mechanisms, and SHALL recommend that the configuration repository carry most managed resources, preferring one group-level resource over one resource per project. It SHALL leave `policy.tf` the project's enumerable, idempotent attributes and any rule the group does not set uniformly.

#### Scenario: Group baseline exists
- **WHEN** the group's configuration repository sets the group's merge request approval settings and the project needs one extra approval rule
- **THEN** `policy.tf` declares that rule alone, and declares nothing the group already sets

### Requirement: Behavior: A group configuration repository is built with a chosen sync mode
When the workspace is a group's configuration repository, the builder SHALL load its configuration-repository reference, SHALL present the sync mode as a harness option defaulting to CI plan and apply where the instance allows it, and SHALL scope any CI credential to that project alone. It SHALL present the state backend as a harness option. The default is the configuration project's GitLab-managed Terraform state, which every tier provides and which works with both Terraform and OpenTofu. The alternatives are stateless tf and a third-party backend.

#### Scenario: Default sync mode on a paid or self-managed instance
- **WHEN** the group is on GitLab.com Premium or Ultimate or on a self-managed instance, and the user accepts the default sync mode
- **THEN** the configuration keeps its state in the configuration project's GitLab-managed Terraform state, and pipeline jobs authenticate to that state with the job token; the delivered pipeline plans on merge requests without taking the state lock, saves a plan on the default branch, and applies only that saved plan from a manual job in a protected environment; the apply credential is a group access token with `api` stored as a masked, protected CI/CD variable of the configuration project only, and the merge request plan uses a separate read-only group access token stored masked and unprotected there, with the builder recording that everyone who can push a branch to the configuration project can use the plan token; the token's expiry within 365 days and its rotation owner are recorded in `platform-settings.md`; and the provider version is pinned with its lock file committed

#### Scenario: GitLab.com Free
- **WHEN** the group is on GitLab.com Free
- **THEN** the builder says group access tokens are unavailable there, offers only the fully local sync mode, and proposes no personal access token as a CI substitute; the state can still be the configuration project's GitLab-managed state, reached with each maintainer's own glab login

#### Scenario: Group labels
- **WHEN** the group's labels are in scope
- **THEN** they are declared in the configuration repository's tf, and project labels stay in each project's label file

#### Scenario: Personal namespace
- **WHEN** the project lives in a personal namespace
- **THEN** the builder offers no configuration-repository mode and offers `policy.tf` instead

### Requirement: Behavior: Labels sync with a project-scoped token and never prune by default
The builder SHALL keep project labels in the label file rather than in tf, SHALL present the label sync mode as a harness option that defaults to a pipeline job using a project access token where the instance provides one and to manual sync where it does not, and SHALL present pruning of labels absent from the file as a harness option that is off by default.

#### Scenario: Default label sync on a self-managed instance
- **WHEN** the project is on a self-managed instance and the user accepts the default
- **THEN** the delivered job runs on default-branch pipelines that change the label file, authenticates with a project access token stored as a masked, protected variable of that project with the least role that manages labels, and does not prune

#### Scenario: CI job token considered
- **WHEN** the agent considers using `CI_JOB_TOKEN` for the label sync
- **THEN** it does not, because the job token cannot call the labels API

#### Scenario: GitLab.com Free label sync
- **WHEN** the project is on GitLab.com Free
- **THEN** the label sync mode is manual, and the knowledge file names the command a maintainer runs

## MODIFIED Requirements

### Requirement: Behavior: Scripts delivered to the target come from management assets the target owns
The builder SHALL deliver every script it gives the target — the commit check, the label sync committed beside the label file, the tf plan guard when tf is delivered, and the pipeline-log digest and next-version helpers of the durable project skill — from its management assets rather than from the scripts the builder runs itself, with no rule, check, or instruction binding a delivered script to one of those (an asset may still match one where the role needs the same code); SHALL change only an asset's marked settings; and SHALL tell the user that the target owns the delivered scripts.

#### Scenario: Durable project skill with CI diagnosis
- **WHEN** agents will diagnose GitLab CI in the target and the builder delivers the durable project skill
- **THEN** the pipeline-log digest inside the project skill comes from the management asset, and nothing delivered to the target requires it to stay identical to a file in the builder's `scripts/`

#### Scenario: SemVer not chosen
- **WHEN** the project did not choose SemVer
- **THEN** no next-version helper is delivered

#### Scenario: Guard for a configuration repository with state
- **WHEN** the configuration repository keeps state
- **THEN** its pipeline runs the same delivered plan guard in its delete-listing mode, not a separate script

#### Scenario: Label file committed
- **WHEN** the approved label taxonomy is committed to the target
- **THEN** the label sync script delivered beside it comes from the builder's management asset, and nothing delivered requires it to stay identical to the builder's `scripts/sync_labels.py`
