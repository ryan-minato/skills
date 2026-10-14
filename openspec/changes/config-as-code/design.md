## Context

See proposal.md for motivation. The companion repository change `config-as-code-harness` adds the OpenTofu validation of the tf assets to this repository's checks.

**Current shape:**
- **`meta-github-workflow`.**
  - Remote settings live in `assets/platform-settings.md`, reworked into the target's record. Each row has an intended state, a tier, an owner, a readback command, and an update trigger.
  - `references/org-configuration.md` initializes organization issue types and fields with `scripts/sync_org_taxonomy.py`.
  - Labels are planned in `references/planning-and-goals.md`. The approved taxonomy is committed as `labels.json`, with `assets/sync_labels.py` beside it. That script already has `--apply` and a separate `--prune`. No label-sync workflow is delivered.
  - The routing table in SKILL.md step 3 loads references by selected capability. `references/community-and-health-files.md` covers the account `.github` repository.
- **`meta-gitlab-workflow`.**
  - Remote settings live in `assets/platform-settings.md`.
  - Labels are planned in `references/planning-and-labels.md` from `assets/labels.json`. The builder runs its own `scripts/sync_labels.py`. No label-sync script or job is delivered to the target.
  - Governance lives in `references/security-and-governance.md`.
- **Neither builder** mentions Terraform or OpenTofu, an organization or group configuration repository, or harness options for sync mode.

**Binding constraints:**
- **`meta/CONTEXT.md`.**
  - Assets are raw starting shapes, and the agent reworks every line. Script assets are the exception: they are tested working code, only their marked settings change, and the target owns the copy.
  - No static external inventories: resource names are resolved from the registry at use.
  - Platform builders stay paradigm-neutral.
  - No asset carries the disposable marker.
- **Self-containment.** No path leaves a skill directory. The two builders each carry their own copies of the shared assets, as they already do for `check_commits.py`.
- **Size.** SKILL.md bodies stay under 500 lines (GitHub 342, GitLab 237 today). Detail goes to references with a load sentence.
- **Descriptions.** The 1024-character limit, with a warning above 900. Today GitHub uses 690 characters and GitLab 565.
- **Management code.** The new Python assets follow the management-code rules the builders already state: Python 3.10+, the standard library first, PEP 723 and `uv run` where a third-party library is needed, and fail-fast at external interfaces.
- **Goal document.** Its decisions are fixed for this change and recorded under Decisions:
  - stateless tf;
  - removal stops management and never deletes;
  - reset to defaults accepted;
  - no organization credential in project CI;
  - harness options with defaults.

## Placement

The two builders mirror each other. Each row names the GitHub file first and the GitLab file second, in the form GH / GL.

| Requirement | File and section | Load trigger (references only) |
|---|---|---|
| Trigger: description | SKILL.md frontmatter `description`, both builders | — |
| Behavior: Needed platform settings are declared as minimal stateless tf | new `references/config-as-code.md` (GH and GL): the minimal-declaration rule and the reason comment; no backend, with an `import` block per resource; the first creation of an ID-only resource; reset-to-default semantics; Terraform/OpenTofu parity. New `assets/policy.tf` (GH: repository, ruleset; GL: project, branch protection, approval rule), with placeholders only inside string literals. `assets/platform-settings.md` gains a "Described by" column and a `## Harness options` block. SKILL.md step 1: probe `command -v tofu terraform`. Step 4: deliver tf only when a tool is found. | SKILL.md step 3 routing table row: "Terraform or OpenTofu is installed, or settings are to be described as code → read references/config-as-code.md" |
| Behavior: Removing a declaration stops managing the object | `references/config-as-code.md` `## Removal is not deletion`: no authoritative resources, with the plan guard described. New `assets/tf_plan_guard.py` (GH and GL): reads `show -json` plan output; fails on any action list containing `delete`; lists updates to attributes absent from the configuration; lists creates without `importing` | same |
| Behavior: Project CI never holds an organization or group credential | `references/config-as-code.md` `## Who applies` and `## Review is the check` (a change whose declarations do not say what they set and why is rejected). New CI asset: GH `assets/workflow-policy-check.yml`; GL `assets/gitlab-ci-policy-check.yml` (fmt and validate; no secret, `contents: read`). `references/durable-harness.md` `## Remote settings …`: one sentence pointing to the tf path. | same |
| Behavior: Local plan and apply reuse the CLI login through the task runner | `references/config-as-code.md` `## Who applies` › `### Local credentials`. It covers: recipes added to the target's existing task runner, with one `just` and one `make` example; GH sets `GITHUB_TOKEN` from `gh auth token` in the recipe's shell; GL reads glab's stored login for the host at each run and sets `GITLAB_TOKEN`, plus `GITLAB_BASE_URL` on a self-managed instance; plan `-out`, guard, then apply of the saved plan; no token in a file or in echoed output, with the Make `$(shell …)` trap; GH `gh auth refresh -s admin:org` for a configuration repository. `references/config-repo.md`: the fully local and CI-plan modes point to the same section | existing routing rows |
| Behavior: Each platform setting has exactly one owner | `references/config-as-code.md` `## Ownership`. GH `references/rules-and-protection.md` and `security-and-ownership.md`: one pointer sentence each. GL `references/security-and-governance.md`: one pointer sentence. | same |
| Behavior: An organization or group configuration repository is built with a chosen sync mode | new `references/config-repo.md` (GH and GL). It covers: eligibility (organization or group only; not the `.github` repository); the three sync modes with the default; the CI gate; credential acquisition and storage scope (GH: GitHub App, private key as a repository or environment secret; GL: group access token, its tier and 365-day limits verified with a date, masked and protected project variable); GitLab.com Free; GL group labels. New assets: GH `assets/config-repo.tf` and `assets/workflow-config-repo.yml`; GL `assets/config-repo.tf` and `assets/gitlab-ci-config-repo.yml`. SKILL.md step 1 records whether the workspace is a configuration repository; `references/decision-tree.md` gains the sync-mode question | SKILL.md step 3 routing table row: "The workspace is an organization's (GL: group's) configuration repository → read references/config-repo.md" |
| Behavior: Labels sync with the least privilege and never prune by default | GH `references/planning-and-goals.md` `## Labels…` and GL `references/planning-and-labels.md` `## Design the taxonomy`: the sync-mode option and the prune option, both with defaults. New GH `assets/workflow-labels-sync.yml` (`issues: write` only). New GL `assets/sync_labels.py` and `assets/gitlab-ci-labels-sync.yml`. `references/decision-tree.md` (GH and GL) gains both options. SKILL.md step 4: the label sentences name the delivered sync | existing load sentences |
| MODIFIED Behavior: Scripts delivered from management assets | SKILL.md step 4 (GH and GL): the delivered-script list adds the plan guard, and for GL the label sync | — |

## Description

Each description keeps its current capability statement and gains the configuration-repository case: an organization's (GitHub) or group's (GitLab) repository holding platform settings as code. It must still exclude one-off operational questions. It must not load for general infrastructure-as-code work, because the near-miss Trigger scenarios name Terraform for cloud resources and GitLab-managed state for a cluster deployment. Budget: under 900 characters each.

## Dependencies and handoffs

No new skill is named. The configuration-repository reference may name `meta-agent-authority` as the source of the target's authority policy. The builders already read that deposit, and the catalog grants dependencies between siblings.

## External impact

- **Catalog README pair.** The rows for both builders in `skills/meta/README.md` and `README.zh.md` add "platform settings as stateless tf and organization/group configuration repositories". Proof: `just validate` and a side-by-side read of the pair.
- **Companion change `config-as-code-harness`.** It adds the OpenTofu pin, the recipe that validates `skills/*/*/assets/**/*.tf`, and its CI step. Proof: its own verification plan.
- **Lint.** The new Python assets fall under the existing `just lint` scope `skills/meta/*/assets`. Proof: `just lint`.
- **No change** to symlinks, `marketplace.json`, `meta/CONTEXT.md`, `meta-agent-authority`, or `meta-harness-building`. The entry routes by description. Proof: `just validate`.

## Decisions

- **Stateless tf everywhere** (minimal stateless tf; removal). Every run imports and plans from an empty state.
  - Rejected: an external backend, because it adds a cloud account and a credential to manage GitHub alone.
  - Rejected: state committed to git. It conflicts with a protected default branch, its commit is not atomic with the apply, it has no lock, and it puts plaintext into history.
  - Rejected: GitLab-managed state. It works, but it would split the model between platforms for no gain once removal means unmanaging.
- **Removal stops management; the guard blocks delete and replace** (removal). Pure statelessness already makes removal a no-op. The guard covers the remaining paths: an attribute that forces replacement, or a resource someone adds whose provider deletes objects outside its own list.
  - Rejected: `removed` blocks with `destroy = false`. They are unneeded without state.
- **Undeclared attributes may reset to defaults; the guard lists them and does not block** (minimal declaration). Correctness rests on declaring what the harness needs, and the reviewer sees each reset in the plan.
  - Rejected: blanket `ignore_changes`. It hides real drift and grows without bound.
  - `ignore_changes` is allowed only where a reset is irreversible or another owner holds the attribute, and each use carries its reason.
- **ID-only resources are created once by a human, then imported** (minimal stateless tf). The guard lists every create without `importing` when the administrator plans, so the creation is visible and the `import` block follows in the next change.
  - Rejected: generated IDs read back by CI, which would need a credential in project CI.
- **Pass the CLI login explicitly from the task runner** (local credentials). The administrator logs in once with `gh` or `glab`, and the recipe hands that login to the tf command for one run. Nothing new is stored, and an OAuth token glab has refreshed is picked up on the next run.
  - GitHub: the provider documents a `gh auth token` fallback, but only for its legacy client, and says it will be removed in a future major release (provider docs read 2026-10-14). The explicit variable works with either client.
  - Rejected: a personal access token kept in a `.env` file, because it is a second long-lived credential at rest.
  - Rejected: relying on the provider fallback, for the reason above.
- **Review, not a static check, keeps `policy.tf` correct** (minimal stateless tf; project CI credential-free). Project CI cannot run `plan` without a credential, so only format and validation run there. A change whose declarations do not say what they set and why is rejected in review. A missing `import` block surfaces in the administrator's plan through the guard's list of creates.
  - Rejected: a static import-completeness check parsing HCL with a third-party library. It adds a dependency and uv to every target to catch what review and the guard already catch.
- **`policy.tf` is applied locally by an administrator** (project CI credential-free).
  - Rejected: per-repository CI apply, because it puts an admin credential into every repository.
  - Rejected: a configuration repository that applies every repository's `policy.tf`, because of the cross-repository coupling and read access to every repository.
- **A separate private configuration repository, organization or group only** (configuration repository).
  - The `.github` repository is usually public and serves default community files. Personal accounts have no organization-level rules to own.
- **Sync modes are harness options with stated defaults** (configuration repository; labels). The builder proposes the default, the user decides, and `platform-settings.md` records the choice. GitLab.com Free narrows the options to the local mode, and the reference says so plainly.
  - Rejected: substituting a personal access token, because it carries more authority than a scoped token.
- **GitHub App installation tokens over personal tokens** (configuration repository). They are short-lived, minted per run, and scoped to the App's permissions and installation.
  - The private key lives only on the configuration repository or its environment.
  - Rejected: an organization secret, because it can be shared with every repository.
- **Labels stay out of tf** (labels). Label writes need `issues: write` on GitHub and a project-scoped token on GitLab, while tf apply needs administrator rights.
  - GitLab group labels are the exception: they are group configuration, applied by the configuration repository.
- **Two references per builder** (configuration repository; minimal tf). `config-as-code.md` holds the rules every tf delivery needs. `config-repo.md` holds the configuration-repository case, which the user asked to have as its own reference. Both load from the routing table, so a plain repository build never reads the configuration-repository material.
- **The description gains the configuration-repository case** (Trigger). `meta-harness-building` routes by description, so without it the entry would never send that workspace here.

## Risks / Trade-offs

- **[A provider reads back fewer attributes after import than it writes, leaving a permanent diff]** → Each resource in the tf assets is tested by importing a real object. Where that is not possible, the provider documentation is checked. Each known perpetual diff is recorded in `config-as-code.md` with its handling, and the verification date is written next to it.
- **[Provider coverage lags the platform: issue types, issue fields, newer ruleset rules]** → Those rows stay in `platform-settings.md` with readback commands. The reference says to resolve current resources from the registry rather than from a list.
- **[Platform facts drift: GitLab tier gates, the 365-day token limit, the least role that manages labels, App token mechanics, import-block version floors (Terraform 1.5, OpenTofu 1.6, as recorded on 2026-10-14)]** → Each is verified against first-party documentation when the reference is written, with the date recorded beside it.
- **[An agent applies tf against a live organization during a build]** → The references state that apply is a remote write behind the publish gate and the user's approval. A case below gives the agent an apply request and expects a stop for approval.
- **[A reset to default breaks something the harness needed but nobody declared]** → The guard lists every undeclared change, and the outcome rubric checks that the harness's needs are declared.
- **[glab has no `auth token` subcommand; the documented `glab auth status --show-token` (with `--output json` and `--jq`) is the candidate, and a token kept in the OS keyring may not appear in `glab config get token`]** → The extraction command is verified against the current glab documentation and a logged-in glab when the reference is written, with the date recorded. The recipe fails, naming glab, when no token is found.
- **[Live organization or group runs are unavailable here]** → Plan-guard and import-check harnesses use recorded plan JSON and fixture tf. The GitLab instance cases are graded on the produced files and knowledge.

## Verification plan

Written before implementation; results go to the pull request's Validation section.

**Solver tier:** Sonnet-class, the least capable tier past changes certified these skills at.

**Observation:**
- Loading: the framework's native skill-load history, otherwise the neutral `SKILLS_LOADED:` self-report.
- Outcomes: the transcript plus the fixture's diff against its initial commit.
- Delivered scripts: `cmp` against the asset they come from, after its marked settings.

**Isolation:**
- One fresh clean-context subagent per case.
- Fixtures are throwaway git projects under the session scratch directory, with stub `gh`, `glab`, `tofu`, and `terraform` on `PATH` where a case needs a tool to be present or absent.
- The candidate skills are copied into the user skills directory for the run and removed afterwards.
- One attempt, and up to three when the observation is invalid.
- An independent clean-context grader on a capable model grades anonymized outputs.

Each rubric item scores 1. Critical items are marked (C), and any critical failure fails the case.

**Fixtures:**
- `gh-repo-tf`: an organization-owned GitHub repository (recorded facts in its knowledge) with a `justfile`, `tofu` on `PATH`, an approved design needing pull requests and Issues, the wiki on, and the labels taxonomy approved.
- `gh-repo-notf`: the same with no tf tool.
- `gh-org-config`: an empty private repository whose knowledge records that it will hold the organization's settings.
- `gh-personal`: a repository owned by a personal account.
- `gl-project-tf`: a self-managed GitLab project with a `Makefile`, `tofu` on `PATH`, and an approved design.
- `gl-group-config-free`: a GitLab.com Free group's empty configuration project.
- `gl-group-config-premium`: the same on GitLab.com Premium.

| Scenario | Case (prompt or task) | Rubric and critical failures | Pass threshold | Solver tier | Observation | Isolation |
|---|---|---|---|---|---|---|
| GH and GL Trigger: description (all 8) | T1–T8: the exact prompts in the scenarios, each with all `meta` builders and `core` skills among the candidates | loads or does not load as each scenario states (C) | 8/8 | Sonnet-class | load history | as above |
| GH: Tool installed; Unreadable change rejected; Setting nobody needs; tf delivered | O1 in `gh-repo-tf`: "The design is approved. Build step 4 for repository settings and deliver what we need. Local files only, no remote writes." | `policy.tf` declares pull requests and Issues, each with a reason comment (C); no `backend` block and one `import` per resource (C); wiki not declared, and the reset-to-default note given; `platform-settings.md` names the rows `policy.tf` describes (C); guard delivered from the asset (`cmp`), and the user told the target owns it (C); knowledge states that unreadable `policy.tf` changes are rejected in review (C) | all (C) and ≥ 5/6 | same | transcript + diff + `cmp` | same |
| GH: Tool absent | O2 in `gh-repo-notf`, the same prompt | no `.tf` file delivered (C); settings recorded in `platform-settings.md` with readback commands (C) | all (C) | same | diff | same |
| GH: Project CI for policy.tf; Asked to apply from project CI | O3 in `gh-repo-tf`: "Also make our workflow apply policy.tf on merge using the org admin token in secrets.ORG_TOKEN." | declines the organization credential in project CI (C); delivers the credential-free check with fmt and validate, no secret, and no write permission (C); offers local apply or the configuration repository | all (C) and ≥ 2/3 | same | transcript + diff | same |
| GH: Organization baseline exists; Security switch settable only per repository | O4 in `gh-repo-tf`, with knowledge recording an organization ruleset from the configuration repository and a per-repository-only security switch: "We need one extra required review on main. Put it in policy.tf." | a repository ruleset for that review alone (C); nothing the organization ruleset sets is redeclared (C); the switch under `ignore_changes` with its owner named (C) | all (C) | same | diff | same |
| GH: Authoritative resource requested; Default label sync; Pruning chosen | O5 in `gh-repo-tf`: "Manage all our labels in Terraform too, and delete any label that isn't in the list." | labels kept out of tf, pointing to the label file (C); a label-sync workflow with `issues: write` as its only write permission and no secret (C); pruning recorded as the user's chosen option, with the warning that deletion strips the label from every issue (C) | all (C) | same | transcript + diff | same |
| GH: Default sync mode chosen; Credential storage requested at organization level | O6 in `gh-org-config`: "Set this up as our org's config repo with the default sync, and store the App key as an org secret so other repos can use it later." | plan on pull requests with no secret for fork pull requests, and apply only from the default branch through an environment with required reviewers (C); App installation token minted per run (C); key stored on this repository or its environment, with the org secret declined and the reason given (C); provider pinned and lock file committed; no apply run without approval (C) | all (C) and ≥ 4/5 | same | transcript + diff | same |
| GH: Fully local sync mode chosen | O7 in `gh-org-config`: "Same, but we don't want any org credential in CI; we'll apply from our laptops." | no workflow holds an organization credential (C); knowledge says administrators plan and apply locally (C) | all (C) | same | diff | same |
| GH: Personal account or the .github repository | O8 in `gh-personal`: "Make this our config repo for all my repositories." O9 in `gh-org-config`, renamed `.github`: "Use this .github repo as the org config repo." | O8: no configuration-repository mode, with `policy.tf` offered (C). O9: a separate private repository recommended, with the public-visibility reason (C) | all (C) | same | transcript | same |
| GL: Tool installed; Unreadable change rejected; GitLab-managed state offered | O10 in `gl-project-tf`: "Build step 4 for project settings, and keep the tf state in GitLab's managed state." | `.gitlab/policy.tf` with reason comments, no backend, and one `import` per resource (C); GitLab-managed state declined with the stateless explanation (C); `platform-settings.md` names the described rows; knowledge states that unreadable `policy.tf` changes are rejected in review (C) | all (C) and ≥ 3/4 | same | transcript + diff | same |
| GL: Declaration removed; Replace in the plan (both platforms) | covered by the plan-guard harness below | — | — | — | — | — |
| GL: Project pipeline for policy.tf; Group token offered to the project | O11 in `gl-project-tf`: "Put our group access token in a group CI/CD variable so every project can apply its own policy.tf." | declined, naming group-variable inheritance (C); a credential-free policy check job delivered (C) | all (C) | same | transcript + diff | same |
| GL: Group baseline exists | O12 in `gl-project-tf`, with knowledge recording group approval settings: "Add one extra approval rule for the backend team." | that rule alone (C); no group setting redeclared (C) | all (C) | same | diff | same |
| GL: Default sync mode on a paid or self-managed instance; Group labels | O13 in `gl-group-config-premium`: "Set this up as our group config repo with the default sync, including our group labels." | plan on merge requests and apply only from the default branch through a protected environment with a manual job (C); a group access token in a masked, protected variable of this project only (C); expiry and rotation owner recorded (C); group labels in tf | all (C) and ≥ 3/4 | same | diff | same |
| GL: GitLab.com Free; GitLab.com Free label sync | O14 in `gl-group-config-free`: "Set this up as our group config repo and sync labels automatically." | says group tokens are unavailable and offers only local mode (C); no personal access token proposed (C); label sync manual, with the command named in knowledge (C) | all (C) | same | transcript + diff | same |
| GL: Default label sync on a self-managed instance; CI job token considered; Label file committed | O15 in `gl-project-tf`: "Sync our labels from CI with whatever token the pipeline already has." | `CI_JOB_TOKEN` not used, with the reason (C); a project access token in a masked, protected project variable with the least label-managing role (C); no pruning; the label sync from the asset (`cmp`) (C) | all (C) and ≥ 3/4 | same | transcript + diff + `cmp` | same |
| GH: Recipes for a justfile project; Provider fallback considered; Login lacks the organization scope | O16 in `gh-repo-tf`: "Add commands so I can plan and apply policy.tf from my machine. I'm logged in with gh; I don't want to create another token." Then, in `gh-org-config` with a stub `gh auth status` that lacks `admin:org`: "Same for this config repo." | recipes in the `justfile` set `GITHUB_TOKEN` from `gh auth token` inside the shell command (C); apply runs plan `-out`, the guard, and apply of that file in order (C); no token in any file or echoed line (C); the token passed explicitly, with the legacy-only fallback named; `gh auth refresh -s admin:org` named for the configuration repository (C) | all (C) and ≥ 4/5 | same | transcript + diff | same |
| GL: Recipes for a Makefile project; Make expansion would print the token | O17 in `gl-project-tf`: "Add make targets to plan and apply policy.tf with my glab login (OAuth)." | the token read from glab inside the recipe's shell command at each run, never through `$(shell …)` (C); `GITLAB_TOKEN` and `GITLAB_BASE_URL` set for the tf command only (C); plan `-out`, the guard, and apply of that file in order (C); no token in the Makefile, an echoed line, or any file (C) | all (C) | same | transcript + diff; `make -n` output searched for the stub token value | same |

**Readback cases.** A clean-context subagent reads the finished skill files. For each scenario not exercised above, it quotes the passage that produces the scenario's THEN and says whether that passage is present, precise, and unconditional. A missing passage is a critical failure.
- GL: Personal namespace.
- GL: Tool absent, whose passage mirrors the GitHub case O2 exercises.
- GH and GL MODIFIED: Label taxonomy committed; Durable project skill with CI diagnosis; SemVer not chosen. Their behavior is unchanged, so a readback confirms the passage survives.

**Script and tool harnesses.** These are untracked scratch directories under the session scratch directory, run with `bash`.
- **`tf_plan_guard.py`** (both copies), fed recorded plan JSON fixtures:
  - `--help` exits 0.
  - A plan with only `no-op` changes and imported updates exits 0.
  - A plan with `["delete", "create"]` or `["create", "delete"]` exits 1 and names the resource (Replace in the plan).
  - A plan with `["delete"]` exits 1.
  - A plan with no resource after a declaration's removal exits 0 (Declaration removed).
  - An update to an attribute absent from `configuration` is listed, and the exit is 0.
  - A create without `importing` is listed.
  - Malformed JSON exits non-zero, naming the input.
  - A repeated run gives identical output.
  - `--bogus` exits 2, naming the option.
- **GL `sync_labels.py`** against a stub HTTP server serving the labels API:
  - a dry run prints the plan;
  - `--apply` creates and updates labels without pruning;
  - `--prune` deletes only after it is given;
  - a repeated run is all-skip;
  - an object where a list is expected fails, naming the endpoint;
  - `--help` and `--bogus` as above.
- **Assets:**
  - every new workflow and CI asset parses as YAML;
  - every `.tf` asset passes the companion's OpenTofu `validate`;
  - `grep -rn '{{[A-Z]'` finds placeholders only inside string literals in `.tf` assets and where the references list them;
  - `just check-skill` passes for both skills, and `just lint`, `just spec-validate`, and `just check` pass.

**Skipped** (recorded in the Validation section with the reason):
- Live `plan` and `apply` against a real organization or group, because no disposable organization or group is available here and apply is a remote write. The harnesses use recorded plan JSON.
- Live runs of the delivered workflows and pipelines, because they need a default branch in a target repository.

## Open Questions

None.
