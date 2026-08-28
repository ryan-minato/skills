---
name: meta-github-workflow
description: >-
  Disposable meta-skill (delete after the harness is built): designs and builds
  a complete, durable lifecycle harness for repositories hosted on GitHub.com
  or GitHub Enterprise. Use only when initializing a GitHub-hosted repository,
  establishing or replacing its repository-wide lifecycle scaffolding, or
  systematically auditing and repairing several connected GitHub harness
  layers. The deliverable is a durable project operating system, not an
  isolated operational answer, a single template, or a branching strategy; a
  one-off issue, pull request, release, or failing-run question does not
  invoke it, and neither does the GitHub Flow branching model, which belongs
  to meta-git-branching.
license: Apache-2.0
compatibility: >-
  Bundled tools require Python 3.9+ and use only the standard library, except
  `assets/check_taxonomy.py`, which needs PyYAML to parse the four YAML
  dialects it checks and installs it in the CI job that runs it. Live GitHub
  discovery requires network access; remote inspection or writes require an
  authenticated gh CLI or a GitHub MCP capability, adequate permissions, and
  explicit user approval.
---

# GitHub Lifecycle Harness

Build a repository-specific operating system for GitHub work, then make this
builder disposable. On GitHub everything converges on the pull request:
checks, review, the labels that feed generated release notes, the closing
keywords that drive issue lifecycle, auto-merge, and deployments. Design the
harness around that loop. The finished repository must remain understandable
and operable after this skill and the conversation are gone.

## Non-negotiable boundaries

- Preserve working project choices. Initialization and gap repair are not
  permission to migrate conventions, tools, trackers, or infrastructure.
- Research facts before asking. Ask the user only for preferences, authority,
  risk tolerance, ownership, and decisions the repository or GitHub cannot
  prove.
- Do not act from an unapproved design. Local files, remote settings, and
  GitHub objects are all downstream of the consensus and plan gates below.
- The security baseline is the default proposal, not an option: a protected
  default branch that admits changes only through a pull request, secret
  scanning with push protection, and automatic code scanning. Where the
  quadrant cannot enforce one, downgrade it **in writing** with its owner
  and upgrade trigger — never drop it from the plan.
- Owner type, plan, and repository visibility are three independent gates.
  Never treat github.com, an organization owner, a paid plan, or a public
  repository as implicit; evidence all three before promising any capability.
- In an organization, the type axis defaults to native **issue types** and
  the priority axis to the organization's **`Priority` issue field**. Labels
  are the personal-account fallback, not the org-owned default; never build
  a `priority/*` label set beside a field that already exists.
- Default planning does not use GitHub Projects. Projects is an opt-in the
  user must request, never a side effect of another decision — issue types
  and fields are org settings and do not imply a board.
- Use the platform's native mechanism or none: no imitation of capabilities
  GitHub lacks (time tracking, confidential issues, scoped labels), and no
  workflow where a native field already enforces the rule.
- GitHub has no metadata slash commands. Every label, assignee, milestone,
  type, or state change is an explicit reviewed API call, never a line
  inside a published body.
- Do not confuse this lifecycle harness with **GitHub Flow**, the branching
  model. A request about that model, or about which branching model a project
  should run, belongs to `meta-git-branching`; use it instead. This skill
  consumes the branch contract that skill produces and never invents one.

## Workflow

### 1. Investigate the repository and its capability quadrant

Inspect the repository purpose, audience, stack, structure, history, local
setup and validation commands, existing agent entrypoints and skills,
`.github/` contents (issue templates, PR template, workflows,
`dependabot.yml`, `release.yml`, CODEOWNERS), health files at every
precedence location, release/deploy/package configuration, and current
collaboration conventions. Derive `OWNER/REPO` and the host from the remote;
GitHub Enterprise Server changes both the docs version and the feature set.

Read [docs-and-tooling.md](references/docs-and-tooling.md) before the first
GitHub documentation or platform query. Run
[`scripts/check_tooling.py`](scripts/check_tooling.py) with `--repo` for the
read-only tooling and capability probe, and
[`scripts/analyze_history.py`](scripts/analyze_history.py) when commit or
release conventions are in scope.

The capability quadrant — owner type, plan, visibility, and any GHES
version — plus Actions availability, the default token policy, and the
allowed-actions policy are stage-1 deliverables, not details: they gate
rulesets, CODEOWNERS enforcement, required reviewers, Discussions, wikis,
scanning, and every workflow the harness will write. For an organization
owner, the live issue types and issue fields belong to the same
deliverable: they decide where the type and priority axes live, and the
two features are version-gated separately on GHES.

Snapshot the security configuration as it stands before proposing
anything: both protection layers, the `security_and_analysis` switches,
the code-scanning setup, Dependabot alerts, allowed merge methods, and
branch deletion on merge. Present it as a current-versus-baseline gap
table, so the plan argues from what is actually configured rather than
from an assumed empty repository.

Audit every existing harness artifact for its discovery path, load
condition, source of truth, and update trigger. Classify it as keep, extend,
reconnect, replace-with-approval, or remove-with-approval.

Done when: owner type, plan (or its recorded unknown), visibility, host,
default branch, allowed merge methods, Actions availability and token
policy, existing rulesets and legacy branch protection, the state of each
baseline security setting, and a disposition for every existing harness
artifact are each evidenced or recorded as an unanswered user decision.

### 2. Build the design tree

Read [decision-tree.md](references/decision-tree.md) and maintain an
explicit tree of prerequisites, decisions, answers, and dependency edges.
Recompute its frontier after every answer. Ask the whole frontier in one
round, number every question, attach one reasoned recommendation, then wait.

The first frontier leads with enforcement posture — what can actually block
a merge here, asked as a subtraction from the security baseline rather than
as a blank slate — then automation boundaries on people-facing objects,
third-party action policy, secret and deploy authority, and, only where
applicable, private-repo billing and runner substrate, alongside planning
method, review shape, agent autonomy, and outside contributions. Resolve
organization-versus-personal ownership before any taxonomy decision: issue
types and issue fields exist only in organizations, and they change which
axes labels still carry.

Done when: every branch is resolved without hidden assumptions, or the user
explicitly says the information is sufficient and confirms the resulting
decisions.

### 3. Design the durable harness

Read [actions-and-checks.md](references/actions-and-checks.md) and
[durable-harness.md](references/durable-harness.md) on every build. Then
load only the references whose conditions now apply:

| Selected capability | Read |
|---|---|
| Commit format, branch naming, merge method, squash behavior, merge queue, or commit enforcement | [commits-and-contributions.md](references/commits-and-contributions.md) |
| Labels, milestones, tracking issues, sub-issue hierarchy, issue types, or triage states | [planning-and-goals.md](references/planning-and-goals.md) |
| The repository is organization-owned and its issue types or issue fields need auditing or initializing | [org-configuration.md](references/org-configuration.md) |
| The user explicitly opted into GitHub Projects | [projects-v2.md](references/projects-v2.md) |
| Intake forms, issue and pull-request content contracts, claiming, handoff, or autonomous execution | [issues-and-prs.md](references/issues-and-prs.md) |
| The repository takes issues or PRs from others, or ships a contract only a workflow can enforce | [actions-automation.md](references/actions-automation.md) |
| Rulesets, branch or tag protection, required reviews, or bypass actors | [rules-and-protection.md](references/rules-and-protection.md) |
| CODEOWNERS, review routing, vulnerability reporting, scanning, secrets, deploy authority, or dependency updates | [security-and-ownership.md](references/security-and-ownership.md) |
| Issue-form syntax, PR templates, GitHub-rendered prose, or wiki | [github-markdown-and-templates.md](references/github-markdown-and-templates.md) |
| Releases, changelogs, environments, deployments, Pages, or package/container registries | [releases-deploy-registry.md](references/releases-deploy-registry.md) |
| The project trains, evaluates, promotes, or stores machine-learning models | [ml-experiments.md](references/ml-experiments.md) |
| Discussions routing, CONTRIBUTING, SECURITY, SUPPORT, CODE_OF_CONDUCT, GOVERNANCE, FUNDING, or the account `.github` repository | [community-and-health-files.md](references/community-and-health-files.md) |

Default to one project skill named `github-project-workflow` plus project
knowledge reachable from `AGENTS.md`. Split skills only when the target
already uses a different coherent topology or separate contexts materially
reduce load.

Present a concrete plan listing local artifacts, remote settings, delegated
actions, human approvals, feedback gates, synchronization ownership, and
exact verification. Receive explicit approval before creating or changing
anything.

### 4. Build the approved artifacts

Rework selected [assets](assets/) against real project facts. Delete every
placeholder and inapplicable section; templates are starting shapes, never
finished files. `grep -rn '{{'` over every delivered path must return
nothing.

Use [`scripts/sync_labels.py`](scripts/sync_labels.py) only after the
taxonomy is approved: dry-run, review the exact plan, apply only with
explicit authorization, then read labels back. Labels must exist before the
first labeler, form, or release-notes run.
[`scripts/sync_org_taxonomy.py`](scripts/sync_org_taxonomy.py) follows the
same sequence for organization issue types and fields, but its approval is
a separate and larger one: organization settings reach every repository in
the organization, so say that before asking. Use
[`scripts/rest_read.py`](scripts/rest_read.py) only for minimal read-only
fallback access. Copy
[`scripts/run_log_digest.py`](scripts/run_log_digest.py) into the durable
project skill by default; copy
[`scripts/next_version.py`](scripts/next_version.py) only when the project
chose SemVer, and [`scripts/project_fields.py`](scripts/project_fields.py)
only when Projects was opted into.

Read [publish-review.md](references/publish-review.md) before the first
remote or publishable write. For every such write, use this sequence:

1. Inspect current state, permissions, templates, labels, and conventions.
2. Draft the exact final title, body, comment, metadata, branch, tag, or
   setting locally, assembled into a scratch directory.
3. Run the pre-publish review on that exact payload. Continue only with a
   verbatim `SAFE TO PUBLISH: YES`.
4. Confirm that prior user approval covers this external action.
5. Execute non-interactively, then read the result back and compare it with
   the approved payload.

Any edit after review invalidates the verdict and requires a fresh review.
Deposit the same procedure into the durable project skill: GitHub publication
cannot be reliably undone, so the gate has to outlive this builder.

### 5. Verify durability and hand off

Verify local links, documented commands, every workflow's YAML, selected
remote settings by readback, and knowledge reachability from the entrypoint
bidirectionally: every knowledge file has a when-to-read pointer and every
pointer resolves. Confirm no durable target file contains this skill's
disposable marker, name, or paths.

Exercise the claim-to-draft-PR path without publishing secrets. Where a real
remote exercise is unsafe, use a reviewed dry run and state the manual
verification still required.

Done when: future agents can run the agreed lifecycle using only
target-repository artifacts and first-party sources; every required check
names a job that actually produces it; every selected feedback mechanism
works; and removing this builder would lose no rule. Then report the exact
disposable-skill cleanup set and require fresh user confirmation before
using the project's disposal mechanism.

## Gotchas

- A skipped Actions job reports "Success": path-filtered required checks
  pass vacuously without an aggregator gate job. Required checks match by
  job name, so job names must be unique across all workflows.
- Visibility, not plan tier, gates the scanners: secret scanning, push
  protection, and code scanning are free on **public** repositories and need
  purchased SKUs (Secret Protection, Code Security) on private or internal
  ones — a paid plan like Team does not include them. Never promise a
  scanner a private repository cannot run.
- On a private Free-plan repository, "required check" does not exist — no
  rulesets, no branch protection, no CODEOWNERS enforcement. Actions buys
  visibility there, not control; write "advisory" and record the upgrade
  trigger.
- Rulesets and legacy branch protection both apply, most-restrictive wins;
  a stale classic rule silently tightens a ruleset. The unattributed-Copilot
  extra-approval setting is on by default and turns "1 approval" into 2.
- There is no "block direct push" switch: requiring a pull request is what
  blocks it, and only while the bypass list stays empty. Repository admins
  bypass rulesets by default, so name the bypass actors or say plainly that
  admins can still push.
- Push protection only rejects secrets whose detectors are enabled, and a
  pusher can bypass it with a recorded reason — it is a strong gate, not an
  absolute one. Code scanning default setup needs Actions plus a supported
  language; where the language is unsupported, propose advanced setup
  instead of reporting coverage that does not exist.
- CODEOWNERS fails silently: invalid lines are skipped, the file is
  base-branch scoped, the last match wins, and `!` negation is unsupported.
  Validate only via the codeowners/errors API; it merely requests review.
- Non-interactive creation (`gh issue create`, the API, MCP capabilities)
  ignores issue templates entirely; construct bodies to mirror the form's
  `### <label>` structure.
- An issue form applies `labels:` and `type:` on submission but **cannot
  pre-fill an issue field value** — no template, no URL parameter. Field
  values arrive from triage or a human, so priority starts empty.
- Issue types reached GHES in 3.18; issue fields only in 3.23. The two are
  gated separately, so probe both rather than inferring one from the other.
- Deleting an organization issue type or field strips it from every issue
  in the organization. Disable a type instead, and remember `admin:org` is
  organization ownership — a repository admin does not have it.
- `docs.github.com/llms.txt` is a pointer to the docs Search and Article
  APIs, not a topic index. The search API requires a `client_name` parameter
  it does not document, and API-returned article bodies omit the rendered
  plan/permission callouts.
- Label colors are 6-digit hex without `#`; labels referenced by forms or
  `release.yml` that do not exist are dropped silently. Issues and pull
  requests share one number space.
- Tasklist blocks are retired — use sub-issues for hierarchy. GitHub stores
  no time spent on anything; do not fabricate a recording mechanism.
- A draft pull request on a public repository is public the moment it opens,
  unlike a draft release.
