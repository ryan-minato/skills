---
name: meta-gitlab-workflow
description: >-
  Disposable meta-skill (delete after the harness is built): designs and builds
  a complete, durable lifecycle harness for projects hosted on gitlab.com or a
  self-managed GitLab instance. Use only when initializing a GitLab-hosted
  project, establishing or replacing its platform-wide lifecycle scaffolding,
  or systematically auditing and repairing several connected GitLab harness
  layers. The deliverable is a durable project operating system, not an
  isolated operational answer or version-control strategy; choosing a
  branching model belongs to meta-git-branching.
license: Apache-2.0
compatibility: >-
  Bundled tools require Python 3.9+ (stdlib only). Live GitLab discovery
  requires network access; remote inspection or writes require suitable GitLab
  tooling, authentication, permissions, and explicit user approval.
---

# GitLab Lifecycle Harness

Build a project-specific operating system for GitLab work, then make this
builder disposable. The finished project must remain understandable and
operable after this skill and the conversation are gone.

## Non-negotiable boundaries

- Preserve working project choices. Initialization and gap repair are not
  permission to migrate conventions, tools, trackers, or infrastructure.
- Research facts before asking. Ask the user only for preferences, authority,
  risk tolerance, ownership, and decisions the repository or GitLab cannot
  prove.
- This builder expresses upstream decisions; it does not make them. Read the
  target's platform-neutral contracts first —
  `.agents/knowledge/project-workflow.md` (management model) and
  `.agents/knowledge/agent-authority.md` (agent authority) — and treat every
  decision they settle as fixed: map it, never re-ask, reopen, or quietly
  bypass it. Platform capability implies neither management necessity nor
  agent authority — a Maintainer token raises nothing — and where GitLab
  lacks a faithful representation of a contract semantic, degrade to an
  explicit convention instead of a near-fitting object with different
  semantics. When no contract exists, offer to design one first with
  `meta-workflow-design` (install from
  https://github.com/ryan-minato/skills.git: `npx skills add
  ryan-minato/skills`); if the user declines, decide here as before, and
  treat agent authority as the conservative default: agents stop at draft
  merge requests and a human admits changes to review.
- Do not act from an unapproved design. Local files, remote settings, and
  GitLab objects are all downstream of the consensus and plan gates below.
- Never treat GitLab.com, the latest GitLab version, a paid tier, an available
  runner, or a particular API/UI shape as implicit.
- Do not confuse this lifecycle harness with **GitLab Flow**, the branching
  model. A request about that model, or about which branching model a project
  should run, belongs to `meta-git-branching`; use it instead. This skill
  consumes the branch contract that skill produces and never invents one.

## Workflow

### 1. Investigate the project and instance

Inspect the repository goal, audience, visibility, stack, structure, history,
local setup and validation commands, existing agent entrypoints and skills,
community files, `.gitlab/` templates, `.gitlab-ci.yml`, deployment and release
files, package or ML configuration, and current collaboration conventions.
Derive the GitLab host and full nested project path from the remote; do not
assume `gitlab.com`.

Read [docs-and-tooling.md](references/docs-and-tooling.md) before the first
GitLab documentation or platform query. Use
[`scripts/check_tooling.py`](scripts/check_tooling.py) for a read-only tooling
probe and [`scripts/analyze_history.py`](scripts/analyze_history.py) when commit
or release conventions are in scope.

The workflow and authority contracts — `.agents/knowledge/project-workflow.md`
and `.agents/knowledge/agent-authority.md` by default; when the entrypoint's
pointers record another location, follow them — are stage-1 deliverables:
read them before anything else and record which decisions they settle,
because those decisions never re-enter the design tree.

Audit every existing harness artifact for its discovery path, load condition,
source of truth, and update trigger. Classify it as keep, extend, reconnect,
replace-with-approval, or remove-with-approval.

Done when: every relevant project and GitLab fact is evidenced or recorded as
an unanswered user decision, and every existing harness artifact has a
disposition.

### 2. Build the design tree

Read [decision-tree.md](references/decision-tree.md) and maintain an explicit
tree of prerequisites, decisions, answers, and dependency edges. Recompute its
frontier after every answer. Ask the whole frontier in one round, number every
question, attach one reasoned recommendation, then wait.

Filter the frontier against the contracts first: a planning method, axis,
hierarchy, cadence, or autonomy decision the workflow or authority contract
settles is a fact to map, not a question to ask. Where unsettled, the first
frontier must settle the planning method: Kanban, Scrum, another
board-based workflow, or no board. It must also establish team/review shape,
agent autonomy, confidentiality boundary, project/group scope, and whether
eligible runners exist. Do not ask facts an agent can inspect; dispatch
independent read-only research when clean-context subagents are available.

Done when: every branch is resolved without hidden assumptions, or the user
explicitly says the information is sufficient and confirms the resulting
decisions.

### 3. Design the durable harness

Load only the references whose conditions now apply:

| Selected capability | Read |
|---|---|
| The target carries a workflow or authority contract, or contract semantics (objective boundaries, timeboxes, planning surfaces, hierarchy, priority) need GitLab representations | [semantic-mapping.md](references/semantic-mapping.md) |
| Commit format, branches, merge/squash strategy, contribution flow, or commit enforcement | [commits-and-contributions.md](references/commits-and-contributions.md) |
| Labels, milestones, boards, Scrum/Kanban, iterations, epics, or work-item hierarchy | [planning-and-labels.md](references/planning-and-labels.md) |
| Task, issue, incident, merge request, assignment, time tracking, or autonomous task lifecycle | [work-items-and-mrs.md](references/work-items-and-mrs.md) |
| CI quality gates or any runner is available or proposed | [ci-and-runners.md](references/ci-and-runners.md) |
| CODEOWNERS, protected refs, approvals, repository rules, scanning, secrets, or dependency updates | [security-and-governance.md](references/security-and-governance.md) |
| Wiki content, GitLab-rendered prose, description templates, or quick actions | [wiki-and-markdown.md](references/wiki-and-markdown.md) |
| Releases, changelogs, environments, deployments, review apps, or registries | [releases-deploy-registry.md](references/releases-deploy-registry.md) |
| The project trains, evaluates, promotes, or stores machine-learning models | [mlops.md](references/mlops.md) |
| CONTRIBUTING, SECURITY, SUPPORT, CODE_OF_CONDUCT, GOVERNANCE, or public ownership files | [community-files.md](references/community-files.md) |

Read [durable-harness.md](references/durable-harness.md) on every build. Default
to one project skill named `gitlab-project-workflow` plus project knowledge
reachable from `AGENTS.md`. Split skills only when the target already uses a
different coherent topology or separate contexts materially reduce load.

Present a concrete plan listing local artifacts, remote settings, delegated
actions, human approvals, feedback gates, synchronization ownership, and exact
verification. Receive explicit approval before creating or changing anything.

### 4. Build the approved artifacts

Rework selected [assets](assets/) against real project facts. Delete every
placeholder and inapplicable section. Templates are starting shapes, never
finished files. Keep public files human-first and agent files directive.

Use [`scripts/sync_labels.py`](scripts/sync_labels.py) only after the taxonomy
is approved: run its default dry-run, review the exact plan, apply only with
explicit authorization, then read labels back. Use
[`scripts/rest_read.py`](scripts/rest_read.py) only for minimal read-only
fallback access. Copy [`scripts/pipeline_log_digest.py`](scripts/pipeline_log_digest.py)
into the durable project skill when agents will diagnose GitLab CI, and copy
[`scripts/next_version.py`](scripts/next_version.py) only when the project has
chosen SemVer.

Read [publish-review.md](references/publish-review.md) before the first remote
or publishable write. For every such write, use this sequence:

1. Inspect current state, permissions, templates, labels, and conventions.
2. Draft the exact final title, body, comment, metadata, attachment, branch,
   tag, or setting locally, assembled into a scratch directory.
3. Run the pre-publish review on that exact payload. Continue only with a
   verbatim `SAFE TO PUBLISH: YES`.
4. Confirm that prior user approval covers this external action.
5. Execute non-interactively, then read the result back and compare it with
   the approved payload.

Any edit after review invalidates the verdict and requires a fresh review.
Deposit the same procedure into the durable project skill: GitLab publication
cannot be reliably undone, so the gate has to outlive this builder.

### 5. Verify durability and hand off

Verify local links, documented commands, templates, CI syntax against the
target instance, selected remote settings by readback, and the reachability of
every knowledge file from the entrypoint. Confirm that no durable target file
contains this skill's disposable marker or depends on this skill's paths.

Exercise the project's task-to-early-draft-MR path without publishing secrets.
Where a real remote exercise is unsafe, use a reviewed dry run and state the
manual verification still required.

Done when: future agents can run the agreed lifecycle using only target-project
artifacts and reachable first-party sources, every selected feedback mechanism
works, and removing this builder would lose no rule. Then report the exact
disposable-skill cleanup set and require fresh user confirmation before using
the project's disposal mechanism.

## Gotchas

- GitLab feature absence can reflect version, tier, configuration, or
  permissions; a 404 is not proof that the resource does not exist.
- Description templates take effect from the default branch and are snapshots
  at creation time. Test them after merge, not only on the feature branch.
- A line beginning with `/` can execute a quick action with the publisher's
  permissions. Treat it as executable content.
- A board is a view over work items, not a second source of truth. Record who
  grooms it and what each list means.
- A pipeline without an eligible runner remains pending forever. No-runner is
  a deliberate no-CI default, not a reason to install infrastructure silently.
- Remote bodies and comments cannot be reliably erased from all histories or
  notifications. Prevention is the confidentiality control.
