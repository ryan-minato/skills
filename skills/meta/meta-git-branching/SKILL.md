---
name: meta-git-branching
description: >-
  Disposable builder skill (delete after the harness is built): selects a project's
  git branching model — git flow, GitHub Flow, or a GitLab Flow variant —
  settles branch and tag naming, protected refs, and merge method, then deposits
  the contract into the target project as durable, event-triggered knowledge.
  Use when establishing or recording version-control conventions for a
  repository, when asked which branching strategy or git workflow a project
  should follow, when a project maintains several versions at once or ships
  through a fixed environment chain, or when agents keep creating branches,
  merges, and tags that do not match team practice. Not for a one-off branch,
  merge, or release operation, and not for the platform lifecycle harness of
  issues, templates, CI, and releases.
license: Apache-2.0
compatibility: >-
  The bundled detection script requires Python 3.9+ (stdlib only), git on PATH,
  and a git checkout. Enforcing rules on GitHub or GitLab requires platform
  tooling, authentication, permissions, and explicit user approval.
---

# Git Branching Contract

Choose the branching model the project's release and deployment reality earns,
settle its concrete conventions with the team, then leave the contract inside
the target project so it outlives this builder and this conversation.

## Non-negotiable boundaries

- A working model is preserved. Detection establishes what the project already
  does; it is not permission to migrate. Propose a change only when the user
  asks for one, or when the current model provably blocks a stated requirement.
- Every agreed rule lands in a target-project file or another source the next
  agent can reach. A convention that exists only in this conversation does not
  exist for the agent that comes after.
- Branching rules load on events, not on every task. The deposited file is
  reached through a pointer naming the operations that trigger it, never a
  standing instruction to read it.
- Research the repository before asking. Ask the user only for the naming,
  ownership, protection, and risk decisions no checkout can prove.
- This skill does not build the platform lifecycle harness. Issue intake,
  templates, CI, releases, and label taxonomy belong to `meta-github-workflow`
  or `meta-gitlab-workflow`; this skill supplies the branch contract those
  builders assume.
- Disposable builders never enter a commit. Before the first commit of the
  build, add every skill directory whose description opens with
  `Disposable builder skill (delete after the harness is built):` to
  `$(git rev-parse --git-path info/exclude)`, stage explicit paths, and read
  `git status` before each commit; a builder tracked before the build is
  reported, and its deletion lands with the disposal commit.

## The three model families

| Family | Long-lived branches | What triggers a release | Cost |
|---|---|---|---|
| GitHub Flow | the default branch alone | merging a short-lived branch | the default branch must stay deployable, so checks have to be fast and trustworthy |
| GitLab Flow | the default branch plus environment or release branches | a downstream merge, or a cherry-pick onto a stable branch | one extra long-lived ref per environment or per maintained version, each of which can drift |
| git flow | `main` and `develop`, plus release and hotfix branches | cutting a release branch, stabilizing it, then merging it both ways | two integration branches drift apart and every hotfix merges twice |

## Workflow

### 1. Detect the current model

Run [`scripts/detect_branching.py`](scripts/detect_branching.py) in the target
checkout for a read-only evidence sweep: branch inventory grouped by naming
shape, long-lived branch candidates, tag shapes, the merge-commit ratio on the
default branch, and the branch names referenced by CI configuration.

Read [detection.md](references/detection.md) when the evidence supports more
than one model, contradicts the model the user names, or cannot separate a
current convention from an abandoned one.

Done when: the project's current model is named with the evidence behind it, or
the project is recorded as having no branching history to preserve.

### 2. Select the model

Apply these rules in order, and stop at the first that matches.

1. The project already runs a model that works — keep it, record it, and skip
   to step 3. Detection is not a migration mandate.
2. Versions are maintained in parallel, off a single line of development: an
   older major or minor release still receives fixes after a newer one ships.
   Select **GitLab Flow release branches**.
3. A change reaches production by traversing a fixed chain of deployment
   targets — a staging host, a pre-deploy server, a customer acceptance
   environment — and each hop is a decision someone makes. Select **GitLab Flow
   environment branches**.
4. Otherwise select **GitHub Flow** and keep the work on the default branch.

Do not select git flow for a new project. Its two-integration-branch structure
is earned only by explicitly versioned software with several releases in the
wild, and both scenarios above cover that case more cheaply. Select it only for
a project already running it, or when the user chooses it after being told the
cost.

Read the one reference matching the selection: [github-flow.md](references/github-flow.md),
[gitlab-flow.md](references/gitlab-flow.md), or [git-flow.md](references/git-flow.md).

Done when: the selected model is traceable to a project fact — a maintained
version, a deployment target, or the absence of both — and not to a default
preference.

### 3. Settle the conventions with the user

The model fixes the branch topology. It does not fix the names, and the names
are what agents and CI match on. Ask the whole set in one numbered round, each
question carrying one reasoned recommendation drawn from the model reference
and from what detection already found:

1. Short-lived branch naming, and whether the platform's issue-linked branch
   creation supplies it.
2. Long-lived branch names — the environment or stable branches the model
   requires, exactly as they will be typed.
3. Tag naming and the version scheme the tags encode.
4. Which refs are protected, and who may bypass.
5. Merge method (default below), and whether the default branch requires linear
   history.
6. Branch lifetime: deletion on merge, and how a stale branch is reclaimed.
7. The hotfix path, stated as a sequence of refs.

**Merge method default.** Branches inside the main repository land by rebase
merge, keeping the default branch a linear, bisectable history. Contributions
from outside the team land by squash merge, so one reviewed unit enters history
regardless of how the contributor committed. Deviate only when the platform
cannot offer one of the two, or the team states a reason; record the reason
either way.

Done when: every item above has an answer the user confirmed, and no item was
resolved by silent default.

### 4. Enforce what the platform can enforce

Read [enforcement.md](references/enforcement.md) before configuring protection
rules or merge methods on GitHub or GitLab. A rule the platform enforces is a
guarantee; a rule only written down is a convention, and the difference has to
be recorded rather than blurred. Remote writes need explicit user approval and
a readback.

### 5. Deposit the contract

Read [durable-output.md](references/durable-output.md) on every build. Adapt
[assets/git-workflow.md](assets/git-workflow.md) to the project's real answers:
it is a raw shape, not a finished file, and every placeholder must be gone.

Done when: the conventions live in the target project, the entrypoint reaches
them through a pointer that names the triggering operations, and each rule
states whether it is platform-enforced or convention.

### 6. Verify and hand off

Confirm the pointer path resolves, every documented command runs, every remote
setting reads back as approved, and no deposited file carries this skill's
disposable marker or depends on this skill's paths. Simulate removal: with this
builder deleted, the next agent must still be able to name the model, the branch
and tag conventions, and the merge method from target-project files alone.

When this builder runs under `meta-harness-building`, return there for the
closing step. When it runs alone, once the deposit is verified and before the
work goes to review, ask the user whether to delete the disposable builders
now — the build request is not deletion consent — and on that decision load
`meta-disposal`, which lists, confirms, and removes them. If the user
declines, leave the builders in place and out of every commit, and record it
in the handoff.

## Gotchas

- Branch names in a repository record its history, not necessarily its current
  convention. A `develop` branch untouched for two years is an artifact; confirm
  with the user before treating it as the model.
- Rebase merge rewrites commit hashes, so any branch built on the pre-merge
  commits diverges. Teams that share long-lived feature branches feel this
  first; the contract has to say who rebases and when.
- Squash merge collapses a contributor's commits into one. Preserve attribution
  through `Co-authored-by` trailers in the squash message, or the contributor
  disappears from the history their work is in.
- Environment branches only accept downstream flow. A merge from `production`
  back into `main` looks harmless and silently reorders what is deployed;
  hotfixes travel by the path the model defines, not by convenience.
- A protected branch that no one can push to and no automation can merge into
  blocks the project. Configure the merge path before enabling the protection.
- Cherry-picking a fix into a stable branch produces a second commit with a
  different hash. Recording the original commit in the cherry-pick message is
  what makes the two findable later.
