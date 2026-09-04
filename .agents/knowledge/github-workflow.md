# GitHub Workflow

Read this before creating a branch, opening or updating an issue or pull
request, applying labels, creating a milestone, or proposing any new
management structure. GitHub is this repository's only remote and its only
task-management platform; there is no platform-neutral layer and no
migration path, so every rule here is written in GitHub terms.

Repository: `ryan-minato/skills` (personal account, public). Owner and
maintainer: `ryan-minato`. Issue types and issue fields do not exist on a
personal account, so type and priority live on labels.

## What this repository is

A library other projects install by copying skill directories out of it.
A merge changes what future installations receive; installed copies are
never updated in place. Acceptance therefore checks installed behavior and
compatibility, and rollback is `git revert` on `main`.

## Branches and merging

- Model: GitHub Flow. `main` is the only long-lived branch; it must stay
  installable at every commit. No release branches, environment branches,
  version tags, or releases exist (`backup-*` tags are historical and
  unused).
- Short-lived branch name: `<type>/<slug>`, or `<type>/<issue>-<slug>`
  when an issue exists. `<type>` is the dominant Conventional Commit type
  of the change. The slug is also the OpenSpec change name: branch
  `feat/spec-driven-development` pairs with
  `openspec/changes/spec-driven-development/`; a companion repository
  change on the same branch is `<slug>-harness` (see
  `.agents/knowledge/spec-workflow.md`).
- Cut from current `origin/main`; never commit on `main` directly.
- Merge method: rebase merge for branches in this repository (branch
  commits land, so every commit subject follows Conventional Commits);
  squash merge for pull requests from forks (the pull request title
  becomes the subject). Merge commits are disabled in the repository
  settings. Branches are deleted on merge by repository setting.

| Rule | Tier | Where enforced |
|---|---|---|
| Changes reach `main` only through a pull request | enforced | ruleset `Default` |
| Force pushes and deletion of `main` are blocked | enforced | ruleset `Default` |
| Required checks pass before merge | enforced | ruleset `Default`, see `.agents/knowledge/github-checks.md` |
| Review threads resolved before merge | enforced | ruleset `Default` |
| Rebase for in-repo branches, squash for forks | convention | maintainer at merge time; both methods enabled |
| Branch naming | convention | the `change-workflow` project skill |

The live state of every enforced rule is recorded in
`.agents/knowledge/github-settings.md`.

## Objects in use

| Object | Meaning here | What is lost without it |
|---|---|---|
| Issue | Work that needs its own priority, owner, discussion, or acceptance lifecycle. A small self-contained change needs no issue; its pull request explains it. | Work loses its context and priority before implementation. |
| Pull request | Every change to `main`; one per branch. A draft is work in progress, never a placeholder for planned work. | No durable boundary around review evidence and the exact change set. |
| Acceptance | The scenarios of the linked OpenSpec change pass, `just check` and the required checks are green, and the maintainer decides to merge. | Green automation gets mistaken for product acceptance. |
| `bug` / `enhancement` / `task` labels | The type axis, applied by the issue forms. A bug's acceptance baseline is the current spec; a task's is the linked change's scenarios. | Intake cannot route to the right form or baseline. |
| `priority/critical` … `priority/low` | Exactly one per issue; orders the open list. | Urgent work is indistinguishable from the queue. |
| `catalog/<name>` and `catalog/repository` | The area axis: which catalog a change touches, or the repository harness itself. Every applicable one. | Changes spanning the library cannot be filtered or routed. |
| `status/needs-triage` | Applied by every form; removed by automation once a type label and exactly one priority label exist. | New issues sit indistinguishable from triaged ones. |
| `status/blocked` | Applied by a human when an issue cannot proceed; the body says on what. | Blocked work looks abandoned. |
| Milestone + `goal` tracking issue | A theme spanning several pull requests (for example the spec-driven-development series). The maintainer creates or confirms both. The tracking issue carries vision, non-goals, the specs whose scenarios define completion (linked, not restated), and its sub-issues; the milestone is only the date bucket. Close milestones, never delete them. | Multi-PR themes have no completion criterion and no progress view. |
| Sub-issues | One level only, under a tracking issue. | Theme progress cannot be counted. |
| Actions checks | Deterministic enforcement of structure, safety, and pull-request policy. | Mechanical review drifts between contributors and agents. |

Deliberately not used, with the trigger that would reopen the decision:

| Object | Enable when |
|---|---|
| GitHub Projects boards, iterations | Three or more people work in parallel and filtered issue lists stop being enough. |
| Blocked-by dependency links | Issues are regularly blocked by other tracked issues. |
| Releases, version tags, `release.yml` | Consumers start depending on named versions instead of `main`. |
| Severity labels | An operational incident stream appears. |
| Decision records, CODEOWNERS | A second maintainer joins. |
| A "retired labels" mechanism | Never: a label that no longer applies is deleted once, with authorization, after listing the issues that carry it. |

## Decomposition

Create a separate issue only when the piece independently needs its own
priority, owner, discussion, or acceptance. Otherwise it stays a line in
the change's `tasks.md`. Issues for planned work derive from that task
list, each linking the scenarios it closes; acceptance criteria are never
copied into an issue.

## Triage

1. Every form applies its type label plus `status/needs-triage`.
2. The `issues / triage` check derives `priority/*` and `catalog/*` from
   the form's Priority and Catalog answers and removes
   `status/needs-triage` when the type label and one priority label are
   present.
3. The maintainer reviews `is:open label:status/needs-triage` weekly,
   converts questions and ideas to Discussions, and sets priority where
   the automation could not.

## Planning view

Filtered issue and pull-request lists, ordered by `priority/*`; no board.
The open draft pull requests are the "in progress" view. Nothing here is a
record: every fact lives on the issue, the pull request, or the spec.

## Update this file when

- A second maintainer joins, or contributors work in parallel.
- A merge method, protection rule, or required check changes (update
  `.agents/knowledge/github-settings.md` in the same change).
- A "not used" trigger fires.
- The label taxonomy in `.github/labels.json` changes.
