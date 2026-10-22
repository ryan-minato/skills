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
  version tags, or releases exist.
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
| Required checks pass before merge | enforced | ruleset `Default`; `.agents/knowledge/github-checks.md` |
| Review threads resolved before merge | enforced | ruleset `Default` |
| No local commit on `main` | enforced locally | the `no-commit-to-branch` pre-commit hook |
| Rebase for in-repo branches, squash for forks | convention | maintainer at merge time; both methods enabled |
| Branch naming | convention | review; the `change-workflow` project skill applies it |

The declared state of every enforced rule is in
`.agents/knowledge/github-settings.md`.

## Objects in use

| Object | Meaning here |
|---|---|
| Issue | Work that needs its own priority, owner, discussion, or acceptance lifecycle; a small change needs none. |
| Pull request | Every change to `main`; one per branch. A draft is work in progress, never a placeholder for planned work. |
| Acceptance | The linked change's scenarios pass, `just check` and the required checks are green, the maintainer merges. |
| `bug` / `enhancement` / `task` | The type axis, applied by the issue forms. A bug's baseline is the current spec; a task's is its change. |
| `priority/critical` … `priority/low` | Exactly one per issue; orders the open list. |
| `catalog/<name>`, `catalog/repository` | The area axis: every catalog a change touches, or the repository harness itself. |
| `status/needs-triage` | Applied by every form; removed by automation once a type label and exactly one priority label exist. |
| `status/blocked` | Applied by a person when an issue cannot proceed; the body says on what. |
| `spec/pending` / `spec/approved` / `spec/archived` | The one spec state of a pull request, kept by `spec / labels` (`spec-workflow.md` `## Approval gates`). |
| Milestone + `goal` tracking issue | A theme spanning several pull requests; see below. |
| Sub-issues | One level only, under a tracking issue. |
| Actions checks | Deterministic enforcement of structure, safety, and pull-request policy. |

A milestone and its `goal` tracking issue are created or confirmed by the
maintainer. The tracking issue carries the vision, the non-goals, the
specs whose scenarios define completion (linked, not restated), and its
sub-issues; the milestone is only the date bucket. Milestones are closed,
never deleted.

Deliberately not used, with the trigger that would reopen the decision:

| Object | Enable when |
|---|---|
| GitHub Projects boards, iterations | Three or more people work in parallel and filtered issue lists stop being enough. |
| Blocked-by dependency links | Issues are regularly blocked by other tracked issues. |
| Releases, version tags, `release.yml` | Consumers start depending on named versions instead of `main`. |
| Severity labels | An operational incident stream appears. |
| Decision records, CODEOWNERS | A second maintainer joins. |
| A "retired labels" mechanism | Never: a label that no longer applies is deleted once, with authorization, after listing its issues. |

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
