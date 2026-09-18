---
name: spec-kit-workflow
description: >-
  Runs Spec-Kit features through pull or merge requests — the approval
  package (the specification and the plan; tasks after approval),
  completion before ready as every task ticked since the kit archives
  nothing, the declaration that locks a feature when the kit gives no
  artifact to freeze, the `/spec show` and `/spec status` comment commands
  and the progress labels over touched features — and installs that
  automation on GitHub or GitLab. Use when a project that runs Spec-Kit
  asks whether the plan is approved before tasks, what must be complete
  before the PR or MR is ready, what locks a feature for approval, what
  the `/spec` commands or `spec/*` labels do, or to add the
  feature check, the comment commands, or the progress labels to a
  repository. Not for running the kit's own commands (`specify`, its
  slash commands) to create or implement one feature, for choosing a spec
  tool or adopting the practice, or for a project on another framework.
license: Apache-2.0
compatibility: >-
  The bundled script requires Python 3.10+ (stdlib only) and git.
---

# Spec-Kit Features Through Pull and Merge Requests

Precedence: an explicit user instruction, then the project's specification
contract (`.agents/knowledge/spec-workflow.md` or the file the agent
entrypoint points to), then this skill's defaults. Two loop facts hold
everywhere: the approval package precedes the task list, and the
approval applies to a version that was locked first.

## The feature directory and the approval package

Layout verified against the kit's templates and feature script on
2026-09-17 (re-check when the kit updates: it has renamed its command set
before):

- One constitution per repository under the kit's hidden directory; the
  plan step reads it, so it is the home of engineering principles.
- One numbered directory per feature under `specs/` (`NNN-name`), created
  by the kit's feature script — which creates the directory and **no git
  branch**; any branch follows the project's branching rules. It holds
  `spec.md` and `plan.md` (required), `tasks.md` (`- [ ] T001 …`
  checkboxes), and optional research, data-model, contract, and
  quickstart files.
- The draft opens once the approval package is complete: `spec.md` and
  `plan.md`. The plan bounds the approach — constraints, preferences,
  rejected alternatives — and lists no steps; a plan written as an ordered
  procedure is rewritten before the draft opens. It is committed, so it
  carries no secret or private data.
- `tasks.md` is after-approval material even when the kit generated it:
  push it as a draft, mark it as after-approval on the request, keep it out
  of the review, and finish it after the discussion is closed.
- The kit ships no validator: check the feature against the kit's template
  headings and say that no programmatic check exists. Every kit command
  runs through the kit, verified from its `--help` and current
  documentation; a feature directory made by hand lacks what the kit's
  later steps read.

## Completion, then the lock

Spec-Kit has no archive operation. A request is ready when every task of
every feature it touches is ticked, and a task is ticked only when its
verification ran; the installed check fails a ready request otherwise and
warns while it is a draft. Under the split shape it passes one such
request: the specification request, which carries the specification and
the plan and implements nothing — `--shape split` allows a touched
feature with no ticked task, and fails the moment one is ticked.
Combined projects leave the flag at its default and get no exception. When the project's level is spec-anchored, the
project's own rule says how the living specification is updated — nothing
in the kit enforces it, so name the rule rather than assume it.

Because there is nothing to archive, the lock the approval applies to is
a **declaration**, not a commit: when the gate owner closes the
deliberation on the finished implementation, reconcile it, then record on
the request that the deliberation closed with the specification and the
plan unchanged, naming the commit they stand at. From there a change to
either is a new round. Watch for a spent lock the same way an archiving
tool watches for a stale freeze: a commit after the one the declaration
names means the approved version no longer exists — say so before
pushing or handing over, and ask for the gate again. A contract may name
a different lock; apply the contract's.
[`scripts/spec_kit_features.py`](scripts/spec_kit_features.py) reports the
touched features and their state:

```bash
python3 scripts/spec_kit_features.py status --base origin/main --head HEAD
```

## Commands and labels on a request

A request's *touched features* are the feature directories whose files its
diff touches. The installed automation exposes them as:

- `/spec show [<feature>] [spec|plan|tasks|all]` — the documents of one
  touched feature, or of every touched feature; long output is truncated
  with a link to the file on the branch.
- `/spec status [<feature>]` — the table (required files present, done and
  open tasks) and the first open tasks.
- `spec/not-started` | `spec/in-progress` | `spec/done` — the progress
  label the labels workflow derives from the touched features' task lists.
  Read it; never apply or remove it by hand. A request that touches no
  feature carries none.

On GitLab the same output comes from manual jobs and the label takes effect
on the next pipeline; the platform reference says how.

## Installing the automation

Read [references/github.md](references/github.md) when installing or
changing the automation in a GitHub repository, and
[references/gitlab.md](references/gitlab.md) in a GitLab project. Both
place the script, the check, the commands, and the label, and list the
maintainer actions and the fork-safety rules.

One rule governs every job that runs with a writable token on someone
else's request: **no object authored by the request reaches the runner.**
Such a job checks out the base and reads the head through
`spec_kit_features.py snapshot`, which pulls the file list and the
documents from the platform's API and parses them. Reading the head with
a `git fetch` of its SHA works and is not exploitable on its own, but it
leaves the request's objects one command away from being checked out by a
later edit; the snapshot removes them from the runner instead.

## Handoffs

- Whether or at which level to adopt spec-driven development, what a good
  specification is, the generic loop → the methodology skill
  `spec-driven-development` of the `sdd` catalog of
  https://github.com/ryan-minato/skills. If it is not installed, load the
  `ryan-minato-skills-installing` skill and install it as it directs;
  never run an install command yourself. If the user declines, answer the
  Spec-Kit question at hand from the two loop facts above and say the
  practice-level question stays open.
- Setting or changing the project's rules — the approval mode, the
  completion rule, what locks a feature, the request shape → the spec
  workflow builder `meta-spec-workflow` of the `meta` catalog, installed
  whole through the same installing skill. If the user declines, apply the
  contract as it stands or these defaults — every task ticked before
  ready, both gates closed in conversation, and the lock declared on the
  request — say so, and name the contract update as remaining work.

## Gotchas

- The kit's feature script creates a numbered spec directory, not a git
  branch; a harness that says "the tool creates the branch" sends every
  agent onto the wrong ref.
- The kit's template text ("NEEDS CLARIFICATION" markers, branch-name
  fields) stays inside its files; do not import it into the project's
  contract.
- The kit does not oblige anyone to keep a shipped feature's spec current;
  at spec-anchored that obligation is a written project rule.
- A touched feature whose task list has no checkbox at all counts as not
  started, not as done.
- With no archive step, nothing mechanical marks the moment a feature
  stops changing. A project that skips the declaration has no version for
  its approval to name, and the approval silently follows the branch.
