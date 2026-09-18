---
name: openspec-workflow
description: >-
  Runs OpenSpec changes through pull or merge requests — the approval
  package (proposal, delta specs, design when warranted; tasks after
  approval), the spec-less marker, the validator's moments, the
  implementer's archive command inside the request, the `/spec show` and
  `/spec status` comment commands, the archived and progress status
  labels — and installs that automation on GitHub or GitLab. Use when a
  project that runs OpenSpec asks how a change is approved or archived
  before its PR or MR merges, what the `/spec` commands or `spec/*` labels
  do, how to wire the OpenSpec check into CI, or to add the comment
  commands or the status labels to a repository. Not for the
  tool's own command that creates, applies, or archives one change, for
  choosing a spec tool or adopting the practice, or for a project on
  another framework.
license: Apache-2.0
compatibility: >-
  The bundled script requires Python 3.10+ (stdlib only) and git; its
  `check` and `archive` subcommands also need the OpenSpec CLI on PATH.
---

# OpenSpec Changes Through Pull and Merge Requests

Precedence: an explicit user instruction, then the project's specification
contract (`.agents/knowledge/spec-workflow.md` or the file the agent
entrypoint points to), then this skill's defaults. Two loop facts hold
everywhere: the approval package precedes the task list, and the
approval applies to a version that was frozen first.

## What the tool generates and assumes

- One `openspec/` directory: `specs/<domain>/spec.md` is the source of
  truth for each domain's current behavior; `changes/<name>/` holds one
  change in flight — `proposal.md`, delta specs under `specs/`, `design.md`,
  `tasks.md`, and a `.openspec.yaml` the tool writes; `changes/archive/`
  holds completed changes as `<YYYY-MM-DD>-<name>`; a project schema may
  reshape the artifacts.
- Every operation the CLI has a command for — initializing, creating a
  change, validating, archiving — runs through that command, verified from
  the CLI's `--help` and current documentation first; the command set has
  been renamed once already. A change directory made by hand lacks the
  metadata the archive reads and fails later.
- Main specs change only through archived changes; the one hand edit is a
  domain's purpose line when a change reshapes the domain, named in the
  proposal.
- The strict validator proves structure, never content; it runs after each
  artifact edit, before publishing the draft, before ready, and after
  archiving.

## Records and the approval package

- The draft opens once the approval package is complete: `proposal.md`,
  the delta specs, and `design.md` when the project's rule warrants one —
  by default when more than one reasonable approach exists, or the change
  touches structure, interfaces, dependencies, or files outside the
  record; a wording change inside one section needs none. The design
  bounds the approach (constraints, preferences, rejected alternatives)
  and lists no steps; it is committed, so it carries no secret or private
  data.
- `tasks.md` is after-approval material even when the propose step
  generated it with the rest: push it as a draft, mark it as after-approval
  on the request, keep it out of the review, and finish it after the
  discussion is closed.
- A change to the project's own harness, tooling, checks, workflows, or
  documents sets `skip_specs: true` in its `.openspec.yaml` and carries a
  proposal, a design, and tasks with no delta spec.
- Under the split request shape a delta written against a domain another
  change archived later may no longer apply: re-validate when
  implementation starts.

## Ready, then the freeze

The request is marked ready while its change is still open, and the
deliberation on the finished implementation runs against that. The check
fails a ready request holding an unarchived related change, so the
request is red for the whole deliberation; that red is the merge block,
and it is expected — never work around it, and never archive early to
clear it.

When the gate owner closes that deliberation, reconcile it, then archive
inside the request, so the integration branch never holds an unarchived
change. The executor is a person on the request's branch — every task
ticked, then the archive command with its confirmation-skipping flag,
plus its spec-skipping flag for a change marked spec-less, then the
strict validator, then a commit.
[`scripts/spec_changes.py`](scripts/spec_changes.py) does this for every
related change at once and refuses all of them when any has an open task:

```bash
# `<target>` is the branch the request merges into, not always `main`.
python3 scripts/spec_changes.py archive --base origin/<target> --head HEAD
```

No job archives. A platform token cannot push to a fork at all, and a job
that could push to the branch would be the only one needing write access
to the repository's contents; the command above is what it would have run.
On a fork, the executor is whoever holds the branch: its author, or the
maintainer after pulling it locally.

The archive commit is the freeze the approval names. Nothing revokes a
closing made in conversation, so watch for a spent one: a commit after
the archive commit means the approved version no longer exists — say so
before pushing or handing over, and ask for the gate again.

Never tick a task whose verification did not run, never archive a change
with an open task, and never edit an archived record: a defect review finds
afterwards goes to the request's validation section or a follow-up change.

## Commands and labels on a request

A request's *related changes* are the change directories whose files its
diff touches; an archived directory counts under its change name. The
installed automation exposes them as:

- `/spec show [<change>] [proposal|design|tasks|specs|all]` — the documents
  of one related change, or of every related change; long output is
  truncated with a link to the file on the branch.
- `/spec status [<change>]` — the progress table (done and open tasks per
  change) and the first open tasks.
- `spec/archived` | `spec/unarchived`, and `spec/not-started` |
  `spec/in-progress` | `spec/done` — facts the labels workflow derives from
  the related changes' task lists. Read them; never apply or remove them by
  hand. A request with no related change carries none. The check fails a
  ready request that still holds an unarchived related change and warns
  while it is a draft.

On GitLab the same output comes from manual jobs and the labels take effect
on the next pipeline; the platform reference says how.

## Installing the automation

Read [references/github.md](references/github.md) when installing or
changing the automation in a GitHub repository, and
[references/gitlab.md](references/gitlab.md) in a GitLab project. Both
place the script, the check, the commands, the archive executor, and the
labels, and list the maintainer actions and the fork-safety rules the
files rely on.

One rule governs every job that runs with a writable token on someone
else's request: **no object authored by the request reaches the runner.**
Such a job checks out the base and reads the head through
`spec_changes.py snapshot`, which pulls the file list and the documents
from the platform's API and parses them. No installed job checks the head
out, fetches it, installs from it, or runs it. Reading the head with a
`git fetch` of its SHA works and is not exploitable on its own, but it
leaves the request's objects one command away from being checked out by a
later edit; the snapshot removes them from the runner instead.

## Handoffs

- Whether or at which level to adopt spec-driven development, what a good
  specification is, the generic loop → the methodology skill
  `spec-driven-development` of the `sdd` catalog of
  https://github.com/ryan-minato/skills. If it is not installed, load the
  `ryan-minato-skills-installing` skill and install it as it directs;
  never run an install command yourself. If the user declines, answer the
  OpenSpec question at hand from the two loop facts above and say the
  practice-level question stays open.
- Setting or changing the project's rules — the approval mode, when a
  design is warranted, the archive executor, the request shape → the spec
  workflow builder `meta-spec-workflow` of the `meta` catalog, installed
  whole through the same installing skill. If the user declines, apply the
  contract as it stands or these defaults — a person archives once the
  implementation deliberation closes, and both gates close in
  conversation — say so, and name the contract update as remaining
  work.

## Gotchas

- A tool's lowercase `design.md` is a technical design file. `DESIGN.md`
  at the repository root is a reserved name for the visual-design format;
  never rename one into the other.
- A `MODIFIED` delta block replaces the whole requirement at archive time:
  copy every existing scenario, or the validator refuses the change.
- The archive directory name carries the archive date, so the `Spec:` link
  in a request description changes after archiving; both forms are valid.
- A ready request is red until the archive commit lands, and that is the
  design: the red blocks the merge through the deliberation. Reading it
  as a defect leads to archiving early, which hands the reviewer a frozen
  record.
- A related change whose task list has no ticked task is refused as
  incomplete, not archived as empty.
