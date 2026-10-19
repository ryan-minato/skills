# Agent Authority Policy

Read this before marking a pull request ready, requesting review,
approving, merging, or changing any repository setting, and whenever it is
unclear whether an action is yours to take.

Level: **H1** — the maintainer decided on 2026-09-03 that an agent may
admit its own work to review; integration stays human. Outside the scope
below, the level is H0: the agent stops at a draft pull request.

## What agents may do

Take issues; plan, implement, and test; self-review; create atomic
commits; push an authorized branch; open and update a draft pull request;
run `gh pr ready` and request review from the maintainer; respond to
review; prepare the acceptance-evidence report.

`gh pr ready` is permitted only when all of these hold:

- the OpenSpec change's package deliberation on the draft was closed by
  the maintainer in conversation and reconciled with nothing open
  (`.agents/knowledge/spec-workflow.md`), and every task in its
  `tasks.md` is done; or the pull request carries `Spec: none — <reason>`
  (see `.agents/knowledge/spec-workflow.md`). The change is **not**
  archived yet: marking the pull request ready is what opens the
  deliberation that precedes the freeze;
- `just check` passes locally and every required check is green except
  `checks / spec`, which stays red until the freeze;
- every scenario of the change passed, with the evidence in the pull
  request's Validation section;
- the publish gate in the `change-workflow` project skill returned
  `SAFE TO PUBLISH: YES` for the exact pull request payload;
- the user authorized remote writes in the current conversation.
  Authority under this policy never substitutes for that authorization.

## What agents may not do

Approve a pull request; merge; arm auto-merge; create milestones or
tracking issues without a human confirming them; change a ruleset, a
required check, a repository setting, this file, or `AGENTS.md` in a way
that widens agent authority; mark a pull request ready while a required
check is failing, unavailable, or unreliable; execute a task whose change
is missing or unapproved.

## Gates

| Gate | Meaning | Owner |
|---|---|---|
| Review admission | The implementation is complete enough for formal review: draft to ready, review requested. | the agent, under the conditions above |
| Integration | The change enters `main`; engineering responsibility transfers. | the maintainer (`ryan-minato`), by merging |

## Escalation — stop and hand to the maintainer when

- The scope grows beyond the issue or change that authorized it.
- Acceptance cannot be evaluated: no approved change, ambiguous scenarios,
  or a scenario that cannot be executed.
- A required check is unavailable, unreliable, or contradicts other
  evidence.
- The change unexpectedly touches a published skill's behavior that no
  change record covers, `scripts/check_commit_safety.py`, secret
  scanning, or any workflow's permissions.
- Reverting the merge would not restore the previous installed behavior.
- Review evidence conflicts, or confidence is insufficient for a decision
  this policy authorizes.
- A structural decision appears that the change's `design.md` does not
  settle.

Escalating earlier is always allowed. Bypassing a closed gate never is.

## At a human gate, hand over this report

- Goal, and how the change addresses it (the pull request's Specification
  section links the change record).
- Tests executed and results (the pull request's Validation section); the
  state of every required check.
- Scope actually touched, including anything beyond the original intent.
- Known risks and remaining limitations.
- The decisions available to the maintainer: request fixes, reject, or
  merge (rebase for this repository's branches, squash for forks).

## No self-escalation

An agent may exercise less authority than granted; it may never grant
itself more. Editing this file, `AGENTS.md`, the ruleset, required
checks, or any workflow's permissions to relax an agent's own limits is
prohibited. At a policy boundary: stop; propose the change to the
maintainer with benefit, risk, and exact scope; wait; resume only after
the maintainer updates this file. When refusing a request that exceeds
this policy, state this path — the limit is changeable, but only by the
maintainer.

## Pre-authorized actions

What an agent *can* do is set by the dev container and the `gh` login
the user forwards; what it *may* do without a fresh decision is set here
and by the conversation. The permission file `.claude/settings.json`
allows the read-only and recipe commands every task runs and denies the
common forms of the known-dangerous ones (force and batch pushes, hard
resets, deletion of `main`, merge, approve, deletes and settings writes
on the platform); a reordered flag gets past a prefix rule, which is why
it is fool-proofing against a careless agent and the container is the
boundary. It carries no `ask` entry.

- Reversible internal actions — editing files, branching, local commits,
  running recipes — proceed without asking.
- A push goes only to the branch the change's proposal names in its
  `## Deliverable and pre-authorized actions` section, one named ref per
  push, and only after the publish gate; the conversation's authorization
  (change-workflow §1) is still required and covers exactly the writes it
  lists.
- Every other outward write — a comment, an issue, a label, a review
  request, a setting — is pre-authorized in the conversation or queued as
  a pending decision; a refusal is a decision to re-plan or queue, never
  to route around with an equivalent command.

## Threat model

Recorded 2026-10-19. No unattended agent runs in this repository: every
agent session is directed by the maintainer, who reviews results before
they leave. The workflows that read issue and comment text
(`issues / triage`, `spec / command`, `spec / labels`) are deterministic
scripts without a model, run from the default branch with job-level
permissions (`.agents/knowledge/github-checks.md`). The dev container
holds no credential beyond the `gh` login the user forwards and no host
mounts; session logs stay in the container user's home and are not
collected. The strict isolation tier therefore applies to no role here;
revisit this section before adding any scheduled or unattended agent run.

## Update this file when

- The maintainer changes the level, a condition, or a gate owner.
- The verification strength this level was priced against degrades: a
  required check is removed, behavioral tests are skipped by default, or
  revert stops being a complete rollback.
- The specification workflow changes what "approved", "frozen", or
  "archived" means, or a gate gains a recorded approval.
- `.claude/settings.json` gains or loses a deny entry, an unattended agent
  run is added, or the dev container gains a credential or a host mount.
