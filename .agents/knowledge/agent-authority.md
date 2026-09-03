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

- the OpenSpec change is approved and archived, or the pull request
  carries `Spec: none — <reason>` (see `.agents/knowledge/spec-workflow.md`);
- `just check` passes locally and every required check is green;
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

- Goal, and how the change addresses it (link the change record).
- Tests executed and results; the state of every required check.
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

## Update this file when

- The maintainer changes the level, a condition, or a gate owner.
- The verification strength this level was priced against degrades: a
  required check is removed, behavioral tests are skipped by default, or
  revert stops being a complete rollback.
- The specification workflow changes what "approved" or "archived" means.
