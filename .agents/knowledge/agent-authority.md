# Agent Authority Policy

Read this before marking work ready, requesting review, approving, merging,
releasing, or deploying, and whenever authority is uncertain.

Level: H1 — Autonomous Author. The maintainer selected this level on
2026-09-02 so an agent may prepare a complete change for review while a human
retains integration responsibility.

## What agents may do

After the goal, scope, non-goals, constraints, and acceptance criteria are
complete, an agent may plan, implement, test, self-review, create atomic
commits, push an authorized branch, create and update a draft pull request,
mark it ready, request review, respond to review, and prepare acceptance
evidence.

Every remote write still requires explicit authorization in the current
conversation. Authority under this policy never supplies that authorization.

## What agents may not do

An agent may not approve, merge, release, deploy, change the authority level,
or weaken an approval requirement, protection rule, or required check. It may
not mark work ready while the specification is incomplete or a required check
is failing, unavailable, or unreliable.

## Gates

| Gate | Meaning | Owner |
|---|---|---|
| Review Admission | Accept the implementation as complete enough for formal review, including draft-to-ready and requesting review. | The author agent, under this policy. |
| Integration | Decide that the change enters the long-lived branch and assume engineering responsibility for it. | The human maintainer. |

## Escalation

Stop and hand control to the human maintainer when:

- The requested scope or propagation exceeds the agreed boundary.
- Acceptance criteria are ambiguous or cannot be evaluated.
- A required check is unavailable, unreliable, or conflicts with other
  evidence.
- The change unexpectedly affects security-sensitive behavior, public skill
  behavior, or compatibility.
- No practical revert or rollback path exists.
- A significant architectural decision was not settled in the specification.
- A remote write was not explicitly authorized for the current conversation.

Escalating earlier is always allowed. Bypassing a closed gate never is.

## Acceptance-evidence report

At the integration gate, report:

- The goal and how the change addresses it.
- Tests executed and their results, including CI state.
- The scope actually touched and any deviation from the agreed scope.
- Known risks and remaining limitations.
- The human's available decisions: request fixes, reject the change, or merge
  it using the repository's approved method.

## No self-escalation

An agent may exercise less authority than granted; it may never grant itself
more. Editing this file, the agent entrypoint, approval requirements,
protection rules, or required checks to relax an agent's own limits is
prohibited. At a policy boundary, stop; propose the change to the human with
its benefit, risk, and exact scope; wait; and resume only after the human
explicitly updates this policy.

## Update this file when

- A human changes the authority level or either gate owner.
- Test, CI, rollback, or review strength changes enough to alter the price of
  H1.
- The workflow contract changes a semantic used by either gate.
