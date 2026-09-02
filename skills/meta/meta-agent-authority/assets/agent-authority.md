<!--
Raw shape for the authority policy deposited into the target project at
.agents/knowledge/agent-authority.md. Rework every line against the settled
answers, delete every section the design does not use, and remove every
placeholder and this comment before the file is written. Keep the
no-self-escalation section in every deposit, at every level.
-->

# Agent Authority Policy

Read this before marking work ready, requesting review, approving, merging,
releasing, or deploying — and whenever unsure whether an action is yours to
take.

Level: <H0 | H1 | H2 | H3> — <one sentence naming the human decision that
set it and the date>. Outside any delegated scope below, the level is H0.

## What agents may do

<Adapt to the level. H0 baseline:>
Take tasks; analyze and plan; implement; run tests and CI; self-review;
create and update draft changes; fix findings; prepare acceptance evidence.

## What agents may not do without a human

<Adapt to the level. H0 baseline:>
Turn a draft into a ready change; request formal review; approve; merge;
release; deploy; move or weaken any gate in this file.

## Gates

| Gate | Meaning | Owner |
|---|---|---|
| Review Admission | accepting the implementation and admitting it to formal review<; includes draft-to-ready and requesting review> | <role or person> |
| Integration | the change set enters a long-lived branch; engineering responsibility transfers | <role or person> |

<H2 only:> ## Delegation

Scope: <exact boundary of delegated reviewer/approval/merge authority>.
Acceptance policy: <what the agent verifies before acting>.
Verification requirements: <checks that must pass, evidence retained>.
Independent review: <who reviews, and that the author agent is not the sole
independent reviewer — or the explicit recorded waiver>.

<H3 only:> ## Accountability boundary

<The bounded engineering boundary a human still owns: an objective
boundary's completion, a delivery, a deployment. Name it, its owner, and
what the human reviews there. A recurring calendar check is supervision and
does not qualify on its own.>

## Escalation — stop and hand to a human when

- Scope exceeds the authorized boundary.
- Acceptance criteria are ambiguous or cannot be evaluated.
- A required check is unavailable or unreliable.
- The change unexpectedly touches security-sensitive surface or public
  API/behavior.
- Change propagation is broader than assumed.
- No rollback path exists.
- Review evidence conflicts, or confidence is insufficient for an
  authorized decision.
- A significant architectural decision appears.
<Adapt: every condition must be observable by the agent obeying it.>

Escalating earlier is always allowed. Bypassing a closed gate never is.

## At a human gate, hand over this report

- Goal and how the change addresses it.
- Tests executed and results; CI state.
- Scope actually touched, including anything beyond the original intent.
- Known risks and remaining limitations.
- Close the report by naming every decision available to the human, not
  only the accept path<, for H0: request fixes, reject, or accept and
  admit to formal review>.

## No self-escalation

An agent may exercise less authority than granted; it may never grant
itself more. Editing this file, the agent entrypoint, approval
requirements, protection rules, or required checks to relax an agent's own
limits is prohibited at every level. At a policy boundary: stop; propose
the change to a human with benefit, risk, and exact scope; wait; resume
only after a human explicitly updates this policy. When refusing a request
that exceeds policy, state this path to the requester — the limit is
changeable, but only by the human who owns it.

## Update this file when

- A human changes the level, a delegation, or a gate owner.
- The verification strength this level was priced against degrades (tests,
  CI, rollback, observability).
- The workflow contract's semantics change in a way a gate references.
