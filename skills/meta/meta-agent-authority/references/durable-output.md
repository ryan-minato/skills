# Durable Output

Read on every build before depositing, and when defining what an agent does
at a gate or a policy boundary.

## Where the policy lives

Deposit the adapted policy at `.agents/knowledge/agent-authority.md` (or the
project's existing knowledge location, recorded in the hand-off). Add one
event-triggered pointer to the agent entrypoint:

> Read `.agents/knowledge/agent-authority.md` before marking work ready,
> requesting review, approving, merging, releasing, or deploying — and
> whenever unsure whether an action is yours to take.

The policy is the single source of truth on agent authority. The workflow
contract may point to it; nothing restates it.

## The acceptance-evidence report

At a human gate the agent hands over a decision-ready report instead of
asking what to do next. It contains, briefly and verifiably:

- What the goal was and how the change addresses it.
- Tests executed and their results; CI state.
- Scope actually touched, including anything beyond the original intent.
- Known risks and remaining limitations.
- The decisions now available to the human (for the default level: request
  fixes, reject, or accept and admit to formal review).

Deposit this format inside the policy file so runtime agents produce it
without this builder.

## The policy-change proposal

When an agent reaches a policy boundary — including any temptation to edit
the policy, entrypoint, approval requirements, protection rules, or
required checks in its own favor — the deposited procedure is:

1. Stop the action that crossed the boundary.
2. Propose the policy change to a human: what limit was reached, the
   benefit of changing it, the risk, and the exact scope of the proposed
   change.
3. Wait. Only a human updates the policy.
4. Resume under the new policy only after it is explicitly updated.

## Survival rules

- The deposited file carries the may/may-not lists, both gates with named
  owners, the escalation conditions, the report format, the proposal
  procedure, and the no-self-escalation rule — the runtime rules must not
  depend on this builder existing.
- Platform vocabulary for actions: the policy names what an agent may and
  may not do as the platform's own operations (for GitHub: `gh pr ready`,
  requesting review, approving, merging, publishing a release), and defines
  Review Admission and Integration in the file. A term only this builder
  defines must not appear without its definition.
- Platform enforcement is welcome but separate: what a platform can enforce
  is configured by the platform lifecycle builder and recorded there with
  its enforcement tier. Policy is authoritative even where enforcement is
  absent — "the platform would let me" never overrides it.
- No trace of the builder: the deposited file never carries this skill's
  disposable marker, name, or paths.
