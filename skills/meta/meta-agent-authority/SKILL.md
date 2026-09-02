---
name: meta-agent-authority
description: >-
  Disposable meta-skill (delete after the harness is built): sets the project
  policy for what coding agents may do on their own — whether an agent may
  mark a PR or MR ready, request review, approve, merge, or release — via the
  H0–H3 authority levels, the review-admission and integration gates,
  escalation conditions, and the no-self-escalation rule. Use when deciding or asked
  whether the agent can merge, release, or send its work to review by itself;
  when the user wants an agent to have more (or less) autonomy as official
  project policy; or when an agent marked work ready, requested review, or
  merged without asking and the project needs rules so it does not recur. Not
  for harness permission allowlists or tool approval settings (configuration,
  not project policy), platform token scopes or branch protection (the
  platform lifecycle builders), or how finely a human phrases task
  instructions.
license: Apache-2.0
---

# Human–Agent Authority Policy

Decide, with the human developers, which project-management actions their
coding agents may take on their own, then leave that policy in the target
project where every future agent will read it. The default answer is
conservative: agents prepare complete work; humans admit it to review and
answer for what enters long-lived branches.

## Non-negotiable boundaries

- **No self-escalation, ever.** An agent may voluntarily exercise less
  authority than granted; it may never grant itself more. Editing the agent
  entrypoint, this policy, approval requirements, protected-branch rules, or
  required checks to relax its own limits is prohibited at every level,
  including the most autonomous. On reaching a policy boundary the agent
  stops, proposes the policy change with benefit, risk, and scope, and
  resumes only after a human explicitly updates the policy. This rule is
  itself part of the deposited policy.
- Planning autonomy is not project-management authority. Whether a human
  directs work step by step or hands over a whole goal changes implementation
  autonomy only; it never implicitly changes review-admission, approval, or
  merge authority.
- Effective authority is the minimum of governance policy, harness
  capability, and platform capability. A token that technically can merge
  grants nothing; capability discovered downstream never raises what policy
  granted.
- Requesting review is a management action, not an execution step. It spends
  reviewer attention, enters formal queues, and asserts the change deserves
  formal review — so by default a human decides it, after their own
  acceptance of the agent's work.
- Self-review is author-side validation. It never counts as independent
  review; an independent reviewer is a human or a separately authorized
  reviewer agent that is not the author. The user may waive independence,
  but only as an explicit recorded decision.
- A human accountability gate may move later — from review admission to
  integration to a bounded delivery boundary — but may never disappear.
  Every reduction in human intervention is purchased with stronger
  specification, verification, containment, observability, rollback, and
  auditability; refuse to record a level whose price is unpaid unless the
  user explicitly accepts the recorded gap.

## Workflow

### 1. Establish the ground

Read the project's workflow contract at
`.agents/knowledge/project-workflow.md` if present; its settled semantics —
acceptance, ownership, objective boundaries, deliveries — are the anchors
authority attaches to. Its absence blocks nothing: a project with no tracker
still needs an authority policy. Then evidence the project's real
verification strength yourself — test coverage and trustworthiness, CI
gates, rollback and revert paths, observability — because these facts price
the higher levels, and they are inspected, never asked.

Done when: the contract's settled decisions (or its absence) and the
project's verification strength are recorded as facts.

### 2. Place the two gates

Every change set that enters a long-lived branch is an engineering
responsibility unit. Two gates structure its path — both human-owned at the
default level; a rising level moves an owner downstream, never off the path:

- **Review Admission** — accepting the implementation and admitting it to
  formal review. By default this includes turning a draft into a ready
  change *and* requesting review; split them only if the project records in
  writing that ready is a technical state that does not enter a review
  queue.
- **Integration** — the final decision that the change set enters the
  long-lived branch, and with it the engineering responsibility.

Done when: both gates have a named owner at the settled level — a human role
or person (never "the team"), or the agent under a delegation this policy
records — and a named human still owns a point on the path: a gate, or at
the most autonomous level the accountability boundary.

### 3. Select the authority level

Read [authority-profiles.md](references/authority-profiles.md) and put the
choice to the user as a structured single-select, leading with **H0 — the
default** — and its reason. The levels are presets, not maturity grades;
higher is not better. For each level above H0, verify its purchase price
against the facts from step 1 before offering it as recommendable; where the
price is unpaid, present the gap and refuse to record the level unless the
user explicitly accepts the gap in writing.

Done when: the level is settled, its required delegations and waivers are
each an explicit user decision, and any accepted gap is recorded.

### 4. Settle the escalation policy

Autonomous work halts and returns to a human gate when any of these fire:

- Scope exceeds the authorized boundary.
- Acceptance criteria are ambiguous or cannot be evaluated.
- A required check is unavailable or unreliable.
- The change unexpectedly touches security-sensitive surface or public
  API/behavior.
- Change propagation turns out broader than assumed.
- No rollback path exists for the change.
- Review evidence conflicts, or confidence is insufficient for a decision
  the level authorizes.
- A significant architectural decision appears.

Adapt the list to the project — every condition must be observable by the
agent that must obey it — and confirm it with the user. An agent may always
escalate earlier than required; it may never bypass a gate that a condition
has closed.

Done when: the escalation list is confirmed and each condition is
observable.

### 5. Deposit and verify

Read [durable-output.md](references/durable-output.md) on every build. Adapt
[assets/agent-authority.md](assets/agent-authority.md) to the settled
answers; every placeholder and inapplicable section must be gone, and the
deposited file must carry the may/may-not lists, both gates with owners, the
escalation list, the acceptance-evidence report format, the policy-change
proposal procedure, and the no-self-escalation rule verbatim in substance.

Verify by removal simulation: with this builder deleted, a future agent must
be able to state its own limits, both gate owners, and what to do at a
policy boundary from target-project files alone. Check the level's
requirements one last time — an H2 record without delegation scope and a
reviewer-independence decision, or an H3 record whose accountability
boundary is a recurring calendar check rather than a bounded engineering
boundary, is not depositable. Confirm the deposited file carries neither
this skill's disposable marker nor its name or paths.

Then hand off by name: the platform lifecycle builder for the evidenced host
enforces what the platform can enforce and maps the rest to convention. If
it is not installed, install it from
https://github.com/ryan-minato/skills.git — recommend the whole `meta`
catalog, since its builders stack:

    npx skills add ryan-minato/skills

## Gotchas

- Green CI is evidence, not acceptance. At the default level the agent's
  work ends at a draft change plus an acceptance-evidence report — never at
  ready, review requested, or merged.
- "A human looks things over every Friday" is supervision, and worth
  keeping — but it is not an accountability boundary and cannot replace an
  integration gate or a bounded delivery boundary.
- A delegated reviewer agent is a second authorization, not a property of
  the author agent. The author being excellent at self-review changes
  nothing about independence.
- Authority granted for one scope does not travel: delegation names its
  boundary, and outside it the level is H0 again.
- The policy governs agents in this project regardless of which harness runs
  them; harness-side permission systems implement the policy, they do not
  define it.
