---
name: meta-workflow-design
description: >-
  Disposable meta-skill (delete after the harness is built): designs how a
  project tracks, plans, and accepts work — settling with the humans
  which structures the project actually earns (work items, hierarchy,
  objectives, timeboxes, priorities, planning views) and depositing a
  platform-neutral workflow contract the platform builders then
  express. Use when asked how a project or team should track, plan, organize,
  or prioritize its work; whether it needs sprints, epics, boards, milestones,
  or priorities; when a solo project grows into a team and the process must be
  rethought; or when an issue tracker or backlog is a mess and the management
  process needs redesign ("帮我设计项目管理流程"). Not for expressing the
  contract on GitHub or GitLab (the platform builders), choosing a branching model
  (meta-git-branching), agent permissions (meta-agent-authority), or one-off
  issue and pull-request operations.
license: Apache-2.0
---

# Project Workflow Contract

Establish, with the human developers, the smallest project management model
that expresses this project's real coordination facts, then leave it in the
target project as a platform-neutral contract. The contract is what the
platform lifecycle builders map onto a host; it must read correctly with no
platform in mind and survive this builder and this conversation.

## Non-negotiable boundaries

- The contract is platform-neutral. No platform object — a board product, an
  epic object, a label system, a sprint feature — may appear in the model or
  in the deposited contract; platform names appear only in the sentence that
  hands off to the next builder. Platform capability never implies management
  necessity: nothing enters the model because some host has a feature for it.
- No entity without necessity. Every work-item kind, hierarchy level, status,
  attribute, planning surface, or boundary must answer the deletion test —
  what management ability is lost if it is removed — and its benefit must
  exceed its creation, maintenance, classification, synchronization, and
  cognitive cost. Start minimal and formalize progressively as real
  coordination problems appear, never by enabling everything up front.
- Facts are the agent's job; decisions are the human's. Investigate everything
  the repository, its history, and its platform can prove before asking, and
  ask only for intent, collaboration, and governance decisions no inspection
  can answer.
- A recommendation is not a decision. Attach one reasoned recommendation to
  every question, and let the user overrule it. When a choice carries a cost
  the user may not see, state the trade-off once and accept their confirmed
  answer; a settled decision becomes a downstream constraint that later
  phases and later builders must not reopen or quietly bypass.
- Nothing lands before agreement. Keep the working design in the
  conversation; write into the target project only after the user approves
  the complete design summary. Preserve working conventions — redesign is
  not permission to migrate a system that already serves the team.

## Workflow

### 1. Inspect the project

Establish the facts without asking: repository purpose and audience, single
repo or monorepo, stack, contributor count and cadence from history, release
and tag history, existing CI and checks, current tracker conventions and
templates, existing planning or governance files, which hosting platform is
in use and which of its features are already active, and any existing agent
entrypoints. Sort everything into three lists: known facts with evidence,
unknown facts still discoverable, and decisions only a human can make.

Done when: every question the repository can answer is answered with
evidence, and the human-decision list contains nothing an inspection could
have resolved.

### 2. Establish intent — the first frontier

Read [design-tree.md](references/design-tree.md) before the first question,
and run every questioning round by it. The first frontier asks only the most
upstream human decisions:

1. The project's driver — what a period of work is organized around. Read
   [profiles.md](references/profiles.md) to match the evidenced facts to a
   base profile and to phrase this question with a recommendation.
2. Expected collaboration scale and shape over the design horizon.
3. Overlay characteristics, asked as a multi-select. Read
   [overlays-and-propagation.md](references/overlays-and-propagation.md)
   when the facts suggest any overlay or any change propagation beyond the
   repository itself.
4. How the project's changes propagate to consumers, when step 1's
   inspection has not already proven it.
5. Whether the team intends agents to work autonomously here — record the
   intent only; the design belongs to `meta-agent-authority`.

Done when: base profile, overlays, and change propagation are settled by the
user (or confirmed from evidence), each with its selecting fact recorded.

### 3. Design the workflow — the second frontier

Only now, with the profile settled, decide which semantics the project earns.
Read [management-model.md](references/management-model.md) for the semantic
catalog, and ask only the questions the profile leaves genuinely open — a
semantic the profile already requires, or one nothing in the facts argues
for, is not a question. Typical open decisions: whether any objective
boundary exists, whether work needs a hierarchy, whether priority or status
would change behavior, whether a timebox matches how the team actually
plans, and whether any planning surface is worth its upkeep.

For every semantic proposed, present the management problem it solves in
this project and its deletion-test answer; for every semantic omitted,
record the trigger that would justify enabling it later. Actively challenge
structure the user requests without a supporting fact — deep hierarchies,
long status ladders, ceremonial timeboxes, a planning surface for a dozen
work items — exactly once, with the cost stated, then follow their decision.

Done when: every enabled semantic has a deletion-test answer, every omitted
semantic has a re-enable trigger, and no open question remains whose answer
a settled decision or an evidenced fact already determines.

### 4. Present the design and get approval

Present one compact design summary: base profile, overlays, change
propagation, enabled semantics with their justifications, intentionally
omitted semantics with their triggers, the work-decomposition rule, the
objective and timebox policy, the planning policy, and what remains for the
governance and platform builders. Offer the decision as a structured choice —
apply, revise, or design only — and stop there without an explicit apply.

### 5. Deposit the contract

Read [durable-output.md](references/durable-output.md) on every build. Adapt
[assets/project-workflow.md](assets/project-workflow.md) to the settled
answers: it is a raw shape, and every placeholder and inapplicable section
must be gone. Wire the entrypoint pointer with the events that trigger
reading the contract.

Done when: the contract lives in the target project; every enabled semantic
in it carries its deletion-test answer and every omitted semantic its
trigger; and this exact platform-vocabulary check over the deposited file
returns nothing:

    grep -inwE 'github|gitlab|jira|epics?|sprints?|boards?|labels?|milestones?|iterations?|backlog|tickets?' <file>

A hit that looks platform-neutral is still a hit: rewrite the sentence in
contract vocabulary (Objective Boundary, Timebox, Planning Surface) rather
than waiving the check — this skill's own references phrase every semantic
without those words.

### 6. Verify and hand off

Simulate removal: with this builder deleted, the next agent must be able to
name the profile, the enabled and omitted semantics, and the decomposition
rule from target-project files alone. Confirm the deposited file does not
carry this skill's disposable marker, name, or paths.

Then hand off by name, in order. Governance next: design human–agent
authority with `meta-agent-authority`. Platform expression after that: run
the lifecycle builder for the evidenced host platform. If either is not
installed, install it from https://github.com/ryan-minato/skills.git —
recommend the whole `meta` catalog, since its builders stack:

    npx skills add ryan-minato/skills

If the user declines, record in the hand-off which decisions remain
unexpressed.

## Gotchas

- A platform having a feature is the weakest possible argument for a
  semantic. The model is derived from coordination facts; expression comes
  later and may lawfully be a downgrade.
- A planning surface is a view, never the record. Deleting one must lose no
  project fact; anything that would be lost belongs in tracked work, change
  requests, deliveries, or decision records instead.
- "We might need it later" is not a deletion-test answer. Record the trigger
  under omitted semantics and add the entity when the trigger fires.
- A draft change describes work already happening; it is never a placeholder
  for planned future work. Future work is tracked work, ordered by priority
  or timebox — not a dependency edge, which asserts only that one item
  factually cannot proceed before another.
- Priority is what the team chooses to do first; severity is how bad a
  problem objectively is. Conflating them destroys both signals.
- An objective boundary says what will be achieved; a timebox says when work
  is planned to happen. A project may need either, both, or neither.
- Significant objective boundaries are created or confirmed by humans. An
  agent proposing one is fine; an agent inventing goals because they make
  its own planning convenient is not.
- The work-type attribute is named `Type` in the contract, whatever any
  platform calls its equivalent. Keep names stable so downstream mapping
  stays mechanical.
