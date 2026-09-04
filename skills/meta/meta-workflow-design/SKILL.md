---
name: meta-workflow-design
description: >-
  Disposable builder skill (delete after the harness is built): designs how a
  project tracks, plans, and accepts work — settling with the humans
  which structures the project actually earns (work items, hierarchy,
  objectives, timeboxes, priorities, planning views) — designing in a
  platform-neutral model and then recording the decisions in the project's
  own platform vocabulary for the platform builders to implement. Use when asked how a project or team should track, plan, organize,
  or prioritize its work; whether it needs sprints, epics, boards, milestones,
  or priorities; when a solo project grows into a team; or when an issue
  tracker or backlog is a mess and the management process needs redesign
  ("帮我设计项目管理流程"). Not for expressing the
  contract on GitHub or GitLab (the platform builders), choosing a branching model
  (meta-git-branching), agent permissions (meta-agent-authority), the spec
  workflow (meta-spec-workflow), or one-off issue and pull-request
  operations.
license: Apache-2.0
---

# Project Workflow Contract

Establish, with the human developers, the smallest project management model
that expresses this project's real coordination facts, then leave it in the
target project as a workflow file written in the project's own platform
vocabulary. The model is platform-neutral so that the design is not shaped
by whatever the host happens to offer; the deposited file is not, because
the agents that read it every day work on that host. The platform lifecycle
builders implement what the file names; it must survive this builder and
this conversation.

## Non-negotiable boundaries

- The model is platform-neutral; the deposit is not. Inspection, every
  question, and the design summary use the management model's vocabulary
  (Tracked Work, Change Request, Objective Boundary, Timebox, Planning
  Surface) so that no platform object — a board product, an epic object, a
  label system, a sprint feature — enters the design because the host has
  it. Platform capability never implies management necessity. Only at
  deposit, after approval, is every enabled semantic translated into the
  evidenced platform's object through the platform builder's mapping, and
  the deposited file then speaks that vocabulary and nothing else: a
  builder-defined noun that survives into it is a defect the step-5 check
  catches.
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
- Disposable builders never enter a commit. Before the first commit of the
  build, add every skill directory whose description opens with
  `Disposable builder skill (delete after the harness is built):` to
  `$(git rev-parse --git-path info/exclude)`, stage explicit paths, and read
  `git status` before each commit; a builder tracked before the build is
  reported, and its deletion lands with the disposal commit.

## Workflow

### 1. Inspect the project

Establish the facts without asking: repository purpose and audience, single
repo or monorepo, stack, contributor count and cadence from history, release
and tag history, existing CI and checks, current tracker conventions and
templates, existing planning or governance files, which hosting platform is
in use (the remote, the CI directory, the platform's template directories)
and which of its features are already active, and any existing agent
entrypoints. Sort everything into three lists: known facts with evidence,
unknown facts still discoverable, and decisions only a human can make. The
platform is a required fact: when nothing evidences it, it becomes the
first question of step 2, and no file is deposited until it is answered.

Done when: every question the repository can answer is answered with
evidence, and the human-decision list contains nothing an inspection could
have resolved.

### 2. Establish intent — the first frontier

Read [design-tree.md](references/design-tree.md) before the first question,
and run every questioning round by it. The first frontier asks only the most
upstream human decisions:

1. The hosting platform, only when step 1 could not evidence it — the
   deposit is written in its vocabulary, so nothing lands without it.
2. The project's driver — what a period of work is organized around. Read
   [profiles.md](references/profiles.md) to match the evidenced facts to a
   base profile and to phrase this question with a recommendation.
3. Expected collaboration scale and shape over the design horizon.
4. Overlay characteristics, asked as a multi-select. Read
   [overlays-and-propagation.md](references/overlays-and-propagation.md)
   when the facts suggest any overlay or any change propagation beyond the
   repository itself.
5. How the project's changes propagate to consumers, when step 1's
   inspection has not already proven it.
6. Whether the team intends agents to work autonomously here — record the
   intent only; the design belongs to `meta-agent-authority`.
7. Whether the team intends to work from written specifications — record
   the intent only; the level, the tooling, and the specification contract
   belong to `meta-spec-workflow`.

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

Present one compact design summary in the model's vocabulary — no platform
object names yet: base profile, overlays, change propagation, enabled
semantics with their justifications, intentionally omitted semantics with
their triggers, the work-decomposition rule, the objective and timebox
policy, the planning policy, and what remains for the governance and
platform builders. Offer the decision as a structured choice — apply,
revise, or design only — and stop there without an explicit apply.

### 5. Deposit the workflow file

Read [durable-output.md](references/durable-output.md) on every build. Then
translate: load the evidenced platform builder's semantic mapping
(`references/semantic-mapping.md` of `meta-github-workflow` or
`meta-gitlab-workflow`; when neither covers the platform, load the
`ryan-minato-skills-installing` skill and install the whole `meta` catalog
at project scope as it directs — never run an install command yourself — and
if no builder exists for the platform, name each object from the platform's
own documentation) and replace every enabled semantic with the platform
object the mapping gives it, including the recorded fallback and what it
loses; write every omitted semantic as the platform object deliberately not
used, with its trigger. Adapt
[assets/platform-workflow.md](assets/platform-workflow.md) to the settled
answers: it is a raw shape, and every placeholder and inapplicable section
must be gone. Wire the entrypoint pointer with the platform events that
trigger reading the file.

Done when: the file lives in the target project at
`.agents/knowledge/<platform>-workflow.md`; every object in use carries its
deletion-test answer and every object not used its trigger; and this exact
model-vocabulary check over the deposited file returns nothing:

    grep -inE 'objective boundar|timebox|planning surface|tracked work|change request|draft change|source marker|work hierarchy' <file>

A hit means a design word leaked into the file agents read daily: replace it
with the platform object (milestone, iteration, board, issue, pull or merge
request, tag) rather than waiving the check. Design vocabulary belongs in
the conversation and in this skill's references, never in the deposit.

### 6. Verify and hand off

Simulate removal: with this builder deleted, the next agent must be able to
name the profile, the platform objects in use and deliberately not used with
their triggers, and the decomposition rule from target-project files alone,
without ever meeting a term only this builder defines. Confirm the deposited
file does not carry this skill's disposable marker, name, or paths.

Then hand off by name, in order. Specification workflow first, when
spec-driven intent was recorded: settle the level, tool, and specification
contract with `meta-spec-workflow`. Governance next: design human–agent
authority with `meta-agent-authority`. Platform expression after that: run
the lifecycle builder for the evidenced host platform. If any is not
installed, load the `ryan-minato-skills-installing` skill and install the
whole `meta` catalog at project scope as it directs — its builders stack and
are disposed together; never run an install command yourself.

If the user declines, record in the hand-off which decisions remain
unexpressed.

When this builder runs under `meta-harness-building`, return there for the
closing step. When it runs alone, once the deposit is verified and before the
work goes to review, ask the user whether to delete the disposable builders
now — the build request is not deletion consent — and on that decision load
`meta-disposal`, which lists, confirms, and removes them. If the user
declines, leave the builders in place and out of every commit, and record it
in the handoff.

## Gotchas

- A platform having a feature is the weakest possible argument for a
  semantic. The model is derived from coordination facts; the platform
  object is named only at deposit and may lawfully be a downgrade.
- A planning surface is a view, never the record. Deleting one must lose no
  project fact; anything that would be lost belongs in tracked work, change
  requests, deliveries, or decision records instead.
- "We might need it later" is not a deletion-test answer. Record the trigger
  under omitted semantics and add the entity when the trigger fires.
- A draft change describes work already happening; it is never a placeholder
  for planned future work. A draft whose first content is a specification
  is work that has begun. Future work is tracked work, ordered by priority
  or timebox — not a dependency edge, which asserts only that one item
  factually cannot proceed before another.
- Priority is what the team chooses to do first; severity is how bad a
  problem objectively is. Conflating them destroys both signals.
- An objective boundary says what will be achieved; a timebox says when work
  is planned to happen. A project may need either, both, or neither.
- Significant objective boundaries are created or confirmed by humans. An
  agent proposing one is fine; an agent inventing goals because they make
  its own planning convenient is not.
- The work-type attribute is named `Type` in the model, whatever any
  platform calls its equivalent; the platform builders' mappings are indexed
  by the model's names, so keep them stable in the design and translate
  only at deposit.
