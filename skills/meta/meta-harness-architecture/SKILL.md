---
name: meta-harness-architecture
description: >-
  Disposable builder skill (delete after the harness is built): the
  architecture practice manual for agent harness layers — entrypoints and
  architecture documents, knowledge files and external backends, project
  skills (warrant, design, retrofit), sync and entropy mechanisms, periodic
  reclamation, multi-agent topology, advanced autonomy, and public-convention
  files — with the design axes and starting-shape assets for each. Use when
  a harness build selects an artifact and needs the practice for that layer,
  when auditing what an existing harness gets wrong, or when asked how a
  specific harness layer should be designed. Not the build workflow itself:
  investigation, clarification, planning, approval, build order,
  verification, and cleanup belong to the entry builder
  meta-harness-building.
license: Apache-2.0
---

# Harness Architecture Manual

The practice behind every layer of an agent harness, loaded one layer at a
time by whatever workflow is building it — normally `meta-harness-building`,
which owns investigation, planning, approval, verification, and cleanup and
returns here for each artifact. Every durable decision and artifact must land
in the target project; this manual and the conversation are not part of the
finished harness.

## Harness Methodology

A harness is everything agent-visible that helps agents meet the project's
expectations: the environment they run in, the information and tools they can
reach, the constraints on what they produce, and the feedback that lets them
correct mistakes. It works in two directions: feed-forward context expands and
guides capability before action; feedback from tests, linting, CI, runtime
observation, review, and users narrows results toward the intended target.

### Research before design

Inspect discoverable project facts before asking questions or choosing artifacts.
Read the goal, structure, stack, commands, checks, deployment environment, existing
instructions, and maintained sources of truth. Ask only for team decisions,
permissions, risk tolerance, and preferences that the project cannot reveal.
Preserve working choices unless the user explicitly requests a migration.

### Write for agents and progressive loading

Agent-first files put facts and direct instructions first, name their source of
truth, state when they load, define checkable completion boundaries, and contain no
unresolved placeholders. Public-convention files such as README, CONTRIBUTING, and
SECURITY remain human-first because their audience is broader.

Split content by loading behavior, not by abstract topic. Keep information needed
on every task in the entrypoint; put conditional detail in a reference behind a
precise when-to-read pointer; encode repeated, fragile, ordered, or branchy work as
a project skill; put deterministic repeated logic in a script. A split that does
not reduce what loads for a task only adds navigation cost.

AGENTS.md is the map for progressive loading, not an exhaustive manual. It states
the project's purpose, always-applicable constraints, validation entrypoints, and
the exact conditions for reading deeper material. Aim for about 100 lines. A light
project whose entrypoint is the whole harness may approach 200 lines, but global
safety rules stay visible even when that exceeds the budget.

### Calibrate thickness to the project

Do not assign one maturity level to the whole harness. Rate each capability or
constraint layer omitted, light, medium, or thick according to project lifecycle,
error cost, maintenance horizon, environment isolation, team size, and agent
autonomy. Short-lived, exploratory, or closely supervised work benefits from a
thin harness; long-lived, high-risk, or agent-driven work justifies stronger
machine feedback and more durable knowledge. Build only what a current need earns.

### Manage harness entropy

Implementation and harness artifacts decay as projects change. Agent-driven
development compresses the change cycle, so contradictions, stale commands,
unreachable knowledge, and excessive constraints can accumulate faster than in a
human-only workflow. Every harness design, audit, or improvement must account for
the project's lifecycle and choose an entropy strategy.

There are three complementary controls:

1. Reduce complexity and keep a human in the loop. Leave more judgment to the user
   and agent, reducing the surface that can drift.
2. Add real-time consistency controls. Name concrete forward triggers from
   implementation changes to harness updates and reverse triggers from harness
   edits back to the implementation facts that must be checked.
3. Run periodic entropy reclamation. Inspect harness-to-harness and
   harness-to-implementation consistency for stale paths, dead commands,
   duplication, contradictions, unreachable content, and unjustified thickness.

Use simplification as the primary control for short-lived, actively supervised
projects. Long-lived or agent-driven projects need real-time consistency controls;
long-lived, high-change projects combine those with periodic reclamation.

### Make the method durable

Every rule future agents need must land in project-visible files, registered tools,
or reachable sources. Conversation-only decisions do not exist for the next run.
Record team workflow choices without inventing automation, keep one source of truth
per fact, and verify that feed-forward guidance and feedback mechanisms remain
reachable after any temporary harness-building skills are removed.

## Design axes

Rate every capability and constraint layer omitted, light, medium, or thick:

- Capability: environment, information tools, workflow tools, capability
  tools.
- Constraint: target, implementation, quality, workflow, repository safety.

Choose separately:

- Evolution mode: fixed, compromise, or self-evolving.
- Agent topology: single agent by default; multiple roles only when real
  context boundaries and coordination support justify them.
- Sync family: entrypoint/knowledge rules or dedicated project skills.
- Model class: weaker/local models need thinner documents and more mechanical
  lookup aids.
- Entropy controls: simplification, real-time consistency, periodic
  reclamation, or the lifecycle-appropriate combination.

Do not collapse these axes into a maturity level. Record one value and one
reason for every decision. Read [references/layers.md](references/layers.md)
when designing a full harness or recalibrating several layers.

## Load by artifact

Load only the reference for the artifact being audited or built:

- Auditing, repairing, slimming, or de-conflicting an existing harness:
  read [references/audit.md](references/audit.md) first.
- Entrypoints and architecture documents: read
  [references/entrypoint.md](references/entrypoint.md); when architecture
  detail exceeds the entrypoint budget, also read
  [references/entrypoint-offload.md](references/entrypoint-offload.md).
- Agent knowledge, including external backends: read
  [references/knowledge.md](references/knowledge.md). Read
  [references/knowledge-external-backend.md](references/knowledge-external-backend.md)
  before choosing a remote source of truth, or
  [references/knowledge-reorganize.md](references/knowledge-reorganize.md)
  when existing knowledge is scattered or structurally mixed.
- Durable project skills: read
  [references/project-skill.md](references/project-skill.md). Read
  [references/project-skill-warrant.md](references/project-skill-warrant.md)
  when the warrant is unclear, or
  [references/project-skill-retrofit.md](references/project-skill-retrofit.md)
  when an existing project skill misfires or has drifted.
- Real-time consistency or periodic reclamation: read
  [references/maintenance.md](references/maintenance.md), then the selected
  mechanism family in [references/sync-entrypoint.md](references/sync-entrypoint.md)
  or [references/sync-project-skill.md](references/sync-project-skill.md).
  Read [references/periodic-reclamation.md](references/periodic-reclamation.md)
  when the lifecycle calls for a scheduled entropy review. For every changing
  relationship choose one owner; a real-time control is bidirectional, and
  two mechanisms for one concern is one too many. Failure-driven
  self-updates require the advanced-autonomy approval, audit, and rollback
  controls.
- More than one agent role: read
  [references/multi-agent.md](references/multi-agent.md) before choosing it.
- README, LICENSE, SECURITY, CONTRIBUTING, or architecture documents: read
  [references/public-files.md](references/public-files.md) before changing
  them.
- Self-evolution, persistent memory, unattended operation, autonomous
  routing, or multiple agent roles: follow
  [references/advanced-autonomy.md](references/advanced-autonomy.md).

## Using the assets

Use the corresponding [assets](assets/) only as starting shapes. Rework every
line against inspected facts, delete inapplicable sections, and remove every
placeholder. AGENTS.md is the discovery root: skills and registered tools may
self-announce, but every knowledge or reference file needs an explicit
when-to-read pointer.

For scripts, tests, linters, CI, hooks, task runners, and framework settings,
implement the approved feedback and safety layers. Custom checks must explain
what failed, why it matters, and the likely fix.

## Gotchas

- Name categories in durable guidance; enumerate products only where the
  project has actually selected one.
- A document with no discovery path does not exist for future agents.
- A generic “keep docs in sync” rule has no actionable trigger.
- Templates deployed unchanged are unfinished decisions, not scaffolding.
