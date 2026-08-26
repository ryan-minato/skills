---
name: meta-harness-architecture
description: >-
  Disposable meta-skill (delete after the harness is built): investigates,
  plans, builds, audits, or systematically restructures a project's complete
  agent harness. Use together with the core meta-harness methodology when a
  concrete project needs AGENTS.md or CLAUDE.md, architecture and knowledge
  files, project skills, development and feedback tooling, autonomy boundaries,
  synchronization, or entropy management implemented as one coherent system.
  Also use when an existing harness has become stale, contradictory, invisible,
  or too thick. Not for an isolated artifact edit that does not affect the
  surrounding harness.
license: Apache-2.0
compatibility: Live catalog discovery requires Python 3.11+ and network access.
---

# Meta-Harness Architecture

This is a complete, self-contained harness builder. It carries the full core
methodology so it still works when the durable `meta-harness` skill is absent.
Every durable decision and artifact must land in the target project; this
disposable skill and the conversation are not part of the finished harness.

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

## Route by task

- Read [references/audit.md](references/audit.md) before auditing, repairing,
  slimming, or de-conflicting an existing harness.
- Read [references/advanced-autonomy.md](references/advanced-autonomy.md) when
  self-evolution, persistent memory, unattended operation, autonomous routing,
  or multiple agent roles are requested or seriously considered.
- Read [references/layers.md](references/layers.md) when designing a full
  harness or recalibrating several layers.
- Read [references/multi-agent.md](references/multi-agent.md) before choosing
  more than one agent role, and
  [references/public-files.md](references/public-files.md) before changing
  README, LICENSE, SECURITY, CONTRIBUTING, or architecture documents.
- Otherwise continue with the workflow below and load builder references only
  when their artifact is selected.

## Architecture workflow

### 1. Investigate the project

Inspect the repository before asking questions. Record the target and exposed
interfaces; layout and dependency skeleton; technology stack; development,
test, deployment, and CI environments; validation commands; error cost;
lifecycle and maintenance horizon; existing agent files and tools; version-
control workflow; and maintained sources of truth.

Then resolve team facts that the repository cannot prove: team and review
structure, confidentiality boundary, agent involvement, approvals, delegated
external actions, and where findings reach humans.

Done when: every fact is either supported by inspected evidence or appears as
a concrete unanswered user decision.

### 2. Audit what already exists

For every entrypoint, knowledge file, project skill, registered tool, script,
check, CI workflow, and framework configuration, record how future agents find
it, its load condition, source of truth, and keep-current mechanism. Classify
problems as stale, duplicated, contradictory, invisible, missing, excessive,
or orphaned. Preserve working choices and plan only justified gaps.

Done when: every existing artifact has an evidence-backed keep, update, move,
merge, split, reconnect, thin, or remove disposition.

### 3. Design the independent axes

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
reason for every decision.

When the needed builder is unknown, run the bundled
[live discovery script](scripts/discover.py):

```bash
python3 scripts/discover.py --catalog meta
```

It reads the current repository inventory rather than carrying a static catalog.

### 4. Plan before building

Present a harness plan that groups files by layer and states:

- the entrypoint map and every when-to-read route;
- feed-forward context and feedback mechanisms;
- each layer's thickness and evidence;
- delegated agent actions versus human approvals;
- implementation-to-harness and harness-to-implementation sync ownership;
- the entropy strategy and periodic routine, if any;
- exact verification and completion criteria;
- assumptions and unresolved user decisions.

Do not create or change target artifacts until the user approves this plan.
If the project already has a harness, plan a gap repair rather than a rebuild.
Use [assets/harness-plan-template.md](assets/harness-plan-template.md) only as
a shape, then remove every fill instruction and inapplicable row.

### 5. Build approved artifacts

Load only the references for selected artifacts:

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
  when the lifecycle calls for a scheduled entropy review.
- Advanced autonomy: follow
  [references/advanced-autonomy.md](references/advanced-autonomy.md).

Use the corresponding [assets](assets/) only as starting shapes. Rework every
line against inspected facts, delete inapplicable sections, and remove every
placeholder. AGENTS.md is the discovery root: skills and registered tools may
self-announce, but every knowledge or reference file needs an explicit
when-to-read pointer.

For scripts, tests, linters, CI, hooks, task runners, and framework settings,
implement the approved feedback and safety layers. Custom checks must explain
what failed, why it matters, and the likely fix.

Done when: every approved artifact exists, is reachable from the entrypoint,
and contains only real project information.

### 6. Install entropy controls

For every changing relationship, choose one owner. A real-time control is
bidirectional: a concrete implementation change triggers the dependent
harness update, and a harness edit names the implementation evidence to
re-check. Never install two mechanisms for the same concern.

For long-lived, high-change projects, add a periodic reclamation workflow or
project skill that checks stale paths, dead commands, duplication,
contradictions, unreachable content, entrypoint growth, and unjustified
constraint thickness. Failure-driven self-updates require the advanced-
autonomy approval, audit, and rollback controls.

### 7. Verify and hand off

Verify that:

- future agents can reach every rule and knowledge source from the entrypoint;
- AGENTS.md remains a map near its 100-line target;
- local links resolve and documented commands run;
- feed-forward context and feedback mechanisms form a usable loop;
- each sync concern has one owner in both directions;
- the entropy strategy matches the lifecycle;
- no durable target artifact carries the disposable marker;
- every temporary builder can be removed without losing a rule.

Report every artifact changed, its role, validation results, and any human or
platform action still required. If the harness is complete and verified,
route explicit cleanup requests to `meta-disposal`; never treat the earlier
build request as deletion consent.

## Gotchas

- A plan is not approval to delete, publish, change remote settings, or grant
  agent autonomy.
- Name categories in durable guidance; enumerate products only where the
  project has actually selected one.
- A document with no discovery path does not exist for future agents.
- A generic “keep docs in sync” rule has no actionable trigger.
- Templates deployed unchanged are unfinished decisions, not scaffolding.
