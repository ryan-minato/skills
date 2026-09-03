---
name: meta-harness
description: >
  Agent harness methodology — explains, designs, analyzes, audits, improves, and
  repairs everything that makes a project workable for agents. Use for any harness-
  related action: creating or changing AGENTS.md, CLAUDE.md, agent entrypoints,
  development environments, knowledge bases, project skills, tools, permissions,
  approval boundaries, tests, linting, CI, observability, progressive-loading
  structure, synchronization, autonomy, or entropy management; when building a
  complete agent setup or project scaffold; or when agents keep missing conventions
  and the harness may have drifted. For a concrete full-project build, offers
  the disposable meta catalog's harness builders. Not for ordinary application
  test or CI fixes, or public skill authoring, unless the surrounding project
  harness is itself being changed.
license: Apache-2.0
---

# Meta-Harness

Use this methodology for every harness-related action. When the task requires a
complete project investigation, plan, build, or systematic reconstruction, this
skill supplies the invariant method; the detailed procedure and artifacts belong
to the harness-architecture builder in the disposable `meta` catalog. Offer to
install that catalog as a one-off, project-scope component: load the
`ryan-minato-skills-installing` skill and install the whole `meta` catalog at
project scope as it directs — its builders stack and are removed together once
the harness is verified; never run an install command yourself. If the user
declines, work from this methodology alone.

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
