# Project Workflow Contract

Read this before creating, splitting, or closing tracked work, before
starting a change request, and before proposing any new management
structure.

Profile: Shared Infrastructure / Governance. This repository supplies
reusable instructions and scaffolds to agents and downstream projects.

Overlays: Template / Scaffold and Change-driven. Consumers copy installed
skills out of this repository, while small, self-contained changes often do
not need a separate tracked item.

Change propagation: Copy. A merge changes what future installations receive;
existing installations are not updated automatically. Acceptance therefore
requires checking installed behavior, compatibility, and rollback by revert.

## Enabled semantics

| Semantic | Meaning here | What is lost without it |
|---|---|---|
| Tracked Work | Record work whose problem, priority, ownership, or acceptance needs a lifecycle. Small changes may be explained only in their change request. | Important work could lose its context or priority before implementation. |
| Change Request | Every repository change uses an isolated change request. A draft is active work that is not yet admitted to formal review. | Review evidence and the exact change set would have no durable boundary. |
| Priority | One priority value orders tracked work when order matters. | The maintainer could not distinguish urgent work from the normal queue. |
| Area | Catalog areas identify affected public collections; Repository identifies harness-wide work. | Changes that span the library could not be routed or filtered consistently. |
| Acceptance | A change is accepted only when its stated criteria, repository checks, task-specific tests, and human integration decision agree. | Passing automation could be mistaken for product acceptance. |
| Automation | Deterministic checks enforce repository structure, safety, and pull-request policy. | Repeated mechanical review would drift between contributors and agents. |

## Intentionally omitted

| Semantic | Enable when |
|---|---|
| Work Hierarchy | Several independently owned or discussed items need one shared outcome boundary. |
| Dependency | Work regularly becomes blocked by another independently tracked item. |
| Objective Boundary | Several tracked items must converge on one human-confirmed outcome. |
| Timebox | The maintainer begins planning work into explicit start-and-end windows. |
| Planning Surface | Parallel work makes filtered tracked-work lists insufficient. |
| Status beyond open/closed | An intermediate state repeatedly changes what the maintainer does next. |
| Type | Classification would change routing or acceptance rather than only the intake questions asked. |
| Severity | Operational incidents require an impact measure distinct from priority. |
| Delivery and Source Marker | Consumers adopt named versions rather than the current repository state. |
| Deliberation and Decision Record | Multiple maintainers or hard-to-reverse decisions require a durable decision process. |

## Work decomposition

Create a separate tracked item only when it needs its own priority, owner,
discussion, dependency, or acceptance. Keep implementation steps that move
together in one item and explain them in the change request.

## Planning

Use filtered tracked-work and change-request lists. Priority orders tracked
work; no separate planning surface or timebox exists.

## Agent authority

Governed by `.agents/knowledge/agent-authority.md`.

## Update this file when

- The collaboration scale or work driver changes enough to strain this
  profile.
- An omitted semantic's enable trigger fires.
- An enabled semantic goes unused for a sustained period.
- Installation or propagation stops being copy-based.
