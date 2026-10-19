# Harness Plan

Read when building, changing, or questioning any part of this project's
agent harness. This file records the approved decisions; the harness built
from it must match it.

> Fill-in guidance appears as quoted blocks like this one. Resolve and
> delete every one of them before presenting the draft — a surviving
> guidance block means that decision was not made.

## Project Profile

> One line per fact, from inspection first, the user second.

- Stack:
- Layout:
- Error cost (what a wrong change breaks):
- Maintenance horizon (one-off / short / long-lived):
- Environment isolation (host / sandboxed host / container or VM; what it exposes):
- Agent autonomy (attended / unattended; external inputs agents read; where a human reviews results before they leave):
- Team size (solo / multi-person; review process):
- Pre-authorized irreversible actions (branch pushed, outward writes allowed):
- Hard read-only control-plane files:
- Model class (frontier / weak or local):
- Existing checks and conventions:

## Audit Findings

> Only when a harness already existed. One row per finding; delete the
> section otherwise.

| Component | Class (stale / duplicated / invisible / excessive / orphaned) | Disposition |
|---|---|---|
|  |  |  |

## Axis Decisions

### 1. Thickness Per Layer

> Name the preset the plan starts from (exploratory / engineering), then
> rate each of the six layers omitted / light / medium / thick with the
> variable that sets it; mark every deviation from the preset. For omitted
> layers, record the trigger that would justify building later. Six rows,
> never one level for the project.

Preset:

| Layer | Rating | Variable and reason / build-later trigger | Deviates from preset |
|---|---|---|---|
| Environment and sandbox |  |  |  |
| Knowledge and routing |  |  |  |
| Hard gates |  |  |  |
| Authorization boundary |  |  |  |
| Specification and completion criteria |  |  |  |
| Entropy management |  |  |  |

### 2. Evolution Mode

- Mode (self-evolving / fixed / compromise):
- Reason:
- Reconsideration trigger:

### 3. Agent Topology

- Topology (single / multi):
- Reason:
- Fallback and reconsideration trigger:

### 4. Sync Mechanism Family

- Family (project-skill / entrypoint-or-knowledge-doc):
- Reason:

### 5. Model Class Provisions

- Weak-model target (yes / no):
- Provisions (section lookup in the entrypoint, thinner docs):

## Public-Convention Files Touched

> List only files this plan will create or change; note the convention each
> follows. Delete the section if none.

## Build List

> Ordered. One row per artifact. "Form" is skill / entrypoint-or-doc where
> the artifact could take either shape. A builder that runs in two phases
> occupies one row per phase: a paradigm builder's contract row precedes
> the authority and platform rows, and its shaping row follows the
> platform row. The removal meta-skill runs last, after the user verifies
> the harness.

| # | Artifact | Builder skill to install and run | Form | Done when |
|---|---|---|---|---|
| 1 |  |  |  |  |

## Open Questions

## Approval

- Approved by:
- Date:
