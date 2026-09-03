<!--
Raw shape for the workflow contract deposited into the target project at
.agents/knowledge/project-workflow.md. Rework every line against the settled
answers, delete every section the design does not use, and remove every
placeholder and this comment before the file is written. The file must stay
platform-neutral: no platform product or object names anywhere.
-->

# Project Workflow Contract

Read this before creating, splitting, or closing tracked work, before
starting a change request, and before proposing any new management
structure.

Profile: <base profile>. Selected because <one sentence naming the project
fact that decided it>.
Overlays: <list, or "none">. <One selecting fact per overlay.>
Change propagation: <Local | Copy | Dependency | Inherited/Enforced |
State-mutating> — <one sentence on what a change here reaches and what that
demands of acceptance and rollback>.

## Enabled semantics

| Semantic | Meaning here | What is lost without it |
|---|---|---|
| Tracked Work | <what earns an item in this project — and what stays a note in a change description> | <deletion-test answer> |
| Change Request | <when a change stands alone vs. links to tracked work; what a draft signifies> | <deletion-test answer> |
| Acceptance | <who accepts what, under which conditions> | <deletion-test answer> |
| <further semantic> | <meaning> | <deletion-test answer> |

## Intentionally omitted

| Semantic | Enable when |
|---|---|
| <e.g. Planning Surface> | <concrete trigger, e.g. parallel work by three or more people makes a filtered list insufficient> |
| <e.g. Timebox> | <concrete trigger> |

## Work decomposition

<When a child item is created: only when it independently earns its own
state, owner, discussion, dependency, or lifecycle. State the rule as this
project applies it, with one concrete example.>

## Objectives and timeboxes

<Whether objective boundaries exist here, who creates and confirms them, and
how completion is judged. Whether timeboxes exist and what a window means.
Delete whichever half the design omits — the omission table above already
records the trigger.>

## Planning

<How current work is viewed and ordered: a filtered list, a dedicated
surface, or several. State that any surface is a rebuildable view and where
the underlying facts live.>

## Agent authority

Governed by `.agents/knowledge/agent-authority.md`. <Delete this section if
no authority policy exists yet, and record that gap instead.>

## Specifications

Governed by `.agents/knowledge/spec-workflow.md`: a change request's
acceptance is the scenarios of the specification it implements, and tracked
work links the specification instead of restating it. <Delete this section
if no specification contract exists yet, and record that gap instead.>

## Update this file when

- The collaboration scale or driver changes enough to strain the profile.
- An omitted semantic's enable-trigger fires.
- An enabled semantic goes unused for a sustained period — remove it, do not
  let it rot.
- Change propagation changes (new consumers, a new instantiation mechanism,
  new governed targets).
