<!--
Raw shape for the workflow file deposited into the target project at
.agents/knowledge/<platform>-workflow.md (github-workflow.md,
gitlab-workflow.md). Rework every line against the settled answers, delete
every section the design does not use, and remove every placeholder and
this comment before the file is written. Write the file in the platform's
own vocabulary — the objects the semantic mapping gave each enabled
semantic — and never in the model's: no "tracked work", "change request",
"objective boundary", "timebox", or "planning surface" may remain.
-->

# <Platform> Workflow

Read this before creating a branch, opening or updating <an issue or pull
request | an issue or merge request>, applying labels, creating a
milestone, or proposing any new management structure. <Platform> is this
repository's <only | primary> remote and task-management platform, so every
rule here is written in <Platform> terms.

<Repository or project path> (<owner type: personal account | organization
| group>, <visibility>). Owner and maintainer: <role or handle>. <One
sentence on which native features exist here and which do not — for
example, issue types and issue fields exist only in organizations, so type
and priority live on labels.>

## What this repository is

<Two or three sentences: what the project delivers, how a change reaches
its consumers (the change propagation, in plain words), and what that
demands of acceptance and rollback.>

## Objects in use

| Object | Meaning here | What is lost without it |
|---|---|---|
| <Issue | Work item> | <what earns one in this project — and what stays a note in a pull or merge request description> | <deletion-test answer> |
| <Pull request | Merge request> | <every change to the default branch; what a draft signifies — work in progress, including a draft whose first content is a specification; never a placeholder for planned work> | <deletion-test answer> |
| Acceptance | <who accepts what, under which conditions — link the specification contract if one exists> | <deletion-test answer> |
| <label set, milestone, iteration, board, sub-issues, tag, release …> | <meaning> | <deletion-test answer> |

## Deliberately not used

| Object | Enable when |
|---|---|
| <e.g. Projects boards, iterations> | <concrete trigger, e.g. three or more people work in parallel and filtered lists stop being enough> |
| <e.g. Releases and version tags> | <concrete trigger> |

## Decomposition

<When a separate <issue | work item> is created: only when the piece
independently earns its own priority, owner, discussion, or acceptance.
State the rule as this project applies it, with one concrete example, and
where smaller pieces live instead (a task list in the change record, a
checklist in the description).>

## Triage

<Who reviews new items, how often, which labels or fields every item must
carry to leave triage, and what automation applies. Delete if the design
has no triage.>

## Planning view

<How current work is viewed and ordered: filtered lists, a board, or
several. State that any view is rebuildable and where the underlying facts
live — on the item, the pull or merge request, or the specification.>

## Agent authority

Governed by `.agents/knowledge/agent-authority.md`. <Delete this section if
no authority policy exists yet, and record that gap instead.>

## Specifications

Governed by `.agents/knowledge/spec-workflow.md`: a <pull request | merge
request>'s acceptance is the scenarios of the specification it implements,
and <issues | work items> link the specification instead of restating it;
the change request shape and archive mode are settled there, and the rules
here must not contradict them. <Delete this section if no specification
contract exists yet, and record that gap instead.>

## Update this file when

- The collaboration scale or driver changes enough to strain the profile.
- A "not used" trigger fires.
- An object in use goes unused for a sustained period — remove it, do not
  let it rot.
- Change propagation changes (new consumers, a new instantiation mechanism,
  new governed targets).
- <The platform's native features change for this owner type or plan.>
