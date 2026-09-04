# Expressing the Specification Contract on GitLab

Read when the target carries a specification contract
(`.agents/knowledge/spec-workflow.md` by default) or a spec tool's
directories sit in the project. The contract decides *where*
specifications live, *who* approves them, and that tracked work links
them; this reference decides only how GitLab objects express that. Never
re-decide the level, the tool, or the approval owner here.

## One rule, applied everywhere

Acceptance criteria exist in exactly one place — the specification's
scenarios. Every GitLab object that would otherwise restate them links the
specification instead.

## Description templates

- The **Task** template carries the optional `## Specification` section
  from the assets (path of the spec or change record). Keep it only under
  a contract; delete it otherwise.
- The Task template's acceptance section accepts either executable
  criteria or a link to the specification's scenarios — never both, as the
  asset's comment states.
- The **Issue** and **Incident** templates are unchanged: an issue's
  expected behavior is, under a spec-anchored contract, the spec's current
  requirement; the fix's change record carries the delta.
- Do not add a specification label or work-item type: a spec is a document
  in the repository, and its lifecycle lives in the tool's own layout.

## Merge requests

- The default MR template's related-work section carries a `Spec:` line
  naming the change record or specification the MR implements.
- The checklist reads "The change satisfies the linked acceptance criteria,
  or the scenarios of the linked specification" and adds "The specification
  is updated or the change record archived where behavior changed".
- When the tool ships a validator and an eligible runner exists, add it as
  a pipeline job with `rules:` limited to changes under the spec
  directories; its failure output must name the file to fix. Without a
  runner or validator, the checklist is the only gate; record that.

## Milestones, epics, and boards

A milestone's observable completion links the specifications whose
scenarios define it; it does not restate them. A specification never
becomes an epic or a board list: the goal is human-endorsed, the spec is
the content that serves it, and a board is a view.

## The project workflow skill

- **Take work:** before claiming, confirm the linked specification is
  approved per the contract's approval gate and treat its scenarios as the
  acceptance criteria. A work item whose spec is missing or unapproved is
  escalated, not executed.
- **Create work:** items for planned work derive from the change record's
  task list, one per task that independently earns state, each linking the
  specification and the scenarios it closes; never copy acceptance
  criteria into the description.
- **Finish:** before the decision-ready report, confirm the spec-side step
  (update, archive, or converge per the contract's lifecycle) is done or
  listed as remaining work.

## Knowledge deposit

Record in the Specifications section of `.agents/knowledge/gitlab-workflow.md`:
the contract's location, which template sections and MR lines exist
because of it, the validator job if any, and the update trigger — "when the
spec directory or tool changes, re-check every template link". Do not copy
the contract's tables there.

## Platform-native option

When the contract records committed specification documents with no tool,
express the same rules with a repository path in the `## Specification`
section and the archive step on the MR checklist. Do not promote work items
or the Wiki into the specification store — the contract placed
specifications in the repository, and a closed item reads as "done", not
as a requirement.
