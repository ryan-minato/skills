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
  or the scenarios of the linked specification" and adds two items: "The
  change record was approved by the gate owner before implementation, or
  this merge request carries the specification only" and "The
  specification is updated, or the change record is archived, or every
  task is complete and archiving runs after merge".
- When the tool ships a validator and an eligible runner exists, add it as
  a pipeline job with `rules:` limited to changes under the spec
  directories; its failure output must name the file to fix. Without a
  runner or validator, the checklist is the only gate; record that.

## Change request shape and archive mode

The contract records both; express them, never re-decide them.

- **Combined shape.** The draft merge request opens the moment the change
  record is committed — its first push is the proposal and the delta
  specs, before any plan or task list. Request the gate owner's review
  explicitly. The gate owner records approval as a comment naming the
  approved commit, never as an MR approval: GitLab removes approvals when
  commits are added by default, and the approval state is reserved for
  implementation review after the draft flag is removed. The description's
  `Spec:` line names the change record and a phase marker says whether the
  MR is in its specification or implementation phase. Removing the draft
  flag follows the scenarios' verification and, under in-request
  archiving, the archive.
- **Split shape.** The specification merge request uses the same template
  with the `Spec:` line and the phase marker "specification", references
  the work item without a closing pattern — the closing pattern would close
  the work when the spec merges — and takes its title from the project's
  commit convention for specification changes. Its approval and merge are
  the gate. Implementation merge requests link the merged specification MR
  and the record's path; the last one carries the closing reference.
- **Automated archiving with OpenSpec.** Adapt the asset
  `assets/gitlab-ci-spec-archive.yml`:
  a job on the default branch pipeline, serialized by
  `resource_group: spec-archive`, that installs the pinned OpenSpec CLI,
  runs the project's `scripts/archive_completed_changes.py` (the script
  ships with the spec-driven development methodology skill — if the target
  lacks it, load the `ryan-minato-skills-installing` skill and install that
  skill as it directs, never run an install command yourself — and is
  copied into the project's `scripts/`), validates strictly, commits under
  the project's commit convention, and pushes without retry. The push needs
  a project access token or deploy token allowed to push to the protected
  default branch, stored as a masked CI variable: record it as a
  maintainer action in the platform-settings knowledge, and state in the
  contract that in-request archiving is in force until it exists. Verify
  the CLI's non-interactive archive flag from its help before shipping the
  asset.
- **Automated archiving with any other tool.** Spec-Kit, Kiro, and
  committed documents have no fixed archive operation. Do not adapt the
  OpenSpec asset or copy its commands: take from the contract the
  project-defined completion criterion and post-processing step (the
  specification builder records them, and asks the user for them if
  missing), and design the job with the user from the same skeleton —
  default-branch trigger, resource group, rescanning, self-validating, no
  retry.
- **In-request archiving.** No job; the MR checklist's archive item is the
  gate.

## Milestones, epics, and boards

A milestone's observable completion links the specifications whose
scenarios define it; it does not restate them. A specification never
becomes an epic or a board list: the goal is human-endorsed, the spec is
the content that serves it, and a board is a view.

## The project workflow skill

- **Take work:** follow the contract's shape. Combined: a work item with
  no specification yet is taken by committing the change record to the
  draft MR first and waiting for the gate owner's approval comment; one
  whose record exists but is unapproved waits the same way. Split: a work
  item whose specification MR is not merged is escalated, not executed. In
  both, the approved specification's scenarios are the acceptance
  criteria.
- **Create work:** items for planned work derive from the change record's
  task list, one per task that independently earns state, each linking the
  specification and the scenarios it closes; never copy acceptance
  criteria into the description.
- **Finish:** before the decision-ready report, confirm the spec-side step
  per the contract's archive mode — the record archived (in-request) or
  every task ticked so the archive job takes it (automated) — is done or
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
