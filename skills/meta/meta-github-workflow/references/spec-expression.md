# Expressing the Specification Contract on GitHub

Read when the target carries a specification contract
(`.agents/knowledge/spec-workflow.md` by default) or a spec tool's
directories sit in the repository. The contract decides *where*
specifications live, *who* approves them, and that tracked work links
them; this reference decides only how GitHub objects express that. Never
re-decide the level, the tool, or the approval owner here.

## One rule, applied everywhere

Acceptance criteria exist in exactly one place — the specification's
scenarios. Every GitHub object that would otherwise restate them links the
specification instead. The failure this prevents is the issue that says
one thing, the spec another, and the PR a third.

## Issue forms

- **Task** and **Feature request** forms carry the optional `Specification`
  input from the assets (path of the spec or change record). Keep it only
  under a contract; delete it otherwise. It stays optional because a bug
  report has no spec and triage may create the spec after intake.
- The Task form's acceptance field accepts either executable criteria or a
  link to the specification's scenarios — never both. State that in the
  field description as the assets do.
- The **Bug report** form is unchanged: a bug's acceptance baseline is the
  expected behavior, which under a spec-anchored contract is the spec's
  current requirement; the fix's change record carries the delta.
- Do not add a "Specification" issue type or label: a spec is a document
  in the repository, and its lifecycle (proposed, approved, implemented,
  archived) lives in the tool's own layout, not in issue metadata.

## Pull requests

- The PR template's related-work section carries a `Spec:` line naming the
  change record or specification the PR implements.
- The checklist reads "Acceptance criteria of the linked issue, or the
  scenarios of the linked specification, are met" and adds
  "Specification updated or change record archived where behavior
  changed". Keep the security line's wording intact — the checklist
  workflow keys on it.
- When the tool ships a validator, add it as a job in the checks workflow
  running on pull requests that touch the spec directories, with a path
  filter and the aggregator gate the harness already uses. Its failure
  message must name the file to fix. Without a validator, the checklist is
  the only gate; say so in `checks.md`.

## Tracking issues and milestones

A tracking issue's "Observable completion" links the specifications whose
scenarios define the goal; it does not restate them. A milestone remains
the date bucket. Neither replaces a specification, and a specification
never becomes a tracking issue: the goal is human-endorsed, the spec is
the content that serves it.

## The project workflow skill

- **Take work:** before claiming, confirm the linked specification is
  approved per the contract's approval gate and treat its scenarios as the
  acceptance criteria. An issue whose spec is missing or unapproved is
  escalated, not executed.
- **Create issues:** issues for planned work derive from the change
  record's task list, one issue per task that independently earns state,
  each linking the specification and the scenarios it closes. Keep the
  spec-tool step that converts tasks to issues if the tool offers one, but
  its output must obey the form's headings and the label and type rules —
  read the created issues back.
- **Finish:** before the decision-ready report, confirm the spec-side
  step (update, archive, or converge per the contract's lifecycle) is done
  or listed as remaining work.

## Knowledge deposit

Record in `planning.md`: the contract's location, which form fields and
template lines exist because of it, the validator job if any, and the
update trigger — "when the spec directory or tool changes, re-check every
template link". Do not copy the contract's tables into `planning.md`.

## Platform-native option

When the contract records committed specification documents with no tool,
express the same rules: the `Specification` field holds a repository path,
the PR checklist carries the archive step, and no validator job exists.
Do not promote issues or Discussions into the specification store — the
contract placed specifications in the repository, and a closed issue reads
as "done", not as a requirement.
