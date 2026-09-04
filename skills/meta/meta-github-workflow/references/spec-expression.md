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
  scenarios of the linked specification, are met" and adds two items: "The
  change record was approved by the gate owner before implementation, or
  this pull request carries the specification only" and "The specification
  is updated, or the change record is archived, or every task is complete
  and archiving runs after merge". Keep the security line's wording intact
  — the checklist workflow keys on it — and keep the word "secrets" out of
  the new items.
- When the tool ships a validator, add it as a job in the checks workflow
  running on pull requests that touch the spec directories, with a path
  filter and the aggregator gate the harness already uses. Its failure
  message must name the file to fix. Without a validator, the checklist is
  the only gate; say so in `checks.md`.

## Change request shape and archive mode

The contract records both; express them, never re-decide them.

- **Combined shape.** The draft pull request opens the moment the change
  record is committed — its first push is the proposal and the delta
  specs, before any plan or task list. Request the gate owner's review
  explicitly (`gh pr edit --add-reviewer`): CODEOWNERS is not auto-requested
  on drafts. The gate owner records approval as a comment naming the
  approved commit, never as a review approval, which a ruleset dismissing
  stale approvals (or the next push) removes; the review-approval state is
  reserved for implementation review after ready. The PR body's `Spec:`
  line names the change record and a phase marker says whether the PR is
  in its specification or implementation phase. Ready follows the
  scenarios' verification and, under in-request archiving, the archive.
- **Split shape.** The specification pull request uses the same template
  with the `Spec:` line and the phase marker "specification", references
  the issue with `Refs #N` — never `Closes`, which would close the work
  when the spec merges — and takes its title from the project's commit
  convention for specification changes. Its review approval and merge are
  the gate. Implementation pull requests link the merged specification PR
  and the record's path; the last one carries `Closes #N`.
- **Automated archiving with OpenSpec.** Adapt the asset
  `assets/workflow-spec-archive.yml`:
  a workflow on `push` to the default branch, serialized by a
  `concurrency` group with `cancel-in-progress: false`, that installs the
  pinned OpenSpec CLI, runs the project's `scripts/archive_completed_changes.py`
  (the script ships with the spec-driven development methodology skill —
  if the target lacks it, load the `ryan-minato-skills-installing` skill
  and install that skill as it directs, never run an install command
  yourself — and is copied into the project's `scripts/`), validates
  strictly, commits under the project's commit convention, and pushes
  without retry. `GITHUB_TOKEN` pushes trigger no further workflows, so the
  job validates itself. The workflow's identity needs a bypass on the
  default branch's ruleset: record it in `platform-settings.md` as a
  maintainer action, and state in the contract that in-request archiving
  is in force until it is granted. Verify the CLI's non-interactive archive
  flag from its help before shipping the asset.
- **Automated archiving with any other tool.** Spec-Kit, Kiro, and
  committed documents have no fixed archive operation. Do not adapt the
  OpenSpec asset or copy its commands: take from the contract the
  project-defined completion criterion and post-processing step (the
  specification builder records them, and asks the user for them if
  missing), and design the job with the user from the same skeleton —
  push trigger, serialized, rescanning, self-validating, no retry.
- **In-request archiving.** No workflow; the PR checklist's archive item
  is the gate.

## Tracking issues and milestones

A tracking issue's "Observable completion" links the specifications whose
scenarios define the goal; it does not restate them. A milestone remains
the date bucket. Neither replaces a specification, and a specification
never becomes a tracking issue: the goal is human-endorsed, the spec is
the content that serves it.

## The project workflow skill

- **Take work:** follow the contract's shape. Combined: an issue with no
  specification yet is taken by committing the change record to the draft
  PR first and waiting for the gate owner's approval comment; an issue
  whose record exists but is unapproved waits the same way. Split: an
  issue whose specification PR is not merged is escalated, not executed.
  In both, the approved specification's scenarios are the acceptance
  criteria.
- **Create issues:** issues for planned work derive from the change
  record's task list, one issue per task that independently earns state,
  each linking the specification and the scenarios it closes. Keep the
  spec-tool step that converts tasks to issues if the tool offers one, but
  its output must obey the form's headings and the label and type rules —
  read the created issues back.
- **Finish:** before the decision-ready report, confirm the spec-side
  step per the contract's archive mode — the record archived (in-request)
  or every task ticked so the archive job takes it (automated) — is done
  or listed as remaining work.

## Knowledge deposit

Record in the Specifications section of `.agents/knowledge/github-workflow.md`:
the contract's location, which form fields and template lines exist
because of it, the validator job if any, and the update trigger — "when the
spec directory or tool changes, re-check every template link". Do not copy
the contract's tables there.

## Platform-native option

When the contract records committed specification documents with no tool,
express the same rules: the `Specification` field holds a repository path,
the PR checklist carries the archive step, and no validator job exists.
Do not promote issues or Discussions into the specification store — the
contract placed specifications in the repository, and a closed issue reads
as "done", not as a requirement.
