# Running the Loop Against Tracked Work

Read when a step of the loop meets the project's tracked work: publishing
the draft, waiting for the approval, drafting the change request body,
archiving. Everything here is stated in tool-neutral and platform-neutral
terms; the project's contract and templates use the platform's own words
(issue, pull request, merge request, workflow, job), and they win.

## What the contract answers

| Fact | Where it is recorded | Default when no contract exists |
|---|---|---|
| Level and tool | the contract's header and artifact map | the tool the project already runs; otherwise the family fitting the situation, offered to the user |
| Change request shape | the contract's shape section | combined |
| Approval owner and record | the contract's approval gate | the gate owner's comment on the draft naming the approved commit |
| Archive mode | the contract's archive section | in-request |
| What tracked work links | the contract's tracked-work section | the record's path; acceptance criteria never copied |

## The two shapes at run time

| Step | Combined | Split |
|---|---|---|
| Publish (loop step 2) | Commit the record; open a draft change request whose first content is the record, with the phase marker "specification"; stop | Commit the record; open a specification change request carrying only the record, referencing the work item without closing it, titled per the project's commit convention for specification changes; stop |
| Approval | Recorded on the draft as the contract says | The specification change request is approved and merged |
| Plan, tasks, implementation | On the same branch, after the approval | On one or more implementation change requests, each linking the merged record; re-validate the delta first — a domain spec another change archived since may have moved |
| Ready | Implementation review of the scenarios against the result | The same, per implementation change request |
| Merge | Closes the work item | The last implementation change request closes it |

Under split, a record on the integration branch that is approved but has
no open work item owning its implementation is stale: assign it or remove
it through a change request, never by hand.

## Waiting for the approval

After publishing, do not write design, tasks, or code. Say what is being
waited for: the gate owner's comment on the draft naming the approved
commit, or the merge of the specification change request. A platform
review approval is not the record, because later pushes dismiss it and
the approval then points at a tip the implementation replaces. Request
the reviewer explicitly: drafts do not auto-request code owners.

## Archive modes at run time

- **Automated.** Leave the completed change for the archive job: every
  task ticked, the delta validated. The job runs one at a time, rescans
  every completed change on each run, and fails without retry when its
  push is rejected — the run the competing merge triggered archives the
  rest. Until the harness records the job's push path, the in-request
  mode is in force.
- **In-request.** Archive the record inside the change request before it
  is marked ready, so the integration branch never holds an unarchived
  record.

## Per-tool loop notes

- **OpenSpec.** A change is a directory (proposal, delta specs, optional
  design, tasks). Propose generates design and task files with the
  specification; the gate reviews the proposal and the delta specs only.
  The tool archives after merge or inside the pull request; the automated
  mode is the former made mechanical.
- **Spec-Kit.** Under split, the feature directory's specification merges
  in the specification change request; plan and tasks join the
  implementation request. There is no archive operation: a spec-first
  feature is complete when delivered, and a spec-anchored project's rule
  is what updates the living spec.
- **Kiro.** Under split, the requirements file merges first; design and
  tasks join implementation. Ticks in the tasks file are status: tick
  only inside the implementing change request.
- **Committed documents.** The contract names who merges the delta into
  the domain spec and when, because nothing does it automatically.
