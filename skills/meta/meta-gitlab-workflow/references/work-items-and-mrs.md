# Work Items and Merge Requests

Read when the harness defines task, issue, incident, merge request, assignment,
time tracking, or autonomous work execution.

Resolve the current `work items`, `tasks`, `issues`, `incidents`, `description
templates`, `merge requests`, `draft merge requests`, `quick actions`, and
`time tracking` topics through `llms.txt`. GitLab versions differ in whether
these are separate resources or work-item types; encode intended semantics,
then use the capability the target actually exposes.

## Content contracts

Rework the description-template assets for the selected types:
`assets/issue-template-task.md`, `assets/issue-template-issue.md`,
`assets/issue-template-incident.md`, and `assets/mr-template-default.md`.

### Task

Use for planned project evolution. Include content, explicit outcome, context
and references, architecture-level solution direction and work decomposition,
executable acceptance criteria, out-of-scope items, and optional cautions.
Do not prescribe an imagined implementation line by line; preserve the
executor's feedback loop and technical judgment. A goal such as “improve X” is
not executable until its observable target is stated.

### Issue

Use for an observed mismatch with intended behavior. Include what is
inconsistent, expected state, actual state as the repair acceptance baseline,
and relevant context/references. Do not invent solution design for an
unexpected defect.

### Incident

Use for an unplanned operational, hardware, software, availability, safety, or
access event. Keep the template flexible but require confirmed facts, impact,
time observations, current response state, confidentiality handling, and
follow-up linkage. Separate facts from hypotheses.

### Merge request

An MR responds to a task, issue, or incident unless the approved project policy
allows a bare MR. Its description includes:

- **What and why:** the behavior or outcome changed and why it matters.
- **Changes:** concise locations/components and what changed there.
- **Related work:** closing or related references with correct semantics.
- **Checklist:** the project's fixed contribution and quality gates.

Add local test results, screenshots, rollout evidence, or review instructions
only when the project requires them. Set the author as assignee plus applicable
labels and milestone. Set reviewers only when policy requires them or the
author explicitly plans to review with that person.

## Assignment and execution state machine

For agent-initiated work:

1. Read the object and verify it is open.
2. Inspect assignees. If another person owns it, stop and ask whether duplicate
   work is intended. Otherwise assign the acting identity, announce the start,
   and record the start timestamp.
3. Create a policy-compliant branch, push it, and open a draft MR as soon as the
   initial branch exists. The draft is the ownership and collaboration surface,
   not a completion claim.
4. Keep the MR description current. Add comments for major discoveries,
   changed assumptions, evidence, or decisions that future reviewers need.
5. Log time to the work item or MR on completion, material change, pause, or
   abandonment using the target version's verified mechanism.
6. If abandoning, explain the state and remaining work, remove the assignee,
   and leave the object open for pickup unless the user chooses otherwise.
7. When acceptance criteria pass and the MR is reviewable, update the exact
   final description, complete the checklist, and remove draft status.

For human-authored objects, use the consensus tree before step 1. If project
policy does not assign at creation and the creator will not begin immediately,
leave assignee empty; ask a human creator whether they intend to self-assign.

## Unattended operation

Enable only after explicit user approval and verified isolation, rollback,
validation, audit, and external-write boundaries. Delegations come from the
target's authority policy (`.agents/knowledge/agent-authority.md`) when one
exists — never from a green pipeline or an available role. The durable
project skill must state which reads, assignments, comments, branch pushes,
draft MRs, metadata edits, ready transitions, and merges are delegated. By
default the agent stops at the draft MR with a decision-ready report, and
ready transitions, review requests, merging, and security-sensitive
settings remain human-approved.

## Confidentiality

Assume descriptions, comments, quick-action effects, attachments, branches,
commits, and notifications cannot be completely erased after publication.
Review the exact payload before every publish. Use confidential work items and
private security channels only after verifying how the target version exposes
them; never put a credential or sensitive personal data there merely because
visibility is restricted.
