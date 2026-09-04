# Issues, Pull Requests, and Autonomous Execution

Read when the harness defines intake forms, issue or pull-request content
contracts, claiming, handoff, or autonomous execution.

## Intake: the GitHub trichotomy

Intake is issue forms named the GitHub way — **Bug report**, **Feature
request**, **Task** — aligned with the organization's native issue types
(whose defaults are exactly Task, Bug, Feature) and with the default
labels. "Issue" is the container on GitHub, never a type name. Questions,
ideas, and support are not issues: route them to Discussions through
`config.yml` contact links, and convert misfiled issues during triage.

Content contracts per form:

- **Bug report**: what is inconsistent, expected behavior, actual behavior
  (the acceptance baseline for the fix), reproduction, context and
  references. No solution design — a bug is unplanned by definition.
- **Feature request**: the outcome and who benefits, context, acceptance
  sketch, explicitly out-of-scope items.
- **Task** (planned work, often agent-executed): content, observable
  outcome, context and references that bridge the designer-executor gap
  (name the URL or repo being imitated, not "like A"), architecture-level
  solution direction and decomposition — never line-by-line prescriptions
  that turn execution into feed-forward imagination — executable
  acceptance criteria (no "improve X") or, under a specification
  contract, a link to the specification's scenarios instead — never both
  (see [spec-expression.md](spec-expression.md)) — out-of-scope, optional
  cautions.
- **Incident** is an opt-in shape for projects with a real operational
  event stream, not a default form: confirmed facts separated from
  hypotheses, impact, a timezone-stamped timeline, response state, and
  follow-up links. Add it only when the design tree selected it.

A complete issue carries its type, priority, area when one applies, and a
milestone when it serves a goal; leave the assignee empty unless the
creator will execute it themselves — ask a human creator which. In an
organization "type" is the native issue type and "priority" the `Priority`
field value; elsewhere both are labels.

## Pull requests: the hub

A PR responds to an issue unless approved policy allows bare PRs. Its
description holds **what and why** (the outcome, not the diff), **changes**
(where and what, briefly — the how), **related work** (`Closes #N` — the
closing keyword is the issue's lifecycle driver), and the project's fixed
checklist. Local test evidence or screenshots join only where project
policy asks. Set assignee and labels on the PR — its labels are what the
generated release notes consume — plus milestone and reviewer per policy.

## Claim and execute (the autonomous state machine)

1. Read the issue; confirm it is open.
2. Inspect assignees. If another identity holds it, stop and ask whether
   duplicate work is intended.
3. Claim: assign the acting identity, then **re-read and confirm it is the
   sole assignee** — GitHub allows ten assignees, so a race yields two
   assignees, not a rejection. Verify the identity is assignable at all;
   GitHub drops non-collaborator assignees silently.
4. Branch natively: `gh issue develop -c <number>` creates and checks out
   the issue-linked branch. Push it and open a **draft pull request
   immediately** with `Closes #N` in the body. The draft PR is the public
   claim and the work log — no separate announcement comment, and no time
   recording anywhere: GitHub stores no durations and the harness does not
   fabricate them. On a public repository the draft's diff and commit
   messages are public from that moment; the confidentiality boundary from
   the design tree applies to every push.
5. Under a specification contract with the combined shape, the draft's
   first content is the change record: push the proposal and delta specs,
   set the phase marker to "specification", request the gate owner's
   review explicitly, and start planning and implementation only after the
   approval comment naming the commit (see
   [spec-expression.md](spec-expression.md)). Under split, link the merged
   specification PR instead.
6. Keep the PR description current; add comments for major discoveries,
   changed assumptions, and decisions reviewers will need.
7. Abandon by un-assigning, closing the draft with a comment stating the
   state and remaining work, and leaving the issue open for pickup.
8. When acceptance criteria and checks pass, update the final description
   and complete the checklist, including the spec-side step the contract's
   archive mode requires. What happens next is set by the project's
   authority policy (`.agents/knowledge/agent-authority.md`, or the location
   the entrypoint records), not by green checks: by default `gh pr ready` and requesting review are the human's
   acceptance decision, and the agent stops at the draft with a
   decision-ready report — goal addressed, tests and CI state, actual
   scope, known risks, remaining limitations. Run `gh pr ready` yourself
   only where the deposited policy grants it. Auto-merge
   (`gh pr merge --auto`) may arm only where approved policy says so, and
   the unattributed-Copilot extra-approval default can demand a second
   review — check it rather than waiting on a phantom.
9. Merge closes the linked issue via the closing keyword. Verify the
   closure landed; do not close by hand what the keyword already handles.

Every metadata change in this machine — labels, assignees, milestone,
type — is an explicit `gh`/API call: there are no slash commands, and
nothing written in a body mutates state except closing keywords.

## Confidentiality

There are no confidential issues: on a public repository every issue, PR,
comment, and push is public immediately and cannot be reliably erased.
Prevention is the only control — run the publish gate from SKILL.md on
every payload, and route security reports through the channel selected in
[security-and-ownership.md](security-and-ownership.md), never the tracker.
