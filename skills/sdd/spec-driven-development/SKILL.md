---
name: spec-driven-development
description: >-
  Spec-driven development (SDD) — writes specifications before code and
  runs the specify, clarify, approve, tasks, implement, verify loop from
  them; judges whether the discipline pays and at which level; settles how specs meet tracked work — when the
  draft opens, what the approval package holds and where the design sits,
  whether a spec needs its own PR, how a change is archived, what issues
  and PRs link instead of restate; and converts existing code into a
  spec-driven project without backfilling specs. Use when adopting or
  starting SDD, to "write the spec first" or "do this spec-driven", which
  approach family fits, whether the design is approved before tasks, how
  issues, PRs, and specs fit together, where a spec is reviewed or when it
  is archived, when a prototype needs specs, when what was built drifts
  from what was agreed, or when issues and specs disagree. Not for
  defining goals, building the platform harness, or a tool's own change
  command.
license: Apache-2.0
---

# Spec-Driven Development

A specification is a structured, behavior-oriented artifact in natural
language that states what software must do and serves as the instruction an
agent implements from. Spec-driven development (SDD) makes that artifact the
first thing written and the thing verification is judged against, so the
agreement about *what* survives the session that wrote the code.

This skill is the methodology, in three layers that live in three places:
the practice and its loop (here); the project's own rules — level, tool,
approval mode, archive executor — in the project's specification contract,
written by a harness builder; and each tool's records, commands, and
request automation in that tool's framework skill. Nothing here names a
tool's command or a platform's object.

- Read [references/adoption-decision.md](references/adoption-decision.md)
  when the user asks whether, at which level, or with which approach
  family to adopt spec-driven development, what the rejected alternatives
  cost, or before recommending a tool.
- Read [references/tracked-work.md](references/tracked-work.md) when a
  loop step meets the platform's tracked work: publishing the draft,
  waiting for approval, reconciling, drafting the request body, archiving
  before ready.
- Read [references/adopting-existing-code.md](references/adopting-existing-code.md)
  when the project already contains code that was not written from a
  specification — a prototype, a vibe-coded app, a brownfield codebase.

## The loop

Run every change through these steps with the chosen tool's equivalent
command for each. This skill pairs with `plan-clarification` for step 2,
the interrogation the clarify step needs. If it is not installed, load the
`ryan-minato-skills-installing` skill and install `plan-clarification` as
it directs; never run an install command yourself. If the user declines,
run step 2 from the questions in this file.

1. **Specify.** Create the change record with the tool's command, then
   write the requirements and their scenarios for this change only, plus
   non-goals. Done when: every requirement has at least one scenario a
   reader who has never seen the code could judge, and the validator
   passes.
2. **Clarify.** Interrogate every ambiguity, unstated assumption, and
   missing edge case with the user; record answers in the spec, not in the
   chat. Done when: no `[NEEDS CLARIFICATION]`-class marker remains.
3. **Bound the approach, then publish.** When a design is warranted (see
   [The approval package](#the-approval-package)), write it: the approach,
   its technical constraints, the preferences, the rejected alternatives —
   never steps. Validate, commit the change record, and open the draft
   change request (or the specification change request, under the split
   shape) with the complete approval package as its first content. Then
   stop: no task list and no code until the contract's approval mode is
   satisfied (the gate owner closes the discussion in conversation, or
   posts the fixed comment). Done when: the package is complete and
   published, and the agent is waiting and says what it waits for.
4. **Reconcile, then tasks.** When the gate owner closes the discussion,
   read the request's comments and review threads with their resolution
   state; list every unresolved thread, every adjustment requested in the
   discussion that the package does not carry, and every pair of
   conclusions that contradict each other; ask the gate owner to confirm
   the open items; record the closing on the request. Only then break the
   approved package into ordered, independently verifiable tasks with the
   scenarios each one closes. A task list the tool generated with the
   specification is a draft the implementer finishes now; it was never the
   gate's object. Done when: nothing is open or the open items are
   confirmed, no task lacks a scenario, and no scenario lacks a task.
5. **Implement.** Work task by task; when the code must deviate from the
   spec or the design, stop and change the record first, with the user's
   approval. Done when: every task is closed or its deviation is recorded
   as an approved change.
6. **Verify.** Execute the scenarios — tests, commands, observed states —
   against the running result, not against the diff. When the change
   first creates a domain and the project's schema requires a baseline
   block, verify its scenarios like any other; a baseline scenario that
   fails for behavior the change does not touch is filed as a separate
   defect with the evidence, and the record is narrowed to what is
   verified — never widened silently. Done when: every scenario has
   passed, is recorded as a spec change, or is filed as a defect.
7. **Archive or converge, then ready.** Write the delivered behavior back
   into the source-of-truth spec (spec-anchored) or archive the change
   record (spec-first) inside the request, through the tool's archive
   command or the automation the framework skill installed, as the
   contract's executor says; validate afterwards; only then mark the
   request ready. Done when: the spec and the code describe the same
   system and the integration branch will receive no unarchived record.

## Tool commands first, then the validator

When the project uses a specification tool, every operation the tool has
a command for — initializing, creating a change or feature record,
validating, archiving — runs through that command, verified from the
tool's `--help` and current documentation first. Never create the tool's
directory tree or its generated files by hand, even when asked because it
is faster: a hand-made record lacks the metadata the tool's later steps
read, and the validator or the archive fails on it long after the
shortcut was taken. Hand edits stop at the record's text itself.

Run the tool's validator — strict mode where it exists — after each
artifact edit, before publishing the draft, before marking the request
ready, and after archiving; fix what it reports before proceeding. When
the tool ships no validator, check the artifacts against the tool's
documented structure and say that no programmatic check exists. A passing
validation proves structure, not content; it never replaces the approval
review.

## Specification quality

- One requirement, one normative statement (SHALL or MUST); split compound
  requirements.
- Every requirement carries at least one scenario in a given / when / then
  shape, and one of them covers an edge or failure path.
- Non-goals are written down; an unstated exclusion is a future dispute.
- Scope requirements as *now*, *later*, and *out of scope*; only *now* enters
  the design.
- Requirements and design live in separate files; a requirement that names
  a class or a table is a design decision in disguise.
- A spec describes observable behavior, never the diff: "the API returns
  409 on a duplicate email" is a requirement; "add a unique index" is not.

## The approval package

The gate reviews the *approval package*: the specification plus the design
when one is warranted. What the package holds depends on the tool — a
spec-anchored change workflow: the proposal, the delta specs, and the
design; a spec-first kit with a constitution: the specification and the
plan; an IDE's native spec files: the requirements and the design;
committed specification documents: the specification with its approach
section — and the framework skill for the tool carries the details. The
task list always follows approval.

A design is warranted when more than one reasonable approach exists, when
the change touches structure, interfaces, dependencies, or files outside
the record, or when the project's schema or contract requires one; a
wording change inside one section needs none. The design is a broad record
of the chosen approach, the technical constraints, the preferences, and
the rejected alternatives — the plan-mode document of an agent harness,
not a procedure. It bounds *how* so the implementation stays controllable
while leaving the steps to the implementer; a design that reads as a
numbered procedure is a task list in disguise and is rewritten before the
draft opens. Because it is committed and published like every record, it
carries no secret, credential, internal hostname, or private data —
constraints are where those leak first.

The review examines the outcome description — goals and scope,
terminology and the domain model, behavior, invariants, constraints and
rules, states and their transitions, interface contracts, data contracts,
exceptions and edge cases, security and permissions, metrics and
acceptance criteria — and the design's bounds. It never reviews the task
list or a step breakdown: those describe how the outcome is built and are
judged by implementation review. A reviewer handed a task list is being
asked to approve a method, not an outcome.

## Project rules live in the contract

The project's specification contract — `.agents/knowledge/spec-workflow.md`
by default, or the file the agent entrypoint points to — records the
level, the tool and its framework skill, the change request shape
(combined or split), the approval owner and mode, when a design is
warranted, the archive executor, the specification scope, and what tracked
work links. Apply those facts without asking them again; the specification
owns *what*, *why*, and the acceptance criteria, tracked work owns *who*,
*when*, and *status* and links the record, and acceptance criteria exist in
exactly one place.

When no contract exists, apply these defaults and say so in the reply,
naming the spec workflow builder of the `meta` catalog (offered through
the installing skill, as the last section says) as the way to settle and
record them — never run a questioning round of your own:

- **Shape**: combined — one change request carries the record from the
  moment the package is committed, opens as a draft, and the gate is
  exercised on that draft; split only where consumers depend on a stable
  contract.
- **Approval**: discussion-closed on the complete package — the gate
  owner discusses on the draft, directs record changes in conversation,
  and declares the discussion closed in conversation; the record of
  approval is that closing plus the request's discussion state, and the
  reconciliation precedes the task list. A blocking comment is the
  alternative only where the contract or the user asks for one. A platform
  review approval is never the record, because later pushes dismiss it.
- **Design**: warranted by the rule above.
- **Archive executor**: by hand, inside the request before it is marked
  ready, with the tool's archive command; the automation the framework
  skill installs (a trigger label whose bot archives, commits, pushes, and
  removes the label) only when the contract names it. On a fork the author
  runs the commands the bot posts. The archived record is frozen: a defect
  review finds afterwards goes to the request's validation section or a
  follow-up change. Archiving after the merge by an automation that pushes
  to the integration branch is not proposed.
- **Scope**: specification domains cover the product the project
  delivers; a change to the project's own harness, tooling, checks,
  workflows, or documents is a spec-less change carried by the tool's
  marker, linked by tracked work the same way. Refuse to create a domain
  for tooling.
- **Request body**: it navigates to the record and carries no
  implementation until ready — the default shape is in
  [references/tracked-work.md](references/tracked-work.md).

## The framework skill

When the project runs a specification tool, its usage — creating and
validating records, composing the approval package, archiving or
completing, the comment commands and labels a request carries, installing
that automation — belongs to the framework skill for the tool in the `sdd`
catalog of https://github.com/ryan-minato/skills (`openspec-workflow`,
`spec-kit-workflow`). Load it; if it is not installed, load the
`ryan-minato-skills-installing` skill and install it as it directs, never
running an install command yourself. If the user declines, or no framework
skill exists for the tool, run the loop with the tool's commands verified
from its own help, quote none, and say that the request automation is not
installed.

## Setting up or improving the project's rules

Initializing a project's spec-driven rules, or improving them — the
contract, the templates and forms, the project skill's steps — is the work
of the spec workflow builder in the `meta` catalog of the same repository,
a disposable harness builder that settles each rule with the user,
deposits the contract, and hands tool adoption and automation to the
framework skill. When the user asks for that, load the
`ryan-minato-skills-installing` skill and install the whole `meta`
catalog at project scope as it directs — its builders stack and are
disposed together; never run an install command yourself.

If the user declines, or the catalog is unavailable, record the defaults
above — and which of them the project departs from — in the project's
knowledge base, and list the harness build as remaining work. Do not edit
templates, forms, checks, automation, or a project skill: those stay the
builder's.

## Gotchas

- Tools rename their commands between releases; a command remembered from a
  blog post is the most common way an SDD setup fails on day one.
- A change directory written by hand usually lacks the tool's metadata
  file or a required heading; the validator or the archive step fails on
  it later, when the shortcut is forgotten.
- Backfilling specs for code nobody is changing feels productive and rots
  immediately: nothing forces those specs to track reality.
- The goal document (what the project must achieve) sits above every spec;
  a constitution or steering file may cite it, and no spec may contradict it.
- Requirements written by observing a prototype record what the prototype
  happens to do; only the user can say which of that behavior was intended.
- A drifted spec is an active source of falsehood: when the spec and the
  code disagree, the next agent trusts the spec and builds on a lie. Fix
  the spec or delete it before any other work.
- A review approval left on a draft does not survive the implementation
  pushes — removed by default on GitLab, by a stale-approval rule on
  GitHub, and stale in meaning everywhere; the record of approval is the
  closing of the discussion or the fixed comment, as the contract says.
- Closing the discussion without reading the threads implements the
  record as the author remembers it, not as the discussion left it; the
  reconciliation is what makes the closing an approval.
- Opening the draft before the design is written, or after the task list
  is, turns the gate into a formality: the reviewer sees either half a
  package or a method.
