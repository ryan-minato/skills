---
name: spec-driven-development
description: >-
  Spec-driven development (SDD) — writes specifications before code and
  runs the specify, clarify, approve, tasks, implement, verify, freeze
  loop from them; judges whether the discipline pays and at which level;
  settles how specs meet tracked work — when the draft opens, what the
  approval package holds, whether a spec needs its own PR, when a record
  is frozen and who approves the frozen version, what issues and PRs link
  instead of restate; and converts existing code into a spec-driven
  project without backfilling specs. Use when adopting or starting SDD,
  to "write the spec first" or "do this spec-driven", which approach
  family fits, whether the design is approved before tasks, whether an
  approval must be recorded, how issues, PRs, and specs fit together,
  where a spec is reviewed or when it is archived, when a prototype needs
  specs, when what was built drifts from what was agreed, or when issues
  and specs disagree. Not for
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
  waiting for either gate, reconciling, drafting the request body, marking
  the request ready, freezing the record.
- Read [references/adopting-existing-code.md](references/adopting-existing-code.md)
  when the project already contains code that was not written from a
  specification — a prototype, a vibe-coded app, a brownfield codebase.

## The loop

Run every change through these steps with the chosen tool's equivalent
command for each. This skill pairs with `plan-clarification` for step 2,
the interrogation the clarify step needs. If it is not installed, load the
`ryan-minato-skills-installing` skill and install `plan-clarification` as
it directs; never run an install command yourself. If the user declines,
run step 2 by interrogating the draft against the specification quality
rules below until none of them is unanswered.

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
7. **Publish the finished implementation.** Mark the change request ready
   and say what is now open: the implementation, against the package the
   first gate approved. The record is still unfrozen, so the project's
   specification check reports the request as unfinished — that red is the
   merge block for the whole deliberation, not a defect to chase. Done
   when: the request is ready, the deliberation is asked for, and the
   agent is waiting and says what it waits for.
8. **Freeze, then hand the decision over.** When the gate owner closes
   this deliberation, reconcile it as step 4 reconciles the first one,
   then write the delivered behavior back into the source-of-truth spec
   (spec-anchored) or archive the change record (spec-first) inside the
   request, through the tool's archive command, and validate. The freeze
   is what the approval applies to: after it, a change to the record is a
   new round, and a commit pushed after it spends the closing. Done when:
   the spec and the code describe the same system, the integration branch
   will receive no unarchived record, and the approval the contract asks
   for names the frozen version.

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
- **Approval**: two gates, both closed in conversation — the gate owner
  declares the package deliberation closed before the task list, and the
  implementation deliberation closed before the freeze; the agent
  reconciles each before acting, and the freeze commit is the record that
  a version was fixed. A **recorded approval** — any mechanism that
  leaves a durable, attributable mark on a named version, canonically the
  platform's review approval paired with the setting that dismisses it on
  a new commit — is added only where someone outside the conversation
  must verify for themselves that a named person accepted a named
  version. Recommend it from who reads the record, never from team size
  or pipeline maturity.
- **Design**: warranted by the rule above.
- **Archive executor**: a person on the request's branch, with the tool's
  archive command, once the implementation deliberation closes. No job
  archives: no platform token can push to a fork, so such a job would
  never serve an external contribution, and on a branch it can reach it
  would be the only automation needing write access to the repository's
  contents. On a fork the executor is whoever holds the branch — its
  author, or a maintainer who pulled it. The frozen record stays frozen: a
  defect review finds afterwards goes to the request's validation section
  or a follow-up change. Archiving after the merge by an automation that
  pushes to the integration branch is not proposed.
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
- A platform review approval is dismissed by the next push — by default
  on GitLab, by an opt-in rule on GitHub. That is the feature, not the
  flaw: the pair is what makes a review approval a *recorded* approval of
  one named version. It misleads only when it is taken before the freeze,
  where it points at a tip the implementation replaces.
- Nothing revokes a closing made in conversation. A commit pushed after
  the freeze leaves the request carrying an approval of a version that no
  longer exists; detect it by comparing the branch tip with the freeze
  commit, say so before pushing or handing over, and ask for the gate
  again.
- Closing the discussion without reading the threads implements the
  record as the author remembers it, not as the discussion left it; the
  reconciliation is what makes the closing an approval.
- Opening the draft before the design is written, or after the task list
  is, turns the gate into a formality: the reviewer sees either half a
  package or a method.
- Freezing before the implementation deliberation closes has the same
  effect on the second gate: the reviewer is handed an archived record
  and can only ask for a follow-up change.
