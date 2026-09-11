---
name: spec-driven-development
description: >-
  Spec-driven development (SDD) — writes specifications before code and
  runs the specify, clarify, plan, tasks, implement, verify loop from them;
  judges whether the discipline pays and at which level (spec-first,
  spec-anchored, spec-as-source); settles how specs meet tracked work —
  when the draft opens, whether a spec needs its own PR, when to archive,
  what issues and PRs link instead of restate; and converts existing code
  — a prototype or brownfield codebase — into a spec-driven project
  without backfilling specs. Use when adopting or starting spec-driven
  development or SDD; to "write the spec first" or "do this spec-driven";
  which of Spec-Kit, OpenSpec, Kiro, or plain documents to use; how
  issues, PRs, and specs fit together, where a spec is reviewed, or when
  to archive a change; when a prototype needs specs before more features;
  when what was built drifts from what was agreed; or when issues and
  specs disagree about acceptance. Not for defining goals, or for building
  the platform harness a builder owns.
license: Apache-2.0
---

# Spec-Driven Development

A specification is a structured, behavior-oriented artifact in natural
language that states what software must do and serves as the instruction an
agent implements from. Spec-driven development (SDD) makes that artifact the
first thing written and the thing verification is judged against, so the
agreement about *what* survives the session that wrote the code.

This skill is the methodology: what the practice is, when it pays, how the
loop runs, what a good specification looks like, and how existing code
enters. The project's own rules — level, tool, how change requests carry
the record, who approves and how, who archives and when — are not set
here: they live in the project's specification contract, and a harness
builder in the `meta` catalog writes that contract (see [Setting up or
improving the project's rules](#setting-up-or-improving-the-projects-rules)).

## Three levels

| Level | The spec is... | Obligation after the change ships |
|---|---|---|
| spec-first | written before the change and used to build it | none — it may be archived or discarded |
| spec-anchored | kept as the living description of the feature | every behavior change updates the spec first |
| spec-as-source | the only file humans edit; code is regenerated | humans never patch code by hand |

Spec-anchored fits anything maintained beyond one release and spec-first a
bounded delivery nobody will evolve; spec-as-source is experimental and
only worth it when the toolchain regenerates code reliably. The level fixes
what must be maintained, so it is chosen before the tool.

**A drifted spec is an active source of falsehood.** When the spec and the
code disagree, the next agent trusts the spec and builds on a lie. Fix the
spec or delete it before any other work; never leave a stale spec standing
because "the code is right".

## When it pays

Use SDD when requirements can be stated before the code exists, the change
outlives one session, several people or agents touch the same behavior, or
acceptance keeps being argued after the fact. Skip it for throwaway
exploration, a spike whose purpose is to discover the requirements, and a
prototype still in its validation window — there, code is the cheapest
spec. Default: adopt SDD the moment a prototype gets its first user who is
not its author.

## Approach families

Each family fits a situation; a spec tool the project already runs is the
answer unless the user asks to change it. When asked directly which to
use, give the fitting family with its reason and say that the choice, and
the rules that go with it, are recorded by the project's harness builder,
not by this skill.

- **A spec-first kit with a project constitution and per-feature spec,
  plan, and task files** (GitHub Spec-Kit) fits a new application
  delivered feature by feature: the heavier, whole-process shape a new
  codebase needs to form habits. The kit does not oblige anyone to keep a
  spec current after the feature ships; at spec-anchored that obligation
  is a written project rule.
- **A spec-anchored change workflow with specs organized by capability**
  (OpenSpec) fits a library, framework, or infrastructure with no code
  yet — a library's contract is its capabilities, not a sequence of
  features — and any existing code: it is lighter, built for existing
  systems, and never asks for specs of code that is not changing.
- **An IDE's native requirements, design, and tasks files** (Kiro) fit a
  team that lives in that IDE; whether other agents honor them is
  unverified, so a tool-agnostic family fits better when several agents
  work the repository.
- **Committed specification documents** under one directory, linked from
  tracked work, fit a team that refuses tooling: the same discipline with
  a hand-run loop.
- **A custom layout** fits only a stated constraint none of the above
  meets, and costs every future agent the tool's validation and
  conventions.

Tool commands and file layouts change between releases: verify them from
the tool's own `--help` and current documentation before running or
describing one. This skill deliberately lists none.

## The loop

Run every change through these steps with the chosen tool's equivalent
command for each. This skill pairs with `plan-clarification` for step 2,
the interrogation the clarify step needs. If it is not installed, load the
`ryan-minato-skills-installing` skill and install `plan-clarification` as
it directs; never run an install command yourself. If the user declines,
run step 2 from the questions in this file.

1. **Specify.** Write the requirements and their scenarios for this change
   only, plus non-goals. Done when: every requirement has at least one
   scenario a reader who has never seen the code could judge.
2. **Clarify, then publish.** Interrogate every ambiguity, unstated
   assumption, and missing edge case with the user; record answers in the
   spec, not in the chat. Then commit the change record and open the draft
   change request (or the specification change request, under the split
   shape) at once, before any plan exists, so review starts on the
   specification — and stop there: no design, tasks, or code until the
   contract's approval mode is satisfied (the gate owner closes the
   discussion in conversation, or posts the fixed comment). Done when: no
   `[NEEDS CLARIFICATION]`-class marker remains, the draft is published,
   and the agent is waiting.
3. **Reconcile, then plan.** When the gate owner closes the discussion,
   read the request's comments and review threads with their resolution
   state; list every unresolved thread, every adjustment requested in the
   discussion that the record does not carry, and every pair of
   conclusions that contradict each other; ask the gate owner to confirm
   the open items; record the closing on the request. Only then derive the
   technical design from the spec and the project's constraints
   (constitution, architecture, conventions), keeping design out of the
   requirements file. Done when: nothing is open or the open items are
   confirmed, every requirement maps to a design decision, and every
   decision names the requirement it serves.
4. **Tasks.** Break the plan into ordered, independently verifiable tasks
   with the scenarios each one closes. A tool that generated design and
   task files together with the spec has produced drafts; they are the
   implementer's to finish now, and they were never the approval gate's
   object. Done when: no task lacks a scenario and no scenario lacks a task.
5. **Implement.** Work task by task; when the code must deviate from the
   spec, stop and change the spec first, with the user's approval. Done
   when: every task is closed or its deviation is recorded as an approved
   spec change.
6. **Verify.** Execute the scenarios — tests, commands, observed states —
   against the running result, not against the diff. Done when: every
   scenario has passed or is recorded as a spec change.
7. **Converge or archive.** Write the delivered behavior back into the
   source-of-truth spec (spec-anchored) or archive the change record
   (spec-first), when and by whom the contract's archive mode says. Done
   when: the spec and the code describe the same system.

Read [references/tracked-work-lifecycle.md](references/tracked-work-lifecycle.md)
when a step meets the project's tracked work — publishing the draft,
waiting for the approval, drafting the request body, archiving — for what
the agent does under each shape and mode.

## Specification quality

- One requirement, one normative statement (SHALL or MUST); split compound
  requirements.
- Every requirement carries at least one scenario in a given / when / then
  shape, and one of them covers an edge or failure path.
- Non-goals are written down; an unstated exclusion is a future dispute.
- Scope requirements as *now*, *later*, and *out of scope*; only *now* enters
  the plan.
- Requirements and design live in separate files; a requirement that names
  a class or a table is a design decision in disguise.
- A spec describes observable behavior, never the diff: "the API returns
  409 on a duplicate email" is a requirement; "add a unique index" is not.

## What specification review examines

The approval gate reviews the description of the outcome, each item as the
project needs it: goals and scope; terminology and the domain model;
behavior; invariants; constraints and rules; states and their transitions;
interface contracts; data contracts; exceptions and edge cases; security
and permissions; metrics and acceptance criteria. It never reviews tasks
or design: those describe how the outcome is built, belong to the
implementer after approval, and are judged by implementation review. A
reviewer who is handed the task list is being asked to approve a method,
not an outcome; hand them the specification instead.

## Project rules live in the contract

The project's specification contract — `.agents/knowledge/spec-workflow.md`
by default, or the file the agent entrypoint points to — records the
level, the tool and its artifact map, the change request shape (combined
or split), the approval owner and how approval is recorded, the archive
mode, and what tracked work links. Apply those facts without asking them
again; the specification owns *what*, *why*, and the acceptance criteria,
tracked work owns *who*, *when*, and *status* and links the record, and
acceptance criteria exist in exactly one place.

When no contract exists, apply these defaults, say so, and name the
harness builder below as the way to settle and record them — never run a
questioning round of your own:

- **Shape**: combined — one change request carries the record from the
  moment it is committed, opens as a draft, and the gate is exercised on
  that draft; split only where consumers depend on a stable contract.
- **Approval**: discussion-closed — the gate owner discusses on the
  draft, directs record changes in conversation, and declares the
  discussion closed in conversation; the record of approval is that
  closing plus the request's discussion state, and the reconciliation
  above precedes any design. A blocking comment is the alternative only
  where the contract or the user asks for one. A platform review approval
  is never the record, because later pushes dismiss it.
- **Archive mode**: in-request — the request archives its record before
  it is marked ready; automated archiving needs a push path the harness
  builder records.

## Adopting existing code

Read [references/adopting-existing-code.md](references/adopting-existing-code.md)
when the project already contains code that was not written from a
specification — a prototype, a vibe-coded app, or a brownfield codebase.
Its last step hands the friction log to the harness builder below;
adoption is not finished until that hand-off has been offered and its
outcome recorded in the project.

## Setting up or improving the project's rules

Initializing a project's spec-driven rules, or improving them — the
contract, the tool's layout, the templates and forms, the project skill's
steps, the archive automation — is the work of the spec workflow builder
in the `meta` catalog of https://github.com/ryan-minato/skills, a
disposable harness builder that settles each rule with the user and
deposits the contract and the platform expression. When the user asks for
that, load the `ryan-minato-skills-installing` skill and install the whole
`meta` catalog at project scope as it directs — its builders stack and are
disposed together; never run an install command yourself. (If that
installer skill is absent too, it lives in the `core` catalog of the same
repository.)

If the user declines, or the catalog is unavailable, record the defaults
above — and which of them the project departs from — in the project's
knowledge base, and list the harness build as remaining work. Do not edit
templates, forms, checks, automation, or a project skill: those stay the
builder's.

## Gotchas

- Tools rename their commands between releases; a command remembered from a
  blog post is the most common way an SDD setup fails on day one.
- Spec-Kit's feature script creates a numbered spec directory, not a git
  branch; branch creation follows the project's branching contract.
- A tool's lowercase `design.md` is a technical design file. `DESIGN.md` at
  the repository root is a reserved name for the visual-design format; never
  rename one into the other.
- Backfilling specs for code nobody is changing feels productive and rots
  immediately: nothing forces those specs to track reality.
- The goal document (what the project must achieve) sits above every spec;
  a constitution or steering file may cite it, and no spec may contradict it.
- Requirements written by observing a prototype record what the prototype
  happens to do; only the user can say which of that behavior was intended.
- A review approval left on a draft does not survive the implementation
  pushes — removed by default on GitLab, by a stale-approval rule on
  GitHub, and stale in meaning everywhere; the record of approval is the
  closing of the discussion or the fixed comment, as the contract says.
- Closing the discussion without reading the threads implements the
  record as the author remembers it, not as the discussion left it; the
  reconciliation is what makes the closing an approval.
- Under the split shape a delta written against a domain spec that another
  change archived later may no longer apply; re-validate the delta when
  implementation starts.
