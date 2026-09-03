---
name: spec-driven-development
description: >-
  Spec-driven development (SDD) — writes specifications before code and
  runs the specify, clarify, plan, tasks, implement, verify loop from them;
  judges whether the discipline pays and at which level (spec-first,
  spec-anchored, spec-as-source); and converts existing code — a vibe-coded
  or hacked-together prototype, a brownfield codebase — into a spec-driven
  project without backfilling specs. Use when asked to adopt, introduce, or
  start spec-driven development or SDD, or where to start making a
  hacked-together app maintainable; to "write the spec first" or "do this
  spec-driven"; which of Spec-Kit, OpenSpec, Kiro, or plain documents to
  use; when a weekend prototype needs proper specs before more features;
  when what was built drifts from what was agreed; or when issues and specs
  disagree about acceptance. Not for defining goals, or for the harness,
  tracker, and template setup a builder owns.
license: Apache-2.0
---

# Spec-Driven Development

A specification is a structured, behavior-oriented artifact in natural
language that states what software must do and serves as the instruction an
agent implements from. Spec-driven development (SDD) makes that artifact the
first thing written and the thing verification is judged against, so the
agreement about *what* survives the session that wrote the code.

## Three levels

Pick the level before the tool: the level fixes what must be maintained.

| Level | The spec is... | Obligation after the change ships |
|---|---|---|
| spec-first | written before the change and used to build it | none — it may be archived or discarded |
| spec-anchored | kept as the living description of the feature | every behavior change updates the spec first |
| spec-as-source | the only file humans edit; code is regenerated | humans never patch code by hand |

Default to **spec-anchored** for anything maintained beyond one release and
**spec-first** for a bounded delivery nobody will evolve. Treat
spec-as-source as experimental: choose it only when the user names it and
the toolchain regenerates code reliably.

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

## Choose the approach

One default per situation; the user decides, and a recommendation is not a
decision. A spec tool the project already runs is the answer unless the
user asks to change it. Present the default with its reason and the one
deviation.

- **No code yet, and the product is an application delivered feature by
  feature:** a spec-first kit with a project constitution and per-feature
  spec, plan, and task files — GitHub Spec-Kit. It is the heavier,
  whole-process option, organized around requirements and a solution per
  feature; that weight is what a new application needs to form habits.
  The kit does not oblige anyone to keep a spec current after the feature
  ships; at spec-anchored, write that obligation down as a project rule.
  Deviate to a lighter change workflow when the team already resists
  process.
- **No code yet, and the product is a library, framework, or
  infrastructure:** a spec-anchored change workflow whose specs are
  organized by capability — OpenSpec. A library's contract is its
  capabilities, not a sequence of features, and a per-capability spec that
  each change amends is the shape that stays true; the feature-shaped kit
  fragments such a contract into deliveries. Deviate to the kit when the
  library ships as a product with user-facing features.
- **Existing code (brownfield, a vibe-coded prototype), whatever the
  product shape:** the spec-anchored change workflow — OpenSpec. It is
  lighter, built for existing systems, and never asks for specs of code
  that is not changing. Deviate to the kit only when the codebase is being
  rewritten feature by feature from scratch.
- **The team lives in Kiro:** its native requirements, design, and tasks
  files. Whether other agents honor them is unverified; deviate to a
  tool-agnostic option when more than one agent works the repository.
- **The team refuses tooling:** committed specification documents under one
  directory, linked from tracked work. Same discipline, hand-run loop.
- **Custom layout:** only when none of the above fits a stated constraint,
  and only after the user hears that a custom format costs every future
  agent the tool's validation and conventions.

Tool commands and file layouts change between releases: verify them from
the tool's own `--help` and current documentation before running or
describing one. This skill deliberately lists none.

## The loop

Run every change through these steps, using the chosen tool's equivalent
command for each. This skill pairs with `plan-clarification` for step 2,
the interrogation the clarify step needs. If it is not installed, load the
`ryan-minato-skills-installing` skill and install `plan-clarification` as
it directs; never run an install command yourself. If the user declines,
run step 2 from the questions in this file.

1. **Specify.** Write the requirements and their scenarios for this change
   only, plus non-goals. Done when: every requirement has at least one
   scenario a reader who has never seen the code could judge.
2. **Clarify.** Interrogate every ambiguity, unstated assumption, and missing
   edge case with the user; record answers in the spec, not in the chat.
   Done when: no `[NEEDS CLARIFICATION]`-class marker remains and the user
   has approved the spec.
3. **Plan.** Derive the technical design from the spec and the project's
   constraints (constitution, architecture, conventions). Keep design out of
   the requirements file. Done when: every requirement maps to a design
   decision and every decision names the requirement it serves.
4. **Tasks.** Break the plan into ordered, independently verifiable tasks
   with the scenarios each one closes. Done when: no task lacks a scenario
   and no scenario lacks a task.
5. **Implement.** Work task by task; when the code must deviate from the
   spec, stop and change the spec first, with the user's approval. Done
   when: every task is closed or its deviation is recorded as an approved
   spec change.
6. **Verify.** Execute the scenarios — tests, commands, observed states —
   against the running result, not against the diff. Done when: every
   scenario has passed or is recorded as a spec change.
7. **Converge or archive.** Write the delivered behavior back into the
   source-of-truth spec (spec-anchored) or archive the change record
   (spec-first). Done when: the spec and the code describe the same system.

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

## Specs versus tracked work

The specification owns *what* and *why* and the acceptance criteria.
Tracked work — issues, tasks, tickets, change requests — owns *who*, *when*,
and *status*, and links the specification it serves. Acceptance criteria
exist in exactly one place, the spec; an issue that restates them will
disagree with the spec within a release.

- Create work items from the spec's task list, each linking the scenarios it
  closes; never write a work item whose acceptance cannot be found in a
  spec.
- A change request (pull or merge request) names the spec or change record
  it implements and states whether the spec was updated.
- Spec lifecycle (proposed, approved, implemented, archived) and work-item
  status are different facts; neither substitutes for the other.
- Objectives, milestones, and roadmaps are human-endorsed goals; a spec
  serves one, it does not become one.

## Adopting existing code

Read [references/adopting-existing-code.md](references/adopting-existing-code.md)
when the project already contains code that was not written from a
specification — a prototype, a vibe-coded app, or a brownfield codebase.
Its last step is the harness alignment below; adoption is not finished
until that hand-off has been offered and its outcome recorded in the
project.

## Harness alignment

Once the level and tool are settled, the project's agent harness — its
entrypoint, knowledge base, issue and pull-request templates, and tracker
conventions — must state the same facts the tool assumes: where specs live,
which file is the source of truth for behavior, and that work items link
specs instead of restating them. This skill pairs with the disposable
harness builder for spec workflows, `meta-spec-workflow`. If it is not
installed, load the `ryan-minato-skills-installing` skill and install the
whole `meta` catalog at project scope as it directs — its builders stack and
are disposed together; never run an install command yourself. (If that
installer skill is absent too, it lives in the `core` catalog of
https://github.com/ryan-minato/skills.) If the user declines, record the
level, tool, artifact paths, and source-of-truth decisions in the project's
knowledge base by hand and list every harness file that still restates a
requirement.

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
