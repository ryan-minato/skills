# Adopting Spec-Driven Development in Existing Code

Read when the project already contains code that was not written from a
specification: a prototype, a vibe-coded application, or a brownfield
codebase. The loop in SKILL.md still applies; this reference covers what
happens before the first change enters it.

## Rule one: specify only what changes next

Do not document the whole codebase. Write specifications for the behavior
the next change touches, and nothing else. Specs for untouched code have no
change to keep them honest; they drift the day they are written and then
mislead every agent that trusts them. The substitute for a full spec set is
the codebase map below, which is cheap to refresh and never claims to be
normative.

## 1. Assess what exists

1. Run the software and observe its actual behavior; do not infer behavior
   from reading code alone. A prototype's code is a reference
   implementation, not a source of truth.
2. Inventory every document that states requirements or acceptance: README
   sections, design notes, open issues, comments in the code, chat exports
   the user offers. Each will later become a spec, a pointer to a spec, or
   be deleted.
3. Separate intended behavior from accidental behavior. List every observed
   behavior the code exhibits that no document explains, and ask the user
   to rule on each: keep as a requirement, keep as a non-goal, or fix.
   Never promote an accident to a requirement on your own.
4. Record the verification the code already has — tests, manual checks,
   nothing — because the first spec scenarios must be runnable against it.

Done when: the user has ruled on every unexplained behavior, and every
requirement-bearing document has a disposition.

## 2. Build the codebase map

The map is an as-built description for agents, not a specification. One
file, or one file per module for a large codebase, holding for each module:
responsibility, entry points, conventions it follows, known problems, and
integration points with other modules and external systems. Name it by the
project's convention for agent knowledge; never `DESIGN.md`, which is a
reserved name.

- Open every map file with `Mapped at commit <sha> on <date>`.
- Add a drift gate to the harness: before executing tasks in a module,
  compare the recorded commit with `HEAD` for that module's paths and
  re-map the module first when they differ. A map with no gate is a spec
  drift problem under another name.
- When clean-context subagents are available, map modules in parallel and
  reconcile their outputs; each subagent reads only its module and reports
  in the same shape.

Done when: every module the next change touches is mapped, the gate is
registered where the harness runs task execution, and the map claims no
requirement.

## 3. Choose the path

Default: **spec-anchor and evolve**. The existing code stays; every change
from now on enters through the loop with a delta spec, and the
source-of-truth spec grows only where changes land. Choose a **spec-first
rewrite** only when the user confirms the prototype cannot be maintained and
agrees to rebuild it feature by feature from specs, with the old code kept
as a reference until each feature's scenarios pass on the new one.

Then pick the tool with the defaults in SKILL.md — a spec-anchored change
workflow is the default for existing code — and let the user decide. Write
the chosen path, the recommended tool with its reason, and every decision
still open into the project's plan file for this adoption; a
recommendation that lives only in the conversation is lost by the next
session.

Done when: the plan file names the path, the tool recommendation, and the
open decisions, and the user has been told where it is.

## 4. Run a pilot change

Pick one small, safe, user-visible change and take it through the entire
loop before touching anything else: proposal, delta spec with scenarios,
clarification, plan, tasks, implementation, verification, archive or
converge. The pilot exists to expose friction cheaply:

- a scenario that cannot be executed reveals missing test infrastructure;
- a requirement that contradicts the harness or the goal document reveals a
  conflict to resolve before scale;
- a step the tool's command does not fit reveals a convention to record.

Log every friction point; it is the input to harness alignment.

Done when: the pilot's scenarios pass, its spec is archived or merged into
the source of truth, and the friction log exists.

## 5. Reconcile the documents that restate requirements

For every document inventoried in step 1:

- content that is a requirement for behavior being changed → move it into
  the spec and replace the original with a pointer;
- content that is a requirement for untouched behavior → leave it, mark it
  as descriptive rather than normative, and add it to the map;
- open issues carrying acceptance criteria → move the criteria into the
  spec (or the change record) and leave the issue holding a link, owner,
  and status;
- duplicated statements with no stated precedence → keep one, delete the
  rest.

Done when: no two files claim to be the source of truth for the same
behavior, and every open work item's acceptance can be found in a spec.

## 6. Hand the friction log to harness alignment

The harness must now state where specs live, which file rules on behavior,
and that work items link specs. Follow the harness-alignment section in
SKILL.md; if the user declines the builder, write those facts into the
project's knowledge base by hand from the friction log.

Done when: the hand-off was offered, and either the builder ran or the
plan file records the harness files that still restate a requirement and
who will fix them.
