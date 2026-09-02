# The Management Model

Read when deciding which semantics the project earns. This is the complete
platform-neutral vocabulary: every concept below expresses a real management
fact that exists whether or not any platform has an object for it. Do not
extend the catalog for completeness — a new concept enters only with a
management fact nothing here can express and a deletion-test answer of its
own.

For each concept: what it asserts, and the deletion test — the question to
answer before enabling it in a given project. An entity with no concrete
answer is not enabled.

## Work

- **Tracked Work** — a piece of work, problem, or investigation whose
  lifecycle is worth following independently. Not every TODO: a note in a
  change description is cheaper than an item nobody will groom.
  *Deletion test: without it, would anyone lose track of what is open, who
  has it, or why it exists?*
- **Work Hierarchy** — parent/child decomposition of tracked work. A child
  exists only when it independently earns its own state, owner, discussion,
  dependencies, or lifecycle. Hierarchy follows management complexity, never
  code size or duration: a three-day training run with simple semantics is a
  child task; a ten-line change needing its own interface discussion is
  independent tracked work.
  *Deletion test: without the parent, would the children be hard to
  understand or coordinate?*
- **Dependency** — a real execution constraint: while A is unfinished, B
  technically, factually, or by authority cannot start or finish. Planned
  order is not dependency; sequence preferences are expressed by priority,
  timebox, or a planning surface. The dependency graph stays sparse and
  true.
- **Change Request** — a change to code, configuration, or specification
  that has actually begun. A **Draft Change** is one its author considers
  not yet ready for formal acceptance; it is never a placeholder for future
  work.
- **Delivery** — a versioned or otherwise acceptable result actually handed
  to consumers.
- **Source Marker** — a named, stable source state (a tag- or
  checkpoint-like fact).

## Direction

- **Objective Boundary** — a human-endorsed staged goal with a clear scope
  and a decidable completion condition, usually reached by several pieces of
  work ("CUDA 13 Support", "Public Beta", "v3.0"). It says *what will be
  achieved*.
  *Deletion test: is there a real theme whose completion people need to
  judge, or is work genuinely piecemeal?*
- **Timebox** — a planned execution window with a start and an end. It says
  *when work is planned to happen*. Never merged with Objective Boundary: a
  project may need either, both, or neither.
- **Planning Surface** — a constructed view for filtering, ordering, and
  observing tracked work. It is a view, not a record: it can be deleted and
  rebuilt with no loss of project fact, because the facts live in tracked
  work, change requests, deliveries, and decision records.
  *Deletion test: with it gone, could people still quickly understand
  current work from a filtered list? If yes, it costs more than it earns.*

## Attributes

- **Type** — the nature of a piece of tracked work (Bug, Feature,
  Maintenance, Migration, RFC, Investigation). The name is `Type`, fixed,
  and orthogonal to Work Hierarchy.
- **Status / Priority / Severity / Area / Platform / Owner / Target /
  Risk** — candidate attributes, none universal. An attribute exists only
  when its value would change someone's management behavior. Priority is
  what the team decides to do first; Severity is how objectively bad a
  problem is; the two are never merged. Every added status or field is a
  classification decision every contributor pays on every item.

## Responsibility

- **Ownership** — who is responsible for driving, maintaining, reviewing,
  or answering for a scope.
- **Acceptance** — under what conditions, and by whom, a change or result is
  accepted.
- **Automation** — what machines verify, execute, or deliver without a
  human in the loop.

## Deciding

- **Deliberation** — the open discussion, exploration, and trade-off work
  before a decision exists. A process.
- **Decision Record** — a decision worth keeping: what was decided, why,
  the alternatives, who accepted it, when, and when to revisit. A result.
  Keep the two separate: archiving a deliberation thread is not recording a
  decision, and a decision record is not a place to continue arguing.

## Change Propagation

A cross-cutting dimension, not an entity: how a change in this repository
reaches consumers, instances, governed projects, or real environments. Its
modes and consequences are cataloged in
[overlays-and-propagation.md](overlays-and-propagation.md); record the mode
in every contract, because it drives compatibility, validation, rollout,
rollback, blast radius, and how much human acceptance a change needs.
