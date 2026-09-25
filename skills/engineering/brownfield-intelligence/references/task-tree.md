# Task Tree

Read when creating, updating, or resuming the task tree.

## Where it lives

In the workspace as `task-tree.md`, beside the scope record, the ledger,
and the decision records. Without a workspace, keep the tree in the
conversation and restate it whenever it changes.

## Node fields

- **Id** — stable across sessions (`T1`, `T1.2`).
- **Question** — what the node answers or produces.
- **Member** — the suite member that owns it, and the depth.
- **Status** — one of: pending, running, blocked (naming the decision id
  or the missing member), done, stale, abandoned (tentative work on a
  branch a ruling did not choose).
- **Depends on** — node ids and decision ids.
- **Evidence** — the finding ids it produced or relies on.
- **Confidence** — the lowest confidence among the findings it rests on.
- **Open questions** — what remains unknown, each marked as answerable by
  more investigation or only by a person.
- **Tentative** — set on exploration done ahead of a pending decision, so
  nobody mistakes it for settled work.

## Starting trees

Adapt to the scope; these are the usual shapes.

**Onboarding**

- repository map (ORIENT)
  - documentation reconciliation (ORIENT)
  - domain model (ORIENT)
  - runtime flow for two or three representative capabilities (ORIENT)
  - test safety map: does the suite run (ORIENT)
- onboarding material, after all of the above

**Specification**

- repository map (ESTABLISH)
  - contract candidates per boundary (ESTABLISH)
  - runtime flow and data and state for the boundaries in scope
  - test safety map for the candidates
- classification and decision rounds, after the evidence
- promotion and hardening, per ruled contract

**Migration**

- repository map (EXHAUSTIVE for the components being replaced)
  - runtime flow, data and state, contract candidates per boundary
  - test safety map; history for behavior whose origin matters
- compatibility baseline and decision rounds
- characterization tests, per boundary
- equivalence envelope, then its approval
- verification plan

## Updating

- Update a node when its status changes, and record the finding ids it
  produced.
- When a ruling arrives, unblock every node that waited on it, and mark
  tentative work on the branch not chosen as abandoned rather than
  deleting it.
- On resume, mark stale every node whose evidence became stale under the
  drift gate, and re-run it before any node that depends on it.
