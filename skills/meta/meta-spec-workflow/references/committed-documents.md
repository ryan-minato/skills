# Committed Documents and Custom Layouts

Read when the selected approach is committed specification documents with
no tool, or a custom layout. Both mean the loop runs by hand, so the layout
has to supply the structure a tool would otherwise impose.

## Minimal layout

One directory, named by the project's convention (`specs/` is the common
choice), holding:

- `specs/<domain>.md` or `specs/<domain>/spec.md` — the current behavior of
  each domain: numbered requirements, one normative statement each, with
  scenarios in a given / when / then shape.
- `specs/changes/<change-name>/` — one directory per change in flight:
  a proposal (why, scope, non-goals), the delta (added, modified, and
  removed requirements against the domain spec), an optional design, and
  a task list.
- `specs/changes/archive/<date>-<change-name>/` — completed changes, after
  their deltas were merged into the domain spec by hand.
- A principles file when the project has principles worth stating once —
  place it beside the specs, not in the entrypoint.

For spec-first delivery the change directory is the whole artifact and the
domain spec is omitted; say which in the contract.

## What the layout must supply that a tool would

- **Validation:** without a validator, the change-request template's
  checklist is the only gate. Put "every requirement has a scenario; the
  delta names every changed requirement" on it, and add a lightweight
  check to the project's checks when the team will maintain one.
- **Archiving:** a written rule for who merges the delta into the domain
  spec and when — at change acceptance, by the change's author — because
  nothing does it automatically.
- **Discovery:** the entrypoint pointer from the contract is the only way
  an agent finds the layout; there are no tool-installed commands.

## Custom layout

Choose a custom layout only for a constraint none of the tools meets, and
record that constraint in the contract. Keep the same three roles — domain
spec, change record, archive — under whatever names the constraint
dictates, so a future migration to a tool is a move, not a rewrite. Every
future agent pays for the missing tool with judgment on every change; the
user must hear that before choosing it.

## Collision points with a harness

| Fact | Harness file that tends to restate it | Resolution |
|---|---|---|
| Behavior of a domain | knowledge-base files, README behavior sections | The domain spec rules; others link |
| Acceptance of a change | intake templates, work items | Work items link the change; acceptance is never copied |
| Principles | `AGENTS.md`, conventions files | The principles file rules; the entrypoint points |

## Record for the contract

Level, the layout with every path, the hand-run archive rule, the
validation substitute, the discovery pointer, and — for a custom layout —
the constraint that justified it.
