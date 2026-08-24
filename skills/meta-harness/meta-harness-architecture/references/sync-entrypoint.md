# Entrypoint-Family Sync Mechanisms

Read when the plan's sync family is the entrypoint or knowledge documents.
Without skill auto-triggering, the rules must sit where agents already
look: the entrypoint's workflow section and a sync table.

## The Sync Table

One row per concern, in the entrypoint (or a knowledge doc the entrypoint
points to with a load condition). Each row names the change, the document
it updates, and what to inspect before editing that document — forward and
reverse in one line. Keep rows concrete: "renamed a section in the
architecture document" beats "changed docs".

## Workflow Entries

Changes that recur in every task (commands, checks, conventions) earn a
line in the entrypoint's task workflow — "when your change alters a
command or check, update the entrypoint's command table in the same
change" — because agents reread the workflow every session, which is the
closest this family gets to auto-triggering.

## The Reverse Rule

Add one standing entrypoint rule covering all documents: before updating
any harness document, inspect the artifacts its sync-table row names, and
fix or report every mismatch found — not only the one that prompted the
edit. This is what makes entropy reclamation happen on every doc touch
instead of never.

## Boundaries

- The table and the workflow entries must not both own the same concern;
  pick per concern.
- If the table outgrows roughly ten rows, the project has enough moving
  parts that the skill family would serve better — say so to the user
  rather than growing the table silently.
