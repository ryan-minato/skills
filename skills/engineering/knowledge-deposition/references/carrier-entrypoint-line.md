# Carrier: Entrypoint Line

Load this when the chosen carrier is an entrypoint line.

## Re-check the bar first

Before writing, ask once more: must every session see this, even sessions
doing unrelated work? The entrypoint is paid for in every conversation, so
only safety constraints, never-violate invariants, and red lines around
destructive or irreversible operations clear the bar. If the honest answer
is "only sessions doing X need it", this is a knowledge-file deposit with
an event pointer — switch carriers and read that reference instead.

## Write the line

- Imperative, one or two sentences at most. If it takes more, that is
  evidence it belongs in a file behind a pointer.
- Where the rule leaves judgment room, keep the reason as one clause:
  "Never run migrations against the shared staging database — other teams'
  test runs depend on its state."
- Search the entrypoint for an existing line with the same meaning before
  adding one; if a line already covers it, sharpen that line rather than
  adding a second statement of the same rule.

## Place the line

Put it beside the entrypoint's related existing rules, inside whatever
grouping the file already uses — a constraints section, a bulleted
conventions block. Match the neighbors' format exactly; do not introduce a
new section for one line unless no group fits.

## Demote when lines pile up

An entrypoint that accumulates lines loses the very visibility that made it
worth the rent. When adding a line to a cluster of related ones, propose
moving the non-critical members of the cluster into a knowledge file behind
one event pointer, keeping only the must-see-always core inline.
