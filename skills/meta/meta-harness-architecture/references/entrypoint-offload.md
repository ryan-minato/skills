# Offloading To The Architecture Document

Read when AGENTS.md exceeds its budget, or when an architecture document
(conventionally `ARCHITECTURE.md`) is being created or edited.

## What Moves, What Stays

Move to the architecture document: system structure, tech stack detail,
directory maps, key flows, and recorded design decisions — material agents
need occasionally and humans read too. Keep in AGENTS.md: rules agents need
every session (conventions, commands, checks, safety limits) and every
routing pointer. The architecture document is human-readable prose that
follows public conventions; the entrypoint stays agent-terse. One rule
lives in exactly one of the two.

## The Section-Locating Pointer

Every offloaded topic leaves a pointer at its old position in AGENTS.md.
Only the locating part is fixed; the surrounding sentence is free-form.
A conforming pointer contains both of:

1. a link to the target file, and
2. the target's heading line reproduced byte-exactly in inline code,
   hashes included.

Example (demonstration only — any phrasing works as long as both parts are
present):

```markdown
Stack details and directory map: see [ARCHITECTURE.md](ARCHITECTURE.md),
section `## Tech Stack`.
```

The byte-exact heading is what makes the pointer mechanical: agents locate
the section with heading line lookup rather than reading the whole file —

```bash
grep -n '^## Tech Stack' ARCHITECTURE.md
```

— which composes with the weak-model section-lookup instructions without
any extra convention.

## Keeping Pointers True

Renaming or moving a target section breaks every pointer to it. Update the
pointers in the same change that moves the heading — and hand this rule to
whatever sync mechanism the harness installs, as a forward trigger on the
architecture document.
