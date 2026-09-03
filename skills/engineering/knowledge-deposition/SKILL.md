---
name: knowledge-deposition
description: >
  Deposits a confirmed piece of knowledge into a project so future agents
  find and follow it — probes where the project already keeps agent-facing
  guidance, chooses the right carrier (entrypoint line, knowledge-base
  file, project skill, or nothing yet), writes the deposit as standalone
  instructions, and registers an event-triggered pointer. Use when the user
  says "record this for the project", "remember this from now on", "add
  this to AGENTS.md or CLAUDE.md", "write this down so we stop repeating
  it", or approves findings from a session retrospective; or whenever a
  lesson worth keeping has just been confirmed and needs a durable home.
  Not for mining a session for lessons — that is a retrospective's job —
  and not for building or restructuring a project's whole agent setup; it
  adds one piece of knowledge to the conventions that already exist.
license: Apache-2.0
---

# Knowledge Deposition

Input: one or more confirmed pieces of knowledge — approved retrospective
findings, or something the user just told you to record. Output: each piece
deposited where the project already keeps such knowledge, reachable through
an event-triggered pointer, and usable by an agent that never saw this
conversation. The unit of work is the deposit: carrier chosen, content
written, pointer registered, all three verified.

## Only confirmed knowledge

Deposit only what the user has confirmed, item by item. A finding produced
by an agent — this session's or another skill's — is a proposal until the
user rules on it; "an upstream process approved it" never substitutes for
that verdict.

## Probe before creating

The project's existing conventions decide where knowledge lives; never
assume a platform or a layout. In order:

1. Find the agent entrypoint — AGENTS.md, CLAUDE.md, or whatever this
   platform reads first. Its name and contents reveal the conventions.
2. Follow the entrypoint's pointers to where knowledge already lives: a
   knowledge directory beside it, a docs tree it points into, an existing
   conventions document.
3. Find where project skills, commands, or workflow definitions live, if
   the platform has such a mechanism.

Deposit into what you find, matching the neighbors — same directory, same
naming style, same file shape. Create a new location only when none exists,
and then the lightest one that works.

## Choose the carrier

Prefer the knowledge forms the project already uses; introduce a new form
only when the project has none that fits. Within the available forms:

- **Entrypoint line** — the entrypoint loads in every session, which makes
  it the most expensive location in the project. Reserve it for what every
  session must see: safety constraints, invariants that must never be
  violated, red lines around destructive operations. Everything else
  reaches agents through a pointer instead.
  Read [references/carrier-entrypoint-line.md](references/carrier-entrypoint-line.md)
  when the chosen carrier is an entrypoint line.
- **Knowledge-base file plus an event pointer** — the default destination
  for most deposits: facts, conventions, short contracts, where-things-live
  knowledge — anything needed only when a specific event occurs.
  Read [references/carrier-knowledge-file.md](references/carrier-knowledge-file.md)
  when the chosen carrier is a knowledge-base file.
- **Project skill** — or the platform's equivalent workflow or command
  unit — for a procedure that is recurring, non-obvious, and at least one
  of fragile, order-sensitive, or branchy. A skill a plain sentence or a
  single file could replace charges description rent on every session
  forever; take the cheaper carrier instead.
  Read [references/carrier-skill.md](references/carrier-skill.md) when the
  chosen carrier is a project skill.
- **Nothing yet** — knowledge that is confirmed but has not recurred, or
  is still changing, stays in the findings list or the conversation.
  Tell the user it is parked and what recurrence would justify depositing
  it; a premature deposit is rent with no payoff.

Related items consulted at the same moment go into one file — one lookup,
one file. Anti-patterns: non-critical information promoted into the
entrypoint; a skill wrapping one command; a "general tips" deposit no
trigger can fire; one procedure split across carriers.

## Write the deposit

Rules that hold for every carrier: state the knowledge as an instruction
the next agent executes, not as history — "we found that the seeder
breaks…" is history; "run the migration before seeding, because the seeder
validates against the live schema" is a deposit. Nothing in the deposit
may reference this session. Carrier-specific mechanics live in the
reference chosen above.

## Verify the deposit

Before reporting done, check each deposit:

- From the entrypoint alone, the pointer (or skill description) resolves,
  and its trigger names a real event an agent will recognize.
- From the deposited content alone, a fresh agent can act correctly.
- Nothing in the deposit names this conversation, the retrospective, or
  this skill.

## Scope and precedence

`agentic-writing` applies as the baseline when it is also active: this
skill decides what is worth keeping and where it lives; that one owns how
the text itself is written. This skill grows a project's guidance one
confirmed lesson at a time during normal work — it does not design
entrypoints, reorganize knowledge trees, install sync or review
mechanisms, or build an agent setup from scratch. When a project has no
conventions at all, deposit the minimum — one file plus one pointer — and
tell the user a proper setup pass is a separate job.

## Pairs with

To mine a session for lessons before depositing, this skill hands off to
`session-retrospective`. If it is not installed, load the
`ryan-minato-skills-installing` skill and install `session-retrospective` as
it directs; never run an install command yourself. If the user declines,
deposit the lesson they already confirmed and skip the mining step.
