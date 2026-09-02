---
name: session-retrospective
description: >
  Distills a work session into durable project lessons — scans the
  conversation for repeated failures, fixes that linters and tests kept
  demanding, information that took many hops to find, experiments run to a
  verdict, agent choices the user overruled, and behavior that contradicted
  the docs — then presents a ranked findings list for approval. Use when a
  task or session wraps up, or when asked to reflect on the work — "what did
  we learn here", "summarize the takeaways", "let's not repeat that
  mistake", "remember this for next time", "do a retrospective or postmortem
  of this session". Produces findings only; nothing is written until the
  user approves each one. Not for saving an already-confirmed lesson into
  the project — that is knowledge-deposition — and not for status reports or
  code review.
license: Apache-2.0
---

# Session Retrospective

The session history is evidence that evaporates the moment the session ends:
every wrong turn, correction, and hard-won discovery disappears unless it is
mined now. This skill walks back through the conversation, extracts the
lessons worth keeping, and turns them into findings the user rules on. Its
deliverable is the approved findings list — never an edited file.

## Findings first

This skill reads history and writes nothing. A failure pattern, however
often it repeated, justifies a finding and a recommendation — not an edit to
the project's guidance files. Every finding passes through the user's
verdict before anything durable happens, so the retrospective ends at the
approval gate, and depositing approved findings is a separate act.

## Mine the six signals

Walk the session from the beginning and collect candidates in each class:

| Signal | What to look for | What the finding captures |
|---|---|---|
| Repeated failures | The same error class hit more than once, or the user saying it has happened before — user testimony counts as recurrence evidence across sessions | The cause and the working avoidance or fix |
| Tool-flagged corrections | Issues linters, formatters, type checkers, or tests raised repeatedly and were fixed the same way each time | The convention that prevents the flag at write time |
| Expensive discoveries | Information located only after many hops, searches, or dead ends | Where the information lives — not the search that found it |
| Experiment verdicts | Alternatives tried in parallel or sequence and evaluated to a choice | The chosen option and the one-line reason, so losing branches are not re-run |
| Overruled defaults | A practice chosen as best practice that the user rejected, plus the replacement they accepted | The correction and its reason — the user's way is the knowledge |
| Docs–reality mismatches | Test or runtime behavior that contradicted the project's own docs or a reference document | What is actually true and which document is wrong |

## The bar: recurrence times impact

One occurrence is an anecdote. Propose recording it only when it recurred,
when the user testifies it recurs across sessions, or when a single
occurrence was expensive enough that one more is unacceptable. Weigh
against that the rent every recorded lesson charges: it occupies context in
future sessions whether or not it ever fires again. Skip candidates the
environment answers on demand — a fact one `--help` call or file read
reveals is already durable, and the tool never goes stale while a note
about it can.

When a candidate sits on the line, propose it anyway and let the verdict
settle it: the approval gate exists precisely so borderline findings are
cheap to raise and cheap to drop.

## Draft the findings

Read [references/finding-format.md](references/finding-format.md) when
drafting the findings list. Each finding states its proposed rule as an
instruction a future agent could follow without ever seeing this session —
no "as discussed above", no references to this conversation.

## Present and gate

Present the findings in the conversation, ranked by recurrence times
impact, each with its recommended destination attached. Ask for a verdict
on each finding — keep, edit, or drop — rather than a blanket yes; a
blanket approval hides the one finding the user would have vetoed.

Done when: every finding has an explicit per-item verdict.

## Hand off approved findings

Approved findings need a durable home chosen by the project's own
conventions. This skill pairs with `knowledge-deposition` for that step. If
it is not installed, install it from
https://github.com/ryan-minato/skills.git:

    npx skills add ryan-minato/skills --skill knowledge-deposition

If the user declines the install, the approved findings list presented in
the conversation is the deliverable; write findings only to locations the
user explicitly names.

## Scope and precedence

`agentic-writing` applies as the baseline when it is also active — anything
this skill helps the user save is an agent-first document. This skill
evolves an existing setup from lived evidence during normal work; designing
or restructuring a project's agent setup from scratch is a different job
and does not trigger here.
