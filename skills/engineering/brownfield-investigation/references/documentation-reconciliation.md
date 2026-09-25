# Documentation Reconciliation

Read when the question involves what documents, comments, or decision
records claim, or when a finding contradicts a document. The documents are
evidence to check, never the answer, and never something to edit.

## Inventory

List every source that states how the system works: READMEs, the docs
directory, architecture and design documents, decision records, API
descriptions, runbooks, changelogs, code comments that state behavior,
agent instruction files, and anything the user supplies from outside the
repository. Record each one's location, apparent date or version, and
status where it has one (a decision record marked "proposed" describes a
plan, not the system).

## Extract material claims

A material claim is one a reader would act on: how to build and run it,
what a component is responsible for, how a behavior works (limits, retries,
errors, ordering), who owns which data, what an interface accepts and
returns, and which decision was made. Skip aspiration ("we aim to") and
opinion. Quote each claim with its location.

## Check each claim

Check the claim against the authoritative source for its type (see the
claim-type table in SKILL.md) and mark it:

- **verified** — the source agrees; cite it;
- **CONTRADICTED** — the source disagrees; cite both locations, and raise
  which one is intended as an open question unless an authority already
  settled it;
- **unverifiable** — no reachable evidence can settle it; say what would;
- **stale reference** — a path, command, setting, or link that no longer
  exists at the pinned revision.

A contradiction is not a verdict on which side is wrong. The code may carry
a bug, or the document may be outdated; that is a decision for a human.

## Depth

| Depth | Claims checked |
|---|---|
| ORIENT | the claims a newcomer acts on first: setup and run steps, the architecture overview, the main flows |
| ESTABLISH | every behavioral and interface claim within scope |
| EXHAUSTIVE | every claim within scope, including comments at the boundaries being migrated |

## Output

Beyond the per-claim findings, give each document a disposition — reliable,
partly reliable, or unreliable — with the counts behind it, so later work
knows what it can reuse by link and what it must rebuild.

## Gotchas

- Documentation for another deployment, version, or branch reads as wrong
  when it is only elsewhere; check what it describes before calling it
  contradicted.
- Run documented commands when the user permits; reading a command proves
  nothing about whether it works.
- Generated API descriptions can be stale when their generation step is
  not part of the build.
