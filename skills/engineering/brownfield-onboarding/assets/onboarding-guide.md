# <Project name>: onboarding guide

Describes revision <revision> as of <date>. Statements marked unverified
were not checked; everything else cites the evidence behind it.

## Purpose

<What the system does and for whom, in two or three sentences.>

## Running it

| Task | Command or steps | Verified |
|---|---|---|
| Build | <command> | <yes / unverified: reason> |
| Run | <command> | <yes / unverified: reason> |
| Test | <command> | <yes / unverified: reason> |

<Prerequisites, and a link to the setup document when it checked out.>

## Components

| Component | Responsibility | Location | Entry points |
|---|---|---|---|
| <name> | <one line> | <path> | <routes, consumers, jobs, commands> |

## Vocabulary

| Term | Meaning here | Defined in | Notes |
|---|---|---|---|
| <term> | <meaning> | <path or document> | <synonyms; other meanings elsewhere> |

## Architecture overview

<Components, the boundaries between them, external systems, and which
component owns which data — as observed at the revision above.>

## Representative flows

### <Capability>

<Entry point → main hops → state changes and side effects → what the
caller receives, each hop with its location.>

## Where to change what

| To change | Start at | Also moves |
|---|---|---|
| <kind of change> | <path> | <tests, configuration, other components> |

## Risks and unknowns

| Item | Status | Evidence | Next step |
|---|---|---|---|
| <claim or behavior> | <CONTRADICTED / PENDING_DECISION / unprotected / UNKNOWN> | <locations> | <where to look, or whom to ask> |

## Further reading

1. <Document that checked out> — <why read it, and when>
