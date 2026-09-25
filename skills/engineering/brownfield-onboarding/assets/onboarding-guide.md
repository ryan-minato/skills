# <Project name>: onboarding guide

Describes revision <revision> as of <date>. Every statement cites the
finding behind it; a command marked unverified could not be run here, and
the reason is given.

## Purpose

<What the system does and for whom, in two or three sentences.>

## Running it

<Link the setup section of the project's documentation when it checked
out; list below only what it lacks or gets wrong.>

| Task | Where documented, or the command when it is not | Verified |
|---|---|---|
| Build | <link, or command> | <yes / unverified: reason / fails: evidence> |
| Run | <link, or command> | <yes / unverified: reason / fails: evidence> |
| Test | <link, or command> | <yes / unverified: reason / fails: evidence> |

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
