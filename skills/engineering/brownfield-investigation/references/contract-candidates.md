# Contract Candidates

Read when collecting behavior that consumers may depend on, or when asked
what must not change. This lens lists candidates with their evidence; it
never decides which of them are contracts.

## What counts as a candidate

A candidate is behavior at a boundary that some consumer could rely on and
that a complete reimplementation could change. Boundaries are wherever a
party outside the component observes it, whether or not that party lives
in another repository: HTTP or RPC interfaces, events and messages, shared
tables, files and formats, command-line output and exit codes, a library's
public interface, plugin hooks, webhooks, and messages sent to people.

Internal structure — classes, call paths, frameworks, the storage engine —
is not a candidate unless a consumer can observe it.

## Procedure

For each boundary in scope, record every behavior of these kinds: inputs
accepted and rejected, outputs and their shape, errors and their codes or
bodies, observable side effects, ordering, consistency windows,
idempotency, limits, and invariants. For each candidate record:

- the boundary and the behavior, as input → output, error, or effect;
- the evidence and its kind;
- the known consumers, with evidence, or UNKNOWN when none was found and
  none can be ruled out;
- the tests that pin it, if any;
- a normative status: CONFIRMED_CONTRACT only with an authority,
  DE_FACTO_COMPATIBILITY when known consumers depend on it, and otherwise
  PENDING_DECISION or UNKNOWN.

Look in route and schema definitions, serializers, event schemas, error
mappings, published package exports, retry and deduplication code, the
order of writes and publishes, integration and contract tests, consumer
lists, and documented guarantees.

## Depth

| Depth | Covers |
|---|---|
| ORIENT | rarely needed; the boundaries themselves, without behavior detail |
| ESTABLISH | every boundary behavior in scope, including error semantics |
| EXHAUSTIVE | plus edge cases: null versus missing, empty collections, limits, encodings, pagination and ordering, timing |

## Gotchas

- Accidental behavior is still a candidate: an unhandled error that returns
  HTTP 500 is observable, and a client may have learned to depend on it.
  Record it as PENDING_DECISION, not as a contract and not as a bug to fix.
- Side effects consumers rely on are easy to miss because nothing returns
  them: an email sent, a record left behind, an event emitted twice.
- Consumers outside the repository are invisible from inside it. Say so
  instead of concluding there are none.
