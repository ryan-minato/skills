# Contract Classification

Read when classifying a candidate, or when a candidate fits two classes.

## The two tests, applied

**Rewrite test.** Imagine a team rebuilding the system from its formal
requirements alone, in another language, with another design. Would they be
free to change this? If yes, it is not a specification candidate.

**Consumer test.** List who would notice the change: clients of an
interface, subscribers to an event, other systems reading a shared table,
scripts parsing command output, people receiving a message. Evidence of a
consumer comes from the repository, from consumer lists, from traffic, or
from people. The absence of evidence is not the absence of a consumer.

## Examples

| Candidate | Class | Why |
|---|---|---|
| A repeated submission with the same idempotency key never creates a second order | SPECIFICATION | clients retry on timeouts and rely on it; any rebuild must keep it |
| An event is visible to consumers only after the order is persisted | SPECIFICATION | consumers read the order when the event arrives |
| Missing and null fields mean different things in a request | SPECIFICATION | clients send both and get different results |
| A legacy ID format still accepted for one known client until it migrates | COMPATIBILITY | needed now, with an end condition |
| A field carrying an identifier from a system being retired, inside an event other teams consume | COMPATIBILITY, on its own row; the event itself may still be a SPECIFICATION candidate | the field exists for the transition, not for the event's lasting purpose |
| Orders are stored in a relational database | ARCHITECTURE | a structural choice; no consumer sees the engine |
| The order service calls the pricing service | ARCHITECTURE or IMPLEMENTATION | internal collaboration, invisible at the boundary |
| A helper caches prices for sixty seconds | IMPLEMENTATION, unless consumers observe stale prices and rely on the window | the consumer test decides |
| An event field no repository reads, on a topic shared with other teams | UNKNOWN | consumers outside the repository cannot be ruled out |

## Candidates that fit two classes

- **A table another system reads or writes.** Its shape is a boundary, so
  it is SPECIFICATION or COMPATIBILITY, never merely ARCHITECTURE.
- **Error codes and bodies.** Contractual when clients branch on them; an
  unhandled error that returns a generic failure is a candidate with status
  PENDING_DECISION, not a contract by default.
- **Performance.** Latency or throughput is a contract only when a
  consumer depends on it and it can be measured at the boundary; otherwise
  it is a quality goal, recorded outside the specification.
- **Ordering and timing.** Contractual when a consumer's correctness
  depends on it (ordering of events for one entity); implementation when
  only speed depends on it.
- **Defaults.** A default a client relies on by omitting a field is
  observable, and therefore a candidate.

## From class to status

The class says where a behavior goes; the normative status says what a
human decided. SPECIFICATION becomes CONFIRMED_CONTRACT only by a ruling;
COMPATIBILITY becomes DE_FACTO_COMPATIBILITY with an end condition;
IMPLEMENTATION becomes IMPLEMENTATION_DETAIL; UNKNOWN stays PENDING_DECISION
or UNKNOWN until evidence or a ruling moves it. ARCHITECTURE carries no
normative status here; an approved structural rule becomes an architecture
policy.
