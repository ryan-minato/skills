# Runtime Flow

Read when tracing what one capability does from its entry point to its
outputs, state changes, and side effects.

## Procedure

Trace one capability as a chain, recording the path and line of each hop:

1. **Entry.** The route, handler, consumer, job, or command where the
   request arrives, and the middleware, interceptors, or decorators that
   run before it.
2. **Input.** Parsing, validation, defaults, and authorization — what is
   rejected, and how.
3. **Hops.** Each call that carries the work forward, including hops
   through queues or events (follow the topic or queue name when the call
   chain breaks).
4. **State changes.** Every write, to which store, inside which
   transaction.
5. **Side effects.** Events published, external calls, messages sent,
   files written — with their payloads when the depth needs them.
6. **Outputs.** The response or result, its shape and status.
7. **Failure paths.** Which errors arise where, which are caught, retried,
   or swallowed, and what the caller finally sees.
8. **Ordering and guarantees.** What happens before and after the commit,
   retries and their counts at every layer, idempotency, and concurrency
   controls.

## Depth

| | ORIENT | ESTABLISH | EXHAUSTIVE |
|---|---|---|---|
| Representative success path | required | required | required |
| Other important paths | not required | the important ones | high coverage |
| Failure paths | a few key ones | those touching a contract | systematic |
| Side effects | overview | at contract level | exact, with payloads |
| Ordering | when it matters to the question | where a contract depends on it | high priority |
| Runtime evidence | optional | recommended | strongly recommended |
| Edge cases | a few | at external contracts | broad |

## Runtime evidence

When the user or the scope record permits, run the path — through its
tests, a local instance, or recorded logs — with a representative input,
and record what was observed and how. An observation outranks a reading of
the code for what actually happens; it does not decide what should happen.

## Gotchas

- Retries often exist at more than one layer (a client library and the
  code around it); the effective count is their product.
- Transaction boundaries hidden in decorators or framework configuration
  decide whether a side effect can happen without the write.
- Errors caught and logged look like success to the caller; record what
  the caller receives, not what the log says.
- Framework conventions route requests without explicit wiring; find the
  convention before concluding a handler is unused.
