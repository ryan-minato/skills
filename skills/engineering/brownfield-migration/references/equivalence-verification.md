# Equivalence Verification

Read when planning how the new system is compared with the old —
conformance runs, differential or shadow comparison, state or event
comparison — or when a comparison fails.

## Methods

- **Conformance run.** The characterization suite runs against the new
  system. Cheap, repeatable, and limited to the inputs the suite holds.
- **Differential comparison.** The same inputs — recorded requests,
  fixtures, generated cases — go to both systems. Outputs are normalized
  only as the envelope's rules allow, then compared.
- **Shadow traffic.** Real requests are mirrored to the new system and its
  responses compared and discarded. Its side effects must be suppressed or
  sandboxed: a shadow system that charges a card or sends an email twice
  is an incident, not a test.
- **State comparison.** After the same operations, the persisted records
  of both systems are compared under a stated field mapping.
- **Event and outbound-call comparison.** The events emitted and the calls
  made to other systems are captured and compared: payloads under the
  envelope, ordering per entity where the envelope requires it.

Choose the methods the envelope needs: strictly identical rows need
exact comparison, and a boundary with real traffic and side effects needs
more than the conformance run.

## Tolerances

State each tolerance with its reason and the decision that approved it:
numeric precision, timestamp granularity, ordering where the envelope
allows any order, fields allowed to differ. A tolerance not in the approved
envelope does not exist.

## When a comparison fails

Report the difference with both outputs and the input as a failure, then
triage:

Take the first row that matches:

| Cause | Action |
|---|---|
| The difference is in ordering, timestamps, or generated identifiers | ask the user whether it may differ, proposing a semantic rule for the envelope as a decision item |
| Any other difference on a strictly identical row whose old output is deterministic | the new system is wrong: fix it |
| The old behavior looks like a bug | a decision item: INTENTIONALLY_CHANGE or preserve |
| The cause is unclear | a decision item describing the difference; acceptance stays blocked until it is ruled on |

Deciding on your own which side is wrong is only safe in the second row.

Never widen a tolerance, loosen a normalization, or reclassify a row to make
a comparison pass without a recorded ruling.

## Exit criteria

Agree with the user what ends each phase — for example, a conformance run
with every normative and compatibility test passing, then a shadow period
of an agreed volume or duration with zero unexplained differences — and
record them with the envelope.
