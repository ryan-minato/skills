# Characterization Tests

Read when writing, running, or tagging characterization tests. A
characterization test records what the existing system does. Its passing
says "unchanged", never "correct".

## Choose the boundaries

Test where consumers observe the system: its interfaces, its error
responses, the state it persists, the events it emits, the calls it makes
to other systems. Prefer driving the running system from outside, so the
suite needs no change to the legacy code and can later run unchanged
against the new system.

When the code must be entered from inside, limit changes to mechanical
seams (extracting a parameter, injecting a clock) that alter no behavior,
and record each one.

## Capture, do not predict

Run the existing system with the input and record the output it actually
produces as the expected value. Never write an expected value from reading
the code: that tests your reading, not the system.

Cover, per boundary in scope, the success paths and the failure paths the
baseline lists, including the inputs the old system handles oddly.
Reproduce legacy data shapes in the fixtures, odd ones included.

## Tags

Every test carries exactly one tag, matching its baseline row:

- **normative** — pins a CONFIRMED_CONTRACT;
- **compatibility** — pins DE_FACTO_COMPATIBILITY or a PRESERVE_TEMPORARILY
  ruling;
- **pending-decision** — pins behavior not yet ruled on, including
  behavior that looks like a bug.

Put the tag and the baseline row's id in the test's name or metadata, so a
failing test leads straight to its decision. Behavior that looks like a bug
keeps its current expected value and its pending-decision tag until a
ruling says otherwise.

## Nondeterminism

- Freeze or inject the clock; seed randomness; make generated identifiers
  predictable where the boundary allows.
- Keep ordering assertions unless the envelope says ordering does not
  matter; sort only under that rule.
- Where a value cannot be controlled, scrub it — replace the field with a
  placeholder — and keep asserting everything else. Dropping the whole
  assertion to avoid one timestamp throws away the test.
- For large outputs, compare against a recorded master copy with the same
  scrubbing.
- For calls to other systems, record and replay at the boundary, or use a
  fake whose behavior was itself captured from the real system. Note that
  a fake pins the fake.

## Stability

Run the suite twice in a row against the unchanged existing system — in
random order too, when the test framework supports it. Any test that
changes result is fixed or removed before the suite is relied on, and the
removal is recorded.
