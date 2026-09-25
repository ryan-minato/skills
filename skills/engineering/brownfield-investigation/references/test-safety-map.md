# Test Safety Map

Read when asked which behavior tests protect, or before a change or
migration relies on the existing tests.

## Procedure

1. **Can the suite run?** Find how the tests run and, when permitted, run
   them. Record which tests fail, which are skipped or disabled, and which
   need services that are not available.
2. **Behaviors to map.** Take the important behaviors from the runtime-flow
   and contract-candidate findings, or from the question.
3. **Map each behavior to its tests.** A test protects a behavior when it
   would fail if the behavior changed. Read the assertions: a test that
   exercises the path but asserts only a status code does not protect the
   body.
4. **Grade each behavior:**
   - protected — at least one running test asserts it precisely;
   - weakly protected — tests exercise it but assert loosely, mock the part
     that matters, or are flaky;
   - unprotected — no running test would notice a change.
5. **Rank the gaps** by risk: consumer-facing behavior first, then money or
   data-loss paths, then code that changes often.

At EXHAUSTIVE depth, and with the user's permission, confirm a protection
claim by breaking the behavior in a disposable copy of the repository and
watching a test fail. Never break anything in the working tree.

## Depth

| Depth | Covers |
|---|---|
| ORIENT | whether the suite runs, and protection of the main paths |
| ESTABLISH | protection of every contract candidate in scope |
| EXHAUSTIVE | protection of every boundary behavior in scope, failure paths included |

## Output

A table of behavior → tests (with locations) → level (unit, integration,
end to end, contract) → grade → runs or not, followed by the ranked list of
unprotected and weakly protected behaviors.

## Gotchas

- Coverage percentages measure execution, not assertion; a line can run
  under a test that checks nothing about it.
- A test that mocks the component it claims to test protects the mock.
- Snapshot tests that are routinely regenerated protect nothing.
- A skipped test reads like protection in the file listing.
