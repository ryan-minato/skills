# Test-Style Doctrine: Vocabulary and Trade-offs

Read when a default in the recorded doctrine is being challenged or
tailored — the user wants BDD phrasing, heavy mocking already exists, or
the project must choose between example-based and property-based rigor.

## Structure: AAA vs Given-When-Then

Arrange-Act-Assert and Given-When-Then are the same skeleton for
different audiences. AAA (the default) speaks to developers reading test
code. GWT phrases the same three beats as behavior specification —
choose it only when stakeholder-readable specs are an actual project
requirement, because it usually brings a BDD toolchain with it. Either
way: one behavior per test, and the act step is one line.

## Schools: Classical vs Mockist

- **Classical (Detroit, the default):** a unit's collaborators are real
  unless they cross a boundary the project does not own. Tests assert on
  outcomes, so they survive refactoring — the implementation can be
  rewritten and green still means correct.
- **Mockist (London):** every collaborator is replaced; tests specify the
  conversation between objects. This pins the design in place: renaming a
  method call breaks tests even when behavior is unchanged.

The mockist stance is legitimate in one narrow case: the interaction *is*
the contract ("charges the gateway exactly once", "emits the audit event
before committing"). Then verify the interaction — that is the behavior.

## The Double Taxonomy

| Double | Is |
|---|---|
| Dummy | Fills a parameter list; never used. |
| Stub | Returns canned answers; no verification. |
| Spy | A stub that also records what happened to it, for later assertions. |
| Mock | Pre-programmed with expected calls; the test verifies the interaction. |
| Fake | A working lightweight implementation (in-memory repository, temp filesystem). |

The preference ladder — **real object > fake > stub > mock** — orders
doubles by how loosely they couple the test to the implementation. Each
step down encodes more of "how" into the test: a real object knows
nothing of the test, a fake shares only the interface, a stub hard-codes
specific answers, a mock hard-codes the exact conversation. Climb down
only when the step above is impractical.

## State vs Behavior Verification

Default to state: assert on what the system ended up as (return values,
stored records, emitted output). Behavior verification (asserting calls
happened) is reserved for owned contracts with unowned systems — the
mockist exception above. A state assertion survives an internal rewrite;
a behavior assertion is a bet that the wiring never changes.

## Parametrization

Collapse families of cases into one parametrized test instead of
copy-pasted bodies. Keep case IDs readable — a failure report naming
`invalid-utf8` beats `case_7`. If cases need different assertions, they
are different tests, not parameters.

## Property-Based vs Example-Based

Example-based tests state "for this input, this output" — cheap to write
and to read, the default. Property-based tests state invariants ("decode
inverts encode", "output is always sorted") and let the framework hunt
counterexamples — stronger, but each property costs design thought.
Use properties where the code is invariant-rich (algorithms, parsers,
serialization) or the project demands rigor; they compose with examples,
not compete.

## Error Paths

Expected exceptions and error paths are part of a unit's definition of
done: every documented raise gets a test, and the assertion covers the
exception type plus the message when the message is part of the contract.
A suite that only proves the happy path proves half the function.
