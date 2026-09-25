# Guardrails

Read when turning an approved contract or policy into a test, a check, or
an agent instruction. A guardrail is only as good as the proof that it
fails when the rule is broken.

## The enforcement ladder

Take the first rung that can hold the rule:

1. **Contract test at the boundary** — for behavioral contracts. It drives
   the interface a consumer uses and asserts the guarantee: the response,
   the error, the persisted state, the emitted event.
2. **Static check with a failing exit code** — for structural policies:
   dependency direction, forbidden imports, required files, schema rules.
   Use the project's existing linters or architecture-test tooling before
   adding a new tool.
3. **Ratchet** — when existing code already violates an approved policy,
   check new code strictly and list the existing violations explicitly, so
   the list can only shrink. The list is itself a recorded decision.
4. **Report-only check** — only as a dated, temporary step toward rung 2 or
   3, with the date recorded.
5. **Prose instruction** — only for a rule no check can express. Put it
   where the agent meets the work (see below).

A rule written as prose when a check could hold it will be broken by the
first agent that did not read that paragraph.

## Proving a guardrail

For every guardrail:

1. Run it against the current system; it passes.
2. Introduce a minimal violation in the working copy — remove the
   duplicate check, add the forbidden import.
3. Run it; it fails, and its message names the rule.
4. Remove the violation; run it again; it passes. Confirm the working copy
   is back to its original state.
5. Record the violation used and the failure seen next to the decision id.

A guardrail that never failed has not been shown to guard anything.

## Wiring

- Run guardrails where the project already runs its checks, so a violation
  blocks the change instead of being noticed later.
- Name each guardrail after the rule it enforces, and cite the decision id
  in it, so a failing run leads back to why the rule exists.

## Agent instructions

- Put an instruction line in the agent entrypoint only when every task
  needs it; otherwise put it beside the code it governs or in the project's
  knowledge files, with a pointer whose trigger names the event that makes
  it relevant.
- Prefer a searchable marker in the code itself (a comment marking legacy,
  generated, or frozen code) over a paragraph elsewhere: agents find the
  marker while working on the file.
- Write only approved rules, with the decision id. An observed convention
  written as an instruction becomes a rule nobody chose.
