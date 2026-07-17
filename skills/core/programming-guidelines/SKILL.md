---
name: programming-guidelines
description: >
  Applies universal coding standards to all programming work. Use when writing,
  editing, debugging, reviewing, or refactoring code in any language or framework;
  when the user asks to keep it simple or not over-engineer; when they ask for minimal
  changes that touch only what was requested; when requirements are ambiguous enough
  to need explicit assumptions, tradeoffs, or success criteria before coding; or when
  produced code drifts beyond the request — unrequested features, extra abstractions,
  adjacent edits. Not for prose, documentation, or commit messages.
license: MIT
metadata:
  references: >
    https://x.com/karpathy/status/2015883857489522876
    https://github.com/forrestchang/andrej-karpathy-skills
---

# Programming Guidelines

Apply these standards by default for programming work, with proportional
judgment for small one-off tasks.

## 1. Think Before Coding

Do not assume, hide confusion, or skip tradeoffs.

Before implementing:

- State assumptions explicitly. If uncertainty affects the result, ask.
- If multiple interpretations exist, present them before choosing.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear enough to make implementation risky, stop, name the
  ambiguity, and ask.

## 2. Simplicity First

Write the minimum code that solves the problem.

- Do not add features beyond what was asked.
- Do not add abstractions for single-use code.
- Do not add flexibility or configurability that was not requested.
- Do not add error handling for impossible scenarios.
- If the implementation is much longer than the problem requires, simplify it.

Ask: would a senior engineer call this overcomplicated? If yes, simplify.

## 3. Surgical Changes

Touch only what the request requires.

When editing existing code:

- Do not improve adjacent code, comments, or formatting unless required.
- Do not refactor things that are not broken.
- Match existing style, even when another style would also work.
- If you notice unrelated dead code, mention it instead of deleting it.

When your changes create unused code:

- Remove imports, variables, functions, or files that your changes made unused.
- Do not remove pre-existing dead code unless asked.

Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

Turn tasks into verifiable goals and loop until verified.

- "Add validation" -> write checks for invalid inputs, then make them pass.
- "Fix the bug" -> reproduce the failure, then verify the fix.
- "Refactor X" -> ensure relevant checks pass before and after when practical.

For multi-step programming tasks, state a brief plan:

```text
1. [Step] -> verify: [check]
2. [Step] -> verify: [check]
3. [Step] -> verify: [check]
```

Strong success criteria allow independent progress. Weak criteria, such as
"make it work", require clarification.
