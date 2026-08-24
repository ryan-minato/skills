---
name: <check-x>
description: >-
  Runs this project's <what is being validated> checks and reports
  pass/fail with fixes. Use when the user asks to <validate / audit /
  review X>, or before <the event that requires it>. Not for <the
  deeper review this does not replace>.
---

# <Validation Name>

<One sentence: what passing means, and what this gate protects.>

## Checks

| # | Check | How | Passes when |
|---|---|---|---|
| 1 | <what> | `<command or inspection>` | <criterion> |
| 2 | <what> | <how> | <criterion> |

Run non-interactive commands only; never prompt mid-run.

## On Failure

- Check <n> fails → <the fix, or who decides>.
- Never silence a check to make the run green; fix the cause or report
  it.

Done when: every check reports pass, or each failure is reported with its
cause and proposed fix.

Update this skill in the same change that adds, removes, or reworks any
check above.
