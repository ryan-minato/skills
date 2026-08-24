# Docstring Completeness Defaults

Read when the user did not specify how thorough docstrings must be and the
harness needs a completeness default to record.

## Two Tiers

Pick the tier by one question: do consumers outside this repository read
the API?

- **Application tier (default)** — the code's readers are its editors.
  Docstrings carry what the name and signature cannot.
- **Library tier** — the public API is a published contract. Everything
  public is documented as if the reader cannot open the source.

## Per-Object-Kind Table

| Object | Application tier | Library tier |
|---|---|---|
| Module | One-line purpose statement | Purpose plus a short usage overview for public modules |
| Package `__init__` | One-line package purpose | Package purpose and what it exports |
| Public class | Summary line; document attributes only when state is non-trivial | Summary, attributes, and usage notes |
| Public function / method | Summary line always; full parameter/return/exception sections only when the signature is not self-evident | Full sections always |
| Private helper (`_name`) | One-liner, or none when the name and type hints suffice | Same as application tier |
| Dunder method | None; a non-trivial `__init__` is documented on the class | Same as application tier |
| Property | One-line noun phrase ("The active session's user.") | Same, plus raising behavior if any |
| Test function | None — the test name carries the intent | Same as application tier |
| Override | Document only deviations from the base class contract | Same as application tier |

Two universal rules on top of the table:

- A docstring that restates the signature ("Args: x: the x") is worse than
  none; if there is nothing to add beyond names and types, stay with the
  summary line.
- Exceptions worth documenting are the ones callers are expected to
  handle, not every error that could theoretically propagate.

## Enforcement

Enforcement is optional and separate from the requirement: most linters
expose a docstring-convention setting that checks presence and formatting.
Record the requirement even when nothing enforces it, and fetch the
enforcement syntax from the chosen linter's docs when the project wants
it checked.
