# Differences Between Starlark and Python

## Top-Level and Execution Restrictions

* **Global Variables**: Global variables are immutable.
* **No Top-Level `for`**: `for` statements are not allowed at the top-level. Use them within functions instead. (In `BUILD` files, you may use list comprehensions).
* **No Top-Level `if`**: `if` statements are not allowed at the top-level. However, `if` expressions can be used: `first = data[0] if len(data) > 0 else None`.
* **No Recursion**: Recursion is not allowed.
* **BUILD vs .bzl**: In `BUILD` files, declaring functions is illegal, and `*args` and `**kwargs` arguments are not allowed.

## Data Types and Operations

* **Strings**: Strings aren't iterable. They are represented with double-quotes (such as when you call `repr`).
* **Dictionaries**: Deterministic order for iterating through Dictionaries. Dictionary literals cannot have duplicated keys (e.g., `{"a": 4, "b": 7, "a": 1}` is an error).
* **Integers**: Int type is limited to 32-bit signed integers. Overflows will throw an error.
* **Iteration Mutation**: Modifying a collection during iteration is an error.
* **Cross-Type Comparisons**: Except for equality tests, comparison operators `<`, `<=`, `>=`, `>`, etc. are not defined across value types. In short: `5 < 'foo'` will throw an error and `5 == "5"` will return false.

## Syntax Strictness

* **Tuples**: In tuples, a trailing comma is valid only when the tuple is between parentheses — when you write `(1,)` instead of `1,`. Starlark does not permit a trailing comma to appear in an unparenthesized tuple expression.
* **Unparenthesized Tuples in Loops**: Starlark does not accept an unparenthesized tuple or lambda expression as the operand of a `for` clause in a comprehension.

## Unsupported Python Features

The following standard Python features are **not supported** in Starlark:

* `while` and `yield` loops.
* `class` definitions (use `struct` function instead).
* `import` statements (use `load` statement instead).
* `is` identity operator (use `==` instead).
* `try`, `raise`, `except`, `finally` block for exceptions (see `fail` for fatal errors).
* `global` and `nonlocal` keywords.
* Chained comparisons (such as `1 < x < 5`).
* Implicit string concatenation (must use explicit `+` operator).
* Generators and generator expressions.
* Float and set types (depending on the strictness of the Bazel environment, though generic Starlark supports them).
* Most Python builtin functions and most methods.
