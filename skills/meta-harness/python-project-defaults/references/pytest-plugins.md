# pytest Plugins By Need

Read when selecting pytest plugins to meet a concrete project need —
coverage, mocking, async code, a slow suite, hangs, order coupling, or
property-based rigor.

## The Table

| Need | Plugin | Package |
|---|---|---|
| Coverage measurement | pytest-cov, wrapping coverage.py | `pytest-cov` |
| Mocking with cleanup | pytest-mock — a fixture wrapper over stdlib `unittest.mock` that undoes patches automatically | `pytest-mock` |
| Async test functions | pytest-asyncio; or anyio when tests must run against multiple async backends | `pytest-asyncio` / `anyio` |
| Parallel execution for a slow suite | pytest-xdist | `pytest-xdist` |
| Hanging tests | pytest-timeout; pytest's built-in `--durations=N` reports the slowest tests without any plugin | `pytest-timeout` |
| Hidden test-order coupling | pytest-randomly — randomizes order and seeds random number generators | `pytest-randomly` |
| Property-based rigor | Hypothesis — generates inputs against stated invariants | `hypothesis` |

Fetch current install and usage from the selected plugin's official
first-party documentation before writing commands or configuration.

## Selection Rules

- The default set for a no-preference project is **pytest + pytest-cov**;
  every further plugin enters only with a named need.
- Record each plugin in the harness with its need attached — an
  unattributed plugin cannot be safely removed later.
- Prefer the framework's built-in answer before adding a plugin (e.g.
  `--durations` before a timing plugin).
