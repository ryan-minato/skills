# Python (uv) Script Specification

## Inline Dependency Declaration (PEP 723)

Place this block near the top of the file, before imports:

```python
# /// script
# dependencies = [
#   "httpx>=0.27,<1",
#   "beautifulsoup4>=4.12",
# ]
# requires-python = ">=3.11"
# ///
```

Version specifiers: prefer `>=X.Y,<X+1` for libraries; exact `==X.Y.Z` for tools.
Avoid unpinned dependencies.

## Shebang

```python
#!/usr/bin/env -S uv run
```

## Running

```bash
uv run scripts/extract.py
```

`uv run` creates an isolated virtual environment, installs declared dependencies, and
runs the script. Subsequent runs reuse the cached environment.

## Design Rules

**No interactive prompts** — agents run in non-interactive shells; `input()` hangs
indefinitely. Accept all input via CLI flags or stdin.

**`--help` required** — the primary interface discovery mechanism. `argparse` generates
it automatically from `add_argument` calls. Include a one-line description, all flags
with types and defaults, and at least one usage example (use `epilog`).

**Structured output** — data to stdout (JSON preferred for machine consumption);
diagnostics and progress to stderr.

**Exit codes:**

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments (`parser.error()` uses 2 automatically) |

**Idempotent** — agents may retry on transient failures; design operations to be safely
re-runnable.

**Large output** — if output may exceed ~10,000 characters, default to a summary and
provide `--full` or `--output FILE` flags; agent harnesses may silently truncate stdout.

## Helpful Error Messages

Write error messages the agent can act on:

```python
# Bad
raise ValueError("invalid input")

# Good
parser.error(
    f"--format must be one of: json, csv, table. Got: {args.format!r}"
)
```
