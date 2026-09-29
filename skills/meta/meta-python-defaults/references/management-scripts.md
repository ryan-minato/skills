# Python Management Scripts

Read when the project has, or the harness build adds, Python scripts that
never ship in the product: checks, hooks, CI or administration scripts,
and scripts inside project skills. These conventions sit beside the
project's own dependency management; they do not replace it.

## Dependencies and invocation

- A script with third-party dependencies declares them in PEP 723 inline
  metadata, so the script carries its own environment:

  ```python
  # /// script
  # requires-python = ">=3.10"
  # dependencies = ["pyyaml>=6"]
  # ///
  ```

- Run such a script with `uv run <script>` only when uv exists in every
  environment it runs in: developer machines, the dev container, and each
  CI job that calls it. `uv run` gives the script its declared
  dependencies and ignores the project's.
- When uv is not in every one of those environments, and the standard
  library solves the task cleanly (reading JSON, running git, walking
  files), keep the script standard-library-only, give it no PEP 723
  header, and run it with the interpreter (`python3 <script>`).
- When the task needs a mature library (YAML, schema validation, HTTP
  with retries), do not reimplement it on the standard library: recommend
  adding uv to the environments that lack it, and declare the library in
  PEP 723.
- Record the chosen invocation beside the script's entry in the harness
  (the validation table or the knowledge file that lists commands).

## Error handling

Present these as the fail-fast philosophy, not as a procedure: a script is
readable first, an unexpected failure is left to crash, and no fallback
postpones or hides it.

- Check each interface the script relies on where its data enters, and
  exit with a message naming the interface and the value received:
  - a subprocess: its exit status, and for structured output, that it
    parses and has the shape read;
  - an HTTP response: the status and the JSON shape;
  - a file read: that it parses and carries the keys read.
  `sys.exit(f"...")` prints the message to stderr and exits 1.
- After the check, read the data directly (`data["labels"]`, not
  `data.get("labels") or []`).
- A bare or blind `except`, a `try`/`except`/`pass`, and a handler that
  only restates the exception each need a stated reason in a comment; they
  are never a default habit.
- A finding in a checked file prints the file and the fix and exits
  non-zero without a traceback. A traceback is for an unexpected failure.
- A value the data contract documents as possibly null is domain logic
  (`body or ""` for a pull request body). Retries stay narrow: known
  transient errors only, a bounded count.

Before, a failure reads as "no labels":

```python
out = subprocess.run(["gh", "pr", "view", "--json", "labels"], capture_output=True, text=True).stdout
labels = json.loads(out or "{}").get("labels") or []
```

After, the interface is checked once:

```python
result = subprocess.run(["gh", "pr", "view", "--json", "labels"], capture_output=True, text=True)
if result.returncode != 0:
    sys.exit(f"`gh pr view` exited {result.returncode}: {result.stderr.strip()}")
try:
    labels = json.loads(result.stdout)["labels"]
except (ValueError, KeyError) as exc:
    sys.exit(f"`gh pr view --json labels` printed unexpected output ({exc}): {result.stdout[:200]!r}")
```

## Locking dependencies in CI

When a PEP 723 script with third-party dependencies runs in a CI job, its
dependencies resolve on every run unless they are locked. Present locking
as a recommendation and let the user decide; write no lock the user did
not choose. The two options (checked against uv's documentation on
2026-09-29):

- `uv lock --script <script>` writes `<script>.lock` beside the script,
  and later `uv run --script` runs reuse it.
- An `exclude-newer` cutoff in the script's `[tool.uv]` table limits
  resolution to distributions released before an RFC 3339 timestamp:

  ```python
  # [tool.uv]
  # exclude-newer = "2026-09-01T00:00:00Z"
  ```

Default the recommendation by the job's actual risk:

| The job | Default |
|---|---|
| Privileged trigger (`pull_request_target`, `workflow_run`, a comment trigger), a token that can write, or secrets | Recommend locking; name both options |
| Read-only token, no secrets, unprivileged trigger | Say locking is optional for this job |

Ask, then write the option the user chose, or none.
