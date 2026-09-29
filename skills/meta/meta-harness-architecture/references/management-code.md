# Management Code

Management code is everything in the harness that never ships in the
product: quality checks, environment preparation, CI and administration
scripts, git hooks, inline workflow shell, task-runner recipes, and scripts
inside project skills. The project owns it and fixes it with its own tools,
so write it for whoever reads it next.

## Written for its role

- Write each script for the job that calls it (a CI check, a hook, a
  project-skill helper) and put it in the project's own paths.
- When a skill used in the build already ships a script that does the job,
  do not bind the project to it. No rule, check, or instruction may require
  the project's script to stay identical to a skill's bundled script, and no
  step runs a script from a skill's install path. Write or deposit a script
  for the role. Its content may match the skill's script where the role
  needs the same code; matching content is not the defect, a binding is.
- Say in the handoff that the project owns the script and that later fixes
  to the skill do not reach it.

## Language

Choose by how widely and maturely the project's community writes scripts
in its language, not by whether the language has a build step:

- Use the project's language when its community widely and maturely
  scripts in it, however short the check. A Go project gets Go checks run
  with `go run`. A Node project gets TypeScript or JavaScript run by its
  package manager.
- Otherwise use Python or Deno. A C or C++ project whose build has no
  scripting practice of its own gets Python or Deno checks, not C++
  programs and not long Bash scripts.
- Bash is for a few lines of glue that certainly run only in a controlled
  Linux environment, such as the body of one CI step or a dev container
  hook. A check that developers also run, or that grows past a few lines,
  is not a Bash case.
- For a borderline ecosystem, check the language's current official
  documentation for its scripting practice (a run command, a script mode, a
  tools layout) before choosing.

## Dependencies and invocation

- Where the ecosystem has a convention for tool dependencies, follow it: a
  Node project adds the package to `devDependencies` in `package.json`,
  and a Go project adds the module to `go.mod`.
- Where it has none, keep the script self-contained. A Python script
  declares its packages in PEP 723 inline metadata. The conventions of a
  Python project's own scripts belong to the `meta-python-defaults`
  builder.
- Use the standard library only when it solves the task cleanly. Parsing
  a format such as YAML or TOML, validating a schema, or HTTP with retries
  needs its mature library, even when today's files look like a small
  subset: a hand-written parser would need its own tests before anyone
  could trust it, and the next file breaks it. In Python, declare the
  library in PEP 723 and recommend adding uv to the harness where it is
  missing; in Deno, import the standard library's module (`jsr:@std/yaml`
  for YAML).
- Invoke a script with `uv run` only when uv exists in every environment
  the script runs in: developer machines, the dev container, and each CI
  job that calls it. Otherwise call the interpreter, or add uv to the
  environments that lack it first. Never write `uv run` for an environment
  without uv.
- Record the language and the exact command that runs each management
  script in the target's harness knowledge (the entrypoint's validation
  table or a knowledge file), so the next agent runs it the same way.

## Failing fast

Fail-fast is a design philosophy aimed at one thing: a fallback that
postpones or hides an unexpected error. It is not a rule against handling
or deferring a failure by design. Readability comes first, and an
unexpected failure is left to crash rather than turned into a pass or a
wrong answer. Nothing is built on a result the code knows is bad.

Check each interface the code relies on where its data enters, and fail
there with a message naming the interface, the value received, and what
to fix:

- an external command: its exit status, and for structured output, that it
  parses and has the shape the code reads;
- an API: the status, the JSON shape, and documented flags such as
  truncation or a next page;
- a file another step wrote: that it exists, parses, and carries the keys
  the code reads.

After the check, read the data directly instead of guarding it again at
every use.

Remove the fallbacks that hide a failure:

- a default standing in for data the interface guarantees: `?? {}`,
  `|| []`, `.get(key) or {}`, a 404 read as "empty";
- a swallowed exception, `except: pass`, or a catch-all handler at the
  entry point;
- a handler that only restates the exception it caught;
- a guard on a field the platform guarantees.

Keep what is not a fallback:

- A value the data contract documents as possibly null is domain logic. A
  pull request body is null when it is empty, so treat null as empty text.
- Structural safety (dry-run by default, refusing a dirty tree),
  idempotence, and narrow retries on known-transient errors such as rate
  limits and server errors stay.

A finding is not a crash. A check that validates committed files prints
each finding with the file and the fix, then exits non-zero without a
traceback; a traceback is for an unexpected failure of the check itself.
When the user wants every problem reported in one run, collect the
findings and fail once at the end, and do not restructure the check to
stop at the first finding. When a failure surfaces is a judgment made for
readability and for the person who fixes it.

A failure the design expects may be handled or deferred: an exit status
that carries meaning, a finding to collect, a result a later step or gate
decides on. Write such a deferral explicitly (see Shell), so a reader sees
that it is on purpose.

Before, the failure hides:

```python
out = subprocess.run(["gh", "pr", "view", "--json", "labels"], capture_output=True, text=True).stdout
labels = (json.loads(out or "{}").get("labels") or [])
```

After, the interface is checked once and the data is read directly:

```python
result = subprocess.run(["gh", "pr", "view", "--json", "labels"], capture_output=True, text=True)
if result.returncode != 0:
    sys.exit(f"`gh pr view` exited {result.returncode}: {result.stderr.strip()}")
try:
    labels = json.loads(result.stdout)["labels"]
except (ValueError, KeyError) as exc:
    sys.exit(f"`gh pr view --json labels` printed unexpected output ({exc}): {result.stdout[:200]!r}")
names = [label["name"] for label in labels]
```

## Shell

A shell step hides a failure in a few recurring ways. Write each step so
that a failed command either fails it or is handled on purpose:

- A command tested inside a condition: `if cmd | grep -q x` reads a failed
  `cmd` as "no match". Run the command outside the condition, save its
  output, then test the output.
- An `|| true` wider than the exit it was written for: `grep` exits 1 for
  no match and 2 for an error. Tolerate the exit you mean and let the rest
  fail.
- A pipe whose earlier status is dropped: without `pipefail` only the last
  command's status counts, and a process substitution `< <(cmd)` never
  reports `cmd`'s status. Write the output to a file first, or turn on
  `pipefail` for that step.
- A sourced script or an inherited option that changes what fails.

Choose the shell and its options for what each step needs, knowing the
runner's defaults (verified 2026-09-29): GitHub Actions runs a `run:` step
with no `shell` as `bash -e {0}` and one with `shell: bash` as
`bash --noprofile --norc -eo pipefail {0}`; GitLab Runner's generated bash
script sets `errexit`, and `pipefail` where the shell supports it.

Write a deliberate deferral where it happens, so it reads as one and the
failure still reaches where the design sends it:

- An advisory check that reports on every run but must not block a merge:
  mark it on the step or job (`continue-on-error: true` in GitHub Actions,
  `allow_failure: true` in GitLab CI) with a comment saying why, and keep
  its report visible in the log, a summary, or an annotation.
- A report step that must run after a failure: `if: always()` or
  `if: failure()`, while the failed step still fails the job.
- A gate over collected results: an aggregator job that runs whatever its
  dependencies did and fails when any of them failed or was cancelled.
- Output posted before failing: record the status (`cmd || rc=$?`), post
  the output, then exit with the recorded status.
