---
name: meta-python-defaults
description: >-
  Disposable meta-skill (delete after the harness is built): chooses and
  records coherent defaults for a Python project's documentation style,
  testing approach, and development toolchain. Use during a harness build
  when a Python project lacks settled docstring, comment, test, dependency,
  lint, format, type-check, task, hook, or documentation conventions. Preserve
  every working project or user choice; this is not a migration skill and is
  not for writing the project's production code.
license: Apache-2.0
---

# Python Project Defaults

Supply missing decisions, not a universal Python stack. Inspect
`pyproject.toml`, lockfiles, test layout, task files, hooks, CI, documentation,
and existing code style before proposing anything. A working convention or an
explicit user preference always wins.

## Workflow

1. Inventory each decision as settled, missing, or inconsistent:
   documentation style and completeness; comment philosophy and markers;
   testing framework and doctrine; dependency management; lint/format/type
   tools; task runner and hooks; documentation generator.
2. Resolve only missing or explicitly inconsistent decisions. Present one
   recommended default and explain deviations from it; do not offer an
   unranked menu.
3. Documentation:
   - Read [references/docstring-styles.md](references/docstring-styles.md)
     when selecting Google, NumPy, or reStructuredText style.
   - Read
     [references/completeness-defaults.md](references/completeness-defaults.md)
     when deciding which public and private objects require docstrings.
   - Read [references/comment-markers.md](references/comment-markers.md)
     when standardizing TODO, FIXME, warning, or tooling markers.
   - Default to self-documenting code, docstrings for public contracts and
     non-obvious behavior, and comments for why rather than what.
4. Testing:
   - Read [references/frameworks.md](references/frameworks.md) when framework
     choice is unresolved.
   - Use pytest by default; retain unittest or doctest when the project has a
     working reason.
   - Read [references/pytest-plugins.md](references/pytest-plugins.md) only
     when a named need exists; never install plugins speculatively.
   - Read [references/test-doctrine.md](references/test-doctrine.md) before
     setting test style. Default to contract-focused tests and real objects,
     using mocks only at expensive, nondeterministic, or external boundaries.
5. Toolchain:
   - Read
     [references/dependency-managers.md](references/dependency-managers.md)
     only when dependency management is unsettled.
   - Read [references/quality-tools.md](references/quality-tools.md) when
     lint, format, or type-check choices are missing or conflict.
   - Read
     [references/task-runners-and-hooks.md](references/task-runners-and-hooks.md)
     when commands or local gates need a stable entrypoint.
   - Read [references/doc-generators.md](references/doc-generators.md) only
     when the project actually publishes generated documentation.
6. Verify volatile commands, versions, configuration syntax, and current
   capabilities against official first-party documentation at execution
   time. Do not create a documentation-URL registry or copy one into the
   harness.
7. Record the selected conventions where the target harness keeps durable
   implementation and quality constraints. Include the evidence for retained
   existing choices and the trigger for revisiting each new default.

Done when: every in-scope decision is either inherited from the project or
explicitly chosen, no working choice was replaced without a request, and
future agents can reach the recorded conventions from the entrypoint.

## Default baseline

Use this only for genuinely empty choices:

- Google-style docstrings for a general application; NumPy style for
  scientific APIs already shaped around that ecosystem.
- Pytest with no plugin until a concrete test need requires one.
- A classical testing style: behavior and contracts over implementation
  details, with fast tests in the default suite and expensive tests explicit.
- uv for dependency and environment management, Ruff for formatting and
  linting, ty for type checking when the project benefits from static checks,
  just for task entrypoints, and pre-commit for fast local gates.
- Documentation generation only when the project has an audience and a
  publication path.

These defaults yield to lifecycle, platform, team familiarity, dependency
constraints, and every existing working choice.

## Gotchas

- A default is not permission to migrate.
- Installing every pytest plugin or quality tool makes the feedback loop
  slower and the harness thicker without evidence.
- Marker comments without ownership or cleanup rules become permanent noise.
- Tool names and URLs go stale; record selected categories and project
  decisions, then verify current tool details when implementing them.
