# Task Runners and Git Hooks

Read when deciding how the project's checks and dev tasks are invoked by
contributors and agents, and whether git hooks enforce them.

## Task Runners

| Tool | One line |
|---|---|
| just (default) | Command runner with make-like recipes and none of make's build semantics |
| Poe the Poet | Tasks defined inside `pyproject.toml` |
| Invoke | Tasks as Python functions in a `tasks.py` |
| Nox | Sessions in Python across multiple environments |
| tox | The classic environment-matrix runner, ini-configured |
| Make | Ubiquitous, but build-oriented: phony targets, tab-sensitive syntax, portability tax |

Fetch current install and syntax from the selected tool's official first-party
documentation before writing recipes into the target.

Two families, one distinction: plain command runners (just, Poe, Invoke,
Make) run named commands; environment-matrix runners (Nox, tox) create
isolated environments per Python version and only earn their weight for
libraries that test across versions — applications pinned to one version
do not need them. Poe fits teams that want everything inside
`pyproject.toml`.

Default recipe rule: one aggregate recipe runs the formatter, linter,
type checker, and test suite in a single command (e.g. `just verify`),
plus one recipe per individual tool — agents and CI call the aggregate,
humans iterate on the parts.

## Git Hooks

pre-commit is the default and the only major option. Fetch current install and
hook configuration from its official first-party documentation.

- Hooks run the same commands the task runner runs — one source of truth
  for what "passing" means; a hook that checks something the aggregate
  recipe does not (or vice versa) splits the truth in two.
- pre-commit pins each hook's tool version in `.pre-commit-config.yaml`,
  independent of the project's own dependency pins — record the rule that
  the two stay aligned, and how the project updates them.
- Hooks gate the commit; CI gates the merge. Record both, and keep them
  running the same aggregate.
