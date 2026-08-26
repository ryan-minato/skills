# Linters, Formatters, Type Checkers

Read when choosing the linter, formatter, or type checker, when the user
names an alternative to the Ruff/ty defaults, or when maturity
requirements rule out a young tool.

## Linters

| Tool | One line |
|---|---|
| Ruff (default) | Fast linter reimplementing the Flake8/isort/pyupgrade rule families under one config |
| Pylint | Deepest analysis and opinions, at a real speed cost |
| Flake8 | The classic pluggable linter |
| Pyflakes | Error-only checking, no style opinions |

## Type Checkers

| Tool | One line |
|---|---|
| ty (default) | Fast checker and language server from the Ruff/uv team |
| mypy | The reference implementation, most mature |
| Pyright | Fast, strict, powers VS Code's Pylance |
| Pyre | Meta's checker for very large codebases |

**ty maturity caveat:** ty is young. Before recording it, fetch its docs
and confirm it supports the project's Python version and the typing
features the codebase uses; when the user needs a settled checker, record
mypy or Pyright instead.

## Formatters

| Tool | One line |
|---|---|
| Ruff formatter (default) | Black-compatible formatter in the same binary as the linter |
| Black | The uncompromising formatter Ruff's is modeled on |
| isort | Import sorting only |
| autopep8 | Minimal fixer that only corrects PEP 8 violations |
| docformatter | Formats docstrings to PEP 257 |

Fetch current install commands and rule configuration from the selected
tool's official first-party documentation before writing them into the target.

## Selection Rules

- Exactly one linter and one formatter; overlapping tools disagree at the
  margins and contributors lose.
- Import sorting goes through the linter's rules when it offers them
  (Ruff does); a separate isort next to Ruff is redundant.
- Pylint's depth is a deliberate speed trade for projects that want its
  extra analysis — a valid preference, not a default.
- Type-checking strictness is a recorded dial, not a virtue: start
  permissive on legacy code and strict on new code, and record where the
  dial sits and why. Never max it by default on an untyped codebase.
