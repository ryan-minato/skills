# Commit Conventions

<!-- Rework against analyze_history.py evidence and the agreed decision;
this document, scripts/check_commits.py CONFIG, CI, and CONTRIBUTING must
state one identical rule. -->

- Format: {{CONVENTION — e.g. Conventional Commits 1.0.0, or a plain
  subject rule}}, written in {{LANGUAGE}}.
- Validated surface: {{PR_TITLE_OR_RANGE — PR titles under squash merge;
  the commit range otherwise}}.
- Types: {{TYPES_TABLE_IF_STRUCTURED}}
- Scopes: {{SCOPE_RULE_IF_ANY}}
- Subject: imperative mood, no trailing period, ≤ {{MAX_LEN}} characters.
- Exemptions: merge and revert subjects{{OTHER_EXEMPTIONS}}.
- Enforcement: `python3 scripts/check_commits.py` locally and the
  `commits / messages` job in CI — the same script, same CONFIG.
