# Contributing

This repository accepts occasional external fixes and improvements. Open an
Issue when the work needs its own discussion or priority; a small,
self-contained pull request may put `N/A — <reason>` in the template's
`Related issue` section instead.

## Prepare the change

Create a dedicated branch from the latest `main`. Keep commits atomic and use
the Conventional Commit format documented in [AGENTS.md](../AGENTS.md) for
every non-merge/revert commit.

The dev container supplies the development tools. Outside it, install `just`,
`pre-commit`, and `ruff` before running the commands below.

Run the repository checks before opening a pull request:

```bash
just setup
just check
```

If a public Skill changes, also follow the Skill authoring and behavioral-test
rules linked from [AGENTS.md](../AGENTS.md).

## Open the pull request

Use the pull request template and leave the PR as a draft until the requested
work is complete, the tests pass, and every checklist item is complete. Link
the Issue with `Closes #N`, or explain why no Issue exists with
`N/A — <reason>`.

A maintainer makes the final integration decision. Maintainers rebase-merge
branches in `ryan-minato/skills` and squash-merge pull requests from forks;
contributors should not rewrite a fork's history solely to satisfy the
repository's commit-subject convention.

Do not put credentials, private data, or unpublished security reports in an
Issue, pull request, commit, or comment. Use the private reporting path in
[SECURITY.md](SECURITY.md) for vulnerabilities.
