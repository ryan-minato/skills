# Contributing

> The knowledge base — [AGENTS.md](../AGENTS.md) and
> [.agents/knowledge/](../.agents/knowledge/) — governs agents and holds
> every rule in full. This page is the human summary; where the two
> disagree, the knowledge base wins.

Thank you for improving this skill library.

## Where things go

- Bugs, feature requests, and planned tasks: the
  [issue forms](https://github.com/ryan-minato/skills/issues/new/choose).
- Questions and ideas:
  [Discussions](https://github.com/ryan-minato/skills/discussions).
- Vulnerabilities: [SECURITY.md](SECURITY.md), never a public issue.

## Setup and checks

Open the repository in its dev container, or install the latest
[Deno](https://deno.com/), `just`, `pre-commit`, and `ruff` yourself; no
Node.js is needed. Then:

```bash
just setup
just check
```

`just setup` installs OpenSpec through Deno at the version the `justfile`
pins (the one pinned tool) and the latest OpenTofu, and installs the
pre-commit hooks. `just check` is exactly what CI runs (see
[.agents/knowledge/github-checks.md](../.agents/knowledge/github-checks.md)).

## Working on a change

1. Open an issue first for anything that needs discussion or priority; a
   small, self-contained fix may skip it and say `N/A — <reason>` in the
   pull request instead.
2. Branch from `main` as `<type>/<slug>`. A change to a skill's behavior
   or to the repository itself goes through an OpenSpec change under
   `openspec/changes/<slug>/`
   ([spec-workflow.md](../.agents/knowledge/spec-workflow.md)).
3. Write the approval package first — the proposal, the delta specs, and
   the design — and open a draft pull request with it, using the
   template. The maintainer discusses the package there and closes it by
   adding the `spec/approved` label; the task list and the implementation
   follow.
4. Commits from branches in this repository land by rebase merge, so each
   commit subject follows Conventional Commits (`type(scope): subject`).
   Pull requests from forks are squash-merged: only the pull request title
   must conform, and you need not rewrite your branch history.
5. Mark the pull request ready once every task is done, `just check`
   passes, and the Validation section records what you ran. Ready opens
   the second discussion, on the finished implementation; meanwhile the
   required `checks / archived` waits unreported and nothing is red.
6. When the maintainer closes that discussion, archive the change with
   `just spec-changes archive --base origin/main --head HEAD` and commit
   it; then the maintainer merges. Green checks are evidence, not
   acceptance.

## Changing a skill

Follow the `skill-authoring` project skill: public skills must be
self-contained (no references outside their directory), documented in both
the catalog `README.md` and `README.zh.md`, and behavior changes are tested
against the scenarios of their OpenSpec change.
