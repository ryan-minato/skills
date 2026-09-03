# Contributing

Thank you for improving this skill library. This page is the human-facing
summary; the full conventions live in [AGENTS.md](../AGENTS.md) and are the
same for people and agents.

## Where things go

- Bugs, feature requests, and planned tasks: the
  [issue forms](https://github.com/ryan-minato/skills/issues/new/choose).
- Questions and ideas:
  [Discussions](https://github.com/ryan-minato/skills/discussions).
- Vulnerabilities: [SECURITY.md](SECURITY.md), never a public issue.

## Setup and checks

Open the repository in its dev container, or install `just`, `pre-commit`,
`ruff`, Python 3.10+, and Node.js 20.19+ yourself, then:

```bash
just setup
just check
```

`just check` is exactly what CI runs (see
[.agents/knowledge/github-checks.md](../.agents/knowledge/github-checks.md)).

## Working on a change

1. Open an issue first for anything that needs discussion or priority; a
   small, self-contained fix may skip it and say `N/A — <reason>` in the
   pull request instead.
2. Branch from `main` as `<type>/<slug>`. Behavior changes to a skill or a
   repository tool go through an OpenSpec change under
   `openspec/changes/<slug>/`; see
   [.agents/knowledge/spec-workflow.md](../.agents/knowledge/spec-workflow.md).
3. Commits from branches in this repository land by rebase merge, so each
   commit subject follows Conventional Commits (`type(scope): subject`).
   Pull requests from forks are squash-merged: only the pull request title
   must conform, and you need not rewrite your branch history.
4. Open a draft pull request early, using the template. Mark it ready once
   `just check` passes, the checklist is complete, and the Validation
   section records what you ran.
5. The maintainer reviews and merges. Green checks are evidence, not
   acceptance.

## Changing a skill

Follow the `skill-authoring` project skill: public skills must be
self-contained (no references outside their directory), documented in both
the catalog `README.md` and `README.zh.md`, and behavior changes are tested
against the scenarios of their OpenSpec change.
