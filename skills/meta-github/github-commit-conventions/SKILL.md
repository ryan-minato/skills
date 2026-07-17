---
name: github-commit-conventions
description: >
  Commit conventions for a GitHub repository — author, enforce in CI, and teach the
  repo's commit-message rules. Use when standardizing commit messages ("set up
  conventional commits", "enforce commit format in CI", "define our commit types");
  when commit history is inconsistent and should be fixed going forward; when a repo
  gains contributors and needs written commit rules; when an existing commitlint or
  .gitmessage setup needs enforcement built around it; when asked what commit types or
  scopes a project should use; or when creating a skill for writing commits in this
  repo. Not for drafting one commit message or committing changes
  (conventional-commits, git-commit), nor for GitLab (gitlab-commit-conventions).
license: Apache-2.0
compatibility: >
  scripts/analyze_history.py and the shipped assets/check_commits.py
  require Python 3.9+ (stdlib only) and git.
---

# GitHub Commit Conventions

Author the files that define how a repository's commit messages are
written and enforced: a convention document grounded in the history the
repo already has, a validator that ships *into* the repository, a CI
workflow that runs it on every PR, and a project-level skill (or
AGENTS.md section) that teaches agents to write compliant messages. This
skill writes local files — only its outputs land in the repository.
Day-to-day PR operations belong to `github-pull-requests`; release and
tag policy to `github-release-conventions`.

## Assess the project first

Before authoring anything, inventory what the repository already has:
run [scripts/analyze_history.py](scripts/analyze_history.py) —

```bash
python3 scripts/analyze_history.py --max 500
```

— which prints one JSON object: how many recent titles already follow a
`type:` prefix or gitmoji style, the type and scope frequencies, subject
lengths, and the trailer keys in use. Also check: an existing
`commitlint` config or `.gitmessage` template, commit rules already
stated in `CONTRIBUTING.md` / `AGENTS.md` / `CLAUDE.md`, existing
workflows in `.github/workflows/` (name collisions), whether the repo
squash-merges (`gh repo view -R O/R --json squashMergeAllowed,
mergeCommitAllowed`), and where project skills live — use
`.claude/skills/` if it exists, else `.agents/skills/` if it exists, else
plan to create `.agents/skills/`. Never invent structure parallel to what
the project already defines: build on what exists, or get the user's
explicit approval to replace it.
Done when: the inventory is written down and each deliverable below is
marked "new", "extends existing", or "replaces (approved)".

## Choose the deliverable

The default deliverable for workflow guidance is a **project-level agent
skill** in the skills directory found during assessment. When the project's
harness does not support skills, or the user prefers documentation, deliver
an `AGENTS.md` section (create the file if missing) or a standalone doc
instead. Ask the user once, before generating, and record the choice. All
other artifacts (templates, configs, workflows, validators) ship regardless
of this choice.

## Define the convention

Copy [assets/commit-conventions.md](assets/commit-conventions.md) to
`docs/commit-conventions.md` (or the project's docs location) and settle
every `{{...}}` placeholder **with the user**, informed by the analyzer
report: keep the types the history actually uses, add missing standard
ones deliberately, decide whether scopes are required and from what set,
and fix the subject length limit. The defaults are Conventional Commits
1.0.0 with the eleven standard types and a 72-character subject cap.

Done when: the convention doc has no `{{...}}` left and the user has
approved the types table and scope rule.

## Install the validator

Copy [assets/check_commits.py](assets/check_commits.py) to
`scripts/check_commits.py` in the target repository and edit its `CONFIG`
block (top of file) to match the convention doc exactly: types, scope
rule, subject cap. The validator is python3-stdlib-only, so it runs on
any CI runner and any contributor machine with no installation.

Smoke-test it against the repo's own recent history:

```bash
python3 scripts/check_commits.py --range HEAD~20..HEAD || true
```

Findings on historical commits are expected when the convention is new —
the CI workflow below validates only new PR commits, never history.

Done when: the validator exits 0 on a compliant test message
(`python3 scripts/check_commits.py --message "feat: add x"`) and reports
findings, not a crash, on the history sample.

## CI validation

Copy [assets/workflow-commit-check.yml](assets/workflow-commit-check.yml)
to `.github/workflows/commit-check.yml`. It runs on `pull_request`,
checks out with `fetch-depth: 0` (mandatory — a shallow clone makes the
range empty), and validates `base.sha..head.sha` with the committed
validator. First-party actions only (`actions/checkout`); the validation
itself is a plain `run:` step, so there is no third-party action or npm
dependency. If the repository squash-merges, read
[references/rule-customization.md](references/rule-customization.md) for
the PR-title variant.

Done when: the workflow file parses and references the validator path it
was actually installed at.

## Generate the project-level skill

For the default deliverable, copy
[assets/project-skill-commits.md](assets/project-skill-commits.md) to
`<skills-dir>/<repo-name>-commits/SKILL.md` and fill every
`{{PLACEHOLDER}}`:

| Placeholder | Fill with |
|---|---|
| `{{REPO_NAME}}` | Repository name, lowercase, hyphens only |
| `{{CONVENTION_DOC_PATH}}` | Where the convention doc was installed |
| `{{TYPES_TABLE}}` | The convention doc's type → use-for table |
| `{{TYPES_LIST}}` | The same types as one comma-separated line |
| `{{SCOPE_RULE}}` | The scope rule sentence from the convention doc |
| `{{SUBJECT_MAX}}` / `{{BODY_LINE_MAX}}` | The limits set in CONFIG |

For the AGENTS.md fallback, copy
[assets/agents-md-commits-section.md](assets/agents-md-commits-section.md)
into the project's `AGENTS.md` and fill the same placeholders (it uses
the list form, not the table).
Refinement beyond the template pairs with `great-skill-writer`
(`npx skills add ryan-minato/skills --skill great-skill-writer`).

Done when: the generated deliverable contains no `{{...}}` placeholder
and (for a skill) its frontmatter `name` matches its directory name.

## Deliver

Everything this skill wrote is local files — nothing is published yet. Hand
the changes to the project's normal git flow (branch, commit, review); that
flow, not this skill, publishes them and carries its own review gates.
Done when: the user has the list of every file created or changed, one line
each on what it does, and any follow-up steps (the first PR to watch the
workflow on, branch protection to require the check).

## Gotchas

- `fetch-depth: 0` on the checkout is load-bearing: the default shallow
  clone has no base commit, so the range cannot resolve and git fails
  the check on every PR with "bad object" (see
  references/rule-customization.md for a cheaper depth on huge repos).
- Merge, revert, and `fixup!`/`squash!` commits are exempted by pattern
  in the validator — hand-tightening the regexes to catch them again
  breaks normal GitHub merge flows.
- Validate only the PR range. Running the validator over all history as
  a required check makes the convention retroactive and every PR red.
- In squash-merge repositories the PR **title** becomes the squash
  commit's subject — validating commits without validating the title
  enforces the wrong thing (see references/rule-customization.md).
- The validator and the convention doc drift independently — the CONFIG
  block is the enforced truth; when the user changes the doc, change
  CONFIG in the same commit.
