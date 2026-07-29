# Commit conventions: doc, validator, CI enforcement

Loaded when the task standardizes commit messages. Inputs from "Assess
the project first": the analyzer report
(`python3 scripts/analyze_history.py --max 500`), any existing commitlint
config or `.gitmessage`, commit rules in `CONTRIBUTING.md` / `AGENTS.md`,
and whether the repo squash-merges.

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
[rule-customization.md](rule-customization.md) for the PR-title variant.

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

Done when: the generated deliverable contains no `{{...}}` placeholder
and (for a skill) its frontmatter `name` matches its directory name.

## Customization branches

- Read [rule-customization.md](rule-customization.md) when the shipped
  CONFIG needs custom scopes, types, or required trailers, when the
  repository squash-merges (validate the PR title instead), or for a
  cheaper `fetch-depth` on huge repositories.
- Read [commitlint-alternative.md](commitlint-alternative.md) only when
  the user explicitly wants commitlint instead of the shipped validator.

## Gotchas

- `fetch-depth: 0` on the checkout is load-bearing: the default shallow
  clone has no base commit, so the range cannot resolve and git fails
  the check on every PR with "bad object".
- Merge, revert, and `fixup!`/`squash!` commits are exempted by pattern
  in the validator — hand-tightening the regexes to catch them again
  breaks normal GitHub merge flows.
- Validate only the PR range. Running the validator over all history as
  a required check makes the convention retroactive and every PR red.
- In squash-merge repositories the PR **title** becomes the squash
  commit's subject — validating commits without validating the title
  enforces the wrong thing.
- The validator and the convention doc drift independently — the CONFIG
  block is the enforced truth; when the user changes the doc, change
  CONFIG in the same commit.
