# Commit conventions: doc, validator, MR-pipeline enforcement

Loaded when the task standardizes commit messages. Inputs from "Assess
the project first": the analyzer report
(`python3 scripts/analyze_history.py --max 500`), commit rules in
`CONTRIBUTING.md` / `AGENTS.md`, a `.gitmessage` template, the pipeline
style, and whether the project squash-merges (`squash_option`).

The convention includes the **`Changelog:` Git trailer** by default — it
is what `glab changelog generate` and the release-conventions domain
build on.

## Define the convention

Copy [assets/commit-conventions.md](assets/commit-conventions.md) to
`docs/commit-conventions.md` (or the project's docs location) and settle
every `{{...}}` placeholder **with the user**, informed by the analyzer
report: keep the types the history actually uses, add missing standard
ones deliberately, decide whether scopes are required and from what set,
fix the subject length limit, and decide the `Changelog:` trailer rule —
the default is "every user-facing commit carries `Changelog: <category>`",
because that single habit is what makes `glab changelog generate` work
downstream.

Done when: the convention doc has no `{{...}}` left and the user has
approved the types table, scope rule, and trailer rule.

## Install the validator

Copy [assets/check_commits.py](assets/check_commits.py) to
`scripts/check_commits.py` in the target repository and edit its `CONFIG`
block (top of file) to match the convention doc exactly: types, scope
rule, subject cap. The validator is python3-stdlib-only, so it runs on
any CI runner and any contributor machine with no installation.

Smoke-test it against the project's own recent history:

```bash
python3 scripts/check_commits.py --range HEAD~20..HEAD || true
```

Findings on historical commits are expected when the convention is new —
the CI job below validates only new MR commits, never history.

Done when: the validator exits 0 on a compliant test message
(`python3 scripts/check_commits.py --message "feat: add x"`) and reports
findings, not a crash, on the history sample.

## CI validation

Copy the job in [assets/commit-check-job.yml](assets/commit-check-job.yml)
into the project's pipeline config — `.gitlab-ci.yml`, or a file it
`include:`s (job `rules:` are evaluated the same either way; create
`.gitlab-ci.yml` with just this job if the project has none). The job
runs in merge request pipelines (Free), is tokenless (safe on fork MRs —
never add secrets to it), sets `GIT_DEPTH: "0"` so the merge base is
present, and validates
`$CI_MERGE_REQUEST_DIFF_BASE_SHA..$CI_COMMIT_SHA` with the committed
validator. If the assessment found classic branch pipelines, adding an
MR-event job can spawn duplicate pipelines — resolve that with the user
per the MR-conventions domain. If the project squash-merges, read
[rule-customization.md](rule-customization.md) for the MR-title variant.

Validate the edited `.gitlab-ci.yml` with `glab ci lint` when glab is
authenticated for the host.

Done when: the job is in the pipeline config, the YAML parses (or
`glab ci lint` passes), and it references the validator path it was
actually installed at.

## Generate the project-level skill

For the default deliverable, copy
[assets/project-skill-commits.md](assets/project-skill-commits.md) to
`<skills-dir>/<project-name>-commits/SKILL.md` and fill every
`{{PLACEHOLDER}}`:

| Placeholder | Fill with |
|---|---|
| `{{PROJECT_NAME}}` | Project name, lowercase, hyphens only |
| `{{CONVENTION_DOC_PATH}}` | Where the convention doc was installed |
| `{{TYPES_TABLE}}` | The convention doc's type → use-for table |
| `{{TYPES_LIST}}` | The same types as one comma-separated line |
| `{{SCOPE_RULE}}` | The scope rule sentence from the convention doc |
| `{{TRAILER_RULE}}` | The Changelog-trailer rule sentence |
| `{{SUBJECT_MAX}}` / `{{BODY_LINE_MAX}}` | The limits set in CONFIG |

For the AGENTS.md fallback, copy
[assets/agents-md-commits-section.md](assets/agents-md-commits-section.md)
into the project's `AGENTS.md` and fill the same placeholders (it uses
the list form, not the table).

Done when: the generated deliverable contains no `{{...}}` placeholder
and (for a skill) its frontmatter `name` matches its directory name.

## Customization branches

- Read [rule-customization.md](rule-customization.md) when the shipped
  CONFIG needs custom scopes, types, or required trailers, or when the
  project squash-merges (validate the MR title instead).
- Read [push-rules-premium.md](push-rules-premium.md) when the project
  wants hard server-side commit enforcement and has Premium.

## Gotchas

- `GIT_DEPTH: "0"` on the job is load-bearing: GitLab's default shallow
  clone may not contain the merge base, so the range cannot resolve and
  git fails the job with "bad object" — not a silent pass.
- `$CI_MERGE_REQUEST_DIFF_BASE_SHA` (and the other `CI_MERGE_REQUEST_*`
  variables) exist only in merge request pipelines — the job's `rules:`
  keeps it out of branch and tag pipelines, where the range would be
  empty.
- Merge, revert, and `fixup!`/`squash!` commits are exempted by pattern
  in the validator — hand-tightening the regexes to catch them again
  breaks normal GitLab merge flows.
- Validate only the MR range. Running the validator over all history as
  a blocking job makes the convention retroactive and every MR red.
- In squash-merge projects the **MR title** becomes the squash commit's
  subject by default — validate it too, or the enforced range disappears
  at merge time.
- The validator and the convention doc drift independently — the CONFIG
  block is the enforced truth; when the user changes the doc, change
  CONFIG in the same commit.
