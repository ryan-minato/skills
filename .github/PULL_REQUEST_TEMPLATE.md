<!-- scripts/check_pr_policy.py reads the `## ` headings of this file from
the base branch and requires each one in the PR body. Keep the word
"secrets" in the security checklist line: the check keys on it. Sections
beyond these may be added or updated when the change needs them; every
section passes the publish gate before it is published. -->

<!-- Opening paragraph, no heading: the goal of the change — what is true
once it merges — not the work done. -->

## Why

<!-- The value: why this is worth merging now. Not the diff. -->

## Specification

<!-- The OpenSpec change this PR implements, as a link reviewers can click
(one line per change; a companion repository change gets its own):
`Spec: [openspec/changes/<slug>](https://github.com/ryan-minato/skills/tree/<branch>/openspec/changes/<slug>)`.
`Spec: none — <reason>` only for a change too small to plan. -->
Spec: [openspec/changes/<slug>](https://github.com/ryan-minato/skills/tree/<branch>/openspec/changes/<slug>)
<!-- `specification` until the maintainer's approval comment; `implementation` after. -->
Phase: specification

<!-- The change's records, one link each; design and tasks arrive after approval. -->
| Record | What it holds |
|---|---|
| [proposal.md](https://github.com/ryan-minato/skills/blob/<branch>/openspec/changes/<slug>/proposal.md) | Why, what changes, skills touched, installed behavior, impact, non-goals |
| [specs/<catalog>/<skill-name>/spec.md](https://github.com/ryan-minato/skills/blob/<branch>/openspec/changes/<slug>/specs/<catalog>/<skill-name>/spec.md) | Delta spec: requirements and scenarios (omit for a repository change) |
| design.md, tasks.md | Written after the approval comment |

<!-- Keep the approval line until the comment exists, then replace it with
"Approval: <date>, comment on this draft". The comment covers the record as
of the last push before it; a later push to the record needs a fresh one. -->
Approval: pending — a maintainer comment on this draft; it covers the record as of the last push before it. Review the proposal and the delta specs only; design and tasks are not part of the approval. To approve, post this comment (one line, copy as is):

```
Specification approved
```

## Related work

<!-- `Closes #N`, or `N/A — <reason>` when no issue exists. -->
Closes #

## Changes

<!-- Filled in when the PR is marked ready. Each touched file or directory
as a permalink to the commit that changed it: the exact lines
(`blob/<sha>/<path>#L10-L20`) for a local change, the whole file or
directory (`blob/<sha>/<path>`, `tree/<sha>/<dir>`) for a broad one. -->
_Reserved: filled in when the pull request is marked ready, as permalinks to its commits — the exact lines for a local change, the whole file or directory for a broad one._

## Validation

<!-- Filled in when the PR is marked ready. Every command or behavioral
test run and its result: scores, evidence, isolation degradations, skipped
cases with reasons. Name the scenarios that passed and link the change's
design.md verification plan for the cases and rubric instead of restating
them. -->
_Reserved: filled in when the pull request is marked ready, naming each scenario with its result and linking the verification plan in `design.md`._

## Checklist

- [ ] `just check` passes locally
- [ ] The change record was approved on this draft before implementation (a maintainer comment `Specification approved` later than the record's last push), or `Spec: none` is justified
- [ ] The scenarios of the linked change, or the acceptance criteria of the linked issue, are met and recorded above
- [ ] Every task of the linked change is done and it is archived (or, once the `spec-archive` workflow can push, left for that workflow), or `Spec: none` is justified
- [ ] No secrets, credentials, or personal data in the diff, description, or commits
- [ ] Documentation and paired `README.zh.md` translations are updated where required
- [ ] Catalog READMEs and `marketplace.json` are synchronized where a public skill changed
