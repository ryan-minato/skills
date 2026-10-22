<!-- `pr / policy` reads the `## ` headings of this file from the base
branch and requires each one in the PR body; it keys on the word "secrets"
in the security checklist line. Sections beyond these may be added; every
section passes the publish gate (.agents/knowledge/agent-authority.md). -->

<!-- Opening paragraph, no heading: the goal of the change — what is true
once it merges — not the work done. -->

## Why

<!-- The value: why this is worth merging now. Not the diff. -->

## Specification

<!-- One `Spec:` line per change, linking its directory on this branch, or
`Spec: none — <reason>`; Phase and Approval lines as
.agents/knowledge/spec-workflow.md (Approval gates) says. -->
Spec: [openspec/changes/<slug>](https://github.com/ryan-minato/skills/tree/<branch>/openspec/changes/<slug>)
Phase: specification

<!-- The approval package, one link per file; drop a line the change does not have. -->
- [proposal.md](https://github.com/ryan-minato/skills/blob/<branch>/openspec/changes/<slug>/proposal.md)
- [specs/<catalog>/<skill-name>/spec.md](https://github.com/ryan-minato/skills/blob/<branch>/openspec/changes/<slug>/specs/<catalog>/<skill-name>/spec.md)
- [design.md](https://github.com/ryan-minato/skills/blob/<branch>/openspec/changes/<slug>/design.md)
- tasks.md: written after the package deliberation is closed

<!-- Replace this line at each closing, then at the freeze, with the date and commit. -->
Approval: package deliberation open on this draft (`.agents/knowledge/spec-workflow.md`, Approval gates).

## Related work

<!-- `Closes #N`, or `N/A — <reason>` when no issue exists. -->
Closes #

## Changes

<!-- Keep the reserved line until the PR is marked ready; `pr / policy`
fails a ready PR that still holds it. Then list each touched file or
directory as a permalink to the commit that changed it: the exact lines
(`blob/<sha>/<path>#L10-L20`) for a local change, the whole file or
directory (`blob/<sha>/<path>`, `tree/<sha>/<dir>`) for a broad one. -->
_Reserved: filled in when the pull request is marked ready._

## Validation

<!-- Keep the reserved line until the PR is marked ready. Then record every
command or behavioral test run and its result: scores, evidence, isolation
degradations, skipped cases with reasons. Name each scenario with its
result and link the change's design.md verification plan instead of
restating the cases or rubric. -->
_Reserved: filled in when the pull request is marked ready._

## Checklist

<!-- Every item is true when the PR is marked ready; `pr / policy` fails a ready PR with an open box. -->
- [ ] `just check` passes locally
- [ ] The package approval is recorded and reconciled, or `Spec: none` is justified
- [ ] The scenarios of the linked change, or the acceptance criteria of the linked issue, are met and recorded above
- [ ] Every task of the linked change is done and verified, or `Spec: none` is justified
- [ ] No secrets, credentials, or personal data in the diff, description, or commits
- [ ] Documentation and paired `README.zh.md` translations are updated where required
- [ ] Catalog READMEs and `marketplace.json` are synchronized where a public skill changed
