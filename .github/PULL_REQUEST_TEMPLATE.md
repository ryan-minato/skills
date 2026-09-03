<!-- scripts/check_pr_policy.py reads the `## ` headings of this file from
the base branch and requires each one in the PR body. Keep the word
"secrets" in the security checklist line: the check keys on it. -->

## What and why

<!-- The outcome or behavior this changes and why it matters. Not the diff. -->

## Changes

<!-- Where and what, briefly: skill or path, then the change. -->

## Related work

<!-- `Closes #N`, or `N/A — <reason>` when no issue exists. -->
Closes #

<!-- The OpenSpec change this PR implements, or `Spec: none — <reason>`
for changes that alter no behavior. -->
Spec: openspec/changes/

## Validation

<!-- Every command or behavioral test run and its result. Name the scenarios
that passed. -->

-

## Checklist

- [ ] `just check` passes locally
- [ ] The scenarios of the linked change, or the acceptance criteria of the linked issue, are met and recorded above
- [ ] The change is archived (delta merged into `openspec/specs/`), or `Spec: none` is justified
- [ ] No secrets, credentials, or personal data in the diff, description, or commits
- [ ] Documentation and paired `README.zh.md` translations are updated where required
- [ ] Catalog READMEs and `marketplace.json` are synchronized where a public skill changed
