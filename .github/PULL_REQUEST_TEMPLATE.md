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

<!-- The OpenSpec change this PR implements (one line per change; a
companion repository change gets its own), or `Spec: none — <reason>` for
a change too small to plan. -->
Spec: openspec/changes/
<!-- `specification` until the maintainer's approval comment names the
approved commit; `implementation` after. -->
Phase: specification

## Validation

<!-- Every command or behavioral test run and its result: scores, evidence,
isolation degradations, skipped cases with reasons. Name the scenarios that
passed and link the change's design.md verification plan for the cases and
rubric instead of restating them. -->

-

## Checklist

- [ ] `just check` passes locally
- [ ] The change record was approved on this draft before implementation (a maintainer comment naming the commit), or `Spec: none` is justified
- [ ] The scenarios of the linked change, or the acceptance criteria of the linked issue, are met and recorded above
- [ ] Every task of the linked change is done and it is archived (or, once the `spec-archive` workflow can push, left for that workflow), or `Spec: none` is justified
- [ ] No secrets, credentials, or personal data in the diff, description, or commits
- [ ] Documentation and paired `README.zh.md` translations are updated where required
- [ ] Catalog READMEs and `marketplace.json` are synchronized where a public skill changed
