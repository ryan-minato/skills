<!-- .github/pull_request_template.md — the checklist workflow parses the
headings of this file, so renaming a heading and updating the workflow is
one change, not two. -->

## What and why

<!-- The outcome or behavior this changes and why it matters. Not the diff. -->

## Changes

<!-- Where and what, briefly: component or path, then the change. -->

## Related work

<!-- Closes #N — the closing keyword drives the issue lifecycle. -->
{{SPEC_LINE — under a specification contract, the lines
"Spec: <path or change name>" and "Phase: specification | implementation";
delete this placeholder otherwise}}

## Checklist

- [ ] {{LOCAL_CHECK_COMMAND}} passes locally
- [ ] Acceptance criteria of the linked issue, or the scenarios of the linked specification, are met
{{SPEC_CHECK — under a specification contract, the two items
"- [ ] The change record was approved by the gate owner before implementation, or this pull request carries the specification only" and
"- [ ] The specification is updated, or the change record is archived, or every task is complete and archiving runs after merge";
delete this placeholder otherwise}}
- [ ] No secrets, credentials, or personal data in the diff, description, or commits
