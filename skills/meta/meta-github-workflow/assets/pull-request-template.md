<!-- .github/pull_request_template.md — the checklist workflow parses the
headings of this file, so renaming a heading and updating the workflow is
one change, not two. -->

## What and why

<!-- The outcome or behavior this changes and why it matters. Not the diff. -->

## Changes

<!-- Where and what, briefly: component or path, then the change. -->

## Related work

<!-- Closes #N — the closing keyword drives the issue lifecycle. -->
{{SPEC_LINE — under a specification contract, the comment line
"<!-- Spec: <path or change name> -->"; delete this placeholder otherwise}}

## Checklist

- [ ] {{LOCAL_CHECK_COMMAND}} passes locally
- [ ] Acceptance criteria of the linked issue, or the scenarios of the linked specification, are met
{{SPEC_CHECK — under a specification contract, the item
"- [ ] Specification updated or change record archived where behavior changed";
delete this placeholder otherwise}}
- [ ] No secrets, credentials, or personal data in the diff, description, or commits
