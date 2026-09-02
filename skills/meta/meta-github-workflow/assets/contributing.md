# Contributing to {{PROJECT_NAME}}

<!-- Rework to the flow this harness actually built; this file and the
internal workflow are one system and change in the same PR. -->

## Setup and checks

{{SETUP_COMMANDS}} · run {{LOCAL_CHECK_COMMAND}} before pushing — CI runs
the same commands ({{CHECKS_DOC_PATH}} maps jobs to commands).

## Where things go

Bugs, features, and tasks → the issue forms. Questions and ideas →
{{DISCUSSIONS_LINK_OR_CHANNEL}}. Vulnerabilities →
{{SECURITY_CHANNEL — never public issues}}.

## Working on a change

1. Comment on the issue you are taking, or open one first for anything
   non-trivial; wait for triage on {{TRIAGE_EXPECTATION}}.
2. Branch from `{{DEFAULT_BRANCH}}` ({{BRANCH_NAMING — gh issue develop
   creates <number>-<slug>}}), and open a draft PR early with
   `Closes #N`.
3. Commits: {{COMMIT_RULE_SUMMARY — under squash merge, the PR title is
   what must conform}}.
4. Green checks are evidence, not acceptance. Mark the PR ready when the
   checklist is complete and {{REVIEW_EXPECTATION — approvals required and
   from whom}} is met; agents follow the authority policy at
   {{AUTHORITY_POLICY_PATH — e.g. .agents/knowledge/agent-authority.md}},
   which by default reserves ready and review requests for a human.

## Merging and releases

{{MERGE_METHOD}} · releases: {{RELEASE_SUMMARY_POINTER}}.
