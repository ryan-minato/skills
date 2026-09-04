## GitLab lifecycle

- Host and project: <verified host and full nested project path>.
- Planning: <Kanban, Scrum, other, or no board>; the source of truth is
  <work-item/query location>; <owner> grooms it <cadence>.
- Branches, issues, merge requests, labels, and milestones: read
  `.agents/knowledge/gitlab-workflow.md` before creating a branch, opening
  or editing an issue or merge request, or creating a planning object.
- Work execution: use the project GitLab workflow at <project skill or workflow
  path> before taking a task, issue, or incident; open an early draft MR and
  keep its state current.
- Publishing: review the exact final payload and diff for credentials,
  confidential information, sensitive personal data, internal identifiers,
  and unintended quick actions. Publish only with `SAFE TO PUBLISH: YES` and
  applicable external-write approval.
- CI: <job-to-local-command mapping or no-runner/no-CI decision and revisit
  trigger>.
- Ownership and protected settings: read <reachable settings path> before
  changing CODEOWNERS, protected refs, approvals, scanners, variables,
  environments, packages, or releases.
- Update triggers: <concrete implementation-to-harness relationships>.
