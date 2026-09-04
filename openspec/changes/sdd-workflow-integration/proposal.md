## Why

The SDD skill (`engineering`) and the `meta` contract and platform builders
say specifications own acceptance and tracked work links them, but never say
*when* the two interact: when the work item opens, when the specification is
published for review, where approval is recorded, whether it travels with
its implementation or in its own change request, and when a completed change
is archived. Projects improvise those moments, and this repository's harness
contradicts itself (approval "on the draft", yet the draft opens after
implementation). A second defect: the contract builders deposit
platform-neutral files that a platform builder then translates, although the
file agents read daily should speak the project's platform vocabulary.

## What Changes

- `spec-driven-development`: adds the change request shape (combined or
  split), the archive mode (automated or in-request), the approval record,
  the eight-step timing between requirement, specification, draft, review,
  implementation, archive, and merge, the scope of specification review
  (outcome description, never tasks), and a reference that lets the skill
  guide workflow and template design itself when the harness builder is
  declined or absent. Rewords the description and the tracked-work rules.
- `meta/meta-spec-workflow`: asks and records change request shape, archive mode,
  default specification author, and where approval is recorded; the deposited
  contract and every tool reference carry them. The contract is deposited in
  the platform's vocabulary.
- `meta/meta-workflow-design`: designs in the platform-neutral model but deposits
  the workflow file in the platform's vocabulary at
  `.agents/knowledge/<platform>-workflow.md`; the platform becomes a
  required fact; the vocabulary check inverts; a specification-only draft is
  work in progress.
- `meta/meta-agent-authority`: the approval gate precedes review admission; the
  gate passing on an agent-authored specification satisfies H1's
  specification precondition; the policy is deposited in platform verbs.
- `meta-github-workflow` and `meta-gitlab-workflow`: read the platform-worded
  deposits and implement them without a second planning file; express the
  change request shape in the project skill, templates, and state machine;
  ship a runnable OpenSpec archive job (serialized, idempotent) as an asset
  with a shared script, and design guidance only for tools without a fixed
  archive operation.
- The repository's own harness alignment (archive workflow, knowledge files,
  `change-workflow`, pull request template, schema and layout) is the
  companion repository change `sdd-workflow-integration-harness`.

## Skills touched

- `engineering/spec-driven-development` (new): change request shape, archive mode, approval record, timing, review scope, harness design without a builder, description triggers.
- `meta/meta-spec-workflow` (new): questions and deposit for shape, archive mode, author, approval record; platform-vocabulary deposit; tool references.
- `meta/meta-workflow-design` (new): neutral-model design with platform-vocabulary deposit, platform as a required fact, specification-only drafts.
- `meta/meta-agent-authority` (new): approval gate versus review admission, H1 under a specification contract, platform-verb deposit.
- `meta/meta-github-workflow` (new): consuming platform-worded deposits, change request shape in the project skill and templates, the OpenSpec archive workflow asset, design guidance for other tools.
- `meta/meta-gitlab-workflow` (new): the same for GitLab.

## Installed behavior

- `spec-driven-development`: gains capabilities (shape, archive mode, timing, review scope, builder-less harness design) → `feat`.
- `meta-spec-workflow`, `meta-github-workflow`, `meta-gitlab-workflow`: gain the shape and archive-mode questions, expressions, and assets → `feat`; their platform-vocabulary deposit corrects a misleading rule → `fix`, landed as its own commit.
- `meta-workflow-design`: the platform-neutral deposit and the "a draft is never a placeholder" gotcha were wrongly restrictive → `fix`.
- `meta-agent-authority`: H1's "the human supplies a complete specification" was wrongly restrictive under a specification contract → `fix`.

## Impact

- Skills: `skills/engineering/spec-driven-development/`,
  `skills/meta/meta-spec-workflow/`, `skills/meta/meta-workflow-design/`,
  `skills/meta/meta-agent-authority/`, `skills/meta/meta-github-workflow/`,
  `skills/meta/meta-gitlab-workflow/`; catalog `CONTEXT.md`, `README.md`,
  `README.zh.md` rows for `engineering`, `meta`, and one sentence in
  `scaffold/CONTEXT.md`.

## Non-goals

- Writing specifications for skills or tools this change does not touch.
- Changing `meta-git-branching`, which already deposits in platform terms.
- Building automated archiving for Spec-Kit, Kiro, or committed documents;
  those get design guidance only.
- Changing `scripts/check_pr_policy.py` or the required checks.

## Tracked work

No issue: user-directed change planned in conversation.
