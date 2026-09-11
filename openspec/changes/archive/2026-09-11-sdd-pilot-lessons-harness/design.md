## Context

See proposal.md. This repository's harness pins the archive script's origin
in `scripts/validate_harness.py` (mirror pair list) and in the register
`.agents/knowledge/harness-maintenance.md`; its specification contract
`.agents/knowledge/spec-workflow.md` still says the approval is the fixed
comment on the draft; the `change-workflow` project skill §3 and §7, the
pull request template's approval line and checklist item 2, and
`agent-authority.md`'s ready condition repeat that rule; the `code-review`
project skill and `ARCHITECTURE.md` know two builder kinds.
`scripts/check_pr_policy.py` reads only headings, reserved lines, the
`Phase:` line, the `Spec:` line, and checkbox state, so the approval-line
wording is free. The push path for `spec-archive.yml` and the three
"Actions identity on the bypass list" sentences are out of scope
(`automated-archive`).

## Placement

| What Changes bullet | Files | Proof |
|---|---|---|
| Archive mirror | `scripts/validate_harness.py` (mirror pair and docstring), `.agents/knowledge/harness-maintenance.md` (origin row, new slot-table row), `.github/workflows/spec-archive.yml` (header comment) | `just validate`; `diff` of the pair empty |
| Specification gate | `.agents/knowledge/spec-workflow.md` (Lifecycle step 2–3, tracked-work approval bullet), `.agents/skills/change-workflow/SKILL.md` §3 (publish, stop; on closing pull comments and review threads and reconcile) and §7, `.github/PULL_REQUEST_TEMPLATE.md` (approval line, checklist item 2), `.agents/knowledge/agent-authority.md` (ready condition) | clean-context readback of the four files; `scripts/check_pr_policy.py` against a draft and a ready body built from the new template |
| Purpose-line exception | `.agents/knowledge/spec-workflow.md` main-spec rule | readback; `just spec-validate` after the hand edits land |
| Contract flow | `.agents/skills/code-review/SKILL.md` contract-flow rule, `ARCHITECTURE.md` catalog description, `.claude-plugin/marketplace.json` `meta` description | readback; `just gen-marketplace` leaves the description as edited |

## External impact

The skill change on the same branch supplies the new origin file; until
its commit lands, `just validate` fails on the mirror pair, so the
companion's mirror commit follows the `meta-spec-workflow` commit in the
rebase order. No public skill, README row, or symlink changes here.

## Decisions

- **The gate switches in the same pull request** (maintainer's decision):
  this change itself runs the discussion-closed flow, which is the first
  evidence of the rule.
- **`check_pr_policy.py` is unchanged**: the `Phase:` and reserved-line
  rules hold in both approval modes; a draft-aware body check is a later
  change.
- **The bypass sentences stay** until `automated-archive` settles the push
  path.

## Risks / Trade-offs

- [The mirror check fails between the script move and the origin
  repoint] → one commit order, verified by `just validate` at each commit
  that touches either side.
- [The template's approval line and the contract disagree] → the readback
  asks both files the same question.

## Verification plan

- Archive mirror: `just validate` green; `diff
  scripts/archive_completed_changes.py
  skills/meta/meta-spec-workflow/scripts/archive_completed_changes.py`
  empty; `grep -n 'meta-spec-workflow' scripts/validate_harness.py
  .agents/knowledge/harness-maintenance.md .github/workflows/spec-archive.yml`
  hits all three.
- Specification gate: a clean-context readback given `spec-workflow.md`,
  `change-workflow/SKILL.md`, the PR template, and `agent-authority.md`
  answers "what happens after the draft is published", "what records the
  approval", "what the agent does when the maintainer closes the
  discussion", and "when may the agent mark ready" consistently across
  the four files and names no fixed comment; `python3
  scripts/check_pr_policy.py` passes a draft body from the new template
  and fails a ready body whose Changes section is still reserved.
- Purpose-line exception and contract flow: the same readback answers
  "when may a main spec be edited by hand" and "how many builder kinds
  exist and which runs twice"; `just spec-validate`; `just check`.
- This pull request itself: stopped after the draft, records adjusted on
  instruction, discussion closed in conversation, threads reconciled
  (none existed), then design and implementation — recorded in the
  Validation section.

## Open Questions

None.
