## Context

See proposal.md for motivation. `openspec/changes/spec-command-reply/design.md` carries the decisions for the workflow's admission, parse, header, and head, with their rejected alternatives; this change applies them to this repository's copy and does not restate them.

**Current state:**
- `.github/workflows/spec-command.yml` differs from `skills/sdd/openspec-workflow/assets/github/workflow-spec-command.yml` only in the header comment and the pinned checkout (`diff`, 2026-09-30).
- `.agents/knowledge/github-checks.md` line 14 says the job runs on "a pull request comment starting with `/spec` by a collaborator" and "is skipped for other comments and for bots".
- The permission comment at `spec-command.yml` line 34 says "reading the request and its files is a pull-request read".

**Binding rules:**
- The register row for the sdd workflow assets in `.agents/knowledge/harness-maintenance.md`: this repository's workflow has the asset's steps, with placeholders resolved and the same `scripts/spec_changes.py` path.
- `.agents/knowledge/skill-quality.md` `## Management code`, R5 and R6, as for the asset.
- `spec / command` runs from the default branch, so nothing in this pull request's own runs exercises the new steps.

## Placement

| What Changes bullet | File and section | Proof |
|---|---|---|
| Workflow takes the asset's steps | `.github/workflows/spec-command.yml`: the job's `if:`, the resolving step, the reply step, the `pull-requests: read` comment. The header comment and the checkout pin stay. | `diff` against the asset; the step harness below |
| `github-checks.md` admission | `.agents/knowledge/github-checks.md`, `spec / command` row, the "Runs on" cell | readback; `just validate` |

## Decisions

- **The workflow follows the asset line for line** (workflow bullet). The register asks for the same steps, and the two differ today only in the header and the pin; a local variant would make the next asset change a merge. Rejected: fixing only the parse here and keeping the old admission, which leaves this repository's workflow off the asset's steps.
- **The permission comment is reworded in this change, not #101** (workflow bullet; maintainer decision, recorded with its alternative in the skill change's design).
- **`github-checks.md` names the admitted shapes, not the expression** (github-checks bullet). The row says what a reader needs to predict a run: a comment whose first word is `/spec`, alone or followed by a space, a tab, or a line break. Rejected: quoting the `if:`, a second copy of the workflow that would drift.

## Risks / Trade-offs

- **[The new admission or parse misbehaves live, where only `main`'s workflow runs]** → The step harness below runs this file's bodies before the merge; the maintainer's test pull request after the merge covers admission, including the evaluation-error mode the skill change's design leaves to be verified. A failure found live is fixed by a new fix change on `main`, alongside the skill change's, since this change is archived by then.
- **[Drift between this file and the asset]** → `diff` in verification; the register row names the pair.
- **[#101 edits the same resolving step]** → Merge order #99, this branch, #101; #101 rebases onto this change.

## Verification plan

Per What Changes bullet. Scratch files and stubs live under the session scratch directory and stay untracked.

- **Workflow takes the asset's steps.**
  - `diff .github/workflows/spec-command.yml skills/sdd/openspec-workflow/assets/github/workflow-spec-command.yml` shows only the header comment and the checkout pin.
  - The skill change's harness runs this file's two step bodies with `scripts/spec_changes.py` of this repository, on every case of its table: CR LF, tab, bare `/spec` with more text, unknown command, other case, the two heads with `show` and `status`, closed request, and a snapshot without a head. The expected bodies and exits are those of the skill change's plan.
  - Regression: a `status` exiting 1 posts the reply and fails the step; one exiting 2 posts it and passes.
  - `just validate` passes: the job names still match `github-checks.md`.
  - `grep -n 'git .*fetch\|checkout'` finds only the base checkout.
- **`github-checks.md` admission.** A readback: the row names the admitted shapes and still says the run is skipped for other comments and for bots.
- **Everything.** `just check` and `git diff --stat origin/main...HEAD`.
- **After the merge** (maintainer action): the live admission check of the skill change's verification plan, run on this repository, is this file's live check too: every admitted case gets one reply naming the current head, and `/specs`, `/specification`, `/speckit.plan`, and a plain comment show the job as skipped, not failed or errored.
