## Context

See proposal.md for motivation. `openspec/changes/label-sync-workflow-asset/design.md` covers the skill change this companion serves.

**Current state:**
- `.agents/knowledge/harness-maintenance.md` `## Synchronization register` lists the pairs no script checks. The `openspec-workflow` asset row (`.agents/knowledge/harness-maintenance.md:31`) is the precedent: it pairs that skill's workflow assets with this repository's `.github/workflows/spec-{command,labels}.yml` as "placeholders resolved; same steps". Its "which nothing keeps identical to the asset" refers only to this repository's `scripts/spec_changes.py`, not to the workflows.
- `.github/workflows/labels-sync.yml` (job `labels / sync`) runs `scripts/sync_labels.py --apply` on a path-filtered push to `main`, weekly, and on dispatch. It never passes `--prune` and reports prune candidates as warnings. Its record is `openspec/changes/archive/2026-09-18-labels-sync-workflow/`.
- The management-code change (#95) removed the copies check. No validator pairs the skill's `sync_labels.py` with this repository's, and none will pair the workflow asset with this workflow.

**Binding rules:**
- `.agents/knowledge/skill-quality.md` `## Management code` (R1): management code need not be identical to another copy, and no rule, check, or instruction keeps two copies identical. A register row records that two files share steps, so a change to one is looked at in the other. It does not bind their content.
- The Extension-slots row (`.agents/knowledge/harness-maintenance.md:32`) is unaffected: the skill change stays outside `## Extension slots`.

## Placement

| What Changes bullet | File and section | Proof |
|---|---|---|
| Synchronization register | `.agents/knowledge/harness-maintenance.md` `## Synchronization register`: one row after the `openspec-workflow` asset row (`:31`) | readback of the row against both files; `just validate` |

## Decisions

- **Register the pair** (settled by the maintainer while scoping the change; the decisions after it are for the maintainer to confirm on the draft).
  - The asset is shaped on this repository's workflow, and each will be fixed independently. The register is the existing place for such pairs, as the `openspec-workflow` asset row shows.
  - Rejected: no row. The pair would drift silently, and a fix proven here, such as the report step, would not reach targets.
- **The mirror is "the same triggers, grants, guard, and steps", not the whole file.** Two differences are named in the row. The schedule is a marked setting in the asset, optional for a target, and this repository keeps it. The asset's apply step derives the GitHub Enterprise Server host from the run's server URL and passes the enterprise token variable (the skill design's GHES decision); this repository's workflow runs on github.com and does not carry those environment lines. Rejected: "placeholders resolved, identical", which would bind the files against R1 and fail on both differences.
- **No mechanical check.** Rejected: a validator comparing the steps. It rebuilds the binding R1 removed, for a pair that changes rarely.
- **Owner: author**, like the other rows for files an agent edits in the same pull request. Rejected: maintainer, which would put a routine step sync into the maintainer's queue.

## Risks / Trade-offs

- **[The row lands before the asset exists]** → The companion lands on the same branch as the skill change, and its row is written in the same pull request, after the asset is committed.
- **[A reader takes the row as an identity requirement]** → The row carries its own clause saying nothing keeps the two files identical. The `openspec-workflow` row's clause cannot be reused for this: it speaks of `scripts/spec_changes.py` only.
- **[A textual conflict with another open change editing the register]** → The table is append-only in practice. Whichever change lands second rebases, and the conflict is one line.

## Verification plan

- **Synchronization register.**
  - A readback of the new row. Both paths it names exist. The steps it names (triggers, job grants, branch guard, apply step, report step) appear in both files. Of the two differences it names, the schedule is a marked setting in the asset and present in this repository's workflow, and the GitHub Enterprise Server environment lines appear only in the asset's apply step.
  - The row says nothing keeps the files identical.
  - `git diff --stat origin/main...HEAD -- .github scripts` is empty for this companion.
- **Everything.** `just validate` and `just check`.
