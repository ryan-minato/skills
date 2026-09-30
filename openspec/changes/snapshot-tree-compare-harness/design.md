## Context

See proposal.md for motivation. `openspec/changes/snapshot-tree-compare/design.md` covers the assets this change follows, with the decisions, the docs quotes, and the live evidence for the comparison's merge base.

**Current state** (origin/main at 10e29c0):
- **`scripts/spec_changes.py`** (1064 lines). Its snapshot region (`Api`, `listing_at`, `change_dirs`, `build_snapshot`, lines 360-540) is identical to the `openspec-workflow` asset's. It differs from the asset only in the header, the `run_openspec` wording, and `cmd_archive`. `MAX_FILES = 3000` is at :75, `pull_files` at :365, the call at :487, and the flag at :992.
- **Callers.**
  - `spec / labels` (`spec-labels.yml:46`) and `spec / command` (`spec-command.yml:62`) run `snapshot` with no cap flag and read the document in the same job.
  - `checks / spec` runs `check` through `just spec-check`, with the git source.
  - `validate_harness.py` runs `labels --taxonomy`.
  - Agents run `archive` with the git source.
- **Register.** The `.agents/knowledge/harness-maintenance.md` row for `skills/sdd/openspec-workflow/assets/…` requires "the same subcommands, flags, and output in this repository's own `scripts/spec_changes.py`".
- **Prose.** `.agents/knowledge/github-checks.md` item 5 (:44) says the snapshot "pulls the file list and the documents from the REST API". A whole-word search for "file list", "files touched", `max-files`, and `MAX_FILES` outside `skills/` and `openspec/changes/` finds only item 5 and the script itself. The one other stale sentence is the `spec-command.yml` permission comment ("reading the request and its files"), which that search does not match.

**Binding rules:**
- **Commits.** Every commit passes `just check` on its own, because branches are rebase-merged.
- **Management code.** `.agents/knowledge/skill-quality.md` `## Management code`: the standard library only, `python3`, an interface checked where its data enters, and no fallback standing in for a failure.
- **Privileged runs.** `spec / command` and `spec / labels` run from `main`, never from this pull request.

## Placement

| What Changes bullet | File and section | Proof |
|---|---|---|
| `snapshot` by tree comparison | `scripts/spec_changes.py`: the same snapshot, `SnapshotSource`, and source-interface edits as the asset (see the skill change's Placement); `MAX_FILES`, `pull_files`, and `--max-files` removed; module docstring and epilog | script harness below |
| Knowledge wording | `.agents/knowledge/github-checks.md`, item 5 of the privileged-jobs list | readback |
| Workflows unchanged | `.github/workflows/spec-labels.yml`, `spec-command.yml` (read, not edited) | read-through below |

## Decisions

- **Apply the asset's decisions unchanged** (snapshot bullet). The tree comparison against the comparison's merge base, the base-tip listing kept for `at_base`, the removal of `--max-files`, and `spec-snapshot/2` with `head.sha` kept all follow the skill change's decisions. The register row requires the same flags and output, and this repository's workflows are the skill's workflow assets with placeholders resolved.
  - Rejected: leaving this script on the file list until a later change. The register row would be broken, and this repository would keep the failure the skill change removes.
- **Keep the snapshot code the asset's line for line where the roles match** (snapshot bullet). The two are management scripts and nothing binds them. Identical code in the snapshot region keeps the register row true with the least review.
- **Leave the permission comment to the #96 and #100 change** (non-goals). That change rewrites the same step of `spec-command.yml`. Editing the comment here would make two branches touch one hunk. If that change lands without the edit, this branch rewords the comment after rebasing.

**To confirm on the draft.** The skill change's list applies here unchanged, and this change adds one item: this repository's script follows the asset in the same pull request rather than in a later change (Decisions, first bullet).

## Risks / Trade-offs

- **[The live jobs cannot be exercised by this pull request]** → The stub-server harness below. After the merge, the next pull request of this repository that carries a `/spec` command is read back, and its labels are checked against `just spec-changes status`.
- **[The script drifts from the asset]** → A `diff` of the snapshot region against the asset after the edit shows no difference beyond what the header and `archive` already make.
- **[`archive` breaks with the source-interface change]** → The harness runs `archive` on a complete change and on an identical second run.
- **[Merge order with the other sdd changes]** → The skill change's risk applies here too. This branch rebases after #99 and after the #96 and #100 change, and the `spec-command.yml` comment is read after each rebase.

## Verification plan

Per What Changes bullet. The scratch repositories and the stub server live under the session scratch directory, stay untracked, and are removed afterwards.

- **`snapshot` by tree comparison.**
  - `--help` exits 0 and lists the six subcommands. `snapshot --max-files 10` and `--bogus` each exit 2 naming the option.
  - The skill change's stub-server fixture (a request of more than 3000 files, a change edited only on `main`, a change moved into the archive, and a variant touching nothing under `openspec/changes/`), run against `scripts/spec_changes.py`:
    - `snapshot` exits 0 and never requests `pulls/{n}/files`;
    - `status --json`, `status`, `show`, and `labels --json --current ''` are byte-identical to the git source run with `--base <base.sha> --head <head.sha>`, the full SHAs the document records, since `status --json` echoes the refs as given;
    - the base-only change is absent, and the archived change appears once;
    - the variant yields no related change and no desired label;
    - `head.sha` equals the branch tip.
  - The comparison without `merge_base_commit`, a list, and a 404 each fail naming the endpoint, with no `--out` file. A `/1` document read by `status --snapshot` exits 1 naming the schema.
  - The stub answering 502 to `compare/{b}...{h}` on every attempt: its log shows exactly three `compare` requests, `snapshot` exits 1 naming the `compare` endpoint and HTTP 502, and no `--out` file is written.
  - `archive --base main --head HEAD` in the scratch repository archives a complete change, and an identical second run changes nothing.
  - A second `snapshot` gives a byte-identical document.
  - `diff` of the snapshot region against `skills/sdd/openspec-workflow/assets/spec_changes.py` shows no difference.
  - Read-only live: `snapshot --repo ryan-minato/skills --pr 95`, then `status`, matches `status --base <base.sha> --head <head.sha>` over the local clone. The same runs against this change's draft pull request.
- **Knowledge wording.** A readback of `github-checks.md` item 5 against the script. `git grep -n -i -w -e 'file list' -e 'files touched' -e 'max-files' -e 'MAX_FILES' -- ':!skills/' ':!openspec/changes/'` is empty; on main it lists `github-checks.md:44` and the script's cap code. The `spec-command.yml` permission comment is read after each rebase (Risks).
- **Workflows unchanged.**
  - A read-through of `spec-labels.yml` and `spec-command.yml`: no cap flag, `contents: read` in both jobs, and the `pull-requests` scope each already has (`write` for the labels, `read` for the command), which `pulls/{n}` needs.
  - `just validate` passes, including the `spec` label check that runs `labels --taxonomy`.
- **Everything.** `just check`, and `git diff --stat origin/main...HEAD`.
