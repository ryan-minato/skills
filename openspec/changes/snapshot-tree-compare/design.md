## Context

See proposal.md for motivation. The companion repository change `snapshot-tree-compare-harness` carries this repository's own `scripts/spec_changes.py` and knowledge prose.

**Current shape** (origin/main at 10e29c0):
- **`openspec-workflow` `assets/spec_changes.py`.** `build_snapshot` reads `pulls/{n}` for the base SHA, the head SHA, and `changed_files`, then `pull_files` pages `pulls/{n}/files`. It fails over `--max-files` (`MAX_FILES = 3000`), and fails when the listing is shorter than `changed_files`. The listed paths, with each rename's `previous_filename`, become the document's `changed_paths`, from which `change_names` derives the related changes. `change_dirs` already lists the tree SHA of every directory under the changes directory and under `archive/`, at the base tip (`base.dirs`) and at the head (`head.dirs`). The base side feeds only `state_at(…, "base")`, the `at_base` field of `status --json`.
- **`spec-kit-workflow` `assets/spec_kit_features.py`.** It has the same `pull_files`, and lists only the head's specs directory. Its document records no base listing.
- **The git source** of both scripts takes the touched paths from `git diff --no-renames base...head`, which diffs the head against the merge base. It has no cap, and it is what `checks / spec` runs with the same `pull_request.base.sha` and `head.sha` that `pulls/{n}` reports.
- **Callers.** The GitHub workflow assets and this repository's workflows call `snapshot` with no cap flag, and read the document in the same job: `spec-labels.yml` builds it and runs `labels` on it, and `spec-command.yml` builds it and runs `show` or `status` on it. The change for issues #96 and #100 plans to read `head.sha` from the document in the command workflow.
- **Prose.** `SKILL.md` of each skill (`openspec-workflow` :156, `spec-kit-workflow` :125) and each `references/github.md` `## Fork safety` (openspec :63 and :81-84, spec-kit :49 and :61-64) say the snapshot pulls the file list and caps "files touched".

**Binding constraints:**
- **`skills/sdd/CONTEXT.md`.** Automation is fork-safe by construction: the head is read as data, never fetched, checked out, or run.
- **`.agents/knowledge/skill-quality.md` `## Management code`.** R5 applies to the assets: each interface is checked where its data enters and fails naming it, and no fallback stands in for a failure. Assets are Python 3.10 or later, standard library only.
- **The request budget.** `--max-calls` (200) bounds the REST reads one request can spend.
- **Self-containment.** Nothing outside the two skill directories changes in this change; the repository's copy is the companion's.

## Placement

| Requirement | File and section | Load trigger (references only) |
|---|---|---|
| OSW — MODIFIED `Script: assets/spec_changes.py` | `assets/spec_changes.py`: `Api` gains a read of the comparison `compare/{base}...{head}` (one page, one commit) that checks for `merge_base_commit.sha`; `pull_files`, `MAX_FILES`, and `--max-files` go; `build_snapshot` lists the changes directory and its archive at the base tip, the merge base, and the head, and records the touched directories; `SNAPSHOT_SCHEMA`, `SNAPSHOT_KEYS`, and `SnapshotSource` move to `spec-snapshot/2`; the source interface returns the touched change directories (the git source derives them from its diff, as now); module docstring and `--help` epilog | — |
| OSW — prose | `SKILL.md` `## Installing the automation` fork-safety paragraph; `references/github.md` `## Fork safety` (the snapshot sentence and the caps bullet) | existing load sentence |
| SKW — MODIFIED `Script: assets/spec_kit_features.py` | `assets/spec_kit_features.py`: the same, for numbered feature directories under the specs directory, with a new specs-directory listing at the merge base; `spec-kit-snapshot/2` | — |
| SKW — prose | `SKILL.md` fork-safety paragraph; `references/github.md` `## Fork safety` | existing load sentence |

## External impact

- **Companion `snapshot-tree-compare-harness`.** `scripts/spec_changes.py` gets the same snapshot rewrite. `.agents/knowledge/harness-maintenance.md` registers "the same subcommands, flags, and output" between the asset's CLI and this repository's script, so the flag removal and the schema bump land on both. It also rewords `.agents/knowledge/github-checks.md`. Proof: its verification plan.
- **Workflow assets unchanged.** No `assets/github/*.yml` passes `--max-files`. `contents: read`, which the comparison, trees, and blobs need, is already granted. Proof: `grep -n 'max-files' skills/sdd/*/assets/github/*.yml` is empty, and a read-through of the `permissions:` blocks.
- **Stale permission comment.** "reading the request and its files is a pull-request read" in both `assets/github/workflow-spec-command.yml` (and this repository's `spec-command.yml`) goes stale once no file list is read. `pull-requests: read` is still needed for `pulls/{n}`. The change for #96 and #100 rewrites that step and owns the comment's wording, so this change leaves it alone. If that change lands without the edit, this branch rewords the comment after rebasing. Proof: readback after the rebase.
- **No change to** descriptions, the `sdd` README pair rows, symlinks, `marketplace.json`, `skills/sdd/CONTEXT.md`, or the GitLab assets. Proof: `just validate`.

## Decisions

- **Always compare per-directory tree SHAs against the merge base, and drop the file list** (both MODIFIED requirements; maintainer decision, to confirm on the draft).
  - A directory's tree SHA differs between two commits exactly when some path under it differs, so the set of directories whose tree differs is the set the three-dot diff names. It covers renames, deletions, mode changes, and a directory replaced by a file, with no `previous_filename` handling. Verified locally for pull request #95: the directories under `openspec/changes/` and `archive/` whose tree SHA differs between the merge base and the head are exactly the two that `git diff base...head -- openspec/changes` names. The `archive` directory itself also differs, and it is excluded as `change_names` already excludes it.
  - Only `tree` entries count on either side, and a directory present on one side only is touched, which reproduces `change_names`' rule that a file directly under the changes directory or `archive/` belongs to no change.
  - Reads: `pulls/{n}`, the comparison, and about four tree reads for each of the base tip, the merge base, and the head. Today a large request spends up to 30 file pages.
  - Rejected: keep the file list and fall back to trees only when the listing is short. That means two code paths with different rename semantics, and the path used only for large requests is the one least exercised.
- **The merge base comes from the comparison's `merge_base_commit`** (the "edited only on the base branch" scenarios).
  - The comparison takes `BASE...HEAD` and its response schema lists `merge_base_commit` as required (GitHub REST docs, "Compare two commits", read 2026-09-30). Verified live and read-only on 2026-09-30:
    - `compare/10e29c0...<head of #93>?per_page=1` on this repository answered `diverged`, 88 behind, and a `merge_base_commit` equal to `git merge-base`.
    - On `cli/cli`, `compare/<trunk tip>...<fork head SHA>` for two fork pull requests (#14474, #14373) answered `diverged` with a `merge_base_commit` different from the base commit, and the fork head's tree resolved in the base repository.
  - The docs describe `basehead` as branch names, but the endpoint's own description allows commit SHAs, and SHAs worked in every live call. The scripts pass SHAs, as the gate does.
  - Rejected: `pull.base.sha`. It is the base tip, so a directory changed only on the base branch would look touched.
  - Rejected: `merge_commit_sha`, the test merge. It is null or stale while mergeability is unknown or conflicting, and its tree against the base tip misses a directory changed identically on both sides, which the three-dot diff reports.
  - Rejected: the parents of the first commit in `pulls/{n}/commits`. That is wrong once the base has been merged into the head, and the listing stops at 250 commits.
  - Rejected: a `git fetch` of the head to run `git merge-base`, which breaks the fork-safety rule.
- **The base tip is still listed, for `at_base`** (the "Snapshot source matches the git source" scenario). The git source evaluates the base side at the `--base` ref, the tip, so `status --json` stays byte-identical only if the snapshot keeps that listing. The merge base is used to find the touched directories and nothing else. The spec-kit document has no base side today and gains only the merge-base listing it needs.
- **Remove `--max-files` and bump the schema to `/2`** (both MODIFIED requirements; maintainer decision, to confirm on the draft).
  - No delivered workflow passes the flag, and a caller that does gets argparse's exit 2 naming it. A flag kept as a no-op would be a hidden fallback, which `## Management code` rules out.
  - The document drops `changed_paths` and records the touched directories. **`head.sha` keeps its key and meaning**, because the change for #96 and #100 reads it right after `snapshot`. A `/1` document is refused, naming the schema, with the instruction to rebuild it. Every caller builds and reads the snapshot in one job, so no reader of an old document exists.
  - Rejected: keeping `/1` with synthesized `changed_paths`. It would claim paths the snapshot never read and hide the change of contract.
- **MODIFIED, not ADDED** (both requirements; departs from the maintainer's stated preference, to confirm on the draft). The schema's skeleton allows one `Script:` requirement per script, named by its file, and the new behavior belongs to that script's contract. A `Behavior:` requirement takes agent outcome tasks as scenarios, not command runs. The sdd changes in flight prefer an ADDED requirement wherever one can stand alone; this one cannot, and the coordination risk below is the price.
  - Rejected: an ADDED `Behavior:` requirement carrying the new scenarios, which would split one script's contract across two requirements and misuse the Behavior kind.
- **Accept the comparison's cost on huge requests** (maintainer decision, to confirm on the draft).
  - The docs, read 2026-09-30, say of the comparison: "The list of changed files is only shown on the first page of results, and it includes up to 300 changed files for the entire comparison." The compare section carries no sentence about timeouts. The nearest documented one, "Larger diffs may time out and return a 5xx status code", belongs to "Get a commit" with the diff and patch media types. The scoping report's quote "Large responses may experience timeouts" was not found on 2026-09-30, and is treated as unverified.
  - Even so, the comparison computes a diff server-side and returns up to 300 files with their patches, so a 5xx on a very large comparison is plausible. The existing retry (three attempts on 429 and 5xx) applies, and then the job fails naming the endpoint. That is no worse than today's refusal.
  - Rejected: another merge-base source. None is documented, and the test merge is wrong for the reasons above.
- **The bundled `scripts/` stay out of scope** (non-goal; maintainer decision, to confirm on the draft). Their `snapshot` stops at 3000 files silently, which contradicts their own requirement. Following the `management-code` non-goal, that goes to a separate bug, which may propose removing their `snapshot` and `labels` subcommands, since no delivered workflow calls them.

- **The permission comment is left to the change for #96 and #100** (External impact; to confirm on the draft). Both changes would otherwise edit the same hunk of the same three workflow files.
  - Rejected: rewording the comment here as well, which makes two branches conflict on one hunk and lets whichever archives second overwrite the other's wording.

**To confirm on the draft.** Each is settled here and raised on the pull request:
1. Tree comparison against the merge base replaces the file list (Decisions, first bullet).
2. The merge base is the comparison's `merge_base_commit` (Decisions, second bullet).
3. `--max-files` goes and the documents move to `/2`, keeping `head.sha` (Decisions, fourth bullet).
4. MODIFIED rather than ADDED, against the stated preference for ADDED (Decisions, fifth bullet).
5. The comparison's timeout risk is accepted, with retries and then a failure naming the endpoint; the scoping report's timeout quote was not found in the docs (Decisions, sixth bullet).
6. The bundled `scripts/` stay out of scope, and their silent stop at 3000 files goes to a separate bug (Decisions, seventh bullet).
7. The stale permission comment is left to the change for #96 and #100 (Decisions, last bullet; External impact).
8. The merge order #99, then #96 and #100, then this change, with every MODIFIED block re-copied from main before the archive commit (Risks, first bullet).

## Risks / Trade-offs

- **[Another sdd change's clause is reverted at archive]** → The changes for #99, for #96 and #100, and this one all touch `openspec/specs/sdd/*/spec.md`, and #99 also MODIFIES `Script: assets/spec_changes.py`. A MODIFIED block replaces the main block whole at archive. The planned merge order is #99, then #96 and #100, then this change. After each preceding sdd pull request merges, this branch rebases on `main` and re-copies every MODIFIED requirement block from the then-current main spec, re-applying only this change's edits, before its archive commit, so that no archive silently reverts another change's clause. The re-copy is a record change pushed through the publish gate and noted on the draft.
- **[`head.sha` dropped or renamed in `/2`]** → The requirement now says the document records the head commit it read, and the harness reads `head.sha` from the new document.
- **[Criss-cross history: several merge bases, and GitHub and git pick different ones]** → Accepted as rare. The two sources could then disagree on a directory that differs between the candidate bases; the gate, which uses git, stays authoritative.
- **[A tree listing is truncated for an enormous changes or archive directory]** → The existing truncation check fails loudly. A non-recursive listing is far above realistic sizes.
- **[More tree reads per request: three sides instead of two]** → About four reads per side against a 200-request cap, well below the 30 file pages a large request spends today.
- **[Prose still promises a file cap]** → The edits listed under Placement, and the grep below.

## Verification plan

Written before implementation; results go to the pull request's Validation section.

No description, Behavior, or Handoff requirement changes, so there are no trigger or outcome cases. Every scenario is a `Script:` scenario and runs through the harness below. The harness runs in untracked scratch repositories and a stub server under the session scratch directory, with `bash` and `python3`, and is removed afterwards.

| Scenario | Case | Rubric and critical failures | Pass threshold | Solver tier | Observation | Isolation |
|---|---|---|---|---|---|---|
| OSW and SKW `Script:` scenarios | the harness commands below | the exit code and output each scenario states (all critical) | every command | none (command harness) | exit codes, stdout and stderr, written files | scratch repositories and a stub server outside the repository |

**Fixture.** A scratch git repository with `openspec init` run by the pinned CLI (for `spec_changes.py`) or with `specs/NNN-*` features (for `spec_kit_features.py`):
- `main` holds change A (active, one open task), change C, change D (active, every task ticked), and archived change B; for spec-kit, features 001, 002, and 003.
- The request branch leaves `main`. `main` then advances by editing C (feature 002), so that directory differs only on the base branch.
- The request edits A's task list (feature 001's), moves D under `archive/<date>-D` (deletes feature 003), and adds 3500 files outside `openspec/` (`specs/`).
- A variant request adds only files outside the changes directory (specs directory).

**Stub server**, reached through `--api-url`. It answers from git plumbing on the scratch repository: `pulls/{n}` (`base.sha` the `main` tip, `head.sha` the branch tip, `changed_files` the real count, over 3500), `compare/{b}...{h}` (`merge_base_commit` from `git merge-base`), `git/trees` (with `recursive`), and `git/blobs`. It logs each request and answers 404 to `pulls/{n}/files`.

**Per asset:**
- *Help:* `--help` exits 0 and names the five subcommands.
- *Representative run, Repeated run, Unknown change named / feature named, Missing plan:* the existing sequence from the `management-code` harness, re-run unchanged.
- *Bad arguments:* `--bogus` exits 2 naming it, and `snapshot --max-files 10` exits 2 naming `--max-files`.
- *Request past the file-listing limit; Snapshot source matches the git source:*
  - `snapshot` exits 0, and the stub's log shows no `pulls/{n}/files` request.
  - `status --json`, `status`, `show`, and `labels --json --current ''` on the document are byte-identical to the same subcommands with `--base <base.sha> --head <head.sha>`, the full SHAs the document records. `status --json` echoes the refs as given, so the git side passes those SHAs and not `main` and the branch name.
  - `head.sha` in the document equals the branch tip.
- *Change edited only on the base branch / Feature edited only on the base branch:* C (002) is absent from both outputs.
- *Change archived by the request:* D appears once, as `archived`. *Feature removed by the request:* 003 appears as `removed`. Each matches the git source.
- *Nothing under the changes (specs) directory touched:* on the variant, `status --json` lists nothing, and `labels --json --current <every managed label>` desires none and removes each.
- *Partial read refused:* the `--max-calls` cap reached, a tree answering `truncated: true`, and a non-UTF-8 document each exit 1 naming the path, with no `--out` file written.
- *Unexpected API response:* the comparison answering an object without `merge_base_commit`, a list, and 404 each fail naming the `compare` endpoint, with no `--out` file written. A 404 on a listed tree fails naming its endpoint.
- *Comparison timing out (the accepted risk):* the stub answers 502 to `compare/{b}...{h}` on every attempt. The stub's log shows exactly three `compare` requests, `snapshot` exits 1 with a message naming the `compare` endpoint and HTTP 502, and no `--out` file is written.
- *Snapshot of an earlier schema:* `status --snapshot` on a `/1` document exits 1 naming the schema and reports nothing.
- *Repeated run of the snapshot:* a second `snapshot` gives a byte-identical document.

**Live, read-only:**
- With a token, `snapshot` against a merged pull request of this repository that archived a change (for example #95), then `status`, matches `status --base <base.sha> --head <head.sha>` over a clone.
- The same against this change's draft pull request.

**Hygiene:**
- `git grep -n -i -w -e 'file list' -e 'files touched' -e 'max-files' -e 'MAX_FILES' -- 'skills/sdd/*/SKILL.md' 'skills/sdd/*/references/*' 'skills/sdd/*/assets/*'` is empty. The pathspecs leave out the bundled `scripts/`, which keep their cap. On main the same command lists the SKILL.md and `## Fork safety` sentences and both assets' cap code, so it is known to find them.
- `just check-skill skills/sdd/openspec-workflow skills/sdd/spec-kit-workflow`, `just lint`, `just spec-validate`, and `just check` pass.

**Skipped:**
- Behavioral subagent tests. No trigger, Behavior, or Handoff scenario changes, and the `SKILL.md` and reference edits describe the snapshot without changing what an agent does. The `Script:` scenarios are covered by the harness.
- A live pull request of more than 3000 files. Creating one is a remote write outside this change's authorization. The stub carries that case, and the live read-only runs cover the real endpoints.
- Live runs of the delivered workflows, which run from the default branch. After the merge, the next pull request of this repository that carries a `/spec` command is read back.

## Open Questions

None.
