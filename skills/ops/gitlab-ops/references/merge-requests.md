# Merge requests

Loaded when the task operates on merge requests. Use the column chosen in
"Choose your path"; `G/P` comes from "Identify the host and project".

Surface exposure: creating an MR publishes every commit message and the
complete diff of `TARGET...SOURCE`, not just the description. For MR
creation, comments, and discussion replies, the pre-publish gate runs
file-based:

1. Write the exact outgoing content to a scratch directory: title, body,
   each comment or review body, `git log TARGET..SOURCE --format=full >
   commits.txt`, `git diff TARGET...SOURCE > diff.patch`, and any
   attachments.
2. Run the review procedure in
   [publish-review-mr.md](publish-review-mr.md) over that directory. Read
   that file every time — do not review from memory.
3. Publish only after the verdict is exactly `SAFE TO PUBLISH: YES`. On
   `NO`, fix every finding, rebuild the files from the fixed content, and
   review again.

Pure reads, draft/ready flips, approvals, and merges of already-reviewed
MRs carry no new content and skip the gate.

## Operations

Multi-line text through files: `-d "$(cat BODY.md)"` /
`-m "$(cat COMMENT.md)"`. Never `--fill`.

| Task | glab command | MCP capability (min GitLab) |
|---|---|---|
| Create MR | `glab mr create -R G/P -s BRANCH -b TARGET -t "TITLE" -d "$(cat BODY.md)" [--draft] -y` | create an MR (18.5) |
| Comment | `glab mr note N -R G/P -m "$(cat COMMENT.md)"` | comment on an MR (19.2) |
| Read MR | `glab mr view N -R G/P` (`-F json` for fields) | read MR details (18.4) |
| Read comments/threads | `glab mr view N -R G/P --comments` (`--unresolved` filters) | read MR comments and threads (19.2) |
| Read diff | `glab mr diff N -R G/P` | read MR diffs (18.4) |
| Read commits | `glab api "projects/:fullpath/merge_requests/N/commits"` | read MR commits (18.4) |
| Pipeline status | `glab ci get --merge-request N -R G/P` | read an MR's pipelines (18.4) |
| Who approved | `glab mr approvers N -R G/P` | — |
| List / filter MRs | `glab mr list -R G/P [--merged] [--search "TEXT"]` | — |
| Close / reopen | `glab mr close N -R G/P` / `glab mr reopen N -R G/P` | — |
| Draft → ready | `glab mr update N -R G/P --ready` | — |
| Edit labels / milestone | `glab mr update N -R G/P [-l NEW] [-u OLD] [-m "TITLE"]` | — |
| Approve / revoke approval | `glab mr approve N -R G/P` / `glab mr revoke N -R G/P` | — |
| Merge | `glab mr merge N -R G/P --squash --yes` | — |

After any create or comment, report the returned URL to the user.
Done when: the URL is reported.

**Merge semantics.** `glab mr merge` has `--auto-merge` on by default:
with a running pipeline the command sets the MR to merge when checks pass
and returns immediately — report that as "set to auto-merge", not
"merged". Pass `--auto-merge=false` only when the user wants an immediate
merge regardless of pipeline state. `--squash` follows this skill's
default of linear history; the merge-settings conventions row tells you
when the project says otherwise.

On the REST fallback tier, the read rows map to `rest_read.py mrs` and
`mr` (with `--diff`, `--commits`, `--notes`, `--approvals`,
`--pipelines`) — see [rest-fallback.md](rest-fallback.md).

- Read [pipelines.md](pipelines.md) when a pipeline on the MR is failing
  and you need to see why.
- Read [discussions-and-approvals.md](discussions-and-approvals.md) when
  replying to or resolving a discussion thread, commenting on a diff
  line, or working with approval state.
- Read [mr-recipes.md](mr-recipes.md) when the needed MR operation is not
  in the table above (rebase, checkout, target-branch change, reviewers,
  linked issues, squash/remove-source options).

## Gotchas

- Draft state is a `Draft:` title prefix under the hood — manage it with
  `--draft` (create/update) and `--ready` (update), never by editing the
  title yourself.
- `glab mr note` subcommands (`create`, `resolve`, ...) are marked
  experimental and may change between glab versions; the bare
  `glab mr note N -m` comment form is stable. If a subcommand is missing,
  run `glab mr note --help` and use the `glab api` fallback in
  [discussions-and-approvals.md](discussions-and-approvals.md).
- `glab mr view` and `glab mr list` both use `-F/--output json`, unlike
  `glab issue list` (`-O`) — check `--help` before assuming output flags.
- On `glab mr update`, label/assignee flags behave like the issue ones:
  unknown `-l NAME` silently creates a label.
- Approving requires approvals to be enabled on the project; *required*
  approval rules are Premium, but the approve/revoke buttons exist on
  Free. A 404 on approval endpoints usually means a tier gate, not a
  wrong path.
- Merging is blocked while the MR is draft, discussions are unresolved
  (when the project requires resolution), or a required pipeline is red
  — the error names the blocker; fix that instead of retrying.
