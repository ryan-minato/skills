# Splitting Commits and Recovering From Failures

Read this when the atomicity gate fails (the staged diff mixes concerns) or when
a hook or commit fails partway through the workflow.

## Splitting a mixed change set

Group by concern, not by file type. A feature and its tests belong together; two
unrelated bug fixes do not, even when they touch the same file.

1. List the changes and propose groups, one line of rationale each:

   ```
   Group 1 — fix: null check in session refresh (src/auth/session.ts)
   Group 2 — docs: correct env var name (README.md)
   ```

2. Confirm the grouping with the user before touching the index.
3. Unstage everything so staging starts clean: `git reset`.
4. For each group, in order:
   - Stage whole files with `git add <paths>`.
   - When one file contains hunks from different groups, stage interactively:
     `git add -p <file>` — `y` stages the hunk, `n` skips it, `s` splits a hunk
     that is still too coarse, `q` stops. When hunks are too entangled even for
     `s`, use `git add -e <file>` and delete the unwanted `+` lines from the
     patch (do not touch context lines).
   - Verify the staged set matches the group: `git diff --cached --stat`, then
     `git diff --cached` for the hunk-level view.
   - Run the message and validation steps from SKILL.md, then commit.
5. After the last commit, `git status` must show only intentionally uncommitted
   files. Anything unexpected belongs to a group that was missed.

To test whether a staged subset builds or passes checks on its own, run
`git stash push --keep-index` (stashes only the unstaged remainder), run the
checks, then `git stash pop`. Resolve any pop conflicts immediately — they mean
the staged and unstaged halves overlap and the grouping needs revisiting.

## Recovering from failures

**A pre-commit hook rewrote files (formatters, import sorters).** The working
tree now differs from the index. Re-stage the same paths (`git add <paths>`) and
run `git commit` again with the same message. During a split, confirm the rewrite
did not pull in hunks that belong to a later group (`git diff --cached`).

**The commit was created with the wrong content or grouping.** If it is the most
recent commit and not yet pushed: `git reset --soft HEAD~1` returns the changes
to the index for restaging. Prefer creating a corrective commit for anything
already pushed.

**Wrong message on the most recent, unpushed commit.**
`git commit --amend` (add `--no-edit` to keep the message while amending
content). Avoid amending pushed commits unless the user asks — it rewrites
shared history.

**A check fails against the staged subset but passes against the full change.**
The grouping split a dependency (e.g. the fix in group 1 relies on a helper in
group 2). Reorder the groups or merge them — a commit sequence where
intermediate states fail checks defeats the point of splitting.

**Merge conflict from `git stash pop` during a split.** The staged and unstaged
halves touched the same lines. Abort the current grouping (`git checkout --merge`
the conflicted files or resolve by hand), then regroup with a cleaner boundary.
