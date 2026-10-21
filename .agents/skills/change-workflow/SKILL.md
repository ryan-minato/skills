---
name: change-workflow
description: Runs this repository's change lifecycle on GitHub — the OpenSpec change loop, the draft pull request opened at the change record, approval on the draft, atomic commits, the publish gate, and review admission under the H1 authority policy. Use when tracked files are about to change; when starting or resuming repository work; when taking or creating an issue; when preparing commits, pushing, opening or readying a PR; or when diagnosing a red check. Read-only analysis does not trigger it.
metadata:
  internal: true
---

# Change Workflow

Follow this for every tracked-file change. It orders the steps and holds
their mechanics; each rule it follows has one owner, named where the step
needs it:

- `.agents/knowledge/spec-workflow.md` — the lifecycle, the approval
  gates, the archive executor and the freeze.
- `.agents/knowledge/agent-authority.md` — publishing, review admission,
  pending decisions, what is never yours.
- `.agents/knowledge/github-workflow.md` — branches, merging, issues,
  labels, triage.

When a check, policy, or permission blocks you, follow
`agent-authority.md ## No self-escalation`.

## 1. Gate tooling and authorization

```bash
command -v gh && gh --version && gh auth status
command -v openspec && openspec --version
```

If `gh` is missing, stop and direct the user to <https://cli.github.com/>;
if authentication fails, ask the user to run `gh auth login` — never start
it yourself. If `openspec` is missing, run `just setup`.

List the remote writes the task may need — issue, push, PR, ready, review
request, labels — and obtain the authorization
`agent-authority.md ## Publishing` requires for them.

Done when: `gh auth status` succeeds and the allowed remote operations are
recorded, or the workflow has stopped with exact setup or authorization
needs.

## 2. Take or create the issue

Derive `OWNER/REPO` from `git remote get-url origin`. An issue is optional
(`github-workflow.md`, Decomposition):

- **Taking one**: read it, confirm it is open and nobody else is assigned,
  and confirm its acceptance — the scenarios of the change its
  Specification field names, or executable criteria. An issue with no
  change yet is taken by proposing the change (step 3); an issue whose
  change is still under discussion waits for the maintainer's closing.
  Assign yourself, re-read, confirm you are the sole assignee.
- **Creating one**: only when authorized. Build the body by mirroring the
  form's `### <label>` headings in `.github/ISSUE_TEMPLATE/`
  (non-interactive creation ignores forms) and apply the form's labels
  explicitly; `issues / triage` derives `priority/*` and `catalog/*`.
  Issues for planned work derive from the change's `tasks.md`, each
  naming the scenarios it closes (`spec-workflow.md ## Specifications
  and tracked work`).

Record the issue number or a one-line reason there is none.

## 3. Establish the branch, the change, and its approval

1. Confirm the worktree is clean enough to isolate the requested change;
   fetch `origin` when remote reads are allowed.
2. Create the branch from `origin/main` as
   `github-workflow.md ## Branches and merging` names it.
3. Write the approval package on that branch with the `openspec-*`
   skills. A skill change: propose (`openspec/changes/<slug>/`), clarify
   with `plan-clarification`, write the design the schema asks for. A
   repository change: `openspec new change <slug>`, then
   `skip_specs: true` in its `.openspec.yaml`, then the proposal and the
   design. Run `just spec-validate` and commit the package. The
   `spec-driven-development` skill supplies the loop's general rules;
   `spec-workflow.md` overrides it here.
4. Run the publish gate (step 6) and open the draft pull request with the
   package as its first content, at the point
   `spec-workflow.md ## Lifecycle` sets. Then stop.
5. Push each record change the maintainer directs in conversation through
   the publish gate. When asking for the package approval, ask the
   maintainer to add `spec/approved`, then push nothing to the package and
   poll `just spec-approval <n>` until the record appears
   (`spec-workflow.md ## Approval gates`, including the bootstrap).
6. Reconcile each closing — this one and the implementation closing in
   step 7:
   - read the comments with `gh pr view <n> --json comments` and the
     review threads with `just review-threads <n>` (field `isResolved`);
   - list every unresolved thread, every requested adjustment the record
     does not carry, and every contradiction;
   - ask the maintainer to confirm the open items; proceed only when
     nothing is open or the open items are confirmed;
   - update the `Approval:` line, and after the package closing set
     `Phase: implementation`, through the publish gate.
7. Write `tasks.md`, run `just spec-validate`, commit, then implement.

A change too small to plan skips the loop with `Spec: none — <reason>`
(`spec-workflow.md ## Specifications and tracked work`).

## 4. Implement in atomic commits

Work task by task from `tasks.md`. Each commit is the smallest
independently valid logical change:

1. Group files that must land together for checks to pass; split what can
   stand alone.
2. Stage one group; run the applicable checks (`just check-skill <dir>` for
   a skill, `just spec-validate` after touching `openspec/`).
3. Commit. The `commit-safety` pre-commit hook checks the author email
   and staged credentials; the `git-commit` skill's gates and `AGENTS.md`
   Commits govern the message.

A deviation from the spec stops the work (`spec-workflow.md
## Lifecycle`).

Checkpoints inside a task are commits, never stashes: commit drafts as
you go, and before ticking the task squash them into its one atomic
commit non-interactively — `git reset --soft <task start commit>`, then
one commit under the rules above (interactive rebase is unavailable).
Draft commits never leave the machine: push only after the task commit,
as `agent-authority.md ## Publishing` says. The per-commit hooks — secret
scans, validators, lint — run on every draft, so every draft commit must
pass them; keep them fast. A task is done only when no draft commit
remains on the branch.

Never bypass hooks (`--no-verify`), force a failing commit, or add
tool-attribution trailers; author with a GitHub noreply email.
`pr / policy` re-checks all three over the pull request's commits.

## 5. Verify the scenarios

Before marking the PR ready, execute every scenario of the change against
the result, not the diff: for a skill, the behavioral tests in the
`skill-authoring` project skill, planned in the change's `design.md`; for
a repository change, the proofs its `design.md` names. Tick each task as
`spec-workflow.md ## Lifecycle` says, then run `just check`. Leave the
change unarchived (`spec-workflow.md ## Archive executor and the
freeze`); until it is, `checks / archived` stays unreported and nothing
is red. `just spec-check` reproduces `checks / spec` locally;
`just spec-changes status` shows the related changes' progress. Record
what ran and the outcome for the PR's Validation section.

## 6. Publish gate, then the draft PR and every later push

GitHub publication cannot be undone: bodies, comments, commit messages,
and their notification copies survive deletion, and public content is
indexed within minutes. Every remote write passes this gate
(`agent-authority.md ## Publishing`).

1. Assemble the exact payload as files in a scratch directory outside the
   repository. For a pull request: `title.txt`; `body.md` built from
   `.github/PULL_REQUEST_TEMPLATE.md`, whose comments say what each field
   holds and when; `commits.txt` from
   `git log origin/main..HEAD --format=full`; `diff.patch` from
   `git diff origin/main...HEAD`. For an issue or comment: `title.txt` and
   `body.md`. For a later push: the new commits and diff.
2. Review it independently: dispatch a clean-context subagent whose whole
   prompt is the review prompt in
   [references/publish-review.md](references/publish-review.md) with the
   directory path filled in; without subagent support, re-read every file
   from disk and write `Review mode: file-only (not clean-context)` above
   the verdict.
3. Continue only with a verbatim `SAFE TO PUBLISH: YES`. On NO, fix every
   finding, rebuild the directory, review again; a secret that reached
   GitHub is compromised even after deletion — rotate it.
4. Confirm the authorization covers this write, push one named ref
   without force, open the draft PR non-interactively, then read it back
   and compare with the reviewed payload.

Keep the draft's body current as evidence changes; each edit passes this
gate.

## 7. Review admission, the second deliberation, and the freeze

Check every condition of `agent-authority.md ## What agents may do`,
reading the approval with `just spec-approval <n>`. For a red check,
follow `github-checks.md ## Diagnosing a red check`; digest a failed run
instead of reading full logs:

```bash
./.agents/skills/change-workflow/scripts/run_log_digest.ts --repo ryan-minato/skills --run-id <id>
```

Then:

1. Complete the final description as the template's comments direct:
   Changes and Validation filled, `Phase: implementation`, the approval
   line naming the implementation deliberation, the checklist ticked.
2. `gh pr ready <number>`. When the pull request was opened from another
   account, also `gh pr edit <number> --add-reviewer ryan-minato`; GitHub
   silently drops a review request for the author, so on a pull request
   from the maintainer's own account the ready state is the request.
3. Hand over the report in `agent-authority.md ## Escalation`.
4. Wait. When the maintainer closes the implementation deliberation in
   conversation, reconcile it (step 3.6), then freeze with the executor
   in `spec-workflow.md ## Archive executor and the freeze`:

   ```bash
   just spec-changes archive --base origin/main --head HEAD
   git add openspec && git commit -m "docs: archive the <slug> change"
   ```

   Push it through the publish gate and update the approval line with the
   closing's date and the archive commit. The maintainer merges.

If any condition fails, stop at the draft with the same report and name
the failing condition.

## Gotchas

- A branch, issue, or PR does not authorize unrelated cleanup; report
  discovered work separately (`AGENTS.md` Mismatches).
- Any command naming `spec/approved` is denied, reads included
  (`agent-authority.md ## Permission entries`); read the approval with
  `just spec-approval <n>`.
