---
name: change-workflow
description: Runs this repository's change lifecycle on GitHub — the OpenSpec change loop, the draft pull request opened at the change record, approval on the draft, atomic commits, the publish gate, and review admission under the H1 authority policy. Use when tracked files are about to change; when starting or resuming repository work; when taking or creating an issue; when preparing commits, pushing, opening or readying a PR; or when diagnosing a red check. Read-only analysis does not trigger it.
metadata:
  internal: true
---

# Change Workflow

Follow this for every tracked-file change. The rules it executes live in
three knowledge files; this skill orders them and adds the mechanics:

- `.agents/knowledge/github-workflow.md` — branches, merge method, labels,
  issues, milestones, triage.
- `.agents/knowledge/spec-workflow.md` — when a change needs an OpenSpec
  change, its lifecycle, and who approves it.
- `.agents/knowledge/agent-authority.md` — what you may do on your own
  (H1) and when to stop.

Never edit those files, a workflow, the ruleset, or a required check to
unblock your own change.

## 1. Gate tooling and authorization

```bash
command -v gh && gh --version && gh auth status
command -v openspec && openspec --version
```

If `gh` is missing, stop and direct the user to <https://cli.github.com/>;
if authentication fails, ask the user to run `gh auth login` — never start
it yourself. If `openspec` is missing, run `just setup`.

List the remote writes the task may need — issue, push, PR, ready, review
request, labels — and continue only with the user's explicit authorization
for them in the current conversation. One clear authorization covers the
listed operations; an explicit local-only instruction permits branch and
commit work and nothing remote. Absent or ambiguous authorization: do local
work only and say what is waiting.

Done when: `gh auth status` succeeds and the allowed remote operations are
recorded, or the workflow has stopped with exact setup or authorization
needs.

## 2. Take or create the issue

Derive `OWNER/REPO` from `git remote get-url origin`. An issue is optional
(`github-workflow.md`, Decomposition):

- **Taking one**: read it, confirm it is open and nobody else is assigned,
  and confirm its acceptance — the scenarios of the change its
  Specification field names, or executable criteria. An issue with no
  change yet is taken by proposing the change on the draft pull request
  (step 3); an issue whose change exists but is unapproved waits for the
  approval comment. Assign yourself, re-read, confirm you are the sole
  assignee.
- **Creating one**: only when the user authorized it. Build the body by
  mirroring the form's `### <label>` headings in
  `.github/ISSUE_TEMPLATE/` (non-interactive creation ignores forms) and
  apply the form's labels explicitly; the `issues / triage` check derives
  `priority/*` and `catalog/*`. Issues for planned work derive from the
  change's `tasks.md`, each naming the scenarios it closes; never copy
  acceptance criteria into an issue. Milestones and `goal` tracking issues
  are created only after the maintainer confirms them.

Record the issue number or a one-line reason there is none.

## 3. Establish the branch and the change

1. Confirm the worktree is clean enough to isolate the requested change;
   fetch `origin` when remote reads are allowed.
2. Choose the slug. Create the branch `<type>/<slug>` (or
   `<type>/<issue>-<slug>`) from `origin/main`; never work on `main`.
3. If the change alters what a skill does, run the OpenSpec loop on that
   branch with the `openspec-*` skills: propose (the change is
   `openspec/changes/<slug>/`), clarify with the `plan-clarification`
   skill until no assumption is silent, commit the change record, then
   publish at once — run the publish gate (step 6) and open the draft pull
   request with the record as its first content and `Phase:
   specification` in the body, before any design or task list is
   finished. Request the maintainer's review and wait for the approval
   comment naming the commit (`Specification approved at <sha>`); it
   reviews the proposal and the delta specs, never `design.md` or
   `tasks.md`. Finish design and tasks after the approval, then implement.
   `spec-driven-development` supplies the loop's rules; `spec-workflow.md`
   says which domains exist and that specs are never backfilled.
4. A change to the repository itself (environment, harness, tooling,
   checks, workflows, documents) runs the same loop as a repository change:
   `openspec new change` then `skip_specs: true` in its `.openspec.yaml`,
   so it carries a proposal, design, and tasks and no delta spec. Only a
   change too small to plan (a pin bump, a typo) skips the loop, and its PR
   says `Spec: none — <reason>`.

## 4. Implement in atomic commits

Work task by task from `tasks.md`. Each commit is the smallest
independently valid logical change:

1. Group files that must land together for checks to pass; split what can
   stand alone.
2. Stage one group; run the applicable checks (`just check-skill <dir>` for
   a skill, `just spec-validate` after touching `openspec/`).
3. Commit. The `commit-safety` pre-commit hook checks the author email,
   staged credentials, and the harness layers touched; the `git-commit`
   skill's gates and `AGENTS.md` Commits govern the message. When a
   deviation from the spec appears, stop, revise the change with the update
   skill, and get it re-approved before continuing.

Never bypass hooks, force a failing commit, or add tool-attribution
trailers. Use a GitHub noreply author email unless the user explicitly
approves a private one.

## 5. Verify the scenarios

Before marking the PR ready, execute every scenario of the change against
the result, not the diff: for a skill, the behavioral tests in the
`skill-authoring` project skill, planned in the change's `design.md` and
derived from the scenarios; for a repository change, the proofs its
`design.md` names. Tick each task only when its verification ran. Then
run `just check`. Archiving is the `spec-archive` workflow's job after the
merge; while that workflow cannot push (`spec-workflow.md`, Archive mode),
archive the change here with the archive skill so the delta lands in
`openspec/specs/`. Record what ran and the outcome for the PR's Validation
section, linking the plan in `design.md` rather than restating it.

## 6. Publish gate, then the draft PR and every later push

GitHub publication cannot be undone: bodies, comments, commit messages,
and their notification copies survive deletion, and public content is
indexed within minutes. Every remote or publishable write passes this
gate, and any edit after the verdict needs a fresh one.

1. Assemble the exact payload as files in a scratch directory outside the
   repository. For a pull request: `title.txt`, `body.md` (from
   `.github/PULL_REQUEST_TEMPLATE.md`, with `Closes #N` or `N/A — <reason>`,
   the `Spec:` line as a link to the change directory on this branch —
   `[openspec/changes/<slug>](https://github.com/ryan-minato/skills/tree/<branch>/openspec/changes/<slug>)`
   — and the `Phase:` line), `commits.txt` from
   `git log origin/main..HEAD --format=full`, and `diff.patch` from
   `git diff origin/main...HEAD`. For an issue or comment: `title.txt` and
   `body.md`. For a later push, the new commits and diff.
2. Review it independently: dispatch a clean-context subagent whose whole
   prompt is the review prompt in
   [references/publish-review.md](references/publish-review.md) with the
   directory path filled in; without subagent support, re-read every file
   from disk and write `Review mode: file-only (not clean-context)` above
   the verdict. Judge only what the files contain.
3. Continue only with a verbatim `SAFE TO PUBLISH: YES`. On NO, fix every
   finding, rebuild the directory, review again; a secret that reached
   GitHub is compromised even after deletion — rotate it.
4. Confirm the user's authorization covers this write, push without force,
   open the draft PR non-interactively, then read it back and compare with
   the reviewed payload.

The draft is the public ownership signal and the review surface for the
specification; keep its body current as evidence changes, and switch
`Phase:` to `implementation` once the approval comment exists. Do not
publish secrets, private data, or internal context on any surface.

## 7. Review admission (H1)

Mark the PR ready and request the maintainer's review yourself only when
every condition in `agent-authority.md` holds: change approved on the
draft, every task done, archived or left for the `spec-archive` workflow
(or `Spec: none` justified), `just check` green locally and every
required check green (`.agents/knowledge/github-checks.md`; digest a red
run with
`python3 .agents/skills/change-workflow/scripts/run_log_digest.py --repo ryan-minato/skills --run-id <id>`
rather than reading full logs), scenarios verified and recorded, publish
gate passed for the final body, remote writes authorized. Then:

1. Complete the checklist and update the final description.
2. `gh pr ready <number>`. When the pull request was opened from another
   account, also `gh pr edit <number> --add-reviewer ryan-minato`; GitHub
   silently drops a review request for the author, so on a pull request
   from the maintainer's own account the ready state is the request.
3. Hand over the acceptance-evidence report from `agent-authority.md`:
   goal, tests and check state, actual scope, risks, and the maintainer's
   decisions — request fixes, reject, or merge (rebase; squash for forks).

If any condition fails, stop at the draft with the same report and name
the failing condition. Never approve, merge, or arm auto-merge; merge
closes the issue through the closing keyword.

## Gotchas

- Authorization is task-scoped; a successful `gh auth status` is not
  permission to publish.
- Archiving runs after merge; never tick a task that is not done, because
  a fully ticked task list is what the workflow archives. While the
  workflow cannot push, archive before ready.
- The draft exists to review the specification: opening it after
  implementation turns the approval gate into a formality.
- A branch, issue, or PR does not authorize unrelated cleanup; report
  discovered work separately.
- When ending with work unfinished, leave a self-contained handoff in the
  PR or conversation: branch, change, completed work, remaining work,
  current checks.
- Fork pull requests are squash-merged and their title is what the
  `pr / policy` check validates; in-repo branches are validated commit by
  commit.
