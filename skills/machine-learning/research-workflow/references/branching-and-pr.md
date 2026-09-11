# Branching, Snapshots, and the Closing Request

Read when opening the research worktree or branch, making snapshot
commits, or preparing the pull or merge request that closes the task.

## The branch or worktree

- One task, one branch (or one worktree on that branch), named as the
  project's branching convention says; never the integration branch.
- The draft pull or merge request opens when the spec is written, so the
  task is visible and the spec can be discussed where the project
  discusses changes; the request stays a draft until the task closes.

## Snapshot commits

- Before every run, commit the tree as it will execute. The commit
  message names the hypothesis id and what changed ("h3: gated FFN,
  width 2048"); the run records the commit.
- A snapshot commit is not a reviewable unit and need not pass every
  check; it is an identity. The project's hooks stay light for this
  reason.
- Never amend or rebase a snapshot a run has already cited.

## Squashing and reachability

The integration branch keeps the code the verdict needs, arranged as the
project prefers (squash, cherry-pick, or a rewritten history). Before the
research branch is deleted, make every cited snapshot reachable by the
project's retention rule — a tag per run (`run/<run_id>`), a
`research/<task>` branch kept intact, or an archive ref — and check that
the platform's branch cleanup will not remove it.

## The closing request

Fill the request's description with:

1. the objective and the evaluation (or a link to the spec);
2. the verdict: what was found, positive, negative, or inconclusive;
3. the decisive evidence — run ids or tracker links, the comparison, the
   artifacts;
4. what was ruled out, from the hypothesis log;
5. what remains open, if anything, as the seed of a later task;
6. the code changes that land, and why each is needed by the verdict.

Metrics tables belong in the tracker; the request links to them. A
request that restates every run is unreadable and goes stale.

## Archiving the spec

Follow the project's specification contract when it has one (a
research-task change is archived like any other, on completion, whatever
the verdict). Without a contract, the spec and the hypothesis log stay
under `research/<task>/` with a one-line status at the top.
