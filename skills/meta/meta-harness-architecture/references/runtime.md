# Runtime Conventions

Read when depositing how agents run commands, checkpoint their work,
explore alternatives, or hand off. Each rule here lands in the target as a
setting, a recipe, or a sentence in the entrypoint or workflow document;
the conversation keeps none of it.

## Non-interactive execution

Set the environment so no agent process waits on standard input: pagers
off (`PAGER`, `GIT_PAGER`, and the platform CLI's pager variable set to
`cat`), the CI variable set where tools read it to skip prompts, watch
modes and interactive confirmations disabled by flag. Put the settings in
the container or shell profile the agent inherits, not in a document the
agent must remember.

## A fast scoped check beside the full one

Provide, in the entrypoint or the task runner, a seconds-scale command
that checks one file, one module, or the current diff — type check, lint,
and the relevant tests — beside the full check. Without it an agent
polishing a small change waits for the whole build and stops verifying.

## Checkpoints are commits

- A checkpoint is a commit. A stash is addressed by stack position, may
  conflict on restore, has no name, and is invisible in the log; a
  framework's own rewind lives only in the session and is invisible to
  humans. A commit is addressable, named, and inspectable by everyone with
  `git log`.
- A task-level commit is an atomic commit. Define "a verified sub-step" as
  the completion of one task of the specification; each checkpoint is then
  one atomic commit under the project's message convention, with nothing to
  tidy afterwards.
- Draft commits inside a task are squashed into that task's commit when the
  task ends. The unit of squashing is the task, not the change request: a
  request with three tasks lands as three commits. Do not postpone the
  squash to the request's ready step — by then the context no longer knows
  what each draft was. Interactive rebase is unavailable to agents, so
  name the non-interactive sequence: reset softly to the task's start
  commit, then make one commit.
- A task that is not atomic is split at the plan, not at the diff.
  Discovering midway that one task holds two independent changes means
  splitting the task and committing each; staging hunks is error-prone for
  agents, and the signal belongs to the plan, not to commit-time tidying.
- Draft commits never leave the machine. Push only after the task-level
  commit, and push one named ref; batch pushes of all branches, mirrors,
  or tags are denied. Only unpushed history is rewritten; pushed history
  is reverted. A draft that needs remote CI to verify is a publication and
  passes the same gate as any push; unattended runs prefer the local
  scoped check.
- Squashing changes no content. A debug credential file still in the tree
  at task end enters the squashed commit unchanged, so the secret scan runs
  on every checkpoint commit, not only before publication. This is also
  why the pre-commit hook must stay seconds-fast: a slow hook gets
  bypassed, and the secret scan is the one hook that must not be.
- "No draft commits on the branch" is part of each task's done criteria,
  so the squash does not depend on the agent remembering.
- Rollback and the deny list do not conflict. The deny list blocks hard
  resets and discarding the working tree to protect uncommitted work;
  checkpoints exist so the tree is always committed. The precondition for
  a rollback (clean tree, or a checkpoint just made) is enforced by a
  mechanism — a recipe, an alias, or a framework feature the project
  chooses — while the deny list keeps blocking the raw command.

## Exploring alternatives

Encourage branches or worktrees for a doubtful design: build and run each
candidate, let compilation and tests expose what the design stage could
not, keep the best, and roll back the rest. Cleanup is part of the
procedure: the temporary worktrees and experiment branches are removed as
soon as the choice is made.

## Progress and handoff hygiene

Where a specification tool exists, reuse its task tracking; progress notes
exist to expose state to humans early, not as a diary that spends output
tokens. A handoff or scratch note carries debug logs, environment values,
and credentials more easily than source, so it is written only to a
location git does not track — the system temporary directory, or a handoff
directory listed in the ignore file — and treated as a secret. Session logs
are the same class: they never enter the repository or a shared drive, and
any analysis of them happens on the machine that produced them or after
redaction.
