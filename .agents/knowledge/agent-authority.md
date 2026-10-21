# Agent Authority Policy

Read this before any remote write, before marking a pull request ready,
requesting review, approving, merging, or changing any repository
setting, and whenever it is unclear whether an action is yours to take.

Level: **H1** — the maintainer decided on 2026-09-03 that an agent may
admit its own work to review; integration stays human. Outside the scope
below, the level is H0: the agent stops at a draft pull request.

Two gates: **review admission** (draft to ready, review requested) is the
agent's, under the conditions below; **integration** (the change enters
`main`, engineering responsibility transfers) is the maintainer's
(`ryan-minato`), by merging.

## What agents may do

Take issues; plan, implement, and test; self-review; create atomic
commits; push an authorized branch; open and update a draft pull request;
run `gh pr ready` and request review from the maintainer; respond to
review; prepare the acceptance-evidence report.

`gh pr ready` is permitted only when all of these hold:

- the change's package deliberation is closed and reconciled
  (`spec-workflow.md ## Approval gates`; read the approval with
  `just spec-approval <n>`) and every task in its `tasks.md` is done; or
  the pull request carries a justified `Spec: none — <reason>`. The
  change is not archived yet (`spec-workflow.md ## Lifecycle`);
- `just check` passes locally and every required check is green, except
  that `checks / spec`, and through it `checks / gate`, is red until the
  freeze (`github-checks.md`);
- every scenario of the change passed, with the evidence in the pull
  request's Validation section;
- the publish gate returned `SAFE TO PUBLISH: YES` for the exact pull
  request payload, and the user authorized the write (`## Publishing`).

## What agents may not do

- Approve a pull request, merge, or arm auto-merge.
- Add or remove the `spec/approved` label, or edit or delete its record
  comment: the label stands for the maintainer's closing, and agents act
  through the maintainer's `gh` login, so only this rule tells the two
  apart. Read it with `just spec-approval <n>`.
- Create milestones or tracking issues without a human confirming them.
- Change a ruleset, a required check, a repository setting, this file,
  `AGENTS.md`, or `.claude/settings.json` in a way that widens agent
  authority.
- Mark a pull request ready while a required check is failing,
  unavailable, or unreliable for any reason other than the unarchived
  change `checks / spec` (and through it `checks / gate`) reports before
  the freeze.
- Execute a task whose change is missing or unapproved.

## Publishing

Nothing reaches GitHub — push, pull request, issue, comment, label, review
request, setting — without both:

- **Authorization**: the user's explicit authorization in the current
  conversation, covering that write. It is task-scoped: one clear
  authorization covers the operations it lists; an explicit local-only
  instruction permits branch and commit work and nothing remote; a
  successful `gh auth status` permits nothing. Absent or ambiguous
  authorization: do local work only and say what is waiting. Authority
  under this policy never substitutes for it.
- **Publish gate**: a `SAFE TO PUBLISH: YES` verdict from an independent
  review of the exact payload, published byte-identical; any edit after
  the verdict needs a fresh review.

A push names one ref (`git push origin <branch>`), never force, `--all`,
`--mirror`, or `--tags`, and goes only to the branch the change's
proposal names. The procedure — payload directory, review prompt, read
back — is the `change-workflow` skill, §1 and §6.

## Escalation

Stop and hand to the maintainer when:

- The scope grows beyond the issue or change that authorized it.
- Acceptance cannot be evaluated: no approved change, ambiguous scenarios,
  or a scenario that cannot be executed.
- A required check is unavailable, unreliable, or contradicts other
  evidence.
- The change unexpectedly touches a published skill's behavior that no
  change record covers, `scripts/check_commit_safety.ts`, secret
  scanning, or any workflow's permissions.
- Reverting the merge would not restore the previous installed behavior.
- Review evidence conflicts, or confidence is insufficient for a decision
  this policy authorizes.
- A structural decision appears that the change's `design.md` does not
  settle.

Escalating earlier is always allowed. Bypassing a closed gate never is.

At a human gate, hand over: the goal and how the change addresses it (the
Specification section links the record); tests executed and results (the
Validation section) and the state of every required check; the scope
actually touched; known risks and remaining limitations; the
maintainer's decisions — request fixes, reject, or merge
(`github-workflow.md ## Branches and merging`).

## Pending decisions

Blocked work does not stop the session: finish everything that does not
depend on the decision, then record each pending decision in the handoff
with

- where it is stuck;
- the options;
- the recommended option and why;
- the completed work each option affects.

The handoff lives in the pull request or the conversation: branch,
change, completed work, remaining work, current checks, pending
decisions. A handoff file lives only in the scratch directory, never in
the tree. An outward write that the proposal's
`## Deliverable and pre-authorized actions` does not name is queued here
when the implementation reaches it.

## No self-escalation

An agent may exercise less authority than granted; it may never grant
itself more. Never weaken, skip, or delete a check to make it pass, and
never edit a check, ruleset, required check, workflow permission,
`.claude/settings.json`, this file, or `AGENTS.md` to unblock your own
change or relax your own limits. At a policy boundary: stop; propose the
change to the maintainer with benefit, risk, and exact scope; wait;
resume only after the maintainer updates this file. When refusing a
request that exceeds this policy, state this path — the limit is
changeable, but only by the maintainer.

## Pre-authorized actions

What an agent *can* do is set by the dev container and the `gh` login
the user forwards; what it *may* do without a fresh decision is set here
and by the conversation.

- Reversible internal actions — editing files, branching, local commits,
  running recipes — proceed without asking.
- A push to the branch the change's proposal names in its
  `## Deliverable and pre-authorized actions` section, as
  `## Publishing` bounds it.
- Every other outward write — a comment, an issue, a label, a review
  request, a setting — is pre-authorized in the conversation or queued as
  a pending decision. A refusal or a declined prompt is a decision to
  re-plan or queue, never to route around with an equivalent command.

## Permission entries

Two configurations carry one policy: `.claude/settings.json` for Claude
Code and `.codex/config.toml` with `.codex/rules/default.rules` for Codex.
Both agents run in their classifier mode — Claude Code's auto mode, Codex
with `approvals_reviewer = "auto_review"` — so a model reviewer, reading
this knowledge base, judges every command no entry decides. A human prompt
costs a review of an agent-written command and halts the run; under that
load approval degrades into approving everything, so no entry asks.

- **deny** (`forbidden` for Codex) — a clear security risk: force,
  mirror, all-refs, and tag pushes and remote branch deletion; `gh pr
  merge` and approving a review; any command naming `spec/approved`;
  `gh api` with `DELETE`, `PUT`, or `PATCH`; `gh repo delete` and
  `gh repo edit`; printing the `gh` token; `sudo`; broad `rm -rf`; hard
  resets, `git clean`, and discarding the working tree. Claude Code
  denies any command naming `spec/approved`, read-only ones included (read
  the approval with `just spec-approval <n>`); Codex rules match prefixes
  only, so they forbid the `gh pr edit` label forms and the reviewer
  judges the rest.
- **allow** — only the checks every task runs: `just check`, `test`,
  `validate`, `lint`, `spec-validate`, `check-skill`, `spec-check`,
  `review-threads`, `spec-approval`, `commit-gate`. An allow entry skips
  the classifier, so the list stays this short; read-only commands are
  already approved by the agents' built-in read-only handling.
- **ask** — none. Outward writes stay bound by `## Publishing`; the
  classifier holds the agent to it.

Prefix rules are fool-proofing against a careless agent, not a boundary:
a reordered flag or a wrapper gets past them. The container is the
boundary. The mode is the maintainer's setting: Claude Code ignores
`defaultMode: "auto"` and `autoMode` in a project file (auto mode is its
default starting mode), and Codex reads `.codex/` only for a trusted
project. In Claude Code's Manual mode an unmatched command prompts, as
before.

## Threat model

Recorded 2026-10-19. No unattended agent runs in this repository: every
agent session is directed by the maintainer, who reviews results before
they leave. The workflows that read issue and comment text
(`issues / triage`, `spec / command`, `spec / labels`) are deterministic
scripts without a model, run from the default branch with job-level
permissions (`github-checks.md ## Privileged workflows`). The dev
container holds no credential beyond the `gh` login the user forwards and
no host mounts; session logs stay in the container user's home and are
not collected. The strict isolation tier therefore applies to no role
here; revisit this section before adding any scheduled or unattended
agent run.

Accepted risks:

- A workflow on a same-repository branch can post a record-shaped comment
  as `github-actions[bot]`, and any write-access account can add
  `spec/approved`. Write access is the maintainer's and agents act through
  the maintainer's login; a record counts only when it answers the latest
  person-added label event and its head; the maintainer checks the
  recorded commit (`spec-workflow.md ## Approval gates`).
- The deny entry on `spec/approved` blocks read-only commands that name
  the label; `just spec-approval <n>` is the read path.
- Outward writes (pushes, pull request and issue writes, labels) are
  judged by the agent's classifier, not a human prompt. It reads
  `## Publishing` and the conversation; the irreversible acts are denied
  outright, a push lands only on a branch, and `main` changes only through
  the maintainer's merge.

## Update this file when

- The maintainer changes the level, a condition, or a gate owner.
- The verification strength this level was priced against degrades: a
  required check is removed, behavioral tests are skipped by default, or
  revert stops being a complete rollback.
- The specification workflow changes what "approved", "frozen", or
  "archived" means, or how an approval is recorded.
- `.claude/settings.json` or `.codex/` gains or loses an entry, an
  unattended agent run is added, or the dev container gains a credential
  or a host mount.
