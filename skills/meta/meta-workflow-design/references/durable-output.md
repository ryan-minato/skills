# Durable Output

Read on every build, before depositing the workflow file.

## Where the file lives

Deposit the adapted file at `.agents/knowledge/<platform>-workflow.md` in
the target project — `github-workflow.md`, `gitlab-workflow.md` — (create
the directory if the project has no knowledge tree; if the project keeps
agent knowledge elsewhere, follow the existing convention and record the
path in the hand-off). This file is the single source of truth for the
management model as this project runs it: the governance builder reads it to
anchor authority boundaries, the platform lifecycle builder reads it to
implement — never re-decide — the objects it names, and appends the
mechanics it adds (label semantics, tracking conventions, grooming owner) to
this same file rather than opening a second one.

## The entrypoint pointer

Add one pointer to the project's agent entrypoint (`AGENTS.md` or its
equivalent), event-triggered rather than always-read, in the platform's own
words — for GitHub:

> Read `.agents/knowledge/github-workflow.md` before creating a branch,
> opening or updating an issue or pull request, applying labels, creating a
> milestone, or proposing any new management structure.

Do not paste the file's content into the entrypoint; one source of truth per
fact.

## What the file must carry

The deposited file, adapted from the asset and written entirely in the
platform's vocabulary, must state:

- The platform and its evidenced constraints (owner type, plan, what exists
  natively and what does not), then the base profile, overlays, and change
  propagation, each with the selecting fact — in plain words, not the
  profile letter or the model's nouns.
- Every platform object in use with its meaning here and what is lost
  without it (the deletion-test answer, translated).
- Every platform object deliberately not used with the concrete trigger that
  would justify enabling it — this is what makes progressive formalization
  real rather than a slogan.
- The work-decomposition rule, the objective and timebox policy in the
  platform's objects (milestones, iterations, tracking issues), and the
  planning view, phrased so an agent can apply them to a concrete item.
- An "Update this file when" list naming the events that reopen the design.

The file records facts and policies. Planning views are rebuildable at any
time; nothing in the file may exist only inside a view.

## Survival rules

- Platform vocabulary only: the file names the platform's objects and
  operations and never a term that only this builder defines. The
  step-5 check enforces it. If the project ever changes platform — rare, and
  a human decision — the file is rewritten for the new platform from a fresh
  design round, not translated line by line.
- No trace of the builder: the deposited file never carries this skill's
  disposable marker, name, or paths.
- Governance is adjacent, not inlined: agent authority lives in
  `.agents/knowledge/agent-authority.md`, produced by `meta-agent-authority`.
  The file may point to it; it must not restate it.
- Specifications are adjacent, not inlined: the specification discipline —
  level, tool, artifact map, approval gate, change request shape, archive
  mode — lives in `.agents/knowledge/spec-workflow.md`, produced by
  `meta-spec-workflow`. The file may point to it; it must not restate it,
  and a specification is the content of a change request's acceptance,
  never a semantic of its own.
