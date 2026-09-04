# GitHub Projects (v2) — Opt-in Branch

Read only after the user has explicitly opted into GitHub Projects. Do not
read this file to decide whether to use Projects — that decision belongs to
the design tree.

## Ownership and linkage

Projects are user- or organization-owned, never repository-owned; a
repository can only link projects owned by the same account, and the
project **number** is an account-scoped identifier the harness must
discover and record — it cannot be derived from the repository. Projects
(classic) is removed; any column/card guidance is dead. Creating or
restructuring a project is an account-side write requiring its own
approval and, usually, broader credentials than repository work.

## Data model and limits

Up to 50 fields and 50,000 items per project. Field types: text, number,
date, single select, iteration, plus fields surfaced from issues (type,
parent, sub-issue progress). **Organization issue fields also appear here
and count toward that 50** — they are org-level, so a project-level
single-select duplicating one is a second source of truth for the same
axis. Where the organization already owns `Priority`, use it and do not
create a project Priority field. **Single-select fields** are the
platform's real mutual-exclusivity mechanism — prefer one over any
label-pair convention the moment Projects is in play. **Iteration fields** are the
only sprint primitive on GitHub: arbitrary lengths, breaks, and
`@current`/`@next` filters; a Scrum shape lives here or nowhere. Views:
table, board, roadmap.

## Automation: platform first

Projects has built-in workflows (item closed → status, PR merged → status,
auto-add by filter, auto-archive) attributed to
`@github-project-automation`. Use them before writing any Actions
workflow; add an Actions step only for what the built-ins cannot express.
The API surface is GraphQL plus a REST API for Projects v2, including
field endpoints under `orgs/{org}/projectsV2/{number}/fields` — verify
availability against the live instance before depending on it, and mind
the separate `project` token scope, which default tokens lack.

## Deterministic tooling

Copy `scripts/project_fields.py` into the durable
project skill: moving an item programmatically requires GraphQL node IDs
that no human-facing `gh project` command surfaces, and the resolution is
deterministic. It is the only bundled script gated on this branch.

## Durability

A project is account-side state invisible to a checkout. Mirror the field
schema, each view's meaning, and the grooming owner into
`.agents/knowledge/github-workflow.md`, and register the project number and
link in `platform-settings.md` with readback commands.

Done when: the project link, field schema, view meanings, automation
ownership (built-in versus Actions), and grooming owner are all recorded in
repository knowledge, and no rule exists only inside the project UI.
