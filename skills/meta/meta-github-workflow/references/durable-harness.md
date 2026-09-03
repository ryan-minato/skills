# Durable Harness Deposit

Read on every approved build. This is the deposit contract that makes the
builder removable.

## Entrypoint

Use the target's existing agent entrypoint. Otherwise create `AGENTS.md` as
a compact map: project purpose, always-on safety and validation rules, the
GitHub workflow skill location, and exact when-to-read pointers to each
knowledge file. Do not turn it into the full lifecycle manual, and do not
create a competing entrypoint beside an existing one.

## Default structure

- `.agents/skills/github-project-workflow/` (or the target's existing skill
  directory, probed in order `.claude/skills/`, `.agents/skills/`) — the
  recurring intake, claim, pull-request, release, and approved
  platform-operation procedures, organized around the PR loop.
- `.agents/knowledge/github/` — `planning.md` (label semantics, tracking
  issue conventions, milestone policy, grooming owner), `platform-settings.md`
  (the remote-settings register), `checks.md` (job-name-to-command map,
  required list, aggregator gates, and what a healthy run of each workflow
  looks like — how to tell "not running" from "passing"), plus
  `ownership-and-security.md`, `releases-and-deploys.md`, and, only when
  selected, `experiments.md`.
- `.github/` — issue forms, `ISSUE_TEMPLATE/config.yml`, the PR template,
  workflows, `dependabot.yml`, `release.yml`, CODEOWNERS at one chosen
  location, and health files at one consistent precedence level.

If the framework cannot load project skills, place the recurring procedure
in an existing workflow document reachable from the entrypoint. Never
generate an undiscoverable skill.

## Remote settings become knowledge

Rulesets, legacy branch protection, org issue types and fields, Projects
field schemas, environment reviewers, security features, and Actions
policies are account-side state no checkout can prove or recreate. Record
each in `platform-settings.md` with: intended setting, enforcement tier
(`enforced` / `advisory` / `convention`), owner, exact readback command or
UI path, last-verified evidence, and the update trigger. Secrets and
variables are recorded by name, never by value. Where the tier is
advisory-only, pre-compute the upgrade trigger ("if this becomes public or
the plan upgrades, enable the ruleset with these exact job names").

## Synchronization ownership

Register every copied pair with an owner and update trigger — and prefer
deriving over copying wherever possible (a checklist workflow parses the PR
template's headings; a tag check reads its pattern from one committed
config):

- local validation command ↔ CI job, and CI job name ↔ required-check name;
- path filter ↔ aggregator gate job;
- `merge_group` event ↔ every required-check workflow (when merge queue is
  on);
- organization issue types ↔ issue-form `type:` ↔ the triage
  workflow ↔ the `platform-settings.md` row recording them;
- label taxonomy ↔ `release.yml` categories ↔ issue-form `labels:` ↔
  labeler config (mechanically checked by the deposited taxonomy check, and
  its paths filter ↔ the files it checks);
- PR-template security checklist wording ↔ the checklist workflow's
  `SECURITY_KEYWORD` (rewording the line without it fails every pull
  request, fail-closed and unexplained);
- directory or module ↔ CODEOWNERS pattern ↔ area label;
- ruleset ↔ legacy branch protection (record both layers or drift is
  invisible);
- release tag and changelog ↔ package or deployment automation;
- public contribution and security statements ↔ the internal workflow.

## Proportionality

A solo repository gets the checks workflow and nothing else; add
enforcement only for contracts that have actually been violated. Every
failing check must print a fix-it message naming the file to edit. The
harness must never be the thing that breaks contributions.

## Disposal test

Before the closing step asks the user about deletion, simulate removal: search target files for this
skill's name, path, disposable marker, and conversation-only references;
verify every remaining link and procedure; confirm every knowledge file is
reachable from the entrypoint and vice versa. Cleanup is a fresh, explicit
user action after the exact disposable set is shown; building the harness
is not deletion consent.
