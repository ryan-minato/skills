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
- `.agents/knowledge/github-workflow.md` — the workflow file the workflow
  builder deposited (objects in use and not used, decomposition, triage,
  planning view), to which this builder appends label semantics, tracking
  issue conventions, milestone policy, and the grooming owner; created here
  in the same shape when no builder ran.
- `.agents/knowledge/github/` — `platform-settings.md`
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

Register every copied pair with an owner and update trigger in a
`## Synchronization` table of `.agents/knowledge/github-workflow.md` — and
prefer deriving over copying wherever possible (a checklist workflow parses
the PR template's headings; a tag check reads its pattern from one
committed config):

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
- every filled extension slot ↔ the paradigm contract it was filled from
  (the filling builder registers the row; a contract that changes without
  its slot text, or the reverse, leaves the harness saying two things);
- ruleset ↔ legacy branch protection (record both layers or drift is
  invisible);
- release tag and changelog ↔ package or deployment automation;
- public contribution and security statements ↔ the internal workflow.

## Extension slots

The base is paradigm-neutral: nothing in the templates, forms, project
skill, or knowledge presupposes a development paradigm. A paradigm
builder — one whose description claims the contract the entrypoint points
to — fills these slots after this builder has delivered. A slot is a
structural location (a heading, a step, a field id), never a surviving
placeholder or an anchor comment, so delivered files read clean and
`grep -rn '{{[A-Z]'` stays empty (Actions expressions are not placeholders).

| Slot | Location in the delivered base | What a fill inserts |
|---|---|---|
| `RELATED_WORK_LINES` | PR template, under `## Related work`, after the closing-keyword comment | lines linking the paradigm's record and its phase |
| `ACCEPTANCE_ITEM` | PR template, the checklist item beginning "Acceptance criteria" | an alternative acceptance source, inserted before "are met" |
| `CHECKLIST_ITEMS` | PR template, between the acceptance item and the security item | further checklist items; the security item is never edited |
| `INTAKE_LINK_FIELD` | task and feature forms, immediately before the field with id `acceptance` | one optional input linking the paradigm's artifact |
| `ACCEPTANCE_SOURCE` | task and feature forms, the `acceptance` field's `description` | the alternative source of acceptance, appended |
| `COMPLETION_SOURCE` | tracking-issue body, `## Observable completion` | what the goal's completion links instead of restating |
| `TAKE_WORK_PRECONDITION` | project skill, `## Take work` step 1, after the executable-criteria sentence | what must exist before the issue is taken |
| `DRAFT_FIRST_CONTENT` | project skill, `## Take work` step 3, after "the claim and the work log" | what the draft's first push carries and what the agent then waits for |
| `CREATE_WORK_RULE` | project skill, `## Create issues`, before the tracking-issue sentence | how issues derive from the paradigm's artifacts |
| `FINISH_STEP` | project skill, `## Finish` step 2, after "update the final description" | the paradigm's step before the authority policy applies |
| `KNOWLEDGE_SECTION` | `.agents/knowledge/github-workflow.md`, appended as one `## <Paradigm>` section | the contract's location, the slots filled because of it, the update trigger |
| `SYNC_ROW` | `.agents/knowledge/github-workflow.md`, the `## Synchronization` table | one row per filled slot ↔ its contract |
| `MAINTAINER_ACTION` | `platform-settings.md`, one row | a setting the paradigm's automation needs, recorded as a maintainer action with its readback |

Fill contract, for the paradigm builder: locate each slot by its structure,
never by a marker; insert, never reword base text; before inserting, grep
for the sentence about to be inserted and skip the slot when it is already
present, so a second run changes nothing; keep the security checklist item
byte-identical; a paradigm's own local check joins the command the checks
workflow already runs rather than a new workflow; register one `SYNC_ROW`
per insertion; then rerun this builder's step-5 checks — placeholders
(`grep -rn '{{[A-Z]'`), links, workflow YAML, checklist parsing against
the template, and a clean-context readback of the project skill.

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
