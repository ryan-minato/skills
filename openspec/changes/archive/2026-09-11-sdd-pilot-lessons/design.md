## Context

See proposal.md for motivation. Six existing domains change; no skill
directory is added, renamed, or removed. Binding constraints:

- `skills/engineering/CONTEXT.md`: an engineering skill depends only on
  `core` and hands off to other catalogs by role; its body may name the
  target skill, its description may not. `spec-driven-development` today
  ships `scripts/archive_completed_changes.py`, which the two platform
  builders name as the file their archive workflow calls — a dependency
  across the catalog boundary that `skills/meta/CONTEXT.md` (builders
  depend on `meta` siblings by name) does not allow.
- `skills/meta/CONTEXT.md` `## Contract flow is one-way` knows two kinds
  of builder (contract, platform); the paradigm builder is a third kind
  that runs twice: before the platform base (contract) and after it
  (shaping).
- Sizes: descriptions ≤1024 characters (warn above 900);
  `spec-driven-development`'s is 1018 and stays; `meta-spec-workflow`'s is
  952 and is rewritten under 900. Bodies stay under 500 lines
  (`meta-spec-workflow` SKILL.md ≈330 after the second phase, SDD ≈220
  after the cut).
- Mirrors (`scripts/validate_harness.py`): `scripts/archive_completed_changes.py`
  ↔ the engineering skill's copy today; the companion change
  `sdd-pilot-lessons-harness` repoints it to `meta-spec-workflow`.
- The platform bases' delivery rule: no `{{PLACEHOLDER}}` survives
  (`grep '{{'` empty), and anchor comments are dead weight in files agents
  read on every run — so extension slots are structural (a heading, a
  step number, a field id), registered once in each base's
  `references/durable-harness.md`.
- This repository's own harness was generated from these builders and is
  updated by the companion change in the same pull request.

## Placement

| Requirement | File and section | Load trigger (references only) |
|---|---|---|
| SDD — Change request shape is recommended from the project's consumers | `SKILL.md` `## Project rules live in the contract` (contract-or-default reading; the methodology answer when asked directly) | — |
| SDD — Specification review happens on the published draft with a recorded approval | `SKILL.md` `## The loop` step 2–3 and `## What specification review examines`; `references/tracked-work-lifecycle.md` `## Approval modes` | "how each approval mode is exercised and recorded" |
| SDD — The draft is published as soon as the specification is written | `SKILL.md` `## The loop` step 2 ("publish, then stop") | — |
| SDD — Archive mode is recommended from the automation available | `SKILL.md` `## The loop` step 7; `references/tracked-work-lifecycle.md` `## Archive modes at run time` (freeze rule) | "when the contract records the mode, or when none exists" |
| SDD — Handoff: the harness builder for spec workflows | `SKILL.md` `## Setting up or improving the project's rules` | — |
| SDD — REMOVED Script: archive_completed_changes.py | `scripts/archive_completed_changes.py` deleted; `## The loop` step 7 and gotchas no longer mention it; README pair rows drop the script sentence | — |
| SDD — Level, tool, and lifecycle facts are read from the contract, defaulted when absent | `SKILL.md` `## Project rules live in the contract` (the defaults list) | — |
| SDD — Closing the discussion reconciles the request before implementation | `SKILL.md` `## The loop` step 3 (reconcile before plan); `references/tracked-work-lifecycle.md` `## Reconciliation checklist` | "when the gate owner closes the discussion" |
| SDD — The change request body navigates to the record and carries no implementation until ready | `references/tracked-work-lifecycle.md` `## Default request body` (used when the project has no template) | "when drafting the request body" |
| SDD — Specification domains cover the product; harness changes are spec-less | `SKILL.md` `## Project rules live in the contract` (scope default); `references/adopting-existing-code.md` product boundary sentence | — |
| SDD — Baseline scenarios are verified and untouched defects are filed | `SKILL.md` `## The loop` step 6; `references/adopting-existing-code.md` baseline bullet | — |
| SDD — Spec artifacts are created through the tool's commands and validated programmatically | `SKILL.md` `## Tool commands first, then the validator` (after `## The loop`), one sentence each in steps 1, 2, 7; gotcha on hand-built records; `references/tracked-work-lifecycle.md` `## Per-tool loop notes` (command category and validation point per step) | — |
| MSW — Trigger: description | `SKILL.md` frontmatter `description` | — |
| MSW — Shape, archive mode, and author are settled with a reasoned recommendation | `SKILL.md` step 2 questions 3 (approval mode) and 8 (specification scope); `references/tracked-work-lifecycle.md` (moved from SDD: shape table, timing table, selecting facts) | "before asking questions 3–8" |
| MSW — The deposited contract carries the new facts in platform vocabulary | `assets/spec-workflow.md` (`## Artifact operations`, `## Approval gate`, `## Change request shape`, `## Archive mode`, `## Specifications and <issues>`, `## Scope of specifications`); `references/durable-output.md` `## What the contract must carry` | "on every build, before depositing" |
| MSW — Tool references distinguish fixed and project-defined archive operations | `references/openspec.md`, `references/spec-kit.md`, `references/kiro.md`, `references/committed-documents.md`: a `## Commands and validation` paragraph each (what is created by command, validator and strict mode or the structural check to adopt), the spec-less marker and archive timings in the OpenSpec one | "read the reference for the selected approach" |
| MSW — The platform base is shaped for the contract in a second phase | `SKILL.md` step 1 (base and contract already present?), step 6 hand-off order, new step 7 `Shape the platform`, step 8 close; `references/github-expression.md`, `references/gitlab-expression.md` (one row per slot with the insertion text by shape and mode, the fill contract, the validator-into-check-command step) | "step 7: read the expression reference for the evidenced platform" |
| MSW — Take-work and the draft follow the change request shape on the platform | `assets/github/project-skill-steps.md`, `assets/gitlab/project-skill-steps.md` (TAKE_WORK_PRECONDITION, DRAFT_FIRST_CONTENT, CREATE_WORK_RULE texts) | — |
| MSW — Templates carry the specification block and checklist items | `assets/github/template-lines.md`, `assets/gitlab/template-lines.md` (RELATED_WORK_LINES, ACCEPTANCE_ITEM, CHECKLIST_ITEMS, INTAKE_LINK_FIELD, ACCEPTANCE_SOURCE, COMPLETION_SOURCE texts) | — |
| MSW — Closing the discussion reconciles the request before implementation | `assets/<platform>/project-skill-steps.md` (the reconcile step, platform commands described by category) | — |
| MSW — OpenSpec projects get a runnable archive job; other tools get design guidance | `assets/github/workflow-spec-archive.yml`, `assets/gitlab/ci-spec-archive.yml` (moved from the bases); `references/<platform>-expression.md` `## Push path by owner type` and MAINTAINER_ACTION row; `references/tracked-work-lifecycle.md` `## Archive modes` (job rules, per-tool notes) | "when the contract records automated archiving" |
| MSW — Script: archive_completed_changes.py | `scripts/archive_completed_changes.py` (moved byte-identical from the engineering skill, then one sentence on `skip_specs` in the docstring and epilog) | — |
| GH/GL — REMOVED four specification requirements | `references/spec-expression.md` and the archive asset deleted; `{{SPEC_*}}` placeholders, the `spec` form field, and the specification passages removed from `SKILL.md`, `decision-tree.md`, `durable-harness.md`, `issues-and-prs.md` / `work-items-and-mrs.md`, `semantic-mapping.md`, `rules-and-protection.md`, `ci-and-runners.md`, `security-and-governance.md`, the templates and forms, `project-skill.md` | — |
| GH/GL — Templates and the project skill carry paradigm-neutral extension slots | `references/durable-harness.md` `## Extension slots` (slot table and fill contract); `SKILL.md` step 5 ("confirm each slot's structure exists") and the closing hand-back sentence | "step 5, before the delivery checks" |
| GH — A required check is named in the ruleset only after it has run on the default branch | `references/rules-and-protection.md` required-checks paragraph | — |
| GL — The protected branch baseline includes thread resolution and pre-existing pipeline jobs | `references/security-and-governance.md` baseline list | — |
| AA — The approval gate precedes review admission and pays H1's price | `SKILL.md` gate paragraphs; `references/authority-profiles.md` H1 rows; `assets/agent-authority.md` Gates table ("contract-defined gate, if any") | — |
| AA — The acceptance-evidence report points at the change request's record and results | `references/durable-output.md` report items; `assets/agent-authority.md` report section | — |
| WD — The hand-off order runs the paradigm builder's second phase after the platform base | `SKILL.md` closing hand-off order; `assets/platform-workflow.md` `## Other contracts` | — |

## Description

`meta-spec-workflow` only. It must state the capability (settling and
depositing a project's spec-driven rules, then shaping a delivered
platform base for that contract), load on direct requests to initialize or
improve a project's spec-driven rules, on indirect ones where a harness,
template, or tracker contradicts the spec tool, and on "the platform
harness is built, now make it follow our contract"; it must not load for
writing a specification or deciding whether to adopt the practice. Budget:
under 900 characters (currently 952), which means dropping the enumerated
tool names from the "Use when" clause. `spec-driven-development`'s
description is unchanged by decision.

## Dependencies and handoffs

- `spec-driven-development` → `plan-clarification` (`core`, in range,
  unchanged) and → the spec workflow builder of the `meta` catalog, by
  role in the description-free body, through `ryan-minato-skills-installing`,
  installing the catalog whole; fallback when declined: record the
  defaults in the knowledge base, list the harness build as remaining,
  edit no harness file.
- `meta-spec-workflow` → `meta-github-workflow`, `meta-gitlab-workflow`,
  `meta-agent-authority`, `meta-workflow-design` by name (`meta` siblings,
  allowed); the second phase requires a delivered base and ends after the
  contract deposit when none exists.
- The platform bases → "the builder whose description claims the contract
  the entrypoint points to" (role, not name), keeping the bases free of
  any paradigm's vocabulary.
- `meta-workflow-design` names the paradigm builder's two runs in its
  hand-off order (by name, allowed).

## External impact

- `skills/engineering/README.md` + `README.zh.md` (SDD row: apply the
  contract, no bundled script) and `skills/meta/README.md` + `README.zh.md`
  (meta-spec-workflow: second phase and script; the two bases:
  paradigm-neutral with extension slots) — `just validate` (README pair
  check) and a read of both rows.
- `skills/meta/CONTEXT.md` (paradigm builders as a third kind; the
  contract-flow rule; the sibling dependency list) and
  `skills/engineering/CONTEXT.md` (disambiguation row) — `just validate`.
- `.claude-plugin/marketplace.json` `meta` plugin description (human-owned
  field, edited by hand; `just gen-marketplace` must leave it as edited).
- `openspec/specs/meta/meta-github-workflow/spec.md` and
  `.../meta-gitlab-workflow/spec.md` `## Purpose` lines, hand-edited under
  the exception the companion records — `just spec-validate`.
- `meta-harness-building/assets/harness-plan-template.md` Build List rows
  for phased builders (no domain; documentation-only, listed in
  proposal.md Impact) — `just check-skill skills/meta/meta-harness-building`.
- The mirror `scripts/archive_completed_changes.py`, its origin in
  `scripts/validate_harness.py`, the register row, this repository's
  contract, project skill, template, and reviewer rule — the companion
  change `sdd-pilot-lessons-harness` (`just validate`, `diff` of the
  mirror pair).
- Symlinks and the `marketplace.json` skill lists: unaffected (`git
  status --short` shows no `.agents/skills/` change).

## Decisions

- **Structural slots, not placeholders or anchors** (serves the bases'
  extension-slot requirement and MSW's second phase). A slot is a heading,
  a step number, or a field id the base already has; the fill contract is
  locate by structure, insert only, grep the sentence first so a second
  run is a no-op, register a sync row. Rejected: surviving `{{SLOT}}`
  markers (the base's own delivery check forbids them) and HTML anchor
  comments (dead text in every agent read).
- **The script moves byte-identical, then gains one sentence** (serves
  Script: archive_completed_changes.py). The companion repoints the mirror
  in the same pull request so `just validate` never sees two origins.
  Rejected: leaving a stub in the engineering skill — a builder may not
  depend on it.
- **Discussion-closed is the default the methodology teaches and the
  builder recommends** (serves the approval, reconciliation, and gate
  requirements). The record of approval is the closing instruction plus
  the request's discussion state; blocking stays available as a
  contract choice. Rejected: making the methodology skill ask — SDD never
  runs a questioning round.
- **The validator joins the project's local check command, not the
  base's checks workflow** (serves Validator joins the check command and
  the SDD tool-commands requirement). The base already runs that command
  in CI, so no workflow edit and no new slot is needed. Rejected: a
  `CHECK_STEP` slot in the checks workflow — a second file to keep in
  sync for the same effect.
- **Per-platform expression references and assets live in
  `meta-spec-workflow`** (serves the second phase). They are the moved
  `spec-expression.md` files rewritten as edit tables keyed by slot.
  Rejected: keeping them in the bases behind a "paradigm" flag — the
  bases would still carry SDD text.
- **SDD's description stays** by decision (1018 characters; its trigger
  vocabulary is unchanged); the responsibility cut is expressed in the
  body and the README row.
- **Purpose lines are corrected by hand** in the two platform domains,
  because the schema allows `## Purpose` only for new domains and the
  current sentences describe removed capabilities; the companion records
  the exception in `spec-workflow.md`.
- **One change, one pull request, BREAKING commits** — the three moves
  are interdependent (script origin, slot lists, hand-off order) and would
  leave installed skills inconsistent if split across releases.

## Risks / Trade-offs

- [Installed bases stop expressing specifications; installed SDD loses
  its script and template fallback] → **BREAKING** in proposal and commit
  subjects; the migration line names `meta-spec-workflow`.
- [A base sentence survives in paradigm vocabulary] → readback r1 fails
  on any spec/OpenSpec/archive_completed occurrence beyond generic
  "specification" as in "a standards document"; `git grep` listed in the
  verification plan.
- [The slot table in the bases and the slot table in
  `meta-spec-workflow` drift] → registered as a human-synchronized pair in
  `harness-maintenance.md` by the companion.
- [Description rewrite breaks the loads that work today] → t1–t3 in a
  fixture project at the Sonnet tier; the near-miss stays.
- [Solvers inside this repository answer from `.agents/knowledge`] → every
  trigger and outcome case runs in a fixture project outside the
  repository; readbacks receive only the changed files.
- [o5 needs the OpenSpec CLI in the fixture] → installed through `npx`
  in the fixture; if the network refuses, o5 is skipped with the reason
  and r3 covers the scenario as a readback question.
- [The archive script's harness needs a real `openspec archive`] → run in
  a scratch worktree after `openspec archive --help`, as the pilot did.

## Verification plan

Solver tier: Sonnet-class for every case (the least capable tier these
skills claim). Fixture: a scratch project outside the repository with the
skills under test temp-installed; trigger cases append a neutral
`SKILLS_LOADED:` self-report; one attempt per case, up to three on an
invalid observation. Readbacks are clean-context Sonnet subagents given
only the changed files, answering a numbered questionnaire graded item by
item.

| Scenario | Case (prompt or task) | Rubric and critical failures | Pass threshold | Solver tier | Observation | Isolation |
|---|---|---|---|---|---|---|
| MSW Trigger: Delivered base awaits shaping | t1 "our platform harness is built; now make the templates and project skill follow our OpenSpec contract" | loads `meta-spec-workflow` (critical) | 1/1 | Sonnet | `SKILLS_LOADED:` | fixture project |
| MSW Trigger: Harness alignment request | t2 "our issue template and openspec specs keep contradicting each other; make the harness match the tool" | loads (critical) | 1/1 | Sonnet | same | same |
| MSW Trigger: Writing a specification (near-miss) | t3 "write the spec for the export feature before we code it" | does not load (critical) | 1/1 | Sonnet | same | same |
| SDD: Draft body in the specification phase; Implementation offered for the draft | o1 fixture contract records the blocking mode; "change export-csv is proposed on feat/export-csv; draft the pull request description" | goal paragraph, a value section, `Spec:` link, `Phase: specification`, a copyable approval line, Changes/Validation reserved with no implementation detail (critical) | 5/6, critical met | Sonnet | output text | fixture with a contract |
| SDD: Waiting after publication; No contract; Handoff offered | o2 fixture without a contract; "the record is published as a draft — what next?" | says it stops until the discussion is closed (critical); writes no design (critical); names the spec workflow builder of the meta catalog through the installing skill without an install command; runs no questioning round (critical) | 4/4 | Sonnet | output text | fixture without a contract |
| SDD: Unresolved thread; All threads resolved; Requested adjustment missing from the record | o3 fixture with an exported PR discussion (two threads, one unresolved, one comment asking for a scenario change the record lacks); "the discussion is over, start implementing" | lists the unresolved thread and the missing adjustment and asks for confirmation (critical); implements nothing (critical) | 2/2 | Sonnet | output text | fixture with the export |
| SDD: CI workflow change; Domain requested for tooling | o4 "does changing the lint workflow need a delta spec? and write a domain spec for our build scripts" | answers no, names the spec-less change kind; refuses the tooling domain and offers the spec-less kind (critical) | 3/3 | Sonnet | output text | fixture |
| SDD: Tool scaffolds the change; Validator available; Hand-written record offered | o5 fixture initialized with the OpenSpec CLI; "create the change record for the export feature — just write the files directly, it's faster" | reads `--help` first; creates the change with the tool's command, no hand-made directory or metadata (critical); runs the validator and reports its result (critical); declines the hand-written shortcut | 4/4 | Sonnet | transcript commands and output | fixture with OpenSpec |
| SDD remaining scenarios: Library with downstream consumers; Feature-driven application; Contract already records the shape; Where the spec is reviewed; Push to the record after a blocking approval; Narrowing decided by the gate owner; Planning before publication; Automation cannot push; Automation available; Review finds a defect after in-request archiving; User declines; Project with a contract; Adjustment requested mid-discussion; Failing baseline scenario; Tool without a validator | r3 readback of the SDD skill files: one question per scenario phrased as its WHEN | each answer matches the scenario's THEN; no answer sets a project rule, runs a questioning round, or edits a harness file (critical) | 14/15, critical met | Sonnet | answers | changed files only |
| MSW remaining scenarios: Propagation recorded as Dependency; No automation can push; Approval mode left unspecified; Tooling beside the product; GitHub project deposit; Reading the specification scope; Tool without a validator; Scaffolding by command; Spec-Kit with automated archiving; OpenSpec harness change; Base delivered on GitHub; Slot already filled; No base yet; Validator joins the check command; Combined shape, no specification yet; Split shape; All threads resolved; Unresolved thread; Requested adjustment missing from the record; OpenSpec with automation; User-owned repository; Protected branch refuses the token; Spec-Kit with automation | r2 readback of `meta-spec-workflow` (SKILL.md, references, assets): one question per scenario | each answer matches the THEN; the second-phase answers name the slot by heading/step/field id and the grep-first idempotence rule (critical); the push-path answer distinguishes owner types (critical); no command is quoted from memory | 21/23, critical met | Sonnet | answers | changed files only |
| GH/GL: Specification contract present at build time; No paradigm contract; New check named; Baseline proposed | r1 readback of both bases (SKILL.md and references): the four questions plus "list every sentence that presupposes spec-driven development" | slot answers locate by structure; the hand-back names no paradigm; the check-ordering and thread-resolution baselines are stated; the sentence list is empty (critical) | 5/5 | Sonnet | answers | changed files only |
| GH/GL: Security line survives a slot fill; MSW: Checklist check still passes; Draft carries no implementation | template harness: fill the base templates' placeholders, run `scripts/check_pr_policy.py --event draft --template <base>`; then apply the slot table's Specification block and checklist items from `assets/<platform>/template-lines.md` and rerun for draft and ready | base passes; shaped draft passes; shaped ready with reserved sections fails; security line byte-identical before and after (critical) | 4/4 | — | exit codes and `diff` | scratch copies |
| AA: H1 offered with an agent-authored specification; Agent asked to approve and ready; Discussion-closed gate under H1; Reading the deposited report format | r5 readback of `meta-agent-authority` changed files | gate is contract-defined and generic; agent never approves its own artifact; discussion-closed passes when the owner closes and reconciliation finds nothing open (critical); report points at the Specification block and Validation section | 4/4 | Sonnet | answers | changed files only |
| WD: Hand-off order names the second phase; Deposited file without a paradigm contract | r4 readback of `meta-workflow-design`, `meta-spec-workflow` SKILL.md step 6, `skills/meta/CONTEXT.md`, and the harness-plan template | one consistent order: contract → authority → platform base → paradigm second phase (critical); `## Other contracts` is generic | 2/2 | Sonnet | answers | changed files only |

Script and tool harnesses (`S=skills/meta/meta-spec-workflow/scripts/archive_completed_changes.py`,
run in a scratch worktree after `openspec archive --help` confirms the
current flags):
- Help: `python3 $S --help` → usage names `--dry-run` and the spec-less
  case; exit 0.
- Nothing completed: `python3 $S --dry-run` in a worktree whose changes all
  have open tasks → exit 0, no output.
- Representative run: a change with every task ticked and a delta spec →
  moved under `openspec/changes/archive/`, main spec carries the delta,
  `openspec validate --all --strict` passes.
- Spec-less change: a `skip_specs: true` change with every task ticked →
  moved, main specs unchanged (`git diff --stat openspec/specs` empty),
  strict validation passes.
- Repeated run: the representative command again → `git status --short`
  unchanged.
- Bad arguments: `python3 $S --bogus` → exit 2, stderr names `--bogus`.
- Mirror: `diff scripts/archive_completed_changes.py $S` empty.
- Residue: `git grep -n -i 'openspec\|archive_completed\|{{SPEC_\|spec-expression' skills/meta/meta-github-workflow skills/meta/meta-gitlab-workflow` prints nothing; `grep -rn '{{' skills/meta/meta-spec-workflow/assets` prints only the documented placeholders of the moved assets.
- `just check-skill` for the seven directories (SDD, meta-spec-workflow,
  both bases, meta-agent-authority, meta-workflow-design,
  meta-harness-building), `just lint`, `just validate`,
  `just spec-validate`, `just check`.

Skipped:
- None planned. If the fixture cannot install the OpenSpec CLI, o5 is
  recorded as skipped with the reason and its three scenarios are covered
  by r3 questions.

## Open Questions

None.
