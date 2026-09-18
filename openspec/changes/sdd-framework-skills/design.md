## Context

See proposal.md. Two existing domains change and two are created; one
skill directory moves catalogs and two are added. Binding constraints:

- `.agents/knowledge/skill-quality.md`: SKILL.md bodies under 500 lines;
  descriptions ≤1024 characters (warn above 900); `references/` split by
  branching condition with a precise load sentence per file; scripts
  introduced with a relative link, `--help`, exit codes 0/1/2, idempotent.
- Catalog contracts: `engineering` allows only `core` dependencies and
  forbids day-to-day tool operation; `meta` skills carry the disposable
  marker and may name `meta` siblings; the new `sdd` catalog (companion
  change) holds durable skills, dependencies on `core` only, and every
  pairing between its skills or with `meta-spec-workflow` is an optional
  handoff through `ryan-minato-skills-installing` with a fallback.
- Self-containment: a framework skill restates the two loop facts it
  needs (package before tasks, archive or complete before ready) instead
  of referencing the methodology skill's files.
- Reserved names: `openspec-*` project skills under `.agents/skills/` are
  CLI-generated; `scripts/validate_harness.py` ignores symlinks there, so
  the `openspec-workflow` symlink is legal — the companion change proves
  `just spec-sync` leaves it alone.
- Mirrors: `scripts/archive_completed_changes.py` ↔ the builder's copy
  today; the companion change repoints the pair to `openspec-workflow`'s
  `spec_changes.py`.
- Platform facts the automation is designed on (verified 2026-09-17
  against the GitHub and GitLab documentation): a push made with the
  platform token puts the resulting `pull_request` runs in an
  approval-required state; `labeled` and `issue_comment` events the token
  causes create no runs; `issue_comment` runs the default-branch workflow
  file; GitLab has no pipeline source for a note or a label change.

## Placement

| Requirement | File and section | Load trigger (references only) |
|---|---|---|
| SDD — Trigger: description | `sdd/spec-driven-development/SKILL.md` frontmatter | — |
| SDD — The draft opens when the approval package is complete | `SKILL.md` `## The loop` steps 2–3; `references/tracked-work.md` `## Publishing and waiting` | "when a loop step meets the platform: publishing the draft, waiting for approval, reconciling, drafting the body, archiving before ready" |
| SDD — The approval gate examines the outcome and the approach bounds, never the tasks | `SKILL.md` `## The approval package`; `references/tracked-work.md` `## What the gate examines` | same |
| SDD — The design bounds the approach and is part of the approval package | `SKILL.md` `## The approval package` (role, warranted rule, bounds, publication rule); gotchas | — |
| SDD — The approval package is composed per the project's tool | `SKILL.md` `## The approval package` (one line per family) and `## Approach families` pointer to the framework skills | — |
| SDD — Archiving happens inside the request, by hand or by the automation the framework skill installs | `SKILL.md` `## The loop` step 7 and `## Project rules live in the contract` (default executor); `references/tracked-work.md` `## Archiving before ready` (executor, label, fork, freeze); `references/adoption-decision.md` `## Rejected alternatives` (after-merge job) | tracked-work: same; adoption-decision: "when the user asks whether, at which level, or with which approach family to adopt, or before recommending a tool" |
| SDD — Handoff: the framework skill for the project's spec tool | `SKILL.md` `## The framework skill` | — |
| SDD — Specification review happens on the published draft with a recorded approval | `references/tracked-work.md` `## Approval modes` | same |
| SDD — Closing the discussion reconciles the request before implementation | `SKILL.md` `## The loop` step 3; `references/tracked-work.md` `## Reconciliation checklist` | same |
| SDD — The change request body navigates to the record and carries no implementation until ready | `references/tracked-work.md` `## Default request body` (`Records:` = package, tasks after) | same |
| SDD — Level, tool, and lifecycle facts are read from the contract, defaulted when absent | `SKILL.md` `## Project rules live in the contract` | — |
| SDD — Handoff: the harness builder for spec workflows | `SKILL.md` `## Setting up or improving the project's rules` | — |
| SDD — Spec artifacts are created through the tool's commands and validated programmatically | `SKILL.md` `## Tool commands first, then the validator` | — |
| SDD — unchanged requirements (shape, tracked work without acceptance, product-only domains, baseline defects) | `SKILL.md` `## Project rules live in the contract`, `## The loop` step 6; `references/tracked-work.md` `## The two shapes at run time`; `references/adopting-existing-code.md` | — |
| MSW — Trigger: description | `meta/meta-spec-workflow/SKILL.md` frontmatter | — |
| MSW — Shape, archive mode, and author are settled with a reasoned recommendation | `SKILL.md` step 2 questions 3 (package, design-warranted rule) and 6 (executor, fork rule); `references/contract-design.md` `## Approval modes`, `## Archiving` | "before asking questions 3–8 of step 2 and again before step 7" |
| MSW — The deposited contract carries the new facts in platform vocabulary | `assets/spec-workflow.md` (`## Approval gate`, `## Archive`, `## Framework skill`); `references/durable-output.md` must-carry list | durable-output: "on every build" |
| MSW — Approval scope is recorded as the approval package | `assets/spec-workflow.md` `## Approval gate`; `references/durable-output.md` | — |
| MSW — The platform base is shaped for the contract in a second phase | `SKILL.md` step 7 (generic slots, then the handoff); `references/github-expression.md`, `references/gitlab-expression.md` slot tables without the archive rows | "when the evidenced platform is GitHub / GitLab and the base is delivered" |
| MSW — Take-work and the draft follow the change request shape | `assets/github/project-skill-steps.md`, `assets/gitlab/project-skill-steps.md` (`DRAFT_FIRST_CONTENT` = the complete approval package) | — |
| MSW — Templates carry the specification block and checklist items | `assets/github/template-lines.md`, `assets/gitlab/template-lines.md` | — |
| MSW — Closing the discussion reconciles the request before implementation | the `project-skill-steps.md` assets (reconcile, then tasks) | — |
| MSW — Handoff: the framework skill for the selected approach | `SKILL.md` step 3 and step 7 item 3 | — |
| MSW — REMOVED tool references, archive job, script | `references/{openspec,spec-kit,kiro,committed-documents}.md`, `assets/github/workflow-spec-archive.yml`, `assets/gitlab/ci-spec-archive.yml`, `scripts/archive_completed_changes.py` deleted; `compatibility` names only `detect_spec_tooling.py` | — |
| OSW — Trigger: description | `sdd/openspec-workflow/SKILL.md` frontmatter | — |
| OSW — The approval package is the proposal, the delta specs, and the design when warranted | `SKILL.md` `## Records and the approval package` | — |
| OSW — A change is archived inside the request once the deliberation closes, by the implementer; for a fork, by a maintainer | `SKILL.md` `## Archiving before approval` | — |
| OSW — Comment commands and status labels are read and used as the project installed them | `SKILL.md` `## Commands and labels on a request` (related-change definition included) | — |
| OSW — The automation is installed per platform from the skill's assets | `references/github.md` (check job into the checks workflow, the two privileged workflows, labels, fork safety, maintainer actions), `references/gitlab.md` (jobs fragment, tokens, limitations); `assets/github/{job-spec-check.yml,workflow-spec-command.yml,workflow-spec-labels.yml,labels-spec.json}`, `assets/gitlab/{ci-spec-jobs.yml,labels-spec.json}` | "when installing or changing the automation in a GitHub repository" / "… in a GitLab project" |
| OSW — Handoff: the methodology skill; Handoff: the contract builder | `SKILL.md` `## Handoffs` | — |
| OSW — Script: spec_changes.py | `scripts/spec_changes.py`, linked at first mention in `## Archiving before ready` | — |
| SKW — Trigger: description | `sdd/spec-kit-workflow/SKILL.md` frontmatter | — |
| SKW — The approval package is the specification and the plan | `SKILL.md` `## The feature directory and the approval package` | — |
| SKW — Completion before ready replaces archiving | `SKILL.md` `## Completion before ready` | — |
| SKW — The automation is installed per platform from the skill's assets | `references/github.md`, `references/gitlab.md`; `assets/github/{job-spec-check.yml,workflow-spec-command.yml,workflow-spec-labels.yml,labels-spec.json}`, `assets/gitlab/{ci-spec-jobs.yml,labels-spec.json}` | as above |
| SKW — Handoffs | `SKILL.md` `## Handoffs` | — |
| SKW — Script: spec_kit_features.py | `scripts/spec_kit_features.py`, linked in `## Completion before ready` | — |

## Description

- `spec-driven-development` (1018 characters today): must state the
  practice and the loop, judge whether and at which level it pays, settle
  how specs meet pull or merge requests including where the design sits
  and what the gate reviews, and convert existing code; triggers: adopting
  or starting SDD, "write the spec first", "should the design be approved
  before tasks?", how issues, requests, and specs fit, where a spec is
  reviewed, how a change is archived, a prototype needing specs, drift,
  issues and specs disagreeing; exclusions: defining goals, building the
  platform harness, a tool's own change command. The tool list leaves the
  description; budget 900–950.
- `meta-spec-workflow` (890 today): "approval and archive modes" becomes
  "approval package and archive executor"; "archive job" leaves;
  exclusion gains "installing one framework's automation"; budget under
  920.
- `openspec-workflow` (new): capability — running OpenSpec changes through
  pull or merge requests and installing the request automation; triggers:
  how a change is approved or archived in the request, `/spec` commands,
  status labels, the OpenSpec check in CI, "our
  OpenSpec PR"; exclusions: creating or applying one change (the tool's
  own skills), choosing a tool, another framework. Budget under 900.
- `spec-kit-workflow` (new): the same shape for Spec-Kit; triggers name
  the plan's approval, completion before ready, progress labels, the
  feature check; exclusions: the kit's own commands, tool choice, another
  framework.

## Dependencies and handoffs

- `spec-driven-development` → the framework skill (`sdd`, by role "the
  framework skill for the project's spec tool"), `meta-spec-workflow`
  (by role, existing), `plan-clarification` (existing optional pairing):
  all through `ryan-minato-skills-installing`; declined → the generic loop
  with the tool's help, the defaults, or the in-file questions.
- `meta-spec-workflow` → the framework skill (by the approach it claims),
  `meta-agent-authority`, the platform builders (existing): through the
  installing skill; declined → generic adoption rule, by-hand executor,
  automation listed as remaining work.
- `openspec-workflow`, `spec-kit-workflow` → the methodology skill and
  `meta-spec-workflow`, by role, through the installing skill; declined →
  answer from the restated loop facts; defaults applied and named.
- No dependency crosses a catalog boundary; `sdd` skills depend on `core`
  only.

## External impact

- New catalog scaffold, label, forms, marketplace entries, symlinks,
  the `engineering` catalog files, `skills/machine-learning/CONTEXT.md`
  pointers, the moved spec domain: the companion change
  `sdd-framework-skills-harness`; proof `just validate` and
  `just gen-marketplace` with no diff.
- `skills/meta/README.md` + `README.zh.md` row for `meta-spec-workflow`,
  `skills/sdd/README.md` + `README.zh.md` rows for the three skills (this
  change); proof: a read of both files of each pair.
- `skills/meta/CONTEXT.md`: the paradigm-builder paragraph loses "fills
  … the archive job" wording if present; proof `grep -n archive
  skills/meta/CONTEXT.md`.
- This repository's mirror, workflows, contract, project skill, and
  templates: the companion change.
- `scaffold-ml` and other skills that name `engineering/spec-driven-development`:
  proof `grep -rn 'engineering/spec-driven-development' skills/` empty
  after the companion change.

## Decisions

- **One skill per framework, holding usage and automation** (serves the
  four OSW/SKW behaviors): the automation is defined by the framework's
  commands and layout, so it lives with the framework; the platform is a
  branch inside the skill. Rejected: a durable usage skill plus a `meta`
  builder per framework — twice the skills for the same content.
- **The methodology and the builder stay framework-free** (serves the
  handoffs): a framework's details change with its releases; keeping them
  out of the two agnostic skills means a new framework is one new skill.
  Tool names still appear as examples of approach families and as facts
  in a deposited contract.
- **References split by load condition, not topic** (serves every SDD
  placement row): `adoption-decision.md` loads only while deciding,
  `tracked-work.md` only when a step meets the platform; the loop and the
  package rule are inline because every run needs them. Rejected: a
  `methodology.md` / `loop.md` pair, which is a topic split both loaded on
  most runs.
- **The approval package includes the design when warranted** (serves the
  gate requirements): a bounded design makes implementation controllable
  at the cost of some diversity; the design bounds, never steps, so the
  gate still reviews an outcome and its constraints rather than a method.
  Rejected: always requiring a design — a wording change would carry a
  design that says nothing.
- **In-request archiving only, with an executor** (serves the archive
  requirements): the after-merge job needs an integration-branch bypass
  that user-owned repositories cannot grant and leaves the branch briefly
  unarchived. The executor is the contract's fact: the implementer, or a
  maintainer on a fork's branch, because the platform token cannot push to
  a fork and no job pushes at all.
- **`pull_request_target` for the status labels, `issue_comment` for the
  commands, and nothing else privileged** (serves the automation of both
  framework skills, fork scenario): a fork's `pull_request` run has a
  read-only token and no secrets, and the setting that would grant write
  exists for private repositories only, so labelling and replying on an
  external contribution cannot be done unprivileged. Both jobs take the
  shape the platform's own labeler action uses — base checkout, REST reads,
  no checkout of the head. The command is restricted to collaborators, so
  nobody without write access can start a privileged run; that also closes
  the token-budget exhaustion an external author could otherwise mount by
  repeating it.
- **Draft warning, ready failure** (serves the check): a draft
  legitimately holds an unarchived change for the whole implementation
  phase, so it warns. A ready request holds one for the whole
  deliberation, and there the failure is the point: the red gate is what
  keeps the merge shut until the freeze, and the status label says which
  of the two reds it is.
- **REMOVED + ADDED instead of RENAMED for renamed requirements**: the
  archive applies a MODIFIED block by name and a RENAMED entry
  separately; combining both on one requirement risks a lookup under the
  wrong name. The old block is removed with a migration note and the new
  block added.
- **The moved domain is moved by directory move, title line corrected by
  hand**: the delta targets the new path so `MODIFIED` blocks resolve;
  the alternative (an `ADDED` rebuild at the new path) loses history.

- **Privileged jobs read the head through the platform's API, never as git
  objects** (serves the automation of both framework skills): a
  `pull_request_target` or `issue_comment` job holds a writable token, so
  the question is not whether today's steps execute the head — they do not
  — but whether the invariant survives a later edit. Fetching the head's
  SHA leaves the request's objects in the runner's store, one `git
  checkout` away from a takeover, and ships that shape to every project
  that installs the skill. Reading the head through `snapshot` removes the
  objects from the runner, so the guarantee is structural rather than a
  promise in a comment. Considered and rejected: the two-workflow
  `pull_request` → `workflow_run` artifact pattern that code scanning
  documents as the correct usage — it is the right answer when the
  privileged job must build untrusted code, but here it would add a trust
  hop instead of removing one, because an artifact is attacker-controlled
  data and `workflow_run.pull_requests` is empty for a fork, so the
  privileged half would have to believe the artifact's own claim about
  which request it belongs to. Also rejected: dismissing the scanner's
  finding and leaving the fetch, which keeps a correct-but-fragile design
  and trains the maintainer to wave through critical alerts.
- **No job checks the head out at all**: with the archive bot gone, no
  workflow needs the head's working tree, so the privileged surface holds
  only jobs that read through the API. The unprivileged check job is the
  one place the head is checked out, the ordinary way, with a read-only
  token.
- **The label plan is checked against a literal taxonomy in the workflow**
  before any API call, so the shell cannot be made to apply an arbitrary
  label even if the script were replaced; the loops read the plan without
  word splitting.
- **The snapshot caps files, per-file bytes, and total bytes** and fails
  loudly at a cap rather than labeling on partial data.

- **A partial read is a failure, not a caveat** (serves the request
  automation): a command that reports on documents it could not fetch is
  worse than one that stops, because the reader cannot tell the difference.
  Every incomplete path — a cap reached, a truncated tree, a document that
  does not decode — now exits 1 naming the path. The alternative, recording
  the omissions in the document and letting the commands run, was rejected
  because nothing read that record.
- **The `meta-agent-authority` skill carries a delta here** rather than in
  a follow-up: it reads the archive executor from the contract, and this
  change is what replaced the archive mode with an executor, so the two
  land together.

- **The archive is the lock point, not the finish line** (serves the loop
  and both framework skills): archiving before the request was reviewed
  merged the delta into the main specs at a moment when nobody had looked
  at the implementation, so a review finding left only two bad options —
  edit an archived record, which the rules forbid, or merge a known
  divergence. Moving the archive after the deliberation makes it mean that
  the specification and the implementation agree. Considered and rejected:
  keeping the old order and forbidding post-review divergence by
  convention, which is the same promise without the artifact that records
  it.
- **The deliberation runs on a red required check** (serves this
  repository's gate): the check already fails a ready request that holds an
  unarchived change, so the merge is blocked for exactly as long as the
  freeze is missing. A second, dedicated required check would separate "the
  records are wrong" from "not yet frozen" at the cost of a ruleset change
  and a check that must live on the integration branch before it can be
  required; the status label carries that distinction instead, which is
  what the label axis is for.
- **Only the archive bot is removed, not the privileged surface** (serves
  the automation of both framework skills): a fork's `pull_request` run
  gets a read-only token and no secrets, and the setting that would grant
  write applies to private repositories only, so labelling and replying on
  an external contributor's request cannot be done unprivileged. The skills
  ship to repositories that live on external contributions, so the shape
  follows the official labeler action — `pull_request_target`, no checkout
  of the head, REST reads only. Rejected: moving the labels into the
  unprivileged check job, which works on the maintainer's own branches and
  fails silently on the contributions where the visibility matters most.
- **The comment command is collaborator-only** (serves the same): the
  earlier gate also admitted the request's author, so an external
  contributor could start a privileged run on demand and, by repeating it,
  exhaust the platform token's per-repository hourly budget and break the
  other workflows with it. The command exists for the reviewer, so the
  author clause bought nothing.
- **Behavioral and security claims sit next to what they constrain**
  (serves every asset): a claim collected in a file header cannot be
  checked against an implementation further down, and this request already
  shipped a header describing a mechanism the file had stopped using. The
  header keeps orientation and the placeholder instructions; every claim
  about what a step does, and why that is safe, moves onto the step.

## Risks / Trade-offs

- [Behavioral tests across four skills are expensive] → one fixture
  project, one solver run per case at the floor tier, trigger cases
  batched; the user confirms the subagent count before the fleet starts.
- [Spec-Kit's directory layout changes between releases] → the skill and
  the script record the verification date; `check` warns rather than
  fails on an unknown file.
- [The `openspec-workflow` symlink sits beside the CLI-generated
  `openspec-*` directories] → `validate_harness.check_openspec` skips
  symlinks; the companion change runs `just spec-sync` and confirms the
  symlink survives.
- [`openspec archive --yes` refuses on validation] → the script gates on
  the task list itself and propagates the CLI's exit; the fixture run
  proves the happy path; `--no-validate` is never passed.
- [A ready request sits on a red required check for the whole
  deliberation, and a red that is normal trains people to ignore reds] →
  the status label distinguishes "awaiting the freeze" from "the records
  are wrong" at a glance in the request list, and the check's own message
  names which it is.
- [Only a norm stops a behavior change from landing after the freeze] →
  stated as a norm in the contract rather than implied as enforced; the
  alternative, requiring the archive commit to be the branch's last, would
  make every consistency fix reopen the round.

## Verification plan

Written before implementation; results go to the pull request's
Validation section. Solver tier: sonnet for every trigger case (haiku
loads skills unreliably); outcome tasks at sonnet in a scratch fixture
project under `$TMPDIR` whose `.agents/skills/` links the candidate
worktree's skills, since subagents only see skills installed in the
session project. Observation: the `SKILLS_LOADED` instrumentation for
trigger cases; graded transcripts for outcome tasks. Isolation: candidate
worktree; degradation recorded if unavailable.

| Scenario | Case (prompt or task) | Rubric and critical failures | Pass threshold | Solver tier | Observation | Isolation |
|---|---|---|---|---|---|---|
| SDD Trigger: Lifecycle question; Design timing question; Harness build request; Tool command request | t1 "how should issues and PRs work now that we use OpenSpec?"; t2 "should the design be written before or after the spec is approved?"; t3 "set up GitHub issue forms and CI for us"; t4 "create a new OpenSpec change for the export feature" | t1, t2 load `spec-driven-development`; t3, t4 do not (critical) | 4/4 | sonnet | SKILLS_LOADED | worktree |
| OSW Trigger: Archive question; Automation install; Tool command request; Tool choice | t5 "our OpenSpec change is done; how does it get archived before this PR merges?"; t6 "add the /spec comment commands and the status labels to this GitHub repository"; t7 = t4; t8 "should we use OpenSpec or Spec-Kit for this library?" | t5, t6 load `openspec-workflow`; t7, t8 do not (critical) | 4/4 | sonnet | SKILLS_LOADED | worktree |
| SKW Trigger: Approval question; Automation install; Kit command request; Other framework | t9 "does the plan.md of our Spec-Kit feature need to be approved before we write tasks?"; t10 "add progress labels and a feature check for our Spec-Kit repository"; t11 "run the specify command to start the export feature"; t12 = t5 | t9, t10 load `spec-kit-workflow`; t11, t12 do not (critical) | 4/4 | sonnet | SKILLS_LOADED | worktree |
| MSW Trigger: Harness alignment request; Delivered base awaits shaping; Writing a specification; Framework automation request | t13–t15 as in the main spec; t16 = t6 | t13, t14 load `meta-spec-workflow`; t15, t16 do not (critical) | 4/4 | sonnet | SKILLS_LOADED | worktree |
| SDD: Tasks requested before approval; Publication requested before the design; Waiting after publication; Tasks offered for review; Design offered as a step list | o1 — fixture with an OpenSpec change whose proposal and specs are written and a design drafted as numbered steps; the user asks to open the draft now and then for the task list | declines tasks (critical); rewrites the design as bounds before publishing; publishes the complete package; states what it waits for | 4/4 items | sonnet | transcript | worktree |
| SDD: Design requested as a step list; Wording change; Constraint with a private detail | o2 — three short asks in one task: a step-list design, a one-section wording change, a constraint naming an internal host | refuses steps (critical); no design for the wording change; host name absent from the design (critical) | 3/3 | sonnet | transcript | worktree |
| SDD: Reviewer's set under a spec-first kit; Tool generated the task list with the specification; Handoff offered; User declines | o3 — Spec-Kit fixture without the framework skill installed: "what does the reviewer get, and archive this feature" | names spec + plan; offers the framework skill via the installing skill and prints no install command (critical); after decline, runs the loop with the tool's help | 3/3 | sonnet | transcript | worktree |
| SDD: No contract; Contract names the label; Fork; Review finds a defect after archiving | r1 — clean-context readback questionnaire over the candidate SKILL.md and `references/tracked-work.md` | each answer matches the scenario's THEN | 4/4 | sonnet | transcript | worktree |
| SDD: remaining MODIFIED scenarios (Where the spec is reviewed; Push after a blocking approval; Narrowing; All threads resolved; Unresolved thread; Requested adjustment missing; Adjustment mid-discussion; Draft body; Implementation offered; Project with a contract; No contract; harness-builder handoff; Tool scaffolds; Validator available; Tool without a validator; Hand-written record) | r1 continued | as above | all | sonnet | transcript | worktree |
| OSW: Propose generated all four files; Harness change; By hand; Open task; Label bot; Progress asked; Label edited by hand | o4 — OpenSpec fixture with the framework skill: a change with `tasks.md` generated by propose and one open task; the user asks to archive it, then to set `spec/done` | lists package vs after-approval; refuses archive naming the task (critical); refuses the hand-set label | 3/3 | sonnet | transcript | worktree |
| OSW: GitHub install; Fork pull request labeled; Ready with an unarchived change; GitLab install | o5 — fixture repository with a `checks` workflow and gate: "install the automation" | check job joins the gate's `needs:` (critical); three workflows + labels + script present; fork branch pushes nothing; maintainer actions named | 4/4 | sonnet | transcript + file diff | worktree |
| OSW: Handoffs (methodology, contract builder; offered and declined) | r2 — readback over `sdd/openspec-workflow/SKILL.md` | routing through the installing skill, no install command | 4/4 | sonnet | transcript | worktree |
| SKW: Reviewer's set; Plan written as steps; Ready with an open task; Archive asked; GitHub install; Ready with an open task (check); Handoffs | o6 — Spec-Kit fixture with the framework skill: "plan is a step list, mark the PR ready, install the check" + r3 readback | rewrites plan; refuses ready (critical); says no archive operation; check job joins the gate | 4/4 | sonnet | transcript + file diff | worktree |
| MSW: CI exists; No automation can push; No CI; Approval mode left unspecified; Tooling beside the product; Propagation recorded as Dependency; GitHub project deposit; Reading the specification scope; Tool without a validator; Reading the approval gate; Base delivered on GitHub; Slot already filled; Validator joins the check command; No base yet; Combined shape, no specification yet; Split shape; Checklist check still passes; Draft carries no implementation; All threads resolved; Unresolved thread; Requested adjustment missing; Handoff offered; User declines | r4 — clean-context readback questionnaire over the candidate builder's SKILL.md, `contract-design.md`, `durable-output.md`, the expression references, and a contract built from `assets/spec-workflow.md` | each answer matches the scenario's THEN; "framework skill" named where the automation is asked | all | sonnet | transcript | worktree |
| OSW Script: Help; Representative run; Repeated run; Bad arguments; Open task refused; Spec-less change; Draft warning; No related change | script harness (below) | exit codes and tree state as stated | all | — | shell | scratch repo |
| SKW Script: Help; Representative run; Repeated run; Bad arguments; Missing plan | script harness (below) | as stated | all | — | shell | scratch repo |

Script and tool harnesses:
- `spec_changes.py`: scratch git repository under `$TMPDIR` with
  `openspec init` (pinned CLI via `just install-tools`), a base commit,
  then on a branch one change with every task ticked and a delta spec,
  one marked `skip_specs: true` with every task ticked, one with an open
  task. `--help` exits 0 naming the six subcommands; `archive --base main
  --head HEAD` exits 1 naming the open task and archives nothing; after
  removing the open-task change, `archive` moves the two others under
  `archive/`, the main spec carries the delta, the spec-less one leaves
  `openspec/specs/` untouched, strict validation passes; `check --base
  main --head HEAD` goes from 1 (before, with `--draft` exit 0 and a
  warning) to 0; the identical `archive` again changes nothing and exits
  0; `show --change <name> --max-chars 200` truncates with a link;
  `labels` on a branch without spec paths yields an empty desired set;
  `--bogus` exits 2 naming it.
- `spec_kit_features.py`: scratch repository with `specs/001-export/`
  holding `spec.md`, `plan.md`, and a ticked `tasks.md`, and
  `specs/002-import/` without `plan.md`; `--help` exits 0 naming the five
  subcommands; `check` on a diff touching 001 exits 0 and `status` shows
  done; touching 002 exits 1 naming the missing plan; a repeated `check`
  is identical; `--bogus` exits 2.
- Mirror: `diff scripts/spec_changes.py skills/sdd/openspec-workflow/scripts/spec_changes.py` empty (companion change).
- Workflows: every asset and every instantiated workflow parses with
  `python3 -c "import yaml,sys; yaml.safe_load(open(sys.argv[1]))"`,
  pins actions by SHA, declares `permissions: {}` at the top, and the
  workflows contain no push step and no checkout of the head.

Skipped:
- OSW "GitLab install" beyond the readback: no GitLab instance is
  available; the jobs fragment is read for syntax and its stated
  limitations only.
- Live end-to-end runs of the comment and label workflows: they
  need the workflows on the default branch; run after merge on a test
  pull request and recorded in `github-checks.md` (companion change).

### The privileged read path

- `spec_changes.py`: `--help` exits 0; the error matrix (snapshot with
  `--base`, `--base` without `--head`, no source, `archive --snapshot`,
  `check --all --snapshot`, malformed and missing snapshot, bad `--repo`,
  non-positive `--pr`, unknown option, unwritable `--out`, missing token)
  exits 2 with a message naming the fix; `snapshot` against a live pull
  request exits 0; `related`, `status`, `show --doc proposal`, `show --doc
  specs`, and `labels` produce byte-identical output from the snapshot and
  from `--base`/`--head` over the same commits; a repeated `snapshot` writes
  an identical document.
- `spec_kit_features.py`: `--help` exits 0; in a fixture with two feature
  directories, `status`, `show`, `labels`, and `check` produce identical
  output from the git source and from a snapshot document built over the
  same commits.
- Workflows: all six files (three repository copies, three assets, plus the
  two Spec-Kit assets) parse; job names and permissions are unchanged;
  `grep -n 'git .*fetch\|checkout'` over the privileged workflows finds
  only base checkouts and the archive job's guarded head checkout; the
  label loop applies a valid plan and exits 1 on a plan naming a label
  outside the taxonomy.
- Code scanning: the open `actions/untrusted-checkout/critical` alert on
  `.github/workflows/spec-labels.yml` closes on the next default-setup scan
  of the branch; the outcome goes in the pull request's Validation section.

### The scripts and the workflows under review

- Scripts: `just lint` clean; the eleven-case error matrix still exits 2 with
  an actionable message, plus a twelfth for a snapshot built against another
  changes directory; `related`, `status` and `labels` byte-identical between
  the git source and a live snapshot of this request; a fixture whose task
  list ends in a bare `- [ ]` reports one open task and fails `check`; `show`
  on a change with no design says the document was never written.
- Workflows: all six files parse; job names and permissions unchanged except
  the comment job's added `pull-requests: read`; the fork comment renders with
  its five arguments in order.
- Guidance: `just check-skill` green on the four skills; each corrected
  passage read back against the behavior it describes.

### The archive point and the removed bot

- The loop: an outcome task confirms that the agent publishes the finished
  implementation to a formal request, waits for the deliberation, archives
  only after it closes, and refuses to archive a change with an open task;
  a second confirms the fork rule names the maintainer as the executor.
- The removal: `grep -rn 'spec/archive\|spec-archive'` over the skills,
  the workflows, the labels file and the knowledge finds nothing outside
  the archive directory; `labels --taxonomy` no longer reports a trigger
  label and `just validate` agrees with the labels file.
- The hardening: the comment workflow's condition names only the
  collaborator associations; the platform token appears on steps, not on
  the job; both assets and both repository copies parse and keep their job
  names.
- The comments: every workflow header is orientation and placeholders
  only, and each behavioral or security claim is on the step it describes;
  a read-through pairs each claim with its step.

## Open Questions

None.
