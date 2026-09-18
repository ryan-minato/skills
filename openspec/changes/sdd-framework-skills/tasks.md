## 1. Skill and repository files

- [x] 1.1 Move `skills/engineering/spec-driven-development` to `skills/sdd/spec-driven-development` with its symlink retargeted, the `engineering` and `sdd` README pairs and `CONTEXT.md` files adjusted, `skills/machine-learning/CONTEXT.md` pointers, and `just gen-marketplace`; verify `just validate` — closes nothing by itself (placement precondition)
- [x] 1.2 `sdd/spec-driven-development/SKILL.md`: rewrite as the framework-agnostic body — definition, the loop with the package steps, specification quality, tool commands first, the approval package and the design's role, contract-or-default reading with the by-hand executor, the framework-skill handoff, the builder handoff, gotchas; description under 950 characters; verify `just check-skill skills/sdd/spec-driven-development` — closes SDD: Design timing question, Tasks requested before approval, Package completed, Waiting after publication, Tasks offered for review, Design offered as a step list, Design requested as a step list, Wording change, Constraint with a private detail, Reviewer's set under a spec-first kit, Tool generated the task list with the specification, No contract, Project with a contract, Handoff offered, User declines (both handoffs), Tool scaffolds the change, Validator available, Tool without a validator, Hand-written record offered
- [x] 1.3 `sdd/spec-driven-development/references/adoption-decision.md` (levels, when it pays, approach families, rejected alternatives) and `references/tracked-work.md` (contract facts, shapes, approval modes, what the gate examines, reconciliation checklist, request body, archiving before ready); delete `references/tracked-work-lifecycle.md`; repoint `references/adopting-existing-code.md`; verify `just check-skill` — closes SDD: Where the spec is reviewed, Push to the record after a blocking approval, Narrowing decided by the gate owner, All threads resolved, Unresolved thread, Requested adjustment missing from the record, Adjustment requested mid-discussion, Draft body in the specification phase, Implementation offered for the draft, Contract names the label, Fork, Review finds a defect after archiving
- [x] 1.4 `sdd/openspec-workflow/` new skill: `SKILL.md`, `references/github.md`, `references/gitlab.md`, `assets/github/{job-spec-check.yml,workflow-spec-archive.yml,workflow-spec-command.yml,workflow-spec-labels.yml,labels-spec.json}`, `assets/gitlab/{ci-spec-jobs.yml,labels-spec.json}`, `scripts/spec_changes.py`; symlink, README rows, marketplace; verify `just check-skill skills/sdd/openspec-workflow` and `python3 skills/sdd/openspec-workflow/scripts/spec_changes.py --help` — closes OSW: every Trigger, Behavior, and Handoff scenario
- [x] 1.5 `sdd/spec-kit-workflow/` new skill: `SKILL.md`, `references/github.md`, `references/gitlab.md`, `assets/github/{job-spec-check.yml,workflow-spec-command.yml,workflow-spec-labels.yml,labels-spec.json}`, `assets/gitlab/{ci-spec-jobs.yml,labels-spec.json}`, `scripts/spec_kit_features.py`; symlink, README rows, marketplace; verify `just check-skill skills/sdd/spec-kit-workflow` and the script's `--help` — closes SKW: every Trigger, Behavior, and Handoff scenario
- [x] 1.6 `meta/meta-spec-workflow`: delete `references/{openspec,spec-kit,kiro,committed-documents}.md`, `assets/github/workflow-spec-archive.yml`, `assets/gitlab/ci-spec-archive.yml`, `scripts/archive_completed_changes.py`; rename `references/tracked-work-lifecycle.md` to `references/contract-design.md` and rewrite it; edit `SKILL.md` (questions 2, 3, 6; steps 3 and 7; gotchas; description; compatibility), `references/durable-output.md`, `references/github-expression.md`, `references/gitlab-expression.md`, `assets/spec-workflow.md`, `assets/*/template-lines.md`, `assets/*/project-skill-steps.md`; verify `just check-skill skills/meta/meta-spec-workflow` and `grep -rniE 'openspec|spec-kit|kiro|archive_completed' skills/meta/meta-spec-workflow` returns only contract-fact mentions — closes MSW: every Trigger, Behavior, and Handoff scenario

## 2. External impact

- [x] 2.1 `skills/meta/README.md` + `README.zh.md` row for `meta-spec-workflow`; `skills/sdd/README.md` + `README.zh.md` rows for the three skills; `skills/meta/CONTEXT.md` archive-job wording; verify `just validate` and a read of each pair
- [x] 2.2 Companion change `sdd-framework-skills-harness` implemented (catalog scaffold, this repository's automation, contract, project skill, templates, knowledge); verify `just validate` and `diff scripts/spec_changes.py skills/sdd/openspec-workflow/scripts/spec_changes.py` empty
- [x] 2.3 `grep -rn 'engineering/spec-driven-development\|tracked-work-lifecycle\|archive_completed_changes' skills/ .agents/ .github/ scripts/ justfile *.md openspec/config.yaml openspec/specs` empty outside `openspec/changes/archive/`

## 3. Tests

- [x] 3.1 Trigger cases t1–t16 in the fixture project — closes SDD, OSW, SKW, MSW Trigger scenarios
- [x] 3.2 Outcome tasks o1, o2 — closes SDD package and design scenarios
- [x] 3.3 Outcome task o3 — closes SDD: Reviewer's set under a spec-first kit, framework-skill Handoff offered, User declines
- [x] 3.4 Readback r1 — closes the SDD MODIFIED scenarios listed in the plan
- [x] 3.5 Outcome tasks o4, o5 and readback r2 — closes the OSW Behavior and Handoff scenarios
- [x] 3.6 Outcome task o6 and readback r3 — closes the SKW Behavior and Handoff scenarios
- [x] 3.7 Readback r4 — closes the MSW scenarios listed in the plan
- [x] 3.8 Script harness for `spec_changes.py` — closes OSW Script: Help, Representative run, Repeated run, Bad arguments, Open task refused, Spec-less change, Draft warning, No related change
- [x] 3.9 Script harness for `spec_kit_features.py` — closes SKW Script: Help, Representative run, Repeated run, Bad arguments, Missing plan
- [x] 3.10 Workflow read-through (YAML parse, pinned SHAs, top-level permissions, no push on the fork branch); record skipped cases and their reasons for the pull request's Validation section

## 4. The privileged read path

- [x] 4.1 `spec_changes.py`: a head-source abstraction (git plumbing or `--snapshot`), a `snapshot` subcommand over the GitHub REST API with file, per-file, and total byte caps, `archive` restricted to the git source; verify `just lint`, `--help`, the error matrix, and parity of `status`, `show`, `labels` between the two sources on a live pull request
- [x] 4.2 `spec_kit_features.py`: the same abstraction, subcommand, and caps; verify `just lint`, `--help`, and source parity in a fixture
- [x] 4.3 `openspec-workflow` assets: labels and command workflows read the head through the snapshot and fetch nothing; the archive workflow reads a fork's head through the snapshot and checks the head out only under the literal `head.repo.full_name == github.repository` condition; the label plan is checked against the literal taxonomy before it is applied; verify YAML parse, job names, permissions, and a grep for head fetches
- [x] 4.4 `spec-kit-workflow` assets: the same for the labels and command workflows; verify YAML parse and the grep
- [x] 4.5 Both skills' `SKILL.md` and `references/github.md` state the rule (no object authored by the request reaches a privileged runner), the snapshot, the literal same-repository guard, and the caps; the verification lists gain the grep and the parity check; verify `just check-skill` on both

## 5. Script, workflow, and guidance corrections

- [x] 5.1 Scripts: a partial read of any kind fails the snapshot (cap, truncated tree, undecodable document); a snapshot built for another changes or specs directory is refused; `show` tells a document that was never written from one the source does not hold; the clean-tree guard before archiving covers every path the tool writes; an unchecked box with no text is an open task; a request touching exactly the file cap is inside it — verify `just lint`, the error matrix, source parity on a live request, and a fixture whose task list ends in a bare checkbox
- [x] 5.2 Workflows: the comment job grants `pull-requests: read` for the reads it makes; the fork instructions fetch from `upstream` because a fork clone's `origin` is the fork — verify YAML parse, job names and permissions, and a rendering of the fork comment
- [x] 5.3 Guidance: both GitLab references refuse the parent-project pipeline as a fork workaround; the archiving bullet says a GitLab label change starts no pipeline; the by-hand command takes the request's target branch; the catalog README pair, the deposited draft rule, the entrypoint pointer, the clarify fallback, and the catalog's adding-a-framework rule match what the skills do — verify `just check-skill` on the four skills and a read of each corrected passage
- [x] 5.4 Delta specs extended to cover the above and a `meta/meta-agent-authority` delta added for the archive executor it now reads — verify `just spec-validate`

## 6. The archive point and the removed bot

- [ ] 6.1 Remove the archive bot from both framework skills: the GitHub archive workflow asset, the `spec:archive` job and `SPEC_ARCHIVE_TOKEN` from the GitLab fragment, the trigger label from both label files, and every passage in `SKILL.md` and the two platform references that describes the bot — verify `grep -rn 'spec/archive\|spec-archive\|SPEC_ARCHIVE_TOKEN'` over `skills/` is empty and `just check-skill` passes
- [ ] 6.2 Move the archive after the implementation discussion in the methodology skill, `references/tracked-work.md`, both framework skills, and the builder's questioning round, contract asset, and both `project-skill-steps.md`; state that approval follows the freeze and that only specification-consistency changes are expected after it — verify a read-through of each passage and `just check-skill`
- [ ] 6.3 Record the fork executor: the maintainer archives on the contributor's branch after the deliberation, which needs maintainer edits enabled and cannot be done by the platform token; the contributor archiving is the alternative; archiving after the merge stays excluded — verify the passage exists in the framework skill, the contract asset, and the methodology's contract facts
- [ ] 6.4 Name the Spec-Kit asymmetry: the kit has no archive, so the contract must say what marks the specification locked, and no mechanism enforces it — verify the passage in `spec-kit-workflow/SKILL.md`
- [ ] 6.5 Harden the comment command: the condition admits only the collaborator associations, and the platform token descends from the job to the steps that use it, in both assets — verify both files parse, job names and permissions are unchanged, and the condition names no author clause
- [ ] 6.6 Move every behavioral and security claim next to the step or function it constrains, in the five workflow assets, the GitLab fragments, and both scripts; headers keep orientation and placeholder instructions only — verify a read-through pairing each claim with its step, and `just lint`
- [ ] 6.7 Delta specs rewritten for the new archive point, the removed bot, the collaborator-only command, and the fork executor, across the five domains — verify `just spec-validate`

## 7. Tests for the archive point

- [ ] 7.1 Outcome: an agent finishing an implementation publishes to a formal request, waits for the deliberation, and archives only after it closes; it refuses to archive a change with an open task
- [ ] 7.2 Outcome: asked how a fork's change is archived, the agent names the maintainer as the executor and never offers an after-merge job
- [ ] 7.3 Readback: a clean context reads the deposited contract and states the archive point, the executor, and the fork rule

## 8. Finish

- [x] 8.1 Run `just check`, write the results to the pull request's Validation section linking the verification plan, archive both changes inside this pull request by hand
