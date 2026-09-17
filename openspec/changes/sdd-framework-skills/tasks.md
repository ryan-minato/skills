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

## 4. Finish

- [x] 4.1 Run `just check`, write the results to the pull request's Validation section linking the verification plan, archive both changes inside this pull request by hand
