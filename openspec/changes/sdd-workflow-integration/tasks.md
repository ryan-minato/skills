## 1. Skill and tool files

- [ ] 1.1 Rework `meta-workflow-design` (SKILL.md, references/durable-output.md, management-model.md, rename assets/project-workflow.md → assets/platform-workflow.md) so the summary uses the model and the deposit uses platform objects at `<platform>-workflow.md`, the platform is a required question, the model-noun check replaces the platform-word check, and a specification-only draft is work in progress; verify `just check-skill` — closes meta-workflow-design "Workflow design request", "Platform expression request", "GitHub-hosted project", "Local repository without a remote", "Draft carrying only the change record"
- [ ] 1.2 Invert `meta-spec-workflow`'s deposit rule (SKILL.md step 5, references/durable-output.md, assets/spec-workflow.md header and vocabulary) and read `<platform>-workflow.md` in step 1; verify `just check-skill` — closes meta-spec-workflow "Harness alignment request", "Writing a specification", "GitHub project deposit"
- [ ] 1.3 Add the platform-verb deposit rule to `meta-agent-authority` (SKILL.md deposit step, references/durable-output.md, assets/agent-authority.md); verify `just check-skill` — closes meta-agent-authority "Reading the deposited policy"
- [ ] 1.4 Point `meta-github-workflow` and `meta-gitlab-workflow` at the platform-worded deposits (SKILL.md, references/semantic-mapping.md, decision-tree.md, durable-harness.md, spec-expression.md knowledge deposit, planning references, assets pointers) and remove the separate planning file; verify `git grep -n 'project-workflow.md\|planning\.md' skills/meta` shows only intended hits — closes meta-github-workflow and meta-gitlab-workflow "Deposit already present"
- [ ] 1.5 Rewrite `skills/engineering/spec-driven-development/SKILL.md` (description, "Specs and tracked work", loop steps 2–4 and 7, "What specification review examines", reference pointer, harness-alignment fallback, gotchas); verify `just check-skill` reports no error — closes SDD "Library with downstream consumers", "Feature-driven application", "Where the spec is reviewed", "Planning before publication", "Tasks offered for review", "Automation cannot push", "Automation available", "Acceptance pasted into the issue", "Lifecycle question", "Harness build request"
- [ ] 1.6 Write `references/tracked-work-lifecycle.md` (shapes, timing table, review scope, archive mode with automation rules, template guidance, per-tool notes) with no path outside the skill and no tool command in prose; verify `just check-skill` — closes SDD "Handoff offered", "User declines"
- [ ] 1.6a Add `scripts/archive_completed_changes.py` to `spec-driven-development` (stdlib only, `--help`, `--dry-run`, exit 2 on bad arguments, non-interactive archive verified from the CLI's help) and name it from the reference's archive-mode section; verify the script harness of the verification plan and `just lint` — closes SDD "Help", "Representative run", "Nothing completed", "Repeated run", "Bad arguments"
- [ ] 1.7 Add questions 5–7 (shape, archive mode, author) to `meta-spec-workflow/SKILL.md` step 2, extend step 6's removal simulation, add the deliberation gotcha; verify `just check-skill` — closes meta-spec-workflow "Propagation recorded as Dependency", "No automation can push"
- [ ] 1.8 Extend `meta-spec-workflow/assets/spec-workflow.md` (approval gate scope, change request shape, archive mode, lifecycle recording points, tracked-work timing) and `references/durable-output.md`; verify a clean-context readback answers the gate scope — closes "Reading the approval gate"
- [ ] 1.9 Update `meta-spec-workflow/references/openspec.md`, `spec-kit.md`, `kiro.md`, `committed-documents.md` with archive timings, integration-branch state, and the project-defined completion rule; verify `just check-skill` — closes "Spec-Kit with automated archiving"
- [ ] 1.10 Add the gate-precedes-admission and H1 clauses to `meta-agent-authority/SKILL.md` and `references/authority-profiles.md`; verify `just check-skill` — closes "H1 offered with an agent-authored specification", "Agent asked to approve and ready"
- [ ] 1.11 Add the change-request-shape and archive-mode section to both `references/spec-expression.md`, make Take-work shape-aware in both `assets/project-skill.md`, add the spec phase to `issues-and-prs.md` and `work-items-and-mrs.md`; verify `just check-skill` — closes "Combined shape, no specification yet", "Split shape"
- [ ] 1.12 Split the SPEC_CHECK placeholder into two items in `assets/pull-request-template.md` and `assets/mr-template-default.md`; verify the checklist workflow's parsing against a body built from the template — closes "Checklist check still passes"
- [ ] 1.13 Add `assets/workflow-spec-archive.yml` and `assets/gitlab-ci-spec-archive.yml` (wrappers calling the project's `scripts/archive_completed_changes.py`, routed to the SDD skill by role through the installer), the bypass/token notes, and the design-only guidance for other tools; verify the wrapper read-through — closes "OpenSpec with automation", "Spec-Kit with automation"
- [ ] 1.14 Update both `references/durable-harness.md` sync registers; verify `just check-skill`

## 2. External impact

- [ ] 2.1 Update `skills/meta/CONTEXT.md`, `skills/meta/README.md`, `README.zh.md`, the sentence in `skills/scaffold/CONTEXT.md`, `skills/engineering/CONTEXT.md` disambiguation, and the `engineering` README pair row; verify `just validate`
- [ ] 2.2 Confirm the old `project-workflow.md` asset name has no remaining reference in `skills/`; verify `git grep -n project-workflow.md skills` is empty

## 3. Tests

- [ ] 3.1 Run the SDD trigger cases (two loads, two non-loads) at the floor tier with the observation method recorded — closes SDD "Lifecycle question", "Harness build request"
- [ ] 3.2 Run the SDD outcome case (semver library) with an independent clean-context grader — closes SDD "Library with downstream consumers"
- [ ] 3.3 Run the meta-spec-workflow outcome case (dependency propagation) with an independent grader — closes meta-spec-workflow "Propagation recorded as Dependency"
- [ ] 3.4 Run the script harness for `spec-driven-development/scripts/archive_completed_changes.py` and the wrapper read-throughs — closes SDD "Help", "Representative run", "Nothing completed", "Repeated run", "Bad arguments"; meta-github/gitlab "OpenSpec with automation"
- [ ] 3.5 Perform the clean-context readbacks for the skipped domains and record every skipped case with its reason for the pull request's Validation section

## 4. Finish

- [ ] 4.1 Run `just check`, write the results to the pull request's Validation section linking the verification plan, and archive the change in-request under the rule in force on `main` (together with the companion repository change), then `just spec-validate`
