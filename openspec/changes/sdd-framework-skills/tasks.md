## 1. The sdd catalog and the moved methodology skill

- [x] 1.1 `spec-driven-development` moves from `engineering` into `sdd` with its spec domain (directory move, title line corrected by hand); verify `just spec-validate` and `git log --follow` on the moved spec
- [x] 1.2 `SKILL.md` rewritten platform- and framework-agnostically: the loop with the approval package, the specification-quality rules, the contract facts with their defaults (the implementer archives inside the request once the deliberation closes), and the two handoffs; verify `just check-skill skills/sdd/spec-driven-development` — closes SDD: every Behavior and Handoff scenario
- [x] 1.3 `references/adoption-decision.md` and `references/tracked-work.md` split by load condition, each pointer naming the branch that reaches it; `references/tracked-work-lifecycle.md` deleted; verify a read of each load condition against the branch it serves

## 2. The framework skills

- [x] 2.1 `sdd/openspec-workflow/`: `SKILL.md`, `references/{github,gitlab}.md`, `assets/github/{job-spec-check.yml,workflow-spec-command.yml,workflow-spec-labels.yml,labels-spec.json}`, `assets/gitlab/{ci-spec-jobs.yml,labels-spec.json}`, `scripts/spec_changes.py`; no asset archives, commits, or pushes; symlink, README rows, marketplace; verify `just check-skill` and the script's `--help` — closes OSW: every Trigger, Behavior and Handoff scenario
- [x] 2.2 `sdd/spec-kit-workflow/`: the same shape for Spec-Kit, with no archive operation and the contract's lock declaration named; verify `just check-skill` and the script's `--help` — closes SKW: every Trigger, Behavior and Handoff scenario
- [x] 2.3 Both scripts take their head from git plumbing or from a snapshot; the `snapshot` subcommand reads the request through the platform's API under file, byte, and call caps and fails rather than report on a partial read; `show` tells a document never written from one the source does not hold; `archive` accepts only the git source, refuses an open task, and refuses a dirty tree under any path the tool writes; verify `just lint`, the error matrix, source parity on a live request, and the fixture cases — closes both Script requirements
- [x] 2.4 The privileged assets carry one rule: base checkout, the head read through the API, no fetch and no checkout of it; the command admits only the collaborator associations; the platform token sits on the steps that use it; the label plan is checked against the literal taxonomy before it is applied; verify the files parse, job names and permissions, and the greps
- [x] 2.5 Every asset and script keeps its behavioral and security claims next to the step or function they constrain, headers carrying orientation and placeholders only; verify a read-through pairing each claim with its step

## 3. The builder and the authority skill

- [x] 3.1 `meta/meta-spec-workflow` framework-agnostic: the four tool references, the archive assets, and the archive script deleted; `references/contract-design.md` replaces `tracked-work-lifecycle.md`; the questioning round settles the approval package and the archive executor; step 3 and step 7 hand adoption and automation to the framework skill; the deposited contract and the slot texts carry the package, the executor, and the freeze in platform vocabulary with no framework name, command, or script; verify `just check-skill` and the vocabulary grep — closes MSW: every Trigger, Behavior and Handoff scenario
- [x] 3.2 `meta/meta-agent-authority` reads the archive executor as the freeze the approval applies to, never a job after merge; verify `just check-skill` — closes the authority Behavior scenarios

## 4. Catalog rows and residue

- [x] 4.1 `skills/meta/README.md` + `README.zh.md`, `skills/sdd/README.md` + `README.zh.md`, and `skills/meta/CONTEXT.md` describe the skills as they ship; verify `just validate` and a read of each pair
- [x] 4.2 The companion change `sdd-framework-skills-harness` carries this repository's copy; verify `just validate` and the script mirror diff empty
- [x] 4.3 `grep -rn 'engineering/spec-driven-development\|tracked-work-lifecycle\|archive_completed_changes\|spec/archive\|spec-archive'` over `skills/`, `.agents/`, `.github/`, `scripts/`, `justfile`, the root documents and `openspec/specs` is empty outside `openspec/changes/archive/`

## 5. Tests

- [x] 5.1 Trigger cases t1–t16 in the fixture project — closes SDD, OSW, SKW, MSW Trigger scenarios
- [x] 5.2 Outcome o1 and readback r1 — closes the SDD approval-package and design scenarios
- [x] 5.3 Outcome o3 — closes SDD: Reviewer's set under a spec-first kit, framework-skill Handoff offered, User declines
- [x] 5.4 Outcome o4 and o5 and readback r2 — closes the OSW Behavior and Handoff scenarios, including the refusal to install an archive bot
- [x] 5.5 Outcome o6 and readback r3 — closes the SKW Behavior and Handoff scenarios, including what locks the specification
- [x] 5.6 Readback r4 — closes the MSW scenarios
- [x] 5.7 Outcome o7: the agent publishes the finished implementation for deliberation, archives only after it closes, and names the maintainer as a fork's executor — closes the archive-freeze scenarios of SDD, OSW and the authority skill
- [x] 5.8 Script harnesses for both scripts, the workflow read-through, and the skipped cases recorded for the pull request's Validation section

## 6. Finish

- [x] 6.1 Run `just check`, write the results to the pull request's Validation section linking the verification plan, and archive both changes inside this pull request after the deliberation closes
