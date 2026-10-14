## 1. Skill and repository files

- [x] 1.1 `meta/meta-github-workflow` and `meta/meta-gitlab-workflow` descriptions: add the organization (GitLab: group) configuration-repository case, under 900 characters each. Verify `just check-skill` on both skills — closes GH and GL Trigger: description (all scenarios).
- [x] 1.2 `meta-github-workflow` new `references/config-as-code.md`, covering: minimal declarations with reasons; no backend, with an `import` per resource; reset to defaults; the first creation of ID-only resources; `## Removal is not deletion`; `## Who applies` with `### Local credentials` (just and make examples, `gh auth token` passed as `GITHUB_TOKEN`, the provider's legacy-only fallback with its date, `gh auth refresh -s admin:org`, plan `-out` → guard → apply of that file, no token in files or echoed lines); `## Review is the check`; `## Ownership`; known perpetual diffs with their dates. Then the SKILL.md step 1 tool probe, the step 3 routing row, and the step 4 delivery sentences. Verify `just check-skill skills/meta/meta-github-workflow` — closes GH: Tool installed; Unreadable change rejected; Tool absent; Setting nobody needs; Declaration removed; Authoritative resource requested; Project CI for policy.tf; Asked to apply from project CI; Recipes for a justfile project; Provider fallback considered; Login lacks the organization scope; Organization baseline exists; Security switch settable only per repository.
- [x] 1.3 `meta-github-workflow` new assets:
  - `policy.tf`, with placeholders only inside string literals;
  - `tf_plan_guard.py`;
  - `workflow-policy-check.yml`, running fmt and validate with no secret and `contents: read`;
  - `platform-settings.md`, gaining the "Described by" column and `## Harness options`.

  Verify `just check-skill`, `just lint`, the YAML parse, and the companion's `just tf-validate` — closes GH: tf delivered; Replace in the plan.
- [x] 1.4 `meta-github-workflow` new `references/config-repo.md` and assets `config-repo.tf` and `workflow-config-repo.yml`. The reference covers:
  - eligibility, and why the `.github` repository is not used;
  - the three sync modes and the default;
  - the CI gate;
  - the GitHub App installation token, with the key stored on the repository or its environment only and never as an organization secret;
  - the local modes, pointing to `### Local credentials`.

  Also add the routing row and the `decision-tree.md` sync-mode question. Verify `just check-skill`, the YAML parse, and `just tf-validate` — closes GH: Default sync mode chosen; Fully local sync mode chosen; Credential storage requested at organization level; Personal account or the .github repository.
- [x] 1.5 `meta-github-workflow` labels:
  - `references/planning-and-goals.md` gains the label sync mode and the prune option, with defaults;
  - new `assets/workflow-labels-sync.yml`, with `issues: write` only, no secret, and no prune unless chosen;
  - `decision-tree.md` gains both options;
  - SKILL.md step 4's delivered-script list adds the plan guard.

  Also add the pointer sentences in `durable-harness.md`, `rules-and-protection.md`, and `security-and-ownership.md`. Verify `just check-skill` and the YAML parse — closes GH: Default label sync; Pruning chosen; MODIFIED Label taxonomy committed.
- [x] 1.6 `meta-gitlab-workflow` new `references/config-as-code.md`, mirroring 1.2 for GitLab:
  - no GitLab-managed state, with the reason;
  - `### Local credentials` reads glab's stored login for the host at each run into `GITLAB_TOKEN`, with `GITLAB_BASE_URL` on a self-managed instance. The extraction command is verified against current glab documentation and a logged-in glab, with the date recorded. The recipe fails, naming glab, when no token is found. The Make `$(shell …)` trap is covered.

  Then the step 1 probe, the routing row, and the step 4 sentences. Verify `just check-skill skills/meta/meta-gitlab-workflow` — closes GL: Tool installed; Unreadable change rejected; Tool absent; GitLab-managed state offered; Declaration removed; Project pipeline for policy.tf; Group token offered to the project; Recipes for a Makefile project; Make expansion would print the token; Group baseline exists.
- [x] 1.7 `meta-gitlab-workflow` new assets: `policy.tf`, `tf_plan_guard.py`, `gitlab-ci-policy-check.yml`, and the `platform-settings.md` additions. Verify `just check-skill`, `just lint`, the YAML parse, and `just tf-validate` — closes GL: Replace in the plan.
- [x] 1.8 `meta-gitlab-workflow` new `references/config-repo.md` and assets `config-repo.tf` and `gitlab-ci-config-repo.yml`. The reference covers:
  - eligibility: a group only, and no configuration repository for a personal namespace;
  - the sync modes;
  - the protected environment with a manual apply job;
  - the group access token, with its tier and 365-day facts verified and dated, stored as a masked, protected variable of this project only, with the rotation owner recorded;
  - GitLab.com Free, where only the local mode is available and no personal access token is offered as a substitute;
  - group labels in tf.

  Also add the routing row and the `decision-tree.md` question. Verify `just check-skill`, the YAML parse, and `just tf-validate` — closes GL: Default sync mode on a paid or self-managed instance; GitLab.com Free; Group labels; Personal namespace.
- [x] 1.9 `meta-gitlab-workflow` labels:
  - new `assets/sync_labels.py` (dry run by default, `--apply`, `--prune` only when given) and `assets/gitlab-ci-labels-sync.yml`, which uses a project access token in a masked, protected project variable with the least role that manages labels (verified and dated) and never `CI_JOB_TOKEN`;
  - `references/planning-and-labels.md` gains the sync and prune options, plus the manual command for GitLab.com Free;
  - `decision-tree.md` gains both options;
  - SKILL.md step 4's delivered-script list;
  - the `security-and-governance.md` and `durable-harness.md` pointers.

  Verify `just check-skill`, `just lint`, and the YAML parse — closes GL: Default label sync on a self-managed instance; CI job token considered; GitLab.com Free label sync; MODIFIED Label file committed.

## 2. External impact

- [x] 2.1 The `skills/meta/README.md` and `README.zh.md` rows for both builders state the new capability, with content identical across the pair. Verify `just validate`.
- [x] 2.2 The companion change `config-as-code-harness` is in place, and `just tf-validate` passes on every new `.tf` asset. Verify `just check`.
- [x] 2.3 There is no change to symlinks, `marketplace.json`, `meta/CONTEXT.md`, `meta-agent-authority`, or `meta-harness-building`. Verify `just validate` and `git diff --stat origin/main...HEAD -- .claude-plugin skills/meta/CONTEXT.md skills/meta/meta-agent-authority skills/meta/meta-harness-building .agents/skills`.

## 3. Tests

- [x] 3.1 Plan-guard harness, run on both copies: help; a no-op and imported-update plan; `["delete","create"]`, `["create","delete"]`, and `["delete"]`; the plan after a declaration is removed; an undeclared-attribute update that is listed; a create without `importing` that is listed; malformed JSON; a repeated run; `--bogus` — closes GH and GL: Replace in the plan; Declaration removed (guard part).
- [x] 3.2 GitLab `sync_labels.py` harness against a stub labels API: a dry run; `--apply` without pruning; `--prune` only when given; a repeated all-skip run; an object where a list is expected; help and `--bogus` — closes GL: Label file committed (script part).
- [x] 3.3 Asset hygiene: every new workflow and CI asset parses as YAML; `grep -rn '{{[A-Z]'` finds placeholders only inside string literals in `.tf` assets and where the references list them; `just tf-validate` passes.
- [ ] 3.4 Trigger cases T1–T8 — closes GH and GL Trigger: description.
- [ ] 3.5 Outcome O1 — closes GH: Tool installed; Unreadable change rejected; Setting nobody needs; tf delivered.
- [ ] 3.6 Outcome O2 — closes GH: Tool absent.
- [ ] 3.7 Outcome O3 — closes GH: Project CI for policy.tf; Asked to apply from project CI.
- [ ] 3.8 Outcome O4 — closes GH: Organization baseline exists; Security switch settable only per repository.
- [ ] 3.9 Outcome O5 — closes GH: Authoritative resource requested; Default label sync; Pruning chosen.
- [ ] 3.10 Outcome O6 — closes GH: Default sync mode chosen; Credential storage requested at organization level.
- [ ] 3.11 Outcome O7 — closes GH: Fully local sync mode chosen.
- [ ] 3.12 Outcomes O8 and O9 — closes GH: Personal account or the .github repository.
- [ ] 3.13 Outcome O10 — closes GL: Tool installed; Unreadable change rejected; GitLab-managed state offered.
- [ ] 3.14 Outcome O11 — closes GL: Project pipeline for policy.tf; Group token offered to the project.
- [ ] 3.15 Outcome O12 — closes GL: Group baseline exists.
- [ ] 3.16 Outcome O13 — closes GL: Default sync mode on a paid or self-managed instance; Group labels.
- [ ] 3.17 Outcome O14 — closes GL: GitLab.com Free; GitLab.com Free label sync.
- [ ] 3.18 Outcome O15 — closes GL: Default label sync on a self-managed instance; CI job token considered; Label file committed.
- [ ] 3.19 Outcome O16 — closes GH: Recipes for a justfile project; Provider fallback considered; Login lacks the organization scope.
- [ ] 3.20 Outcome O17 — closes GL: Recipes for a Makefile project; Make expansion would print the token.
- [ ] 3.21 Readback cases — closes GL: Personal namespace; GL: Tool absent; GH and GL MODIFIED: Durable project skill with CI diagnosis; SemVer not chosen.
- [ ] 3.22 Record the skipped cases (live plan and apply; live workflow and pipeline runs) and their reasons for the Validation section.

## 4. Finish

- [ ] 4.1 Run `just check`, write the results to the pull request's Validation section linking the verification plan, and leave the record unarchived until the maintainer closes the implementation deliberation.
