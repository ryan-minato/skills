## 1. Skill and repository files

- [x] 1.1 `justfile`: an `opentofu_version` pin, and an `install-tools` step that installs that version through OpenTofu's official standalone method with its checksum verified (the method checked against OpenTofu's installation documentation, dated in the recipe comment). Verify that `just install-tools` followed by `tofu version` prints the pin, and that a repeated `install-tools` leaves the same binary.
- [x] 1.2 `justfile`: a `tf-validate` recipe that runs `tofu fmt -check` on every `skills/*/*/assets/**/*.tf`, then `init -backend=false` and `validate` on a temp copy, failing with the asset named; `check` depends on it. Verify that it passes on the `config-as-code` assets, that a broken scratch copy fails naming the asset, and that `git status --porcelain` is unchanged after a run.
- [x] 1.3 `scripts/validate_harness.py`: reject an OpenTofu version hard-coded in `checks.yml` or `devcontainer.json`, beside the OpenSpec rule. Verify that a scratch hard-coded version fails `just validate` naming the file, then revert.
- [x] 1.4 `.github/workflows/checks.yml`: no change, because the recipes call `.tools/bin/tofu` by its path. Verify that `checks / quality` runs `tf-validate` on the pull request.

## 2. External impact

- [x] 2.1 Add a `tf-validate` row to the `AGENTS.md` Validation table, name the recipe in `ARCHITECTURE.md` `## Quality Gates`, add OpenTofu to the `checks / quality` row of `.agents/knowledge/github-checks.md`, and add an `opentofu_version` register row to `.agents/knowledge/harness-maintenance.md`. Verify with a read-through and `just validate`.

## 3. Tests

- [x] 3.1 Every Placement proof in `design.md` ran and its result is recorded.

## 4. Finish

- [x] 4.1 `just check` passes locally and `checks / quality` passes on the pull request; the results are recorded in the Validation section.
