## 1. Skill and repository files

- [x] 1.1 Create `skills/machine-learning/CONTEXT.md`, `README.md`, and `README.zh.md` (content-identical), add the `machine-learning` bullet to `ARCHITECTURE.md` `## Catalogs`, the rows and install-example mention to the root `README.md` and `README.zh.md`, `catalog/machine-learning` to `.github/labels.json`, and the option to the three issue forms; verify `just validate` passes and the label dry run lists one label to create — proves Catalog scaffold, Architecture map, Root README pair, Labels and issue forms
- [x] 1.2 Add the `machine-learning` plugin entry to `.claude-plugin/marketplace.json` in the same commit as the first skill and run `just gen-marketplace` after each skill; add the five symlinks; verify `just validate` passes and `git diff --exit-code .claude-plugin/marketplace.json` is empty after the generator — proves Marketplace entry, Symlinks
- [x] 1.3 Run the entropy review the register requires after a catalog change and record its last-run date in `.agents/knowledge/harness-maintenance.md`; verify `just validate` passes — proves Entropy review

## 2. External impact

- [x] 2.1 Confirm `scripts/` is untouched (`git diff --stat origin/main...HEAD -- scripts` empty) and record the post-merge `sync_labels.py --apply` maintainer action in the pull request handover; verify `just validate`

## 3. Tests

- [x] 3.1 Run every command of the verification plan (validate, label dry run, marketplace generator diff, symlink listing, README pair read-through) and note each result — proves every What Changes bullet
- [x] 3.2 Record skipped cases (expected none) for the pull request's Validation section

## 4. Finish

- [x] 4.1 Run `just check`, write the results to the pull request's Validation section linking this plan, and archive this change in-request beside `machine-learning-catalog`, then `just spec-validate`
