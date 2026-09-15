## 1. Skill and repository files

- [ ] 1.1 `experiment-provenance`: `assets/run_manifest.py` (`sha256_file`, `no_image_digest`, docstring) in the same commit as the scaffold's copy; `SKILL.md` manifest paragraph, run-equation validator sentence, tracker-alias sentence, Gotchas reduced to two; verify `just check-skill skills/machine-learning/experiment-provenance` and `cmp` of the two copies — closes Empty image digest declared, Run record contents
- [ ] 1.2 `research-workflow`: `assets/research-spec.md` bare skeleton with unchanged headings; Gotchas reduced to one; verify `just check-skill` and the heading-list diff against the scaffold's template
- [ ] 1.3 `experiment-code-conventions`: `references/config-surface.md` mechanism block, `references/vendoring-research-code.md` Gotchas dissolved into the procedure and the modes table, `references/tensor-tests-and-docs.md` positive rule only, `SKILL.md` Gotchas reduced to three; verify `just check-skill`
- [ ] 1.4 `training-instrumentation`: `SKILL.md` `## Health measurements in the loop` in place of the PyTorch skeleton, Gotchas reduced to one; `references/model-health-metrics.md` one sentence in place of the Adam block; verify `just check-skill`
- [ ] 1.5 `training-diagnosis`: `SKILL.md` evidence-chain pointer, one resident numerical-instability rule, Gotchas reduced to four; verify `just check-skill`

## 2. External impact

- [ ] 2.1 Confirm `cmp` of the two manifest copies is silent and the two research-spec heading lists are identical; `just gen-marketplace` then `git diff --exit-code .claude-plugin/marketplace.json`; verify `just validate`

## 3. Tests

- [ ] 3.1 Run the empty-digest harness (declared empty, set, unset) — closes Empty image digest declared
- [ ] 3.2 Run the readback over the five skills, every scenario, and every removed Gotchas bullet; fix any GAP and rerun
- [ ] 3.3 Record skipped cases and their reasons for the pull request's Validation section

## 4. Finish

- [ ] 4.1 Run `just check`, write the results to the pull request's Validation section linking the verification plan, run `just spec-validate`; archiving waits for the maintainer's instruction
