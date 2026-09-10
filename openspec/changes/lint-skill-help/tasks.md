## 1. Skill and repository files

- [ ] 1.1 `scripts/lint_skill.py`: add `import_yaml()`, call it first in `parse_frontmatter`, drop the module-level import, update the docstring exit codes and the argparse epilog; verify `just check-skill skills/core/great-skill-writing` and `just lint` — closes Help, Bad arguments, Missing skill path, Missing dependency, Representative run, Repeated run

## 2. External impact

- [ ] 2.1 Confirm nothing outside the script moves (description unchanged, no file added or removed); verify `just validate` and `git status --short` lists only the script and this change record

## 3. Tests

- [ ] 3.1 Run the three Trigger cases as clean-context Sonnet-class subagents in the fixture project — closes "Skill authoring request", "Human documentation (near-miss)", "Human skills (near-miss)"
- [ ] 3.2 Run the PyYAML-free harness (fresh venv interpreter): `--help`, `--bogus`, no `--skill`, missing SKILL.md path, real lint — closes Help, Bad arguments, Missing skill path, Missing dependency
- [ ] 3.3 Run the `uv run --offline` harness: representative run (text, `--json`, `--help`) and the repeated run with `diff` and `git status` — closes Representative run, Repeated run
- [ ] 3.4 Record skipped cases (expected none) and isolation degradations for the pull request's Validation section; remove the venv, fixture, and outputs

## 4. Finish

- [ ] 4.1 Run `just check`, write the results to the pull request's Validation section linking this plan, fill Changes with permalinks, archive this change in-request (bypass not granted), run `just spec-validate`, and point the `Spec:` link at the archive path
