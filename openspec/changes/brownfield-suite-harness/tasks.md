## 1. Skill and repository files

- [x] 1.1 `skills/engineering/CONTEXT.md`: the class list, the suite grant in `## Dependencies`, the suite prefix in `## Naming`, and the split brownfield route plus the named "two skills" in `## Disambiguation`. Verify with a readback against the placement rows. Proves the grant, class list, suite prefix, and split route.
- [x] 1.2 `skills/sdd/CONTEXT.md`: one clause in `## Scope` and one route in `## Disambiguation` pointing to the suite. Verify with a readback. Proves the `sdd` pointer.
- [x] 1.3 Update two mirrors of the grant:
  - the "today:" parenthesis in `.agents/knowledge/skill-quality.md`;
  - the `engineering` bullet in `ARCHITECTURE.md` `## Catalogs`, naming the suite by prefix, with no backticked path.

  Verify that `just validate` passes. Proves the grants sentence and the architecture bullet.
- [x] 1.4 `skills/engineering/README.md` and `README.zh.md`: the introduction paragraph and one install example listing the five `--skill` names. Verify that both files are content-identical. Proves the README introduction.
- [x] 1.5 `scripts/validate_harness.py`: add the shared-section check to `main()` beside `check_copies`, and the `suite` entry to the docstring. Verify that `just validate` and `just lint` pass with no member present. Proves the shared-material check (step 5).

## 2. External impact

- [x] 2.1 Confirm that no other file under `scripts/`, `.github/`, or `.agents/knowledge/` changes, apart from `skill-quality.md`: `git diff --stat origin/main...HEAD -- scripts .github .agents/knowledge`. Verify with `just validate`.

## 3. Tests

- [x] 3.1 Once the five skills exist, run the verification plan's steps 1–4 of the shared-material check in a disposable worktree:
  1. `just validate` passes on the complete tree.
  2. Change a line in the `### Workspace and records` subsection. The check fails.
  3. Rename the `## Evidence discipline` heading. The check fails.
  4. Remove the source skill. The check fails.

  After each failure, confirm that the error names the file, then revert. Proves the shared-material check.
- [x] 3.2 Run a clean-context readback of the `CONTEXT.md` pair, `skill-quality.md`, `ARCHITECTURE.md`, and the README pair against the placement rows. Proves every text bullet.

## 4. Finish

- [x] 4.1 Run `just check` and write the results to the pull request's Validation section, linking this plan. Once the maintainer closes the deliberation on the finished implementation, archive this change beside `brownfield-suite` and run `just spec-validate`.

## 5. Implementation deliberation

- [ ] 5.1 `scripts/validate_harness.py`: compare the shared section exactly as matched, without trimming, so that trailing spaces or blank lines at its end count as drift (Copilot review). Verify in a disposable worktree: steps 1–4 of the plan again, plus two extra blank lines at the end of a member's section, which must fail and name the file; then `just lint`.
