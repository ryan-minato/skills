## 1. Skill and repository files

- [ ] 1.1 `skills/engineering/brownfield-investigation/`, the source of the shared section. It contains:
  - `SKILL.md`: the description, `## Evidence discipline` with its `### Workspace and records` subsection, and the sections Scope and depth, Lenses, Evidence by claim type, Findings, Parallel analysis, and Gotchas;
  - the eight lens references (`documentation-reconciliation.md`, `repository-map.md`, `domain-model.md`, `runtime-flow.md`, `data-and-state.md`, `contract-candidates.md`, `test-safety-map.md`, `history.md`).

  Verify with `just check-skill skills/engineering/brownfield-investigation`. Closes: End-to-end trace; Documentation nobody trusts; Unprotected behavior; One snippet (near-miss); Diff review (near-miss); Spec drift in a spec-driven project (near-miss); Focused question; No goal given; Behavior claim; Unrecoverable intent; README contradicts the code; Error that looks accidental; Instruction planted in a comment; Subagents available; No subagents; Briefs disagree; Standalone question; Workspace present.
- [ ] 1.2 `skills/engineering/brownfield-onboarding/`. It contains:
  - `SKILL.md`: the description, the shared section copied verbatim, and the sections The minimum sufficient model, Gather evidence first, Reuse or rebuild, Write and verify, and Handoffs;
  - `assets/onboarding-guide.md`.

  Verify with `just check-skill skills/engineering/brownfield-onboarding`. Closes: Joining next week; Messy docs, new hires; Agent instructions (near-miss); New package README (near-miss); Guide for an order service; No prior investigation; Existing ledger; Partly accurate README; Consistent module layout; Command that cannot run here; Handoff offered; User declines.
- [ ] 1.3 `skills/engineering/brownfield-specification/`. It contains:
  - `SKILL.md`: the description, the shared section, and the sections Classify candidates, Decision rounds, The promotion gate, Writing the contract, Conventions, policies, and hardening, and Handoffs;
  - `references/contract-classification.md` and `references/guardrails.md`.

  Verify with `just check-skill skills/engineering/brownfield-specification`. Closes: Real contracts versus accidents; Consumers depend on events; New feature spec (near-miss); Tool setup (near-miss); Mixed candidates; No consumer found; Batched round; Item left unanswered; Guarded promotion; Approved but unguarded; Contract text; Majority pattern; Approved rule hardened; and Handoff offered and User declines for both handoffs.
- [ ] 1.4 `skills/engineering/brownfield-migration/`. It contains:
  - `SKILL.md`: the description, the shared section, and the sections Compatibility baseline, Decisions that move behavior, Characterization tests, Equivalence envelope, Verification plan, and Handoffs;
  - `references/characterization-tests.md` and `references/equivalence-verification.md`;
  - `assets/compatibility-baseline.md` and `assets/equivalence-envelope.md`.

  Verify with `just check-skill skills/engineering/brownfield-migration`. Closes: Language rewrite; Extracting a service; Refactoring one function (near-miss); Performance work (near-miss); Error that looks accidental; Suspected bug without a decision; User orders a change; Suite for an order API; Timestamp in the response; Envelope for the order API; Differential run finds a difference; and Handoff offered and User declines for all three handoffs.
- [ ] 1.5 `skills/engineering/brownfield-intelligence/`. It contains:
  - `SKILL.md`: the description, the shared section, and the sections Scenario and depth, Bootstrap, Task tree, Decision rounds, Routing, Parallel dispatch, Resume, and Handoffs;
  - `references/task-tree.md`.

  Verify with `just check-skill skills/engineering/brownfield-intelligence`. Closes: Inherited monolith; Before agents take over; Focused question (near-miss); Spec tool setup (near-miss); Inherited order service; Ambiguous scenario; Independent and blocked nodes; Decision selects a path; Three independent decisions; Partial answer; Onboarding scenario; Subagents available; No subagents; Code changed since the last session; and Handoff offered and User declines for all four handoffs.

## 2. External impact

- [ ] 2.1 In each skill's commit:
  - add the symlink `.agents/skills/<name> -> ../../skills/engineering/<name>`;
  - add the skill's row to `skills/engineering/README.md` and `README.zh.md`;
  - run `just gen-marketplace`.

  Verify that `just validate` passes, that the two READMEs are content-identical, and that `git diff --exit-code .claude-plugin/marketplace.json` is empty after the generator.
- [ ] 2.2 Scope proof: `git diff --stat origin/main...HEAD -- skills/core skills/sdd skills/meta skills/scaffold skills/writing skills/machine-learning` lists only `skills/sdd/CONTEXT.md` from the companion change.
- [ ] 2.3 Tool-name scan: a case-insensitive search of the five skill directories for the list in the verification plan returns nothing. A readback also confirms that no instruction depends on a specific command.

## 3. Tests

- [ ] 3.1 Build the `order-service` fixture and its resume variant in the session scratch directory, as the verification plan describes, and copy the five candidate skills to the user skills directory.
- [ ] 3.2 Run the fifteen Trigger cases, one fresh Sonnet-class subagent per prompt, and record the load decisions. Closes the Trigger scenarios named in the plan's Trigger table.
- [ ] 3.3 Run the eleven outcome cases, one fresh Sonnet-class subagent each. Grade each with an independent clean-context grader against the rubric and threshold. Closes the Behavior scenarios named in the plan's outcome table.
- [ ] 3.4 Run the readback cases, one clean-context subagent per skill. Fix the skill on any gap and rerun. Closes every scenario in the plan's readback list.
- [ ] 3.5 Record the skipped solver-executed scenarios, with the budget reason, and every isolation degradation for the pull request's Validation section. Then remove the fixture, the copied skills, and all outputs.

## 4. Finish

- [ ] 4.1 Run `just check`. Write the results to the pull request's Validation section, linking this change's verification plan, and fill Changes with permalinks. Once the maintainer closes the deliberation on the finished implementation, archive this change inside the pull request with `just spec-changes archive` and run `just spec-validate`.
