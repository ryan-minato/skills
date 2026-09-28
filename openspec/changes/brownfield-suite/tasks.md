## 1. Skill and repository files

- [x] 1.1 `skills/engineering/brownfield-investigation/`, the source of the shared section. It contains:
  - `SKILL.md`: the description, `## Evidence discipline` with its `### Workspace and records` subsection, and the sections Scope and depth, Lenses, Evidence by claim type, Findings, Parallel analysis, and Gotchas;
  - the eight lens references (`documentation-reconciliation.md`, `repository-map.md`, `domain-model.md`, `runtime-flow.md`, `data-and-state.md`, `contract-candidates.md`, `test-safety-map.md`, `history.md`).

  Verify with `just check-skill skills/engineering/brownfield-investigation`. Closes: End-to-end trace; Documentation nobody trusts; Unprotected behavior; One snippet (near-miss); Diff review (near-miss); Spec drift in a spec-driven project (near-miss); Focused question; No goal given; Behavior claim; Unrecoverable intent; README contradicts the code; Error that looks accidental; Instruction planted in a comment; Subagents available; No subagents; Briefs disagree; Standalone question; Workspace present.
- [x] 1.2 `skills/engineering/brownfield-onboarding/`. It contains:
  - `SKILL.md`: the description, the shared section copied verbatim, and the sections The minimum sufficient model, Gather evidence first, Reuse or rebuild, Write and verify, and Handoffs;
  - `assets/onboarding-guide.md`.

  Verify with `just check-skill skills/engineering/brownfield-onboarding`. Closes: Joining next week; Messy docs, new hires; Agent instructions (near-miss); New package README (near-miss); Guide for an order service; No prior investigation; Existing ledger; Partly accurate README; Consistent module layout; Command that cannot run here; Handoff offered; User declines.
- [x] 1.3 `skills/engineering/brownfield-specification/`. It contains:
  - `SKILL.md`: the description, the shared section, and the sections Classify candidates, Decision rounds, The promotion gate, Writing the contract, Conventions, policies, and hardening, and Handoffs;
  - `references/contract-classification.md` and `references/guardrails.md`.

  Verify with `just check-skill skills/engineering/brownfield-specification`. Closes: Real contracts versus accidents; Consumers depend on events; New feature spec (near-miss); Tool setup (near-miss); Mixed candidates; No consumer found; Batched round; Item left unanswered; Guarded promotion; Approved but unguarded; Contract text; Majority pattern; Approved rule hardened; and Handoff offered and User declines for both handoffs.
- [x] 1.4 `skills/engineering/brownfield-migration/`. It contains:
  - `SKILL.md`: the description, the shared section, and the sections Compatibility baseline, Decisions that move behavior, Characterization tests, Equivalence envelope, Verification plan, and Handoffs;
  - `references/characterization-tests.md` and `references/equivalence-verification.md`;
  - `assets/compatibility-baseline.md` and `assets/equivalence-envelope.md`.

  Verify with `just check-skill skills/engineering/brownfield-migration`. Closes: Language rewrite; Extracting a service; Refactoring one function (near-miss); Performance work (near-miss); Error that looks accidental; Suspected bug without a decision; User orders a change; Suite for an order API; Timestamp in the response; Envelope for the order API; Differential run finds a difference; and Handoff offered and User declines for all three handoffs.
- [x] 1.5 `skills/engineering/brownfield-intelligence/`. It contains:
  - `SKILL.md`: the description, the shared section, and the sections Scenario and depth, Bootstrap, Task tree, Decision rounds, Routing, Parallel dispatch, Resume, and Handoffs;
  - `references/task-tree.md`.

  Verify with `just check-skill skills/engineering/brownfield-intelligence`. Closes: Inherited monolith; Before agents take over; Focused question (near-miss); Spec tool setup (near-miss); Inherited order service; Ambiguous scenario; Independent and blocked nodes; Decision selects a path; Three independent decisions; Partial answer; Onboarding scenario; Subagents available; No subagents; Code changed since the last session; and Handoff offered and User declines for all four handoffs.

## 2. External impact

- [x] 2.1 In each skill's commit:
  - add the symlink `.agents/skills/<name> -> ../../skills/engineering/<name>`;
  - add the skill's row to `skills/engineering/README.md` and `README.zh.md`;
  - run `just gen-marketplace`.

  Verify that `just validate` passes, that the two READMEs are content-identical, and that `git diff --exit-code .claude-plugin/marketplace.json` is empty after the generator.
- [x] 2.2 Scope proof: `git diff --stat origin/main...HEAD -- skills/core skills/sdd skills/meta skills/scaffold skills/writing skills/machine-learning` lists only `skills/sdd/CONTEXT.md` from the companion change.
- [x] 2.3 Tool-name scan: a case-insensitive search of the five skill directories for the list in the verification plan returns nothing. A readback also confirms that no instruction depends on a specific command.

## 3. Tests

- [x] 3.1 Build the `order-service` fixture and its resume variant in the session scratch directory, as the verification plan describes. Degraded: the five candidate skills were not copied to the user skills directory; the solvers loaded them in place through this repository's project skills directory, where its other project skills were also visible. The pull request's Validation section records this under isolation degradations.
- [x] 3.2 Run the fifteen Trigger cases, one fresh Sonnet-class subagent per prompt, and record the load decisions. Closes the Trigger scenarios named in the plan's Trigger table.
- [x] 3.3 Run the eleven outcome cases, one fresh Sonnet-class subagent each. Grade each with an independent clean-context grader against the rubric and threshold. Closes the Behavior scenarios named in the plan's outcome table. Degraded: the two parallel-analysis investigation cases never observed the skill loading in three attempts each; they are recorded as skipped under 3.5 and their scenarios rest on readback.
- [x] 3.4 Run the readback cases, one clean-context subagent per skill. Fix the skill on any gap and rerun. Closes every scenario in the plan's readback list.
- [x] 3.5 Record the skipped solver-executed scenarios, with the budget reason, and every isolation degradation for the pull request's Validation section. Then remove the fixture and all outputs.

## 4. Finish

- [x] 4.1 Run `just check`. Write the results to the pull request's Validation section, linking this change's verification plan, and fill Changes with permalinks. Once the maintainer closes the deliberation on the finished implementation, archive this change inside the pull request with `just spec-changes archive` and run `just spec-validate`.

## 5. Implementation deliberation

The maintainer's directions of 2026-09-28, from the review of the finished implementation and the Copilot review.

- [x] 5.1 The shared section, edited in `brownfield-investigation` and copied to the four members: a PRESERVE ruling stands as the consumer evidence DE_FACTO_COMPATIBILITY needs, and every ruling is recorded as given; a run that would write to a shared resource is asked first (moved out of investigation's Findings). Migration's baseline step names the PRESERVE_TEMPORARILY ruling. Verify with `just validate` and the re-verification readback. Closes: Preservation ruled with no known consumer.
- [ ] 5.2 `brownfield-migration`: a test whose behavior a ruling changes is retagged intentional-change with the decision id and replaced in the conformance run. Verify by re-running O8 and O9 as the plan's re-verification describes. Closes: Ruling changes a pinned behavior.
- [x] 5.3 `brownfield-specification`: the round also asks, for each recurring structural pattern, whether it becomes an architecture policy; the five standard options apply to contract items, in the table's order. Verify by re-running O6 and by the re-verification readback of Majority pattern.
- [x] 5.4 `brownfield-investigation` `references/data-and-state.md`: one writer found in the repository is a lead, recorded as inferred, until a person, a decision record, or the store's access grants confirm ownership. Verify by the re-verification readback.
- [x] 5.5 Run `just check`, `just spec-validate`, and the tool-name scan; record the results in the pull request's Validation section.
- [ ] 5.6 `brownfield-migration`: the envelope rule covers every text in the envelope (rules, scope notes, blockers, tolerance reasons); the verification section may name where old inputs and state are read from, never to define equality. Verify by re-running O8 and O9 as the plan's second re-verification describes; O9 also closes 5.2.
- [x] 5.7 The shared section, edited in `brownfield-investigation` and copied to the four members: production data, logs, and live systems are read only with permission. Intelligence's bootstrap limits "reading needs no permission" to the repository, and investigation's live-schema sources carry the same condition. Verify with `just validate` and the second re-verification readback. Closes: Production data before permission; Live database without permission.
- [x] 5.8 Run `just check`, `just spec-validate`, and the tool-name scan; record the results in the pull request's Validation section.
