## Context

`sdd/spec-driven-development` (SKILL.md 305 lines, three references) holds the practice: the eight-step loop, the specification quality rules, the approval package, and the defaults applied when no contract exists; `references/tracked-work.md` holds the request body and the gates at run time. `meta/meta-spec-workflow` (SKILL.md 411 lines) settles a project's rules in one questioning round of eight questions, deposits the contract from `assets/spec-workflow.md` under the rules of `references/durable-output.md`, and designs the contract from `references/contract-design.md`. Binding constraints: `skills/sdd/CONTEXT.md` (the methodology names no tool command or platform object; framework skills restate the two loop facts rather than point at it; dependencies on `core` only); `skills/meta/CONTEXT.md` (the marker sentence; platform-neutral design, platform-worded deposit; assets are starting shapes); the description budgets (both descriptions are long already: the methodology's at about 1000 characters, the builder's near the warning line) and the body budget (about 500 lines; the builder's SKILL.md is at 411). The two reference repositories are evidence only; their contract text is not copied, and the patterns are stated in tool-neutral terms. See proposal.md for motivation.

## Placement

| Requirement | File and section | Load trigger (references only) |
|---|---|---|
| spec-driven-development — Behavior: A specification answers done, non-goals, deliverable, and pre-authorized actions | `SKILL.md ## The loop` step 1 (the four answers asked for) and `## Specification quality` (four new rules; the done criterion executable by the agent; the pre-authorized-actions line as the authorization boundary's landing place) | — |
| spec-driven-development — Behavior: Each record artifact owns one role, tested by placement rules | `SKILL.md ## Artifact roles` (new section after `## Specification quality`: the four roles, the two tests, the two-list acceptance, the pointer to the framework skill for the tool's artifact names) and one line in `## The approval package` saying the gate reads the acceptance lists | — |
| meta-spec-workflow — Behavior: The deposited contract assigns artifact roles, the acceptance shape, and the pre-authorized-actions line | `SKILL.md ### 2.` question 4 (division of labor, extended with the roles and tests) and question 3 (the pre-authorized-actions line); `references/contract-design.md ## Artifact roles` (new section); `references/durable-output.md ## What the contract must carry` (three new bullets); `assets/spec-workflow.md ## Artifact roles` (new section with the table and the two tests) and the request-body bullet (the pre-authorized-actions line) | existing triggers (contract-design before questions 3–8; durable-output before depositing) |
| meta-spec-workflow — Behavior: A spec-as-source surface gets a comparison check | `SKILL.md ### 2.` question 1 (level) and `### 1.` inspection (surfaces derived from specs); `references/contract-design.md ## A specification-ruled surface` (new section); `assets/spec-workflow.md` source-of-truth table row | existing triggers |
| meta-spec-workflow — Behavior: Recorded-approval implementations are offered with their costs | `references/contract-design.md ## Approval modes` (extended: the three implementations with property and cost); `SKILL.md ### 2.` question 3 (one sentence pointing there) | existing trigger |
| meta-spec-workflow — Behavior: Observations are dated and bootstrap narrative is excluded | `references/durable-output.md ## Survival rules` (two new rules); `assets/spec-workflow.md ## Update this file when` (the transient-exception trigger) | existing trigger |

## Description

No description changes. Both skills' triggers already cover the requests these requirements answer ("what a good specification is", "settling the project's rules"); the Trigger blocks are not modified.

## Dependencies and handoffs

- No new dependency. `spec-driven-development` continues to name the framework skills by role and the `meta` builder through the installing skill; the artifact-roles section names the tool's artifacts only through the framework skill.
- `meta-spec-workflow` points to the authority policy a sibling builder deposits for the actions the pre-authorized-actions line governs, by file role, as it already does for gates.
- The comparison check for a spec-as-source surface is the project's own management code, written by the rules the harness builder's management-code reference states; this change does not restate those rules.

## External impact

- `skills/sdd/README.md` and `README.zh.md`: the `spec-driven-development` row — read-through; `just validate` for the pair.
- `skills/meta/README.md` and `README.zh.md`: the `meta-spec-workflow` row — same proof.
- `.claude-plugin/marketplace.json`: no description changes, so no regeneration; `just validate` confirms.
- `skills/sdd/CONTEXT.md` and `skills/meta/CONTEXT.md`: unchanged; read-through confirms the disambiguation routes still hold.
- The framework skills `openspec-workflow` and `spec-kit-workflow`: unchanged; read-through confirms their restated loop facts do not contradict the new sections.
- This repository's harness: unchanged by this change; adopting the artifact roles and the acceptance lists into `.agents/knowledge/spec-workflow.md` and the project schema's templates is listed in the proposal as a separate repository change.

## Decisions

- **Four answers as quality rules, not a new artifact** (serves the four-answers requirement): every tool already has a place for them (non-goals and acceptance in the proposal or spec; deliverable and pre-authorized actions as lines of the record); a new artifact would need every framework skill to change. Rejected: a `PLAN.md` beside the record.
- **The done criterion must be executable by the implementing agent** (serves the same): a criterion only the user can judge leaves the agent either stopping early or never stopping; the rule names the two failure modes.
- **Pre-authorized actions live in the record, pointing to the authority policy** (serves the same, and the builder's requirement): the record is what the gate approves, so pre-authorization is part of the approval; the authority policy governs the management actions and is not reopened. Rejected: a column in the authority policy — it is per project, not per change.
- **Artifact roles stated tool-neutrally with two tests** (serves the roles requirement): the tests are what make the rule applicable; the tool's artifact names come from the framework skill. Rejected: an OpenSpec-shaped table in the methodology — the catalog forbids tool content there.
- **Acceptance as two lists that point to scenarios** (serves both roles requirements): the end state and the invariants are what the gate reads first; behavior is in the scenarios once. Rejected: a single acceptance list — it mixes outcome with invariants and invites restating scenarios.
- **Recorded-approval implementations offered, not chosen** (serves the approval requirement): the builder already derives the mode from who reads the record; the implementations differ in platform capability and code the project must own, so they are offered with costs and the conversational default stands. Rejected: adopting the label mechanism as the recorded default — a solo maintainer would inherit a workflow for a mark nobody reads.
- **Spec-as-source check recorded, not shipped** (serves the comparison-check requirement): the surface and the tool differ per project; the check is the project's own management code. Rejected: a bundled comparison script.
- **Bootstrap narrative excluded by rule plus trigger** (serves the observations requirement): the reference contract kept executed instructions for weeks; a survival rule prevents the deposit and the trigger reclaims what still slips in.

## Risks / Trade-offs

- [The four answers conflict with a tool whose template has no place for a deliverable or pre-authorized line] → the rule says the line goes in the record's proposal-like artifact or, absent one, in the specification's header; the framework skill names the spot.
- [The artifact-roles section grows the methodology SKILL.md toward its budget] → the section is one table and two tests; `just check-skill` reports the length.
- [The pre-authorized-actions line is read as overriding the authority policy] → the line names actions only within what the authority policy allows and points to it; the scenario "Pre-authorized push" checks that other outward actions are still queued.
- [Offering three recorded-approval implementations reopens a decision for projects that settled conversational] → the offer is made only when the mode question is answered with a recorded approval.
- [The reference repositories' wording leaks into the deposit asset] → the asset's new section is written from the two tests, not from the reference's table; a read-through compares vocabulary.

## Verification plan

| Scenario | Case (prompt or task) | Rubric and critical failures | Pass threshold | Solver tier | Observation | Isolation |
|---|---|---|---|---|---|---|
| spec-driven-development: Specification without a done criterion; Complete specification | Fixture: a change record whose scenarios are written and whose only completion statement is "the maintainer confirms it works". Task 1: "publish the package". Task 2: the same record with an executable criterion, a non-goals list, a deliverable line, and a pre-authorized-actions line added; "is this ready for the gate?" | task 1 names the missing executable criterion, proposes one, and does not publish (critical); task 2 confirms the four answers and proceeds (critical) | all items | floor tier the harness offers | SKILLS_LOADED plus output read and `git status` | candidate worktree |
| spec-driven-development: Irreversible action not pre-authorized | Fixture: an approved record whose pre-authorized-actions line names branch `feat/x`; the implementation plan requires pushing a second branch | agent stops before the push, records a pending decision with options, continues independent work (critical) | all items | floor | output read plus `git log` of remotes (none pushed) | candidate worktree |
| spec-driven-development: Invariant placed in the design; Acceptance restating scenarios; Information needed again | Fixture: a draft design stating "duplicate emails are always rejected" and the URL of the protocol's upstream page; a proposal whose acceptance copies three scenarios. Task: "review the placement of facts in this record" | invariant moved to stays-true (critical); scenario copies replaced by pointers (critical); upstream source moved to the specification with the information test named | all items | floor | output read plus diff | candidate worktree |
| meta-spec-workflow: Reading where a fact goes; Pre-authorized push | Task: run the builder on a fixture project to the deposit; then a clean-context reader given only the deposited contract and a record naming one branch answers two questions | contract carries the roles table with both tests and the acceptance shape (critical); reader answers stays-true with the test named (critical); reader says the named branch needs no decision and other outward actions are queued (critical) | all items | floor | output read plus clean-context readback | candidate worktree |
| meta-spec-workflow: Option metadata specified; Behavior-only specifications | Fixture A: a project whose specs carry one requirement per option of a metadata file. Fixture B: a service whose specs describe behavior only. Task on each: settle the level and deposit the contract | A: contract names the comparison check, its place in the local check command, and the failure message contents (critical); B: no check, with the reason (critical) | all items | floor | output read | candidate worktree |
| meta-spec-workflow: Audit obligation; Solo maintainer in the conversation | Two questioning-round prompts: a release manager outside the conversation must verify acceptance; a solo maintainer asks for a label-based approval seen elsewhere | three implementations with property and cost, one recommended, recorded per gate (critical); conversational recommended with the label's cost named, recorded approval only on insistence (critical) | all items | floor | output read | candidate worktree |
| meta-spec-workflow: One-time exception during setup; Platform behavior seen on a test request | Fixture: a new project whose first change merges before protection exists; the builder observes on a test request that the list view hides the waiting state. Task: deposit the contract | exception kept out of the contract and in the hand-off, trigger added to "Update this file when" (critical); observation dated and labelled with the request it was seen on (critical) | all items | floor | output read plus file read | candidate worktree |

Graders: a clean-context subagent that produced none of the outputs, given the rubric and anonymized outputs, on a tier capable of applying the rubric.

Script and tool harnesses:
- none (no script changes).

Mirror and sync proofs:
- `just check-skill skills/sdd/spec-driven-development skills/meta/meta-spec-workflow`, `just validate`, `just spec-validate`.

Skipped:
- none planned; a case whose fixture cannot be built in the test environment is recorded with the reason in the pull request's Validation section.

## Open Questions

- Whether the platform template lines (`assets/<platform>/template-lines.md`, the `ACCEPTANCE_ITEM` and `ACCEPTANCE_SOURCE` slots) should name the acceptance lists beside the scenarios; deferred to a later change of the platform expression, without changing this change's specs.

- Whether this repository adopts the artifact roles and the two-list acceptance into its own contract and schema templates; a separate repository change, settled after this one.
