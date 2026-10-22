## 1. Skill and repository files

- [x] 1.1 Rewrite `## Harness Methodology` in `skills/core/meta-harness/SKILL.md` (six layers and five variables; loading modes with explicit routing and subdirectory carriers; hard boundaries placed where enforced with quiet output and error routing; threat model; authorization boundary; non-interactive runtime with commit checkpoints; drift landing and two-tier self-checks; hardening and thinning) and copy it byte-identical into `skills/meta/meta-harness-architecture/SKILL.md`; verify `just check-skill skills/core/meta-harness skills/meta/meta-harness-architecture` — closes core/meta-harness: all seven Behavior requirements
- [x] 1.2 Rewrite `skills/meta/meta-harness-architecture/references/layers.md` around the six layers with light and thick shapes, the variables that push each layer, the two presets, per-layer deviation, hardening, and thinning; verify `just check-skill` — closes Layer calibration uses six layers and five variables
- [x] 1.3 Add `references/security-boundary.md` (three conditions, source reachability, human in the loop, model resistance; sandbox audit; control-plane mutability; readable rejection; verifier fallback); verify `just check-skill` — closes The security boundary is designed from the threat model and audited in the sandbox
- [x] 1.4 Add `references/authorization.md` and `assets/pending-decisions.md`; verify `just check-skill` — closes The authorization boundary is deposited as a reversibility-keyed policy
- [x] 1.5 Add `references/runtime.md`; verify `just check-skill` — closes Runtime conventions are deposited for non-interactive execution and commit checkpoints
- [x] 1.6 Add `references/harness-checks.md`; verify `just check-skill` — closes The harness gets deterministic checks of its own
- [x] 1.7 Extend `references/entrypoint.md` and `assets/agents-md-skeleton.md` (always-applicable part first, four questions, explicit routing, carriers; authorization and runtime sections); verify `just check-skill` — closes The entrypoint is two-part and splits are justified
- [x] 1.8 Extend `references/multi-agent.md` (shared resources, remote branch per agent, coordinator, merge ownership); verify `just check-skill` — closes Multiple agents get isolated resources and one coordinator for remote state
- [x] 1.9 Extend `references/maintenance.md` (drift landing, hook reminders, session-log observation, canary tasks), `references/management-code.md` (quiet output), `references/audit.md` (thinning signal, hardening pass); verify `just check-skill` — supporting the entropy and hard-boundary requirements
- [x] 1.10 Update `skills/meta/meta-harness-architecture/SKILL.md`: description (new layers and phrasings, under 1000 characters), `## Design axes` (six layers, five variables), `## Load by artifact` (new references with their load triggers); verify `just check-skill` — closes meta-harness-architecture Trigger: description
- [x] 1.11 Update `skills/meta/meta-harness-building/SKILL.md` steps 1–3 and `assets/harness-plan-template.md` (six-row thickness table with preset and deviation; profile rows for the five variables, external inputs, human-review point, pre-authorized actions, hard read-only files); verify `just check-skill skills/meta/meta-harness-building` — closes meta-harness-building: The requirement round settles the five variables and the threat-model facts; The plan rates six layers with evidence; Trigger: description

## 2. External impact

- [x] 2.1 Update the `meta-harness` row in `skills/core/README.md` and `README.zh.md`, and the `meta-harness-architecture` and `meta-harness-building` rows in `skills/meta/README.md` and `README.zh.md`; verify `just validate`
- [x] 2.2 Run `just gen-marketplace` for the changed description; verify `just validate`
- [x] 2.3 Hand-correct `## Purpose` in `openspec/specs/core/meta-harness/spec.md`; verify `just spec-validate`
- [x] 2.4 Read through `skills/meta/CONTEXT.md` and `skills/engineering/harness-study/SKILL.md` to confirm the mirror sentence and the six layer names still hold; no edit expected

## 3. Tests

- [x] 3.1 Trigger cases for `meta-harness-architecture` (four load, two near-miss) and `meta-harness-building` (two load, two near-miss) per the verification plan — closes both Trigger blocks
- [x] 3.2 Outcome cases for `core/meta-harness` (seven paired cases) per the verification plan — closes the core Behavior scenarios
- [x] 3.3 Outcome cases for `meta-harness-architecture` (seven paired cases) and `meta-harness-building` (two paired cases) per the verification plan — closes the remaining Behavior scenarios
- [x] 3.4 Record skipped cases and their reasons for the pull request's Validation section

## 4. Finish

- [x] 4.1 Run `just check`, write the results to the pull request's Validation section linking the verification plan, and leave the change unarchived for the maintainer's second deliberation
