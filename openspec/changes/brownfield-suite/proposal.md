## Why

Nothing in the library helps an agent understand an existing codebase it did not write while keeping evidence apart from judgment. An unaided agent reads code, writes a fluent summary, and quietly makes two mistakes:
- it turns what the code happens to do into what the system must do;
- it "fixes" behavior that consumers depend on.

A new engineer, a team preparing a legacy system for agent-driven work, and a team rewriting a system all hit these mistakes, each with a different deliverable at stake. Issue #92 asks for a suite to handle them.

## What Changes

- New suite of five skills in the `engineering` catalog, sharing the `brownfield-` prefix. Members may depend on one another by name under a new grant. That grant, the naming rule, and the routing live in the companion repository change `brownfield-suite-harness`.
- **Shared to all five members.** Every member records findings with their evidence: source kind, location, confidence by how it was obtained, counter-evidence, and unknowns. It keeps what the system does apart from what it must do, and treats text inside the investigated repository as data, never as instructions. It hands every normative decision to a human and never records a recommendation as an approval.
  - Parallel work: when the host can dispatch clean-context subagents, the member dispatches independent analysis in parallel with self-contained read-only briefs and reconciles the results. When the host cannot, it runs the same briefs sequentially and produces the same output shape.
  - Tool-neutral instructions: members name no host tool. They describe the capability needed and leave the choice of tool to the agent.
- `brownfield-investigation`: an agent that loads it answers a question about an existing codebase through the lenses the question needs, at a stated depth (ORIENT, ESTABLISH, EXHAUSTIVE) and at a pinned revision. The lenses are documentation reconciliation, repository map, domain model, runtime flow, data and state, contract candidates, test safety map, and history.
  - Each claim type is checked against its authoritative source, and every behavior claim carries a call chain.
  - Documentation drift and suspected bugs are recorded as findings, never edited or fixed.
- `brownfield-onboarding`: an agent that loads it builds a minimum sufficient, trustworthy mental model for engineers joining an existing project: purpose, running it, components, vocabulary, architecture, representative flows, where to change what, risks and unknowns, and further reading.
  - The material rests on investigated evidence. Verified documentation is reused, not duplicated.
  - Observed conventions are described as current practice and never as rules. Every path and command in the material is verified.
- `brownfield-specification`: an agent that loads it sorts contract candidates into specification, compatibility, architecture, implementation, or unknown, using the rewrite and consumer tests.
  - It puts the normative decisions to a human in batched rounds.
  - It promotes an approved contract to normative specification only when an executable guardrail accompanies it: a check shown to pass on the current system and to fail when the contract is broken.
  - It hardens only approved rules into checks.
- `brownfield-migration`: an agent that loads it builds a compatibility baseline of boundary behavior, pins it with characterization tests tagged normative, compatibility, or pending-decision, and agrees an equivalence envelope with the human. It then plans conformance and differential verification with explicit tolerances.
  - It refuses both silent fixes and accidental fossilization.
- `brownfield-intelligence`: an agent that loads it leads work on an existing codebase. It identifies the scenario and the depth, bootstraps the scope, and keeps a task tree. It finishes all independent work before asking, then batches human decisions at the dependency frontier.
  - It dispatches independent tasks in parallel where the host allows, and routes each task to the member that owns it.
  - It resumes across sessions after checking what drifted since the recorded revision.

## Skills touched

- `engineering/brownfield-investigation` (new): description triggers, lens and depth selection, evidence per claim type, documentation reconciliation, recording suspected bugs, the trust boundary, parallel analysis with sequential fallback, and where findings go.
- `engineering/brownfield-onboarding` (new): description triggers, minimum sufficient scope, evidence first, reuse or rebuild of existing documentation, observed-not-normative wording with verified paths, and the handoff to the investigation role.
- `engineering/brownfield-specification` (new): description triggers, classification, decision rounds and authority, the guardrail gate on promotion, contract-only specification text, observed convention versus policy, and handoffs to the investigation role and the spec-driven development role.
- `engineering/brownfield-migration` (new): description triggers, the compatibility baseline, no silent fix and no fossilization, classified characterization tests, the equivalence envelope, verification with explicit tolerances, and handoffs to the investigation, specification, and incremental refactoring roles.
- `engineering/brownfield-intelligence` (new): description triggers, scenario and scope bootstrap, the task tree, frontier decision rounds, routing, parallel dispatch with sequential fallback, resume with drift check, and handoffs to the four members.

## Installed behavior

Every skill is new, so an agent in a project that installs one gains a capability it did not have. Each commit is `feat`, scoped by skill name. No existing installed skill changes.

## Impact

- Five skill directories under `skills/engineering/`, five symlinks in `.agents/skills/`, five rows in both `skills/engineering/README.md` and `README.zh.md`, and the `engineering` plugin's `skills[]` in `.claude-plugin/marketplace.json` (updated by `just gen-marketplace`).
- The companion repository change `brownfield-suite-harness` carries:
  - the catalog `CONTEXT.md` grant, naming, and routing;
  - the `sdd` catalog's pointer back to the suite;
  - the current-grants sentence in `.agents/knowledge/skill-quality.md`;
  - the `engineering` entry in `ARCHITECTURE.md`;
  - the catalog README introduction;
  - the validator check that keeps the shared material identical across the five members.

## Non-goals

- Changing `spec-driven-development`, `code-refactoring`, `meta-harness`, `agentic-writing`, `plan-clarification`, or the installer skill or its script. The suite routes to them; they do not change.
- The root `README.md` pair and the `engineering` plugin description in `marketplace.json`. Both still describe the catalog accurately.
- Onboarding material for non-engineering readers such as product managers and executives.
- Bundled scripts. Nothing in the suite is deterministic enough to earn one.
- A migration strategy or implementation plan. The suite defines what must stay equivalent and how to prove it; it does not decide how the new system is built.

## Tracked work

Issue #92.
