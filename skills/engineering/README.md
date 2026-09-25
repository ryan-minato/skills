# engineering

[中文](README.zh.md)

General programming **methodology** skills — approaches, workflows, and
practices that apply across languages and frameworks — plus narrowly
scoped **artifact-authoring** workflows (e.g. Dev Container artifacts and
durable visual-design specifications) that do not warrant a catalog of
their own. Building a GitHub or GitLab project's complete lifecycle
harness — collaboration files, conventions, and day-to-day platform
workflows included — belongs to the disposable `meta` catalog.

```bash
npx skills add ryan-minato/skills --skill <skill-name>
```

The **brownfield suite** — five skills prefixed `brownfield-` — works on
an existing codebase: understanding it with evidence, onboarding
engineers, deciding which behaviors are contracts and guarding them, and
preserving boundary behavior through a rewrite. Its members depend on one
another, so install them together:

```bash
npx skills add ryan-minato/skills \
  --skill brownfield-intelligence --skill brownfield-investigation \
  --skill brownfield-onboarding --skill brownfield-specification \
  --skill brownfield-migration
```

## Skills

| Skill | Description |
|---|---|
| [brownfield-investigation](brownfield-investigation/) | Investigate an existing codebase with evidence: choose the lenses a question needs (documentation against code, repository map, domain model, runtime flow, data and state, contract candidates, test safety map, history) and a depth (ORIENT, ESTABLISH, EXHAUSTIVE) at a pinned revision; record each finding with its source, confidence, counter-evidence, and unknowns; report drift and suspected bugs without editing or fixing anything; fan independent lenses out to subagents where the host allows, sequentially otherwise. |
| [brownfield-migration](brownfield-migration/) | Preserve the behavior consumers depend on through a rewrite, re-platforming, or service extraction: build a compatibility baseline of boundary behavior with a ruling for each item (promote to spec, preserve temporarily, intentionally change), pin it with characterization tests tagged normative, compatibility, or pending-decision, agree an equivalence envelope that compares boundaries rather than internals, and plan conformance and differential verification with explicit tolerances — no silent fixes, no accidental fossilization. |
| [brownfield-onboarding](brownfield-onboarding/) | Build onboarding material for engineers joining an existing codebase: a minimum sufficient, trustworthy mental model (purpose, running it, components, vocabulary, architecture, representative flows, where to change what, risks and unknowns) resting on investigated evidence, reusing documentation that checks out, describing observed conventions as current practice rather than rules, and verifying every path and command it gives. |
| [brownfield-specification](brownfield-specification/) | Turn an existing system's observed behavior into normative specification only by decision: classify contract candidates (specification, compatibility, architecture, implementation, unknown) with the rewrite and consumer tests, put the rulings to a human in batched rounds, promote an approved contract only with a guardrail shown to pass on the current system and fail on a violation, write contracts rather than implementation, and harden only approved rules into checks. |
| [code-refactoring](code-refactoring/) | Restructure existing code in small behavior-preserving steps verified by tests: separate structural change from behavior change, decide when to refactor (and when not to), diagnose code smells, and execute the standard named refactoring techniques safely. |
| [devcontainer-authoring](devcontainer-authoring/) | Author, test, and publish Dev Container artifacts — Features (install.sh contract, idempotency and base-image quality bar, independence rule), Templates (option substitution, payload design, smoke-test loop), and prebuilt images (devcontainer build --push, metadata merge semantics) — with bundled repo scaffolds and shared-action CI. |
| [design-md](design-md/) | Author and validate a durable, agent-readable DESIGN.md visual-design specification with optional YAML design tokens, prose guidance, upstream format checks, and an OKLCH calculator. |
| [gitmoji](gitmoji/) | Draft gitmoji commit messages: resolve the project variant (standalone vs CC-combined grammar, unicode vs text codes), pick the one emoji for the dominant intent via a first-match decision list, and validate against a pre-handover checklist. |
| [goal-alignment](goal-alignment/) | Converge with the user on what something should achieve — software, systems, experiments, skills, services — through relentless rounds of questions that carry suggested answers wherever one can be inferred (facts only the user can know are asked directly), then record the consensus in a single source-of-truth goal document: overall goal, tiered concrete goals with verification (hard constraint / optimization target / preference), leveled requirements, and trade-off decisions. Goals only; no plans or architecture. |
| [knowledge-deposition](knowledge-deposition/) | Deposit a confirmed piece of knowledge into a project so future agents find and follow it: probe where agent-facing guidance already lives, choose the right carrier (entrypoint line only for what every session must see, knowledge-base file plus event pointer as the default, project skill for recurring fragile procedures, or park it until it recurs), write it as standalone instructions, and register an event-triggered pointer. |
| [session-retrospective](session-retrospective/) | Distill a work session into durable project lessons: mine the conversation for six signals (repeated failures, tool-flagged corrections, expensive discoveries, experiment verdicts, overruled defaults, docs–reality mismatches), weigh recurrence times impact against the context rent a record charges, and present a ranked findings list for per-item approval — findings only, nothing written until the user approves. |
