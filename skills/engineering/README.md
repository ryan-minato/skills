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

## Skills

| Skill | Description |
|---|---|
| [code-refactoring](code-refactoring/) | Restructure existing code in small behavior-preserving steps verified by tests: separate structural change from behavior change, decide when to refactor (and when not to), diagnose code smells, and execute the standard named refactoring techniques safely. |
| [devcontainer-authoring](devcontainer-authoring/) | Author, test, and publish Dev Container artifacts — Features (install.sh contract, idempotency and base-image quality bar, independence rule), Templates (option substitution, payload design, smoke-test loop), and prebuilt images (devcontainer build --push, metadata merge semantics) — with bundled repo scaffolds and shared-action CI. |
| [design-md](design-md/) | Author and validate a durable, agent-readable DESIGN.md visual-design specification with optional YAML design tokens, prose guidance, upstream format checks, and an OKLCH calculator. |
| [gitmoji](gitmoji/) | Draft gitmoji commit messages: resolve the project variant (standalone vs CC-combined grammar, unicode vs text codes), pick the one emoji for the dominant intent via a first-match decision list, and validate against a pre-handover checklist. |
| [goal-alignment](goal-alignment/) | Converge with the user on what something should achieve — software, systems, experiments, skills, services — through relentless rounds of questions that carry suggested answers wherever one can be inferred (facts only the user can know are asked directly), then record the consensus in a single source-of-truth goal document: overall goal, tiered concrete goals with verification (hard constraint / optimization target / preference), leveled requirements, and trade-off decisions. Goals only; no plans or architecture. |
| [knowledge-deposition](knowledge-deposition/) | Deposit a confirmed piece of knowledge into a project so future agents find and follow it: probe where agent-facing guidance already lives, choose the right carrier (entrypoint line only for what every session must see, knowledge-base file plus event pointer as the default, project skill for recurring fragile procedures, or park it until it recurs), write it as standalone instructions, and register an event-triggered pointer. |
| [session-retrospective](session-retrospective/) | Distill a work session into durable project lessons: mine the conversation for six signals (repeated failures, tool-flagged corrections, expensive discoveries, experiment verdicts, overruled defaults, docs–reality mismatches), weigh recurrence times impact against the context rent a record charges, and present a ranked findings list for per-item approval — findings only, nothing written until the user approves. |
| [spec-driven-development](spec-driven-development/) | Work from written specifications: judge when spec-driven development pays and at which level (spec-first, spec-anchored, spec-as-source), pick the approach with one default per situation (a spec-first kit for a new application, a capability-organized spec-anchored change workflow for libraries, infrastructure, and any existing code) and leave the decision to the user, run the specify–clarify–plan–tasks–implement–verify loop tool-neutrally, keep acceptance criteria in the spec only, and convert a vibe-coded prototype or brownfield codebase one change at a time without backfilling specs. |
