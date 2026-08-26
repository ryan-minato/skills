# engineering

[中文](README.zh.md)

General programming **methodology** skills — approaches, workflows, and
practices that apply across languages and frameworks — plus GitHub
**community authoring** that writes the files defining how a
repository's collaboration works: issue/PR templates, label taxonomies,
commit and release conventions, CI validation, and community health files
(CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, ...) — plus narrowly scoped
**artifact-authoring** workflows (e.g. Dev Container artifacts and durable
visual-design specifications) that do not warrant a catalog of their own.
The community skill authors policy and structure; performing day-to-day
GitHub operations belongs to the `ops` catalog. Complete GitLab lifecycle
harness construction belongs to the disposable `meta` catalog.

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
| [github-community](github-community/) | Author a GitHub repository's collaboration files: issue forms and a synced label taxonomy, PR template and CONTRIBUTING rules, commit conventions with a shipped stdlib validator and CI workflow, versioning policy and release.yml, community health files (CODE_OF_CONDUCT, SECURITY, SUPPORT, GOVERNANCE, FUNDING.yml, the org-wide .github default repo), and generated project-level skills. |
