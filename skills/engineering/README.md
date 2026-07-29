# engineering

[中文](README.zh.md)

General programming **methodology** skills — approaches, workflows, and
practices that apply across languages and frameworks — plus platform
**community authoring** skills that write the files defining how a
repository's collaboration works: issue/PR templates, label taxonomies,
commit and release conventions, CI validation, and community health files
(CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, ...). Community skills author
policy and structure; performing the day-to-day platform operations
belongs to the `ops` catalog.

```bash
npx skills add ryan-minato/skills --skill <skill-name>
```

## Skills

| Skill | Description |
|---|---|
| [code-refactoring](code-refactoring/) | Restructure existing code in small behavior-preserving steps verified by tests: separate structural change from behavior change, decide when to refactor (and when not to), diagnose code smells, and execute the standard named refactoring techniques safely. |
| [gitmoji](gitmoji/) | Draft gitmoji commit messages: resolve the project variant (standalone vs CC-combined grammar, unicode vs text codes), pick the one emoji for the dominant intent via a first-match decision list, and validate against a pre-handover checklist. |
| [github-community](github-community/) | Author a GitHub repository's collaboration files: issue forms and a synced label taxonomy, PR template and CONTRIBUTING rules, commit conventions with a shipped stdlib validator and CI workflow, versioning policy and release.yml, community health files (CODE_OF_CONDUCT, SECURITY, SUPPORT, GOVERNANCE, FUNDING.yml, the org-wide .github default repo), and generated project-level skills. |
| [gitlab-community](gitlab-community/) | Author a GitLab project's collaboration files on gitlab.com or self-managed hosts: issue/MR description templates with quick actions, a scoped-label taxonomy with a sync script, commit conventions with Changelog trailers and tokenless MR-pipeline validation, versioning policy and changelog_config.yml with a tag-pipeline check, community files (CONTRIBUTING, CODE_OF_CONDUCT, SECURITY), and generated project-level skills. |
