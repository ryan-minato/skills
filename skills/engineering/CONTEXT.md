# engineering — Catalog Context

Rules, notes, and references that apply only to skills in this catalog.
(Repo-wide standards live in `.agents/knowledge/skill-quality.md`.)

The catalog holds three classes of skills:

- **Methodology skills** (code-refactoring, gitmoji): approaches that
  transfer across languages and frameworks.
- **Platform community skills** (github-community, gitlab-community):
  authoring the files that define how a repository's collaboration works —
  templates, label taxonomies, commit and release policy, CI validation,
  community health files, and generated project-level skills. These are
  the explicit exemption to the transfer-across-stacks rule: their value
  lands in what they leave behind in the target project, not in the skill
  staying installed. Day-to-day platform *operations* belong to the `ops`
  catalog.
- **Artifact-authoring skills** (devcontainer-authoring): the full
  design→test→publish lifecycle of a specific engineering artifact
  ecosystem that is too narrow to justify its own catalog. Also exempt
  from the transfer-across-stacks rule: toolchain-specific content is
  acceptable; day-to-day operation of a tool is not.

## Requirements — methodology skills

- The guidance must transfer across stacks: examples may use a concrete
  language for illustration, but the instructions themselves must not
  change when the tech stack changes.

## Requirements — platform community skills

- **Local-files doctrine.** Community skills write local files only —
  templates, configs, workflows, docs, generated skills; the project's own
  git flow publishes them. Nothing they do publishes directly, so they
  embed no full pre-publish gate; the *generated* project-level skills
  that publish carry a condensed gate instead (canonical copies live in
  each community skill's `assets/project-skill-*.md`).
- **Assess the project first.** Every community skill opens by
  inventorying what the project already defines (templates, labels,
  workflows, community files, AGENTS.md/CLAUDE.md, skills directory) and
  never invents structure parallel to what exists: build on it, or get
  the user's explicit approval to replace it.
- **Generated project skills are products.** Same quality bar as a
  published skill: frontmatter `name` equals the directory name,
  placeholders use `{{UPPER_SNAKE}}`, and zero leftover `{{...}}`
  placeholders survive delivery.
- **CI validation ships dependency-free and first-party.**
  github-community: first-party actions only (`actions/*`, `github/*`);
  third-party actions only behind explicit user opt-in, pinned to a
  commit SHA. gitlab-community: shipped CI snippets run tokenless (MR-
  event and tag-pipeline checks needing no secrets), default to Free-tier
  mechanisms with tier badges elsewhere, and write hosts as
  `$CI_SERVER_HOST`/`$CI_API_V4_URL`, never literal hostnames. Shipped
  validators are stdlib-only python3 committed into the target repo.
- **Quick actions are a feature and a hazard (gitlab-community).**
  Document every quick action a shipped template embeds — in the template
  and in the skill body.
- **References are split by branching condition, not topic** — same rule
  as the ops catalog: one file = one load condition, pointers state the
  condition verbatim, opt-in automation and schema details are
  sub-branches of their domain file.
- MCP tools, where mentioned, are described by capability, never by name.
- Skills are fully independent of each other: sibling skills (including
  the `ops` skills) are named at most for disambiguation — never
  path-linked, never accompanied by install instructions, and never
  depended on behaviorally (self-containment).
- Exact CLI names, CI keywords, schema keys, and platform file
  conventions must be re-verified before publishing a skill revision —
  GitHub form schemas, actions, and `release.yml` against
  <https://docs.github.com>; GitLab template mechanics, CI YAML, and
  changelog config against <https://docs.gitlab.com>.

## Requirements — artifact-authoring skills

- One skill covers one artifact ecosystem end to end (design, implement,
  test, publish), with references split by execution branch — one file
  per load condition.
- Dev Container content: use exact property and CLI names from the Dev
  Container spec, and include raw spec document links (see References)
  so agents can verify interfaces on demand instead of trusting
  paraphrases.

## Disambiguation

How to approach a cross-stack engineering problem → the methodology
skills · authoring a GitHub repository's conventions or community health
files → `github-community` · a GitLab project's → `gitlab-community` ·
performing platform operations (filing issues, opening PRs/MRs, cutting
releases) → the `ops` catalog's `github-ops`/`gitlab-ops` · authoring
Dev Container artifacts (Features, Templates, prebuilt images) →
`devcontainer-authoring`; consuming them in a project's own
devcontainer.json → `devcontainer-setup` in `core`.

## References

Platform community (scope: `github-community`, `gitlab-community`):

- Community health files (GitHub): <https://docs.github.com/en/communities/setting-up-your-project-for-healthy-contributions/creating-a-default-community-health-file>
- Issue forms schema: <https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/syntax-for-issue-forms>
- Automatically generated release notes (`.github/release.yml`): <https://docs.github.com/en/repositories/releasing-projects-on-github/automatically-generated-release-notes>
- GitHub Actions workflow syntax: <https://docs.github.com/en/actions/writing-workflows/workflow-syntax-for-github-actions>
- GitLab description templates: <https://docs.gitlab.com/user/project/description_templates/>
- GitLab CI/CD YAML: <https://docs.gitlab.com/ci/yaml/> · Predefined CI variables: <https://docs.gitlab.com/ci/variables/predefined_variables/>
- GitLab changelogs: <https://docs.gitlab.com/user/project/changelogs/>
- GitLab labels: <https://docs.gitlab.com/user/project/labels/>
- Contributor Covenant: <https://www.contributor-covenant.org/>

Dev Container (scope: `devcontainer-authoring`):

- Dev Container spec (rendered): <https://containers.dev/implementors/spec/>
- devcontainer.json reference: <https://containers.dev/implementors/json_reference/>
- Features: <https://containers.dev/implementors/features/>
- Features distribution: <https://containers.dev/implementors/features-distribution/>
- Templates: <https://containers.dev/implementors/templates/>
- Templates distribution: <https://containers.dev/implementors/templates-distribution/>
- Spec source (raw markdown): <https://github.com/devcontainers/spec/tree/main/docs/specs>
- Dev Container CLI: <https://github.com/devcontainers/cli>
- Official images: <https://github.com/devcontainers/images>
- CI action: <https://github.com/devcontainers/ci>
- Starters (prior art; superseded by the scaffolds bundled in the
  skill): <https://github.com/devcontainers/feature-starter>,
  <https://github.com/devcontainers/template-starter>
- Third-party feature collection prior art:
  <https://github.com/stacit-ai/devcontainer-features>
