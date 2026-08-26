# engineering — Catalog Context

Rules, notes, and references that apply only to skills in this catalog.
(Repo-wide standards live in `.agents/knowledge/skill-quality.md`.)

The catalog holds two classes of skills:

- **Methodology skills** (code-refactoring, gitmoji, goal-alignment):
  approaches that transfer across languages and frameworks.
- **Artifact-authoring skills** (devcontainer-authoring, design-md): the full
  design→test→publish lifecycle of a specific engineering artifact
  ecosystem that is too narrow to justify its own catalog. Also exempt
  from the transfer-across-stacks rule: toolchain-specific content is
  acceptable; day-to-day operation of a tool is not.

Building a GitHub or GitLab project's complete lifecycle harness —
including its collaboration files and generated project skills — belongs
to the disposable `meta` catalog, not here.

## Requirements — methodology skills

- The guidance must transfer across stacks: examples may use a concrete
  language for illustration, but the instructions themselves must not
  change when the tech stack changes.

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
skills · aligning on what something should achieve and recording it as a
goal document → `goal-alignment`; clarifying ambiguous requirements while
already coding → `programming-guidelines` in `core`; implementation
planning → neither (out of catalog scope) · building or systematically
repairing a GitHub or GitLab project's complete lifecycle harness —
including its conventions, community files, and day-to-day platform
workflows — → the `meta` catalog's `meta-github-workflow` /
`meta-gitlab-workflow` · authoring Dev Container artifacts (Features,
Templates, prebuilt images) → `devcontainer-authoring`; consuming them in
a project's own devcontainer.json → `devcontainer-setup` in `core` ·
durable visual-design specifications → `design-md`.

## References

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
