# engineering — Catalog Context

Rules, notes, and references that apply only to skills in this catalog.
(Repo-wide standards live in `.agents/knowledge/skill-quality.md`.)

## Requirements

Two kinds of skills belong here:

- **Methodology skills** — *how to approach* a class of engineering
  problems. The guidance must transfer across stacks: examples may use a
  concrete language for illustration, but the instructions themselves
  must not change when the tech stack changes.
- **Artifact-authoring workflow skills** — the full design→test→publish
  lifecycle of a specific engineering artifact ecosystem that is too
  narrow to justify its own catalog (currently: Dev Container
  artifacts). Toolchain-specific content is acceptable in these;
  day-to-day operation of a tool is not.

Dev Container content additionally: use exact property and CLI names
from the Dev Container spec, and include raw spec document links (see
References) so agents can verify interfaces on demand instead of
trusting paraphrases.

## References

Dev Container (scope: the `devcontainer-authoring` skill):

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
