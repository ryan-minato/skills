---
name: devcontainer-authoring
description: >
  Dev container artifact authoring — develop, test, and publish Features,
  Templates, and prebuilt dev images. Use when creating, modifying, or
  releasing a devcontainer Feature, Template, or prebuilt image; when a tool
  install should be reusable across projects, a .devcontainer should become
  a reusable starter, or a team wants identical fast-starting containers;
  when an artifact misbehaves — a feature breaks on another base image or
  when applied twice, option substitution mangles a template, a release
  publishes nothing, or baked settings persist after deletion; when asked
  about mechanics — dependsOn vs installsAfter, option variables, metadata
  merge; or when src/*/devcontainer-feature.json, install.sh,
  devcontainer-template.json, or a devcontainers/ci workflow is at hand.
  Not for consuming these in a project's devcontainer.json
  (devcontainer-setup) or for application Docker images.
license: Apache-2.0
compatibility: >
  Testing and publishing require the Dev Container CLI (pinned via
  npx -y @devcontainers/cli@0.87.0 or npm install -g) and a running
  Docker-compatible engine; publishing also needs push access to an OCI
  registry (e.g. ghcr.io). Image CI guidance assumes GitHub Actions
  (devcontainers/ci); a raw-CLI alternative is included for other systems.
  Authoring guidance applies without them.
---

# Devcontainer Authoring

One skill for the three publishable Dev Container artifact kinds:
**Features** (install units), **Templates** (project starters), and
**prebuilt images** (baked environments). Decide which artifact the task
concerns, then read that branch's reference in full before touching any
file — each covers its design, implementation, and gotchas end to end.

## Which artifact

- A tool install or setup script that should be reusable across
  projects' dev containers, running on every container build → a
  **Feature**.
- A starting `.devcontainer` configuration copied once into a project,
  which the user then owns and edits freely → a **Template**.
- A team-shared environment built ahead of time so containers start fast
  and identically everywhere → a **prebuilt image**. Skip it for
  single-user projects that rarely rebuild — a registry image to
  maintain is not free.
- One project's own development environment → not this skill; use the
  `devcontainer-setup` skill (if not installed:
  `npx skills add ryan-minato/skills --skill devcontainer-setup`).
  One-off project setup belongs in `postCreateCommand` or the project's
  Dockerfile, not in a published artifact.

The artifacts compose: a Template's payload references Features and
prebuilt images by OCI address; an image bakes Features' results ahead
of time. If a Template payload is really a tool installer, build a
Feature instead.

## Prerequisites

A Docker-compatible engine and the Dev Container CLI:

```bash
npm install -g @devcontainers/cli@0.87.0   # or npx -y @devcontainers/cli@0.87.0
```

Authoring guidance needs neither; testing and publishing need both.

## Features

A Feature is a self-contained, versioned install unit consumers
reference by OCI address; its `install.sh` runs as root during every
container build.

- Creating or modifying one → read
  [references/feature-authoring.md](references/feature-authoring.md) in
  full first (scaffold, manifest essentials, install.sh contract,
  quality bar, independence rule).
- Manifest property not covered there →
  [references/feature-json-reference.md](references/feature-json-reference.md).
- Writing or debugging tests →
  [references/feature-testing.md](references/feature-testing.md).
- Publishing or releasing, including a release that published nothing →
  [references/feature-publishing.md](references/feature-publishing.md).

Repo skeleton to copy for a new Features repository:
[assets/feature-scaffold/](assets/feature-scaffold/).

## Templates

A Template is a parameterized `.devcontainer` starter applied once into
a project via option substitution.

- Creating or modifying one → read
  [references/template-authoring.md](references/template-authoring.md)
  in full first (manifest and substitution, payload design, smoke-test
  loop).
- Manifest field not covered there →
  [references/template-json-reference.md](references/template-json-reference.md).
- Publishing or releasing →
  [references/template-publishing.md](references/template-publishing.md).

Repo skeleton: [assets/template-scaffold/](assets/template-scaffold/).

## Prebuilt images

A prebuilt image bakes a configuration's Dockerfile, Features, and the
config itself (as the `devcontainer.metadata` label) into a pushable
image consumers reference.

- Prebuilding or publishing one → read
  [references/image-prebuild.md](references/image-prebuild.md) in full
  first (build and push, consumer configuration, what to bake).
- Baked settings misbehave (persist after deletion, appear unbidden), or
  deciding what to bake versus leave to consumers →
  [references/image-metadata-merge.md](references/image-metadata-merge.md).
- Automating the image build in CI →
  [references/image-ci.md](references/image-ci.md).

## Shared publishing model

All three artifact kinds publish to an OCI registry:

- GHCR packages default to **private** — flip visibility after the first
  publish, or consumers get auth errors that look like a missing
  artifact.
- Feature and Template releases are keyed on the manifest `version`: an
  already-published version is skipped — no bump, no publish, no
  force-republish.
- Images are ordinary container images — publish one mutable tag
  consumers track plus one immutable (date or SHA) tag for pinning and
  rollback.

## Spec references

Fetch the authoritative documents when exact behavior matters:

- Rendered spec: <https://containers.dev/implementors/spec/>
- Raw spec documents: <https://github.com/devcontainers/spec/tree/main/docs/specs>

Per-artifact raw spec links live in each branch reference.
